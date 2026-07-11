"""
Trains the Neural Network for MCTSApproximationAgent with Action-Masked Softmax and Auxiliary next-state heads.

Backend selection:
  Set KERAS_BACKEND env-var before running:
    KERAS_BACKEND=tensorflow   (default when TF is installed)
    KERAS_BACKEND=torch
    KERAS_BACKEND=jax
"""
import json
import os
from pathlib import Path

# Configure PyTorch device before Keras is imported if using torch backend
# NOTE: Do NOT call torch.set_default_device("cuda") here — it causes the Keras/PyTorch
# DataLoader sampler to create a CUDA generator while training data is on CPU, resulting in:
#   RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'
if os.environ.get("KERAS_BACKEND") == "torch":
    import torch
    if torch.cuda.is_available():
        print("[PyTorch Backend] CUDA GPU detected.")
    else:
        print("[PyTorch Backend] CUDA GPU NOT detected. Defaulting to 'cpu'.")

import keras
import numpy as np


class PrimaryLossCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        policy_loss = logs.get("val_policy_loss", 0.0)
        value_loss = logs.get("val_value_loss", 0.0)
        # Combined validation loss: 5.0 * policy_loss + 1.0 * value_loss
        logs["val_primary_loss"] = float(policy_loss * 5.0 + value_loss * 1.0)
        
        train_policy = logs.get("policy_loss", 0.0)
        train_value = logs.get("value_loss", 0.0)
        logs["primary_loss"] = float(train_policy * 5.0 + train_value * 1.0)


@keras.saving.register_keras_serializable()
class SliceLayer(keras.layers.Layer):
    """Extracts a slice from a 2-D input tensor along axis 1.

    Uses ``keras.ops.slice`` instead of Python bracket slicing so that the
    operation compiles to static ONNX nodes and survives TorchScript tracing.
    The slice size is derived from ``keras.ops.shape`` so it works on all
    backends (the torch backend does not handle ``-1`` in the shape argument).
    """

    def __init__(self, start, end=None, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, x):
        shape = keras.ops.shape(x)
        if self.end is not None:
            return keras.ops.slice(x, [0, self.start], [shape[0], self.end - self.start])
        return keras.ops.slice(x, [0, self.start], [shape[0], shape[1] - self.start])

    def get_config(self):
        config = super().get_config()
        config.update({"start": self.start, "end": self.end})
        return config


@keras.saving.register_keras_serializable()
class ConstantLayer(keras.layers.Layer):
    """Outputs a constant scalar value broadcast to match the batch size.

    Uses ``keras.ops.slice`` internally instead of bracket slicing so the
    operation is ONNX-export-compatible on all backends.
    """

    def __init__(self, value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.value = float(value)

    def call(self, x):
        shape = keras.ops.shape(x)
        sliced = keras.ops.slice(x, [0, 0], [shape[0], 1])
        return sliced * 0.0 + self.value

    def get_config(self):
        config = super().get_config()
        config.update({"value": self.value})
        return config


@keras.saving.register_keras_serializable()
class ApplyMaskLayer(keras.layers.Layer):
    """Applies a -inf mask to invalid action logits.

    Replaces ``Lambda(lambda inputs: inputs[0] + (1.0 - inputs[1]) * -1e9)``.
    Accepts a list of two tensors [logits, mask].
    """

    def call(self, inputs):
        logits, mask = inputs
        return logits + (1.0 - mask) * -1e9

    def get_config(self):
        return super().get_config()


from battle_agents.mcts_approximation.db.python.database import db
from battle_agents.mcts_approximation.state_encoder import (
    ACTION_SPACE, NUM_ABILITY_INDICES, NUM_DENSE_FEATURES,
    NUM_EMBEDDING_INDICES, NUM_FIELD_FEATURES, NUM_ITEM_INDICES,
    NUM_MOVE_INDICES, NUM_SPECIES_INDICES, OFF_ABILITIES, OFF_ITEMS, OFF_MOVES,
    OFF_SPECIES, TOTAL_FEATURES)


def _split_features(X: np.ndarray):
    n = NUM_DENSE_FEATURES
    X_dense     = X[:, :n].astype(np.float32)
    X_species   = X[:, n + OFF_SPECIES   : n + OFF_SPECIES   + NUM_SPECIES_INDICES  ].astype(np.int32)
    X_moves     = X[:, n + OFF_MOVES     : n + OFF_MOVES     + NUM_MOVE_INDICES     ].astype(np.int32)
    X_items     = X[:, n + OFF_ITEMS     : n + OFF_ITEMS     + NUM_ITEM_INDICES     ].astype(np.int32)
    X_abilities = X[:, n + OFF_ABILITIES : n + OFF_ABILITIES + NUM_ABILITY_INDICES ].astype(np.int32)
    return X_dense, X_species, X_moves, X_items, X_abilities


def _parse_steps(game_data):
    X, y_value, y_policy, X_next, action_masks = [], [], [], [], []
    total_len = len(game_data)
    if total_len == 0:
        return X, y_value, y_policy, X_next, action_masks
        
    N = total_len // 2  # Each side has N turns
    
    # Process P1
    for i in range(N):
        step = game_data[i]
        X.append(step["features"])
        y_value.append(step["value"])
        
        policy_dict = step.get("policy", {})
        policy_array = [policy_dict.get(a, 0.0) for a in ACTION_SPACE]
        s = sum(policy_array)
        policy_array = [p / s for p in policy_array] if s > 0 else [1.0 / len(ACTION_SPACE)] * len(ACTION_SPACE)
        y_policy.append(policy_array)
        
        mask = [1.0 if a in policy_dict else 0.0 for a in ACTION_SPACE]
        action_masks.append(mask)
        
        if i < N - 1:
            X_next.append(game_data[i + 1]["features"])
        else:
            X_next.append([0.0] * TOTAL_FEATURES)
            
    # Process P2
    for i in range(N, 2 * N):
        step = game_data[i]
        X.append(step["features"])
        y_value.append(step["value"])
        
        policy_dict = step.get("policy", {})
        policy_array = [policy_dict.get(a, 0.0) for a in ACTION_SPACE]
        s = sum(policy_array)
        policy_array = [p / s for p in policy_array] if s > 0 else [1.0 / len(ACTION_SPACE)] * len(ACTION_SPACE)
        y_policy.append(policy_array)
        
        mask = [1.0 if a in policy_dict else 0.0 for a in ACTION_SPACE]
        action_masks.append(mask)
        
        if i < 2 * N - 1:
            X_next.append(game_data[i + 1]["features"])
        else:
            X_next.append([0.0] * TOTAL_FEATURES)
            
    return X, y_value, y_policy, X_next, action_masks


def load_data(data_dir: str = "data/games"):
    X, y_value, y_policy, X_next, action_masks = [], [], [], [], []
    for f in Path(data_dir).glob("game_*.json"):
        gx, gv, gp, gn, gm = _parse_steps(json.loads(f.read_text()))
        X += gx; y_value += gv; y_policy += gp; X_next += gn; action_masks += gm
    return (np.array(X, dtype=np.float32), np.array(y_value, dtype=np.float32), 
            np.array(y_policy, dtype=np.float32), np.array(X_next, dtype=np.float32),
            np.array(action_masks, dtype=np.float32))


def load_data_from_files(files):
    X, y_value, y_policy, X_next, action_masks = [], [], [], [], []
    skipped_count = 0
    for f in files:
        try:
            data = json.loads(f.read_text())
            if not data:
                continue
            # Gracefully skip files with legacy feature shapes
            if len(data[0]["features"]) != TOTAL_FEATURES:
                skipped_count += 1
                continue
            gx, gv, gp, gn, gm = _parse_steps(data)
            X += gx; y_value += gv; y_policy += gp; X_next += gn; action_masks += gm
        except Exception as e:
            print(f"Error loading file {f}: {e}")
            
    if skipped_count > 0:
        print(f"Skipped {skipped_count} game files due to feature shape mismatch (expected {TOTAL_FEATURES} features).")
            
    if len(X) == 0:
        return (np.empty((0, TOTAL_FEATURES), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, len(ACTION_SPACE)), dtype=np.float32),
                np.empty((0, TOTAL_FEATURES), dtype=np.float32),
                np.empty((0, len(ACTION_SPACE)), dtype=np.float32))
                
    return (np.array(X, dtype=np.float32),
            np.array(y_value, dtype=np.float32),
            np.array(y_policy, dtype=np.float32),
            np.array(X_next, dtype=np.float32),
            np.array(action_masks, dtype=np.float32))


def extract_aux_targets_batch(X_next, num_species, num_moves):
    batch_size = X_next.shape[0]
    
    # 1. Field conditions (Field indices 636 to 696)
    field = X_next[:, 636:696]
    
    # 2. Boosts
    own_boosts = X_next[:, 312:318]
    opp_boosts = X_next[:, 630:636]
    
    # Initialize active targets
    own_hp = np.zeros((batch_size, 1), dtype=np.float32)
    opp_hp = np.zeros((batch_size, 1), dtype=np.float32)
    
    # Statuses
    own_statuses = np.zeros((batch_size, 5), dtype=np.float32)
    opp_statuses = np.zeros((batch_size, 5), dtype=np.float32)
    
    # Stats
    own_stats = np.zeros((batch_size, 5), dtype=np.float32)
    opp_stats = np.zeros((batch_size, 5), dtype=np.float32)
    
    # Types
    own_types = np.zeros((batch_size, 18), dtype=np.float32)
    opp_types = np.zeros((batch_size, 18), dtype=np.float32)
    
    # Species
    own_species = np.zeros((batch_size,), dtype=np.int32)
    opp_species = np.zeros((batch_size,), dtype=np.int32)
    
    # Moves
    own_moves_multihot = np.zeros((batch_size, num_moves), dtype=np.float32)
    opp_moves_multihot = np.zeros((batch_size, num_moves), dtype=np.float32)
    
    for b in range(batch_size):
        # Find active own Pokémon index
        own_act = -1
        for i in range(6):
            if X_next[b, i * 52 + 7] == 1.0:
                own_act = i
                break
                
        # Find active opponent Pokémon index
        opp_act = -1
        for j in range(6):
            if X_next[b, 318 + j * 52 + 7] == 1.0:
                opp_act = j
                break
                
        if own_act != -1:
            own_hp[b, 0] = X_next[b, own_act * 52]
            own_statuses[b, :] = X_next[b, own_act * 52 + 2 : own_act * 52 + 7]
            own_stats[b, :] = X_next[b, own_act * 52 + 9 : own_act * 52 + 14]
            own_types[b, :] = X_next[b, own_act * 52 + 14 : own_act * 52 + 32]
            # Use NUM_DENSE_FEATURES as dynamic offset
            own_species[b] = np.clip(int(round(X_next[b, NUM_DENSE_FEATURES + own_act])), 0, num_species - 1)
            
            # Extract own active moves
            for k in range(4):
                idx = int(round(X_next[b, NUM_DENSE_FEATURES + OFF_MOVES + own_act * 4 + k]))
                if 0 < idx < num_moves:
                    own_moves_multihot[b, idx] = 1.0
            
        if opp_act != -1:
            opp_hp[b, 0] = X_next[b, 318 + opp_act * 52]
            opp_statuses[b, :] = X_next[b, 318 + opp_act * 52 + 2 : 318 + opp_act * 52 + 7]
            opp_stats[b, :] = X_next[b, 318 + opp_act * 52 + 9 : 318 + opp_act * 52 + 14]
            opp_types[b, :] = X_next[b, 318 + opp_act * 52 + 14 : 318 + opp_act * 52 + 32]
            # Use NUM_DENSE_FEATURES as dynamic offset
            opp_species[b] = np.clip(int(round(X_next[b, NUM_DENSE_FEATURES + 6 + opp_act])), 0, num_species - 1)
            
            # Extract opponent active moves
            for k in range(4):
                idx = int(round(X_next[b, NUM_DENSE_FEATURES + OFF_MOVES + 24 + opp_act * 4 + k]))
                if 0 < idx < num_moves:
                    opp_moves_multihot[b, idx] = 1.0
            
    # Convert species indices to one-hot format
    own_species_onehot = keras.utils.to_categorical(own_species, num_classes=num_species)
    opp_species_onehot = keras.utils.to_categorical(opp_species, num_classes=num_species)
    
    return {
        "aux_field": field,
        "aux_own_hp": own_hp,
        "aux_opp_hp": opp_hp,
        "aux_own_statuses": own_statuses,
        "aux_opp_statuses": opp_statuses,
        "aux_own_boosts": own_boosts,
        "aux_opp_boosts": opp_boosts,
        "aux_own_stats": own_stats,
        "aux_opp_stats": opp_stats,
        "aux_own_types": own_types,
        "aux_opp_types": opp_types,
        "aux_own_species": own_species_onehot,
        "aux_opp_species": opp_species_onehot,
        "aux_own_moves": own_moves_multihot,
        "aux_opp_moves": opp_moves_multihot,
    }


def build_model(num_dense: int, num_moves: int, num_species: int,
                num_items: int, num_abilities: int) -> keras.Model:
    """Builds the multi-input value+policy network with integrated Meta-Planner Transformer and action masking."""

    # Category Inputs
    inp_dense     = keras.layers.Input(shape=(num_dense,), name="dense_features")
    inp_species   = keras.layers.Input(shape=(12,), name="species_indices", dtype="int32")
    inp_moves     = keras.layers.Input(shape=(48,), name="move_indices",    dtype="int32")
    inp_items     = keras.layers.Input(shape=(12,), name="item_indices",    dtype="int32")
    inp_abilities = keras.layers.Input(shape=(12,), name="ability_indices", dtype="int32")
    inp_mask      = keras.layers.Input(shape=(len(ACTION_SPACE),), name="action_mask")

    # --- META-PLANNER TRANSFORMER SUB-NETWORK ---
    # Shared Embeddings for Token Construction
    emb_species_layer = keras.layers.Embedding(num_species, 16, name="meta_emb_species")
    emb_moves_layer   = keras.layers.Embedding(num_moves, 8, name="meta_emb_moves")
    emb_items_layer   = keras.layers.Embedding(num_items, 8, name="meta_emb_items")
    emb_abilities_layer = keras.layers.Embedding(num_abilities, 8, name="meta_emb_abilities")

    # Slice out 60 field conditions (weather, hazards, volatiles) at the end of original dense features [636:696]
    field_conds = SliceLayer(636, 696, name="field_conditions")(inp_dense)

    tokens = []
    for i in range(12):
        if i < 6:
            start_idx = i * 52
            owner_val = 1.0
        else:
            start_idx = 318 + (i - 6) * 52
            owner_val = 0.0
        
        # 1. Dense features for Pokémon i
        p_dense = SliceLayer(start_idx, start_idx + 52, name=f"p{i}_dense")(inp_dense)
        # 2. Extract PP features from the appended block [696:744]
        p_pp    = SliceLayer(696 + i * 4, 696 + i * 4 + 4, name=f"p{i}_pp")(inp_dense)
        # 3. Categorical embeddings
        p_spec = SliceLayer(i, i + 1, name=f"p{i}_species")(inp_species)
        p_item = SliceLayer(i, i + 1, name=f"p{i}_item")(inp_items)
        p_abil = SliceLayer(i, i + 1, name=f"p{i}_ability")(inp_abilities)

        # 4 moves per Pokémon
        m_start = i * 4 if i < 6 else 24 + (i - 6) * 4
        p_moves = SliceLayer(m_start, m_start + 4, name=f"p{i}_moves")(inp_moves)

        spec_emb = keras.layers.Flatten()(emb_species_layer(p_spec))
        item_emb = keras.layers.Flatten()(emb_items_layer(p_item))
        abil_emb = keras.layers.Flatten()(emb_abilities_layer(p_abil))
        moves_emb = keras.layers.Flatten()(emb_moves_layer(p_moves))

        # Owner flag
        owner_flag = ConstantLayer(owner_val, name=f"p{i}_owner")(inp_dense)

        # Concatenate features to form a 181-dimensional Pokémon representation token (52 + 4 + 16 + 8 + 8 + 32 + 1 + 60 = 181)
        token = keras.layers.Concatenate(name=f"p{i}_token")([
            p_dense, p_pp, spec_emb, item_emb, abil_emb, moves_emb, owner_flag, field_conds
        ])
        
        # Expand token to shape (None, 1, 181) for sequence format
        token_expanded = keras.layers.Reshape((1, 181), name=f"p{i}_token_expanded")(token)
        tokens.append(token_expanded)

    # Sequence of 12 tokens: shape (None, 12, 181)
    token_seq = keras.layers.Concatenate(axis=1, name="token_sequence")(tokens)

    # Cross-Attention comparison layer
    attn_out = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, name="meta_attention")(
        query=token_seq, value=token_seq
    )
    attn_out = keras.layers.Add()([token_seq, attn_out])
    attn_out = keras.layers.LayerNormalization()(attn_out)

    # Project token final states to importance scores: shape (None, 12)
    scores = keras.layers.Dense(1, name="token_score_projection")(attn_out)
    scores = keras.layers.Reshape((12,), name="token_scores")(scores)

    # Separate own (P1) vs opponent (P2) scores
    own_scores = SliceLayer(0, 6, name="own_scores")(scores)
    opp_scores = SliceLayer(6, name="opp_scores")(scores)

    # Activation scaling: Softmax for own win-conditions, Sigmoid for opponent threat evaluation
    own_weights = keras.layers.Activation("softmax", name="meta_own_weights")(own_scores)
    opp_weights = keras.layers.Activation("sigmoid", name="meta_opp_weights")(opp_scores)

    # Final 12-dimensional Meta-Plan weights
    meta_plan = keras.layers.Concatenate(name="meta_plan")([own_weights, opp_weights])

    # --- MAIN TACTICAL NETWORK ---
    # Shared Embeddings for the Main network trunk
    emb_species_main   = keras.layers.Flatten()(keras.layers.Embedding(num_species,   32, name="emb_species")(inp_species))
    emb_moves_main     = keras.layers.Flatten()(keras.layers.Embedding(num_moves,     32, name="emb_moves")(inp_moves))
    emb_items_main     = keras.layers.Flatten()(keras.layers.Embedding(num_items,     16, name="emb_items")(inp_items))
    emb_abilities_main = keras.layers.Flatten()(keras.layers.Embedding(num_abilities, 16, name="emb_abilities")(inp_abilities))

    concat_main = keras.layers.Concatenate()(
        [inp_dense, emb_species_main, emb_moves_main, emb_items_main, emb_abilities_main]
    )

    # Fuse the 12-dimensional Meta-Plan directly into the main inputs before the dense trunk
    fused_features = keras.layers.Concatenate(name="fused_features")([concat_main, meta_plan])

    # Trunk (Explicitly named for legacy weight matching)
    x = keras.layers.Dense(512, name="dense")(fused_features)
    x = keras.layers.BatchNormalization(name="batch_normalization")(x)
    x = keras.layers.Activation("relu", name="activation")(x)
    x = keras.layers.Dropout(0.3, name="dropout")(x)

    x = keras.layers.Dense(256, name="dense_1")(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_1")(x)
    x = keras.layers.Activation("relu", name="activation_1")(x)
    x = keras.layers.Dropout(0.2, name="dropout_1")(x)

    x = keras.layers.Dense(128, name="dense_2")(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_2")(x)
    x = keras.layers.Activation("relu", name="activation_2")(x)
    x = keras.layers.Dropout(0.1, name="dropout_2")(x)

    # Core outputs
    out_value  = keras.layers.Dense(1, activation="sigmoid", name="value")(x)
    
    # Policy head with Action-Masked Softmax
    logits = keras.layers.Dense(len(ACTION_SPACE), name="policy_logits")(x)
    masked_logits = ApplyMaskLayer(name="masked_logits")([logits, inp_mask])
    out_policy = keras.layers.Activation("softmax", name="policy")(masked_logits)

    # Auxiliary dynamics outputs
    out_field   = keras.layers.Dense(NUM_FIELD_FEATURES, activation="sigmoid", name="aux_field")(x)
    out_own_hp  = keras.layers.Dense(1, activation="sigmoid", name="aux_own_hp")(x)
    out_opp_hp  = keras.layers.Dense(1, activation="sigmoid", name="aux_opp_hp")(x)
    out_own_statuses = keras.layers.Dense(5, activation="sigmoid", name="aux_own_statuses")(x)
    out_opp_statuses = keras.layers.Dense(5, activation="sigmoid", name="aux_opp_statuses")(x)
    out_own_boosts = keras.layers.Dense(6, activation="tanh", name="aux_own_boosts")(x)
    out_opp_boosts = keras.layers.Dense(6, activation="tanh", name="aux_opp_boosts")(x)
    out_own_stats = keras.layers.Dense(5, activation="sigmoid", name="aux_own_stats")(x)
    out_opp_stats = keras.layers.Dense(5, activation="sigmoid", name="aux_opp_stats")(x)
    out_own_types = keras.layers.Dense(18, activation="sigmoid", name="aux_own_types")(x)
    out_opp_types = keras.layers.Dense(18, activation="sigmoid", name="aux_opp_types")(x)
    out_own_species = keras.layers.Dense(num_species, activation="softmax", name="aux_own_species")(x)
    out_opp_species = keras.layers.Dense(num_species, activation="softmax", name="aux_opp_species")(x)
    out_own_moves = keras.layers.Dense(num_moves, activation="sigmoid", name="aux_own_moves")(x)
    out_opp_moves = keras.layers.Dense(num_moves, activation="sigmoid", name="aux_opp_moves")(x)

    model = keras.Model(
        inputs=[inp_dense, inp_species, inp_moves, inp_items, inp_abilities, inp_mask],
        outputs=[
            out_value, out_policy,
            out_field, out_own_hp, out_opp_hp,
            out_own_statuses, out_opp_statuses,
            out_own_boosts, out_opp_boosts,
            out_own_stats, out_opp_stats,
            out_own_types, out_opp_types,
            out_own_species, out_opp_species,
            out_own_moves, out_opp_moves,
            meta_plan
        ],
    )
    return model


def export_to_onnx(model, onnx_path):
    print(f"Exporting model to ONNX format at {onnx_path}...")

    # Warm up the model with a dummy forward pass (required by model.export / torch tracing)
    dummy = {
        "dense_features": np.zeros((1, NUM_DENSE_FEATURES), dtype=np.float32),
        "species_indices": np.zeros((1, 12), dtype=np.int32),
        "move_indices": np.zeros((1, 48), dtype=np.int32),
        "item_indices": np.zeros((1, 12), dtype=np.int32),
        "ability_indices": np.zeros((1, 12), dtype=np.int32),
        "action_mask": np.ones((1, len(ACTION_SPACE)), dtype=np.float32),
    }
    try:
        model(dummy, training=False)
    except Exception:
        pass

    backend = keras.backend.backend()

    # 1. Try Keras 3 direct model.export (backend-agnostic)
    try:
        model.export(str(onnx_path), format="onnx")
        print("ONNX export completed successfully using model.export!")
        return True
    except Exception as e:
        print(f"[Warning] Direct model.export to ONNX failed: {e}")

    # 2. Backend-specific fallback
    if backend == "torch":
        try:
            import torch
            input_names = ["dense_features", "species_indices", "move_indices",
                           "item_indices", "ability_indices", "action_mask"]
            output_names = ["value", "policy", "aux_field", "aux_own_hp", "aux_opp_hp",
                            "aux_own_statuses", "aux_opp_statuses", "aux_own_boosts",
                            "aux_opp_boosts", "aux_own_stats", "aux_opp_stats",
                            "aux_own_types", "aux_opp_types", "aux_own_species",
                            "aux_opp_species", "aux_own_moves", "aux_opp_moves",
                            "meta_plan"]
            if hasattr(model, "eval"):
                model.eval()
            torch_dummy = tuple(
                torch.from_numpy(dummy[k])
                for k in ["dense_features", "species_indices", "move_indices",
                           "item_indices", "ability_indices", "action_mask"]
            )
            torch.onnx.export(
                model,
                (torch_dummy,),
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                opset_version=18,
                dynamo=False,
                verbose=False,
            )
            print("ONNX export completed successfully using torch.onnx.export!")
            return True
        except Exception as e_torch:
            print(f"[Warning] torch.onnx.export failed: {e_torch}")

    if backend == "tensorflow":
        try:
            import tf2onnx
            import tensorflow as tf
            input_spec = [
                tf.TensorSpec((None, NUM_DENSE_FEATURES), tf.float32, name="dense_features"),
                tf.TensorSpec((None, 12), tf.int32, name="species_indices"),
                tf.TensorSpec((None, 48), tf.int32, name="move_indices"),
                tf.TensorSpec((None, 12), tf.int32, name="item_indices"),
                tf.TensorSpec((None, 12), tf.int32, name="ability_indices"),
                tf.TensorSpec((None, len(ACTION_SPACE)), tf.float32, name="action_mask"),
            ]
            onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_spec, opset=18)
            import onnx
            onnx.save(onnx_model, str(onnx_path))
            print("ONNX export completed successfully using tf2onnx!")
            return True
        except Exception as e_tf:
            print(f"[Warning] tf2onnx export failed: {e_tf}")

    print("[Error] All ONNX export methods failed.")
    return False


def compute_counterfactual_targets(model, inputs, is_scratch=False, batch_size=256):
    """
    Computes Counterfactual Value Drop targets for the Meta-Planner attention weights.
    For each sample, sets each Pokémon's HP to 0 and fainted to 1, and measures
    the resulting drop/increase in predicted win probability.
    """
    import numpy as np
    dense = inputs["dense_features"]
    N = len(dense)
    
    own_targets = np.zeros((N, 6), dtype=np.float32)
    opp_targets = np.zeros((N, 6), dtype=np.float32)
    
    if is_scratch:
        # Fallback to current HP-based targets
        print("Model is uninitialized/scratch. Generating HP-based fallback targets.")
        for i in range(6):
            own_targets[:, i] = dense[:, i * 52] # HP ratio
        # Normalize own_targets
        for i in range(N):
            s = np.sum(own_targets[i])
            if s > 1e-5:
                own_targets[i] /= s
            else:
                own_targets[i] = np.ones(6, dtype=np.float32) / 6.0
                
        for j in range(6):
            opp_targets[:, j] = dense[:, 318 + j * 52] # HP ratio
            
        meta_targets = np.concatenate([own_targets, opp_targets], axis=1)
        return meta_targets

    print("Computing Counterfactual Value Drop targets for Meta-Planner...")
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sub_n = end - start
        
        # Build parallel batch
        sub_dense = dense[start:end]
        sub_species = inputs["species_indices"][start:end]
        sub_moves = inputs["move_indices"][start:end]
        sub_items = inputs["item_indices"][start:end]
        sub_abilities = inputs["ability_indices"][start:end]
        sub_mask = inputs["action_mask"][start:end]
        
        # We need a total of 13 * sub_n states
        parallel_dense = np.tile(sub_dense, (13, 1))
        
        # Modify the tiled dense features for counterfactuals
        # Block 0: base (0 to sub_n) -> no changes
        # Block 1..6: own fainted (1 to 6)
        for i in range(6):
            block_start = (1 + i) * sub_n
            block_end = (2 + i) * sub_n
            parallel_dense[block_start:block_end, i * 52] = 0.0
            parallel_dense[block_start:block_end, i * 52 + 1] = 1.0
            
        # Block 7..12: opp fainted (7 to 12)
        for j in range(6):
            block_start = (7 + j) * sub_n
            block_end = (8 + j) * sub_n
            parallel_dense[block_start:block_end, 318 + j * 52] = 0.0
            parallel_dense[block_start:block_end, 318 + j * 52 + 1] = 1.0
            
        # Tile categorical inputs to match parallel_dense shape
        parallel_species = np.tile(sub_species, (13, 1))
        parallel_moves = np.tile(sub_moves, (13, 1))
        parallel_items = np.tile(sub_items, (13, 1))
        parallel_abilities = np.tile(sub_abilities, (13, 1))
        parallel_mask = np.tile(sub_mask, (13, 1))
        
        # Call model directly
        preds = model(
            [parallel_dense, parallel_species, parallel_moves, parallel_items, parallel_abilities, parallel_mask],
            training=False
        )
        
        # First output is value
        if isinstance(preds, (list, tuple)):
            val_tensor = preds[0]
        else:
            val_tensor = preds
            
        values = keras.ops.convert_to_numpy(val_tensor)
        values = values.reshape((13, sub_n))
        
        # base values
        v_base = values[0]
        
        # Own drop: v_base - v_own_fainted
        for i in range(6):
            v_fainted = values[1 + i]
            own_targets[start:end, i] = np.maximum(0.0, v_base - v_fainted)
            
        # Opp drop: v_opp_fainted - v_base
        for j in range(6):
            v_fainted = values[7 + j]
            opp_targets[start:end, j] = np.maximum(0.0, v_fainted - v_base)
            
    # Normalize own targets
    for i in range(N):
        s = np.sum(own_targets[i])
        if s > 1e-5:
            own_targets[i] /= s
        else:
            own_targets[i] = np.ones(6, dtype=np.float32) / 6.0
            
    opp_targets = np.clip(opp_targets, 0.0, 1.0)
    meta_targets = np.concatenate([own_targets, opp_targets], axis=1)
    return meta_targets


def train(data_dir: str = "data", model_save_path: str = "data/mcts_model.keras", max_games_buffer: int = 2500, epochs: int = 15):
    print(f"[Keras backend: {keras.backend.backend()}]")
    print(f"Locating game files in {data_dir}/gen*...")
    all_files = list(Path(data_dir).glob("gen*/game_*.json"))
    if len(all_files) == 0:
        print("No data found. Please run generate_data.py first.")
        return

    # Replay Buffer: newest files first
    all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    if len(all_files) > max_games_buffer:
        print(f"Replay Buffer: Keeping {max_games_buffer} most recent games out of {len(all_files)}.")
        all_files = all_files[:max_games_buffer]

    np.random.shuffle(all_files)

    split_idx   = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files   = all_files[split_idx:]

    if len(train_files) == 0:
        train_files = all_files
        val_files   = all_files
        print("[WARNING] Very little data found. Using same file for train and validation.")
    elif len(val_files) == 0:
        val_files = train_files[:1]
        print("[WARNING] Very little data found. Sharing some training files for validation.")

    print(f"Split {len(all_files)} game files → {len(train_files)} train / {len(val_files)} val.")

    print("Loading training data...")
    X_train, y_value_train, y_policy_train, X_next_train, action_masks_train = load_data_from_files(train_files)
    print("Loading validation data...")
    X_val, y_value_val, y_policy_val, X_next_val, action_masks_val = load_data_from_files(val_files)

    print(f"Loaded {len(X_train)} training turns and {len(X_val)} validation turns.")

    if len(X_train) == 0:
        print("Error: No training turns found.")
        return

    if X_train.shape[1] != TOTAL_FEATURES:
        print(
            f"[WARNING] Feature count mismatch: got {X_train.shape[1]}, expected {TOTAL_FEATURES}.\n"
            "Your saved data was generated with the old encoder. "
            "Please delete the gen* folders and re-run generate_data.py."
        )
        return

    # Shuffle turns
    def _shuffle(X, yv, yp, xn, am):
        idx = np.random.permutation(len(X))
        return X[idx], yv[idx], yp[idx], xn[idx], am[idx]

    X_train, y_value_train, y_policy_train, X_next_train, action_masks_train = _shuffle(
        X_train, y_value_train, y_policy_train, X_next_train, action_masks_train
    )
    if len(X_val) > 0:
        X_val, y_value_val, y_policy_val, X_next_val, action_masks_val = _shuffle(
            X_val, y_value_val, y_policy_val, X_next_val, action_masks_val
        )

    # Split into 5 named inputs
    splits_train = _split_features(X_train)
    X_dense_train, X_species_train, X_moves_train, X_items_train, X_abilities_train = splits_train

    num_moves     = db.get_num_moves()
    num_species   = db.get_num_species()
    num_items     = db.get_num_items()
    num_abilities = db.get_num_abilities()

    # Extract auxiliary targets
    aux_targets_train = extract_aux_targets_batch(X_next_train, num_species, num_moves)
    train_inputs = {
        "dense_features":   X_dense_train,
        "species_indices":  X_species_train,
        "move_indices":     X_moves_train,
        "item_indices":     X_items_train,
        "ability_indices":  X_abilities_train,
        "action_mask":      action_masks_train,
    }
    train_outputs = {
        "value": y_value_train,
        "policy": y_policy_train,
    }
    train_outputs.update(aux_targets_train)

    val_data = None
    if len(X_val) > 0:
        splits_val = _split_features(X_val)
        X_dense_val, X_species_val, X_moves_val, X_items_val, X_abilities_val = splits_val
        aux_targets_val = extract_aux_targets_batch(X_next_val, num_species, num_moves)
        val_inputs = {
            "dense_features":   X_dense_val,
            "species_indices":  X_species_val,
            "move_indices":     X_moves_val,
            "item_indices":     X_items_val,
            "ability_indices":  X_abilities_val,
            "action_mask":      action_masks_val,
        }
        val_outputs = {
            "value": y_value_val,
            "policy": y_policy_val,
        }
        val_outputs.update(aux_targets_val)
        val_data = (val_inputs, val_outputs)

    print(
        f"  Dense: {X_dense_train.shape[1]}  |  Species: {num_species}  |  "
        f"Items: {num_items}  |  Abilities: {num_abilities}  |  Moves: {num_moves}"
    )

    losses = {
        "value": "mse",
        "policy": "categorical_crossentropy",
        "aux_field": "binary_crossentropy",
        "aux_own_hp": "mse",
        "aux_opp_hp": "mse",
        "aux_own_statuses": "binary_crossentropy",
        "aux_opp_statuses": "binary_crossentropy",
        "aux_own_boosts": "mse",
        "aux_opp_boosts": "mse",
        "aux_own_stats": "mse",
        "aux_opp_stats": "mse",
        "aux_own_types": "binary_crossentropy",
        "aux_opp_types": "binary_crossentropy",
        "aux_own_species": "categorical_crossentropy",
        "aux_opp_species": "categorical_crossentropy",
        "aux_own_moves": "binary_crossentropy",
        "aux_opp_moves": "binary_crossentropy",
        "meta_plan": "mse"
    }

    loss_weights = {
        "value": 1.0,
        "policy": 5.0,
        "aux_field": 0.2,
        "aux_own_hp": 0.5,
        "aux_opp_hp": 0.5,
        "aux_own_statuses": 0.3,
        "aux_opp_statuses": 0.3,
        "aux_own_boosts": 0.2,
        "aux_opp_boosts": 0.2,
        "aux_own_stats": 0.2,
        "aux_opp_stats": 0.2,
        "aux_own_types": 0.2,
        "aux_opp_types": 0.2,
        "aux_own_species": 0.5,
        "aux_opp_species": 0.5,
        "aux_own_moves": 0.5,
        "aux_opp_moves": 0.5,
        "meta_plan": 0.5
    }

    if os.path.exists(model_save_path):
        print(f"Loading existing model from {model_save_path}...")
        try:
            model = keras.models.load_model(
                model_save_path,
                compile=False,
                safe_mode=False,
                custom_objects={"SliceLayer": SliceLayer, "ConstantLayer": ConstantLayer, "ApplyMaskLayer": ApplyMaskLayer},
            )
            if len(model.outputs) != 18:
                raise ValueError(f"Architecture mismatch: expected 18 outputs, got {len(model.outputs)}")
            print("Successfully loaded existing model. Recompiling for fine-tuning...")
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss=losses,
                loss_weights=loss_weights,
                metrics={"value": "mae", "policy": "accuracy"}
            )
        except Exception as e:
            print(f"Model load failed ({e}). Building fresh model from scratch...")
            model = build_model(X_dense_train.shape[1], num_moves, num_species, num_items, num_abilities)
            print("Compiling model...")
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss=losses,
                loss_weights=loss_weights,
                metrics={"value": "mae", "policy": "accuracy"}
            )
    else:
        print("Building new model from scratch...")
        model = build_model(X_dense_train.shape[1], num_moves, num_species, num_items, num_abilities)
        print("Compiling model...")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=losses,
            loss_weights=loss_weights,
            metrics={"value": "mae", "policy": "accuracy"}
        )

    # 3. Compute Counterfactual targets for the Meta-Planner attention heads
    is_scratch = not os.path.exists(model_save_path)
    train_outputs["meta_plan"] = compute_counterfactual_targets(model, train_inputs, is_scratch=is_scratch)
    if val_data is not None:
        val_inputs, val_outputs = val_data
        val_outputs["meta_plan"] = compute_counterfactual_targets(model, val_inputs, is_scratch=is_scratch)
        val_data = (val_inputs, val_outputs)

    model.summary()

    callbacks = [PrimaryLossCallback()]
    
    # Save training history to CSV log file
    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    csv_log_path = save_path.parent / "training_log.csv"
    callbacks.append(keras.callbacks.CSVLogger(str(csv_log_path), append=True))
    
    if val_data is not None:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_primary_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ))

    model.fit(
        train_inputs,
        train_outputs,
        epochs=epochs,
        batch_size=512,
        validation_data=val_data,
        callbacks=callbacks,
    )

    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Model saved to {model_save_path}")

    # Export to ONNX
    onnx_path = save_path.with_suffix(".onnx")
    export_to_onnx(model, onnx_path)


if __name__ == "__main__":
    train()
