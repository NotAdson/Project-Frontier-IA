"""
Trains the Neural Network for MCTSApproximationAgent with Action-Masked Softmax and Auxiliary next-state heads.

Backend selection:
  Set KERAS_BACKEND env-var before running:
    KERAS_BACKEND=tensorflow   (default when TF is installed)
    KERAS_BACKEND=torch
    KERAS_BACKEND=jax
"""
import os
import json
from pathlib import Path
import numpy as np
import keras

from battle_agents.mcts_approximation.db.moves_db import get_num_moves
from battle_agents.mcts_approximation.db.species_db import get_num_species, get_num_items, get_num_abilities
from battle_agents.mcts_approximation.state_encoder import (
    NUM_DENSE_FEATURES,
    NUM_EMBEDDING_INDICES,
    TOTAL_FEATURES,
    NUM_SPECIES_INDICES, NUM_MOVE_INDICES, NUM_ITEM_INDICES, NUM_ABILITY_INDICES,
    OFF_SPECIES, OFF_MOVES, OFF_ITEMS, OFF_ABILITIES,
    ACTION_SPACE,
)


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
    for f in files:
        try:
            gx, gv, gp, gn, gm = _parse_steps(json.loads(f.read_text()))
            X += gx; y_value += gv; y_policy += gp; X_next += gn; action_masks += gm
        except Exception as e:
            print(f"Error loading file {f}: {e}")
            
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


def extract_aux_targets_batch(X_next, num_species):
    batch_size = X_next.shape[0]
    
    # 1. Field conditions (Field indices 636 to 654)
    field = X_next[:, 636:654]
    
    # 2. Boosts
    own_boosts = X_next[:, 312:318]
    opp_boosts = X_next[:, 630:636]
    
    # Initialize active targets
    own_hp = np.zeros((batch_size, 1), dtype=np.float32)
    opp_hp = np.zeros((batch_size, 1), dtype=np.float32)
    
    own_statuses = np.zeros((batch_size, 5), dtype=np.float32)
    opp_statuses = np.zeros((batch_size, 5), dtype=np.float32)
    
    own_stats = np.zeros((batch_size, 5), dtype=np.float32)
    opp_stats = np.zeros((batch_size, 5), dtype=np.float32)
    
    own_types = np.zeros((batch_size, 18), dtype=np.float32)
    opp_types = np.zeros((batch_size, 18), dtype=np.float32)
    
    own_species = np.zeros((batch_size,), dtype=np.int32)
    opp_species = np.zeros((batch_size,), dtype=np.int32)
    
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
            own_species[b] = np.clip(int(round(X_next[b, 654 + own_act])), 0, num_species - 1)
            
        if opp_act != -1:
            opp_hp[b, 0] = X_next[b, 318 + opp_act * 52]
            opp_statuses[b, :] = X_next[b, 318 + opp_act * 52 + 2 : 318 + opp_act * 52 + 7]
            opp_stats[b, :] = X_next[b, 318 + opp_act * 52 + 9 : 318 + opp_act * 52 + 14]
            opp_types[b, :] = X_next[b, 318 + opp_act * 52 + 14 : 318 + opp_act * 52 + 32]
            opp_species[b] = np.clip(int(round(X_next[b, 654 + 6 + opp_act])), 0, num_species - 1)
            
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

    # Slice out 18 field conditions (weather, hazards) at the end of dense features [636:654]
    field_conds = keras.layers.Lambda(lambda x: x[:, 636:654], name="field_conditions")(inp_dense)

    tokens = []
    for i in range(12):
        if i < 6:
            start_idx = i * 52
            owner_val = 1.0
        else:
            start_idx = 318 + (i - 6) * 52
            owner_val = 0.0
        
        # 1. Dense features for Pokémon i
        p_dense = keras.layers.Lambda(lambda x, idx=start_idx: x[:, idx : idx + 52], name=f"p{i}_dense")(inp_dense)
        # 2. Categorical embeddings
        p_spec = keras.layers.Lambda(lambda x, idx=i: x[:, idx:idx+1], name=f"p{i}_species")(inp_species)
        p_item = keras.layers.Lambda(lambda x, idx=i: x[:, idx:idx+1], name=f"p{i}_item")(inp_items)
        p_abil = keras.layers.Lambda(lambda x, idx=i: x[:, idx:idx+1], name=f"p{i}_ability")(inp_abilities)
        
        # 4 moves per Pokémon
        m_start = i * 4 if i < 6 else 24 + (i - 6) * 4
        p_moves = keras.layers.Lambda(lambda x, idx=m_start: x[:, idx : idx + 4], name=f"p{i}_moves")(inp_moves)

        spec_emb = keras.layers.Flatten()(emb_species_layer(p_spec))
        item_emb = keras.layers.Flatten()(emb_items_layer(p_item))
        abil_emb = keras.layers.Flatten()(emb_abilities_layer(p_abil))
        moves_emb = keras.layers.Flatten()(emb_moves_layer(p_moves))

        # Owner flag
        owner_flag = keras.layers.Lambda(lambda x, val=owner_val: x[:, :1] * 0.0 + val, name=f"p{i}_owner")(inp_dense)

        # Concatenate features to form a 135-dimensional Pokémon representation token
        token = keras.layers.Concatenate(name=f"p{i}_token")([
            p_dense, spec_emb, item_emb, abil_emb, moves_emb, owner_flag, field_conds
        ])
        
        # Expand token to shape (None, 1, 135) for sequence format
        token_expanded = keras.layers.Reshape((1, 135), name=f"p{i}_token_expanded")(token)
        tokens.append(token_expanded)

    # Sequence of 12 tokens: shape (None, 12, 135)
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
    own_scores = keras.layers.Lambda(lambda x: x[:, :6], name="own_scores")(scores)
    opp_scores = keras.layers.Lambda(lambda x: x[:, 6:], name="opp_scores")(scores)

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

    # Trunk
    x = keras.layers.Dense(512)(fused_features)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(128)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.1)(x)

    # Core outputs
    out_value  = keras.layers.Dense(1, activation="sigmoid", name="value")(x)
    
    # Policy head with Action-Masked Softmax
    logits = keras.layers.Dense(len(ACTION_SPACE), name="policy_logits")(x)
    masked_logits = keras.layers.Lambda(
        lambda inputs: inputs[0] + (1.0 - inputs[1]) * -1e9,
        name="masked_logits"
    )([logits, inp_mask])
    out_policy = keras.layers.Activation("softmax", name="policy")(masked_logits)

    # Auxiliary transition/dynamics outputs
    out_field   = keras.layers.Dense(18, activation="sigmoid", name="aux_field")(x)
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

    model = keras.Model(
        inputs=[inp_dense, inp_species, inp_moves, inp_items, inp_abilities, inp_mask],
        outputs=[
            out_value, out_policy,
            out_field, out_own_hp, out_opp_hp,
            out_own_statuses, out_opp_statuses,
            out_own_boosts, out_opp_boosts,
            out_own_stats, out_opp_stats,
            out_own_types, out_opp_types,
            out_own_species, out_opp_species
        ],
    )
    return model


def export_to_onnx(model, onnx_path):
    print(f"Exporting model to ONNX format at {onnx_path}...")
    
    # 1. Try Keras 3 direct model.export
    try:
        model.export(str(onnx_path), format="onnx")
        print("ONNX export completed successfully using model.export!")
        return True
    except Exception as e:
        print(f"[Warning] Direct model.export to ONNX failed: {e}")

    # 2. Try torch backend specific export if Keras is using torch backend
    try:
        import keras
        if keras.backend.backend() == "torch":
            print("Keras is using 'torch' backend. Attempting torch.onnx.export...")
            import torch
            
            dummy_inputs = (
                torch.zeros((1, NUM_DENSE_FEATURES), dtype=torch.float32),  # dense_features
                torch.zeros((1, 12), dtype=torch.int32),    # species_indices
                torch.zeros((1, 48), dtype=torch.int32),    # move_indices
                torch.zeros((1, 12), dtype=torch.int32),    # item_indices
                torch.zeros((1, 12), dtype=torch.int32),    # ability_indices
                torch.zeros((1, len(ACTION_SPACE)), dtype=torch.float32),  # action_mask
            )
            
            torch.onnx.export(
                model,
                dummy_inputs,
                str(onnx_path),
                input_names=[
                    "dense_features", "species_indices", "move_indices", "item_indices", "ability_indices", "action_mask"
                ],
                output_names=[
                    "value", "policy", "aux_field", "aux_own_hp", "aux_opp_hp", "aux_own_statuses", "aux_opp_statuses",
                    "aux_own_boosts", "aux_opp_boosts", "aux_own_stats", "aux_opp_stats", "aux_own_types", "aux_opp_types",
                    "aux_own_species", "aux_opp_species"
                ],
                dynamic_axes={
                    "dense_features": {0: "batch_size"},
                    "species_indices": {0: "batch_size"},
                    "move_indices": {0: "batch_size"},
                    "item_indices": {0: "batch_size"},
                    "ability_indices": {0: "batch_size"},
                    "action_mask": {0: "batch_size"},
                    "value": {0: "batch_size"},
                    "policy": {0: "batch_size"},
                },
                opset_version=14,
            )
            print("ONNX export completed successfully using torch.onnx.export!")
            return True
    except Exception as e_torch:
        print(f"[Warning] torch.onnx.export failed: {e_torch}")

    # 3. Try tensorflow backend specific export if Keras is using tensorflow backend
    try:
        import keras
        if keras.backend.backend() == "tensorflow":
            print("Keras is using 'tensorflow' backend. Attempting tf2onnx conversion...")
            import tensorflow as tf
            import tf2onnx
            
            spec = (
                tf.TensorSpec((None, NUM_DENSE_FEATURES), tf.float32, name="dense_features"),
                tf.TensorSpec((None, 12), tf.int32, name="species_indices"),
                tf.TensorSpec((None, 48), tf.int32, name="move_indices"),
                tf.TensorSpec((None, 12), tf.int32, name="item_indices"),
                tf.TensorSpec((None, 12), tf.int32, name="ability_indices"),
                tf.TensorSpec((None, len(ACTION_SPACE)), tf.float32, name="action_mask"),
            )
            
            tf2onnx.convert.from_keras(
                model,
                input_signature=spec,
                opset=14,
                output_path=str(onnx_path)
            )
            print("ONNX export completed successfully using tf2onnx!")
            return True
    except Exception as e_tf:
        print(f"[Warning] tf2onnx export failed: {e_tf}")

    print("[Error] All ONNX export methods failed.")
    return False


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

    num_moves     = get_num_moves()
    num_species   = get_num_species()
    num_items     = get_num_items()
    num_abilities = get_num_abilities()

    # Extract auxiliary targets
    aux_targets_train = extract_aux_targets_batch(X_next_train, num_species)
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
        aux_targets_val = extract_aux_targets_batch(X_next_val, num_species)
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
        "aux_opp_species": "categorical_crossentropy"
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
        "aux_opp_species": 0.5
    }

    if os.path.exists(model_save_path):
        print(f"Loading existing model from {model_save_path}...")
        try:
            # Try to load directly (if model was already built/saved with action_mask and aux heads and deserialization succeeds)
            model = keras.models.load_model(model_save_path, compile=False)
            if len(model.outputs) != 15:
                raise ValueError(f"Architecture mismatch: expected 15 outputs, got {len(model.outputs)}")
            print("Successfully loaded existing model with matching architecture. Recompiling for fine-tuning...")
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss=losses,
                loss_weights=loss_weights,
                metrics={"value": "mae", "policy": "accuracy"}
            )
        except Exception as e:
            print(f"Direct model load failed or mismatched architecture: {e}. Building new model with mask and aux heads, loading weights directly...")
            model = build_model(X_dense_train.shape[1], num_moves, num_species, num_items, num_abilities)
            model.load_weights(model_save_path, skip_mismatch=True)
            print("Successfully loaded weights (using skip_mismatch=True). Compiling for fine-tuning...")
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
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

    model.summary()

    callbacks = []
    if val_data is not None:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss",
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
