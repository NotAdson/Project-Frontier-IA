import os
from pathlib import Path

import keras
import numpy as np

from battle_agents.mcts_approximation.state_encoder import (
    ACTION_SPACE, FIELD_START, MAIN_EMB_ABILITY_DIM, MAIN_EMB_ITEMS_DIM,
    MAIN_EMB_MOVES_DIM, MAIN_EMB_SPECIES_DIM, META_EMB_ABILITY_DIM,
    META_EMB_ITEMS_DIM, META_EMB_MOVES_DIM, META_EMB_SPECIES_DIM,
    NUM_ABILITY_INDICES, NUM_ACTIVE, NUM_BENCH, NUM_BOOSTS, NUM_DENSE_FEATURES,
    NUM_EMBEDDING_INDICES, NUM_FIELD_FEATURES, NUM_ITEM_INDICES, NUM_MOVE_INDICES,
    NUM_MOVES, NUM_SPECIES_INDICES, NUM_STATUS, NUM_STATS, NUM_TYPES,
    OFF_ABILITIES, OFF_FAINTED, OFF_HP, OFF_IS_ACTIVE, OFF_ITEMS, OFF_LEVEL,
    OFF_MOVES, OFF_MOVES_DENSE, OFF_SPECIES, OFF_STATS, OFF_STATUSES, OFF_TYPES,
    OPP_BOOSTS_START, OPP_TEAM_START, OWN_BOOSTS_START, OWN_TEAM_DENSE,
    PER_MON_DENSE, PP_START, TOTAL_FEATURES
)

from battle_agents.mcts_approximation.db.python.knowledge_base import (
    get_species_reference_data,
)
from battle_agents.mcts_approximation.pipeline.differentiable_matching import (
    DifferentiableSpeciesMatching,
)

BASE_WEIGHT_SPECIES = 1.0
BASE_WEIGHT_STATS = 5.0
BASE_WEIGHT_TYPE = 5.0

TOTAL_BASE_WEIGHT = (
    BASE_WEIGHT_SPECIES
    + BASE_WEIGHT_STATS
    + BASE_WEIGHT_TYPE
)

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
    def call(self, inputs):
        logits, mask = inputs
        return logits + (1.0 - mask) * -1e9

    def get_config(self):
        return super().get_config()


def build_models(num_dense: int, num_moves: int, num_species: int,
                num_items: int, num_abilities: int) -> tuple[keras.Model, keras.Model]:
    # (function body copied from original train_nn.py)
    inp_dense     = keras.layers.Input(shape=(num_dense,), name="dense_features")
    inp_species   = keras.layers.Input(shape=(NUM_ACTIVE,), name="species_indices", dtype="int32")
    inp_moves     = keras.layers.Input(shape=(NUM_ACTIVE * NUM_MOVES,), name="move_indices",    dtype="int32")
    inp_items     = keras.layers.Input(shape=(NUM_ACTIVE,), name="item_indices",    dtype="int32")
    inp_abilities = keras.layers.Input(shape=(NUM_ACTIVE,), name="ability_indices", dtype="int32")
    inp_mask      = keras.layers.Input(shape=(len(ACTION_SPACE),), name="action_mask")

    emb_species_layer = keras.layers.Embedding(num_species, META_EMB_SPECIES_DIM, name="meta_emb_species")
    emb_moves_layer   = keras.layers.Embedding(num_moves, META_EMB_MOVES_DIM, name="meta_emb_moves")
    emb_items_layer   = keras.layers.Embedding(num_items, META_EMB_ITEMS_DIM, name="meta_emb_items")
    emb_abilities_layer = keras.layers.Embedding(num_abilities, META_EMB_ABILITY_DIM, name="meta_emb_abilities")

    field_conds = SliceLayer(FIELD_START, FIELD_START + NUM_FIELD_FEATURES, name="field_conditions")(inp_dense)

    tokens = []
    for i in range(NUM_ACTIVE):
        if i < NUM_BENCH:
            start_idx = i * PER_MON_DENSE
            owner_val = 1.0
        else:
            start_idx = OPP_TEAM_START + (i - NUM_BENCH) * PER_MON_DENSE
            owner_val = 0.0

        p_dense = SliceLayer(start_idx, start_idx + PER_MON_DENSE, name=f"p{i}_dense")(inp_dense)
        p_pp    = SliceLayer(PP_START + i * NUM_MOVES, PP_START + i * NUM_MOVES + NUM_MOVES, name=f"p{i}_pp")(inp_dense)
        p_spec = SliceLayer(i, i + 1, name=f"p{i}_species")(inp_species)
        p_item = SliceLayer(i, i + 1, name=f"p{i}_item")(inp_items)
        p_abil = SliceLayer(i, i + 1, name=f"p{i}_ability")(inp_abilities)

        m_start = i * NUM_MOVES if i < NUM_BENCH else (NUM_BENCH * NUM_MOVES) + (i - NUM_BENCH) * NUM_MOVES
        p_moves = SliceLayer(m_start, m_start + NUM_MOVES, name=f"p{i}_moves")(inp_moves)

        spec_emb = keras.layers.Flatten()(emb_species_layer(p_spec))
        item_emb = keras.layers.Flatten()(emb_items_layer(p_item))
        abil_emb = keras.layers.Flatten()(emb_abilities_layer(p_abil))
        moves_emb = keras.layers.Flatten()(emb_moves_layer(p_moves))

        owner_flag = ConstantLayer(owner_val, name=f"p{i}_owner")(inp_dense)

        token = keras.layers.Concatenate(name=f"p{i}_token")([
            p_dense, p_pp, spec_emb, item_emb, abil_emb, moves_emb, owner_flag, field_conds
        ])

        token_dim = (PER_MON_DENSE + NUM_MOVES + META_EMB_SPECIES_DIM
                     + META_EMB_ITEMS_DIM + META_EMB_ABILITY_DIM
                     + NUM_MOVES * META_EMB_MOVES_DIM + 1 + NUM_FIELD_FEATURES)
        token_expanded = keras.layers.Reshape((1, token_dim), name=f"p{i}_token_expanded")(token)
        tokens.append(token_expanded)

    token_seq = keras.layers.Concatenate(axis=1, name="token_sequence")(tokens)

    attn_out = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, name="meta_attention")(
        query=token_seq, value=token_seq
    )
    attn_out = keras.layers.Add()([token_seq, attn_out])
    attn_out = keras.layers.LayerNormalization()(attn_out)

    scores = keras.layers.Dense(1, name="token_score_projection")(attn_out)
    scores = keras.layers.Reshape((NUM_ACTIVE,), name="token_scores")(scores)

    own_scores = SliceLayer(0, NUM_BENCH, name="own_scores")(scores)
    opp_scores = SliceLayer(NUM_BENCH, name="opp_scores")(scores)

    own_weights = keras.layers.Activation("softmax", name="meta_own_weights")(own_scores)
    opp_weights = keras.layers.Activation("sigmoid", name="meta_opp_weights")(opp_scores)

    meta_plan = keras.layers.Concatenate(name="meta_plan")([own_weights, opp_weights])

    emb_species_main   = keras.layers.Flatten()(keras.layers.Embedding(num_species,   MAIN_EMB_SPECIES_DIM, name="emb_species")(inp_species))
    emb_moves_main     = keras.layers.Flatten()(keras.layers.Embedding(num_moves,     MAIN_EMB_MOVES_DIM, name="emb_moves")(inp_moves))
    emb_items_main     = keras.layers.Flatten()(keras.layers.Embedding(num_items,     MAIN_EMB_ITEMS_DIM, name="emb_items")(inp_items))
    emb_abilities_main = keras.layers.Flatten()(keras.layers.Embedding(num_abilities, MAIN_EMB_ABILITY_DIM, name="emb_abilities")(inp_abilities))

    concat_main = keras.layers.Concatenate()(
        [inp_dense, emb_species_main, emb_moves_main, emb_items_main, emb_abilities_main]
    )

    fused_features = keras.layers.Concatenate(name="fused_features")([concat_main, meta_plan])

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

    weight_species_logit = keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(
            np.log(BASE_WEIGHT_SPECIES)
        ),
        name="weight_species_logit",
    )(x)

    weight_stats_logit = keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(
            np.log(BASE_WEIGHT_STATS)
        ),
        name="weight_stats_logit",
    )(x)

    weight_type_logit = keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(
            np.log(BASE_WEIGHT_TYPE)
        ),
        name="weight_type_logit",
    )(x)

    weight_logits = keras.layers.Concatenate(name="weight_logits")([
        weight_species_logit,
        weight_stats_logit,
        weight_type_logit,
    ])
    weight_proportions = keras.layers.Activation("softmax", name="weight_proportions")(weight_logits)
    effective_weights = keras.layers.Rescaling(scale=TOTAL_BASE_WEIGHT, name="effective_weights")(weight_proportions)
    pred_weight_species = SliceLayer(0, 1, name="pred_weight_species",)(effective_weights)
    pred_weight_stats = SliceLayer(1, 2, name="pred_weight_stats")(effective_weights)
    pred_weight_type = SliceLayer(2, 3, name="pred_weight_type")(effective_weights)

    out_value  = keras.layers.Dense(1, activation="sigmoid", name="value")(x)
    
    logits = keras.layers.Dense(len(ACTION_SPACE), name="policy_logits")(x)
    masked_logits = ApplyMaskLayer(name="masked_logits")([logits, inp_mask])
    out_policy = keras.layers.Activation("softmax", name="policy")(masked_logits)

    out_field   = keras.layers.Dense(NUM_FIELD_FEATURES, activation="sigmoid", name="aux_field")(x)
    out_own_hp  = keras.layers.Dense(1, activation="sigmoid", name="aux_own_hp")(x)
    out_opp_hp  = keras.layers.Dense(1, activation="sigmoid", name="aux_opp_hp")(x)
    out_own_statuses = keras.layers.Dense(NUM_STATUS, activation="sigmoid", name="aux_own_statuses")(x)
    out_opp_statuses = keras.layers.Dense(NUM_STATUS, activation="sigmoid", name="aux_opp_statuses")(x)
    out_own_boosts = keras.layers.Dense(NUM_BOOSTS, activation="tanh", name="aux_own_boosts")(x)
    out_opp_boosts = keras.layers.Dense(NUM_BOOSTS, activation="tanh", name="aux_opp_boosts")(x)
    out_own_stats = keras.layers.Dense(NUM_STATS, activation="sigmoid", name="aux_own_stats")(x)
    out_opp_stats = keras.layers.Dense(NUM_STATS, activation="sigmoid", name="aux_opp_stats")(x)
    out_own_types = keras.layers.Dense(NUM_TYPES, activation="sigmoid", name="aux_own_types")(x)
    out_opp_types = keras.layers.Dense(NUM_TYPES, activation="sigmoid", name="aux_opp_types")(x)
    out_own_species = keras.layers.Dense(num_species, activation="softmax", name="aux_own_species")(x)
    out_opp_species = keras.layers.Dense(num_species, activation="softmax", name="aux_opp_species")(x)
    out_own_moves = keras.layers.Dense(num_moves, activation="sigmoid", name="aux_own_moves")(x)
    out_opp_moves = keras.layers.Dense(num_moves, activation="sigmoid", name="aux_opp_moves")(x)

    (
        real_stats_matrix,
        real_types_matrix,
        valid_species_mask,
    ) = get_species_reference_data()

    zero_weight = ConstantLayer(
        0.0,
        name="zero_matching_weight",
    )(pred_weight_species)

    weight_species_output = DifferentiableSpeciesMatching(
        real_stats_matrix=real_stats_matrix,
        real_types_matrix=real_types_matrix,
        valid_species_mask=valid_species_mask,
        temperature=1.0,
        name="weight_species",
    )([
        out_opp_species,
        out_opp_stats,
        out_opp_types,
        pred_weight_species,
        zero_weight,
        zero_weight,
    ])

    weight_stats_output = DifferentiableSpeciesMatching(
        real_stats_matrix=real_stats_matrix,
        real_types_matrix=real_types_matrix,
        valid_species_mask=valid_species_mask,
        temperature=1.0,
        name="weight_stats",
    )([
        out_opp_species,
        out_opp_stats,
        out_opp_types,
        zero_weight,
        pred_weight_stats,
        zero_weight,
    ])

    weight_type_output = DifferentiableSpeciesMatching(
        real_stats_matrix=real_stats_matrix,
        real_types_matrix=real_types_matrix,
        valid_species_mask=valid_species_mask,
        temperature=1.0,
        name="weight_type",
    )([
        out_opp_species,
        out_opp_stats,
        out_opp_types,
        zero_weight,
        zero_weight,
        pred_weight_type,
    ])

    dynamic_matching_output = DifferentiableSpeciesMatching(
        real_stats_matrix=real_stats_matrix,
        real_types_matrix=real_types_matrix,
        valid_species_mask=valid_species_mask,
        temperature=1.0,
        name="dynamic_matching",
    )([
        out_opp_species,
        out_opp_stats,
        out_opp_types,
        pred_weight_species,
        pred_weight_stats,
        pred_weight_type,
    ])

    model_inputs = [inp_dense, inp_species, inp_moves, inp_items, inp_abilities, inp_mask]

    base_outputs = [
        out_value,
        out_policy,
        out_field,
        out_own_hp,
        out_opp_hp,
        out_own_statuses,
        out_opp_statuses,
        out_own_boosts,
        out_opp_boosts,
        out_own_stats,
        out_opp_stats,
        out_own_types,
        out_opp_types,
        out_own_species,
        out_opp_species,
        out_own_moves,
        out_opp_moves,
        meta_plan,
    ]

    # Model that will be saved and used by the agent
    inference_model = keras.Model(
        inputs=model_inputs,
        outputs=[
            *base_outputs,

            pred_weight_species,
            pred_weight_stats,
            pred_weight_type,
        ],
        name="frontier_inference_model",
    )

    # Model used only during model.fit()
    training_model = keras.Model(
        inputs=model_inputs,
        outputs=[
            *base_outputs,

            weight_species_output,
            weight_stats_output,
            weight_type_output,
            dynamic_matching_output,
        ],
        name="frontier_training_model",
    )

    return inference_model, training_model


def export_to_onnx(model, onnx_path):
    print(f"Exporting model to ONNX format at {onnx_path}...")

    dummy = {
        "dense_features": np.zeros((1, NUM_DENSE_FEATURES), dtype=np.float32),
        "species_indices": np.zeros((1, NUM_ACTIVE), dtype=np.int32),
        "move_indices": np.zeros((1, NUM_ACTIVE * NUM_MOVES), dtype=np.int32),
        "item_indices": np.zeros((1, NUM_ACTIVE), dtype=np.int32),
        "ability_indices": np.zeros((1, NUM_ACTIVE), dtype=np.int32),
        "action_mask": np.ones((1, len(ACTION_SPACE)), dtype=np.float32),
    }
    try:
        model(dummy, training=False)
    except Exception:
        pass

    backend = keras.backend.backend()

    try:
        model.export(str(onnx_path), format="onnx")
        print("ONNX export completed successfully using model.export!")
        return True
    except Exception as e:
        print(f"[Warning] Direct model.export to ONNX failed: {e}")

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
                            "meta_plan", "pred_weight_species", "pred_weight_stats",
                            "pred_weight_type"]
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
                tf.TensorSpec((None, NUM_ACTIVE), tf.int32, name="species_indices"),
                tf.TensorSpec((None, NUM_ACTIVE * NUM_MOVES), tf.int32, name="move_indices"),
                tf.TensorSpec((None, NUM_ACTIVE), tf.int32, name="item_indices"),
                tf.TensorSpec((None, NUM_ACTIVE), tf.int32, name="ability_indices"),
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


LOSSES = {
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
    "meta_plan": "mse",
}

TRAINING_LOSSES = {
    **LOSSES,

    "weight_species": "categorical_crossentropy",
    "weight_stats": "categorical_crossentropy",
    "weight_type": "categorical_crossentropy",
    "dynamic_matching": "categorical_crossentropy",
}

LOSS_WEIGHTS = {
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
    "meta_plan": 0.5,
}

TRAINING_LOSS_WEIGHTS = {
    **LOSS_WEIGHTS,

    "weight_species": 0.1,
    "weight_stats": 0.1,
    "weight_type": 0.1,
    "dynamic_matching": 0.1,
}

def get_custom_objects():
    return {"SliceLayer": SliceLayer, "ConstantLayer": ConstantLayer, "ApplyMaskLayer": ApplyMaskLayer}
