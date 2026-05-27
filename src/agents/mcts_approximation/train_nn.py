"""
Trains the Neural Network value and policy functions for MCTSApproximationAgent.

The model uses a multi-input Functional API (8 inputs):
  Own side:
    - dense_features    : 163 continuous features
    - species_indices   : 6 integers   → Embedding(num_species, 16)
    - item_indices      : 6 integers   → Embedding(num_items,    8)
    - ability_indices   : 6 integers   → Embedding(num_abilities, 8)
    - bench_move_indices: 24 integers  → Embedding(num_moves,   16)
    - move_indices      : 4 integers   → Embedding(num_moves,   16)
  Opponent (publicly revealed only):
    - opp_species_indices: 6 integers  → Embedding(num_species, 16)
    - opp_move_indices   : 24 integers → Embedding(num_moves,   16)

Outputs:
  - value: sigmoid scalar in [0, 1] representing P(player wins from this state).
  - policy: softmax vector of size 11 representing P(action | state).
"""
import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

from agents.mcts_approximation.moves_db import get_num_moves
from agents.mcts_approximation.species_db import get_num_species, get_num_items, get_num_abilities
from agents.mcts_approximation.state_encoder import (
    NUM_DENSE_FEATURES,
    NUM_EMBEDDING_INDICES,
    TOTAL_FEATURES,
    NUM_BENCH_MOVE_INDICES,
    NUM_OPP_SPECIES_INDICES,
    NUM_OPP_MOVE_INDICES,
    OFF_SPECIES, OFF_ITEMS, OFF_ABILITIES, OFF_BENCH_MOVES, OFF_ACTIVE_MOVES,
    OFF_OPP_SPECIES, OFF_OPP_MOVES,
    NUM_SPECIES_INDICES, NUM_ITEM_INDICES, NUM_ABILITY_INDICES, NUM_ACTIVE_MOVE_INDICES,
    ACTION_SPACE,
)




def _split_features(X: np.ndarray):
    """
    Splits the flat feature array into the 8 model inputs.
    Embedding index layout (after the 163 dense features):
        [163:169] species           (6)
        [169:175] items             (6)
        [175:181] abilities         (6)
        [181:205] bench_moves       (6 × 4 = 24)
        [205:209] active_moves      (4)
        [209:215] opp_species       (6, 0 if not revealed)
        [215:239] opp_used_moves    (6 × 4 = 24, 0 if not yet seen)
    """
    n = NUM_DENSE_FEATURES
    X_dense        = X[:, :n].astype(np.float32)
    X_species      = X[:, n + OFF_SPECIES     : n + OFF_SPECIES     + NUM_SPECIES_INDICES    ].astype(np.int32)
    X_items        = X[:, n + OFF_ITEMS       : n + OFF_ITEMS       + NUM_ITEM_INDICES       ].astype(np.int32)
    X_abilities    = X[:, n + OFF_ABILITIES   : n + OFF_ABILITIES   + NUM_ABILITY_INDICES    ].astype(np.int32)
    X_bench_moves  = X[:, n + OFF_BENCH_MOVES : n + OFF_BENCH_MOVES + NUM_BENCH_MOVE_INDICES ].astype(np.int32)
    X_moves        = X[:, n + OFF_ACTIVE_MOVES: n + OFF_ACTIVE_MOVES + NUM_ACTIVE_MOVE_INDICES].astype(np.int32)
    X_opp_species  = X[:, n + OFF_OPP_SPECIES : n + OFF_OPP_SPECIES + NUM_OPP_SPECIES_INDICES].astype(np.int32)
    X_opp_moves    = X[:, n + OFF_OPP_MOVES   : n + OFF_OPP_MOVES   + NUM_OPP_MOVE_INDICES   ].astype(np.int32)
    return X_dense, X_species, X_items, X_abilities, X_bench_moves, X_moves, X_opp_species, X_opp_moves


def load_data(data_dir: str = "data/games"):
    X, y_value, y_policy = [], [], []
    for f in Path(data_dir).glob("*.json"):
        game_data = json.loads(f.read_text())
        for step in game_data:
            X.append(step["features"])
            y_value.append(step["value"])
            
            # Extract policy distribution as a fixed-size vector
            policy_dict = step.get("policy", {})
            policy_array = [policy_dict.get(action, 0.0) for action in ACTION_SPACE]
            s = sum(policy_array)
            if s > 0:
                policy_array = [p / s for p in policy_array]
            else:
                policy_array = [1.0 / len(ACTION_SPACE)] * len(ACTION_SPACE)
                
            y_policy.append(policy_array)
            
    return np.array(X, dtype=np.float32), np.array(y_value, dtype=np.float32), np.array(y_policy, dtype=np.float32)


def build_model(num_dense: int, num_moves: int, num_species: int,
                num_items: int, num_abilities: int) -> tf.keras.Model:
    """Builds the multi-input value network (8 inputs)."""

    # Own side inputs
    inp_dense        = tf.keras.layers.Input(shape=(num_dense,), name="dense_features")
    inp_species      = tf.keras.layers.Input(shape=(6,),  name="species_indices",    dtype="int32")
    inp_items        = tf.keras.layers.Input(shape=(6,),  name="item_indices",       dtype="int32")
    inp_abilities    = tf.keras.layers.Input(shape=(6,),  name="ability_indices",    dtype="int32")
    inp_bench_moves  = tf.keras.layers.Input(shape=(24,), name="bench_move_indices", dtype="int32")
    inp_moves        = tf.keras.layers.Input(shape=(4,),  name="move_indices",       dtype="int32")
    # Opponent side inputs (public only — 0 for unrevealed slots)
    inp_opp_species  = tf.keras.layers.Input(shape=(6,),  name="opp_species_indices", dtype="int32")
    inp_opp_moves    = tf.keras.layers.Input(shape=(24,), name="opp_move_indices",    dtype="int32")

    # Own embedding branches
    emb_species      = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(num_species,   16, name="emb_species")(inp_species))
    emb_items        = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(num_items,      8, name="emb_items")(inp_items))
    emb_abilities    = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(num_abilities,  8, name="emb_abilities")(inp_abilities))
    emb_bench_moves  = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(num_moves,     16, name="emb_bench_moves")(inp_bench_moves))
    emb_moves        = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(num_moves,     16, name="emb_moves")(inp_moves))
    # Opponent embedding branches (shared vocabs, separate weights)
    emb_opp_species  = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(num_species,   16, name="emb_opp_species")(inp_opp_species))
    emb_opp_moves    = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(num_moves,     16, name="emb_opp_moves")(inp_opp_moves))

    # Concatenate everything
    concat = tf.keras.layers.Concatenate()(
        [inp_dense,
         emb_species, emb_items, emb_abilities, emb_bench_moves, emb_moves,
         emb_opp_species, emb_opp_moves]
    )

    # Feed-forward trunk
    x = tf.keras.layers.Dense(256, activation="relu")(concat)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64,  activation="relu")(x)
    
    # Value Head (Probability of winning)
    out_value = tf.keras.layers.Dense(1, activation="sigmoid", name="value")(x)
    
    # Policy Head (Probability distribution over actions)
    out_policy = tf.keras.layers.Dense(len(ACTION_SPACE), activation="softmax", name="policy")(x)

    model = tf.keras.models.Model(
        inputs=[inp_dense, inp_species, inp_items, inp_abilities,
                inp_bench_moves, inp_moves, inp_opp_species, inp_opp_moves],
        outputs=[out_value, out_policy],
    )
    model.compile(
        optimizer="adam", 
        loss={"value": "mse", "policy": "categorical_crossentropy"},
        loss_weights={"value": 1.0, "policy": 1.0},
        metrics={"value": "mae", "policy": "accuracy"}
    )
    return model


def train(data_dir: str = "data/games", model_save_path: str = "data/mcts_model.keras"):
    print(f"Loading data from {data_dir}...")
    X, y_value, y_policy = load_data(data_dir)

    if len(X) == 0:
        print("No data found. Please run generate_data.py first.")
        return

    print(f"Loaded {len(X)} samples. Total features per sample: {X.shape[1]} (expected {TOTAL_FEATURES})")

    if X.shape[1] != TOTAL_FEATURES:
        print(
            f"[WARNING] Feature count mismatch: got {X.shape[1]}, expected {TOTAL_FEATURES}.\n"
            "Your saved data was generated with the old encoder. "
            "Please delete data/games/ and re-run generate_data.py."
        )
        return

    X_dense, X_species, X_items, X_abilities, X_bench_moves, X_moves, X_opp_species, X_opp_moves = _split_features(X)

    num_moves     = get_num_moves()
    num_species   = get_num_species()
    num_items     = get_num_items()
    num_abilities = get_num_abilities()

    print(
        f"  Dense: {X_dense.shape[1]}  |  "
        f"Species vocab: {num_species}  |  Items vocab: {num_items}  |  "
        f"Abilities vocab: {num_abilities}  |  Moves vocab: {num_moves}"
    )

    if os.path.exists(model_save_path):
        print(f"Loading existing model from {model_save_path} for fine-tuning...")
        model = tf.keras.models.load_model(model_save_path)
        # Optional: reduce learning rate for fine-tuning if desired
        # tf.keras.backend.set_value(model.optimizer.learning_rate, 0.0001)
    else:
        print("Building new model from scratch...")
        model = build_model(X_dense.shape[1], num_moves, num_species, num_items, num_abilities)
        
    model.summary()

    model.fit(
        {
            "dense_features":     X_dense,
            "species_indices":    X_species,
            "item_indices":       X_items,
            "ability_indices":    X_abilities,
            "bench_move_indices": X_bench_moves,
            "move_indices":       X_moves,
            "opp_species_indices": X_opp_species,
            "opp_move_indices":   X_opp_moves,
        },
        {"value": y_value, "policy": y_policy},
        epochs=15,
        batch_size=64,
        validation_split=0.2,
    )

    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    train()
