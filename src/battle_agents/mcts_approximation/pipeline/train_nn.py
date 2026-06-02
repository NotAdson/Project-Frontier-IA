"""
Trains the Neural Network for MCTSApproximationAgent.

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
    X, y_value, y_policy = [], [], []
    for step in game_data:
        X.append(step["features"])
        y_value.append(step["value"])
        policy_dict = step.get("policy", {})
        policy_array = [policy_dict.get(a, 0.0) for a in ACTION_SPACE]
        s = sum(policy_array)
        policy_array = [p / s for p in policy_array] if s > 0 else [1.0 / len(ACTION_SPACE)] * len(ACTION_SPACE)
        y_policy.append(policy_array)
    return X, y_value, y_policy


def load_data(data_dir: str = "data/games"):
    X, y_value, y_policy = [], [], []
    for f in Path(data_dir).glob("*.json"):
        gx, gv, gp = _parse_steps(json.loads(f.read_text()))
        X += gx; y_value += gv; y_policy += gp
    return np.array(X, dtype=np.float32), np.array(y_value, dtype=np.float32), np.array(y_policy, dtype=np.float32)


def load_data_from_files(files):
    X, y_value, y_policy = [], [], []
    for f in files:
        gx, gv, gp = _parse_steps(json.loads(f.read_text()))
        X += gx; y_value += gv; y_policy += gp
    if len(X) == 0:
        return (np.empty((0, TOTAL_FEATURES), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, len(ACTION_SPACE)), dtype=np.float32))
    return np.array(X, dtype=np.float32), np.array(y_value, dtype=np.float32), np.array(y_policy, dtype=np.float32)


def build_model(num_dense: int, num_moves: int, num_species: int,
                num_items: int, num_abilities: int) -> keras.Model:
    """Builds the multi-input value+policy network (5 inputs, 2 outputs)."""

    # Grouped Category Inputs
    inp_dense     = keras.layers.Input(shape=(num_dense,), name="dense_features")
    inp_species   = keras.layers.Input(shape=(12,), name="species_indices", dtype="int32")
    inp_moves     = keras.layers.Input(shape=(48,), name="move_indices",    dtype="int32")
    inp_items     = keras.layers.Input(shape=(12,), name="item_indices",    dtype="int32")
    inp_abilities = keras.layers.Input(shape=(12,), name="ability_indices", dtype="int32")

    # Embedding branches (shared vocabulary sizes per category)
    emb_species   = keras.layers.Flatten()(keras.layers.Embedding(num_species,   32, name="emb_species")(inp_species))
    emb_moves     = keras.layers.Flatten()(keras.layers.Embedding(num_moves,     32, name="emb_moves")(inp_moves))
    emb_items     = keras.layers.Flatten()(keras.layers.Embedding(num_items,     16, name="emb_items")(inp_items))
    emb_abilities = keras.layers.Flatten()(keras.layers.Embedding(num_abilities, 16, name="emb_abilities")(inp_abilities))

    concat = keras.layers.Concatenate()(
        [inp_dense, emb_species, emb_moves, emb_items, emb_abilities]
    )

    # Feed-forward trunk
    x = keras.layers.Dense(512)(concat)
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

    # Heads
    out_value  = keras.layers.Dense(1, activation="sigmoid", name="value")(x)
    out_policy = keras.layers.Dense(len(ACTION_SPACE), activation="softmax", name="policy")(x)

    model = keras.Model(
        inputs=[inp_dense, inp_species, inp_moves, inp_items, inp_abilities],
        outputs=[out_value, out_policy],
    )
    model.compile(
        optimizer="adam",
        loss={"value": "mse", "policy": "categorical_crossentropy"},
        loss_weights={"value": 1.0, "policy": 1.0},
        metrics={"value": "mae", "policy": "accuracy"},
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
            )
            
            torch.onnx.export(
                model,
                dummy_inputs,
                str(onnx_path),
                input_names=[
                    "dense_features", "species_indices", "move_indices", "item_indices", "ability_indices"
                ],
                output_names=["value", "policy"],
                dynamic_axes={
                    "dense_features": {0: "batch_size"},
                    "species_indices": {0: "batch_size"},
                    "move_indices": {0: "batch_size"},
                    "item_indices": {0: "batch_size"},
                    "ability_indices": {0: "batch_size"},
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
    all_files = list(Path(data_dir).glob("gen*/*.json"))
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
    X_train, y_value_train, y_policy_train = load_data_from_files(train_files)
    print("Loading validation data...")
    X_val, y_value_val, y_policy_val       = load_data_from_files(val_files)

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
    def _shuffle(X, yv, yp):
        idx = np.random.permutation(len(X))
        return X[idx], yv[idx], yp[idx]

    X_train, y_value_train, y_policy_train = _shuffle(X_train, y_value_train, y_policy_train)
    if len(X_val) > 0:
        X_val, y_value_val, y_policy_val = _shuffle(X_val, y_value_val, y_policy_val)

    # Split into 5 named inputs
    splits_train = _split_features(X_train)
    X_dense_train, X_species_train, X_moves_train, X_items_train, X_abilities_train = splits_train

    val_data = None
    if len(X_val) > 0:
        X_dense_val, X_species_val, X_moves_val, X_items_val, X_abilities_val = _split_features(X_val)
        val_data = (
            {
                "dense_features":   X_dense_val,
                "species_indices":  X_species_val,
                "move_indices":     X_moves_val,
                "item_indices":     X_items_val,
                "ability_indices":  X_abilities_val,
            },
            {"value": y_value_val, "policy": y_policy_val},
        )

    num_moves     = get_num_moves()
    num_species   = get_num_species()
    num_items     = get_num_items()
    num_abilities = get_num_abilities()

    print(
        f"  Dense: {X_dense_train.shape[1]}  |  Species: {num_species}  |  "
        f"Items: {num_items}  |  Abilities: {num_abilities}  |  Moves: {num_moves}"
    )

    if os.path.exists(model_save_path):
        print(f"Loading existing model from {model_save_path} for fine-tuning...")
        model = keras.models.load_model(model_save_path)
        print("Recompiling with lower learning rate for fine-tuning (1e-4)...")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss={"value": "mse", "policy": "categorical_crossentropy"},
            loss_weights={"value": 1.0, "policy": 1.0},
            metrics={"value": "mae", "policy": "accuracy"},
        )
    else:
        print("Building new model from scratch...")
        model = build_model(X_dense_train.shape[1], num_moves, num_species, num_items, num_abilities)

    model.summary()

    callbacks = []
    if val_data is not None:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_policy_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ))

    model.fit(
        {
            "dense_features":   X_dense_train,
            "species_indices":  X_species_train,
            "move_indices":     X_moves_train,
            "item_indices":     X_items_train,
            "ability_indices":  X_abilities_train,
        },
        {"value": y_value_train, "policy": y_policy_train},
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
