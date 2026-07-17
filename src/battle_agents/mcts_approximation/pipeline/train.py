import os
from pathlib import Path

import numpy as np
import keras

from battle_agents.mcts_approximation.db.python.database import db
from battle_agents.mcts_approximation.state_encoder import TOTAL_FEATURES

from .data import load_data_from_files, split_features
from .aux_targets import extract_aux_targets_batch, compute_counterfactual_targets
from .model import build_model, export_to_onnx, PrimaryLossCallback, LOSSES, LOSS_WEIGHTS, get_custom_objects


def train(data_dir: str = "data", model_save_path: str = "data/mcts_model.keras", max_games_buffer: int = 2500, epochs: int = 15):
    print(f"[Keras backend: {keras.backend.backend()}]")
    print(f"Locating game files in {data_dir}/gen*...")
    all_files = list(Path(data_dir).glob("gen*/game_*.json"))
    if len(all_files) == 0:
        print("No data found. Please run generate_data.py first.")
        return

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

    splits_train = split_features(X_train)
    X_dense_train, X_species_train, X_moves_train, X_items_train, X_abilities_train = splits_train

    num_moves     = db.get_num_moves()
    num_species   = db.get_num_species()
    num_items     = db.get_num_items()
    num_abilities = db.get_num_abilities()

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
        splits_val = split_features(X_val)
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

    if os.path.exists(model_save_path):
        print(f"Loading existing model from {model_save_path}...")
        try:
            model = keras.models.load_model(
                model_save_path,
                compile=False,
                safe_mode=False,
                custom_objects=get_custom_objects(),
            )
            if len(model.outputs) != 18:
                raise ValueError(f"Architecture mismatch: expected 18 outputs, got {len(model.outputs)}")
            print("Successfully loaded existing model. Recompiling for fine-tuning...")
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss=LOSSES,
                loss_weights=LOSS_WEIGHTS,
                metrics={"value": "mae", "policy": "accuracy"}
            )
        except Exception as e:
            print(f"Model load failed ({e}). Building fresh model from scratch...")
            model = build_model(X_dense_train.shape[1], num_moves, num_species, num_items, num_abilities)
            print("Compiling model...")
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss=LOSSES,
                loss_weights=LOSS_WEIGHTS,
                metrics={"value": "mae", "policy": "accuracy"}
            )
    else:
        print("Building new model from scratch...")
        model = build_model(X_dense_train.shape[1], num_moves, num_species, num_items, num_abilities)
        print("Compiling model...")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=LOSSES,
            loss_weights=LOSS_WEIGHTS,
            metrics={"value": "mae", "policy": "accuracy"}
        )

    is_scratch = not os.path.exists(model_save_path)
    train_outputs["meta_plan"] = compute_counterfactual_targets(model, train_inputs, is_scratch=is_scratch)
    if val_data is not None:
        val_inputs, val_outputs = val_data
        val_outputs["meta_plan"] = compute_counterfactual_targets(model, val_inputs, is_scratch=is_scratch)
        val_data = (val_inputs, val_outputs)

    model.summary()

    callbacks = [PrimaryLossCallback()]
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

    onnx_path = save_path.with_suffix(".onnx")
    export_to_onnx(model, onnx_path)


if __name__ == "__main__":
    train()
