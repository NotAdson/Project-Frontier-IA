import json
from pathlib import Path

import numpy as np

from battle_agents.mcts_approximation.state_encoder import (
    ACTION_SPACE, TOTAL_FEATURES, FIELD_START, MAIN_EMB_ABILITY_DIM,
    MAIN_EMB_ITEMS_DIM, MAIN_EMB_MOVES_DIM, MAIN_EMB_SPECIES_DIM,
    META_EMB_ABILITY_DIM, META_EMB_ITEMS_DIM, META_EMB_MOVES_DIM,
    META_EMB_SPECIES_DIM, NUM_ABILITY_INDICES, NUM_ACTIVE, NUM_BENCH,
    NUM_BOOSTS, NUM_DENSE_FEATURES, NUM_EMBEDDING_INDICES, NUM_FIELD_FEATURES,
    NUM_ITEM_INDICES, NUM_MOVE_INDICES, NUM_MOVES, NUM_SPECIES_INDICES,
    NUM_STATUS, NUM_STATS, NUM_TYPES, OFF_ABILITIES, OFF_FAINTED, OFF_HP,
    OFF_IS_ACTIVE, OFF_ITEMS, OFF_LEVEL, OFF_MOVES, OFF_MOVES_DENSE, OFF_SPECIES,
    OFF_STATS, OFF_STATUSES, OFF_TYPES, OPP_BOOSTS_START, OPP_TEAM_START,
    OWN_BOOSTS_START, OWN_TEAM_DENSE, PER_MON_DENSE, PP_START
)


def split_features(X: np.ndarray):
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
