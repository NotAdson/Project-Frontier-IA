import numpy as np
import keras

from battle_agents.mcts_approximation.state_encoder import (
    FIELD_START, NUM_FIELD_FEATURES, NUM_BOOSTS, NUM_STATUS, NUM_STATS,
    NUM_TYPES, NUM_MOVES, NUM_BENCH, PER_MON_DENSE, NUM_DENSE_FEATURES,
    OFF_IS_ACTIVE, OFF_HP, OFF_STATUSES, OFF_STATS, OFF_TYPES, OFF_SPECIES,
    OFF_MOVES, OPP_TEAM_START, OWN_BOOSTS_START, OPP_BOOSTS_START, PP_START
)


def extract_aux_targets_batch(X_next, num_species, num_moves):
    batch_size = X_next.shape[0]

    # 1. Field conditions
    field = X_next[:, FIELD_START:FIELD_START + NUM_FIELD_FEATURES]

    # 2. Boosts
    own_boosts = X_next[:, OWN_BOOSTS_START:OWN_BOOSTS_START + NUM_BOOSTS]
    opp_boosts = X_next[:, OPP_BOOSTS_START:OPP_BOOSTS_START + NUM_BOOSTS]

    # Initialize active targets
    own_hp = np.zeros((batch_size, 1), dtype=np.float32)
    opp_hp = np.zeros((batch_size, 1), dtype=np.float32)

    # Statuses
    own_statuses = np.zeros((batch_size, NUM_STATUS), dtype=np.float32)
    opp_statuses = np.zeros((batch_size, NUM_STATUS), dtype=np.float32)

    # Stats
    own_stats = np.zeros((batch_size, NUM_STATS), dtype=np.float32)
    opp_stats = np.zeros((batch_size, NUM_STATS), dtype=np.float32)

    # Types
    own_types = np.zeros((batch_size, NUM_TYPES), dtype=np.float32)
    opp_types = np.zeros((batch_size, NUM_TYPES), dtype=np.float32)

    # Species
    own_species = np.zeros((batch_size,), dtype=np.int32)
    opp_species = np.zeros((batch_size,), dtype=np.int32)

    # Moves
    own_moves_multihot = np.zeros((batch_size, num_moves), dtype=np.float32)
    opp_moves_multihot = np.zeros((batch_size, num_moves), dtype=np.float32)

    for b in range(batch_size):
        # Find active own Pokemon index
        own_act = -1
        for i in range(NUM_BENCH):
            if X_next[b, i * PER_MON_DENSE + OFF_IS_ACTIVE] == 1.0:
                own_act = i
                break

        # Find active opponent Pokemon index
        opp_act = -1
        for j in range(NUM_BENCH):
            if X_next[b, OPP_TEAM_START + j * PER_MON_DENSE + OFF_IS_ACTIVE] == 1.0:
                opp_act = j
                break

        if own_act != -1:
            own_hp[b, 0] = X_next[b, own_act * PER_MON_DENSE + OFF_HP]
            own_statuses[b, :] = X_next[b, own_act * PER_MON_DENSE + OFF_STATUSES : own_act * PER_MON_DENSE + OFF_STATUSES + NUM_STATUS]
            own_stats[b, :] = X_next[b, own_act * PER_MON_DENSE + OFF_STATS : own_act * PER_MON_DENSE + OFF_STATS + NUM_STATS]
            own_types[b, :] = X_next[b, own_act * PER_MON_DENSE + OFF_TYPES : own_act * PER_MON_DENSE + OFF_TYPES + NUM_TYPES]
            own_species[b] = np.clip(int(round(X_next[b, NUM_DENSE_FEATURES + own_act])), 0, num_species - 1)

            # Extract own active moves
            for k in range(NUM_MOVES):
                idx = int(round(X_next[b, NUM_DENSE_FEATURES + OFF_MOVES + own_act * NUM_MOVES + k]))
                if 0 < idx < num_moves:
                    own_moves_multihot[b, idx] = 1.0

        if opp_act != -1:
            opp_hp[b, 0] = X_next[b, OPP_TEAM_START + opp_act * PER_MON_DENSE + OFF_HP]
            opp_statuses[b, :] = X_next[b, OPP_TEAM_START + opp_act * PER_MON_DENSE + OFF_STATUSES : OPP_TEAM_START + opp_act * PER_MON_DENSE + OFF_STATUSES + NUM_STATUS]
            opp_stats[b, :] = X_next[b, OPP_TEAM_START + opp_act * PER_MON_DENSE + OFF_STATS : OPP_TEAM_START + opp_act * PER_MON_DENSE + OFF_STATS + NUM_STATS]
            opp_types[b, :] = X_next[b, OPP_TEAM_START + opp_act * PER_MON_DENSE + OFF_TYPES : OPP_TEAM_START + opp_act * PER_MON_DENSE + OFF_TYPES + NUM_TYPES]
            opp_species[b] = np.clip(int(round(X_next[b, NUM_DENSE_FEATURES + NUM_BENCH + opp_act])), 0, num_species - 1)

            # Extract opponent active moves
            for k in range(NUM_MOVES):
                idx = int(round(X_next[b, NUM_DENSE_FEATURES + OFF_MOVES + NUM_BENCH * NUM_MOVES + opp_act * NUM_MOVES + k]))
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
