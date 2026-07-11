"""
Encodes a PokemonState into a flat numeric vector for the Neural Network.

Feature layout (total 738 elements):

  DENSE [0:654]:
    OWN TEAM        — 6 Pokémon × 52 = 312  (hp_ratio, fainted, 5 statuses, is_active, level, 5 actual level-scaled stats, 18 types, 4 moves × 5 dense details)
    OWN ACTIVE      — 6 active boosts (6)
    OPP REVEALED    — 6 Pokémon × 52 = 312  (hp_ratio, fainted, 5 statuses, is_active, level, 5 base stats, 18 types, 4 moves × 5 dense details — public only)
    OPP ACTIVE      — 6 active boosts (6)
    FIELD           — 18  (weather ×4, own hazards/screens ×5, opp hazards/screens ×5,
                           own active volatiles ×4)

  EMBEDDING INDICES [654:738] (integers, 0 = unknown/padding):
    species_indices[12]     → own_species[6] + opp_species[6]
    move_indices[48]        → own_bench_moves[24] + opp_used_moves[24]
    item_indices[12]        → own_items[6] + opp_items[6]
    ability_indices[12]     → own_abilities[6] + opp_abilities[6]
    Total: 12 + 48 + 12 + 12 = 84

The encoder is "Blind" — for the opponent it only encodes what is publicly
visible in a real match. Once an opponent's Pokémon is revealed (sent out or fainted),
we symmetrically encode its species, level, normalized base stats, types, current HP ratio,
active status, and any active conditions. Moves are only encoded once they are actually
used (revealed) in battle, maintaining strict compliance with fog-of-war rules.
"""


import numpy as np

from battle_agents.mcts_approximation.db.python.database import db

STATUSES = ["tox", "brn", "par", "slp", "frz"]
BOOST_STATS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

TYPES = ["Normal", "Fire", "Water", "Grass", "Electric", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Steel", "Dark", "Fairy"]

def _encode_types(types_list: list) -> list:
    """Returns an 18-element binary list indicating the active types."""
    res = [0.0] * len(TYPES)
    if not types_list:
        return res
    for t in types_list:
        if t in TYPES:
            res[TYPES.index(t)] = 1.0
    return res

NUM_BENCH = 6  # Total Pokémon per team
NUM_MOVES = 4  # Move slots per Pokémon

# ─── Helper: parse "xxx/yyy" or "xxx/yyy status" condition strings ─────────────

def _parse_condition(condition: str):
    """Returns (hp, maxhp) from a Showdown condition string like '260/260' or '0 fnt'."""
    try:
        hp_part = condition.split(" ")[0]
        hp, maxhp = map(int, hp_part.split("/"))
        return hp, maxhp
    except Exception:
        return 0, 1

def _hp_ratio(condition: str) -> float:
    hp, maxhp = _parse_condition(condition)
    return hp / maxhp if maxhp > 0 else 0.0

def _is_fainted(condition: str) -> float:
    return 1.0 if "fnt" in condition or _parse_condition(condition)[0] == 0 else 0.0

def _status_onehot(status: str) -> list:
    """5-element one-hot for the 5 status conditions."""
    return [1.0 if s in status else 0.0 for s in STATUSES]

def _normalise_boosts(boosts: dict) -> list:
    """6 boost values (atk/def/spa/spd/spe/accuracy) normalised from [-6,6] to [-1,1]."""
    return [boosts.get(stat, 0) / 6.0 for stat in ["atk", "def", "spa", "spd", "spe", "accuracy"]]

def _species_id_from_details(details: str) -> str:
    """Extracts a normalised species ID from a Showdown details string like 'Blissey, L77, F'."""
    name = details.split(",")[0].strip()
    return name.lower().replace(" ", "").replace("-", "")

def _species_id_from_ident(ident: str) -> str:
    """Extracts species from 'p1: Steelix' → 'steelix'."""
    parts = ident.split(": ")
    if len(parts) > 1:
        return parts[1].strip().lower().replace(" ", "").replace("-", "")
    return ""


# ─── Own team encoding ─────────────────────────────────────────────────────────

def _encode_own_team(state_dict: dict, player: str):
    """
    Encodes all 6 own Pokémon from the full state_dict.
    Returns (dense_features, species_indices, item_indices, ability_indices, bench_move_indices).

    Per Pokémon (52 dense):
        hp_ratio (1) + fainted (1) + 5 statuses + isActive (1) + level (1) + 5 stats + 18 types
        + 4 moves × 5 dense details (20) = 52

    Per Pokémon (4 embedding IDs):
        move IDs from moveSlots (padded with 0 if fewer than 4 moves)
    """
    side_idx = 0 if player == "p1" else 1
    sides = state_dict.get("sides", [])
    own_side = sides[side_idx] if len(sides) > side_idx else {}
    pokemon_list = own_side.get("pokemon", [])

    dense = []
    species_idxs = []
    item_idxs = []
    ability_idxs = []
    bench_move_idxs = []  # 4 move IDs per Pokémon, flattened

    for i in range(NUM_BENCH):
        if i < len(pokemon_list):
            p = pokemon_list[i]

            hp = p.get("hp", 0)
            maxhp = p.get("maxhp", 1)
            hp_ratio = hp / maxhp if maxhp > 0 else 0.0
            fainted = 1.0 if hp == 0 or p.get("fainted", False) else 0.0

            status = p.get("status", "")
            statuses = _status_onehot(status)

            is_active = 1.0 if (p.get("isActive", False) or p.get("active", False)) else 0.0
            level = p.get("level", 100) / 100.0

            p_stats = p.get("stats", p.get("storedStats", p.get("baseStoredStats", {})))
            atk = p_stats.get("atk", 0) / 500.0
            def_ = p_stats.get("def", 0) / 500.0
            spa = p_stats.get("spa", 0) / 500.0
            spd = p_stats.get("spd", 0) / 500.0
            spe = p_stats.get("spe", 0) / 500.0

            # Species ID from speciesState.id or details string
            species_id = ""
            species_state = p.get("speciesState", {})
            if isinstance(species_state, dict):
                species_id = species_state.get("id", "")
            if not species_id:
                details = p.get("details", "")
                species_id = _species_id_from_details(details) if details else ""

            species_data = db.get_species_data(species_id)
            species_types = species_data.get("types", [])
            types_encoded = _encode_types(species_types)

            # 4 moves dense features
            move_slots = p.get("moveSlots", [])
            moves_dense = []
            for j in range(NUM_MOVES):
                if j < len(move_slots):
                    m = move_slots[j]
                    move_id = m.get("id", "")
                    db_move = db.get_move_data(move_id)
                    power = db_move.get("basePower", 0) / 150.0
                    accuracy = db_move.get("accuracy", 100) / 100.0
                    cat = db_move.get("category", "Status")
                    moves_dense.extend([
                        power,
                        accuracy,
                        1.0 if cat == "Physical" else 0.0,
                        1.0 if cat == "Special" else 0.0,
                        1.0 if cat == "Status" else 0.0,
                    ])
                else:
                    moves_dense.extend([0.0] * 5)

            dense.extend([hp_ratio, fainted] + statuses + [is_active, level, atk, def_, spa, spd, spe] + types_encoded + moves_dense)

            species_idxs.append(db.get_species_idx(species_id))
            item_idxs.append(db.get_item_idx(p.get("item", "")))
            ability_idxs.append(db.get_ability_idx(p.get("ability", p.get("baseAbility", ""))))

            # Moves from moveSlots (available for all bench Pokémon in state_dict)
            for j in range(NUM_MOVES):
                if j < len(move_slots):
                    bench_move_idxs.append(db.get_move_idx(move_slots[j].get("id", "")))
                else:
                    bench_move_idxs.append(0)
        else:
            # Pad missing slots
            dense.extend([0.0] * 52)
            species_idxs.append(0)
            item_idxs.append(0)
            ability_idxs.append(0)
            bench_move_idxs.extend([0] * NUM_MOVES)
    return dense, species_idxs, item_idxs, ability_idxs, bench_move_idxs



# ─── Active Pokémon boosts ───────────────────────────────────────────────────


def _encode_active_boosts(side: dict) -> list:
    """
    Encodes the boosts of the currently active Pokémon on the given side.
    Returns 6 floats normalised from [-6, 6] to [-1, 1].
    """
    for p in side.get("pokemon", []):
        if p.get("isActive", False):
            boosts = p.get("boosts", {})
            return _normalise_boosts(boosts)
    return [0.0] * 6


# ─── Field conditions (weather, hazards, screens, volatiles) ─────────────────

# All field conditions listed below are publicly visible to both players.
WEATHERS       = ["raindance", "sunnyday", "sandstorm", "hail"]
SIDE_CONDS     = ["stealthrock", "spikes", "toxicspikes", "reflect", "lightscreen"]
ACTIVE_VOLS    = [
    "choicelock", "substitute", "twoturnmove", "focuspunch", "taunt",
    "encore", "confusion", "leechseed", "yawn", "perishsong",
    "attract", "curse", "nightmare", "destinybond", "charge",
    "defensecurl", "minimize", "torment", "ingrain", "imprison",
    "disable", "bide", "flinch"
]

def _encode_field_conditions(state_dict: dict, player: str) -> list:
    """
    Encodes 60 publicly-visible field features:
      Weather       (4): one-hot flags for rain/sun/sand/hail
      Own side      (5): stealthrock, spikes/3, toxicspikes/2, reflect, lightscreen
      Opp side      (5): same 5 (public — you can see their hazards/screens)
      Own volatiles (23): active volatiles like taunt, encore, confusion, substitute, etc.
      Opp volatiles (23): opponent active volatiles like taunt, encore, confusion, substitute, etc.
    All values are floats in [0, 1].
    """
    own_idx = 0 if player == "p1" else 1
    opp_idx = 1 - own_idx
    sides   = state_dict.get("sides", [{}, {}])
    field   = state_dict.get("field", {})

    # Weather one-hot
    weather = field.get("weather", "").lower()
    weather_feats = [1.0 if w == weather else 0.0 for w in WEATHERS]

    def _side_conds(side: dict) -> list:
        sc = side.get("sideConditions", {})
        feats = []
        for cond in SIDE_CONDS:
            if cond == "spikes":
                # Up to 3 layers — normalise to [0, 1]
                layers = sc.get(cond, {}).get("layers", 1) if cond in sc else 0
                feats.append(layers / 3.0)
            elif cond == "toxicspikes":
                layers = sc.get(cond, {}).get("layers", 1) if cond in sc else 0
                feats.append(layers / 2.0)
            else:
                feats.append(1.0 if cond in sc else 0.0)
        return feats  # 5 values

    own_side = sides[own_idx] if len(sides) > own_idx else {}
    opp_side = sides[opp_idx] if len(sides) > opp_idx else {}
    own_sc = _side_conds(own_side)
    opp_sc = _side_conds(opp_side)

    # Own active Pokémon volatile statuses
    own_vol_feats = [0.0] * len(ACTIVE_VOLS)
    for poke in own_side.get("pokemon", []):
        if poke.get("isActive", False):
            vols = poke.get("volatiles", {})
            own_vol_feats = [1.0 if v in vols else 0.0 for v in ACTIVE_VOLS]
            break

    # Opponent active Pokémon volatile statuses
    opp_vol_feats = [0.0] * len(ACTIVE_VOLS)
    for poke in opp_side.get("pokemon", []):
        if poke.get("isActive", False):
            vols = poke.get("volatiles", {})
            opp_vol_feats = [1.0 if v in vols else 0.0 for v in ACTIVE_VOLS]
            break

    return weather_feats + own_sc + opp_sc + own_vol_feats + opp_vol_feats  # 4+5+5+23+23 = 60


def _encode_opp_team_revealed(state_dict: dict, player: str):
    """
    Encodes the opponent's team using ONLY publicly visible information.

    A Pokémon is 'revealed' when it enters the battle for the first time
    (isActive, newlySwitched, or previouslySwitchedIn > 0) or when it faints.
    Pokémon never sent out are encoded as all-zeros (unknown).

    Per slot (52 dense):
        hp_ratio (1) + fainted (1) + 5 statuses + isActive (1) + level (1) + 5 stats + 18 types
        + 4 moves × 5 dense details (20) = 52
        Unknown slots: all 0.0

    Per slot (embedding IDs):
        species_idx (1): 0 if not yet revealed
        used_move_idxs (4): move ID if used this battle, 0 if not yet seen
        item_idx (1): 0 if not yet revealed
        ability_idx (1): 0 if not yet revealed

    Returns (dense, opp_species_idxs, opp_move_idxs, opp_item_idxs, opp_ability_idxs)
    """
    opp_idx = 1 if player == "p1" else 0
    sides = state_dict.get("sides", [])
    opp_side = sides[opp_idx] if len(sides) > opp_idx else {}
    pokemon_list = opp_side.get("pokemon", [])

    dense = []
    opp_species_idxs = []
    opp_move_idxs = []  # 4 per Pokémon; 0 = not yet revealed
    opp_item_idxs = []
    opp_ability_idxs = []

    for i in range(NUM_BENCH):
        if i < len(pokemon_list):
            p = pokemon_list[i]

            # A Pokémon is publicly known once it has entered the battle.
            # Note: at battle start the engine sets newlySwitched=True for ALL bench
            # slots as an initialisation artifact, so we only count it when the
            # Pokémon is also isActive (i.e. actually on the field right now).
            is_revealed = (
                p.get("isActive", False)
                or (p.get("newlySwitched", False) and p.get("isActive", False))
                or p.get("previouslySwitchedIn", 0) > 0
                or p.get("fainted", False)
            )

            if is_revealed:
                hp = p.get("hp", 0)
                maxhp = p.get("maxhp", 1)
                hp_ratio = hp / maxhp if maxhp > 0 else 0.0
                fainted = 1.0 if p.get("fainted", False) or hp == 0 else 0.0
                status = p.get("status", "")
                is_active = 1.0 if (p.get("isActive", False) or p.get("active", False)) else 0.0
                level = p.get("level", 100) / 100.0

                # Species is known once the Pokémon was sent out
                species_id = ""
                species_state = p.get("speciesState", {})
                if isinstance(species_state, dict):
                    species_id = species_state.get("id", "")
                if not species_id:
                    details = p.get("details", "")
                    species_id = _species_id_from_details(details) if details else ""

                species_data = db.get_species_data(species_id)
                species_types = species_data.get("types", [])
                types_encoded = _encode_types(species_types)

                # Use opponent's species base stats normalized (unscaled by level)
                base_stats = species_data.get("baseStats", {})
                atk = base_stats.get("atk", 0) / 500.0
                def_ = base_stats.get("def", 0) / 500.0
                spa = base_stats.get("spa", 0) / 500.0
                spd = base_stats.get("spd", 0) / 500.0
                spe = base_stats.get("spe", 0) / 500.0

                # 4 moves dense features (only for moves that have been used/revealed in battle)
                move_slots = p.get("moveSlots", [])
                moves_dense = []
                for j in range(NUM_MOVES):
                    if j < len(move_slots) and move_slots[j].get("used", False):
                        move_id = move_slots[j].get("id", "")
                        db_move = db.get_move_data(move_id)
                        power = db_move.get("basePower", 0) / 150.0
                        accuracy = db_move.get("accuracy", 100) / 100.0
                        cat = db_move.get("category", "Status")
                        moves_dense.extend([
                            power,
                            accuracy,
                            1.0 if cat == "Physical" else 0.0,
                            1.0 if cat == "Special" else 0.0,
                            1.0 if cat == "Status" else 0.0,
                        ])
                    else:
                        moves_dense.extend([0.0] * 5)

                dense.extend([hp_ratio, fainted] + _status_onehot(status) + [is_active, level, atk, def_, spa, spd, spe] + types_encoded + moves_dense)
                opp_species_idxs.append(db.get_species_idx(species_id))
                opp_item_idxs.append(db.get_item_idx(p.get("item", "")))
                opp_ability_idxs.append(db.get_ability_idx(p.get("ability", p.get("baseAbility", ""))))

                # Only include moves that have been used (revealed through battle)
                for j in range(NUM_MOVES):
                    if j < len(move_slots) and move_slots[j].get("used", False):
                        opp_move_idxs.append(db.get_move_idx(move_slots[j].get("id", "")))
                    else:
                        opp_move_idxs.append(0)  # Not yet revealed
            else:
                # Pokémon never entered the battle — treat as unknown
                dense.extend([0.0] * 52)
                opp_species_idxs.append(0)
                opp_move_idxs.extend([0] * NUM_MOVES)
                opp_item_idxs.append(0)
                opp_ability_idxs.append(0)
        else:
            dense.extend([0.0] * 52)
            opp_species_idxs.append(0)
            opp_move_idxs.extend([0] * NUM_MOVES)
            opp_item_idxs.append(0)
            opp_ability_idxs.append(0)

    return dense, opp_species_idxs, opp_move_idxs, opp_item_idxs, opp_ability_idxs


# ─── Main encode function ─────────────────────────────────────────────────────

def encode_state(state, player: str = "p1") -> np.ndarray:
    """
    Encodes the full PokemonState into a flat numeric vector.

    Layout (total 738 elements):

    DENSE [0:654]:
        [0:312]   — own team (6 × 52): hp, fainted, 5 statuses, active_flag, level, 5 actual stats, 18 types, 4 moves × 5 dense details
        [312:318] — own active boosts (6)
        [318:630] — opp revealed team (6 × 52): hp, fainted, 5 statuses, active_flag, level, 5 base stats, 18 types, 4 moves × 5 dense details (public only)
        [630:636] — opp active boosts (6)
        [636:654] — field conditions (18): weather, hazards, screens, volatiles

    EMBEDDING INDICES [654:738] (integers):
        [654:666] — species indices (12 total: 6 own species, 6 opp species)
        [666:714] — move indices    (48 total: 24 own bench moves, 24 opp used moves)
        [714:726] — item indices    (12 total: 6 own items, 6 opp items)
        [720:738] — ability indices (12 total: 6 own abilities, 6 opp abilities)
    """
    state_dict = state.state_dict

    # 1. Own full team (dense + embedding IDs)
    team_dense, species_idxs, item_idxs, ability_idxs, bench_move_idxs = \
        _encode_own_team(state_dict, player)

    own_idx = 0 if player == "p1" else 1
    opp_idx = 1 - own_idx
    sides = state_dict.get("sides", [{}, {}])
    own_side = sides[own_idx] if len(sides) > own_idx else {}
    opp_side = sides[opp_idx] if len(sides) > opp_idx else {}

    # 2. Own active Pokémon boosts
    active_boosts = _encode_active_boosts(own_side)

    # 3. Opponent team — only revealed public information
    opp_dense, opp_species_idxs, opp_move_idxs, opp_item_idxs, opp_ability_idxs = \
        _encode_opp_team_revealed(state_dict, player)

    # 4. Opponent active Pokémon boosts
    opp_boosts = _encode_active_boosts(opp_side)

    # 5. Field conditions (weather, hazards, screens, active volatiles)
    field_feats = _encode_field_conditions(state_dict, player)

    # Assemble original features (654 elements)
    dense_orig = (team_dense + active_boosts +
                  opp_dense + opp_boosts + field_feats)
    
    # 6. Extract PP features for both teams to append at the end (keeping old offsets intact)
    # Own team PP features (24): 6 Pokémon * 4 moves
    own_pps = []
    own_side_idx = 0 if player == "p1" else 1
    own_side_data = sides[own_side_idx] if len(sides) > own_side_idx else {}
    own_pokemon = own_side_data.get("pokemon", [])
    for p in own_pokemon:
        move_slots = p.get("moveSlots", [])
        for j in range(NUM_MOVES):
            if j < len(move_slots):
                m = move_slots[j]
                pp = m.get("pp", 0)
                maxpp = m.get("maxpp", 1)
                own_pps.append(pp / maxpp if maxpp > 0 else 0.0)
            else:
                own_pps.append(0.0)
    if len(own_pps) < 24:
        own_pps.extend([0.0] * (24 - len(own_pps)))
        
    # Opponent team PP features (24): 6 Pokémon * 4 moves
    opp_pps = []
    opp_side_idx = 1 if player == "p1" else 0
    opp_side_data = sides[opp_side_idx] if len(sides) > opp_side_idx else {}
    opp_pokemon = opp_side_data.get("pokemon", [])
    for p in opp_pokemon:
        # We only know opponent's move PP if it has been used/revealed in the battle
        move_slots = p.get("moveSlots", [])
        for j in range(NUM_MOVES):
            if j < len(move_slots) and move_slots[j].get("used", False):
                m = move_slots[j]
                pp = m.get("pp", 0)
                maxpp = m.get("maxpp", 1)
                opp_pps.append(pp / maxpp if maxpp > 0 else 0.0)
            else:
                opp_pps.append(0.0)
    if len(opp_pps) < 24:
        opp_pps.extend([0.0] * (24 - len(opp_pps)))

    # Concatenate the new 48 PP features to the dense list
    dense_features = dense_orig + own_pps + opp_pps
    
    # Category Groupings
    species_category = species_idxs + opp_species_idxs  # 12 indices
    moves_category = bench_move_idxs + opp_move_idxs     # 48 indices
    items_category = item_idxs + opp_item_idxs          # 12 indices
    abilities_category = ability_idxs + opp_ability_idxs  # 12 indices

    embedding_indices = species_category + moves_category + items_category + abilities_category

    all_features = dense_features + embedding_indices
    return np.array(all_features, dtype=np.float32)


# ─── Feature shape constants (used by train_nn.py and agent) ─────────────────

# Dense features:
#   6×52 own team (312) + 6 active boosts (6)
#   + 6×52 opp revealed team (312) + 6 opp active boosts (6) + 60 field conditions (696 new)
#   + 48 PP features (24 own PP, 24 opponent PP) = 744 dense
NUM_FIELD_FEATURES   = len(WEATHERS) + len(SIDE_CONDS) * 2 + len(ACTIVE_VOLS) * 2  # = 4+10+46 = 60
NUM_DENSE_FEATURES   = 6 * 52 + 6 + 6 * 52 + 6 + NUM_FIELD_FEATURES + 48   # = 696 + 48 = 744

# Grouped category counts
NUM_SPECIES_INDICES      = 12  # 6 own + 6 opp
NUM_MOVE_INDICES         = 48  # 24 own bench moves + 24 opp used moves
NUM_ITEM_INDICES         = 12  # 6 own items + 6 opp items
NUM_ABILITY_INDICES      = 12  # 6 own abilities + 6 opp abilities

NUM_EMBEDDING_INDICES = (
    NUM_SPECIES_INDICES + NUM_MOVE_INDICES + NUM_ITEM_INDICES + NUM_ABILITY_INDICES
)  # = 12 + 48 + 12 + 12 = 84

TOTAL_FEATURES = NUM_DENSE_FEATURES + NUM_EMBEDDING_INDICES  # = 744 + 84 = 828

# Slice offsets within the embedding block (relative to NUM_DENSE_FEATURES = 744)
OFF_SPECIES      = 0                                                # [744:756]
OFF_MOVES        = OFF_SPECIES      + NUM_SPECIES_INDICES           # [756:804]
OFF_ITEMS        = OFF_MOVES        + NUM_MOVE_INDICES              # [804:816]
OFF_ABILITIES    = OFF_ITEMS        + NUM_ITEM_INDICES              # [816:828]

# Define fixed action space for Policy Network mapping
ACTION_SPACE = [
    "move 1", "move 2", "move 3", "move 4",
    "switch 1", "switch 2", "switch 3", "switch 4", "switch 5", "switch 6",
    "pass"
]

