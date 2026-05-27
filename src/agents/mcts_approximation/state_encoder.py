"""
Encodes a PokemonState into a flat numeric vector for the Neural Network.

Feature layout (total 239 elements):

  DENSE [0:163]:
    OWN TEAM        — 6 Pokémon × 13 = 78  (hp_ratio, fainted, 5 statuses, 6 boosts)
    OWN ACTIVE      — 5 base stats + 4 moves × 5 = 25  (power, accuracy, 3 category flags)
    OPP REVEALED    — 6 Pokémon × 7 = 42  (hp_ratio, fainted, 5 statuses — public only)
    FIELD           — 18  (weather ×4, own hazards/screens ×5, opp hazards/screens ×5,
                           own active volatiles ×4)

  EMBEDDING INDICES [163:239] (integers, 0 = unknown/padding):
    own_species[6]          → species ID per own bench slot
    own_items[6]            → item ID per own bench slot
    own_abilities[6]        → ability ID per own bench slot
    own_bench_moves[6][4]   → 4 move IDs per own bench Pokémon  (24 total)
    own_active_moves[4]     → 4 move IDs for the currently active Pokémon
    opp_species[6]          → species ID once revealed; 0 if never sent out
    opp_used_moves[6][4]    → move IDs used so far; 0 for unrevealed moves  (24 total)
    Total: 6+6+6+24+4+6+24 = 76

The encoder is "Blind" — for the opponent it only encodes what is publicly
visible in a real match: species and HP of Pokémon that have been sent out,
moves that have already been used, and shared field conditions (weather/hazards).
"""

import numpy as np

from agents.mcts_approximation.moves_db import get_move_data, get_move_idx
from agents.mcts_approximation.species_db import get_species_idx, get_item_idx, get_ability_idx

STATUSES = ["tox", "brn", "par", "slp", "frz"]
BOOST_STATS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

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


# ─── Own team from state_dict sides (has full data including bench) ────────────

def _encode_own_team(state_dict: dict, player: str):
    """
    Encodes all 6 own Pokémon from the full state_dict.
    Returns (dense_features, species_indices, item_indices, ability_indices, bench_move_indices).

    Per Pokémon (13 dense):
        hp_ratio (1) + fainted (1) + 5 statuses + 6 boosts = 13

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

            boosts = p.get("boosts", {})
            boost_feats = _normalise_boosts(boosts)

            dense.extend([hp_ratio, fainted] + statuses + boost_feats)

            # Species ID from speciesState.id or details string
            species_id = ""
            species_state = p.get("speciesState", {})
            if isinstance(species_state, dict):
                species_id = species_state.get("id", "")
            if not species_id:
                details = p.get("details", "")
                species_id = _species_id_from_details(details) if details else ""

            species_idxs.append(get_species_idx(species_id))
            item_idxs.append(get_item_idx(p.get("item", "")))
            ability_idxs.append(get_ability_idx(p.get("ability", p.get("baseAbility", ""))))

            # Moves from moveSlots (available for all bench Pokémon in state_dict)
            move_slots = p.get("moveSlots", [])
            for j in range(NUM_MOVES):
                if j < len(move_slots):
                    bench_move_idxs.append(get_move_idx(move_slots[j].get("id", "")))
                else:
                    bench_move_idxs.append(0)
        else:
            # Pad missing slots
            dense.extend([0.0] * 13)
            species_idxs.append(0)
            item_idxs.append(0)
            ability_idxs.append(0)
            bench_move_idxs.extend([0] * NUM_MOVES)

    return dense, species_idxs, item_idxs, ability_idxs, bench_move_idxs


# ─── Active Pokémon moves (from request_dict) ─────────────────────────────────

def _encode_active_moves(request: dict):
    """
    Encodes the 4 move slots of the active Pokémon using the move database.
    Returns (dense_features, move_indices).
    
    Per move (5 dense): base_power + accuracy + 3 category one-hot
    """
    dense = []
    move_idxs = []
    
    active_moves = []
    if request and "active" in request and len(request["active"]) > 0:
        active_moves = request["active"][0].get("moves", [])
    
    for i in range(NUM_MOVES):
        if i < len(active_moves):
            move_info = active_moves[i]
            move_id = move_info.get("id") if isinstance(move_info, dict) else move_info
            
            db_move = get_move_data(move_id)
            power = db_move.get("basePower", 0) / 150.0
            accuracy = db_move.get("accuracy", 100) / 100.0
            cat = db_move.get("category", "Status")
            
            dense.extend([
                power,
                accuracy,
                1.0 if cat == "Physical" else 0.0,
                1.0 if cat == "Special" else 0.0,
                1.0 if cat == "Status" else 0.0,
            ])
            move_idxs.append(get_move_idx(move_id))
        else:
            dense.extend([0.0] * 5)
            move_idxs.append(0)
    
    return dense, move_idxs


# ─── Active Pokémon stats (from state_dict own side) ──────────────────────────

def _encode_active_stats(state_dict: dict, player: str):
    """
    Encodes the normalised base stats of the currently active Pokémon.
    Returns 5 floats: atk, def, spa, spd, spe all / 500.
    """
    side_idx = 0 if player == "p1" else 1
    sides = state_dict.get("sides", [])
    own_side = sides[side_idx] if len(sides) > side_idx else {}
    
    for p in own_side.get("pokemon", []):
        if p.get("isActive", False):
            stats = p.get("storedStats", p.get("baseStoredStats", {}))
            return [
                stats.get("atk", 0) / 500.0,
                stats.get("def", 0) / 500.0,
                stats.get("spa", 0) / 500.0,
                stats.get("spd", 0) / 500.0,
                stats.get("spe", 0) / 500.0,
            ]
    return [0.0] * 5


# ─── Field conditions (weather, hazards, screens, volatiles) ─────────────────

# All field conditions listed below are publicly visible to both players.
WEATHERS       = ["raindance", "sunnyday", "sandstorm", "hail"]
SIDE_CONDS     = ["stealthrock", "spikes", "toxicspikes", "reflect", "lightscreen"]
ACTIVE_VOLS    = ["choicelock", "substitute", "twoturnmove", "focuspunch"]

def _encode_field_conditions(state_dict: dict, player: str) -> list:
    """
    Encodes 18 publicly-visible field features:
      Weather       (4): one-hot flags for rain/sun/sand/hail
      Own side      (5): stealthrock, spikes/3, toxicspikes/2, reflect, lightscreen
      Opp side      (5): same 5 (public — you can see their hazards/screens)
      Own volatiles (4): choicelock, substitute, twoturnmove, focuspunch
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
    vol_feats = [0.0] * len(ACTIVE_VOLS)
    for poke in own_side.get("pokemon", []):
        if poke.get("isActive", False):
            vols = poke.get("volatiles", {})
            vol_feats = [1.0 if v in vols else 0.0 for v in ACTIVE_VOLS]
            break

    return weather_feats + own_sc + opp_sc + vol_feats  # 4+5+5+4 = 18


def _encode_opp_team_revealed(state_dict: dict, player: str):
    """
    Encodes the opponent's team using ONLY publicly visible information.

    A Pokémon is 'revealed' when it enters the battle for the first time
    (isActive, newlySwitched, or previouslySwitchedIn > 0) or when it faints.
    Pokémon never sent out are encoded as all-zeros (unknown).

    Per slot (7 dense):
        hp_ratio (1) + fainted (1) + 5 statuses = 7
        Unknown slots: all 0.0

    Per slot (embedding IDs):
        species_idx (1): 0 if not yet revealed
        used_move_idxs (4): move ID if used this battle, 0 if not yet seen

    Returns (dense, opp_species_idxs, opp_move_idxs)
    """
    opp_idx = 1 if player == "p1" else 0
    sides = state_dict.get("sides", [])
    opp_side = sides[opp_idx] if len(sides) > opp_idx else {}
    pokemon_list = opp_side.get("pokemon", [])

    dense = []
    opp_species_idxs = []
    opp_move_idxs = []  # 4 per Pokémon; 0 = not yet revealed

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
                dense.extend([hp_ratio, fainted] + _status_onehot(status))

                # Species is known once the Pokémon was sent out
                species_id = ""
                species_state = p.get("speciesState", {})
                if isinstance(species_state, dict):
                    species_id = species_state.get("id", "")
                if not species_id:
                    details = p.get("details", "")
                    species_id = _species_id_from_details(details) if details else ""
                opp_species_idxs.append(get_species_idx(species_id))

                # Only include moves that have been used (revealed through battle)
                move_slots = p.get("moveSlots", [])
                for j in range(NUM_MOVES):
                    if j < len(move_slots) and move_slots[j].get("used", False):
                        opp_move_idxs.append(get_move_idx(move_slots[j].get("id", "")))
                    else:
                        opp_move_idxs.append(0)  # Not yet revealed
            else:
                # Pokémon never entered the battle — treat as unknown
                dense.extend([0.0] * 7)
                opp_species_idxs.append(0)
                opp_move_idxs.extend([0] * NUM_MOVES)
        else:
            dense.extend([0.0] * 7)
            opp_species_idxs.append(0)
            opp_move_idxs.extend([0] * NUM_MOVES)

    return dense, opp_species_idxs, opp_move_idxs


# ─── Main encode function ─────────────────────────────────────────────────────

def encode_state(state, player: str = "p1") -> np.ndarray:
    """
    Encodes the full PokemonState into a flat numeric vector.

    Layout (total 239 elements):

    DENSE [0:163]:
        [0:78]    — own team (6 × 13): hp, fainted, 5 statuses, 6 boosts
        [78:83]   — own active base stats (5)
        [83:103]  — own active moves (4 × 5): power, accuracy, 3 category flags
        [103:145] — opp revealed team (6 × 7): hp, fainted, 5 statuses (public only)
        [145:163] — field conditions (18): weather, hazards, screens, volatiles

    EMBEDDING INDICES [163:239] (integers):
        [163:169] — own species        (6)
        [169:175] — own items          (6)
        [175:181] — own abilities      (6)
        [181:205] — own bench moves    (6 × 4 = 24)
        [205:209] — own active moves   (4)
        [209:215] — opp species        (6, 0 if not revealed)
        [215:239] — opp used moves     (6 × 4 = 24, 0 if not yet seen)
    """
    request = state.request_dict if player == "p1" else state.p2_request_dict
    state_dict = state.state_dict

    # 1. Own full team (dense + embedding IDs)
    team_dense, species_idxs, item_idxs, ability_idxs, bench_move_idxs = \
        _encode_own_team(state_dict, player)

    # 2. Own active Pokémon stats
    active_stats = _encode_active_stats(state_dict, player)

    # 3. Own active Pokémon moves (dense + move embedding IDs)
    moves_dense, move_idxs = _encode_active_moves(request)

    # 4. Opponent team — only revealed public information
    opp_dense, opp_species_idxs, opp_move_idxs = _encode_opp_team_revealed(state_dict, player)

    # 5. Field conditions (weather, hazards, screens, active volatiles)
    field_feats = _encode_field_conditions(state_dict, player)

    # Assemble: all dense first, then all embedding indices
    dense_features = team_dense + active_stats + moves_dense + opp_dense + field_feats
    embedding_indices = (species_idxs + item_idxs + ability_idxs +
                         bench_move_idxs + move_idxs +
                         opp_species_idxs + opp_move_idxs)

    all_features = dense_features + embedding_indices
    return np.array(all_features, dtype=np.float32)


# ─── Feature shape constants (used by train_nn.py and agent) ─────────────────

# Dense features:
#   6×13 own team (78) + 5 active stats + 4×5 active moves (20)
#   + 6×7 opp revealed team (42) + 18 field conditions
NUM_FIELD_FEATURES   = len(WEATHERS) + len(SIDE_CONDS) * 2 + len(ACTIVE_VOLS)  # = 4+10+4 = 18
NUM_DENSE_FEATURES   = 6 * 13 + 5 + 4 * 5 + 6 * 7 + NUM_FIELD_FEATURES        # = 145+18 = 163

# Embedding index counts (own side)
NUM_SPECIES_INDICES      = 6
NUM_ITEM_INDICES         = 6
NUM_ABILITY_INDICES      = 6
NUM_BENCH_MOVE_INDICES   = 6 * 4   # = 24
NUM_ACTIVE_MOVE_INDICES  = 4

# Embedding index counts (opponent — publicly revealed only)
NUM_OPP_SPECIES_INDICES  = 6
NUM_OPP_MOVE_INDICES     = 6 * 4   # = 24  (0 for each unrevealed move)

NUM_EMBEDDING_INDICES = (
    NUM_SPECIES_INDICES + NUM_ITEM_INDICES + NUM_ABILITY_INDICES +
    NUM_BENCH_MOVE_INDICES + NUM_ACTIVE_MOVE_INDICES +
    NUM_OPP_SPECIES_INDICES + NUM_OPP_MOVE_INDICES
)  # = 6+6+6+24+4+6+24 = 76

TOTAL_FEATURES = NUM_DENSE_FEATURES + NUM_EMBEDDING_INDICES  # = 163+76 = 239

# Slice offsets within the embedding block (relative to NUM_DENSE_FEATURES = 163)
OFF_SPECIES      = 0                                                # [163:169]
OFF_ITEMS        = OFF_SPECIES      + NUM_SPECIES_INDICES           # [169:175]
OFF_ABILITIES    = OFF_ITEMS        + NUM_ITEM_INDICES              # [175:181]
OFF_BENCH_MOVES  = OFF_ABILITIES    + NUM_ABILITY_INDICES           # [181:205]
OFF_ACTIVE_MOVES = OFF_BENCH_MOVES  + NUM_BENCH_MOVE_INDICES        # [205:209]
OFF_OPP_SPECIES  = OFF_ACTIVE_MOVES + NUM_ACTIVE_MOVE_INDICES       # [209:215]
OFF_OPP_MOVES    = OFF_OPP_SPECIES  + NUM_OPP_SPECIES_INDICES       # [215:239]

# Define fixed action space for Policy Network mapping
ACTION_SPACE = [
    "move 1", "move 2", "move 3", "move 4",
    "switch 1", "switch 2", "switch 3", "switch 4", "switch 5", "switch 6",
    "pass"
]

