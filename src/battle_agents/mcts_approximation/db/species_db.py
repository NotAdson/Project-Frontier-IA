"""
Lookup databases for Pokémon species, items, and abilities.
Provides stable integer index mappings suitable for Embedding layers.
Mirrors the pattern of moves_db.py.
"""
import json
from pathlib import Path

_BASEDIR = Path(__file__).resolve().parent

# ─── Species ──────────────────────────────────────────────────────────────────

_species_db = None
_species_to_idx = None

def _load_species():
    global _species_db, _species_to_idx
    if _species_db is None:
        db_path = _BASEDIR / 'species.json'
        _species_db = json.loads(db_path.read_text()) if db_path.exists() else {}
        sorted_ids = sorted(_species_db.keys())
        # 0 is reserved for unknown/padding
        _species_to_idx = {sid: idx + 1 for idx, sid in enumerate(sorted_ids)}

def get_species_idx(species_id: str) -> int:
    """Returns a stable integer for a species ID (0 = unknown/padding)."""
    _load_species()
    if species_id is None:
        return 0
    # Normalise to lowercase with no spaces
    sid = str(species_id).lower().replace(' ', '').replace('-', '')
    return _species_to_idx.get(sid, _species_to_idx.get(species_id, 0))

def get_species_data(species_id: str) -> dict:
    """Returns full species info including types and base stats."""
    _load_species()
    if species_id is None:
        return {}
    sid = str(species_id).lower().replace(' ', '').replace('-', '')
    return _species_db.get(sid, _species_db.get(species_id, {}))

def get_num_species() -> int:
    _load_species()
    return len(_species_to_idx) + 1 if _species_to_idx else 1

# ─── Items ────────────────────────────────────────────────────────────────────

_items_db = None
_item_to_idx = None

def _load_items():
    global _items_db, _item_to_idx
    if _items_db is None:
        db_path = _BASEDIR / 'items.json'
        _items_db = json.loads(db_path.read_text()) if db_path.exists() else {}
        sorted_ids = sorted(_items_db.keys())
        _item_to_idx = {iid: idx + 1 for idx, iid in enumerate(sorted_ids)}

def get_item_idx(item_id: str) -> int:
    """Returns a stable integer for an item ID (0 = no item / unknown)."""
    _load_items()
    if not item_id:
        return 0
    return _item_to_idx.get(str(item_id).lower(), 0)

def get_num_items() -> int:
    _load_items()
    return len(_item_to_idx) + 1 if _item_to_idx else 1

# ─── Abilities ────────────────────────────────────────────────────────────────

_abilities_db = None
_ability_to_idx = None

def _load_abilities():
    global _abilities_db, _ability_to_idx
    if _abilities_db is None:
        db_path = _BASEDIR / 'abilities.json'
        _abilities_db = json.loads(db_path.read_text()) if db_path.exists() else {}
        sorted_ids = sorted(_abilities_db.keys())
        _ability_to_idx = {aid: idx + 1 for idx, aid in enumerate(sorted_ids)}

def get_ability_idx(ability_id: str) -> int:
    """Returns a stable integer for an ability ID (0 = unknown/padding)."""
    _load_abilities()
    if not ability_id:
        return 0
    return _ability_to_idx.get(str(ability_id).lower(), 0)

def get_num_abilities() -> int:
    _load_abilities()
    return len(_ability_to_idx) + 1 if _ability_to_idx else 1
