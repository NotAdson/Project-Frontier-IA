"""
Unified database for Pokémon species, items, abilities, and moves.
Provides a class-based interface for loading JSON files and retrieving indices.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional

_BASEDIR = Path(__file__).resolve().parents[1] / "data"


class PokémonDatabase:
    """Centralized database for Pokémon data."""

    def __init__(self) -> None:
        self._species_db: Optional[Dict[str, Any]] = None
        self._species_to_idx: Optional[Dict[str, int]] = None
        self._items_db: Optional[Dict[str, Any]] = None
        self._item_to_idx: Optional[Dict[str, int]] = None
        self._abilities_db: Optional[Dict[str, Any]] = None
        self._ability_to_idx: Optional[Dict[str, int]] = None
        self._moves_db: Optional[Dict[str, Any]] = None
        self._move_to_idx: Optional[Dict[str, int]] = None

        self._load_all()

    def _load_all(self) -> None:
        self._load_species()
        self._load_items()
        self._load_abilities()
        self._load_moves()

    def _load_species(self) -> None:
        if self._species_db is not None:
            return
        db_path = _BASEDIR / 'species.json'
        if db_path.exists():
            self._species_db = json.loads(db_path.read_text())
        else:
            self._species_db = {}
        sorted_ids = sorted(self._species_db.keys())
        # 0 reserved for unknown/padding
        self._species_to_idx = {sid: idx + 1 for idx, sid in enumerate(sorted_ids)}

    def _load_items(self) -> None:
        if self._items_db is not None:
            return
        db_path = _BASEDIR / 'items.json'
        if db_path.exists():
            self._items_db = json.loads(db_path.read_text())
        else:
            self._items_db = {}
        sorted_ids = sorted(self._items_db.keys())
        self._item_to_idx = {iid: idx + 1 for idx, iid in enumerate(sorted_ids)}

    def _load_abilities(self) -> None:
        if self._abilities_db is not None:
            return
        db_path = _BASEDIR / 'abilities.json'
        if db_path.exists():
            self._abilities_db = json.loads(db_path.read_text())
        else:
            self._abilities_db = {}
        sorted_ids = sorted(self._abilities_db.keys())
        self._ability_to_idx = {aid: idx + 1 for idx, aid in enumerate(sorted_ids)}

    def _load_moves(self) -> None:
        if self._moves_db is not None:
            return
        db_path = _BASEDIR / 'moves.json'
        if db_path.exists():
            self._moves_db = json.loads(db_path.read_text())
        else:
            self._moves_db = {}
        sorted_ids = sorted(self._moves_db.keys())
        self._move_to_idx = {move_id: idx + 1 for idx, move_id in enumerate(sorted_ids)}

    # --- Species ---
    def get_species_idx(self, species_id: str) -> int:
        """Return stable integer index for a species ID (0 = unknown/padding)."""
        if self._species_db is None:
            self._load_species()
        if species_id is None:
            return 0
        sid = str(species_id).lower().replace(' ', '').replace('-', '')
        return self._species_to_idx.get(sid, self._species_to_idx.get(species_id, 0))

    def get_species_data(self, species_id: str) -> dict:
        """Return full species info including types and base stats."""
        if self._species_db is None:
            self._load_species()
        if species_id is None:
            return {}
        sid = str(species_id).lower().replace(' ', '').replace('-', '')
        return self._species_db.get(sid, self._species_db.get(species_id, {}))

    def get_num_species(self) -> int:
        """Number of species including unknown/padding token."""
        if self._species_db is None:
            self._load_species()
        return len(self._species_to_idx) + 1 if self._species_to_idx else 1

    # --- Items ---
    def get_item_idx(self, item_id: str) -> int:
        """Return stable integer index for an item ID (0 = no item / unknown)."""
        if self._items_db is None:
            self._load_items()
        if not item_id:
            return 0
        return self._item_to_idx.get(str(item_id).lower(), 0)

    def get_num_items(self) -> int:
        """Number of items including unknown/padding token."""
        if self._items_db is None:
            self._load_items()
        return len(self._item_to_idx) + 1 if self._item_to_idx else 1

    # --- Abilities ---
    def get_ability_idx(self, ability_id: str) -> int:
        """Return stable integer index for an ability ID (0 = unknown/padding)."""
        if self._abilities_db is None:
            self._load_abilities()
        if not ability_id:
            return 0
        return self._ability_to_idx.get(str(ability_id).lower(), 0)

    def get_num_abilities(self) -> int:
        """Number of abilities including unknown/padding token."""
        if self._abilities_db is None:
            self._load_abilities()
        return len(self._ability_to_idx) + 1 if self._ability_to_idx else 1

    # --- Moves ---
    def get_move_idx(self, move_id: str) -> int:
        """Return unique integer for a move, suitable for Embedding layer (0 = unknown/padding)."""
        if self._moves_db is None:
            self._load_moves()
        if move_id not in self._move_to_idx and isinstance(move_id, str) and move_id.startswith("hiddenpower"):
            move_id = "hiddenpower"
        return self._move_to_idx.get(move_id, 0)

    def get_move_data(self, move_id: str) -> dict:
        """Return move data dictionary with basePower, accuracy, category, etc."""
        if self._moves_db is None:
            self._load_moves()
        # Default fallback values for an unknown move
        default_move = {
            "id": move_id,
            "name": move_id,
            "basePower": 0,
            "type": "Normal",
            "accuracy": 100,
            "category": "Physical"
        }
        db_move = self._moves_db.get(move_id, default_move)
        # Correct category for Gen 3 mechanics (global type-based split)
        if db_move.get("category") != "Status":
            db_move = dict(db_move)  # copy to avoid in-place mutation of cache
            move_type = db_move.get("type", "Normal").lower()
            special_types = {"fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark"}
            if move_type in special_types:
                db_move["category"] = "Special"
            else:
                db_move["category"] = "Physical"
        return db_move

    def get_num_moves(self) -> int:
        """Number of moves including unknown/padding token."""
        if self._moves_db is None:
            self._load_moves()
        return len(self._move_to_idx) + 1 if self._move_to_idx else 1

    def load_moves(self):
        """Load the moves database if not already loaded."""
        self._load_moves()

    @property
    def move_to_idx(self):
        """Mapping from move ID to index (read-only)."""
        return self._move_to_idx


# Singleton instance
db = PokémonDatabase()