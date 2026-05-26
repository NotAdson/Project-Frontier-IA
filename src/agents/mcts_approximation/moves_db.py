import json
from pathlib import Path

_moves_db = None
_move_to_idx = None

def _load_db():
    global _moves_db, _move_to_idx
    if _moves_db is None:
        db_path = Path(__file__).resolve().parent / 'moves.json'
        if db_path.exists():
            with open(db_path, 'r') as f:
                _moves_db = json.load(f)
        else:
            _moves_db = {}
            
        # Create a stable integer mapping for embeddings (1-indexed, 0 is padding/unknown)
        sorted_moves = sorted(list(_moves_db.keys()))
        _move_to_idx = {move_id: idx + 1 for idx, move_id in enumerate(sorted_moves)}

def get_move_data(move_id):
    _load_db()
    
    # Default fallback values for an unknown move
    default_move = {
        "id": move_id,
        "name": move_id,
        "basePower": 0,
        "type": "Normal",
        "accuracy": 100,
        "category": "Physical"
    }
    
    if move_id not in _moves_db and isinstance(move_id, str) and move_id.startswith("hiddenpower"):
        move_id = "hiddenpower"
        
    return _moves_db.get(move_id, default_move)

def get_move_idx(move_id):
    """Returns a unique integer for the move, suitable for an Embedding layer. 0 is unknown/padding."""
    _load_db()
    if move_id not in _move_to_idx and isinstance(move_id, str) and move_id.startswith("hiddenpower"):
        move_id = "hiddenpower"
    return _move_to_idx.get(move_id, 0)

def get_num_moves():
    """Returns the total number of unique moves + 1 (for padding 0)."""
    _load_db()
    return len(_move_to_idx) + 1 if _move_to_idx else 1
