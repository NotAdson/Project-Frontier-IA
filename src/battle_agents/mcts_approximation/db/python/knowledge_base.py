"""
Knowledge Base Search and Match module.
Combines multiple neural network predictions (species probability, base stats, types)
to query and match the closest real Pokémon in the database.
"""
import numpy as np

from battle_agents.mcts_approximation.db.python.database import db
from battle_agents.mcts_approximation.state_encoder import TYPES, _encode_types

# Cache variables to avoid redundant parsing/calculations during search
_db_loaded = False
_species_ids_by_idx = {}
_real_stats_matrix = None  # shape (num_species, 5)
_real_types_matrix = None  # shape (num_species, 18)

def _init_cache():
    global _db_loaded, _species_ids_by_idx, _real_stats_matrix, _real_types_matrix
    if _db_loaded:
        return
    
    db._load_species()
    num_species = len(db._species_to_idx) + 1
    
    # Pre-allocate numpy matrices
    # Columns for stats: [atk, def, spa, spd, spe]
    _real_stats_matrix = np.zeros((num_species, 5), dtype=np.float32)
    _real_types_matrix = np.zeros((num_species, 18), dtype=np.float32)
    
    _species_ids_by_idx = {0: "unknown"}
    
    for sid, idx in db._species_to_idx.items():
        _species_ids_by_idx[idx] = sid
        data = db._species_db[sid]
        
        # Stats normalized by dividing by 500.0 (same as state_encoder)
        base_stats = data.get("baseStats", {})
        atk = base_stats.get("atk", 0) / 500.0
        def_ = base_stats.get("def", 0) / 500.0
        spa = base_stats.get("spa", 0) / 500.0
        spd = base_stats.get("spd", 0) / 500.0
        spe = base_stats.get("spe", 0) / 500.0
        _real_stats_matrix[idx] = [atk, def_, spa, spd, spe]
        
        # Types encoded as 18-element binary multi-hot array
        types_list = data.get("types", [])
        _real_types_matrix[idx] = _encode_types(types_list)
        
    _db_loaded = True

def find_closest_species(
    pred_species_probs,
    pred_stats,
    pred_types,
    weight_species=1.0,
    weight_stats=1.0,
    weight_types=1.0,
    top_k=5
):
    """
    Finds the top_k closest real Pokémon matching the network predictions.
    
    Parameters:
      - pred_species_probs: probability distribution over species IDs (shape: (num_species,))
      - pred_stats: predicted normalized stats [atk, def, spa, spd, spe] (shape: (5,))
      - pred_types: predicted type probabilities (shape: (18,))
      - weight_species: weight for the species probability component
      - weight_stats: weight for the base stats distance component
      - weight_types: weight for the type signature distance component
      - top_k: number of matches to return
      
    Returns a list of dictionaries with matching details.
    """
    _init_cache()
    
    # Ensure inputs are clean numpy arrays
    pred_species_probs = np.array(pred_species_probs, dtype=np.float32)
    pred_stats = np.array(pred_stats, dtype=np.float32)
    pred_types = np.array(pred_types, dtype=np.float32)
    
    # 1. Species distance component: 1.0 - probability
    d_species = 1.0 - pred_species_probs
    
    # 2. Stats distance component: squared L2 norm
    diff_stats = _real_stats_matrix - pred_stats
    d_stats = np.sum(diff_stats ** 2, axis=1)
    
    # 3. Types distance component: squared L2 norm
    diff_types = _real_types_matrix - pred_types
    d_types = np.sum(diff_types ** 2, axis=1)
    
    # Combined distance score (lower is closer/better)
    total_dist = (weight_species * d_species) + (weight_stats * d_stats) + (weight_types * d_types)
    
    # Exclude index 0 (unknown/padding)
    total_dist[0] = np.inf
    
    # Sort and pick top K
    top_indices = np.argsort(total_dist)[:top_k]
    
    results = []
    for idx in top_indices:
        sid = _species_ids_by_idx[idx]
        data = db._species_db[sid]
        
        results.append({
            "species_id": sid,
            "name": data.get("name", sid),
            "distance": float(total_dist[idx]),
            "d_species": float(d_species[idx]),
            "d_stats": float(d_stats[idx]),
            "d_types": float(d_types[idx]),
            "prob": float(pred_species_probs[idx]),
            "types": data.get("types", []),
            "baseStats": data.get("baseStats", {}),
        })
        
    return results
