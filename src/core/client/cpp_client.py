import ctypes
import json
import os
import subprocess

from core.client.base_client import BaseClient


class CppClient(BaseClient):
    def __init__(self, lib_path, engine_path):
        self.lib_path = lib_path
        self.engine_path = engine_path
        
        # Load the shared library
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"Shared library not found at: {self.lib_path}")
            
        self.lib = ctypes.CDLL(self.lib_path)
        
        # Define argtypes and restypes
        self.lib.create_battle_builder.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        self.lib.create_battle_builder.restype = ctypes.c_void_p
        
        self.lib.add_pokemon.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
        ]
        self.lib.add_pokemon.restype = None
        
        self.lib.add_move.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        self.lib.add_move.restype = None
        
        self.lib.build_battle.argtypes = [ctypes.c_void_p]
        self.lib.build_battle.restype = ctypes.c_void_p
        
        self.lib.run_turn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.run_turn.restype = None
        
        self.lib.is_terminal.argtypes = [ctypes.c_void_p]
        self.lib.is_terminal.restype = ctypes.c_bool
        
        self.lib.get_winner.argtypes = [ctypes.c_void_p]
        self.lib.get_winner.restype = ctypes.c_char_p
        
        self.lib.get_turn_count.argtypes = [ctypes.c_void_p]
        self.lib.get_turn_count.restype = ctypes.c_int
        
        self.lib.get_active_index.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.get_active_index.restype = ctypes.c_int
        
        self.lib.get_team_size.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.get_team_size.restype = ctypes.c_int
        
        self.lib.get_pokemon_info.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.c_char_p, ctypes.c_char_p
        ]
        self.lib.get_pokemon_info.restype = None
        
        self.lib.get_valid_actions.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.get_valid_actions.restype = ctypes.c_int
        
        self.lib.delete_battle.argtypes = [ctypes.c_void_p]
        self.lib.delete_battle.restype = None

        # State cache functions
        self.lib.cache_battle.argtypes = [ctypes.c_void_p]
        self.lib.cache_battle.restype = ctypes.c_int
        
        self.lib.get_cached_battle.argtypes = [ctypes.c_int]
        self.lib.get_cached_battle.restype = ctypes.c_void_p
        
        self.lib.delete_cached_battle.argtypes = [ctypes.c_int]
        self.lib.delete_cached_battle.restype = None
        
        self.lib.clear_cache.argtypes = []
        self.lib.clear_cache.restype = None
        
        # Rollout function
        self.lib.run_rollout.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.run_rollout.restype = ctypes.c_double

        self.battle_ptr = None
        self.teams_data = None
        
    def generate_random_teams(self, formatid="gen3randombattle"):
        script_path = os.path.join(self.engine_path, "generate_teams_with_stats.js")
        proc = subprocess.run(["node", script_path, formatid], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to generate random teams: {proc.stderr}")
        return json.loads(proc.stdout)
        
    def init_battle(self, formatid='gen3randombattle', p1_team=None, p2_team=None, seed=1337):
        if self.battle_ptr:
            self.lib.delete_battle(self.battle_ptr)
            self.battle_ptr = None
            
        self.lib.clear_cache()
            
        # If teams are not provided, generate random ones using Node
        if p1_team is None or p2_team is None:
            teams = self.generate_random_teams(formatid)
            p1_team = teams["p1"]
            p2_team = teams["p2"]
            
        self.teams_data = {"p1": p1_team, "p2": p2_team}
        
        # Build the battle builder
        builder = self.lib.create_battle_builder(b"Player 1", b"Player 2", seed)
        
        # Add P1 pokemon
        for idx, p in enumerate(p1_team):
            self.lib.add_pokemon(
                builder, 1,
                p["species_id"].encode(), p["name"].encode(), p["level"],
                p["hp"], p["max_hp"], p["atk"], p["def"], p["spa"], p["spd"], p["spe"],
                p["type1"].encode(), p["type2"].encode(), p["ability"].encode(), p["item"].encode()
            )
            for move in p["moves"]:
                self.lib.add_move(builder, 1, idx, move.encode())
                
        # Add P2 pokemon
        for idx, p in enumerate(p2_team):
            self.lib.add_pokemon(
                builder, 2,
                p["species_id"].encode(), p["name"].encode(), p["level"],
                p["hp"], p["max_hp"], p["atk"], p["def"], p["spa"], p["spd"], p["spe"],
                p["type1"].encode(), p["type2"].encode(), p["ability"].encode(), p["item"].encode()
            )
            for move in p["moves"]:
                self.lib.add_move(builder, 2, idx, move.encode())
                
        self.battle_ptr = self.lib.build_battle(builder)
        
        # Cache the initial state to get a valid state ID
        initial_state_id = self.lib.cache_battle(self.battle_ptr)
        
        # Build initial response
        state_dict = self.get_state_dict(initial_state_id)
        p1_actions = self.get_valid_actions_list(initial_state_id, "p1")
        p2_actions = self.get_valid_actions_list(initial_state_id, "p2")
        
        resp = {
            "type": "success",
            "state_id": initial_state_id,
            "state": state_dict,
            "request": self.make_request(state_dict, "p1", p1_actions),
            "p2_request": self.make_request(state_dict, "p2", p2_actions),
            "winner": state_dict["winner"],
            "log": []
        }
        return resp
        
    def get_pokemon_list_dict(self, state_id, player):
        player_num = 1 if player == "p1" else 2
        battle_ptr = self.lib.get_cached_battle(state_id)
        
        size = self.lib.get_team_size(battle_ptr, player_num)
        active_idx = self.lib.get_active_index(battle_ptr, player_num)
        
        pokemon_list = []
        for i in range(size):
            hp = ctypes.c_int()
            max_hp = ctypes.c_int()
            name_buf = ctypes.create_string_buffer(100)
            status_buf = ctypes.create_string_buffer(100)
            
            self.lib.get_pokemon_info(battle_ptr, player_num, i, ctypes.byref(hp), ctypes.byref(max_hp), name_buf, status_buf)
            
            # Reconstruct original team details
            orig_p = self.teams_data[player][i]
            
            pokemon_list.append({
                "species": orig_p["species_id"],
                "name": name_buf.value.decode(),
                "level": orig_p["level"],
                "hp": hp.value,
                "maxhp": max_hp.value,
                "status": status_buf.value.decode() if status_buf.value.decode() != "NONE" else "",
                "isActive": i == active_idx,
                "active": i == active_idx,
                "condition": f"{hp.value}/{max_hp.value}" if hp.value > 0 else "0 fnt",
                "stats": {
                    "atk": orig_p["atk"],
                    "def": orig_p["def"],
                    "spa": orig_p["spa"],
                    "spd": orig_p["spd"],
                    "spe": orig_p["spe"]
                },
                "types": [orig_p["type1"], orig_p["type2"]],
                "moves": orig_p["moves"],
                "ability": orig_p["ability"],
                "item": orig_p["item"]
            })
        return pokemon_list
        
    def get_state_dict(self, state_id):
        p1_poke = self.get_pokemon_list_dict(state_id, "p1")
        p2_poke = self.get_pokemon_list_dict(state_id, "p2")
        
        battle_ptr = self.lib.get_cached_battle(state_id)
        winner_str = self.lib.get_winner(battle_ptr).decode()
        winner = winner_str if winner_str else None
        
        state_dict = {
            "sides": [
                {
                    "name": "Player 1",
                    "id": "p1",
                    "pokemon": p1_poke
                },
                {
                    "name": "Player 2",
                    "id": "p2",
                    "pokemon": p2_poke
                }
            ],
            "field": {
                "weather": ""
            },
            "winner": winner,
            "turn": self.lib.get_turn_count(battle_ptr)
        }
        return state_dict
        
    def get_valid_actions_list(self, state_id, player):
        player_num = 1 if player == "p1" else 2
        battle_ptr = self.lib.get_cached_battle(state_id)
        
        max_actions = 20
        action_types = (ctypes.c_int * max_actions)()
        action_indices = (ctypes.c_int * max_actions)()
        
        count = self.lib.get_valid_actions(battle_ptr, player_num, action_types, action_indices)
        
        actions = []
        for i in range(count):
            act_type = action_types[i]
            act_idx = action_indices[i]
            
            if act_type == 0:  # MOVE
                actions.append(f"move {act_idx + 1}")
            elif act_type == 1:  # SWITCH
                actions.append(f"switch {act_idx + 1}")
            else:  # PASS
                actions.append("pass")
        return actions
        
    def make_request(self, state_dict, player, actions):
        pokemon_list = state_dict["sides"][0 if player == "p1" else 1]["pokemon"]
        active_p = next((p for p in pokemon_list if p["isActive"]), None)
        
        force_switch = any("switch" in a for a in actions) and not any("move" in a for a in actions)
        
        req = {
            "side": {
                "pokemon": pokemon_list
            }
        }
        if force_switch:
            req["forceSwitch"] = True
        else:
            moves = []
            if active_p:
                for i, m_id in enumerate(active_p["moves"]):
                    disabled = not f"move {i+1}" in actions
                    moves.append({"move": m_id, "id": m_id, "disabled": disabled})
            req["active"] = [{"moves": moves}]
            
        return req
        
    def get_result(self, state, p1_action, p2_action=None, state_id=None):
        # Map actions to ints
        def parse_action(act):
            if not act or act == "pass":
                return 2, 0  # PASS
            parts = act.split()
            if len(parts) != 2:
                return 2, 0
            act_type_str, idx_str = parts[0], parts[1]
            idx = int(idx_str) - 1
            if act_type_str == "move":
                return 0, idx
            elif act_type_str == "switch":
                return 1, idx
            return 2, 0
            
        p1_type, p1_idx = parse_action(p1_action)
        p2_type, p2_idx = parse_action(p2_action)
        
        # Clone parent cached battle to a new state
        parent_ptr = self.lib.get_cached_battle(state_id)
        new_state_id = self.lib.cache_battle(parent_ptr)
        new_ptr = self.lib.get_cached_battle(new_state_id)
        
        # Run turn on the clone
        self.lib.run_turn(new_ptr, p1_type, p1_idx, p2_type, p2_idx)
        
        # Build response
        state_dict = self.get_state_dict(new_state_id)
        p1_actions = self.get_valid_actions_list(new_state_id, "p1")
        p2_actions = self.get_valid_actions_list(new_state_id, "p2")
        
        resp = {
            "type": "success",
            "state_id": new_state_id,
            "state": state_dict,
            "request": self.make_request(state_dict, "p1", p1_actions),
            "p2_request": self.make_request(state_dict, "p2", p2_actions),
            "winner": state_dict["winner"],
            "log": []
        }
        return resp
        
    def rollout(self, state, player, max_depth, state_id=None):
        player_num = 1 if player == "p1" else 2
        battle_ptr = self.lib.get_cached_battle(state_id)
        reward = self.lib.run_rollout(battle_ptr, player_num, max_depth)
        return reward
        
    def close(self):
        self.lib.clear_cache()
        if self.battle_ptr:
            self.lib.delete_battle(self.battle_ptr)
            self.battle_ptr = None
