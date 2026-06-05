"""
Engine correctness tests for the C++ battle engine.
Run from project root: python3 new_engine/test_engine.py
"""
import sys, os, ctypes, random
sys.path.append(os.path.abspath("src"))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

LIB_PATH    = os.path.abspath("new_engine/libbattle.so")
ENGINE_PATH = os.path.abspath("engine")

from core.client.cpp_client import CppClient
from core.problem.pokemon_problem import PokemonProblem

PASS_MARK = "\033[92m[PASS]\033[0m"
FAIL_MARK = "\033[91m[FAIL]\033[0m"
results = []

def assert_test(name, condition, detail=""):
    mark = PASS_MARK if condition else FAIL_MARK
    print(f"{mark} {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)

def random_game(max_turns=1000):
    client = CppClient(LIB_PATH, ENGINE_PATH)
    problem = PokemonProblem(client, formatid="gen3randombattle")
    state = problem.initial
    turns = 0
    while not problem.is_terminal(state) and turns < max_turns + 5:
        a1 = random.choice(problem.actions(state, "p1"))
        a2 = random.choice(problem.actions(state, "p2"))
        state = problem.result(state, a1, a2)
        turns += 1
    client.close()
    return turns, state

print("\n--- Running engine tests ---\n")

# TEST 1: game terminates within 1005 steps
turns, state = random_game()
assert_test("test_game_terminates", turns < 1005, f"turns={turns}, winner={state.winner}")

# TEST 2: valid_actions never returns switch-only
client = CppClient(LIB_PATH, ENGINE_PATH)
problem = PokemonProblem(client, formatid="gen3randombattle")
state = problem.initial
switch_only_found = False
for _ in range(30):
    if problem.is_terminal(state):
        break
    acts = problem.actions(state, "p1")
    move_acts = [a for a in acts if a.startswith("move")]
    if not move_acts and acts != ["pass"]:
        switch_only_found = True
        break
    a1 = random.choice(acts)
    a2 = random.choice(problem.actions(state, "p2"))
    state = problem.result(state, a1, a2)
client.close()
assert_test("test_no_forced_switch_actions", not switch_only_found)

# TEST 3: is_terminal works
client = CppClient(LIB_PATH, ENGINE_PATH)
problem = PokemonProblem(client, formatid="gen3randombattle")
state = problem.initial
for _ in range(1005):
    if problem.is_terminal(state):
        break
    a1 = random.choice(problem.actions(state, "p1"))
    a2 = random.choice(problem.actions(state, "p2"))
    state = problem.result(state, a1, a2)
client.close()
assert_test("test_is_terminal_detected", problem.is_terminal(state),
            f"winner={state.winner}, turn={state.state_dict.get('turn')}")

# TEST 4: turn count increments
client = CppClient(LIB_PATH, ENGINE_PATH)
problem = PokemonProblem(client, formatid="gen3randombattle")
state = problem.initial
for _ in range(5):
    if problem.is_terminal(state):
        break
    state = problem.result(state, random.choice(problem.actions(state,"p1")),
                           random.choice(problem.actions(state,"p2")))
turn = state.state_dict.get('turn', 0)
client.close()
assert_test("test_turn_count_increments", turn >= 1, f"turn={turn}")

# TEST 5: type effectiveness via raw ctypes
lib = ctypes.CDLL(LIB_PATH)
lib.create_battle_builder.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
lib.create_battle_builder.restype  = ctypes.c_void_p
lib.add_pokemon.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,
    ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
lib.add_pokemon.restype  = None
lib.add_move.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
lib.add_move.restype     = None
lib.build_battle.argtypes = [ctypes.c_void_p]
lib.build_battle.restype  = ctypes.c_void_p
lib.run_turn.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.run_turn.restype     = None
lib.get_pokemon_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.c_char_p, ctypes.c_char_p]
lib.get_pokemon_info.restype = None
lib.delete_battle.argtypes = [ctypes.c_void_p]
lib.delete_battle.restype  = None

def hp_after_flamethrower(def_type1, def_type2, seed=100):
    b = lib.create_battle_builder(b"P1", b"P2", seed)
    lib.add_pokemon(b, 1, b"rattata", b"Rattata", 50,
                    200, 200, 200, 100, 200, 100, 100,
                    b"normal", b"none", b"none", b"none")
    lib.add_move(b, 1, 0, b"flamethrower")
    lib.add_pokemon(b, 2, b"defender", b"Defender", 50,
                    300, 300, 100, 100, 100, 100, 100,
                    def_type1, def_type2, b"none", b"none")
    lib.add_move(b, 2, 0, b"tackle")
    battle = lib.build_battle(b)
    lib.run_turn(battle, 0, 0, 0, 0)
    hp = ctypes.c_int(); max_hp = ctypes.c_int()
    name_buf = ctypes.create_string_buffer(100)
    status_buf = ctypes.create_string_buffer(100)
    lib.get_pokemon_info(battle, 2, 0, ctypes.byref(hp), ctypes.byref(max_hp), name_buf, status_buf)
    lib.delete_battle(battle)
    return hp.value

hp_grass  = hp_after_flamethrower(b"grass",  b"none")
hp_normal = hp_after_flamethrower(b"normal", b"none")
assert_test("test_type_effectiveness", hp_grass < hp_normal,
            f"HP left: vs_grass={hp_grass}, vs_normal={hp_normal}")

print(f"\n{sum(results)}/{len(results)} tests passed {'✓' if all(results) else '✗'}\n")
sys.exit(0 if all(results) else 1)
