import os
import sys
import pytest
from unittest.mock import patch

from core.client.showdown_client import ShowdownClient
from core.problem.pokemon_problem import PokemonProblem
from battle_agents.random.random_agent import RandomAgent
from battle_agents.blind_mcts.blind_mcts_agent import BlindMCTSAgent
from battle_agents.mcts.mcts_agent import MCTSAgent
from battle_agents.mcts_approximation.mcts_approximation_agent import MCTSApproximationAgent

from benchmarks.tournament import TournamentBenchmark
from benchmarks.round_robin import RoundRobinBenchmark

# Setup path constants
ENGINE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../engine'))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/data/mcts_model.keras'))

@pytest.fixture(scope="module")
def client():
    # Use ShowdownClient
    cli = ShowdownClient(ENGINE_PATH)
    yield cli
    cli.close()

@pytest.fixture(scope="module")
def agents_factories():
    # We use very low iterations/depth to keep the test incredibly fast
    return {
        "Random": lambda prob: RandomAgent(prob),
        "BlindMCTS": lambda prob: BlindMCTSAgent(prob, iterations=2, max_rollout_depth=2),
        "MCTS": lambda prob: MCTSAgent(prob, iterations=2, max_rollout_depth=2),
        "MCTS_Approx": lambda prob: MCTSApproximationAgent(prob, iterations=2, max_rollout_depth=2, model_path=MODEL_PATH)
    }

def test_round_robin_benchmark(client, agents_factories):
    # Select only 2 agents to make it very fast
    factories = {k: agents_factories[k] for k in ["Random", "BlindMCTS"]}
    
    benchmark = RoundRobinBenchmark(client, factories, games_per_matchup=1)
    
    # We patch the problem to return terminal after 2 turns to speed up testing
    original_is_terminal = PokemonProblem.is_terminal
    
    def fast_terminal(self, state):
        if state.state_dict.get('turn', 0) >= 2:
            state.winner = "Player 1" # Arbitrary winner
            return True
        return original_is_terminal(self, state)
        
    with patch.object(PokemonProblem, 'is_terminal', fast_terminal):
        benchmark.run()
        
    # Verify results exist
    assert len(benchmark.results) == 1
    assert benchmark.results[0]['p1'] == "Random"
    assert benchmark.results[0]['p2'] == "BlindMCTS"
    
def test_tournament_benchmark(client, agents_factories):
    # Select all 4 agents
    benchmark = TournamentBenchmark(client, agents_factories, games_per_matchup=1, shuffle=False)
    
    # We patch the problem to return terminal after 2 turns to speed up testing
    original_is_terminal = PokemonProblem.is_terminal
    
    def fast_terminal(self, state):
        if state.state_dict.get('turn', 0) >= 2:
            state.winner = "Player 1" # Arbitrary winner
            return True
        return original_is_terminal(self, state)
        
    with patch.object(PokemonProblem, 'is_terminal', fast_terminal):
        benchmark.run()
        
    # Verify tournament completed
    assert hasattr(benchmark, 'standings') or len(benchmark.results) > 0
