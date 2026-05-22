import time
from abc import ABC, abstractmethod
from collections import defaultdict
from core.problem.pokemon_problem import PokemonProblem
from core.client.showdown_client import ShowdownClient

class BaseBenchmark(ABC):
    """
    Abstract Base Class for running AI benchmarks and tracking metrics.
    """
    def __init__(self, client: ShowdownClient, formatid="gen3randombattle"):
        self.client = client
        self.formatid = formatid
        self.results = []
        
    def _count_survivors(self, state, player):
        """ Count how many Pokemon are alive for a given player side """
        player_idx = 0 if player == 'p1' else 1
        side = state.state_dict.get('sides', [{}, {}])[player_idx]
        survivors = 0
        for p in side.get('pokemon', []):
            # In Showdown, condition is usually something like '100/100', or '0 fnt' if fainted.
            condition = p.get('condition', '')
            if not condition.endswith('fnt') and not condition == '0 fnt':
                survivors += 1
        return survivors

    def run_match(self, p1_name, p1_factory, p2_name, p2_factory):
        """ Executes a single match between two agents and tracks metrics. """
        problem = PokemonProblem(self.client, formatid=self.formatid)
        p1_agent = p1_factory(problem)
        p2_agent = p2_factory(problem)
        
        state = problem.initial
        turn = 1
        
        p1_times = []
        p2_times = []
        
        while not problem.is_terminal(state):
            # Player 1 action
            start = time.perf_counter()
            p1_action = p1_agent.get_action(state, player="p1")
            p1_times.append(time.perf_counter() - start)
            
            # Player 2 action
            start = time.perf_counter()
            p2_action = p2_agent.get_action(state, player="p2")
            p2_times.append(time.perf_counter() - start)
            
            state = problem.result(state, p1_action=p1_action, p2_action=p2_action)
            turn += 1
            
        p1_won = problem.is_goal(state)
        
        winner_name = None
        survivors = 0
        
        if state.winner == 'Player 1':
            winner_name = p1_name
            survivors = self._count_survivors(state, 'p1')
        elif state.winner == 'Player 2':
            winner_name = p2_name
            survivors = self._count_survivors(state, 'p2')
        else:
            # Fallback if engine string match fails
            if p1_won:
                winner_name = p1_name
                survivors = self._count_survivors(state, 'p1')
            else:
                winner_name = p2_name
                survivors = self._count_survivors(state, 'p2')

        match_result = {
            'p1': p1_name,
            'p2': p2_name,
            'winner': winner_name,
            'turns': turn - 1,
            'p1_avg_time': sum(p1_times) / len(p1_times) if p1_times else 0,
            'p2_avg_time': sum(p2_times) / len(p2_times) if p2_times else 0,
            'survivors': survivors,
            'log': state.log if state.log else []
        }
        self.results.append(match_result)
        return match_result

    def get_logs(self):
        """ Returns a list of the full battle logs from all completed matches. """
        return [r['log'] for r in self.results]

    @abstractmethod
    def run(self):
        """ Abstract method for triggering the full benchmark suite. """
        pass

    def print_report(self):
        """ Renders a comprehensive statistics table of the results. """
        print("\n" + "="*80)
        print(" "*30 + "BENCHMARK REPORT")
        print("="*80)
        print(f"Total Matches Played: {len(self.results)}")
        
        # Aggregate stats per agent
        stats = defaultdict(lambda: {'wins': 0, 'matches': 0, 'total_time': 0, 'turns': 0, 'total_survivors': 0, 'survivor_wins': 0})
        
        for r in self.results:
            p1, p2, winner = r['p1'], r['p2'], r['winner']
            
            stats[p1]['matches'] += 1
            stats[p2]['matches'] += 1
            
            stats[p1]['total_time'] += r['p1_avg_time']
            stats[p2]['total_time'] += r['p2_avg_time']
            stats[p1]['turns'] += r['turns']
            stats[p2]['turns'] += r['turns']
            
            if winner == p1:
                stats[p1]['wins'] += 1
                stats[p1]['total_survivors'] += r['survivors']
                stats[p1]['survivor_wins'] += 1
            elif winner == p2:
                stats[p2]['wins'] += 1
                stats[p2]['total_survivors'] += r['survivors']
                stats[p2]['survivor_wins'] += 1
                
        print(f"\n{'Agent Name':<20} | {'Win Rate':<10} | {'Avg Time/Turn':<15} | {'Avg Survivors (On Win)':<25}")
        print("-" * 80)
        for agent, data in stats.items():
            win_rate = (data['wins'] / data['matches']) * 100 if data['matches'] > 0 else 0
            avg_time = data['total_time'] / data['matches'] if data['matches'] > 0 else 0
            avg_survivors = data['total_survivors'] / data['survivor_wins'] if data['survivor_wins'] > 0 else 0
            
            print(f"{agent:<20} | {win_rate:>5.1f}%     | {avg_time:>10.3f}s    | {avg_survivors:>5.1f} / 6")
        print("="*80 + "\n")
