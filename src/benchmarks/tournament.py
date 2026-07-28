import random
from collections import defaultdict

from core.benchmark import BaseBenchmark


class TournamentBenchmark(BaseBenchmark):
    """ Benchmark that structures agents into a single-elimination bracket tournament. """
    def __init__(self, client, agent_factories, games_per_matchup=3, formatid="gen3ou", shuffle=True):
        super().__init__(client, formatid)
        self.agent_factories = agent_factories # Dict: name -> factory
        self.games_per_matchup = games_per_matchup
        self.shuffle = shuffle
        self.placements = [] # Records who was eliminated in each round
        
    def run(self):
        agent_names = list(self.agent_factories.keys())
        if self.shuffle:
            random.shuffle(agent_names)
            
        print(f"\nStarting Tournament Bracket ({len(agent_names)} agents)")
        current_round = agent_names
        round_num = 1
        
        while len(current_round) > 1:
            print(f"\n--- Round {round_num} ---")
            next_round = []
            eliminated_this_round = []
            
            for i in range(0, len(current_round), 2):
                if i + 1 == len(current_round):
                    print(f"-> Matchup: {current_round[i]} receives a BYE!")
                    next_round.append(current_round[i])
                else:
                    name1 = current_round[i]
                    name2 = current_round[i+1]
                    print(f"-> Matchup: {name1} vs {name2} (Best of {self.games_per_matchup})")
                    
                    wins = {name1: 0, name2: 0}
                    
                    # Play series
                    for k in range(self.games_per_matchup):
                        print(f"   Game {k+1}/{self.games_per_matchup}...")
                        if k % 2 == 0:
                            res = self.run_match(name1, self.agent_factories[name1], name2, self.agent_factories[name2])
                        else:
                            res = self.run_match(name2, self.agent_factories[name2], name1, self.agent_factories[name1])
                            
                        if res['winner'] in wins:
                            wins[res['winner']] += 1
                            
                    # Sudden death tiebreaker if needed
                    while wins[name1] == wins[name2]:
                        print(f"   TIE! Running sudden death tiebreaker...")
                        res = self.run_match(name1, self.agent_factories[name1], name2, self.agent_factories[name2])
                        if res['winner'] in wins:
                            wins[res['winner']] += 1
                            
                    winner = name1 if wins[name1] > wins[name2] else name2
                    loser = name2 if winner == name1 else name1
                    
                    print(f"   *** {winner} wins the series! ***")
                    next_round.append(winner)
                    eliminated_this_round.append(loser)
                    
            self.placements.append(eliminated_this_round)
            current_round = next_round
            round_num += 1
            
        champion = current_round[0]
        self.placements.append([champion])
        print(f"\n*** TOURNAMENT CHAMPION: {champion} ***")
        return self.results

    def print_report(self):
        """ Renders a comprehensive ranking table of the tournament. """
        print("\n" + "="*80)
        print(" "*28 + "TOURNAMENT RANKINGS")
        print("="*80)
        print(f"Total Matches Played: {len(self.results)}\n")
        
        # Calculate stats for extra info
        stats = defaultdict(lambda: {'total_time': 0, 'matches': 0})
        for r in self.results:
            stats[r['p1']]['total_time'] += r['p1_avg_time']
            stats[r['p1']]['matches'] += 1
            stats[r['p2']]['total_time'] += r['p2_avg_time']
            stats[r['p2']]['matches'] += 1
            
        # Reverse placements so champion is first
        reversed_placements = list(reversed(self.placements))
        
        place = 1
        for i, group in enumerate(reversed_placements):
            if not group: continue
            
            # Formatting places (1st, 2nd, 3rd-4th, 5th-8th)
            if len(group) == 1:
                place_str = f"{place}{'st' if place==1 else 'nd' if place==2 else 'rd' if place==3 else 'th'} Place"
            else:
                place_str = f"Tied {place}th-{place + len(group) - 1}th Place"
                
            print(f"--- {place_str} ---")
            for agent in group:
                avg_t = stats[agent]['total_time'] / stats[agent]['matches'] if stats[agent]['matches'] > 0 else 0
                print(f" > {agent:<25} [Avg Compute: {avg_t:.3f}s/turn]")
            print()
            
            place += len(group)
            
        print("="*80 + "\n")
