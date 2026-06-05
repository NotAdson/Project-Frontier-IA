from core.benchmark import BaseBenchmark


class RoundRobinBenchmark(BaseBenchmark):
    """ Benchmark that matches a dictionary of agents against every other agent in a full tournament format. """
    def __init__(self, client, agent_factories, games_per_matchup=5, formatid="gen3randombattle"):
        super().__init__(client, formatid)
        self.agent_factories = agent_factories # Dict: name -> factory
        self.games_per_matchup = games_per_matchup
        
    def run(self):
        agent_names = list(self.agent_factories.keys())
        print(f"\nStarting Round Robin Tournament ({len(agent_names)} agents, {self.games_per_matchup} games per matchup)")
        
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                name1 = agent_names[i]
                name2 = agent_names[j]
                
                print(f"\n-> Matchup: {name1} vs {name2}")
                for k in range(self.games_per_matchup):
                    print(f"   Game {k+1}/{self.games_per_matchup}...")
                    if k % 2 == 0:
                        self.run_match(name1, self.agent_factories[name1], name2, self.agent_factories[name2])
                    else:
                        self.run_match(name2, self.agent_factories[name2], name1, self.agent_factories[name1])
        return self.results
