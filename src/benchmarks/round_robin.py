"""Round-robin benchmark.

Supports two execution modes backed by :class:`RoundRobinBenchmark`:

* Sequential (``processes <= 1``): uses callable ``agent_factories``.
* Parallel (``processes > 1``): splits the tournament into one job **per game**
  and fans them out across a ``multiprocessing.Pool``. Because spawn workers
  cannot pickle lambdas, the parallel path requires a picklable ``agent_specs``
  mapping (dict of primitive values) instead of factory functions.
"""

import multiprocessing
from pathlib import Path

from core.benchmark import BaseBenchmark
from core.client.showdown_client import ShowdownClient
from core.problem.pokemon_problem import PokemonProblem


def build_agent(spec, problem):
    """Reconstruct an agent from a picklable spec dict (used in workers)."""
    agent_type = spec["type"]
    if agent_type == "mcts_approx":
        from battle_agents.mcts_approximation.mcts_approximation_agent import \
            MCTSApproximationAgent
        return MCTSApproximationAgent(
            problem,
            iterations=spec["iterations"],
            model_path=spec["model_path"],
            dirichlet_epsilon=spec.get("dirichlet_epsilon", 0.0),
        )
    if agent_type == "blind_mcts":
        from battle_agents.blind_mcts.blind_mcts_agent import BlindMCTSAgent
        return BlindMCTSAgent(
            problem,
            iterations=spec["iterations"],
            max_rollout_depth=spec.get("max_rollout_depth", 20),
        )
    if agent_type == "mcts_pure":
        from battle_agents.mcts.mcts_agent import MCTSAgent
        return MCTSAgent(
            problem,
            iterations=spec["iterations"],
            max_rollout_depth=spec.get("max_rollout_depth", 20),
        )
    if agent_type == "random":
        from battle_agents.random.random_agent import RandomAgent
        return RandomAgent(problem)
    raise ValueError(f"Unknown agent spec type: {agent_type!r}")


def _count_survivors(state, player):
    """Number of alive Pokemon for a player side (mirrors BaseBenchmark)."""
    player_idx = 0 if player == "p1" else 1
    side = state.state_dict.get("sides", [{}, {}])[player_idx]
    survivors = 0
    for p in side.get("pokemon", []):
        condition = p.get("condition", "")
        if not condition.endswith("fnt") and condition != "0 fnt":
            survivors += 1
    return survivors


def run_benchmark_match(args):
    """Runs a single match in an isolated worker process.

    ``args`` is a tuple:
        (engine_path, formatid, p1_name, p1_spec, p2_name, p2_spec,
         p1_team, p2_team)

    Returns a match_result dict with the same keys as
    :meth:`BaseBenchmark.run_match`.
    """
    import time

    (engine_path, formatid, p1_name, p1_spec, p2_name, p2_spec,
     p1_team, p2_team) = args

    client = ShowdownClient(engine_path)
    try:
        problem = PokemonProblem(client, formatid=formatid,
                                 p1_team=p1_team, p2_team=p2_team)
        p1_agent = build_agent(p1_spec, problem)
        p2_agent = build_agent(p2_spec, problem)

        state = problem.initial
        turn = 1
        p1_times, p2_times = [], []

        while not problem.is_terminal(state):
            start = time.perf_counter()
            p1_action = p1_agent.get_action(state, player="p1")
            p1_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            p2_action = p2_agent.get_action(state, player="p2")
            p2_times.append(time.perf_counter() - start)

            state = problem.result(state, p1_action=p1_action, p2_action=p2_action)
            turn += 1

        p1_won = problem.is_goal(state)

        winner_name = None
        survivors = 0
        if state.winner == "Player 1":
            winner_name = p1_name
            survivors = _count_survivors(state, "p1")
        elif state.winner == "Player 2":
            winner_name = p2_name
            survivors = _count_survivors(state, "p2")
        else:
            if p1_won:
                winner_name = p1_name
                survivors = _count_survivors(state, "p1")
            else:
                winner_name = p2_name
                survivors = _count_survivors(state, "p2")

        return {
            "p1": p1_name,
            "p2": p2_name,
            "winner": winner_name,
            "turns": turn - 1,
            "p1_avg_time": sum(p1_times) / len(p1_times) if p1_times else 0,
            "p2_avg_time": sum(p2_times) / len(p2_times) if p2_times else 0,
            "survivors": survivors,
            "log": state.log if state.log else [],
        }
    finally:
        client.close()


def _default_engine_path():
    return str(Path(__file__).resolve().parents[3] / "engine")


class RoundRobinBenchmark(BaseBenchmark):
    """Benchmark that matches a set of agents against every other agent.

    Parameters
    ----------
    client: ShowdownClient for the sequential path (unused when processes > 1).
    agent_factories: dict name -> factory (sequential path only).
    games_per_matchup: number of games per agent pair.
    formatid: battle format.
    processes: if > 1, run the games in parallel using picklable agent_specs.
    agent_specs: picklable dict name -> spec, required when processes > 1.
        Spec is itself a dict with a "type" key and type-specific fields, e.g.
        {"type": "mcts_approx", "iterations": 300, "model_path": "x.onnx"}.
    """
    def __init__(self, client, agent_factories=None, games_per_matchup=5,
                 formatid="gen3ou", processes=1, agent_specs=None):
        super().__init__(client, formatid)
        self.agent_factories = agent_factories or {}
        self.agent_specs = agent_specs or {}
        self.games_per_matchup = games_per_matchup
        self.processes = processes

    def _games_for(self, name1, name2):
        """Number of games for a given agent pair.

        ``games_per_matchup`` may be an int (uniform) or a dict mapping the
        unordered pair ``(a, b)`` to a specific game count.
        """
        if isinstance(self.games_per_matchup, int):
            return self.games_per_matchup
        return self.games_per_matchup.get(frozenset((name1, name2)),
                                          self.games_per_matchup.get("default", 5))

    def run(self):
        names = list(self.agent_specs.keys() or self.agent_factories.keys())
        print(f"\nStarting Round Robin Tournament ({len(names)} agents, "
              f"{self.games_per_matchup} games per matchup)")

        if self.processes > 1:
            if not self.agent_specs:
                raise ValueError(
                    "processes > 1 requires picklable agent_specs "
                    "(lambdas are not picklable under spawn)."
                )
            self._run_parallel()
            return self.results

        # --- Sequential path (backward compatible, uses callable factories) ---
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name1, name2 = names[i], names[j]
                count = self._games_for(name1, name2)
                print(f"\n-> Matchup: {name1} vs {name2} ({count} games)")
                for k in range(count):
                    print(f"   Game {k+1}/{count}...")
                    if k % 2 == 0:
                        self.run_match(name1, self.agent_factories[name1],
                                       name2, self.agent_factories[name2])
                    else:
                        self.run_match(name2, self.agent_factories[name2],
                                       name1, self.agent_factories[name1])
        return self.results

    def _run_parallel(self):
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        engine_path = self.client.engine_path if self.client else None
        if engine_path is None:
            engine_path = _default_engine_path()

        from battle_agents.mcts_approximation.db import teams as teams_db

        names = list(self.agent_specs.keys())

        jobs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                count = self._games_for(n1, n2)
                for k in range(count):
                    if k % 2 == 0:
                        p1_name, p1_spec = n1, self.agent_specs[n1]
                        p2_name, p2_spec = n2, self.agent_specs[n2]
                    else:
                        p1_name, p1_spec = n2, self.agent_specs[n2]
                        p2_name, p2_spec = n1, self.agent_specs[n1]
                    jobs.append((
                        engine_path, self.formatid,
                        p1_name, p1_spec, p2_name, p2_spec,
                        teams_db.get_random_team("gen3ou"),
                        teams_db.get_random_team("gen3ou"),
                    ))

        print(f"   Dispatching {len(jobs)} games across {self.processes} processes...")
        with multiprocessing.Pool(self.processes) as pool:
            for res in pool.imap_unordered(run_benchmark_match, jobs):
                self.results.append(res)
