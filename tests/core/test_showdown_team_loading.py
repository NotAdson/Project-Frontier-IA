import os
import unittest
from battle_agents.mcts_approximation.db import teams


class TestShowdownTeamLoading(unittest.TestCase):
    def test_load_local_gen3ou_team_pack(self):
        formatid = 'gen3ou'
        local_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'teams', f'{formatid}.txt')
        local_path = os.path.abspath(local_path)
        self.assertTrue(os.path.isfile(local_path), f"Expected local team file to exist: {local_path}")

        packs = teams.load_team_packs(formatid)
        self.assertIsInstance(packs, list)
        self.assertGreater(len(packs), 0, "Expected at least one team pack from local data/teams/gen3ou.txt")
        self.assertTrue(all(isinstance(pack, str) for pack in packs))

    def test_get_random_team_returns_pack_string(self):
        pack = teams.get_random_team('gen3ou')
        self.assertIsInstance(pack, str)
        self.assertNotEqual(pack.strip(), '')
