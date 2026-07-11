"""Guard test – verify the DB package imports correctly and loads data after the folder restructure."""

from battle_agents.mcts_approximation.db.python.database import db


def test_db_loads_moves():
    assert db.get_num_moves() > 0


def test_db_loads_species():
    assert db.get_num_species() > 0


def test_db_loads_items():
    assert db.get_num_items() > 0


def test_db_loads_abilities():
    assert db.get_num_abilities() > 0
