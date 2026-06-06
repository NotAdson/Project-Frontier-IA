import os
from unittest.mock import MagicMock, patch
from core.client.base_client import BaseClient
from core.client.showdown_client import ShowdownClient

def test_base_client_is_abstract():
    # BaseClient should not be instantiated directly
    try:
        client = BaseClient()
        assert False, "BaseClient should be abstract and not instantiable"
    except TypeError:
        assert True

def test_showdown_client_inherits_base_client():
    assert issubclass(ShowdownClient, BaseClient)

