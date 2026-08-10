"""
tests/conftest.py
Shared fixtures used across all test modules.
"""
import pytest


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Override SESSION_DB_PATH with a fresh temp file for the duration of the test."""
    import config
    db_path = str(tmp_path / "test_session.db")
    monkeypatch.setattr(config, "SESSION_DB_PATH", db_path)
    import tools.session.session_store as ss
    ss._init_db()
    yield db_path


@pytest.fixture
def test_client(temp_db):
    from fastapi.testclient import TestClient
    import api
    client = TestClient(api.app, raise_server_exceptions=False)
    return client
