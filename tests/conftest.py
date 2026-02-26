import pytest


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary database path."""
    return tmp_path / "test.db"
