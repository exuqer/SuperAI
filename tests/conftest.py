from pathlib import Path

import pytest

import server.database as database


@pytest.fixture(autouse=True)
def isolated_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "state.sqlite"
    monkeypatch.setattr(database, "DB_PATH", path)
    database.init_db()
    yield path
