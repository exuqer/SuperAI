"""Shared access to the canonical database implementation."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import server.database as database


def get_db_path() -> Path:
    return database.get_db_path()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    with database.get_connection() as conn:
        yield conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    with get_connection() as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def init_db(schema_fn: Callable[[sqlite3.Connection], None] | None = None) -> None:
    if schema_fn is None:
        database.init_db()
        return
    with transaction() as conn:
        schema_fn(conn)
