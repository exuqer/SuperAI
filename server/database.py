"""Canonical SQLite storage for SuperAI V3.0."""

from __future__ import annotations

import atexit
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Iterator

from server.core.settings import settings


DB_PATH = Path(settings.database_path)
_LOCAL = threading.local()


def _record_database_work(started: float, *, execute: bool) -> None:
    _LOCAL.sql_ms = float(getattr(_LOCAL, "sql_ms", 0.0)) + (
        perf_counter() - started
    ) * 1000.0
    if execute:
        _LOCAL.execute_count = int(
            getattr(_LOCAL, "execute_count", 0)
        ) + 1


class MeasuredConnection(sqlite3.Connection):
    """sqlite3 connection that exposes low-cost process diagnostics."""

    def execute(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        started = perf_counter()
        try:
            return super().execute(*args, **kwargs)
        finally:
            _record_database_work(started, execute=True)

    def executemany(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        started = perf_counter()
        try:
            return super().executemany(*args, **kwargs)
        finally:
            _record_database_work(started, execute=True)

    def executescript(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        started = perf_counter()
        try:
            return super().executescript(*args, **kwargs)
        finally:
            _record_database_work(started, execute=True)

    def commit(self) -> None:
        started = perf_counter()
        try:
            super().commit()
        finally:
            _record_database_work(started, execute=False)

    def rollback(self) -> None:
        started = perf_counter()
        try:
            super().rollback()
        finally:
            _record_database_work(started, execute=False)


def metrics_snapshot() -> tuple[float, int]:
    """Return cumulative SQL wall time and execute count for this worker."""
    return (
        float(getattr(_LOCAL, "sql_ms", 0.0)),
        int(getattr(_LOCAL, "execute_count", 0)),
    )


def get_db_path() -> Path:
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _configure_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, factory=MeasuredConnection)
    conn.row_factory = sqlite3.Row
    # WAL keeps source-level transactions atomic while avoiding a full
    # rollback-journal sync for every short-lived repository transaction.
    # NORMAL still syncs WAL checkpoints and is SQLite's intended balance for
    # WAL-backed application databases.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    # The default 1000-page auto-checkpoint repeatedly stalls the compact
    # source workload.  A bounded 4096-page WAL amortises those checkpoints
    # while keeping automatic recovery and a finite on-disk journal.
    conn.execute("PRAGMA wal_autocheckpoint = 4096")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def close_current_connection() -> None:
    """Release the reusable connection owned by the current worker thread."""
    conn = getattr(_LOCAL, "connection", None)
    if conn is not None:
        conn.close()
    _LOCAL.connection = None
    _LOCAL.path = None
    _LOCAL.identity = None
    _LOCAL.depth = 0


def _database_identity(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Reuse one SQLite handle per worker without sharing active contexts.

    Transactions still commit or roll back at repository boundaries.  A
    nested context receives a temporary handle, while the common sequential
    path avoids reconnecting and repeatedly closing the final WAL reader.
    """
    path = get_db_path()
    depth = int(getattr(_LOCAL, "depth", 0))
    current = getattr(_LOCAL, "connection", None)
    current_path = getattr(_LOCAL, "path", None)
    identity = _database_identity(path)
    stale = (
        current is not None
        and current_path == path
        and (
            identity is None
            or getattr(_LOCAL, "identity", None) != identity
        )
    )
    if current is not None and (current_path != path or stale):
        close_current_connection()
        current = None
        depth = 0
    reusable = depth == 0
    conn = current if reusable else None
    if conn is None:
        conn = _configure_connection(path)
        if reusable:
            _LOCAL.connection = conn
            _LOCAL.path = path
            _LOCAL.identity = _database_identity(path)
    _LOCAL.depth = depth + 1
    try:
        yield conn
    finally:
        # Closing a sqlite3 connection used to roll back an unfinished
        # transaction.  Preserve that context-manager contract when the
        # handle itself stays alive for reuse.
        if conn.in_transaction:
            conn.rollback()
        _LOCAL.depth = max(0, int(getattr(_LOCAL, "depth", 1)) - 1)
        if not reusable:
            conn.close()


atexit.register(close_current_connection)


def init_db() -> None:
    from server.v2.graph_schema import ensure_graph_schema

    with get_connection() as conn:
        ensure_graph_schema(conn)
        conn.commit()
    close_current_connection()
