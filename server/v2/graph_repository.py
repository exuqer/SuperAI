"""Transactional persistence helpers for the SuperAI V3.0 evidence graph."""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, Iterator, Optional

import server.database as database

from .graph_schema import ensure_graph_schema, reset_graph_schema


_TRANSACTION_LOCAL = threading.local()
_SERIALIZATION_LOCAL = threading.local()


def serialization_snapshot() -> float:
    return float(getattr(_SERIALIZATION_LOCAL, "elapsed_ms", 0.0))


def _record_serialization(started: float) -> None:
    _SERIALIZATION_LOCAL.elapsed_ms = float(
        getattr(_SERIALIZATION_LOCAL, "elapsed_ms", 0.0)
    ) + (perf_counter() - started) * 1000.0


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode(value: Any) -> str:
    started = perf_counter()
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    finally:
        _record_serialization(started)


def decode(value: Optional[str], default: Any = None) -> Any:
    if not value:
        return {} if default is None else default
    started = perf_counter()
    try:
        return json.loads(value)
    finally:
        _record_serialization(started)


def stable_id(prefix: str, *values: object, size: int = 20) -> str:
    payload = "\x1f".join(str(value) for value in values)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:size]
    return f"{prefix}-{digest}"


def content_hash(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class GraphRepository:
    def __init__(self) -> None:
        self.ensure_schema()

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        path = database.get_db_path()
        active = getattr(_TRANSACTION_LOCAL, "connection", None)
        if (
            active is not None
            and getattr(_TRANSACTION_LOCAL, "path", None) == path
        ):
            _TRANSACTION_LOCAL.depth = int(
                getattr(_TRANSACTION_LOCAL, "depth", 1)
            ) + 1
            try:
                yield active
            finally:
                _TRANSACTION_LOCAL.depth -= 1
            return
        with database.get_connection() as conn:
            _TRANSACTION_LOCAL.connection = conn
            _TRANSACTION_LOCAL.path = path
            _TRANSACTION_LOCAL.depth = 1
            try:
                yield conn
                # sqlite3 does not open a transaction for plain SELECTs.
                # Avoid issuing a redundant commit for the many read-only
                # graph/query contexts.
                if conn.in_transaction:
                    conn.commit()
            except Exception:
                if conn.in_transaction:
                    conn.rollback()
                raise
            finally:
                _TRANSACTION_LOCAL.connection = None
                _TRANSACTION_LOCAL.path = None
                _TRANSACTION_LOCAL.depth = 0

    def ensure_schema(self) -> None:
        with database.get_connection() as conn:
            ensure_graph_schema(conn)
            conn.commit()

    def reset(self) -> None:
        """Remove all current model state; this operation is intentionally final."""
        with database.get_connection() as conn:
            previous = {
                str(row["key"]): int(row["value"])
                for row in conn.execute(
                    """SELECT key,value FROM graph_meta
                       WHERE key IN ('projection_revision','transition_revision')"""
                ).fetchall()
            }
            reset_graph_schema(conn)
            for name in ("projection_revision", "transition_revision"):
                conn.execute(
                    "UPDATE graph_meta SET value=? WHERE key=?",
                    (str(previous.get(name, 0) + 1), name),
                )
            conn.commit()

    @staticmethod
    def bump_revisions(conn: Any, *, projection: bool = True, transition: bool = True) -> Dict[str, int]:
        """Invalidate process-local derived indices after committed geometry changes."""
        names = []
        if projection:
            names.append("projection_revision")
        if transition:
            names.append("transition_revision")
        for name in names:
            conn.execute(
                """INSERT INTO graph_meta(key,value) VALUES(?, '1')
                   ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER)+1""",
                (name,),
            )
        if not names:
            return {}
        rows = conn.execute(
            "SELECT key,value FROM graph_meta WHERE key IN ({})".format(
                ",".join("?" for _ in names)
            ),
            names,
        ).fetchall()
        return {str(row["key"]): int(row["value"]) for row in rows}

    def graph_meta(self) -> Dict[str, str]:
        with self.transaction() as conn:
            return {
                str(row["key"]): str(row["value"])
                for row in conn.execute(
                    "SELECT key,value FROM graph_meta ORDER BY key"
                ).fetchall()
            }
