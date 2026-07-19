"""Transactional persistence helpers for the V2.7 event graph."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

import server.database as database

from .graph_schema import ensure_graph_schema, reset_graph_schema


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def decode(value: Optional[str], default: Any = None) -> Any:
    if not value:
        return {} if default is None else default
    return json.loads(value)


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
        with database.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

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
