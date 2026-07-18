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
            reset_graph_schema(conn)
            conn.commit()

    def graph_meta(self) -> Dict[str, str]:
        with self.transaction() as conn:
            return {
                str(row["key"]): str(row["value"])
                for row in conn.execute(
                    "SELECT key,value FROM graph_meta ORDER BY key"
                ).fetchall()
            }
