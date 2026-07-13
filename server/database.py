"""Persistent storage for the relation-free concept field."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DB_PATH = Path(".superai/state.sqlite")
SCHEMA_VERSION = 2


def get_db_path() -> Path:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


@contextmanager
def get_connection():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version != SCHEMA_VERSION:
            for table in ("sessions", "words", "connections", "phrases", "training_stats", "concepts"):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(
                """
                CREATE TABLE concepts (
                    id INTEGER PRIMARY KEY,
                    token TEXT NOT NULL UNIQUE,
                    position TEXT NOT NULL,
                    mass REAL NOT NULL CHECK (mass > 0)
                )
                """
            )
            conn.execute("CREATE INDEX idx_concepts_mass ON concepts(mass DESC)")
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()


def _unique_tokens(tokens: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(token for token in tokens if token))


def ensure_concepts(tokens: Sequence[str], initial_position: Tuple[float, float]) -> None:
    unique_tokens = _unique_tokens(tokens)
    if not unique_tokens:
        return

    encoded_position = json.dumps(list(initial_position), separators=(",", ":"))
    with get_connection() as conn:
        for token in unique_tokens:
            row = conn.execute("SELECT id FROM concepts WHERE token = ?", (token,)).fetchone()
            if row:
                conn.execute("UPDATE concepts SET mass = mass + 0.1 WHERE id = ?", (row["id"],))
            else:
                conn.execute(
                    "INSERT INTO concepts (token, position, mass) VALUES (?, ?, 1.0)",
                    (token, encoded_position),
                )
        conn.commit()


def get_concepts() -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, token, position, mass FROM concepts ORDER BY mass DESC, id ASC"
        ).fetchall()
    concepts = []
    for row in rows:
        position = json.loads(row["position"])
        concepts.append(
            {
                "id": int(row["id"]),
                "token": row["token"],
                "position": [float(value) for value in position],
                "mass": float(row["mass"]),
            }
        )
    return concepts


def update_concepts(concepts: Iterable[Tuple[int, Sequence[float], float]]) -> None:
    with get_connection() as conn:
        for concept_id, position, mass in concepts:
            conn.execute(
                "UPDATE concepts SET position = ?, mass = ? WHERE id = ?",
                (json.dumps(list(position), separators=(",", ":")), float(mass), int(concept_id)),
            )
        conn.commit()


def get_stats() -> Dict[str, float | int]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS concepts, COALESCE(SUM(mass), 0) AS total_mass FROM concepts"
        ).fetchone()
    concepts = int(row["concepts"])
    total_mass = round(float(row["total_mass"]), 3)
    return {"concepts": concepts, "total_mass": total_mass, "tokens": concepts}


def reset_space() -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM concepts")
        conn.commit()
