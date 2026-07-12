"""SQLite database module for SuperAI"""
import sqlite3
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


DB_PATH = Path(".superai/state.sqlite")


def get_db_path() -> Path:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


@contextmanager
def get_connection():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initialize database tables."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT 'Session',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS words (
                word TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                mass REAL NOT NULL DEFAULT 1.0,
                x REAL NOT NULL DEFAULT 0.0,
                y REAL NOT NULL DEFAULT 0.0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_words_session ON words(session_id)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS phrases (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                words TEXT NOT NULL,  -- JSON array of words
                count INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_phrases_session ON phrases(session_id)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                tokens INTEGER NOT NULL,
                edges INTEGER NOT NULL,
                phrases INTEGER NOT NULL,
                loss REAL,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_stats_session ON training_stats(session_id)
        """)
        conn.commit()


def create_session(name: str = "Session") -> str:
    """Create a new training session."""
    session_id = str(uuid.uuid4())
    now = time.time()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, name, now, now)
        )
        conn.commit()
    return session_id


def get_session(session_id: str) -> Optional[Dict]:
    """Get session info."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None


def list_sessions() -> List[Dict]:
    """List all sessions."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
        return [dict(row) for row in rows]


def delete_session(session_id: str):
    """Delete a session and all associated data."""
    with get_connection() as conn:
        conn.execute("DELETE FROM words WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM phrases WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM training_stats WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()


def add_words(session_id: str, words: List[str]) -> Dict[str, int]:
    """Add words to session, increment mass for existing words."""
    now = time.time()
    counts = {}
    with get_connection() as conn:
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        
        for word, count in counts.items():
            # Check if word exists
            row = conn.execute(
                "SELECT mass FROM words WHERE word = ? AND session_id = ?",
                (word, session_id)
            ).fetchone()
            
            if row:
                # Existing word - increment mass by 0.1 exactly once per request
                new_mass = row["mass"] + 0.1
                conn.execute(
                    "UPDATE words SET mass = ?, updated_at = ? WHERE word = ? AND session_id = ?",
                    (new_mass, now, word, session_id)
                )
            else:
                # New word - mass 1.0, random position
                conn.execute(
                    "INSERT INTO words (word, session_id, mass, x, y, created_at, updated_at) VALUES (?, ?, 1.0, 0.0, 0.0, ?, ?)",
                    (word, session_id, now, now)
                )
        
        conn.commit()
    return counts


def add_phrase(session_id: str, words: List[str]):
    """Add a phrase (sentence) to session."""
    if len(words) < 2:
        return
    
    # Create a key from sorted words to detect duplicates
    phrase_key = "|".join(sorted(words))
    phrase_id = f"phrase_{hash(phrase_key) & 0x7FFFFFFF:x}"
    now = time.time()
    
    with get_connection() as conn:
        row = conn.execute(
            "SELECT count FROM phrases WHERE id = ? AND session_id = ?",
            (phrase_id, session_id)
        ).fetchone()
        
        if row:
            conn.execute(
                "UPDATE phrases SET count = count + 1, updated_at = ? WHERE id = ? AND session_id = ?",
                (now, phrase_id, session_id)
            )
        else:
            conn.execute(
                "INSERT INTO phrases (id, session_id, words, count, created_at, updated_at) VALUES (?, ?, ?, 1, ?, ?)",
                (phrase_id, session_id, json.dumps(words), now, now)
            )
        conn.commit()


def get_words(session_id: str) -> List[Dict]:
    """Get all words for a session."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT word, mass, x, y FROM words WHERE session_id = ? ORDER BY mass DESC",
            (session_id,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_phrases(session_id: str) -> List[Dict]:
    """Get all phrases for a session."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT words, count FROM phrases WHERE session_id = ? ORDER BY count DESC",
            (session_id,)
        ).fetchall()
        return [{"words": json.loads(row["words"]), "count": row["count"]} for row in rows]


def update_words_positions(session_id: str, positions: Dict[str, tuple]):
    """Update word positions after physics simulation."""
    now = time.time()
    with get_connection() as conn:
        for word, (x, y) in positions.items():
            conn.execute(
                "UPDATE words SET x = ?, y = ?, updated_at = ? WHERE word = ? AND session_id = ?",
                (x, y, now, word, session_id)
            )
        conn.commit()


def get_session_stats(session_id: str) -> Dict[str, int]:
    """Get session statistics."""
    with get_connection() as conn:
        tokens = conn.execute(
            "SELECT COUNT(*) as c FROM words WHERE session_id = ?", (session_id,)
        ).fetchone()["c"]
        total_mass = conn.execute(
            "SELECT SUM(mass) as s FROM words WHERE session_id = ?", (session_id,)
        ).fetchone()["s"] or 0
        phrases = conn.execute(
            "SELECT COUNT(*) as c FROM phrases WHERE session_id = ?", (session_id,)
        ).fetchone()["c"]
        
        # Count edges (unique word pairs in phrases)
        phrase_rows = conn.execute(
            "SELECT words FROM phrases WHERE session_id = ?", (session_id,)
        ).fetchall()
        edges_set = set()
        for row in phrase_rows:
            words = json.loads(row["words"])
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    pair = tuple(sorted([words[i], words[j]]))
                    edges_set.add(pair)
        
        return {
            "tokens": tokens,
            "total_tokens": int(total_mass),
            "phrases": phrases,
            "edges": len(edges_set),
        }


def add_training_stats(
    session_id: str,
    epoch: int,
    tokens: int,
    edges: int,
    phrases: int,
    loss: Optional[float] = None
):
    """Record training statistics."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO training_stats (session_id, epoch, tokens, edges, phrases, loss, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, epoch, tokens, edges, phrases, loss, time.time())
        )
        conn.commit()


def reset_session(session_id: str):
    """Reset (clear) all words and phrases for a session."""
    now = time.time()
    with get_connection() as conn:
        conn.execute("DELETE FROM words WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM phrases WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM training_stats WHERE session_id = ?", (session_id,))
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        conn.commit()