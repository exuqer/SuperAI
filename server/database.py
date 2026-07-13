"""SQLite database module for SuperAI"""
import sqlite3
import json
import time
import uuid
import math
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
        # The first prototype only had mass and coordinates. Keep existing local
        # databases usable while adding the observable learning metrics.
        existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(words)").fetchall()}
        for column, definition in {
            "frequency": "INTEGER NOT NULL DEFAULT 1",
            "halo": "REAL NOT NULL DEFAULT 0",
            "permeability": "REAL NOT NULL DEFAULT 0.5",
            "gravity": "REAL NOT NULL DEFAULT 1.0",
            "unique_neighbors": "INTEGER NOT NULL DEFAULT 0",
            "observations": "INTEGER NOT NULL DEFAULT 0",
            "distinct_sentences": "INTEGER NOT NULL DEFAULT 0",
            "distinct_contexts": "INTEGER NOT NULL DEFAULT 0",
            "confidence": "REAL NOT NULL DEFAULT 0.0",
        }.items():
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE words ADD COLUMN {column} {definition}")
        conn.execute("UPDATE words SET mass = CAST(frequency AS REAL)")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_words_session ON words(session_id)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS connections (
                session_id TEXT NOT NULL,
                word_a TEXT NOT NULL,
                word_b TEXT NOT NULL,
                strength REAL NOT NULL DEFAULT 1.0,
                contexts INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (session_id, word_a, word_b)
            )
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
        conn.execute("DELETE FROM connections WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM phrases WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM training_stats WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()


def add_words(session_id: str, words: List[str]) -> Dict[str, int]:
    """Count every occurrence and derive a slowly growing mass from it."""
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
                row_frequency = conn.execute(
                    "SELECT frequency FROM words WHERE word = ? AND session_id = ?", (word, session_id)
                ).fetchone()["frequency"]
                new_frequency = row_frequency + count
                new_mass = float(new_frequency)
                conn.execute(
                    "UPDATE words SET frequency = ?, mass = ?, updated_at = ? WHERE word = ? AND session_id = ?",
                    (new_frequency, new_mass, now, word, session_id)
                )
            else:
                # New word - mass equals its observed frequency.
                conn.execute(
                    "INSERT INTO words (word, session_id, mass, frequency, x, y, created_at, updated_at) VALUES (?, ?, ?, ?, 0.0, 0.0, ?, ?)",
                    (word, session_id, float(count), count, now, now)
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


def add_connections(session_id: str, sentences: List[List[str]]):
    """Strengthen each unique co-occurrence once per sentence."""
    now = time.time()
    with get_connection() as conn:
        for words in sentences:
            unique = list(dict.fromkeys(words))
            for i, word_a in enumerate(sorted(unique)):
                for word_b in sorted(unique)[i + 1:]:
                    conn.execute(
                        """INSERT INTO connections (session_id, word_a, word_b, strength, contexts)
                           VALUES (?, ?, ?, 1.0, 1)
                           ON CONFLICT(session_id, word_a, word_b) DO UPDATE SET
                           strength = strength + 1.0, contexts = contexts + 1""",
                        (session_id, word_a, word_b),
                    )
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        conn.commit()


def get_connections(session_id: str) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT word_a, word_b, strength, contexts FROM connections WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def refresh_word_metrics(session_id: str):
    """Derive halo, permeability and gravity from simple neighborhood observations."""
    with get_connection() as conn:
        words = conn.execute("SELECT word, frequency FROM words WHERE session_id = ?", (session_id,)).fetchall()
        connections = conn.execute(
            "SELECT word_a, word_b, strength, contexts FROM connections WHERE session_id = ?", (session_id,)
        ).fetchall()
        phrases = conn.execute(
            "SELECT words, count FROM phrases WHERE session_id = ?", (session_id,)
        ).fetchall()
        neighbors = {row["word"]: [] for row in words}
        observations = {row["word"]: 0 for row in words}
        distinct_sentences = {row["word"]: 0 for row in words}
        for edge in connections:
            neighbors.setdefault(edge["word_a"], []).append((edge["strength"], edge["contexts"]))
            neighbors.setdefault(edge["word_b"], []).append((edge["strength"], edge["contexts"]))
        for phrase in phrases:
            phrase_words = set(json.loads(phrase["words"]))
            for word in phrase_words:
                observations[word] = observations.get(word, 0) + phrase["count"]
                distinct_sentences[word] = distinct_sentences.get(word, 0) + 1
        for row in words:
            strengths = neighbors.get(row["word"], [])
            unique_neighbors = len(strengths)
            word_observations = observations.get(row["word"], 0)
            word_contexts = distinct_sentences.get(row["word"], 0)
            confidence = min(1.0, word_observations / 10.0)
            halo_signal = min(1.0, word_contexts / 8.0)
            halo = float(word_contexts)
            # Diversity is independent from frequency and connection strength.
            computed_permeability = 0.1 + 0.8 * halo_signal
            permeability = 0.35 * (1.0 - confidence) + computed_permeability * confidence
            best = sorted((value for value, _ in strengths), reverse=True)[:3]
            average_strength = sum(best) / len(best) if best else 0.0
            computed_gravity = float(row["frequency"]) * average_strength
            gravity = 1.0 * (1.0 - confidence) + computed_gravity * confidence
            conn.execute(
                "UPDATE words SET halo = ?, permeability = ?, gravity = ?, unique_neighbors = ?, observations = ?, distinct_sentences = ?, distinct_contexts = ?, confidence = ? WHERE word = ? AND session_id = ?",
                (halo, permeability, gravity, unique_neighbors, word_observations, word_contexts, word_contexts, confidence, row["word"], session_id),
            )
        conn.commit()


def get_words(session_id: str) -> List[Dict]:
    """Get all words for a session."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT word, mass, frequency, halo, permeability, gravity, unique_neighbors, observations, distinct_sentences, distinct_contexts, confidence, x, y FROM words WHERE session_id = ? ORDER BY mass DESC",
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
        
        edges = conn.execute("SELECT COUNT(*) as c FROM connections WHERE session_id = ?", (session_id,)).fetchone()["c"]
        
        return {
            "tokens": tokens,
            "total_tokens": int(total_mass),
            "phrases": phrases,
            "edges": edges,
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
        conn.execute("DELETE FROM connections WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM phrases WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM training_stats WHERE session_id = ?", (session_id,))
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        conn.commit()
