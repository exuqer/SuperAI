"""Persistent storage for the recursive nebula system."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


DB_PATH = Path(".superai/state.sqlite")
SCHEMA_VERSION = 7


def get_db_path() -> Path:
    path = Path(DB_PATH) if isinstance(DB_PATH, str) else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Initialize the legacy schema without destructively replacing stored data."""
    with get_connection() as conn:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])

        _ensure_base_tables(conn)
        _add_lexeme_tables(conn)
        _add_lexeme_layer(conn)
        _add_chat_tables(conn)
        if version < SCHEMA_VERSION:
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()


def _add_chat_tables(conn: sqlite3.Connection) -> None:
    """Persistent chat sessions, swarm turns, events and working-memory cells."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            turn_index INTEGER NOT NULL DEFAULT 0,
            max_cells INTEGER NOT NULL DEFAULT 24
        );
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user','assistant')),
            text TEXT NOT NULL,
            turn_index INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, turn_index);
        CREATE TABLE IF NOT EXISTS swarm_turns (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            iteration INTEGER NOT NULL DEFAULT 0,
            goal_json TEXT NOT NULL DEFAULT '{}',
            metrics_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_swarm_turns_session ON swarm_turns(session_id, created_at);
        CREATE TABLE IF NOT EXISTS swarm_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            UNIQUE(turn_id, sequence),
            FOREIGN KEY (turn_id) REFERENCES swarm_turns(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS hive_cells (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            label TEXT NOT NULL,
            composition_json TEXT NOT NULL DEFAULT '{}',
            x REAL NOT NULL DEFAULT 0.0,
            y REAL NOT NULL DEFAULT 0.0,
            gravity REAL NOT NULL DEFAULT 0.0,
            visits INTEGER NOT NULL DEFAULT 0,
            source_id TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_hive_cells_session ON hive_cells(session_id, gravity DESC);
        """
    )


def _ensure_base_tables(conn: sqlite3.Connection) -> None:
    """Ensure all base tables exist (idempotent)."""
    tables_sql = [
        # LAYERS
        """CREATE TABLE IF NOT EXISTS layers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            order_index INTEGER NOT NULL,
            scale REAL NOT NULL DEFAULT 1.0,
            layer_type TEXT NOT NULL DEFAULT '',
            config_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_layers_order ON layers(order_index)",
        
        # CLOUDS
        """CREATE TABLE IF NOT EXISTS clouds (
            id INTEGER PRIMARY KEY,
            layer_id INTEGER NOT NULL,
            cloud_type TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            mass REAL NOT NULL DEFAULT 1.0 CHECK (mass > 0),
            density REAL NOT NULL DEFAULT 1.0 CHECK (density >= 0),
            radius REAL NOT NULL DEFAULT 10.0 CHECK (radius >= 0),
            stability REAL NOT NULL DEFAULT 0.0 CHECK (stability >= 0 AND stability <= 1),
            activation REAL NOT NULL DEFAULT 0.0 CHECK (activation >= 0),
            observation_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_activated_at TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (layer_id) REFERENCES layers(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_clouds_layer ON clouds(layer_id)",
        "CREATE INDEX IF NOT EXISTS idx_clouds_type ON clouds(cloud_type)",
        "CREATE INDEX IF NOT EXISTS idx_clouds_name ON clouds(canonical_name)",
        "CREATE INDEX IF NOT EXISTS idx_clouds_stability ON clouds(stability DESC)",
        "CREATE INDEX IF NOT EXISTS idx_clouds_activated ON clouds(last_activated_at DESC)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_clouds_layer_name ON clouds(layer_id, canonical_name)",
        
        # SPACES
        """CREATE TABLE IF NOT EXISTS spaces (
            id INTEGER PRIMARY KEY,
            host_cloud_id INTEGER NOT NULL,
            layer_id INTEGER NOT NULL,
            mode TEXT NOT NULL DEFAULT 'structural',
            coordinate_dimensions INTEGER NOT NULL DEFAULT 2,
            scale REAL NOT NULL DEFAULT 1.0,
            config_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (host_cloud_id) REFERENCES clouds(id),
            FOREIGN KEY (layer_id) REFERENCES layers(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_spaces_host ON spaces(host_cloud_id)",
        "CREATE INDEX IF NOT EXISTS idx_spaces_layer ON spaces(layer_id)",
        "CREATE INDEX IF NOT EXISTS idx_spaces_mode ON spaces(mode)",
        
        # CLOUD_PLACEMENTS
        """CREATE TABLE IF NOT EXISTS cloud_placements (
            id INTEGER PRIMARY KEY,
            space_id INTEGER NOT NULL,
            cloud_id INTEGER NOT NULL,
            x REAL NOT NULL DEFAULT 0.0,
            y REAL NOT NULL DEFAULT 0.0,
            z REAL NOT NULL DEFAULT 0.0,
            radius REAL NOT NULL DEFAULT 10.0 CHECK (radius >= 0),
            density REAL NOT NULL DEFAULT 1.0 CHECK (density >= 0),
            mass REAL NOT NULL DEFAULT 1.0 CHECK (mass > 0),
            activation REAL NOT NULL DEFAULT 0.0 CHECK (activation >= 0),
            velocity_x REAL NOT NULL DEFAULT 0.0,
            velocity_y REAL NOT NULL DEFAULT 0.0,
            velocity_z REAL NOT NULL DEFAULT 0.0,
            fixed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (space_id) REFERENCES spaces(id),
            FOREIGN KEY (cloud_id) REFERENCES clouds(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_placements_space ON cloud_placements(space_id)",
        "CREATE INDEX IF NOT EXISTS idx_placements_cloud ON cloud_placements(cloud_id)",
        "CREATE INDEX IF NOT EXISTS idx_placements_density ON cloud_placements(density DESC)",
        "CREATE INDEX IF NOT EXISTS idx_placements_spatial ON cloud_placements(x, y)",
        
        # STRUCTURAL_COMPONENTS
        """CREATE TABLE IF NOT EXISTS structural_components (
            id INTEGER PRIMARY KEY,
            parent_cloud_id INTEGER NOT NULL,
            child_cloud_id INTEGER NOT NULL,
            child_placement_id INTEGER,
            position_index INTEGER NOT NULL DEFAULT 0,
            phase REAL NOT NULL DEFAULT 0.0,
            weight REAL NOT NULL DEFAULT 1.0,
            role TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (parent_cloud_id) REFERENCES clouds(id),
            FOREIGN KEY (child_cloud_id) REFERENCES clouds(id),
            FOREIGN KEY (child_placement_id) REFERENCES cloud_placements(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_struct_parent ON structural_components(parent_cloud_id)",
        "CREATE INDEX IF NOT EXISTS idx_struct_child ON structural_components(child_cloud_id)",
        "CREATE INDEX IF NOT EXISTS idx_struct_order ON structural_components(parent_cloud_id, position_index)",
        
        # ACTIVATION_EVENTS
        """CREATE TABLE IF NOT EXISTS activation_events (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            cloud_id INTEGER NOT NULL,
            placement_id INTEGER,
            layer_id INTEGER NOT NULL,
            activation_value REAL NOT NULL,
            sequence_position INTEGER NOT NULL DEFAULT 0,
            timestamp TEXT NOT NULL,
            context_window_id TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (cloud_id) REFERENCES clouds(id),
            FOREIGN KEY (placement_id) REFERENCES cloud_placements(id),
            FOREIGN KEY (layer_id) REFERENCES layers(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_activation_session ON activation_events(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_activation_cloud ON activation_events(cloud_id)",
        "CREATE INDEX IF NOT EXISTS idx_activation_layer ON activation_events(layer_id)",
        "CREATE INDEX IF NOT EXISTS idx_activation_time ON activation_events(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_activation_context ON activation_events(context_window_id)",
        
        # COACTIVATION_STATS
        """CREATE TABLE IF NOT EXISTS coactivation_stats (
            cloud_a_id INTEGER NOT NULL,
            cloud_b_id INTEGER NOT NULL,
            layer_id INTEGER NOT NULL,
            coactivation_count INTEGER NOT NULL DEFAULT 0,
            weighted_score REAL NOT NULL DEFAULT 0.0,
            average_sequence_distance REAL NOT NULL DEFAULT 0.0,
            last_updated_at TEXT NOT NULL,
            PRIMARY KEY (cloud_a_id, cloud_b_id, layer_id),
            FOREIGN KEY (cloud_a_id) REFERENCES clouds(id),
            FOREIGN KEY (cloud_b_id) REFERENCES clouds(id),
            FOREIGN KEY (layer_id) REFERENCES layers(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_coact_a ON coactivation_stats(cloud_a_id)",
        "CREATE INDEX IF NOT EXISTS idx_coact_b ON coactivation_stats(cloud_b_id)",
        "CREATE INDEX IF NOT EXISTS idx_coact_layer ON coactivation_stats(layer_id)",
        "CREATE INDEX IF NOT EXISTS idx_coact_score ON coactivation_stats(weighted_score DESC)",
        
        # CONDENSATION_CANDIDATES
        """CREATE TABLE IF NOT EXISTS condensation_candidates (
            id INTEGER PRIMARY KEY,
            source_layer_id INTEGER NOT NULL,
            target_layer_id INTEGER NOT NULL,
            signature_hash TEXT NOT NULL,
            observations INTEGER NOT NULL DEFAULT 0,
            stability REAL NOT NULL DEFAULT 0.0,
            sequence_sensitive INTEGER NOT NULL DEFAULT 1,
            proposed_cloud_id INTEGER,
            status TEXT NOT NULL DEFAULT 'pending',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (source_layer_id) REFERENCES layers(id),
            FOREIGN KEY (target_layer_id) REFERENCES layers(id),
            FOREIGN KEY (proposed_cloud_id) REFERENCES clouds(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_cond_source ON condensation_candidates(source_layer_id)",
        "CREATE INDEX IF NOT EXISTS idx_cond_target ON condensation_candidates(target_layer_id)",
        "CREATE INDEX IF NOT EXISTS idx_cond_hash ON condensation_candidates(signature_hash)",
        "CREATE INDEX IF NOT EXISTS idx_cond_status ON condensation_candidates(status)",
    ]

    for sql in tables_sql:
        conn.execute(sql)


def _add_lexeme_tables(conn: sqlite3.Connection) -> None:
    """Add new tables for lexeme layer and semantic features."""
    now = "2024-01-01T00:00:00"

    new_tables = [
        # LEXEMES
        """CREATE TABLE IF NOT EXISTS lexemes (
            id INTEGER PRIMARY KEY,
            canonical_form TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT 'ru',
            pos_tag TEXT,
            features_json TEXT NOT NULL DEFAULT '{}',
            frequency INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_lexemes_form_lang ON lexemes(canonical_form, language)",
        "CREATE INDEX IF NOT EXISTS idx_lexemes_freq ON lexemes(frequency DESC)",
        
        # WORD_FORM_TO_LEXEME
        """CREATE TABLE IF NOT EXISTS word_form_to_lexeme (
            word_form_cloud_id INTEGER NOT NULL,
            lexeme_id INTEGER NOT NULL,
            is_canonical INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            PRIMARY KEY (word_form_cloud_id, lexeme_id),
            FOREIGN KEY (word_form_cloud_id) REFERENCES clouds(id),
            FOREIGN KEY (lexeme_id) REFERENCES lexemes(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_wf2lex_lexeme ON word_form_to_lexeme(lexeme_id)",
        
        # CONTEXT_VECTORS
        """CREATE TABLE IF NOT EXISTS context_vectors (
            lexeme_id INTEGER NOT NULL,
            context_lexeme_id INTEGER NOT NULL,
            direction INTEGER NOT NULL DEFAULT 1,
            weight REAL NOT NULL DEFAULT 0,
            raw_weight REAL NOT NULL DEFAULT 0,
            count INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (lexeme_id, context_lexeme_id, direction),
            FOREIGN KEY (lexeme_id) REFERENCES lexemes(id),
            FOREIGN KEY (context_lexeme_id) REFERENCES lexemes(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_ctx_vec_lexeme ON context_vectors(lexeme_id)",
        "CREATE INDEX IF NOT EXISTS idx_ctx_vec_context ON context_vectors(context_lexeme_id)",
        "CREATE INDEX IF NOT EXISTS idx_ctx_vec_weight ON context_vectors(weight DESC)",
        
        # CONCEPT_CENTROIDS
        """CREATE TABLE IF NOT EXISTS concept_centroids (
            id INTEGER PRIMARY KEY,
            concept_cloud_id INTEGER NOT NULL,
            centroid_vector_json TEXT NOT NULL,
            member_lexeme_ids_json TEXT NOT NULL,
            stability REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (concept_cloud_id) REFERENCES clouds(id)
        )""",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_centroid_concept ON concept_centroids(concept_cloud_id)",
        
        # LEXEME_CONCEPT_MEMBERSHIP
        """CREATE TABLE IF NOT EXISTS lexeme_concept_membership (
            lexeme_id INTEGER NOT NULL,
            concept_cloud_id INTEGER NOT NULL,
            membership REAL NOT NULL,
            centrality REAL NOT NULL DEFAULT 0.0,
            context_coverage REAL NOT NULL DEFAULT 0.0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (lexeme_id, concept_cloud_id),
            FOREIGN KEY (lexeme_id) REFERENCES lexemes(id),
            FOREIGN KEY (concept_cloud_id) REFERENCES clouds(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_lcm_concept ON lexeme_concept_membership(concept_cloud_id)",
        "CREATE INDEX IF NOT EXISTS idx_lcm_membership ON lexeme_concept_membership(membership DESC)",
        
        # SCENES
        """CREATE TABLE IF NOT EXISTS scenes (
            id INTEGER PRIMARY KEY,
            scene_cloud_id INTEGER NOT NULL,
            sentence_text TEXT NOT NULL,
            word_form_cloud_ids_json TEXT NOT NULL,
            lexeme_ids_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (scene_cloud_id) REFERENCES clouds(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_scenes_cloud ON scenes(scene_cloud_id)",
        
        # SCENE_SIMILARITY
        """CREATE TABLE IF NOT EXISTS scene_similarity (
            scene_a_id INTEGER NOT NULL,
            scene_b_id INTEGER NOT NULL,
            similarity REAL NOT NULL,
            weight REAL NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (scene_a_id, scene_b_id),
            FOREIGN KEY (scene_a_id) REFERENCES scenes(id),
            FOREIGN KEY (scene_b_id) REFERENCES scenes(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_scene_sim_a ON scene_similarity(scene_a_id)",
        "CREATE INDEX IF NOT EXISTS idx_scene_sim_b ON scene_similarity(scene_b_id)",
        "CREATE INDEX IF NOT EXISTS idx_scene_sim_score ON scene_similarity(similarity DESC)",
        
        # SEMANTIC_OVERLAYS
        """CREATE TABLE IF NOT EXISTS semantic_overlays (
            id INTEGER PRIMARY KEY,
            concept_cloud_id INTEGER NOT NULL,
            space_id INTEGER NOT NULL,
            center_x REAL NOT NULL,
            center_y REAL NOT NULL,
            radius REAL NOT NULL,
            member_lexeme_ids_json TEXT NOT NULL,
            member_weights_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (concept_cloud_id) REFERENCES clouds(id),
            FOREIGN KEY (space_id) REFERENCES spaces(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_overlay_concept ON semantic_overlays(concept_cloud_id)",
        "CREATE INDEX IF NOT EXISTS idx_overlay_space ON semantic_overlays(space_id)",
    ]

    for sql in new_tables:
        conn.execute(sql)


def _add_lexeme_layer(conn: sqlite3.Connection) -> None:
    """Add lexeme layer to layers table if missing."""
    now = "2024-01-01T00:00:00"

    # Check if lexeme layer exists
    row = conn.execute("SELECT id FROM layers WHERE name = 'lexeme'").fetchone()
    if not row:
        # Insert lexeme layer with order_index=3
        conn.execute(
            """INSERT INTO layers (name, order_index, scale, layer_type, config_json, created_at)
            VALUES ('lexeme', 3, 0.5, 'lexeme', '{}', ?)""",
            (now,)
        )

    # Ensure all 7 layers exist with correct order
    default_layers = [
        ("signal", 0, 0.001, "signal"),
        ("character", 1, 0.01, "character"),
        ("word_form", 2, 0.1, "word_form"),
        ("lexeme", 3, 0.5, "lexeme"),
        ("concept", 4, 1.0, "concept"),
        ("scene", 5, 10.0, "scene"),
        ("context", 6, 100.0, "context"),
    ]

    for name, order_index, scale, layer_type in default_layers:
        conn.execute(
            """INSERT OR IGNORE INTO layers (name, order_index, scale, layer_type, config_json, created_at)
            VALUES (?, ?, ?, ?, '{}', ?)""",
            (name, order_index, scale, layer_type, now)
        )
        conn.execute(
            "UPDATE layers SET order_index = ?, scale = ?, layer_type = ? WHERE name = ?",
            (order_index, scale, layer_type, name),
        )


# ============================================================
# Legacy compatibility functions (for gradual migration)
# ============================================================

def _unique_tokens(tokens: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(token for token in tokens if token))


def ensure_concepts(tokens: Sequence[str], initial_position: Tuple[float, float]) -> None:
    """Legacy: ensure concepts exist. Maps to character/word_form layers."""
    # This is kept for backward compatibility during migration
    # New code should use CloudRepository directly
    unique_tokens = _unique_tokens(tokens)
    if not unique_tokens:
        return
    
    encoded_position = json.dumps(list(initial_position), separators=(",", ":"))
    with get_connection() as conn:
        for token in unique_tokens:
            # Try to find in word_form layer first
            layer_row = conn.execute("SELECT id FROM layers WHERE name = 'word_form'").fetchone()
            if layer_row:
                layer_id = layer_row["id"]
                row = conn.execute(
                    "SELECT id FROM clouds WHERE layer_id = ? AND canonical_name = ?",
                    (layer_id, token)
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE clouds SET mass = mass + 0.1, observation_count = observation_count + 1 WHERE id = ?",
                        (row["id"],)
                    )
                else:
                    conn.execute(
                        """INSERT INTO clouds 
                        (layer_id, cloud_type, canonical_name, mass, density, radius, stability, activation, 
                         observation_count, created_at, updated_at, metadata_json)
                        VALUES (?, 'word', ?, 1.0, 1.0, 10.0, 0.1, 0.0, 1, ?, ?, '{}')""",
                        (layer_id, token, encoded_position, now(), now())
                    )
        conn.commit()


def get_concepts() -> List[Dict]:
    """Legacy: get all concepts. Returns word_form clouds for compatibility."""
    with get_connection() as conn:
        layer_row = conn.execute("SELECT id FROM layers WHERE name = 'word_form'").fetchone()
        if not layer_row:
            return []
        rows = conn.execute(
            "SELECT id, canonical_name, metadata_json, mass FROM clouds WHERE layer_id = ? ORDER BY mass DESC",
            (layer_row["id"],)
        ).fetchall()
    concepts = []
    for row in rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        position = metadata.get("position", [400.0, 300.0])
        concepts.append({
            "id": int(row["id"]),
            "token": row["canonical_name"],
            "position": [float(v) for v in position],
            "mass": float(row["mass"]),
        })
    return concepts


def update_concepts(concepts: Iterable[Tuple[int, Sequence[float], float]]) -> None:
    """Legacy: update concept positions and masses."""
    with get_connection() as conn:
        for concept_id, position, mass in concepts:
            metadata_json = json.dumps({"position": list(position)}, separators=(",", ":"))
            conn.execute(
                "UPDATE clouds SET mass = ?, metadata_json = ?, updated_at = ? WHERE id = ?",
                (float(mass), metadata_json, now(), int(concept_id))
            )
        conn.commit()


def get_stats() -> Dict[str, Union[float, int]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS concepts, COALESCE(SUM(mass), 0) AS total_mass FROM clouds"
        ).fetchone()
    concepts = int(row["concepts"])
    total_mass = round(float(row["total_mass"]), 3)
    return {"concepts": concepts, "total_mass": total_mass, "tokens": concepts}


def reset_space() -> None:
    with get_connection() as conn:
        for table in (
            "semantic_overlays",
            "scene_similarity",
            "scenes",
            "lexeme_concept_membership",
            "concept_centroids",
            "context_vectors",
            "word_form_to_lexeme",
            "lexemes",
        ):
            conn.execute(f"DELETE FROM {table}")
        conn.execute("DELETE FROM structural_components")
        conn.execute("DELETE FROM cloud_placements")
        conn.execute("DELETE FROM activation_events")
        conn.execute("DELETE FROM coactivation_stats")
        conn.execute("DELETE FROM condensation_candidates")
        conn.execute("DELETE FROM spaces")
        conn.execute("DELETE FROM clouds")
        conn.commit()


def now() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat()
