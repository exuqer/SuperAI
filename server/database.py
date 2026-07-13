"""Persistent storage for the recursive nebula system."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


DB_PATH = Path(".superai/state.sqlite")
SCHEMA_VERSION = 4


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
    """Initialize database with new nebula schema."""
    with get_connection() as conn:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version >= SCHEMA_VERSION and _current_schema_is_complete(conn):
            return

        if version >= SCHEMA_VERSION:
            _migrate_current_schema(conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            return
        
        # Drop old tables if version < 3
        if version < 3:
            for table in ("concepts", "sessions", "words", "connections", "phrases", "training_stats"):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        
        # ============================================================
        # LAYERS - Scale layers
        # ============================================================
        conn.execute("""
            CREATE TABLE layers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                order_index INTEGER NOT NULL,
                scale REAL NOT NULL DEFAULT 1.0,
                layer_type TEXT NOT NULL DEFAULT '',
                config_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX idx_layers_order ON layers(order_index)")
        
        # ============================================================
        # CLOUDS - Global nebula entities
        # ============================================================
        conn.execute("""
            CREATE TABLE clouds (
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
            )
        """)
        conn.execute("CREATE INDEX idx_clouds_layer ON clouds(layer_id)")
        conn.execute("CREATE INDEX idx_clouds_type ON clouds(cloud_type)")
        conn.execute("CREATE INDEX idx_clouds_name ON clouds(canonical_name)")
        conn.execute("CREATE INDEX idx_clouds_stability ON clouds(stability DESC)")
        conn.execute("CREATE INDEX idx_clouds_activated ON clouds(last_activated_at DESC)")
        conn.execute("CREATE UNIQUE INDEX idx_clouds_layer_name ON clouds(layer_id, canonical_name)")
        
        # ============================================================
        # SPACES - Local spaces inside host clouds
        # ============================================================
        conn.execute("""
            CREATE TABLE spaces (
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
            )
        """)
        conn.execute("CREATE INDEX idx_spaces_host ON spaces(host_cloud_id)")
        conn.execute("CREATE INDEX idx_spaces_layer ON spaces(layer_id)")
        conn.execute("CREATE INDEX idx_spaces_mode ON spaces(mode)")
        
        # ============================================================
        # CLOUD_PLACEMENTS - Local appearance of cloud in a space
        # ============================================================
        conn.execute("""
            CREATE TABLE cloud_placements (
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
            )
        """)
        conn.execute("CREATE INDEX idx_placements_space ON cloud_placements(space_id)")
        conn.execute("CREATE INDEX idx_placements_cloud ON cloud_placements(cloud_id)")
        conn.execute("CREATE INDEX idx_placements_density ON cloud_placements(density DESC)")
        conn.execute("CREATE INDEX idx_placements_spatial ON cloud_placements(x, y)")
        
        # ============================================================
        # STRUCTURAL_COMPONENTS - Internal composition (technical, not semantic)
        # ============================================================
        conn.execute("""
            CREATE TABLE structural_components (
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
            )
        """)
        conn.execute("CREATE INDEX idx_struct_parent ON structural_components(parent_cloud_id)")
        conn.execute("CREATE INDEX idx_struct_child ON structural_components(child_cloud_id)")
        conn.execute("CREATE INDEX idx_struct_order ON structural_components(parent_cloud_id, position_index)")
        
        # ============================================================
        # ACTIVATION_EVENTS - Temporal activation log
        # ============================================================
        conn.execute("""
            CREATE TABLE activation_events (
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
            )
        """)
        conn.execute("CREATE INDEX idx_activation_session ON activation_events(session_id)")
        conn.execute("CREATE INDEX idx_activation_cloud ON activation_events(cloud_id)")
        conn.execute("CREATE INDEX idx_activation_layer ON activation_events(layer_id)")
        conn.execute("CREATE INDEX idx_activation_time ON activation_events(timestamp)")
        conn.execute("CREATE INDEX idx_activation_context ON activation_events(context_window_id)")
        
        # ============================================================
        # COACTIVATION_STATS - Joint activation statistics (for physics)
        # ============================================================
        conn.execute("""
            CREATE TABLE coactivation_stats (
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
            )
        """)
        conn.execute("CREATE INDEX idx_coact_a ON coactivation_stats(cloud_a_id)")
        conn.execute("CREATE INDEX idx_coact_b ON coactivation_stats(cloud_b_id)")
        conn.execute("CREATE INDEX idx_coact_layer ON coactivation_stats(layer_id)")
        conn.execute("CREATE INDEX idx_coact_score ON coactivation_stats(weighted_score DESC)")
        
        # ============================================================
        # CONDENSATION_CANDIDATES - Accumulating configurations for condensation
        # ============================================================
        conn.execute("""
            CREATE TABLE condensation_candidates (
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
                FOREIGN KEY (source_layer_id) REFERENCES layers(id),
                FOREIGN KEY (target_layer_id) REFERENCES layers(id),
                FOREIGN KEY (proposed_cloud_id) REFERENCES clouds(id)
            )
        """)
        conn.execute("CREATE INDEX idx_cond_source ON condensation_candidates(source_layer_id)")
        conn.execute("CREATE INDEX idx_cond_target ON condensation_candidates(target_layer_id)")
        conn.execute("CREATE INDEX idx_cond_hash ON condensation_candidates(signature_hash)")
        conn.execute("CREATE INDEX idx_cond_status ON condensation_candidates(status)")
        
        # ============================================================
        # LEXEMES - Normalized word forms with morphological info
        # ============================================================
        conn.execute("""
            CREATE TABLE lexemes (
                id INTEGER PRIMARY KEY,
                canonical_form TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT 'ru',
                pos_tag TEXT,
                features_json TEXT NOT NULL DEFAULT '{}',
                frequency INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE UNIQUE INDEX idx_lexemes_form_lang ON lexemes(canonical_form, language)")
        conn.execute("CREATE INDEX idx_lexemes_freq ON lexemes(frequency DESC)")
        
        # ============================================================
        # WORD_FORM_TO_LEXEME - Links word forms to their lexeme
        # ============================================================
        conn.execute("""
            CREATE TABLE word_form_to_lexeme (
                word_form_cloud_id INTEGER NOT NULL,
                lexeme_id INTEGER NOT NULL,
                is_canonical INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                PRIMARY KEY (word_form_cloud_id, lexeme_id),
                FOREIGN KEY (word_form_cloud_id) REFERENCES clouds(id),
                FOREIGN KEY (lexeme_id) REFERENCES lexemes(id)
            )
        """)
        conn.execute("CREATE INDEX idx_wf2lex_lexeme ON word_form_to_lexeme(lexeme_id)")
        
        # ============================================================
        # CONTEXT_VECTORS - PPMI-weighted context vectors for lexemes
        # ============================================================
        conn.execute("""
            CREATE TABLE context_vectors (
                lexeme_id INTEGER NOT NULL,
                context_lexeme_id INTEGER NOT NULL,
                weight REAL NOT NULL,
                count INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (lexeme_id, context_lexeme_id),
                FOREIGN KEY (lexeme_id) REFERENCES lexemes(id),
                FOREIGN KEY (context_lexeme_id) REFERENCES lexemes(id)
            )
        """)
        conn.execute("CREATE INDEX idx_ctx_vec_lexeme ON context_vectors(lexeme_id)")
        conn.execute("CREATE INDEX idx_ctx_vec_context ON context_vectors(context_lexeme_id)")
        conn.execute("CREATE INDEX idx_ctx_vec_weight ON context_vectors(weight DESC)")
        
        # ============================================================
        # CONCEPT_CENTROIDS - Stable concept centroids from context vectors
        # ============================================================
        conn.execute("")
        conn.execute("""
            CREATE TABLE concept_centroids (
                id INTEGER PRIMARY KEY,
                concept_cloud_id INTEGER NOT NULL,
                centroid_vector_json TEXT NOT NULL,
                member_lexeme_ids_json TEXT NOT NULL,
                stability REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (concept_cloud_id) REFERENCES clouds(id)
            )
        """)
        conn.execute("CREATE INDEX idx_centroid_concept ON concept_centroids(concept_cloud_id)")
        
        # ============================================================
        # LEXEME_CONCEPT_MEMBERSHIP - Fuzzy membership of lexemes in concepts
        # ============================================================
        conn.execute("""
            CREATE TABLE lexeme_concept_membership (
                lexeme_id INTEGER NOT NULL,
                concept_cloud_id INTEGER NOT NULL,
                membership REAL NOT NULL,
                centrality REAL NOT NULL DEFAULT 0.0,
                context_coverage REAL NOT NULL DEFAULT 0.0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (lexeme_id, concept_cloud_id),
                FOREIGN KEY (lexeme_id) REFERENCES lexemes(id),
                FOREIGN KEY (concept_cloud_id) REFERENCES clouds(id)
            )
        """)
        conn.execute("CREATE INDEX idx_lcm_concept ON lexeme_concept_membership(concept_cloud_id)")
        conn.execute("CREATE INDEX idx_lcm_membership ON lexeme_concept_membership(membership DESC)")
        
        # ============================================================
        # SCENES - Ordered sequences of word forms (sentences)
        # ============================================================
        conn.execute("""
            CREATE TABLE scenes (
                id INTEGER PRIMARY KEY,
                scene_cloud_id INTEGER NOT NULL,
                sentence_text TEXT NOT NULL,
                word_form_cloud_ids_json TEXT NOT NULL,
                lexeme_ids_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (scene_cloud_id) REFERENCES clouds(id)
            )
        """)
        conn.execute("CREATE INDEX idx_scenes_cloud ON scenes(scene_cloud_id)")
        
        # ============================================================
        # SCENE_SIMILARITY - Weighted Jaccard similarity between scenes
        # ============================================================
        conn.execute("""
            CREATE TABLE scene_similarity (
                scene_a_id INTEGER NOT NULL,
                scene_b_id INTEGER NOT NULL,
                similarity REAL NOT NULL,
                weight REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (scene_a_id, scene_b_id),
                FOREIGN KEY (scene_a_id) REFERENCES scenes(id),
                FOREIGN KEY (scene_b_id) REFERENCES scenes(id)
            )
        """)
        conn.execute("CREATE INDEX idx_scene_sim_a ON scene_similarity(scene_a_id)")
        conn.execute("CREATE INDEX idx_scene_sim_b ON scene_similarity(scene_b_id)")
        conn.execute("CREATE INDEX idx_scene_sim_score ON scene_similarity(similarity DESC)")
        
        # ============================================================
        # SEMANTIC_OVERLAYS - Concept projections in semantic spaces
        # ============================================================
        conn.execute("""
            CREATE TABLE semantic_overlays (
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
            )
        """)
        conn.execute("CREATE INDEX idx_overlay_concept ON semantic_overlays(concept_cloud_id)")
        conn.execute("CREATE INDEX idx_overlay_space ON semantic_overlays(space_id)")
        
        # ============================================================
        # Insert default layers (including lexeme layer)
        # ============================================================
        default_layers = [
            (0, "signal", 0, 0.001, "signal", "{}"),
            (1, "character", 1, 0.01, "character", "{}"),
            (2, "word_form", 2, 0.1, "word_form", "{}"),
            (3, "lexeme", 3, 0.5, "lexeme", "{}"),
            (4, "concept", 4, 1.0, "concept", "{}"),
            (5, "scene", 5, 10.0, "scene", "{}"),
            (6, "context", 6, 100.0, "context", "{}"),
        ]
        now = "2024-01-01T00:00:00"
        for order, name, idx, scale, ltype, cfg in default_layers:
            conn.execute(
                """INSERT OR IGNORE INTO layers 
                (name, order_index, scale, layer_type, config_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (name, idx, scale, ltype, cfg, now)
            )
        
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()


def _current_schema_is_complete(conn: sqlite3.Connection) -> bool:
    required_tables = {
        "layers",
        "clouds",
        "spaces",
        "cloud_placements",
        "structural_components",
        "activation_events",
        "coactivation_stats",
        "condensation_candidates",
    }
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    if not required_tables.issubset({row[0] for row in rows}):
        return False

    columns = {
        row[1] for row in conn.execute(
            "PRAGMA table_info(condensation_candidates)"
        ).fetchall()
    }
    return {"created_at", "updated_at"}.issubset(columns)


def _migrate_current_schema(conn: sqlite3.Connection) -> None:
    """Apply additive migrations to databases created by an earlier MVP build."""
    columns = {
        row[1] for row in conn.execute(
            "PRAGMA table_info(condensation_candidates)"
        ).fetchall()
    }
    if "created_at" not in columns:
        conn.execute("ALTER TABLE condensation_candidates ADD COLUMN created_at TEXT")
        conn.execute(
            "UPDATE condensation_candidates SET created_at = COALESCE(created_at, datetime('now'))"
        )
    if "updated_at" not in columns:
        conn.execute("ALTER TABLE condensation_candidates ADD COLUMN updated_at TEXT")
        conn.execute(
            "UPDATE condensation_candidates SET updated_at = COALESCE(updated_at, datetime('now'))"
        )

    conn.execute("CREATE INDEX IF NOT EXISTS idx_cond_created ON condensation_candidates(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cond_updated ON condensation_candidates(updated_at)")


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
        conn.execute("DELETE FROM cloud_placements")
        conn.execute("DELETE FROM structural_components")
        conn.execute("DELETE FROM activation_events")
        conn.execute("DELETE FROM coactivation_stats")
        conn.execute("DELETE FROM condensation_candidates")
        conn.execute("DELETE FROM spaces")
        conn.execute("DELETE FROM clouds")
        conn.commit()


def now() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat()
