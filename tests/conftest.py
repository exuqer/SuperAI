"""Pytest configuration - patches database before any server imports."""

import pytest
import tempfile
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

# ============================================================
# Test database setup - MUST run before server imports
# ============================================================

_test_db_path = None

@contextmanager
def test_get_connection():
    """Test-specific connection using temp database."""
    global _test_db_path
    if _test_db_path is None:
        raise RuntimeError("Test database not initialized")
    conn = sqlite3.connect(_test_db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def test_init_db():
    """Initialize test database with nebula schema."""
    global _test_db_path
    with test_get_connection() as conn:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version >= 4:
            return
        
        if version < 3:
            for table in ("concepts", "sessions", "words", "connections", "phrases", "training_stats"):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        
        # LAYERS
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
        
        # CLOUDS
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
        
        # SPACES
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
        
        # CLOUD_PLACEMENTS
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
        
        # STRUCTURAL_COMPONENTS
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
        
        # ACTIVATION_EVENTS
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
        
        # COACTIVATION_STATS
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
        
        # CONDENSATION_CANDIDATES
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
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (source_layer_id) REFERENCES layers(id),
                FOREIGN KEY (target_layer_id) REFERENCES layers(id),
                FOREIGN KEY (proposed_cloud_id) REFERENCES clouds(id)
            )
        """)
        conn.execute("CREATE INDEX idx_cond_source ON condensation_candidates(source_layer_id)")
        conn.execute("CREATE INDEX idx_cond_target ON condensation_candidates(target_layer_id)")
        conn.execute("CREATE INDEX idx_cond_hash ON condensation_candidates(signature_hash)")
        conn.execute("CREATE INDEX idx_cond_status ON condensation_candidates(status)")
        
        # LEXEMES - Normalized word forms with morphological info
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
        
        # WORD_FORM_TO_LEXEME - Links word forms to their lexeme
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
        
        # CONTEXT_VECTORS - PPMI-weighted context vectors for lexemes
        conn.execute("""
            CREATE TABLE context_vectors (
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
            )
        """)
        conn.execute("CREATE INDEX idx_ctx_vec_lexeme ON context_vectors(lexeme_id)")
        conn.execute("CREATE INDEX idx_ctx_vec_context ON context_vectors(context_lexeme_id)")
        conn.execute("CREATE INDEX idx_ctx_vec_weight ON context_vectors(weight DESC)")
        
        # CONCEPT_CENTROIDS - Stable concept centroids from context vectors
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
        
        # LEXEME_CONCEPT_MEMBERSHIP - Fuzzy membership of lexemes in concepts
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
        
        # SCENES - Ordered sequences of word forms (sentences)
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
        
        # SCENE_SIMILARITY - Weighted Jaccard similarity between scenes
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
        
        # SEMANTIC_OVERLAYS - Concept projections in semantic spaces
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
        
        # Insert default layers (including lexeme layer)
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
        
        conn.execute("PRAGMA user_version = 4")
        conn.commit()


# ============================================================
# Patch server.database module
# ============================================================
import server.database as db_module

@contextmanager
def patched_get_connection():
    with test_get_connection() as conn:
        yield conn

def patched_get_db_path():
    global _test_db_path
    if _test_db_path is None:
        raise RuntimeError("Test database not initialized")
    return Path(_test_db_path)

def patched_init_db():
    test_init_db()

# Apply patches
db_module.get_connection = patched_get_connection
db_module.get_db_path = patched_get_db_path
db_module.init_db = patched_init_db


# ============================================================
# Also patch repository modules that import database functions
# ============================================================
import server.repositories.cloud_repository as repo_module
repo_module.get_connection = patched_get_connection


# ============================================================
# Fixture
# ============================================================

@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    """Setup fresh database for each test."""
    global _test_db_path
    
    temp_path = tmp_path / "state.sqlite"
    
    _test_db_path = str(temp_path)
    test_init_db()
    
    yield
    
    # Cleanup
    _test_db_path = None
    if os.path.exists(temp_path):
        os.unlink(temp_path)
