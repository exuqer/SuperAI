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
        if version >= 3:
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
        
        # Insert default layers
        default_layers = [
            (0, "signal", 0, 0.001, "signal", "{}"),
            (1, "character", 1, 0.01, "character", "{}"),
            (2, "word_form", 2, 0.1, "word_form", "{}"),
            (3, "concept", 3, 1.0, "concept", "{}"),
            (4, "scene", 4, 10.0, "scene", "{}"),
            (5, "context", 5, 100.0, "context", "{}"),
        ]
        now = "2024-01-01T00:00:00"
        for order, name, idx, scale, ltype, cfg in default_layers:
            conn.execute(
                """INSERT OR IGNORE INTO layers 
                (name, order_index, scale, layer_type, config_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (name, idx, scale, ltype, cfg, now)
            )
        
        conn.execute("PRAGMA user_version = 3")
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
def setup_db():
    """Setup fresh database for each test."""
    global _test_db_path
    
    with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
        temp_path = f.name
    
    _test_db_path = temp_path
    test_init_db()
    
    yield
    
    # Cleanup
    _test_db_path = None
    if os.path.exists(temp_path):
        os.unlink(temp_path)