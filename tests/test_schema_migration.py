import sqlite3

from server.v2.schema import SCHEMA_VERSION, ensure_schema
from server.v2.training import TrainingPipelineV2


def test_legacy_database_is_migrated_before_morphology_training(isolated_database):
    """A pre-morphology database must remain usable after the schema upgrade."""
    with sqlite3.connect(isolated_database) as conn:
        conn.executescript(
            """
            DROP TABLE schema_meta;
            DROP TABLE spaces;
            DROP TABLE clouds;
            CREATE TABLE clouds (
                id INTEGER PRIMARY KEY,
                cloud_type TEXT NOT NULL CHECK (cloud_type IN
                    ('character','word_form','lexeme','scene','concept_candidate','concept')),
                canonical_name TEXT NOT NULL,
                mass REAL NOT NULL DEFAULT 1.0,
                density REAL NOT NULL DEFAULT 1.0,
                stability REAL NOT NULL DEFAULT 0.0,
                base_activation REAL NOT NULL DEFAULT 0.0,
                observation_count INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE spaces (
                id INTEGER PRIMARY KEY,
                space_type TEXT NOT NULL CHECK (space_type IN
                    ('global_field','scene_space','word_structure_space','concept_space','hive_space')),
                owner_cloud_id INTEGER,
                parent_space_id INTEGER,
                dimensionality INTEGER NOT NULL DEFAULT 2,
                random_seed INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            INSERT INTO clouds VALUES
                (1, 'lexeme', 'старый', 1, 1, 0, 0, 1, '{}', 'now', 'now');
            INSERT INTO spaces VALUES
                (1, 'global_field', NULL, NULL, 2, 0, '{}', 'now');
            """
        )
        ensure_schema(conn)
        assert conn.execute("SELECT canonical_name FROM clouds WHERE id=1").fetchone()[0] == "старый"
        assert conn.execute("SELECT space_type FROM spaces WHERE id=1").fetchone()[0] == "global_field"
        assert conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0] == str(SCHEMA_VERSION)

    result = TrainingPipelineV2().train("Мяч. Мячи.")
    assert result["scenes"]
