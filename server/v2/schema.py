"""Canonical Cloud / Space / Placement schema."""

from __future__ import annotations

import sqlite3


SCHEMA_VERSION = 1


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS clouds (
            id INTEGER PRIMARY KEY,
            cloud_type TEXT NOT NULL CHECK (cloud_type IN
                ('character','word_form','lexeme','scene','concept_candidate','concept')),
            canonical_name TEXT NOT NULL,
            mass REAL NOT NULL DEFAULT 1.0 CHECK (mass >= 0),
            density REAL NOT NULL DEFAULT 1.0 CHECK (density >= 0),
            stability REAL NOT NULL DEFAULT 0.0 CHECK (stability BETWEEN 0 AND 1),
            base_activation REAL NOT NULL DEFAULT 0.0 CHECK (base_activation BETWEEN 0 AND 1),
            observation_count INTEGER NOT NULL DEFAULT 0 CHECK (observation_count >= 0),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS cloud_identity
            ON clouds(cloud_type, canonical_name) WHERE cloud_type <> 'concept';
        CREATE INDEX IF NOT EXISTS cloud_type_idx ON clouds(cloud_type);
        CREATE INDEX IF NOT EXISTS cloud_name_idx ON clouds(canonical_name);

        CREATE TABLE IF NOT EXISTS spaces (
            id INTEGER PRIMARY KEY,
            space_type TEXT NOT NULL CHECK (space_type IN
                ('global_field','scene_space','word_structure_space','concept_space','hive_space')),
            owner_cloud_id INTEGER,
            parent_space_id INTEGER,
            dimensionality INTEGER NOT NULL DEFAULT 2 CHECK (dimensionality IN (2,3)),
            random_seed INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (owner_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY (parent_space_id) REFERENCES spaces(id) ON DELETE CASCADE
        );
        CREATE UNIQUE INDEX IF NOT EXISTS one_global_field
            ON spaces(space_type) WHERE space_type = 'global_field';
        CREATE UNIQUE INDEX IF NOT EXISTS one_scene_space_per_scene
            ON spaces(owner_cloud_id) WHERE space_type = 'scene_space';
        CREATE UNIQUE INDEX IF NOT EXISTS one_word_structure_space
            ON spaces(owner_cloud_id) WHERE space_type = 'word_structure_space';
        CREATE INDEX IF NOT EXISTS space_owner_idx ON spaces(owner_cloud_id);
        CREATE INDEX IF NOT EXISTS space_parent_idx ON spaces(parent_space_id);

        CREATE TABLE IF NOT EXISTS cloud_placements (
            id INTEGER PRIMARY KEY,
            cloud_id INTEGER NOT NULL,
            space_id INTEGER NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL,
            radius REAL NOT NULL DEFAULT 12.0 CHECK (radius >= 0),
            local_activation REAL NOT NULL DEFAULT 0.0 CHECK (local_activation BETWEEN 0 AND 1),
            local_density REAL NOT NULL DEFAULT 1.0 CHECK (local_density >= 0),
            local_gravity REAL NOT NULL DEFAULT 0.0,
            local_stability_modifier REAL NOT NULL DEFAULT 0.0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY (space_id) REFERENCES spaces(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS placement_cloud_idx ON cloud_placements(cloud_id);
        CREATE INDEX IF NOT EXISTS placement_space_idx ON cloud_placements(space_id);
        CREATE INDEX IF NOT EXISTS placement_spatial_idx ON cloud_placements(space_id, x, y);
        CREATE UNIQUE INDEX IF NOT EXISTS one_global_placement_per_cloud
            ON cloud_placements(cloud_id, space_id)
            WHERE json_extract(metadata_json, '$.placement_kind') = 'global';

        CREATE TABLE IF NOT EXISTS structural_components (
            id INTEGER PRIMARY KEY,
            parent_cloud_id INTEGER NOT NULL,
            child_cloud_id INTEGER NOT NULL,
            component_index INTEGER NOT NULL CHECK (component_index >= 0),
            component_role TEXT NOT NULL DEFAULT 'unknown',
            weight REAL NOT NULL DEFAULT 1.0 CHECK (weight >= 0),
            local_x REAL NOT NULL DEFAULT 0.0,
            local_y REAL NOT NULL DEFAULT 0.0,
            local_z REAL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (parent_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY (child_cloud_id) REFERENCES clouds(id),
            UNIQUE(parent_cloud_id, component_index)
        );
        CREATE INDEX IF NOT EXISTS component_child_idx ON structural_components(child_cloud_id);

        CREATE TABLE IF NOT EXISTS lexemes (
            cloud_id INTEGER PRIMARY KEY,
            lemma TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT 'ru',
            pos_tag TEXT,
            frequency INTEGER NOT NULL DEFAULT 0,
            semantic_state TEXT NOT NULL DEFAULT 'unassigned'
                CHECK (semantic_state IN ('unassigned','candidate','stable')),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (cloud_id) REFERENCES clouds(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS word_forms (
            cloud_id INTEGER PRIMARY KEY,
            normalized_form TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT 'ru',
            lexeme_cloud_id INTEGER,
            pos_tag TEXT,
            morphology_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY (lexeme_cloud_id) REFERENCES clouds(id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS word_form_normalized_idx
            ON word_forms(normalized_form, language);
        CREATE INDEX IF NOT EXISTS word_form_lexeme_idx ON word_forms(lexeme_cloud_id);

        CREATE TABLE IF NOT EXISTS semantic_memberships (
            id INTEGER PRIMARY KEY,
            lexeme_cloud_id INTEGER NOT NULL,
            concept_cloud_id INTEGER NOT NULL,
            weight REAL NOT NULL CHECK (weight BETWEEN 0 AND 1),
            confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
            evidence_count INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (lexeme_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY (concept_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            UNIQUE(lexeme_cloud_id, concept_cloud_id)
        );

        CREATE TABLE IF NOT EXISTS scenes (
            cloud_id INTEGER PRIMARY KEY,
            scene_space_id INTEGER NOT NULL UNIQUE,
            sentence_text TEXT NOT NULL,
            canonical_text TEXT NOT NULL,
            fingerprint TEXT NOT NULL UNIQUE,
            parser_version TEXT NOT NULL,
            observation_count INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY (scene_space_id) REFERENCES spaces(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS scene_components (
            id INTEGER PRIMARY KEY,
            scene_cloud_id INTEGER NOT NULL,
            word_form_cloud_id INTEGER NOT NULL,
            lexeme_cloud_id INTEGER,
            placement_id INTEGER NOT NULL UNIQUE,
            token_index INTEGER NOT NULL CHECK (token_index >= 0),
            grammatical_role TEXT NOT NULL,
            dependency_role TEXT,
            head_component_id INTEGER,
            confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
            morphology_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (scene_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY (word_form_cloud_id) REFERENCES clouds(id),
            FOREIGN KEY (lexeme_cloud_id) REFERENCES clouds(id),
            FOREIGN KEY (placement_id) REFERENCES cloud_placements(id) ON DELETE CASCADE,
            FOREIGN KEY (head_component_id) REFERENCES scene_components(id),
            UNIQUE(scene_cloud_id, token_index)
        );

        CREATE TABLE IF NOT EXISTS training_runs (
            id TEXT PRIMARY KEY,
            source_text TEXT NOT NULL,
            source_type TEXT NOT NULL,
            success INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS training_observations (
            id INTEGER PRIMARY KEY,
            training_run_id TEXT NOT NULL,
            source_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            scene_cloud_id INTEGER,
            source_type TEXT NOT NULL DEFAULT 'training',
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (training_run_id) REFERENCES training_runs(id) ON DELETE CASCADE,
            FOREIGN KEY (scene_cloud_id) REFERENCES clouds(id)
        );

        CREATE TABLE IF NOT EXISTS training_change_events (
            id INTEGER PRIMARY KEY,
            training_run_id TEXT NOT NULL,
            event_type TEXT NOT NULL CHECK (event_type IN
                ('CLOUD_CREATED','CLOUD_STRENGTHENED','SPACE_CREATED','PLACEMENT_CREATED',
                 'PLACEMENT_MOVED','STRUCTURE_CREATED','SCENE_REUSED','LEXEME_LINKED',
                 'CANDIDATE_CREATED','ACTIVATION_CHANGED')),
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            value_before_json TEXT,
            value_after_json TEXT,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (training_run_id) REFERENCES training_runs(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS training_event_run_idx
            ON training_change_events(training_run_id, id);

        CREATE TABLE IF NOT EXISTS hives (
            id TEXT PRIMARY KEY,
            space_id INTEGER NOT NULL UNIQUE,
            query_text TEXT NOT NULL DEFAULT '',
            query_json TEXT NOT NULL DEFAULT '{}',
            max_cells INTEGER NOT NULL DEFAULT 24,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (space_id) REFERENCES spaces(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS hive_cells (
            id TEXT PRIMARY KEY,
            hive_id TEXT NOT NULL,
            dominant_cloud_id INTEGER NOT NULL,
            hive_placement_id INTEGER NOT NULL UNIQUE,
            source_cloud_id INTEGER NOT NULL,
            source_placement_id INTEGER,
            source_space_id INTEGER,
            source_scene_cloud_id INTEGER,
            stored_strength REAL NOT NULL CHECK (stored_strength BETWEEN 0 AND 1),
            retention REAL NOT NULL CHECK (retention BETWEEN 0 AND 1),
            local_activation REAL NOT NULL CHECK (local_activation BETWEEN 0 AND 1),
            component_class TEXT NOT NULL DEFAULT 'context',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (hive_id) REFERENCES hives(id) ON DELETE CASCADE,
            FOREIGN KEY (dominant_cloud_id) REFERENCES clouds(id),
            FOREIGN KEY (hive_placement_id) REFERENCES cloud_placements(id) ON DELETE CASCADE,
            FOREIGN KEY (source_cloud_id) REFERENCES clouds(id),
            FOREIGN KEY (source_placement_id) REFERENCES cloud_placements(id),
            FOREIGN KEY (source_space_id) REFERENCES spaces(id),
            FOREIGN KEY (source_scene_cloud_id) REFERENCES clouds(id)
        );
        CREATE INDEX IF NOT EXISTS hive_cell_retention_idx ON hive_cells(hive_id, retention DESC);

        CREATE TABLE IF NOT EXISTS hive_cell_components (
            id INTEGER PRIMARY KEY,
            cell_id TEXT NOT NULL,
            cloud_id INTEGER NOT NULL,
            composition_share REAL NOT NULL CHECK (composition_share BETWEEN 0 AND 1),
            local_activation REAL NOT NULL DEFAULT 0 CHECK (local_activation BETWEEN 0 AND 1),
            source_cloud_id INTEGER NOT NULL,
            source_placement_id INTEGER,
            source_space_id INTEGER,
            provenance_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (cell_id) REFERENCES hive_cells(id) ON DELETE CASCADE,
            FOREIGN KEY (cloud_id) REFERENCES clouds(id),
            FOREIGN KEY (source_cloud_id) REFERENCES clouds(id),
            FOREIGN KEY (source_placement_id) REFERENCES cloud_placements(id),
            FOREIGN KEY (source_space_id) REFERENCES spaces(id),
            UNIQUE(cell_id, cloud_id)
        );

        CREATE TABLE IF NOT EXISTS hive_messages (
            id TEXT PRIMARY KEY,
            hive_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            parsed_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (hive_id) REFERENCES hives(id) ON DELETE CASCADE,
            UNIQUE(hive_id, turn_index)
        );

        CREATE TABLE IF NOT EXISTS hive_query_decisions (
            id TEXT PRIMARY KEY,
            hive_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            external_search_required INTEGER NOT NULL DEFAULT 0,
            anchors_json TEXT NOT NULL DEFAULT '[]',
            unresolved_json TEXT NOT NULL DEFAULT '[]',
            reasons_json TEXT NOT NULL DEFAULT '[]',
            metrics_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (hive_id) REFERENCES hives(id) ON DELETE CASCADE,
            FOREIGN KEY (message_id) REFERENCES hive_messages(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS hive_resonance_events (
            id TEXT PRIMARY KEY,
            hive_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            cell_id TEXT NOT NULL,
            component_cloud_id INTEGER,
            reason TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (hive_id) REFERENCES hives(id) ON DELETE CASCADE,
            FOREIGN KEY (message_id) REFERENCES hive_messages(id) ON DELETE CASCADE,
            FOREIGN KEY (cell_id) REFERENCES hive_cells(id) ON DELETE CASCADE,
            FOREIGN KEY (component_cloud_id) REFERENCES clouds(id)
        );

        CREATE TABLE IF NOT EXISTS hive_cell_matches (
            id INTEGER PRIMARY KEY,
            decision_id TEXT NOT NULL,
            cell_id TEXT NOT NULL,
            component_id TEXT NOT NULL,
            match_type TEXT NOT NULL,
            local_support REAL NOT NULL CHECK (local_support BETWEEN 0 AND 1),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (decision_id) REFERENCES hive_query_decisions(id) ON DELETE CASCADE,
            FOREIGN KEY (cell_id) REFERENCES hive_cells(id) ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
