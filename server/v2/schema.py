"""Isolated, additive SQLite schema for the V2 model."""

from __future__ import annotations

import sqlite3


V2_SCHEMA_VERSION = 3


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS v2_schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS v2_clouds (
            id INTEGER PRIMARY KEY,
            cloud_type TEXT NOT NULL CHECK (cloud_type IN
                ('character','word_form','lexeme','concept_candidate','concept','scene')),
            canonical_name TEXT NOT NULL,
            mass REAL NOT NULL DEFAULT 1.0 CHECK (mass >= 0),
            density REAL NOT NULL DEFAULT 1.0 CHECK (density >= 0),
            stability REAL NOT NULL DEFAULT 0.0 CHECK (stability BETWEEN 0 AND 1),
            base_activation REAL NOT NULL DEFAULT 0.0 CHECK (base_activation >= 0),
            observation_count INTEGER NOT NULL DEFAULT 0 CHECK (observation_count >= 0),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS v2_cloud_identity
            ON v2_clouds(cloud_type, canonical_name) WHERE cloud_type <> 'concept';
        CREATE INDEX IF NOT EXISTS v2_cloud_type_idx ON v2_clouds(cloud_type);
        CREATE INDEX IF NOT EXISTS v2_cloud_name_idx ON v2_clouds(canonical_name);

        CREATE TABLE IF NOT EXISTS v2_spaces (
            id INTEGER PRIMARY KEY,
            space_type TEXT NOT NULL CHECK (space_type IN
                ('global_field','scene_space','word_structure_space','concept_space','hive_space')),
            owner_cloud_id INTEGER,
            parent_space_id INTEGER,
            dimensionality INTEGER NOT NULL DEFAULT 2 CHECK (dimensionality IN (2,3)),
            random_seed INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (owner_cloud_id) REFERENCES v2_clouds(id),
            FOREIGN KEY (parent_space_id) REFERENCES v2_spaces(id)
        );
        CREATE INDEX IF NOT EXISTS v2_space_owner_idx ON v2_spaces(owner_cloud_id);
        CREATE INDEX IF NOT EXISTS v2_space_parent_idx ON v2_spaces(parent_space_id);

        CREATE TABLE IF NOT EXISTS v2_cloud_placements (
            id INTEGER PRIMARY KEY,
            cloud_id INTEGER NOT NULL,
            space_id INTEGER NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL,
            radius REAL NOT NULL DEFAULT 12.0 CHECK (radius >= 0),
            local_activation REAL NOT NULL DEFAULT 0.0 CHECK (local_activation >= 0),
            local_density REAL NOT NULL DEFAULT 1.0 CHECK (local_density >= 0),
            local_gravity REAL NOT NULL DEFAULT 0.0,
            local_stability_modifier REAL NOT NULL DEFAULT 0.0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (cloud_id) REFERENCES v2_clouds(id),
            FOREIGN KEY (space_id) REFERENCES v2_spaces(id)
        );
        CREATE INDEX IF NOT EXISTS v2_placement_cloud_idx ON v2_cloud_placements(cloud_id);
        CREATE INDEX IF NOT EXISTS v2_placement_space_idx ON v2_cloud_placements(space_id);
        CREATE INDEX IF NOT EXISTS v2_placement_spatial_idx ON v2_cloud_placements(space_id, x, y);

        CREATE TABLE IF NOT EXISTS v2_structural_components (
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
            FOREIGN KEY (parent_cloud_id) REFERENCES v2_clouds(id),
            FOREIGN KEY (child_cloud_id) REFERENCES v2_clouds(id),
            UNIQUE(parent_cloud_id, component_index)
        );
        CREATE INDEX IF NOT EXISTS v2_component_child_idx ON v2_structural_components(child_cloud_id);

        CREATE TABLE IF NOT EXISTS v2_word_forms (
            cloud_id INTEGER PRIMARY KEY,
            normalized_form TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT 'ru',
            lexeme_cloud_id INTEGER NOT NULL,
            pos_tag TEXT,
            morphology_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (cloud_id) REFERENCES v2_clouds(id),
            FOREIGN KEY (lexeme_cloud_id) REFERENCES v2_clouds(id)
        );
        CREATE INDEX IF NOT EXISTS v2_word_form_lexeme_idx ON v2_word_forms(lexeme_cloud_id);
        CREATE TABLE IF NOT EXISTS v2_lexemes (
            cloud_id INTEGER PRIMARY KEY,
            lemma TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT 'ru',
            pos_tag TEXT,
            semantic_state TEXT NOT NULL DEFAULT 'unassigned'
                CHECK (semantic_state IN ('unassigned','candidate','stable')),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (cloud_id) REFERENCES v2_clouds(id)
        );
        CREATE TABLE IF NOT EXISTS v2_semantic_memberships (
            id INTEGER PRIMARY KEY,
            lexeme_cloud_id INTEGER NOT NULL,
            concept_cloud_id INTEGER NOT NULL,
            weight REAL NOT NULL CHECK (weight BETWEEN 0 AND 1),
            confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
            evidence_count INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (lexeme_cloud_id) REFERENCES v2_clouds(id),
            FOREIGN KEY (concept_cloud_id) REFERENCES v2_clouds(id),
            UNIQUE(lexeme_cloud_id, concept_cloud_id)
        );

        CREATE TABLE IF NOT EXISTS v2_scenes (
            cloud_id INTEGER PRIMARY KEY,
            scene_space_id INTEGER NOT NULL,
            sentence_text TEXT NOT NULL,
            canonical_text TEXT NOT NULL,
            fingerprint TEXT NOT NULL UNIQUE,
            observation_count INTEGER NOT NULL DEFAULT 1,
            parser_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (cloud_id) REFERENCES v2_clouds(id),
            FOREIGN KEY (scene_space_id) REFERENCES v2_spaces(id)
        );
        CREATE TABLE IF NOT EXISTS v2_scene_components (
            id INTEGER PRIMARY KEY,
            scene_cloud_id INTEGER NOT NULL,
            word_form_cloud_id INTEGER NOT NULL,
            lexeme_cloud_id INTEGER NOT NULL,
            placement_id INTEGER NOT NULL,
            token_index INTEGER NOT NULL CHECK (token_index >= 0),
            grammatical_role TEXT NOT NULL,
            dependency_role TEXT NOT NULL DEFAULT 'unknown',
            head_component_id INTEGER,
            confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
            morphology_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (scene_cloud_id) REFERENCES v2_clouds(id),
            FOREIGN KEY (word_form_cloud_id) REFERENCES v2_clouds(id),
            FOREIGN KEY (lexeme_cloud_id) REFERENCES v2_clouds(id),
            FOREIGN KEY (placement_id) REFERENCES v2_cloud_placements(id),
            FOREIGN KEY (head_component_id) REFERENCES v2_scene_components(id),
            UNIQUE(scene_cloud_id, token_index)
        );
        CREATE TABLE IF NOT EXISTS v2_training_observations (
            id INTEGER PRIMARY KEY,
            source_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            scene_cloud_id INTEGER,
            source_type TEXT NOT NULL DEFAULT 'training',
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (scene_cloud_id) REFERENCES v2_clouds(id)
        );
        CREATE TABLE IF NOT EXISTS v2_legacy_scene_imports (
            legacy_scene_id INTEGER PRIMARY KEY,
            source_updated_at TEXT NOT NULL,
            v2_scene_cloud_id INTEGER NOT NULL,
            imported_at TEXT NOT NULL,
            FOREIGN KEY (v2_scene_cloud_id) REFERENCES v2_clouds(id)
        );

        CREATE TABLE IF NOT EXISTS v2_hives (
            id TEXT PRIMARY KEY,
            space_id INTEGER NOT NULL,
            query_text TEXT NOT NULL,
            query_json TEXT NOT NULL DEFAULT '{}',
            max_cells INTEGER NOT NULL DEFAULT 24,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (space_id) REFERENCES v2_spaces(id)
        );
        CREATE TABLE IF NOT EXISTS v2_hive_cells (
            id TEXT PRIMARY KEY,
            hive_id TEXT NOT NULL,
            dominant_cloud_id INTEGER NOT NULL,
            source_placement_id INTEGER,
            source_scene_cloud_id INTEGER,
            x REAL NOT NULL,
            y REAL NOT NULL,
            stored_strength REAL NOT NULL,
            query_relevance REAL NOT NULL,
            composition_cohesion REAL NOT NULL,
            retention REAL NOT NULL,
            local_activation REAL NOT NULL DEFAULT 0 CHECK (local_activation BETWEEN 0 AND 1),
            component_activation REAL NOT NULL DEFAULT 0 CHECK (component_activation BETWEEN 0 AND 1),
            conversation_focus REAL NOT NULL DEFAULT 0 CHECK (conversation_focus BETWEEN 0 AND 1),
            activation_count INTEGER NOT NULL DEFAULT 0,
            last_activated_at TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            component_class TEXT NOT NULL CHECK (component_class IN ('core','context','background','noise')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (hive_id) REFERENCES v2_hives(id),
            FOREIGN KEY (dominant_cloud_id) REFERENCES v2_clouds(id),
            FOREIGN KEY (source_placement_id) REFERENCES v2_cloud_placements(id),
            FOREIGN KEY (source_scene_cloud_id) REFERENCES v2_clouds(id)
        );
        CREATE INDEX IF NOT EXISTS v2_hive_cell_retention_idx ON v2_hive_cells(hive_id, retention DESC);
        CREATE TABLE IF NOT EXISTS v2_hive_cell_components (
            id INTEGER PRIMARY KEY,
            cell_id TEXT NOT NULL,
            cloud_id INTEGER NOT NULL,
            composition_share REAL NOT NULL CHECK (composition_share BETWEEN 0 AND 1),
            local_activation REAL NOT NULL DEFAULT 0 CHECK (local_activation BETWEEN 0 AND 1),
            activation_count INTEGER NOT NULL DEFAULT 0,
            last_activated_at TEXT,
            source_placement_id INTEGER,
            provenance_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (cell_id) REFERENCES v2_hive_cells(id) ON DELETE CASCADE,
            FOREIGN KEY (cloud_id) REFERENCES v2_clouds(id),
            FOREIGN KEY (source_placement_id) REFERENCES v2_cloud_placements(id)
        );
        CREATE TABLE IF NOT EXISTS v2_hive_messages (
            id TEXT PRIMARY KEY,
            hive_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            parsed_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (hive_id) REFERENCES v2_hives(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS v2_hive_messages_idx ON v2_hive_messages(hive_id, turn_index);
        CREATE TABLE IF NOT EXISTS v2_hive_query_decisions (
            id TEXT PRIMARY KEY,
            hive_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            external_search_required INTEGER NOT NULL DEFAULT 0,
            search_budget_json TEXT NOT NULL DEFAULT '{}',
            anchors_json TEXT NOT NULL DEFAULT '[]',
            unresolved_json TEXT NOT NULL DEFAULT '[]',
            reasons_json TEXT NOT NULL DEFAULT '[]',
            metrics_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (hive_id) REFERENCES v2_hives(id) ON DELETE CASCADE,
            FOREIGN KEY (message_id) REFERENCES v2_hive_messages(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS v2_hive_decisions_idx ON v2_hive_query_decisions(hive_id, created_at);
        CREATE TABLE IF NOT EXISTS v2_hive_resonance_events (
            id TEXT PRIMARY KEY,
            hive_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            cell_id TEXT NOT NULL,
            component_cloud_id INTEGER,
            reason TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (hive_id) REFERENCES v2_hives(id) ON DELETE CASCADE,
            FOREIGN KEY (message_id) REFERENCES v2_hive_messages(id) ON DELETE CASCADE,
            FOREIGN KEY (cell_id) REFERENCES v2_hive_cells(id) ON DELETE CASCADE,
            FOREIGN KEY (component_cloud_id) REFERENCES v2_clouds(id)
        );
        CREATE INDEX IF NOT EXISTS v2_hive_resonance_idx ON v2_hive_resonance_events(hive_id, created_at);
        CREATE TABLE IF NOT EXISTS v2_hive_cell_matches (
            id INTEGER PRIMARY KEY,
            decision_id TEXT NOT NULL,
            cell_id TEXT NOT NULL,
            component_id TEXT NOT NULL,
            match_type TEXT NOT NULL,
            local_support REAL NOT NULL CHECK (local_support BETWEEN 0 AND 1),
            role_compatibility REAL NOT NULL DEFAULT 1,
            component_share REAL NOT NULL DEFAULT 1,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (decision_id) REFERENCES v2_hive_query_decisions(id) ON DELETE CASCADE,
            FOREIGN KEY (cell_id) REFERENCES v2_hive_cells(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS v2_hive_matches_idx ON v2_hive_cell_matches(cell_id, decision_id);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO v2_schema_meta(key, value) VALUES ('schema_version', ?)",
        (str(V2_SCHEMA_VERSION),),
    )
