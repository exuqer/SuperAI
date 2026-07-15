"""Canonical Cloud / Space / Placement schema."""

from __future__ import annotations

import sqlite3


SCHEMA_VERSION = 3


def _needs_constraint_migration(conn: sqlite3.Connection, table: str, expected_value: str) -> bool:
    """Return whether a legacy SQLite CHECK constraint must be rebuilt.

    SQLite cannot alter a CHECK constraint in place.  The first V2 schema only
    allowed the original cloud and space types, so an existing database needs a
    table rebuild before morphology records can be inserted.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return bool(row and expected_value not in (row[0] or ""))


def _migrate_legacy_type_constraints(conn: sqlite3.Connection) -> None:
    """Expand legacy cloud/space type constraints without losing model data."""
    migrate_clouds = _needs_constraint_migration(conn, "clouds", "morph_pattern")
    migrate_spaces = _needs_constraint_migration(conn, "spaces", "morphology_space")
    if not migrate_clouds and not migrate_spaces:
        return

    # This function runs before ensure_schema performs any writes.  Disabling
    # FK checks lets us replace parent tables while retaining all child rows;
    # the copied primary keys keep every existing relation valid.
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        if migrate_clouds:
            conn.executescript(
                """
                CREATE TABLE clouds__v2 (
                    id INTEGER PRIMARY KEY,
                    cloud_type TEXT NOT NULL CHECK (cloud_type IN
                        ('character','word_form','lexeme','scene','concept_candidate','concept',
                         'morpheme_candidate','morpheme','morph_operator','morph_pattern','sentence_frame')),
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
                INSERT INTO clouds__v2
                    SELECT id, cloud_type, canonical_name, mass, density, stability,
                           base_activation, observation_count, metadata_json, created_at, updated_at
                    FROM clouds;
                DROP TABLE clouds;
                ALTER TABLE clouds__v2 RENAME TO clouds;
                """
            )
        if migrate_spaces:
            conn.executescript(
                """
                CREATE TABLE spaces__v2 (
                    id INTEGER PRIMARY KEY,
                    space_type TEXT NOT NULL CHECK (space_type IN
                        ('global_field','scene_space','word_structure_space','morphology_space',
                         'sentence_frame_space','concept_space','hive_space','hive_subspace')),
                    owner_cloud_id INTEGER,
                    parent_space_id INTEGER,
                    dimensionality INTEGER NOT NULL DEFAULT 2 CHECK (dimensionality IN (2,3)),
                    random_seed INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (owner_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
                    FOREIGN KEY (parent_space_id) REFERENCES spaces(id) ON DELETE CASCADE
                );
                INSERT INTO spaces__v2
                    SELECT id, space_type, owner_cloud_id, parent_space_id, dimensionality,
                           random_seed, metadata_json, created_at
                    FROM spaces;
                DROP TABLE spaces;
                ALTER TABLE spaces__v2 RENAME TO spaces;
                """
            )
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def ensure_schema(conn: sqlite3.Connection) -> None:
    _migrate_legacy_type_constraints(conn)
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
                ('character','word_form','lexeme','scene','concept_candidate','concept',
                 'morpheme_candidate','morpheme','morph_operator','morph_pattern','sentence_frame')),
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
                ('global_field','scene_space','word_structure_space','morphology_space',
                 'sentence_frame_space','concept_space','hive_space','hive_subspace')),
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

        CREATE TABLE IF NOT EXISTS cloud_compositions (
            id INTEGER PRIMARY KEY,
            parent_cloud_id INTEGER NOT NULL,
            child_cloud_id INTEGER NOT NULL,
            relation_type TEXT NOT NULL,
            child_order INTEGER NOT NULL DEFAULT 0,
            weight REAL NOT NULL DEFAULT 1.0,
            confidence REAL NOT NULL DEFAULT 0.0,
            evidence_count INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(parent_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY(child_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            UNIQUE(parent_cloud_id, child_cloud_id, relation_type, child_order)
        );
        CREATE INDEX IF NOT EXISTS cloud_composition_parent_idx ON cloud_compositions(parent_cloud_id);
        CREATE INDEX IF NOT EXISTS cloud_composition_child_idx ON cloud_compositions(child_cloud_id);

        CREATE TABLE IF NOT EXISTS word_form_features (
            id INTEGER PRIMARY KEY,
            word_form_cloud_id INTEGER NOT NULL UNIQUE,
            lexeme_cloud_id INTEGER,
            part_of_speech TEXT,
            number TEXT,
            grammatical_case TEXT,
            gender TEXT,
            tense TEXT,
            person TEXT,
            animacy TEXT,
            aspect TEXT,
            degree TEXT,
            confidence REAL NOT NULL DEFAULT 0.0,
            evidence_count INTEGER NOT NULL DEFAULT 0,
            features_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(word_form_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY(lexeme_cloud_id) REFERENCES clouds(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS morph_pattern_data (
            cloud_id INTEGER PRIMARY KEY,
            operator_cloud_id INTEGER,
            input_signature_json TEXT NOT NULL DEFAULT '{}',
            output_template_json TEXT NOT NULL DEFAULT '{}',
            compatibility_json TEXT NOT NULL DEFAULT '{}',
            confidence REAL NOT NULL DEFAULT 0.0,
            evidence_count INTEGER NOT NULL DEFAULT 0,
            successful_uses INTEGER NOT NULL DEFAULT 0,
            failed_uses INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY(operator_cloud_id) REFERENCES clouds(id) ON DELETE SET NULL
        );

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

        CREATE TABLE IF NOT EXISTS hive_subspaces (
            id INTEGER PRIMARY KEY,
            hive_id TEXT NOT NULL,
            parent_cell_id TEXT,
            parent_placement_id INTEGER,
            space_id INTEGER NOT NULL,
            subspace_type TEXT NOT NULL,
            depth INTEGER NOT NULL DEFAULT 0,
            capacity INTEGER NOT NULL DEFAULT 12,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            expansion_reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(hive_id) REFERENCES hives(id) ON DELETE CASCADE,
            FOREIGN KEY(parent_cell_id) REFERENCES hive_cells(id) ON DELETE CASCADE,
            FOREIGN KEY(parent_placement_id) REFERENCES cloud_placements(id) ON DELETE SET NULL,
            FOREIGN KEY(space_id) REFERENCES spaces(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS hive_subspace_hive_idx ON hive_subspaces(hive_id, status);

        CREATE TABLE IF NOT EXISTS hive_generation_candidates (
            id INTEGER PRIMARY KEY,
            hive_id TEXT NOT NULL,
            subspace_id INTEGER,
            sentence_slot_id TEXT,
            source_lexeme_cloud_id INTEGER,
            candidate_text TEXT NOT NULL,
            requested_features_json TEXT NOT NULL DEFAULT '{}',
            applied_patterns_json TEXT NOT NULL DEFAULT '[]',
            character_sequence_json TEXT NOT NULL DEFAULT '[]',
            score_total REAL NOT NULL DEFAULT 0,
            score_semantic REAL NOT NULL DEFAULT 0,
            score_grammar REAL NOT NULL DEFAULT 0,
            score_pattern REAL NOT NULL DEFAULT 0,
            score_orthography REAL NOT NULL DEFAULT 0,
            score_context REAL NOT NULL DEFAULT 0,
            reverse_validation_score REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'GENERATED',
            provenance_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(hive_id) REFERENCES hives(id) ON DELETE CASCADE,
            FOREIGN KEY(subspace_id) REFERENCES hive_subspaces(id) ON DELETE SET NULL,
            FOREIGN KEY(source_lexeme_cloud_id) REFERENCES clouds(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS generation_candidate_hive_idx ON hive_generation_candidates(hive_id, status, score_total DESC);

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
            role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('user', 'assistant')),
            text TEXT NOT NULL,
            parsed_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (hive_id) REFERENCES hives(id) ON DELETE CASCADE,
            UNIQUE(hive_id, turn_index)
        );

        CREATE TABLE IF NOT EXISTS hive_dialogue_scenes (
            id TEXT PRIMARY KEY,
            hive_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            source_role TEXT NOT NULL CHECK(source_role IN ('user', 'assistant')),
            source_text TEXT NOT NULL,
            roles_json TEXT NOT NULL DEFAULT '{}',
            activation REAL NOT NULL DEFAULT 1 CHECK(activation BETWEEN 0 AND 1),
            retention REAL NOT NULL DEFAULT 1 CHECK(retention BETWEEN 0 AND 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (hive_id) REFERENCES hives(id) ON DELETE CASCADE,
            FOREIGN KEY (message_id) REFERENCES hive_messages(id) ON DELETE CASCADE,
            UNIQUE(hive_id, message_id)
        );
        CREATE INDEX IF NOT EXISTS hive_dialogue_scenes_hive_idx
            ON hive_dialogue_scenes(hive_id, updated_at DESC);

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
    # Additive upgrades for databases created by schema v1.
    hive_columns = {
        "conversation_id": "TEXT NOT NULL DEFAULT ''",
        "hive_space_id": "INTEGER",
        "status": "TEXT NOT NULL DEFAULT 'ACTIVE'",
        "capacity": "INTEGER NOT NULL DEFAULT 24",
        "reasoning_step": "INTEGER NOT NULL DEFAULT 0",
        "current_temperature": "REAL NOT NULL DEFAULT 1.0",
        "total_energy": "REAL NOT NULL DEFAULT 0.0",
        "random_seed": "INTEGER NOT NULL DEFAULT 0",
        "last_reasoned_at": "TEXT",
        "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
    }
    existing = {row[1] for row in conn.execute("PRAGMA table_info(hives)")}
    for column, declaration in hive_columns.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE hives ADD COLUMN {column} {declaration}")
    message_columns = {row[1] for row in conn.execute("PRAGMA table_info(hive_messages)")}
    if "role" not in message_columns:
        conn.execute("ALTER TABLE hive_messages ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
    component_columns = {
        "role": "TEXT NOT NULL DEFAULT 'context'",
        "effective_strength": "REAL NOT NULL DEFAULT 0 CHECK (effective_strength BETWEEN 0 AND 1)",
        "component_class": "TEXT NOT NULL DEFAULT 'context'",
    }
    existing_components = {row[1] for row in conn.execute("PRAGMA table_info(hive_cell_components)")}
    for column, declaration in component_columns.items():
        if column not in existing_components:
            conn.execute(f"ALTER TABLE hive_cell_components ADD COLUMN {column} {declaration}")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS structural_signatures (
            cloud_id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            signature_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(cloud_id) REFERENCES clouds(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS structural_index (
            index_type TEXT NOT NULL,
            fragment TEXT NOT NULL,
            cloud_id INTEGER NOT NULL,
            PRIMARY KEY(index_type, fragment, cloud_id),
            FOREIGN KEY(cloud_id) REFERENCES clouds(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS structural_index_lookup_idx
            ON structural_index(index_type, fragment);

        CREATE INDEX IF NOT EXISTS hive_conversation_idx ON hives(conversation_id);
        CREATE UNIQUE INDEX IF NOT EXISTS active_hive_conversation_idx
            ON hives(conversation_id) WHERE status = 'ACTIVE' AND conversation_id <> '';

        CREATE TABLE IF NOT EXISTS hive_node_states (
            hive_id TEXT NOT NULL,
            placement_id INTEGER NOT NULL,
            cloud_id INTEGER NOT NULL,
            node_type TEXT NOT NULL DEFAULT 'temporary_candidate',
            x REAL NOT NULL DEFAULT 0,
            y REAL NOT NULL DEFAULT 0,
            z REAL,
            velocity_x REAL NOT NULL DEFAULT 0,
            velocity_y REAL NOT NULL DEFAULT 0,
            velocity_z REAL,
            local_activation REAL NOT NULL DEFAULT 0 CHECK (local_activation BETWEEN 0 AND 1),
            local_gravity REAL NOT NULL DEFAULT 0 CHECK (local_gravity BETWEEN 0 AND 1),
            stored_strength REAL NOT NULL DEFAULT 0 CHECK (stored_strength BETWEEN 0 AND 1),
            local_stability REAL NOT NULL DEFAULT 0 CHECK (local_stability BETWEEN 0 AND 1),
            retention REAL NOT NULL DEFAULT 0 CHECK (retention BETWEEN 0 AND 1),
            energy REAL NOT NULL DEFAULT 0 CHECK (energy BETWEEN 0 AND 1),
            phase REAL NOT NULL DEFAULT 0,
            frequency REAL NOT NULL DEFAULT 1,
            temperature_response REAL NOT NULL DEFAULT 1,
            age_steps INTEGER NOT NULL DEFAULT 0,
            activation_count INTEGER NOT NULL DEFAULT 0,
            last_activated_step INTEGER NOT NULL DEFAULT 0,
            weakening_steps INTEGER NOT NULL DEFAULT 0,
            eviction_status TEXT NOT NULL DEFAULT 'ACTIVE',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY(hive_id, placement_id),
            FOREIGN KEY(hive_id) REFERENCES hives(id) ON DELETE CASCADE,
            FOREIGN KEY(placement_id) REFERENCES cloud_placements(id) ON DELETE CASCADE,
            FOREIGN KEY(cloud_id) REFERENCES clouds(id)
        );
        CREATE INDEX IF NOT EXISTS hive_node_state_cloud_idx ON hive_node_states(hive_id, cloud_id);

        CREATE TABLE IF NOT EXISTS hive_reasoning_runs (
            id TEXT PRIMARY KEY,
            hive_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'COMPLETED',
            reasoning_steps INTEGER NOT NULL,
            completed_steps INTEGER NOT NULL DEFAULT 0,
            query_json TEXT NOT NULL DEFAULT '{}',
            config_json TEXT NOT NULL DEFAULT '{}',
            random_seed INTEGER NOT NULL DEFAULT 0,
            stop_reason TEXT,
            initial_state_hash TEXT,
            final_state_hash TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            FOREIGN KEY(hive_id) REFERENCES hives(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS hive_reasoning_run_idx ON hive_reasoning_runs(hive_id, created_at);

        CREATE TABLE IF NOT EXISTS hive_reasoning_snapshots (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            hive_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            phase TEXT NOT NULL,
            state_hash TEXT NOT NULL,
            state_json TEXT NOT NULL,
            delta_json TEXT NOT NULL DEFAULT '{}',
            clusters_json TEXT NOT NULL DEFAULT '[]',
            events_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES hive_reasoning_runs(id) ON DELETE CASCADE,
            FOREIGN KEY(hive_id) REFERENCES hives(id) ON DELETE CASCADE,
            UNIQUE(run_id, step, phase)
        );
        CREATE INDEX IF NOT EXISTS hive_snapshot_run_idx ON hive_reasoning_snapshots(run_id, step);

        CREATE TABLE IF NOT EXISTS hive_reasoning_events (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            hive_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            phase TEXT NOT NULL,
            event_type TEXT NOT NULL,
            placement_id INTEGER,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES hive_reasoning_runs(id) ON DELETE CASCADE,
            FOREIGN KEY(hive_id) REFERENCES hives(id) ON DELETE CASCADE,
            FOREIGN KEY(placement_id) REFERENCES cloud_placements(id)
        );

        CREATE TABLE IF NOT EXISTS hive_resonance_clusters (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            hive_id TEXT NOT NULL,
            reasoning_step INTEGER NOT NULL,
            member_placement_ids_json TEXT NOT NULL DEFAULT '[]',
            dominant_cloud_ids_json TEXT NOT NULL DEFAULT '[]',
            cohesion REAL NOT NULL DEFAULT 0,
            total_energy REAL NOT NULL DEFAULT 0,
            average_gravity REAL NOT NULL DEFAULT 0,
            query_relevance REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES hive_reasoning_runs(id) ON DELETE CASCADE,
            FOREIGN KEY(hive_id) REFERENCES hives(id) ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
