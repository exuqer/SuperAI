"""Canonical Cloud / Space / Placement schema."""

from __future__ import annotations

import json
import sqlite3


SCHEMA_VERSION = 10


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {
        str(row["name"] if hasattr(row, "keys") else row[1])
        for row in conn.execute(f"PRAGMA table_info({table})")
    }
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


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
    migrate_clouds = _needs_constraint_migration(conn, "clouds", "'entity'")
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
                         'entity','morpheme_candidate','morpheme','morph_operator','morph_pattern',
                         'sentence_frame')),
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


def _migrate_concept_relation_constraints(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='concept_relations'"
    ).fetchone()
    sql = str((row[0] if row else "") or "")
    if not sql or "ALIAS_OF" in sql:
        return
    conn.executescript(
        """
        DROP INDEX IF EXISTS concept_relation_subject_type_idx;
        DROP INDEX IF EXISTS concept_relation_object_type_idx;
        DROP INDEX IF EXISTS concept_relation_lookup_idx;
        CREATE TABLE concept_relations__v7 (
            id TEXT PRIMARY KEY,
            relation_type TEXT NOT NULL CHECK(relation_type IN
                ('IS_A','INSTANCE_OF','PART_OF','HAS_PART','HAS_PROPERTY','LOCATED_IN',
                 'LOCATED_ON','LOCATED_NEAR','OWNS','USES','PRODUCES','REQUIRES','CAUSES',
                 'RESULTS_IN','BEFORE','AFTER','SIMILAR_TO','OPPOSITE_TO','ALIAS_OF')),
            subject_lexeme_cloud_id INTEGER NOT NULL,
            object_lexeme_cloud_id INTEGER NOT NULL,
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            status TEXT NOT NULL DEFAULT 'STABLE',
            direct INTEGER NOT NULL DEFAULT 1 CHECK(direct IN (0,1)),
            depth INTEGER NOT NULL DEFAULT 1 CHECK(depth >= 1),
            evidence_count INTEGER NOT NULL DEFAULT 0 CHECK(evidence_count >= 0),
            source_type TEXT NOT NULL DEFAULT 'CLASSIFICATION_DEFINITION',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(subject_lexeme_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY(object_lexeme_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            UNIQUE(relation_type, subject_lexeme_cloud_id, object_lexeme_cloud_id)
        );
        INSERT INTO concept_relations__v7
            SELECT id, relation_type, subject_lexeme_cloud_id, object_lexeme_cloud_id,
                   confidence, status, direct, depth, evidence_count, source_type,
                   created_at, updated_at
            FROM concept_relations;
        DROP TABLE concept_relations;
        ALTER TABLE concept_relations__v7 RENAME TO concept_relations;
        CREATE INDEX concept_relation_subject_type_idx
            ON concept_relations(subject_lexeme_cloud_id, relation_type);
        CREATE INDEX concept_relation_object_type_idx
            ON concept_relations(object_lexeme_cloud_id, relation_type);
        CREATE INDEX concept_relation_lookup_idx
            ON concept_relations(subject_lexeme_cloud_id, relation_type, object_lexeme_cloud_id);
        """
    )


def _migrate_query_role_hypothesis_constraints(conn: sqlite3.Connection) -> None:
    if not _needs_constraint_migration(
        conn,
        "query_role_hypotheses",
        "'entity_type'",
    ):
        return
    conn.executescript(
        """
        DROP TABLE IF EXISTS query_role_hypotheses__v10;
        CREATE TABLE query_role_hypotheses__v10 (
            id TEXT PRIMARY KEY,
            query_frame_id TEXT NOT NULL,
            semantic_role TEXT NOT NULL CHECK(semantic_role IN
                ('action','entity','entity_type','agent','patient','theme','object','experiencer','recipient','source',
                 'destination','location','instrument','material','cause','result','purpose',
                 'time','attribute','quantity','owner','possessed','manner')),
            grammatical_slot TEXT,
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            selected INTEGER NOT NULL DEFAULT 0 CHECK(selected IN (0,1)),
            evidence_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(query_frame_id) REFERENCES query_frames(id) ON DELETE CASCADE
        );
        INSERT INTO query_role_hypotheses__v10
            SELECT id,query_frame_id,semantic_role,grammatical_slot,confidence,
                   selected,evidence_json
            FROM query_role_hypotheses;
        DROP TABLE query_role_hypotheses;
        ALTER TABLE query_role_hypotheses__v10 RENAME TO query_role_hypotheses;
        """
    )


def _backfill_v25_scenes(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """SELECT s.cloud_id,s.sentence_text,s.canonical_text,s.parser_version,
                  e.id AS event_id,e.predicate_lemma,e.predicate_surface,
                  e.polarity,e.modality
           FROM scenes s LEFT JOIN events e ON e.source_scene_id=s.cloud_id
           WHERE s.source_interpretation_id IS NULL
           ORDER BY s.cloud_id"""
    ).fetchall()
    for row in rows:
        (
            scene_id_raw,
            sentence_text,
            canonical_text,
            parser_version,
            event_id,
            predicate_lemma,
            predicate_surface,
            polarity,
            modality,
        ) = row
        scene_id = int(scene_id_raw)
        utterance_id = f"utterance-backfill-scene-{scene_id}"
        clause_id = f"clause-backfill-scene-{scene_id}"
        hypothesis_id = f"interpretation-backfill-scene-{scene_id}"
        evidence_id = f"evidence-backfill-scene-{scene_id}"
        predicate_hypotheses = (
            [{
                "lemma": predicate_lemma,
                "surface": predicate_surface,
                "confidence": 0.82,
                "selected": True,
                "evidence": ["legacy_confirmed_event"],
            }]
            if predicate_lemma else []
        )
        conn.execute(
            """INSERT OR IGNORE INTO utterances
               (id,conversation_id,turn_index,speaker_role,raw_text,
                normalized_text,received_at,language,source_type,
                parser_version,interpretation_status,message_id)
               VALUES(?,'',0,'source',?,?,CURRENT_TIMESTAMP,'ru',
                      'legacy_scene',?,'STABLE',NULL)""",
            (
                utterance_id,
                sentence_text,
                canonical_text,
                parser_version,
            ),
        )
        conn.execute(
            """INSERT OR IGNORE INTO clauses
               (id,utterance_id,sentence_index,parent_clause_id,token_start,
                token_end,clause_type,relation_to_parent,
                predicate_hypotheses_json,mode,actuality,evidence_status,
                polarity,negation_scope_json,modality,completion_status,
                temporal_anchor_json,speaker,quoted_speaker,surface,
                evidence_json,alternatives_json,participants_json)
               VALUES(?,?,0,NULL,0,0,'MAIN',NULL,?,'ASSERTION','ACTUAL',
                      'OBSERVED',?,NULL,?,'UNKNOWN',NULL,'source',NULL,?,
                      '[]','[]','[]')""",
            (
                clause_id,
                utterance_id,
                json.dumps(predicate_hypotheses, ensure_ascii=False),
                str(polarity or "positive").upper(),
                modality,
                sentence_text,
            ),
        )
        conn.execute(
            """INSERT OR IGNORE INTO interpretation_hypotheses
               (id,scope_type,scope_id,hypothesis_type,value_json,status,
                support_by_group_json,support,penalties_json,constraints_json,
                unresolved_slots_json,stability_cycles,leader_margin,selected,
                parser_version)
               VALUES(?,'clause',?,'legacy_scene',?,'CONFIRMED',
                      '{"source":0.82}',.82,'[]','[]','[]',2,.82,1,?)""",
            (
                hypothesis_id,
                clause_id,
                json.dumps(
                    {
                        "scene_id": scene_id,
                        "event_id": event_id,
                    },
                    ensure_ascii=False,
                ),
                parser_version,
            ),
        )
        conn.execute(
            """INSERT OR IGNORE INTO interpretation_evidence
               (id,origin,target_hypothesis_id,value_json,support,penalty,
                evidence_type,independent_group,scope_type,scope_id,
                source_token_start,source_token_end,source_object_id,
                parser_version)
               VALUES(?,'schema_backfill',?,? ,.82,0,'legacy_confirmed_scene',
                      'source','clause',?,NULL,NULL,?,?)""",
            (
                evidence_id,
                hypothesis_id,
                json.dumps({"scene_id": scene_id}),
                clause_id,
                str(scene_id),
                parser_version,
            ),
        )
        conn.execute(
            """UPDATE scenes SET source_interpretation_id=?
               WHERE cloud_id=? AND source_interpretation_id IS NULL""",
            (hypothesis_id, scene_id),
        )


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
                 'entity','morpheme_candidate','morpheme','morph_operator','morph_pattern',
                 'sentence_frame')),
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

        CREATE TABLE IF NOT EXISTS semantic_evidence (
            id INTEGER PRIMARY KEY,
            source_scene_cloud_id INTEGER,
            left_lexeme_cloud_id INTEGER NOT NULL,
            right_lexeme_cloud_id INTEGER NOT NULL,
            evidence_type TEXT NOT NULL CHECK (evidence_type IN
                ('definition','shared_category','contextual_similarity')),
            weight REAL NOT NULL CHECK (weight BETWEEN 0 AND 1),
            evidence_weight REAL NOT NULL DEFAULT 0 CHECK (evidence_weight BETWEEN 0 AND 1),
            independence REAL NOT NULL DEFAULT 1 CHECK (independence BETWEEN 0 AND 1),
            evidence_key TEXT,
            evidence_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_scene_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY(left_lexeme_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY(right_lexeme_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            UNIQUE(source_scene_cloud_id, left_lexeme_cloud_id, right_lexeme_cloud_id, evidence_type)
        );
        CREATE INDEX IF NOT EXISTS semantic_evidence_left_idx ON semantic_evidence(left_lexeme_cloud_id);
        CREATE INDEX IF NOT EXISTS semantic_evidence_right_idx ON semantic_evidence(right_lexeme_cloud_id);
        CREATE UNIQUE INDEX IF NOT EXISTS semantic_evidence_key_idx ON semantic_evidence(evidence_key) WHERE evidence_key IS NOT NULL;

        CREATE TABLE IF NOT EXISTS concept_fog_registry (
            concept_cloud_id INTEGER PRIMARY KEY,
            concept_space_id INTEGER NOT NULL UNIQUE,
            evidence_type TEXT NOT NULL,
            stability REAL NOT NULL DEFAULT 0 CHECK (stability BETWEEN 0 AND 1),
            evidence_count INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL,
            FOREIGN KEY(concept_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY(concept_space_id) REFERENCES spaces(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS concept_candidate_registry (
            concept_candidate_cloud_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate','rejected','stabilized')),
            stability_score REAL NOT NULL DEFAULT 0 CHECK (stability_score BETWEEN 0 AND 1),
            is_search_eligible INTEGER NOT NULL DEFAULT 0 CHECK (is_search_eligible IN (0,1)),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL,
            FOREIGN KEY(concept_candidate_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS semantic_backfill_state (
            source_scene_cloud_id INTEGER PRIMARY KEY,
            semantic_extractor_version INTEGER NOT NULL,
            input_fingerprint TEXT NOT NULL,
            result_fingerprint TEXT NOT NULL,
            processed_at TEXT NOT NULL,
            FOREIGN KEY(source_scene_cloud_id) REFERENCES scenes(cloud_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS action_concepts (
            id TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            space_id INTEGER,
            status TEXT NOT NULL CHECK(status IN
                ('OBSERVED','CANDIDATE','PROBABLE','STABLE','CONFLICTED','DEPRECATED')),
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            mass REAL NOT NULL DEFAULT 0.5 CHECK(mass >= 0),
            evidence_count INTEGER NOT NULL DEFAULT 0 CHECK(evidence_count >= 0),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(space_id) REFERENCES spaces(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS semantic_constructions (
            id TEXT PRIMARY KEY,
            predicate_lemma TEXT NOT NULL UNIQUE,
            pattern_type TEXT NOT NULL,
            argument_mapping_json TEXT NOT NULL DEFAULT '{}',
            implied_semantics_json TEXT NOT NULL DEFAULT '{}',
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            evidence_count INTEGER NOT NULL DEFAULT 0 CHECK(evidence_count >= 0)
        );

        CREATE TABLE IF NOT EXISTS action_variants (
            id TEXT PRIMARY KEY,
            action_concept_id TEXT NOT NULL,
            lexeme_cloud_id INTEGER,
            lemma TEXT NOT NULL,
            construction_id TEXT,
            weight REAL NOT NULL CHECK(weight BETWEEN 0 AND 1),
            evidence_count INTEGER NOT NULL DEFAULT 0 CHECK(evidence_count >= 0),
            source_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(action_concept_id) REFERENCES action_concepts(id) ON DELETE CASCADE,
            FOREIGN KEY(lexeme_cloud_id) REFERENCES clouds(id) ON DELETE SET NULL,
            FOREIGN KEY(construction_id) REFERENCES semantic_constructions(id) ON DELETE SET NULL,
            UNIQUE(action_concept_id, lemma)
        );
        CREATE INDEX IF NOT EXISTS action_variant_lemma_idx ON action_variants(lemma);

        CREATE TABLE IF NOT EXISTS scene_concept_projections (
            id TEXT PRIMARY KEY,
            scene_id INTEGER NOT NULL,
            action_concept_id TEXT NOT NULL,
            semantic_frame_json TEXT NOT NULL DEFAULT '{}',
            projection_confidence REAL NOT NULL CHECK(projection_confidence BETWEEN 0 AND 1),
            projection_version INTEGER NOT NULL DEFAULT 1,
            source_type TEXT NOT NULL DEFAULT 'scene_parser',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(scene_id) REFERENCES scenes(cloud_id) ON DELETE CASCADE,
            FOREIGN KEY(action_concept_id) REFERENCES action_concepts(id) ON DELETE CASCADE,
            UNIQUE(scene_id, action_concept_id, projection_version)
        );
        CREATE INDEX IF NOT EXISTS scene_projection_concept_idx
            ON scene_concept_projections(action_concept_id, scene_id);

        CREATE TABLE IF NOT EXISTS concept_relation_evidence (
            id TEXT PRIMARY KEY,
            source_scene_id INTEGER,
            relation_type TEXT NOT NULL,
            source_concept_id TEXT,
            target_concept_id TEXT,
            weight REAL NOT NULL CHECK(weight BETWEEN 0 AND 1),
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            status TEXT NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(source_scene_id) REFERENCES scenes(cloud_id) ON DELETE SET NULL,
            FOREIGN KEY(source_concept_id) REFERENCES action_concepts(id) ON DELETE SET NULL,
            FOREIGN KEY(target_concept_id) REFERENCES action_concepts(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS concept_relations (
            id TEXT PRIMARY KEY,
            relation_type TEXT NOT NULL CHECK(relation_type IN
                ('IS_A','INSTANCE_OF','PART_OF','HAS_PART','HAS_PROPERTY','LOCATED_IN',
                 'LOCATED_ON','LOCATED_NEAR','OWNS','USES','PRODUCES','REQUIRES','CAUSES',
                 'RESULTS_IN','BEFORE','AFTER','SIMILAR_TO','OPPOSITE_TO','ALIAS_OF')),
            subject_lexeme_cloud_id INTEGER NOT NULL,
            object_lexeme_cloud_id INTEGER NOT NULL,
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            status TEXT NOT NULL DEFAULT 'STABLE',
            direct INTEGER NOT NULL DEFAULT 1 CHECK(direct IN (0,1)),
            depth INTEGER NOT NULL DEFAULT 1 CHECK(depth >= 1),
            evidence_count INTEGER NOT NULL DEFAULT 0 CHECK(evidence_count >= 0),
            source_type TEXT NOT NULL DEFAULT 'CLASSIFICATION_DEFINITION',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(subject_lexeme_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            FOREIGN KEY(object_lexeme_cloud_id) REFERENCES clouds(id) ON DELETE CASCADE,
            UNIQUE(relation_type, subject_lexeme_cloud_id, object_lexeme_cloud_id)
        );
        CREATE INDEX IF NOT EXISTS concept_relation_subject_type_idx
            ON concept_relations(subject_lexeme_cloud_id, relation_type);
        CREATE INDEX IF NOT EXISTS concept_relation_object_type_idx
            ON concept_relations(object_lexeme_cloud_id, relation_type);
        CREATE INDEX IF NOT EXISTS concept_relation_lookup_idx
            ON concept_relations(subject_lexeme_cloud_id, relation_type, object_lexeme_cloud_id);

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
        CREATE INDEX IF NOT EXISTS scene_component_lexeme_idx
            ON scene_components(lexeme_cloud_id, scene_cloud_id);
        CREATE INDEX IF NOT EXISTS scene_component_form_idx
            ON scene_components(word_form_cloud_id, scene_cloud_id);
        CREATE INDEX IF NOT EXISTS scene_component_role_idx
            ON scene_components(grammatical_role, lexeme_cloud_id, scene_cloud_id);

        CREATE TABLE IF NOT EXISTS entities (
            cloud_id INTEGER PRIMARY KEY,
            canonical_lemma TEXT NOT NULL,
            display_name TEXT NOT NULL,
            entity_kind TEXT NOT NULL DEFAULT 'entity',
            status TEXT NOT NULL DEFAULT 'OBSERVED',
            confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0 AND 1),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(cloud_id) REFERENCES clouds(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS entity_lemma_idx ON entities(canonical_lemma);

        CREATE TABLE IF NOT EXISTS entity_aliases (
            id INTEGER PRIMARY KEY,
            entity_id INTEGER NOT NULL,
            alias TEXT NOT NULL,
            normalized_alias TEXT NOT NULL,
            lexeme_cloud_id INTEGER,
            source_scene_id INTEGER,
            confidence REAL NOT NULL DEFAULT 1 CHECK(confidence BETWEEN 0 AND 1),
            source_type TEXT NOT NULL DEFAULT 'observation',
            created_at TEXT NOT NULL,
            FOREIGN KEY(entity_id) REFERENCES entities(cloud_id) ON DELETE CASCADE,
            FOREIGN KEY(lexeme_cloud_id) REFERENCES clouds(id) ON DELETE SET NULL,
            FOREIGN KEY(source_scene_id) REFERENCES scenes(cloud_id) ON DELETE SET NULL,
            UNIQUE(entity_id, normalized_alias)
        );
        CREATE INDEX IF NOT EXISTS entity_alias_lookup_idx
            ON entity_aliases(normalized_alias, entity_id);

        CREATE TABLE IF NOT EXISTS entity_mentions (
            id TEXT PRIMARY KEY,
            source_scene_id INTEGER NOT NULL,
            entity_id INTEGER NOT NULL,
            token_start INTEGER NOT NULL CHECK(token_start >= 0),
            token_end INTEGER NOT NULL CHECK(token_end >= token_start),
            head_token_index INTEGER NOT NULL,
            surface TEXT NOT NULL,
            normalized_surface TEXT NOT NULL,
            mention_type TEXT NOT NULL DEFAULT 'noun_phrase',
            entity_type_id INTEGER,
            preposition TEXT NOT NULL DEFAULT '',
            grammatical_features_json TEXT NOT NULL DEFAULT '{}',
            attributes_json TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0 AND 1),
            parser_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_scene_id) REFERENCES scenes(cloud_id) ON DELETE CASCADE,
            FOREIGN KEY(entity_id) REFERENCES entities(cloud_id) ON DELETE CASCADE,
            FOREIGN KEY(entity_type_id) REFERENCES entities(cloud_id) ON DELETE SET NULL,
            UNIQUE(source_scene_id, token_start, token_end)
        );
        CREATE INDEX IF NOT EXISTS entity_mention_scene_idx
            ON entity_mentions(source_scene_id, token_start);
        CREATE INDEX IF NOT EXISTS entity_mention_entity_idx
            ON entity_mentions(entity_id, source_scene_id);

        CREATE TABLE IF NOT EXISTS construction_templates (
            id TEXT PRIMARY KEY,
            predicate_lemma TEXT NOT NULL,
            surface_pattern TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'OBSERVED',
            confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0 AND 1),
            evidence_count INTEGER NOT NULL DEFAULT 0 CHECK(evidence_count >= 0),
            source_type TEXT NOT NULL DEFAULT 'learned',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(predicate_lemma, surface_pattern)
        );
        CREATE INDEX IF NOT EXISTS construction_predicate_idx
            ON construction_templates(predicate_lemma, confidence DESC);

        CREATE TABLE IF NOT EXISTS construction_arguments (
            id TEXT PRIMARY KEY,
            construction_id TEXT NOT NULL,
            argument_index INTEGER NOT NULL,
            grammatical_slot TEXT NOT NULL,
            morphological_constraints_json TEXT NOT NULL DEFAULT '{}',
            semantic_role TEXT NOT NULL CHECK(semantic_role IN
                ('agent','patient','theme','object','experiencer','recipient','source',
                 'destination','location','instrument','material','cause','result','purpose',
                 'time','attribute','quantity','owner','possessed','manner')),
            confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0 AND 1),
            FOREIGN KEY(construction_id) REFERENCES construction_templates(id) ON DELETE CASCADE,
            UNIQUE(construction_id, argument_index, semantic_role)
        );

        CREATE TABLE IF NOT EXISTS construction_evidence (
            id TEXT PRIMARY KEY,
            construction_id TEXT NOT NULL,
            source_scene_id INTEGER NOT NULL,
            evidence_type TEXT NOT NULL,
            weight REAL NOT NULL CHECK(weight BETWEEN 0 AND 1),
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(construction_id) REFERENCES construction_templates(id) ON DELETE CASCADE,
            FOREIGN KEY(source_scene_id) REFERENCES scenes(cloud_id) ON DELETE CASCADE,
            UNIQUE(construction_id, source_scene_id, evidence_type)
        );

        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            source_scene_id INTEGER NOT NULL UNIQUE,
            predicate_lemma TEXT NOT NULL,
            predicate_surface TEXT NOT NULL,
            predicate_lexeme_cloud_id INTEGER,
            construction_id TEXT,
            polarity TEXT NOT NULL DEFAULT 'positive',
            modality TEXT NOT NULL DEFAULT 'fact',
            confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0 AND 1),
            parser_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_scene_id) REFERENCES scenes(cloud_id) ON DELETE CASCADE,
            FOREIGN KEY(predicate_lexeme_cloud_id) REFERENCES clouds(id) ON DELETE SET NULL,
            FOREIGN KEY(construction_id) REFERENCES construction_templates(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS event_predicate_idx
            ON events(predicate_lemma, source_scene_id);
        CREATE INDEX IF NOT EXISTS event_construction_idx
            ON events(construction_id, source_scene_id);

        CREATE TABLE IF NOT EXISTS event_participants (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            entity_id INTEGER NOT NULL,
            mention_id TEXT NOT NULL,
            semantic_role TEXT NOT NULL CHECK(semantic_role IN
                ('agent','patient','theme','object','experiencer','recipient','source',
                 'destination','location','instrument','material','cause','result','purpose',
                 'time','attribute','quantity','owner','possessed','manner')),
            grammatical_slot TEXT NOT NULL,
            participant_index INTEGER NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0 AND 1),
            preposition TEXT NOT NULL DEFAULT '',
            surface TEXT NOT NULL,
            lemma TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
            FOREIGN KEY(entity_id) REFERENCES entities(cloud_id) ON DELETE CASCADE,
            FOREIGN KEY(mention_id) REFERENCES entity_mentions(id) ON DELETE CASCADE,
            UNIQUE(event_id, participant_index)
        );
        CREATE INDEX IF NOT EXISTS event_participant_role_entity_idx
            ON event_participants(semantic_role, entity_id, event_id);
        CREATE INDEX IF NOT EXISTS event_participant_slot_entity_idx
            ON event_participants(grammatical_slot, entity_id, event_id);
        CREATE INDEX IF NOT EXISTS event_participant_event_role_idx
            ON event_participants(event_id, semantic_role);

        CREATE TABLE IF NOT EXISTS event_role_hypotheses (
            id TEXT PRIMARY KEY,
            participant_id TEXT NOT NULL,
            semantic_role TEXT NOT NULL CHECK(semantic_role IN
                ('agent','patient','theme','object','experiencer','recipient','source',
                 'destination','location','instrument','material','cause','result','purpose',
                 'time','attribute','quantity','owner','possessed','manner')),
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            source_type TEXT NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(participant_id) REFERENCES event_participants(id) ON DELETE CASCADE,
            UNIQUE(participant_id, semantic_role, source_type)
        );

        CREATE TABLE IF NOT EXISTS event_modifiers (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            target_participant_id TEXT,
            role TEXT NOT NULL,
            value_entity_id INTEGER,
            value_text TEXT NOT NULL DEFAULT '',
            attributes_json TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0 AND 1),
            source_mention_id TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
            FOREIGN KEY(target_participant_id) REFERENCES event_participants(id) ON DELETE CASCADE,
            FOREIGN KEY(value_entity_id) REFERENCES entities(cloud_id) ON DELETE SET NULL,
            FOREIGN KEY(source_mention_id) REFERENCES entity_mentions(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS concepts (
            id TEXT PRIMARY KEY,
            cloud_id INTEGER,
            concept_kind TEXT NOT NULL,
            canonical_name TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN
                ('OBSERVED','CANDIDATE','PROBABLE','STABLE','CONFLICTED','DEPRECATED')),
            confidence REAL NOT NULL DEFAULT 0 CHECK(confidence BETWEEN 0 AND 1),
            evidence_count INTEGER NOT NULL DEFAULT 0 CHECK(evidence_count >= 0),
            source_type TEXT NOT NULL DEFAULT 'learned',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(cloud_id) REFERENCES clouds(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS concept_kind_status_idx
            ON concepts(concept_kind, status, confidence DESC);

        CREATE TABLE IF NOT EXISTS concept_members (
            id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL,
            member_cloud_id INTEGER,
            member_lemma TEXT NOT NULL,
            member_role TEXT NOT NULL DEFAULT 'variant',
            weight REAL NOT NULL CHECK(weight BETWEEN 0 AND 1),
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            evidence_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(concept_id) REFERENCES concepts(id) ON DELETE CASCADE,
            FOREIGN KEY(member_cloud_id) REFERENCES clouds(id) ON DELETE SET NULL,
            UNIQUE(concept_id, member_lemma, member_role)
        );
        CREATE INDEX IF NOT EXISTS concept_member_lemma_idx
            ON concept_members(member_lemma, concept_id);

        CREATE TABLE IF NOT EXISTS concept_evidence (
            id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL,
            source_scene_id INTEGER,
            source_observation_id INTEGER,
            evidence_type TEXT NOT NULL,
            weight REAL NOT NULL CHECK(weight BETWEEN 0 AND 1),
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            independence_key TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(concept_id) REFERENCES concepts(id) ON DELETE CASCADE,
            FOREIGN KEY(source_scene_id) REFERENCES scenes(cloud_id) ON DELETE SET NULL,
            FOREIGN KEY(source_observation_id) REFERENCES training_observations(id) ON DELETE SET NULL,
            UNIQUE(concept_id, independence_key)
        );

        CREATE TABLE IF NOT EXISTS query_frames (
            id TEXT PRIMARY KEY,
            hive_id TEXT,
            source_text TEXT NOT NULL,
            predicate_lemma TEXT,
            requested_role TEXT,
            requested_slot TEXT,
            status TEXT NOT NULL,
            frame_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS query_frame_hive_idx
            ON query_frames(hive_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS query_role_hypotheses (
            id TEXT PRIMARY KEY,
            query_frame_id TEXT NOT NULL,
            semantic_role TEXT NOT NULL CHECK(semantic_role IN
                ('action','entity','entity_type','agent','patient','theme','object','experiencer','recipient','source',
                 'destination','location','instrument','material','cause','result','purpose',
                 'time','attribute','quantity','owner','possessed','manner')),
            grammatical_slot TEXT,
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            selected INTEGER NOT NULL DEFAULT 0 CHECK(selected IN (0,1)),
            evidence_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(query_frame_id) REFERENCES query_frames(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS query_constraints (
            id TEXT PRIMARY KEY,
            query_frame_id TEXT NOT NULL,
            role TEXT NOT NULL,
            constraint_type TEXT NOT NULL,
            value_json TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1 CHECK(confidence BETWEEN 0 AND 1),
            FOREIGN KEY(query_frame_id) REFERENCES query_frames(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS scene_matches (
            id TEXT PRIMARY KEY,
            query_frame_id TEXT NOT NULL,
            source_scene_id INTEGER NOT NULL,
            retrieval_stage TEXT NOT NULL,
            score REAL NOT NULL CHECK(score BETWEEN 0 AND 1),
            matched_roles_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            FOREIGN KEY(query_frame_id) REFERENCES query_frames(id) ON DELETE CASCADE,
            FOREIGN KEY(source_scene_id) REFERENCES scenes(cloud_id) ON DELETE CASCADE,
            UNIQUE(query_frame_id, source_scene_id, retrieval_stage)
        );

        CREATE TABLE IF NOT EXISTS pre_candidates (
            id TEXT PRIMARY KEY,
            scene_match_id TEXT NOT NULL,
            query_frame_id TEXT NOT NULL,
            entity_id INTEGER,
            target_role TEXT NOT NULL,
            value_json TEXT NOT NULL,
            score REAL NOT NULL CHECK(score BETWEEN 0 AND 1),
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(scene_match_id) REFERENCES scene_matches(id) ON DELETE CASCADE,
            FOREIGN KEY(query_frame_id) REFERENCES query_frames(id) ON DELETE CASCADE,
            FOREIGN KEY(entity_id) REFERENCES entities(cloud_id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS accepted_candidates (
            id TEXT PRIMARY KEY,
            pre_candidate_id TEXT NOT NULL UNIQUE,
            score REAL NOT NULL CHECK(score BETWEEN 0 AND 1),
            evidence_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            FOREIGN KEY(pre_candidate_id) REFERENCES pre_candidates(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS rejected_candidates (
            id TEXT PRIMARY KEY,
            pre_candidate_id TEXT NOT NULL UNIQUE,
            reason_code TEXT NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            FOREIGN KEY(pre_candidate_id) REFERENCES pre_candidates(id) ON DELETE CASCADE
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
            memory_class TEXT NOT NULL DEFAULT 'USER_ASSERTION',
            source_type TEXT NOT NULL DEFAULT 'user_assertion',
            knowledge_status TEXT NOT NULL DEFAULT 'OBSERVED',
            independent_evidence INTEGER NOT NULL DEFAULT 1 CHECK(independent_evidence IN (0,1)),
            eligible_for_fact_retrieval INTEGER NOT NULL DEFAULT 1 CHECK(eligible_for_fact_retrieval IN (0,1)),
            derived_from_json TEXT NOT NULL DEFAULT '[]',
            root_evidence_ids_json TEXT NOT NULL DEFAULT '[]',
            provenance_json TEXT NOT NULL DEFAULT '{}',
            completion_status TEXT NOT NULL DEFAULT 'COMPLETE',
            missing_supported_roles_json TEXT NOT NULL DEFAULT '[]',
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
    _migrate_concept_relation_constraints(conn)
    _migrate_query_role_hypothesis_constraints(conn)
    _add_column_if_missing(conn, "concept_relation_evidence", "concept_relation_id", "TEXT")
    _add_column_if_missing(conn, "concept_relation_evidence", "source_training_observation_id", "INTEGER")
    _add_column_if_missing(conn, "concept_relation_evidence", "evidence_type", "TEXT")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS concept_relation_evidence_source_idx "
        "ON concept_relation_evidence(concept_relation_id, source_scene_id, source_training_observation_id) "
        "WHERE concept_relation_id IS NOT NULL"
    )
    evidence_columns = {row[1] for row in conn.execute("PRAGMA table_info(semantic_evidence)").fetchall()}
    if "evidence_weight" not in evidence_columns:
        conn.execute("ALTER TABLE semantic_evidence ADD COLUMN evidence_weight REAL NOT NULL DEFAULT 0")
    if "independence" not in evidence_columns:
        conn.execute("ALTER TABLE semantic_evidence ADD COLUMN independence REAL NOT NULL DEFAULT 1")
    if "evidence_key" not in evidence_columns:
        conn.execute("ALTER TABLE semantic_evidence ADD COLUMN evidence_key TEXT")
    conn.execute("UPDATE semantic_evidence SET evidence_weight=weight WHERE evidence_weight=0 AND weight>0")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS semantic_evidence_key_idx ON semantic_evidence(evidence_key) WHERE evidence_key IS NOT NULL")
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
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS utterances (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL DEFAULT '',
            turn_index INTEGER NOT NULL DEFAULT 0,
            speaker_role TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            received_at TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT 'ru',
            source_type TEXT NOT NULL DEFAULT 'dialogue',
            parser_version TEXT NOT NULL,
            interpretation_status TEXT NOT NULL,
            message_id TEXT
        );
        CREATE INDEX IF NOT EXISTS utterance_conversation_idx
            ON utterances(conversation_id, turn_index);
        CREATE INDEX IF NOT EXISTS utterance_parser_status_idx
            ON utterances(parser_version, interpretation_status);
        CREATE UNIQUE INDEX IF NOT EXISTS utterance_message_idx
            ON utterances(message_id) WHERE message_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS dialogue_acts (
            id TEXT PRIMARY KEY,
            utterance_id TEXT NOT NULL,
            act_type TEXT NOT NULL,
            token_start INTEGER NOT NULL,
            token_end INTEGER NOT NULL,
            target_act_id TEXT,
            addressee TEXT,
            confidence REAL NOT NULL DEFAULT 0.5,
            evidence_json TEXT NOT NULL DEFAULT '[]',
            alternatives_json TEXT NOT NULL DEFAULT '[]',
            FOREIGN KEY(utterance_id) REFERENCES utterances(id) ON DELETE CASCADE,
            FOREIGN KEY(target_act_id) REFERENCES dialogue_acts(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS dialogue_act_utterance_idx
            ON dialogue_acts(utterance_id, token_start);
        CREATE INDEX IF NOT EXISTS dialogue_act_type_idx
            ON dialogue_acts(act_type, utterance_id);

        CREATE TABLE IF NOT EXISTS clauses (
            id TEXT PRIMARY KEY,
            utterance_id TEXT NOT NULL,
            sentence_index INTEGER NOT NULL,
            parent_clause_id TEXT,
            token_start INTEGER NOT NULL,
            token_end INTEGER NOT NULL,
            clause_type TEXT NOT NULL,
            relation_to_parent TEXT,
            predicate_hypotheses_json TEXT NOT NULL DEFAULT '[]',
            mode TEXT NOT NULL,
            actuality TEXT NOT NULL,
            evidence_status TEXT NOT NULL,
            polarity TEXT NOT NULL,
            negation_scope_json TEXT,
            modality TEXT,
            completion_status TEXT NOT NULL,
            temporal_anchor_json TEXT,
            speaker TEXT NOT NULL,
            quoted_speaker TEXT,
            surface TEXT NOT NULL DEFAULT '',
            evidence_json TEXT NOT NULL DEFAULT '[]',
            alternatives_json TEXT NOT NULL DEFAULT '[]',
            participants_json TEXT NOT NULL DEFAULT '[]',
            FOREIGN KEY(utterance_id) REFERENCES utterances(id) ON DELETE CASCADE,
            FOREIGN KEY(parent_clause_id) REFERENCES clauses(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS clause_utterance_idx
            ON clauses(utterance_id, sentence_index, token_start);
        CREATE INDEX IF NOT EXISTS clause_mode_actuality_idx
            ON clauses(mode, actuality, evidence_status);

        CREATE TABLE IF NOT EXISTS clause_relations (
            id TEXT PRIMARY KEY,
            source_clause_id TEXT NOT NULL,
            target_clause_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            evidence_json TEXT NOT NULL DEFAULT '[]',
            FOREIGN KEY(source_clause_id) REFERENCES clauses(id) ON DELETE CASCADE,
            FOREIGN KEY(target_clause_id) REFERENCES clauses(id) ON DELETE CASCADE,
            UNIQUE(source_clause_id, target_clause_id, relation_type)
        );
        CREATE INDEX IF NOT EXISTS clause_relation_target_idx
            ON clause_relations(target_clause_id, relation_type);

        CREATE TABLE IF NOT EXISTS interpretation_hypotheses (
            id TEXT PRIMARY KEY,
            scope_type TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            hypothesis_type TEXT NOT NULL,
            value_json TEXT NOT NULL,
            status TEXT NOT NULL,
            support_by_group_json TEXT NOT NULL DEFAULT '{}',
            support REAL NOT NULL DEFAULT 0,
            penalties_json TEXT NOT NULL DEFAULT '[]',
            constraints_json TEXT NOT NULL DEFAULT '[]',
            unresolved_slots_json TEXT NOT NULL DEFAULT '[]',
            stability_cycles INTEGER NOT NULL DEFAULT 0,
            leader_margin REAL NOT NULL DEFAULT 0,
            selected INTEGER NOT NULL DEFAULT 0,
            parser_version TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS interpretation_hypothesis_scope_idx
            ON interpretation_hypotheses(scope_type, scope_id, hypothesis_type);
        CREATE INDEX IF NOT EXISTS interpretation_hypothesis_status_idx
            ON interpretation_hypotheses(status, selected, parser_version);

        CREATE TABLE IF NOT EXISTS interpretation_evidence (
            id TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            target_hypothesis_id TEXT NOT NULL,
            value_json TEXT NOT NULL,
            support REAL NOT NULL DEFAULT 0,
            penalty REAL NOT NULL DEFAULT 0,
            evidence_type TEXT NOT NULL,
            independent_group TEXT NOT NULL,
            scope_type TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            source_token_start INTEGER,
            source_token_end INTEGER,
            source_object_id TEXT,
            parser_version TEXT NOT NULL,
            FOREIGN KEY(target_hypothesis_id)
                REFERENCES interpretation_hypotheses(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS interpretation_evidence_target_idx
            ON interpretation_evidence(target_hypothesis_id, independent_group);
        CREATE INDEX IF NOT EXISTS interpretation_evidence_source_idx
            ON interpretation_evidence(source_object_id, parser_version);
        CREATE UNIQUE INDEX IF NOT EXISTS interpretation_evidence_dedupe_idx
            ON interpretation_evidence(
                origin,scope_type,scope_id,target_hypothesis_id,evidence_type,
                COALESCE(source_token_start,-1),
                COALESCE(source_token_end,-1),parser_version
            );

        CREATE TABLE IF NOT EXISTS dialogue_states (
            conversation_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL DEFAULT '{}',
            version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS dialogue_topics (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            status TEXT NOT NULL,
            topic_json TEXT NOT NULL DEFAULT '{}',
            last_active_turn INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS dialogue_topic_conversation_idx
            ON dialogue_topics(conversation_id, status, last_active_turn);

        CREATE TABLE IF NOT EXISTS dialogue_focus_items (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            focus_rank INTEGER NOT NULL,
            role TEXT,
            value_json TEXT NOT NULL DEFAULT '{}',
            activation REAL NOT NULL DEFAULT 0,
            inertia REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS dialogue_focus_conversation_idx
            ON dialogue_focus_items(conversation_id, status, focus_rank);

        CREATE TABLE IF NOT EXISTS dialogue_pending_questions (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            query_frame_id TEXT,
            requested_role TEXT,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS pending_question_conversation_idx
            ON dialogue_pending_questions(conversation_id, status, updated_at);

        CREATE TABLE IF NOT EXISTS speaker_commitments (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            speaker_role TEXT NOT NULL,
            source_utterance_id TEXT NOT NULL,
            source_clause_id TEXT NOT NULL,
            interpretation_id TEXT NOT NULL,
            status TEXT NOT NULL,
            supersedes_commitment_id TEXT,
            content_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_utterance_id)
                REFERENCES utterances(id) ON DELETE CASCADE,
            FOREIGN KEY(source_clause_id)
                REFERENCES clauses(id) ON DELETE CASCADE,
            FOREIGN KEY(supersedes_commitment_id)
                REFERENCES speaker_commitments(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS speaker_commitment_conversation_idx
            ON speaker_commitments(conversation_id, speaker_role, status);

        CREATE TABLE IF NOT EXISTS knowledge_staging (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_key TEXT,
            raw_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            independent_source_key TEXT NOT NULL,
            conversation_id TEXT,
            speaker_role TEXT,
            parser_version TEXT NOT NULL,
            interpretation_status TEXT NOT NULL,
            interpretation_json TEXT NOT NULL DEFAULT '{}',
            validation_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'STAGED',
            supersedes_staging_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(supersedes_staging_id)
                REFERENCES knowledge_staging(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS knowledge_staging_status_idx
            ON knowledge_staging(status, parser_version, created_at);
        CREATE INDEX IF NOT EXISTS knowledge_staging_source_idx
            ON knowledge_staging(source_hash, independent_source_key);

        CREATE TABLE IF NOT EXISTS knowledge_admission_decisions (
            id TEXT PRIMARY KEY,
            staging_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            structural_valid INTEGER NOT NULL DEFAULT 0,
            factuality_valid INTEGER NOT NULL DEFAULT 0,
            source_valid INTEGER NOT NULL DEFAULT 0,
            independent_source_count INTEGER NOT NULL DEFAULT 0,
            reasons_json TEXT NOT NULL DEFAULT '[]',
            evidence_json TEXT NOT NULL DEFAULT '[]',
            materialized_objects_json TEXT NOT NULL DEFAULT '[]',
            parser_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(staging_id)
                REFERENCES knowledge_staging(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS knowledge_admission_staging_idx
            ON knowledge_admission_decisions(staging_id, decision);

        CREATE TABLE IF NOT EXISTS knowledge_retractions (
            id TEXT PRIMARY KEY,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            reason TEXT NOT NULL,
            previous_status TEXT,
            new_status TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS knowledge_retraction_target_idx
            ON knowledge_retractions(target_type, target_id, created_at);

        CREATE TABLE IF NOT EXISTS knowledge_dependencies (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            dependent_type TEXT NOT NULL,
            dependent_id TEXT NOT NULL,
            dependency_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(source_type, source_id, dependent_type, dependent_id,
                   dependency_type)
        );
        CREATE INDEX IF NOT EXISTS knowledge_dependency_source_idx
            ON knowledge_dependencies(source_type, source_id, status);
        CREATE INDEX IF NOT EXISTS knowledge_dependency_target_idx
            ON knowledge_dependencies(dependent_type, dependent_id, status);

        CREATE TABLE IF NOT EXISTS language_patterns (
            id TEXT PRIMARY KEY,
            pattern_type TEXT NOT NULL,
            signature TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'OBSERVED',
            observation_count INTEGER NOT NULL DEFAULT 1,
            independent_source_count INTEGER NOT NULL DEFAULT 1,
            evidence_json TEXT NOT NULL DEFAULT '[]',
            parser_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(pattern_type, signature, parser_version)
        );
        CREATE INDEX IF NOT EXISTS language_pattern_status_idx
            ON language_patterns(status, pattern_type, observation_count);

        CREATE TABLE IF NOT EXISTS response_plans (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            source_utterance_id TEXT,
            response_type TEXT NOT NULL,
            target_act_id TEXT,
            focus_role TEXT,
            plan_json TEXT NOT NULL,
            reverse_validation_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'PLANNED',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS response_plan_conversation_idx
            ON response_plans(conversation_id, created_at);

        CREATE TABLE IF NOT EXISTS derived_answers (
            id TEXT PRIMARY KEY,
            response_plan_id TEXT NOT NULL,
            conversation_id TEXT,
            surface_text TEXT NOT NULL,
            full_surface_text TEXT,
            source_evidence_json TEXT NOT NULL DEFAULT '[]',
            independent_source_count INTEGER NOT NULL DEFAULT 0,
            attribution_json TEXT,
            status TEXT NOT NULL DEFAULT 'DERIVED_ANSWER',
            created_at TEXT NOT NULL,
            FOREIGN KEY(response_plan_id)
                REFERENCES response_plans(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS knowledge_batches (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            config_json TEXT NOT NULL DEFAULT '{}',
            preview_json TEXT NOT NULL DEFAULT '{}',
            metrics_before_json TEXT NOT NULL DEFAULT '{}',
            metrics_after_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS clause_event_projections (
            id TEXT PRIMARY KEY,
            source_clause_id TEXT NOT NULL,
            source_scene_id INTEGER,
            confirmed_event_id TEXT,
            event_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'PROVISIONAL',
            parser_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_clause_id) REFERENCES clauses(id) ON DELETE CASCADE,
            FOREIGN KEY(source_scene_id) REFERENCES scenes(cloud_id) ON DELETE SET NULL,
            FOREIGN KEY(confirmed_event_id) REFERENCES events(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS clause_event_projection_clause_idx
            ON clause_event_projections(source_clause_id, status);
        """
    )
    _add_column_if_missing(
        conn,
        "scenes",
        "source_interpretation_id",
        "TEXT",
    )
    _add_column_if_missing(
        conn,
        "scenes",
        "admission_decision_id",
        "TEXT",
    )
    _add_column_if_missing(
        conn,
        "scenes",
        "knowledge_status",
        "TEXT NOT NULL DEFAULT 'LEGACY_CONFIRMED'",
    )
    _add_column_if_missing(
        conn,
        "events",
        "source_clause_id",
        "TEXT",
    )
    _add_column_if_missing(conn, "hive_dialogue_scenes", "memory_class", "TEXT NOT NULL DEFAULT 'USER_ASSERTION'")
    _add_column_if_missing(conn, "hive_dialogue_scenes", "source_type", "TEXT NOT NULL DEFAULT 'user_assertion'")
    _add_column_if_missing(conn, "hive_dialogue_scenes", "knowledge_status", "TEXT NOT NULL DEFAULT 'OBSERVED'")
    _add_column_if_missing(conn, "hive_dialogue_scenes", "independent_evidence", "INTEGER NOT NULL DEFAULT 1")
    _add_column_if_missing(conn, "hive_dialogue_scenes", "eligible_for_fact_retrieval", "INTEGER NOT NULL DEFAULT 1")
    _add_column_if_missing(conn, "hive_dialogue_scenes", "derived_from_json", "TEXT NOT NULL DEFAULT '[]'")
    _add_column_if_missing(conn, "hive_dialogue_scenes", "root_evidence_ids_json", "TEXT NOT NULL DEFAULT '[]'")
    _add_column_if_missing(conn, "hive_dialogue_scenes", "provenance_json", "TEXT NOT NULL DEFAULT '{}'")
    _add_column_if_missing(conn, "hive_dialogue_scenes", "completion_status", "TEXT NOT NULL DEFAULT 'COMPLETE'")
    _add_column_if_missing(conn, "hive_dialogue_scenes", "missing_supported_roles_json", "TEXT NOT NULL DEFAULT '[]'")
    _add_column_if_missing(
        conn,
        "events",
        "actuality",
        "TEXT NOT NULL DEFAULT 'ACTUAL'",
    )
    _add_column_if_missing(
        conn,
        "events",
        "evidence_status",
        "TEXT NOT NULL DEFAULT 'OBSERVED'",
    )
    _add_column_if_missing(
        conn,
        "events",
        "negation_scope_json",
        "TEXT",
    )
    _add_column_if_missing(
        conn,
        "events",
        "completion_status",
        "TEXT NOT NULL DEFAULT 'UNKNOWN'",
    )
    _add_column_if_missing(
        conn,
        "events",
        "temporal_anchor_json",
        "TEXT",
    )
    _add_column_if_missing(
        conn,
        "events",
        "attribution_json",
        "TEXT",
    )
    _add_column_if_missing(
        conn,
        "events",
        "admission_decision_id",
        "TEXT",
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS scene_interpretation_idx "
        "ON scenes(knowledge_status, source_interpretation_id, "
        "admission_decision_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS event_clause_idx "
        "ON events(source_clause_id, actuality, evidence_status)"
    )
    _backfill_v25_scenes(conn)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
