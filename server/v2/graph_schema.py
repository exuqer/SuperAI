"""Fresh V2.7 SQLite schema for role-free event graphs."""

from __future__ import annotations

import sqlite3

from .graph_models import (
    CONSTRUCTION_MODEL_VERSION,
    EVENT_SCHEMA_VERSION,
    GENERATION_VERSION,
    MIGRATION_VERSION,
    QUERY_GRAPH_VERSION,
    SEMANTIC_CLUSTER_VERSION,
    SLOT_MODEL_VERSION,
)


# Query-operator evidence is an additive extension to V2.8's graph schema.
# Keeping this version stable lets existing V2.8 databases gain the tables
# below through CREATE IF NOT EXISTS instead of being reset.
SCHEMA_VERSION = 36


def _reset_incompatible_schema(conn: sqlite3.Connection) -> None:
    tables = {
        str(row[0])
        for row in conn.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table' AND name NOT LIKE 'sqlite_%'"""
        ).fetchall()
    }
    if not tables:
        return
    compatible = False
    if "graph_meta" in tables:
        try:
            row = conn.execute(
                "SELECT value FROM graph_meta WHERE key='schema_version'"
            ).fetchone()
            compatible = bool(
                row and str(row[0]) == str(SCHEMA_VERSION)
            )
        except sqlite3.DatabaseError:
            compatible = False
    if compatible:
        return
    conn.execute("PRAGMA foreign_keys = OFF")
    views = [
        str(row[0])
        for row in conn.execute(
            """SELECT name FROM sqlite_master
               WHERE type='view' AND name NOT LIKE 'sqlite_%'"""
        ).fetchall()
    ]
    for name in views:
        conn.execute(f'DROP VIEW IF EXISTS "{name.replace(chr(34), chr(34) * 2)}"')
    for name in tables:
        conn.execute(f'DROP TABLE IF EXISTS "{name.replace(chr(34), chr(34) * 2)}"')
    conn.execute("PRAGMA foreign_keys = ON")


def reset_graph_schema(conn: sqlite3.Connection) -> None:
    """Erase every current-memory table and create a fresh role-free schema."""
    conn.execute("PRAGMA foreign_keys = OFF")
    views = [
        str(row[0])
        for row in conn.execute(
            """SELECT name FROM sqlite_master
               WHERE type='view' AND name NOT LIKE 'sqlite_%'"""
        ).fetchall()
    ]
    tables = [
        str(row[0])
        for row in conn.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table' AND name NOT LIKE 'sqlite_%'"""
        ).fetchall()
    ]
    for name in views:
        conn.execute(f'DROP VIEW IF EXISTS "{name.replace(chr(34), chr(34) * 2)}"')
    for name in tables:
        conn.execute(f'DROP TABLE IF EXISTS "{name.replace(chr(34), chr(34) * 2)}"')
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_graph_schema(conn)


def ensure_graph_schema(conn: sqlite3.Connection) -> None:
    _reset_incompatible_schema(conn)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS graph_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS knowledge_sources (
            id TEXT PRIMARY KEY,
            raw_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            source_type TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN
                ('STAGED','CONFIRMED','QUARANTINED','RETRACTED')),
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            independent_key TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS knowledge_source_identity_idx
            ON knowledge_sources(content_hash, independent_key);
        CREATE INDEX IF NOT EXISTS knowledge_source_status_idx
            ON knowledge_sources(status, created_at);

        CREATE TABLE IF NOT EXISTS graph_batches (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK(status IN
                ('PREVIEWED','COMMITTED','PARTIALLY_COMMITTED','ROLLED_BACK')),
            config_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS graph_batch_sources (
            batch_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_order INTEGER NOT NULL CHECK(source_order >= 0),
            PRIMARY KEY(batch_id, source_id),
            FOREIGN KEY(batch_id) REFERENCES graph_batches(id)
                ON DELETE CASCADE,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS graph_batch_source_order_idx
            ON graph_batch_sources(batch_id, source_order);

        CREATE TABLE IF NOT EXISTS graph_tokens (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            token_index INTEGER NOT NULL CHECK(token_index >= 0),
            sentence_index INTEGER NOT NULL DEFAULT 0 CHECK(sentence_index >= 0),
            surface TEXT NOT NULL,
            normalized TEXT NOT NULL,
            selected_hypothesis_id TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(id) ON DELETE CASCADE,
            UNIQUE(source_id, token_index)
        );
        CREATE INDEX IF NOT EXISTS graph_token_normalized_idx
            ON graph_tokens(normalized, source_id);

        CREATE TABLE IF NOT EXISTS graph_morph_hypotheses (
            id TEXT PRIMARY KEY,
            token_id TEXT NOT NULL,
            lemma TEXT NOT NULL,
            part_of_speech TEXT NOT NULL,
            features_json TEXT NOT NULL DEFAULT '{}',
            morph_score REAL NOT NULL CHECK(morph_score BETWEEN 0 AND 1),
            selected INTEGER NOT NULL DEFAULT 0 CHECK(selected IN (0,1)),
            evidence_json TEXT NOT NULL DEFAULT '[]',
            FOREIGN KEY(token_id) REFERENCES graph_tokens(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS graph_morph_lemma_idx
            ON graph_morph_hypotheses(lemma, part_of_speech);
        CREATE INDEX IF NOT EXISTS graph_morph_token_score_idx
            ON graph_morph_hypotheses(token_id, morph_score DESC);

        CREATE TABLE IF NOT EXISTS graph_entities (
            id TEXT PRIMARY KEY,
            canonical_lemma TEXT NOT NULL,
            display_surface TEXT NOT NULL,
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS graph_entity_lemma_idx
            ON graph_entities(canonical_lemma);

        CREATE TABLE IF NOT EXISTS graph_mentions (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            entity_id TEXT,
            head_lemma TEXT NOT NULL,
            head_surface TEXT NOT NULL,
            surface TEXT NOT NULL,
            qualified_key TEXT NOT NULL,
            token_start INTEGER NOT NULL,
            token_end INTEGER NOT NULL,
            token_indices_json TEXT NOT NULL,
            features_json TEXT NOT NULL DEFAULT '{}',
            components_json TEXT NOT NULL DEFAULT '[]',
            preposition TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            created_at TEXT NOT NULL,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(id) ON DELETE CASCADE,
            FOREIGN KEY(entity_id) REFERENCES graph_entities(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS graph_mention_head_idx
            ON graph_mentions(head_lemma, qualified_key);
        CREATE INDEX IF NOT EXISTS graph_mention_entity_idx
            ON graph_mentions(entity_id, source_id);
        CREATE INDEX IF NOT EXISTS graph_mention_component_idx
            ON graph_mentions(qualified_key);

        CREATE TABLE IF NOT EXISTS graph_events (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            predicate_lemma TEXT NOT NULL,
            predicate_concept_id TEXT NOT NULL,
            predicate_surface TEXT NOT NULL,
            predicate_features_json TEXT NOT NULL DEFAULT '{}',
            predicate_token_index INTEGER NOT NULL,
            construction_id TEXT,
            polarity TEXT NOT NULL DEFAULT 'POSITIVE',
            actuality TEXT NOT NULL DEFAULT 'ACTUAL',
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            properties_json TEXT NOT NULL DEFAULT '[]',
            source_surface TEXT NOT NULL,
            token_start INTEGER NOT NULL CHECK(token_start >= 0),
            token_end INTEGER NOT NULL CHECK(token_end >= token_start),
            sentence_index INTEGER NOT NULL DEFAULT 0 CHECK(sentence_index >= 0),
            event_schema_version TEXT NOT NULL,
            slot_model_version TEXT NOT NULL,
            construction_model_version TEXT NOT NULL,
            semantic_cluster_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS graph_event_predicate_idx
            ON graph_events(predicate_concept_id, confidence DESC);
        CREATE INDEX IF NOT EXISTS graph_event_predicate_lemma_idx
            ON graph_events(predicate_lemma, created_at, id);
        CREATE INDEX IF NOT EXISTS graph_event_source_idx
            ON graph_events(source_id);
        CREATE INDEX IF NOT EXISTS graph_event_construction_idx
            ON graph_events(construction_id, predicate_concept_id);

        CREATE TABLE IF NOT EXISTS graph_participants (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            mention_id TEXT NOT NULL,
            observation_signature_json TEXT NOT NULL DEFAULT '{}',
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            ordinal_hint INTEGER NOT NULL CHECK(ordinal_hint >= 0),
            created_at TEXT NOT NULL,
            FOREIGN KEY(event_id) REFERENCES graph_events(id) ON DELETE CASCADE,
            FOREIGN KEY(mention_id) REFERENCES graph_mentions(id) ON DELETE CASCADE,
            UNIQUE(event_id, mention_id)
        );
        CREATE INDEX IF NOT EXISTS graph_participant_event_idx
            ON graph_participants(event_id);
        CREATE INDEX IF NOT EXISTS graph_participant_mention_idx
            ON graph_participants(mention_id, event_id);

        CREATE TABLE IF NOT EXISTS graph_edges (
            id TEXT PRIMARY KEY,
            from_node_id TEXT NOT NULL,
            edge_type TEXT NOT NULL CHECK(edge_type IN
                ('EVENT_HAS_PARTICIPANT','MENTION_HAS_COMPONENT',
                 'VALUE_ATTACHED_TO_NODE','COREFERS_TO','EXCLUDES','CONTINUES',
                 'SUPPORTED_BY','CONTRADICTS')),
            to_node_id TEXT NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            created_at TEXT NOT NULL,
            UNIQUE(from_node_id, edge_type, to_node_id)
        );
        CREATE INDEX IF NOT EXISTS graph_edge_from_idx
            ON graph_edges(from_node_id, edge_type);
        CREATE INDEX IF NOT EXISTS graph_edge_to_idx
            ON graph_edges(to_node_id, edge_type);

        CREATE TABLE IF NOT EXISTS semantic_clusters (
            id TEXT PRIMARY KEY,
            seed_hints_json TEXT NOT NULL DEFAULT '[]',
            context_centroid_json TEXT NOT NULL DEFAULT '{}',
            support_count INTEGER NOT NULL DEFAULT 0 CHECK(support_count >= 0),
            confidence REAL NOT NULL DEFAULT 0 CHECK(confidence BETWEEN 0 AND 1),
            display_label TEXT,
            version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS semantic_cluster_members (
            semantic_cluster_id TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            compatibility REAL NOT NULL CHECK(compatibility BETWEEN 0 AND 1),
            evidence_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY(semantic_cluster_id, entity_id),
            FOREIGN KEY(semantic_cluster_id) REFERENCES semantic_clusters(id)
                ON DELETE CASCADE,
            FOREIGN KEY(entity_id) REFERENCES graph_entities(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS semantic_cluster_entity_idx
            ON semantic_cluster_members(entity_id, compatibility DESC);

        CREATE TABLE IF NOT EXISTS local_slots (
            id TEXT PRIMARY KEY,
            predicate_concept_id TEXT NOT NULL,
            centroid_signature_json TEXT NOT NULL DEFAULT '{}',
            support_count INTEGER NOT NULL DEFAULT 0 CHECK(support_count >= 0),
            contradiction_count INTEGER NOT NULL DEFAULT 0
                CHECK(contradiction_count >= 0),
            domain_diversity INTEGER NOT NULL DEFAULT 0
                CHECK(domain_diversity >= 0),
            confidence REAL NOT NULL DEFAULT 0 CHECK(confidence BETWEEN 0 AND 1),
            status TEXT NOT NULL CHECK(status IN
                ('CANDIDATE','LOCAL','STABLE','GENERALIZED','WEAKENED','DEPRECATED')),
            display_label TEXT,
            slot_model_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS local_slot_predicate_idx
            ON local_slots(predicate_concept_id, status, confidence DESC);

        CREATE TABLE IF NOT EXISTS local_slot_domains (
            local_slot_id TEXT NOT NULL,
            domain_key TEXT NOT NULL,
            observation_count INTEGER NOT NULL DEFAULT 1
                CHECK(observation_count > 0),
            PRIMARY KEY(local_slot_id, domain_key),
            FOREIGN KEY(local_slot_id) REFERENCES local_slots(id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS participant_slot_hypotheses (
            participant_id TEXT NOT NULL,
            local_slot_id TEXT NOT NULL,
            compatibility REAL NOT NULL CHECK(compatibility BETWEEN 0 AND 1),
            selected INTEGER NOT NULL DEFAULT 0 CHECK(selected IN (0,1)),
            evidence_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            PRIMARY KEY(participant_id, local_slot_id),
            FOREIGN KEY(participant_id) REFERENCES graph_participants(id)
                ON DELETE CASCADE,
            FOREIGN KEY(local_slot_id) REFERENCES local_slots(id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS participant_slot_lookup_idx
            ON participant_slot_hypotheses(local_slot_id, compatibility DESC);

        CREATE TABLE IF NOT EXISTS slot_sets (
            id TEXT PRIMARY KEY,
            predicate_concept_id TEXT NOT NULL,
            support_count INTEGER NOT NULL DEFAULT 0 CHECK(support_count >= 0),
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            status TEXT NOT NULL CHECK(status IN
                ('CANDIDATE','LOCAL','STABLE','GENERALIZED','WEAKENED','DEPRECATED')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS slot_set_members (
            slot_set_id TEXT NOT NULL,
            local_slot_id TEXT NOT NULL,
            PRIMARY KEY(slot_set_id, local_slot_id),
            FOREIGN KEY(slot_set_id) REFERENCES slot_sets(id) ON DELETE CASCADE,
            FOREIGN KEY(local_slot_id) REFERENCES local_slots(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS slot_set_predicate_idx
            ON slot_sets(predicate_concept_id, confidence DESC);

        CREATE TABLE IF NOT EXISTS slot_prototypes (
            id TEXT PRIMARY KEY,
            centroid_signature_json TEXT NOT NULL DEFAULT '{}',
            support_count INTEGER NOT NULL DEFAULT 0 CHECK(support_count >= 0),
            domain_diversity INTEGER NOT NULL DEFAULT 0
                CHECK(domain_diversity >= 0),
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            display_label TEXT,
            slot_model_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS slot_prototype_members (
            prototype_id TEXT NOT NULL,
            local_slot_id TEXT NOT NULL,
            compatibility REAL NOT NULL CHECK(compatibility BETWEEN 0 AND 1),
            PRIMARY KEY(prototype_id, local_slot_id),
            FOREIGN KEY(prototype_id) REFERENCES slot_prototypes(id)
                ON DELETE CASCADE,
            FOREIGN KEY(local_slot_id) REFERENCES local_slots(id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS slot_prototype_slot_idx
            ON slot_prototype_members(local_slot_id, compatibility DESC);

        CREATE TABLE IF NOT EXISTS construction_clusters (
            id TEXT PRIMARY KEY,
            structural_signature_json TEXT NOT NULL DEFAULT '{}',
            gap_kind TEXT CHECK(gap_kind IS NULL OR gap_kind IN
                ('EVENT_ATTACHMENT','NODE_COMPONENT','RELATION_VALUE',
                 'EVENT_PROPERTY','BOOLEAN_RESULT','QUANTITY_VALUE','WHOLE_EVENT')),
            support_count INTEGER NOT NULL DEFAULT 0 CHECK(support_count >= 0),
            contradiction_count INTEGER NOT NULL DEFAULT 0
                CHECK(contradiction_count >= 0),
            domain_diversity INTEGER NOT NULL DEFAULT 0
                CHECK(domain_diversity >= 0),
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            status TEXT NOT NULL CHECK(status IN
                ('CANDIDATE','LOCAL','STABLE','GENERALIZED','WEAKENED','DEPRECATED')),
            construction_model_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS construction_gap_idx
            ON construction_clusters(gap_kind, confidence DESC);

        CREATE TABLE IF NOT EXISTS construction_evidence (
            construction_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            structural_signature_json TEXT NOT NULL DEFAULT '{}',
            domain_key TEXT NOT NULL,
            contradicted INTEGER NOT NULL DEFAULT 0 CHECK(contradicted IN (0,1)),
            created_at TEXT NOT NULL,
            PRIMARY KEY(construction_id, source_id),
            FOREIGN KEY(construction_id) REFERENCES construction_clusters(id)
                ON DELETE CASCADE,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS construction_slot_compatibility (
            construction_id TEXT NOT NULL,
            prototype_id TEXT NOT NULL,
            compatibility REAL NOT NULL CHECK(compatibility BETWEEN 0 AND 1),
            support_count INTEGER NOT NULL DEFAULT 0 CHECK(support_count >= 0),
            contradiction_count INTEGER NOT NULL DEFAULT 0
                CHECK(contradiction_count >= 0),
            PRIMARY KEY(construction_id, prototype_id),
            FOREIGN KEY(construction_id) REFERENCES construction_clusters(id)
                ON DELETE CASCADE,
            FOREIGN KEY(prototype_id) REFERENCES slot_prototypes(id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS predicate_perspective_relations (
            source_predicate_concept_id TEXT NOT NULL,
            target_predicate_concept_id TEXT NOT NULL,
            slot_permutation_json TEXT NOT NULL DEFAULT '{}',
            evidence_count INTEGER NOT NULL DEFAULT 0 CHECK(evidence_count >= 0),
            confidence REAL NOT NULL DEFAULT 0 CHECK(confidence BETWEEN 0 AND 1),
            context_support_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(source_predicate_concept_id,target_predicate_concept_id)
        );
        CREATE INDEX IF NOT EXISTS predicate_perspective_target_idx
            ON predicate_perspective_relations(
                target_predicate_concept_id, confidence DESC
            );

        CREATE TABLE IF NOT EXISTS hives (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            max_cells INTEGER NOT NULL DEFAULT 24 CHECK(max_cells > 0),
            active_query_graph_id TEXT,
            state_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS hive_conversation_idx
            ON hives(conversation_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS query_graphs (
            id TEXT PRIMARY KEY,
            hive_id TEXT,
            source_text TEXT NOT NULL,
            graph_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN
                ('READY','AMBIGUOUS','INCOMPLETE','CONFLICTED')),
            continuation_of TEXT,
            query_graph_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(hive_id) REFERENCES hives(id) ON DELETE CASCADE,
            FOREIGN KEY(continuation_of) REFERENCES query_graphs(id)
                ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS query_graph_hive_idx
            ON query_graphs(hive_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS candidate_bindings (
            id TEXT PRIMARY KEY,
            query_graph_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            gap_node_id TEXT NOT NULL,
            resolved_node_id TEXT NOT NULL,
            resolved_concept_id TEXT NOT NULL,
            resolved_lemma TEXT NOT NULL,
            resolved_surface TEXT NOT NULL,
            resolved_features_json TEXT NOT NULL DEFAULT '{}',
            structural_score REAL NOT NULL CHECK(structural_score BETWEEN 0 AND 1),
            signature_score REAL NOT NULL CHECK(signature_score BETWEEN 0 AND 1),
            evidence_score REAL NOT NULL CHECK(evidence_score BETWEEN 0 AND 1),
            total_score REAL NOT NULL CHECK(total_score BETWEEN 0 AND 1),
            status TEXT NOT NULL CHECK(status IN
                ('CANDIDATE','ACCEPTED','REJECTED','SELECTED')),
            failed_constraint TEXT,
            evidence_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            FOREIGN KEY(query_graph_id) REFERENCES query_graphs(id)
                ON DELETE CASCADE,
            FOREIGN KEY(event_id) REFERENCES graph_events(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS binding_query_score_idx
            ON candidate_bindings(query_graph_id, status, total_score DESC);
        CREATE INDEX IF NOT EXISTS binding_gap_query_score_idx
            ON candidate_bindings(query_graph_id, gap_node_id, status, total_score DESC);
        CREATE INDEX IF NOT EXISTS binding_event_idx
            ON candidate_bindings(event_id, query_graph_id);

        /* Search is auditable: a swarm proposes events, while GraphMatcher
           remains the only component that admits a binding. */
        CREATE TABLE IF NOT EXISTS swarm_runs (
            id TEXT PRIMARY KEY,
            query_graph_id TEXT NOT NULL,
            gap_id TEXT NOT NULL,
            deterministic_seed TEXT NOT NULL,
            status TEXT NOT NULL,
            termination_reason TEXT NOT NULL,
            retrieval_mode TEXT NOT NULL,
            budget_json TEXT NOT NULL DEFAULT '{}',
            trace_json TEXT NOT NULL DEFAULT '{}',
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS swarm_run_query_idx
            ON swarm_runs(query_graph_id, gap_id, started_at DESC);

        CREATE TABLE IF NOT EXISTS bee_missions (
            id TEXT PRIMARY KEY,
            swarm_run_id TEXT NOT NULL,
            bee_type TEXT NOT NULL,
            mission_type TEXT NOT NULL,
            seed_json TEXT NOT NULL DEFAULT '{}',
            visited_universes_json TEXT NOT NULL DEFAULT '[]',
            candidate_event_ids_json TEXT NOT NULL DEFAULT '[]',
            successful INTEGER NOT NULL DEFAULT 0 CHECK(successful IN (0,1)),
            termination_reason TEXT NOT NULL,
            FOREIGN KEY(swarm_run_id) REFERENCES swarm_runs(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS bee_steps (
            id TEXT PRIMARY KEY,
            swarm_run_id TEXT NOT NULL,
            bee_id TEXT NOT NULL,
            step_index INTEGER NOT NULL CHECK(step_index >= 0),
            source_universe TEXT NOT NULL,
            target_universe TEXT NOT NULL,
            action TEXT NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(swarm_run_id) REFERENCES swarm_runs(id) ON DELETE CASCADE,
            UNIQUE(swarm_run_id,bee_id,step_index)
        );
        CREATE TABLE IF NOT EXISTS nectar_packets (
            id TEXT PRIMARY KEY,
            swarm_run_id TEXT NOT NULL,
            source_universe TEXT NOT NULL,
            target_universe TEXT NOT NULL,
            event_ids_json TEXT NOT NULL DEFAULT '[]',
            dimension_ids_json TEXT NOT NULL DEFAULT '[]',
            evidence_weight REAL NOT NULL DEFAULT 0,
            provenance_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(swarm_run_id) REFERENCES swarm_runs(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS candidate_event_observations (
            id TEXT PRIMARY KEY,
            swarm_run_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            source_mode TEXT NOT NULL,
            evidence_weight REAL NOT NULL DEFAULT 0,
            admitted INTEGER,
            rejection_reason TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(swarm_run_id) REFERENCES swarm_runs(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS bee_mission_run_idx
            ON bee_missions(swarm_run_id);
        CREATE INDEX IF NOT EXISTS bee_step_run_idx
            ON bee_steps(swarm_run_id);
        CREATE INDEX IF NOT EXISTS nectar_packet_run_idx
            ON nectar_packets(swarm_run_id);
        CREATE INDEX IF NOT EXISTS candidate_observation_run_idx
            ON candidate_event_observations(swarm_run_id);

        CREATE TABLE IF NOT EXISTS binding_configurations (
            id TEXT PRIMARY KEY,
            query_graph_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            bindings_json TEXT NOT NULL DEFAULT '[]',
            all_required_gaps_bound INTEGER NOT NULL CHECK(all_required_gaps_bound IN (0,1)),
            distinct_node_count INTEGER NOT NULL DEFAULT 0,
            configuration_score REAL NOT NULL DEFAULT 0,
            validation_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS binding_configuration_query_idx
            ON binding_configurations(query_graph_id, configuration_score DESC);
        CREATE INDEX IF NOT EXISTS binding_configuration_event_idx
            ON binding_configurations(event_id, status, configuration_score DESC);

        CREATE TABLE IF NOT EXISTS dialogue_turns (
            id TEXT PRIMARY KEY,
            hive_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL CHECK(turn_index >= 0),
            speaker TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            query_graph_id TEXT,
            selected_bindings_json TEXT NOT NULL DEFAULT '[]',
            binding_configuration_id TEXT,
            answer_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(hive_id) REFERENCES hives(id) ON DELETE CASCADE,
            FOREIGN KEY(query_graph_id) REFERENCES query_graphs(id)
                ON DELETE SET NULL,
            UNIQUE(hive_id, turn_index)
        );
        CREATE INDEX IF NOT EXISTS dialogue_turn_hive_idx
            ON dialogue_turns(hive_id, turn_index DESC);

        CREATE TABLE IF NOT EXISTS training_episodes (
            id TEXT PRIMARY KEY,
            utterance TEXT NOT NULL,
            query_graph_id TEXT NOT NULL,
            candidate_bindings_json TEXT NOT NULL DEFAULT '[]',
            selected_bindings_json TEXT NOT NULL DEFAULT '[]',
            binding_configuration_id TEXT,
            event_ids_json TEXT NOT NULL DEFAULT '[]',
            construction_ids_json TEXT NOT NULL DEFAULT '[]',
            slot_hypotheses_json TEXT NOT NULL DEFAULT '[]',
            answer_status TEXT NOT NULL,
            validation_json TEXT NOT NULL DEFAULT '{}',
            user_correction_json TEXT,
            eligible_for_learning INTEGER NOT NULL DEFAULT 0 CHECK(eligible_for_learning IN (0,1)),
            created_at TEXT NOT NULL,
            FOREIGN KEY(query_graph_id) REFERENCES query_graphs(id)
                ON DELETE CASCADE
        );

        /* Query operators are learned from concrete, validated uses.  These
           tables contain no named participant roles or answer-type labels:
           profiles retain only observable projections, local slot evidence
           and the history required to evaluate a shadow prediction. */
        CREATE TABLE IF NOT EXISTS query_operator_profiles (
            id TEXT PRIMARY KEY,
            profile_key TEXT NOT NULL UNIQUE,
            projections_json TEXT NOT NULL DEFAULT '{}',
            compatible_slots_json TEXT NOT NULL DEFAULT '{}',
            support_count INTEGER NOT NULL DEFAULT 0 CHECK(support_count >= 0),
            validated_count INTEGER NOT NULL DEFAULT 0 CHECK(validated_count >= 0),
            rejected_count INTEGER NOT NULL DEFAULT 0 CHECK(rejected_count >= 0),
            confidence REAL NOT NULL DEFAULT 0 CHECK(confidence BETWEEN 0 AND 1),
            status TEXT NOT NULL CHECK(status IN ('SHADOW','LEARNED')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS query_operator_profile_status_idx
            ON query_operator_profiles(status, confidence DESC);

        CREATE TABLE IF NOT EXISTS query_operator_occurrences (
            id TEXT PRIMARY KEY,
            query_graph_id TEXT NOT NULL,
            gap_node_id TEXT NOT NULL,
            profile_id TEXT,
            operator_surface TEXT NOT NULL,
            operator_normalized TEXT NOT NULL,
            token_indices_json TEXT NOT NULL DEFAULT '[]',
            context_json TEXT NOT NULL DEFAULT '{}',
            prediction_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL CHECK(status IN
                ('OBSERVED','OBSERVED_UNTRUSTED','VALIDATED','REJECTED')),
            created_at TEXT NOT NULL,
            FOREIGN KEY(query_graph_id) REFERENCES query_graphs(id)
                ON DELETE CASCADE,
            FOREIGN KEY(profile_id) REFERENCES query_operator_profiles(id)
                ON DELETE SET NULL,
            UNIQUE(query_graph_id, gap_node_id)
        );
        CREATE INDEX IF NOT EXISTS query_operator_occurrence_profile_idx
            ON query_operator_occurrences(profile_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS query_operator_experiences (
            id TEXT PRIMARY KEY,
            occurrence_id TEXT NOT NULL,
            profile_id TEXT,
            outcome TEXT NOT NULL CHECK(outcome IN
                ('OBSERVED_UNTRUSTED','VALIDATED_BINDING','REJECTED_BINDING',
                 'UNSELECTED_CANDIDATE','REJECTED_EVENT')),
            validated INTEGER NOT NULL DEFAULT 0 CHECK(validated IN (0,1)),
            binding_json TEXT NOT NULL DEFAULT '{}',
            rejection_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(occurrence_id) REFERENCES query_operator_occurrences(id)
                ON DELETE CASCADE,
            FOREIGN KEY(profile_id) REFERENCES query_operator_profiles(id)
                ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS query_operator_experience_occurrence_idx
            ON query_operator_experiences(occurrence_id, created_at);

        /* A universe is deliberately generic.  Names are UI metadata, not
           semantic instructions for the learner. */
        CREATE TABLE IF NOT EXISTS universes (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            scale TEXT NOT NULL,
            version TEXT NOT NULL,
            base_space_config_json TEXT NOT NULL DEFAULT '{}',
            discovery_config_json TEXT NOT NULL DEFAULT '{}',
            statistics_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS universe_entities (
            id TEXT PRIMARY KEY,
            universe_id TEXT NOT NULL,
            observable_key TEXT NOT NULL,
            display_value TEXT NOT NULL,
            prototype_vector_json TEXT NOT NULL DEFAULT '{}',
            base_position_json TEXT NOT NULL DEFAULT '[]',
            mass REAL NOT NULL DEFAULT 1,
            gravity REAL NOT NULL DEFAULT 0,
            stability REAL NOT NULL DEFAULT 0,
            frequency INTEGER NOT NULL DEFAULT 0,
            dispersion REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(universe_id) REFERENCES universes(id) ON DELETE CASCADE,
            UNIQUE(universe_id, observable_key)
        );
        CREATE INDEX IF NOT EXISTS universe_entity_lookup_idx
            ON universe_entities(universe_id, frequency DESC, observable_key);

        CREATE TABLE IF NOT EXISTS universe_occurrences (
            id TEXT PRIMARY KEY,
            universe_id TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            parent_occurrence_id TEXT,
            context_id TEXT NOT NULL,
            observable_features_json TEXT NOT NULL DEFAULT '{}',
            context_vector_json TEXT NOT NULL DEFAULT '{}',
            base_position_json TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0 AND 1),
            created_at TEXT NOT NULL,
            FOREIGN KEY(universe_id) REFERENCES universes(id) ON DELETE CASCADE,
            FOREIGN KEY(entity_id) REFERENCES universe_entities(id) ON DELETE CASCADE,
            FOREIGN KEY(parent_occurrence_id) REFERENCES universe_occurrences(id)
                ON DELETE SET NULL,
            UNIQUE(universe_id, source_id, context_id, entity_id)
        );
        CREATE INDEX IF NOT EXISTS universe_occurrence_entity_idx
            ON universe_occurrences(entity_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS universe_occurrence_source_idx
            ON universe_occurrences(universe_id, source_id);

        /* A Word entity is a lexeme.  Surface forms and their concrete
           usages are kept below it so inflection cannot split its mass or
           its learned context. */
        CREATE TABLE IF NOT EXISTS lexemes (
            lexeme_entity_id TEXT PRIMARY KEY,
            language TEXT NOT NULL,
            canonical_lemma TEXT NOT NULL,
            sense_cluster_id TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(lexeme_entity_id) REFERENCES universe_entities(id)
                ON DELETE CASCADE,
            UNIQUE(language, canonical_lemma, sense_cluster_id)
        );
        CREATE INDEX IF NOT EXISTS lexeme_lookup_idx
            ON lexemes(language, canonical_lemma, sense_cluster_id);

        CREATE TABLE IF NOT EXISTS word_forms (
            word_form_entity_id TEXT PRIMARY KEY,
            lexeme_entity_id TEXT NOT NULL,
            language TEXT NOT NULL,
            normalized_surface TEXT NOT NULL,
            display_surface TEXT NOT NULL,
            morphological_features_json TEXT NOT NULL DEFAULT '{}',
            morphology_confidence REAL NOT NULL DEFAULT 0.5
                CHECK(morphology_confidence BETWEEN 0 AND 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(word_form_entity_id) REFERENCES universe_entities(id)
                ON DELETE CASCADE,
            FOREIGN KEY(lexeme_entity_id) REFERENCES universe_entities(id)
                ON DELETE CASCADE,
            UNIQUE(lexeme_entity_id, normalized_surface)
        );
        CREATE INDEX IF NOT EXISTS word_form_lexeme_idx
            ON word_forms(lexeme_entity_id, normalized_surface);

        CREATE TABLE IF NOT EXISTS word_usages (
            usage_occurrence_id TEXT PRIMARY KEY,
            lexeme_entity_id TEXT NOT NULL,
            word_form_entity_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            sentence_index INTEGER NOT NULL CHECK(sentence_index >= 0),
            token_index INTEGER NOT NULL CHECK(token_index >= 0),
            created_at TEXT NOT NULL,
            FOREIGN KEY(usage_occurrence_id) REFERENCES universe_occurrences(id)
                ON DELETE CASCADE,
            FOREIGN KEY(lexeme_entity_id) REFERENCES universe_entities(id)
                ON DELETE CASCADE,
            FOREIGN KEY(word_form_entity_id) REFERENCES word_forms(word_form_entity_id)
                ON DELETE CASCADE,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(id)
                ON DELETE CASCADE,
            UNIQUE(source_id, token_index)
        );
        CREATE INDEX IF NOT EXISTS word_usage_lexeme_idx
            ON word_usages(lexeme_entity_id, source_id, token_index);
        CREATE INDEX IF NOT EXISTS word_usage_form_idx
            ON word_usages(word_form_entity_id, source_id, token_index);

        CREATE TABLE IF NOT EXISTS entity_clouds (
            id TEXT PRIMARY KEY,
            universe_id TEXT NOT NULL,
            core_vector_json TEXT NOT NULL DEFAULT '{}',
            core_entity_ids_json TEXT NOT NULL DEFAULT '[]',
            mass REAL NOT NULL DEFAULT 0,
            gravity REAL NOT NULL DEFAULT 0,
            radius REAL NOT NULL DEFAULT 1,
            density REAL NOT NULL DEFAULT 0,
            dispersion REAL NOT NULL DEFAULT 0,
            stability REAL NOT NULL DEFAULT 0,
            member_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL CHECK(status IN ('candidate','active','weak','pruned')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(universe_id) REFERENCES universes(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS entity_cloud_universe_idx
            ON entity_clouds(universe_id, status, stability DESC);

        CREATE TABLE IF NOT EXISTS cloud_memberships (
            cloud_id TEXT NOT NULL,
            source_type TEXT NOT NULL CHECK(source_type IN ('entity','occurrence')),
            source_id TEXT NOT NULL,
            membership REAL NOT NULL CHECK(membership BETWEEN 0 AND 1),
            radial_distance REAL NOT NULL DEFAULT 0,
            confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0 AND 1),
            PRIMARY KEY(cloud_id, source_type, source_id),
            FOREIGN KEY(cloud_id) REFERENCES entity_clouds(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS cloud_membership_source_idx
            ON cloud_memberships(source_type, source_id, membership DESC);

        CREATE TABLE IF NOT EXISTS latent_dimensions (
            id TEXT PRIMARY KEY,
            canonical_dimension_id TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1 CHECK(revision >= 1),
            universe_id TEXT NOT NULL,
            owner_scope TEXT NOT NULL CHECK(owner_scope IN ('universe','cloud','local')),
            owner_id TEXT,
            representation_type TEXT NOT NULL CHECK(representation_type IN
                ('axis','subspace','cloud','multi_core','manifold','field')),
            basis_json TEXT NOT NULL DEFAULT '{}',
            dimensionality INTEGER NOT NULL DEFAULT 1 CHECK(dimensionality >= 1),
            strength REAL NOT NULL DEFAULT 0,
            stability REAL NOT NULL DEFAULT 0,
            predictive_gain REAL NOT NULL DEFAULT 0,
            retrieval_gain REAL NOT NULL DEFAULT 0,
            compression_gain REAL NOT NULL DEFAULT 0,
            memory_cost REAL NOT NULL DEFAULT 0,
            usage_count INTEGER NOT NULL DEFAULT 0,
            projection_usage_count INTEGER NOT NULL DEFAULT 0,
            retrieval_contribution_count INTEGER NOT NULL DEFAULT 0,
            graph_admitted_contribution_count INTEGER NOT NULL DEFAULT 0,
            validated_answer_contribution_count INTEGER NOT NULL DEFAULT 0,
            evidence_count INTEGER NOT NULL DEFAULT 0,
            entity_support INTEGER NOT NULL DEFAULT 0,
            source_support INTEGER NOT NULL DEFAULT 0,
            domain_support INTEGER NOT NULL DEFAULT 0,
            train_support INTEGER NOT NULL DEFAULT 0,
            holdout_support INTEGER NOT NULL DEFAULT 0,
            continual_support INTEGER NOT NULL DEFAULT 0,
            stability_lower_bound REAL NOT NULL DEFAULT 0,
            holdout_retrieval_gain REAL NOT NULL DEFAULT 0,
            shadow_retrieval_gain REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL CHECK(status IN
                ('candidate','probation','active','shared','weak','merged','split',
                 'pruned','frozen')),
            created_at TEXT NOT NULL,
            activated_at TEXT,
            last_updated_at TEXT NOT NULL,
            last_confirmed_at TEXT,
            FOREIGN KEY(universe_id) REFERENCES universes(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS latent_dimension_universe_idx
            ON latent_dimensions(universe_id, status, stability DESC);
        CREATE INDEX IF NOT EXISTS latent_dimension_canonical_idx
            ON latent_dimensions(canonical_dimension_id, revision DESC);

        CREATE TABLE IF NOT EXISTS dimension_history (
            id TEXT PRIMARY KEY,
            dimension_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            status TEXT NOT NULL,
            snapshot_json TEXT NOT NULL DEFAULT '{}',
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(dimension_id) REFERENCES latent_dimensions(id)
                ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS dimension_lineage (
            canonical_dimension_id TEXT PRIMARY KEY,
            current_revision_id TEXT NOT NULL,
            parent_dimension_ids_json TEXT NOT NULL DEFAULT '[]',
            merged_from_json TEXT NOT NULL DEFAULT '[]',
            split_from_json TEXT NOT NULL DEFAULT '[]',
            replaced_by TEXT,
            lineage_reason TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS dimension_evaluations (
            id TEXT PRIMARY KEY,
            dimension_id TEXT NOT NULL,
            dataset_split TEXT NOT NULL,
            entity_support INTEGER NOT NULL DEFAULT 0,
            source_support INTEGER NOT NULL DEFAULT 0,
            domain_support INTEGER NOT NULL DEFAULT 0,
            stability_point_estimate REAL NOT NULL DEFAULT 0,
            stability_lower_bound REAL NOT NULL DEFAULT 0,
            retrieval_gain REAL NOT NULL DEFAULT 0,
            metrics_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(dimension_id) REFERENCES latent_dimensions(id)
                ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS shadow_retrieval_runs (
            id TEXT PRIMARY KEY,
            query_graph_id TEXT NOT NULL,
            dimension_id TEXT NOT NULL,
            baseline_event_ids_json TEXT NOT NULL DEFAULT '[]',
            shadow_candidate_events_json TEXT NOT NULL DEFAULT '[]',
            shadow_graph_admitted_events_json TEXT NOT NULL DEFAULT '[]',
            shadow_correct_event_rank INTEGER,
            shadow_retrieval_gain REAL NOT NULL DEFAULT 0,
            shadow_false_positive_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(dimension_id) REFERENCES latent_dimensions(id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS dimension_clouds (
            id TEXT PRIMARY KEY,
            dimension_id TEXT NOT NULL,
            core_vector_json TEXT NOT NULL DEFAULT '{}',
            positive_core_json TEXT,
            negative_core_json TEXT,
            boundary_region_json TEXT NOT NULL DEFAULT '{}',
            applicability_region_json TEXT NOT NULL DEFAULT '{}',
            mass REAL NOT NULL DEFAULT 0,
            gravity REAL NOT NULL DEFAULT 0,
            radius REAL NOT NULL DEFAULT 1,
            density REAL NOT NULL DEFAULT 0,
            stability REAL NOT NULL DEFAULT 0,
            FOREIGN KEY(dimension_id) REFERENCES latent_dimensions(id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS projections (
            id TEXT PRIMARY KEY,
            dimension_id TEXT NOT NULL,
            source_type TEXT NOT NULL CHECK(source_type IN ('entity','occurrence')),
            source_id TEXT NOT NULL,
            context_id TEXT,
            coordinates_json TEXT NOT NULL DEFAULT '[]',
            membership REAL NOT NULL CHECK(membership BETWEEN 0 AND 1),
            distance_to_core REAL NOT NULL DEFAULT 1,
            confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0 AND 1),
            calculated_at TEXT NOT NULL,
            FOREIGN KEY(dimension_id) REFERENCES latent_dimensions(id)
                ON DELETE CASCADE,
            UNIQUE(dimension_id, source_type, source_id, context_id)
        );
        CREATE INDEX IF NOT EXISTS projection_dimension_idx
            ON projections(dimension_id, source_type, membership DESC);
        CREATE INDEX IF NOT EXISTS projection_source_idx
            ON projections(source_type, source_id, membership DESC);
        CREATE INDEX IF NOT EXISTS projection_source_dimension_idx
            ON projections(source_type, source_id, dimension_id, membership DESC);

        CREATE TABLE IF NOT EXISTS dimension_relations (
            source_dimension_id TEXT NOT NULL,
            target_dimension_id TEXT NOT NULL,
            relation_type TEXT NOT NULL CHECK(relation_type IN
                ('similar','overlapping','nested','conditional','correlated',
                 'competing','independent','merged_from')),
            weight REAL NOT NULL CHECK(weight BETWEEN 0 AND 1),
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            evidence_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(source_dimension_id, target_dimension_id, relation_type),
            FOREIGN KEY(source_dimension_id) REFERENCES latent_dimensions(id)
                ON DELETE CASCADE,
            FOREIGN KEY(target_dimension_id) REFERENCES latent_dimensions(id)
                ON DELETE CASCADE
        );

        /* User-facing aliases never participate in discovery or retrieval. */
        CREATE TABLE IF NOT EXISTS dimension_aliases (
            dimension_id TEXT PRIMARY KEY,
            alias TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(dimension_id) REFERENCES latent_dimensions(id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS universe_transitions (
            id TEXT PRIMARY KEY,
            source_universe_id TEXT NOT NULL,
            target_universe_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            weight REAL NOT NULL CHECK(weight BETWEEN 0 AND 1),
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            context_id TEXT,
            evidence_count INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_universe_id) REFERENCES universes(id) ON DELETE CASCADE,
            FOREIGN KEY(target_universe_id) REFERENCES universes(id) ON DELETE CASCADE,
            UNIQUE(source_universe_id,target_universe_id,source_type,source_id,
                   target_type,target_id,context_id)
        );
        CREATE INDEX IF NOT EXISTS universe_transition_source_idx
            ON universe_transitions(source_universe_id, source_id, weight DESC);
        CREATE INDEX IF NOT EXISTS universe_transition_target_idx
            ON universe_transitions(target_universe_id, target_id, weight DESC);
        CREATE INDEX IF NOT EXISTS universe_transition_context_idx
            ON universe_transitions(context_id);

        CREATE TABLE IF NOT EXISTS universe_training_events (
            id TEXT PRIMARY KEY,
            universe_id TEXT NOT NULL,
            event_type TEXT NOT NULL CHECK(event_type IN
                ('entity_created','cloud_created','cloud_expanded','cloud_split',
                 'cloud_merged','dimension_candidate_created','dimension_activated',
                 'dimension_shared','dimension_merged','dimension_pruned',
                 'transition_created')),
            subject_id TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(universe_id) REFERENCES universes(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS universe_training_event_idx
            ON universe_training_events(universe_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS experiment_runs (
            id TEXT PRIMARY KEY,
            dataset_version TEXT NOT NULL,
            dataset_split TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            pipeline_versions_json TEXT NOT NULL DEFAULT '{}',
            configuration_hash TEXT NOT NULL,
            random_seed INTEGER NOT NULL,
            training_order_json TEXT NOT NULL DEFAULT '[]',
            batch_boundaries_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL,
            report_json TEXT NOT NULL DEFAULT '{}',
            started_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS experiment_reproducibility_idx
            ON experiment_runs(dataset_version,configuration_hash,random_seed);
        CREATE TABLE IF NOT EXISTS experiment_metrics (
            id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            tolerance REAL NOT NULL DEFAULT 0,
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(experiment_id) REFERENCES experiment_runs(id)
                ON DELETE CASCADE
        );

        /* EventBindingFrame: stable dialogue projection of a selected event.
           Survives multiple short turns and tracks how participants relate
           to questions and bindings across the dialogue. */
        CREATE TABLE IF NOT EXISTS event_binding_frames (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            root_query_graph_id TEXT NOT NULL,
            latest_query_graph_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            predicate_concept_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('ACTIVE','WEAK','CLOSED')),
            confidence REAL NOT NULL DEFAULT 0 CHECK(confidence BETWEEN 0 AND 1),
            state_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS event_binding_frame_conversation_idx
            ON event_binding_frames(conversation_id, status);
        CREATE INDEX IF NOT EXISTS event_binding_frame_event_idx
            ON event_binding_frames(event_id);

        CREATE TABLE IF NOT EXISTS event_binding_frame_participants (
            id TEXT PRIMARY KEY,
            frame_id TEXT NOT NULL,
            participant_node_id TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            resolved_lemma TEXT NOT NULL,
            canonical_surface TEXT NOT NULL,
            morphology_json TEXT NOT NULL DEFAULT '{}',
            origin TEXT NOT NULL CHECK(origin IN
                ('EXPLICIT_ROOT_QUERY','RESOLVED_ROOT_GAP',
                 'RESOLVED_LATER_GAP','INFERRED_EVENT_PARTICIPANT')),
            lineage_root_gap_id TEXT,
            latest_source_gap_id TEXT,
            latest_source_binding_id TEXT,
            source_query_graph_ids_json TEXT NOT NULL DEFAULT '[]',
            local_slot_ids_json TEXT NOT NULL DEFAULT '[]',
            observed_question_profiles_json TEXT NOT NULL DEFAULT '[]',
            compatible_question_profiles_json TEXT NOT NULL DEFAULT '{}',
            binding_confidence REAL NOT NULL DEFAULT 0,
            replaceable INTEGER NOT NULL DEFAULT 1 CHECK(replaceable IN (0,1)),
            last_released_turn INTEGER,
            last_selected_turn INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(frame_id) REFERENCES event_binding_frames(id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS event_binding_frame_participant_frame_idx
            ON event_binding_frame_participants(frame_id);
        CREATE INDEX IF NOT EXISTS event_binding_frame_participant_node_idx
            ON event_binding_frame_participants(participant_node_id);
        CREATE INDEX IF NOT EXISTS event_binding_frame_participant_concept_idx
            ON event_binding_frame_participants(concept_id);

        CREATE TABLE IF NOT EXISTS event_binding_frame_question_profiles (
            frame_participant_id TEXT NOT NULL,
            family_key TEXT NOT NULL,
            question_surface TEXT NOT NULL,
            morphology_signature_json TEXT NOT NULL DEFAULT '{}',
            confidence REAL NOT NULL DEFAULT 0,
            observation_count INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY(frame_participant_id, family_key, question_surface),
            FOREIGN KEY(frame_participant_id)
                REFERENCES event_binding_frame_participants(id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS event_binding_frame_qp_family_idx
            ON event_binding_frame_question_profiles(frame_participant_id, family_key);

        /* Query interpretation hypotheses: competing ways to interpret
           a short turn (continuation, standalone query, event rebind). */
        CREATE TABLE IF NOT EXISTS query_interpretation_hypotheses (
            id TEXT PRIMARY KEY,
            query_graph_id TEXT NOT NULL,
            hypothesis_index INTEGER NOT NULL,
            interpretation_type TEXT NOT NULL CHECK(interpretation_type IN
                ('EXPLICIT_QUERY','STRUCTURAL_CONTINUATION',
                 'ANCHORED_EVENT_REBIND','STANDALONE_GAP_QUERY',
                 'CONTEXT_REFERENCE_QUERY')),
            prior_score REAL NOT NULL DEFAULT 0,
            current_evidence_score REAL NOT NULL DEFAULT 0,
            inherited_context_score REAL NOT NULL DEFAULT 0,
            event_retrieval_score REAL NOT NULL DEFAULT 0,
            graph_validation_score REAL NOT NULL DEFAULT 0,
            total_score REAL NOT NULL DEFAULT 0,
            admitted_event_ids_json TEXT NOT NULL DEFAULT '[]',
            rejection_reason TEXT,
            selected INTEGER NOT NULL DEFAULT 0 CHECK(selected IN (0,1)),
            created_at TEXT NOT NULL,
            FOREIGN KEY(query_graph_id) REFERENCES query_graphs(id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS query_interpretation_hypothesis_graph_idx
            ON query_interpretation_hypotheses(query_graph_id, selected DESC);

        /* GAP release diagnostics: full scoring breakdown for every
           participant candidate when a GAP rotation occurs. */
        CREATE TABLE IF NOT EXISTS gap_release_diagnostics (
            id TEXT PRIMARY KEY,
            query_graph_id TEXT NOT NULL,
            execution_id TEXT NOT NULL DEFAULT '',
            hypothesis_id TEXT NOT NULL DEFAULT '',
            gap_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'SELECTED',
            frame_id TEXT,
            event_id TEXT,
            question_family_key TEXT,
            candidates_json TEXT NOT NULL DEFAULT '[]',
            selected_participant_node_id TEXT,
            selected_score REAL NOT NULL DEFAULT 0,
            second_score REAL NOT NULL DEFAULT 0,
            release_margin REAL NOT NULL DEFAULT 0,
            decision TEXT NOT NULL CHECK(decision IN
                ('RELEASED','AMBIGUOUS','NO_COMPATIBLE_PARTICIPANT')),
            decision_reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(query_graph_id) REFERENCES query_graphs(id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS gap_release_diagnostic_query_idx
            ON gap_release_diagnostics(query_graph_id);

        CREATE TABLE IF NOT EXISTS gap_release_candidate_scores (
            diagnostic_id TEXT NOT NULL,
            participant_node_id TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            resolved_surface TEXT NOT NULL,
            exact_surface_match REAL NOT NULL DEFAULT 0,
            question_family_match REAL NOT NULL DEFAULT 0,
            root_gap_lineage_match REAL NOT NULL DEFAULT 0,
            latest_gap_lineage_match REAL NOT NULL DEFAULT 0,
            local_slot_score REAL NOT NULL DEFAULT 0,
            animacy_score REAL NOT NULL DEFAULT 0,
            case_score REAL NOT NULL DEFAULT 0,
            morphology_score REAL NOT NULL DEFAULT 0,
            frame_confidence REAL NOT NULL DEFAULT 0,
            recency_score REAL NOT NULL DEFAULT 0,
            explicit_current_penalty REAL NOT NULL DEFAULT 0,
            animacy_conflict REAL NOT NULL DEFAULT 0,
            hard_slot_conflict REAL NOT NULL DEFAULT 0,
            final_score REAL NOT NULL DEFAULT 0,
            rank INTEGER NOT NULL,
            accepted INTEGER NOT NULL DEFAULT 0 CHECK(accepted IN (0,1)),
            PRIMARY KEY(diagnostic_id, participant_node_id),
            FOREIGN KEY(diagnostic_id) REFERENCES gap_release_diagnostics(id)
                ON DELETE CASCADE
        );

        /* Dialogue context state: persists which turns are valid context
           sources and prevents UNRESOLVED contamination. */
        CREATE TABLE IF NOT EXISTS dialogue_context_states (
            conversation_id TEXT PRIMARY KEY,
            last_turn_id TEXT,
            last_resolved_turn_id TEXT,
            last_valid_binding_configuration_id TEXT,
            active_event_binding_frame_id TEXT,
            unresolved_turn_ids_json TEXT NOT NULL DEFAULT '[]',
            context_strength REAL NOT NULL DEFAULT 0,
            state_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS dialogue_context_state_conversation_idx
            ON dialogue_context_states(conversation_id);

        /* Predicate hypotheses: multiple lemma interpretations for
           ambiguous verb forms (e.g. стоит -> стоять/стоить). */
        CREATE TABLE IF NOT EXISTS predicate_hypotheses (
            id TEXT PRIMARY KEY,
            utterance_id TEXT NOT NULL,
            token_index INTEGER NOT NULL,
            lemma TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            morphology_confidence REAL NOT NULL DEFAULT 0,
            contextual_confidence REAL NOT NULL DEFAULT 0,
            construction_confidence REAL NOT NULL DEFAULT 0,
            participant_compatibility REAL NOT NULL DEFAULT 0,
            selected INTEGER NOT NULL DEFAULT 0 CHECK(selected IN (0,1)),
            selection_reason TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS predicate_hypothesis_utterance_idx
            ON predicate_hypotheses(utterance_id, token_index);

        /* Question family profiles: learned profiles for interrogative
           families (кто/что/где etc.) without semantic roles. */
        CREATE TABLE IF NOT EXISTS question_family_profiles (
            family_key TEXT PRIMARY KEY,
            canonical_lemma TEXT NOT NULL,
            operator_type TEXT NOT NULL,
            observed_surfaces_json TEXT NOT NULL DEFAULT '[]',
            morphology_distributions_json TEXT NOT NULL DEFAULT '{}',
            animacy_preference TEXT,
            animacy_confidence REAL NOT NULL DEFAULT 0,
            animacy_evidence_count INTEGER NOT NULL DEFAULT 0,
            compatible_slots_json TEXT NOT NULL DEFAULT '{}',
            support_count INTEGER NOT NULL DEFAULT 0,
            contradiction_count INTEGER NOT NULL DEFAULT 0,
            confidence REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL CHECK(status IN ('SHADOW','PROBATION','ACTIVE')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS question_family_observations (
            id TEXT PRIMARY KEY,
            family_key TEXT NOT NULL,
            query_graph_id TEXT NOT NULL,
            gap_node_id TEXT NOT NULL,
            operator_surface TEXT NOT NULL,
            resolved_animacy TEXT,
            animacy_compatible INTEGER,
            validated INTEGER NOT NULL DEFAULT 0 CHECK(validated IN (0,1)),
            created_at TEXT NOT NULL,
            FOREIGN KEY(family_key) REFERENCES question_family_profiles(family_key)
                ON DELETE CASCADE
        );
        """
    )
    versions = {
        "schema_version": str(SCHEMA_VERSION),
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "slot_model_version": SLOT_MODEL_VERSION,
        "construction_model_version": CONSTRUCTION_MODEL_VERSION,
        "semantic_cluster_version": SEMANTIC_CLUSTER_VERSION,
        "query_graph_version": QUERY_GRAPH_VERSION,
        "generation_version": GENERATION_VERSION,
        "migration_version": MIGRATION_VERSION,
        "projection_revision": "0",
        "transition_revision": "0",
    }
    conn.executemany(
        """INSERT INTO graph_meta(key,value) VALUES(?,?)
           ON CONFLICT(key) DO UPDATE SET value=excluded.value
           WHERE graph_meta.key NOT IN ('projection_revision','transition_revision')""",
        list(versions.items()),
    )
