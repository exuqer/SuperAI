"""SQLite persistence shared by the modular-monolith components.

SQLite is intentionally the metadata and queue store in the MVP. Blobs live in
the object store; every aggregate stores a JSON snapshot with an explicit
schema version rather than relying on pickled Python state.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def json_loads(value: Optional[str], default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError("Cannot serialize %s" % type(value).__name__)


class SqliteDatabase:
    """A tiny transaction boundary; domain code never opens ad-hoc databases."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
            else:
                self.connection.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.connection.execute(sql, params)

    def one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        row = self.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def all(self, sql: str, params: Sequence[Any] = ()) -> list[Dict[str, Any]]:
        return [dict(row) for row in self.execute(sql, params).fetchall()]

    def _migrate(self) -> None:
        # Tables are additive in this prototype. A production migration runner
        # would retain the same schema-version discipline with numbered files.
        schema = """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            hive_id TEXT,
            status TEXT NOT NULL,
            contract_json TEXT NOT NULL,
            answer_json TEXT,
            error_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_trace ON tasks(trace_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_tenant ON tasks(tenant_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS work_items (
            command_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            handler TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            max_attempts INTEGER NOT NULL,
            priority INTEGER NOT NULL,
            scheduled_at TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            budget_json TEXT NOT NULL,
            deadline_at TEXT,
            last_error_json TEXT,
            tenant_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(tenant_id, idempotency_key)
        );
        CREATE INDEX IF NOT EXISTS idx_work_claim ON work_items(status, scheduled_at, priority);

        CREATE TABLE IF NOT EXISTS outbox (
            event_id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            published_at TEXT
        );

        CREATE TABLE IF NOT EXISTS trace_spans (
            span_id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL,
            parent_span_id TEXT,
            sequence INTEGER NOT NULL,
            component TEXT NOT NULL,
            operation TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_span_trace ON trace_spans(trace_id, sequence);

        CREATE TABLE IF NOT EXISTS domain_events (
            event_id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            kind TEXT NOT NULL,
            producer TEXT NOT NULL,
            causation_id TEXT,
            payload_json TEXT NOT NULL,
            occurred_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_event_trace ON domain_events(trace_id, sequence);

        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            media_type TEXT NOT NULL,
            schema_name TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            size INTEGER NOT NULL,
            tenant_id TEXT NOT NULL,
            access_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            deleted_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_artifact_hash ON artifacts(content_hash);
        CREATE INDEX IF NOT EXISTS idx_artifact_tenant ON artifacts(tenant_id);

        CREATE TABLE IF NOT EXISTS blobs (
            content_hash TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS artifact_writes (
            tenant_id TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(tenant_id, idempotency_key)
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            snapshot_id TEXT PRIMARY KEY,
            aggregate_type TEXT NOT NULL,
            aggregate_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            artifact_id TEXT NOT NULL,
            state_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_snapshot_aggregate ON snapshots(aggregate_type, aggregate_id, sequence DESC);

        CREATE TABLE IF NOT EXISTS hives (
            hive_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            project_id TEXT,
            conversation_id TEXT NOT NULL,
            state TEXT NOT NULL,
            topic_json TEXT NOT NULL,
            contract_json TEXT NOT NULL,
            state_json TEXT NOT NULL,
            snapshot_id TEXT,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_hive_conversation ON hives(tenant_id, conversation_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS hive_entries (
            entry_id TEXT PRIMARY KEY,
            hive_id TEXT NOT NULL,
            store_name TEXT NOT NULL,
            layer TEXT NOT NULL,
            content_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(hive_id) REFERENCES hives(hive_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_hive_entries ON hive_entries(hive_id, layer, created_at);

        CREATE TABLE IF NOT EXISTS hive_entry_writes (
            hive_id TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(hive_id, idempotency_key)
        );

        CREATE TABLE IF NOT EXISTS evictions (
            eviction_id TEXT PRIMARY KEY,
            hive_id TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            score_before REAL NOT NULL,
            score_after REAL NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sources (
            source_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            title TEXT NOT NULL,
            access_json TEXT NOT NULL,
            status TEXT NOT NULL,
            imported_at TEXT NOT NULL,
            deleted_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_source_dedupe ON sources(tenant_id, content_hash, deleted_at);

        CREATE TABLE IF NOT EXISTS concepts (
            concept_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            label TEXT NOT NULL,
            normalized_label TEXT NOT NULL,
            concept_type TEXT NOT NULL,
            aliases_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(tenant_id, normalized_label)
        );
        CREATE INDEX IF NOT EXISTS idx_concept_label ON concepts(tenant_id, normalized_label);

        CREATE TABLE IF NOT EXISTS claims (
            claim_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_value TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_artifact_id TEXT NOT NULL,
            source_fragment TEXT NOT NULL,
            sector_json TEXT NOT NULL,
            access_json TEXT NOT NULL,
            verification_status TEXT NOT NULL,
            scores_json TEXT NOT NULL,
            valid_from TEXT,
            valid_to TEXT,
            created_at TEXT NOT NULL,
            deleted_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_claim_source ON claims(source_id, deleted_at);
        CREATE INDEX IF NOT EXISTS idx_claim_subject ON claims(tenant_id, subject_id);

        CREATE TABLE IF NOT EXISTS capabilities (
            capability_id TEXT NOT NULL,
            version TEXT NOT NULL,
            kind TEXT NOT NULL,
            manifest_json TEXT NOT NULL,
            health TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(capability_id, version)
        );

        CREATE TABLE IF NOT EXISTS plans (
            plan_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            hive_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            plan_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS skills (
            skill_id TEXT NOT NULL,
            version TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            access_json TEXT NOT NULL,
            state TEXT NOT NULL,
            manifest_json TEXT NOT NULL,
            rollback_version TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY(skill_id, version)
        );

        CREATE TABLE IF NOT EXISTS composts (
            compost_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            access_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS compost_dependencies (
            compost_id TEXT NOT NULL,
            source_artifact_id TEXT NOT NULL,
            PRIMARY KEY(compost_id, source_artifact_id)
        );
        CREATE INDEX IF NOT EXISTS idx_compost_dependency_source ON compost_dependencies(source_artifact_id);

        CREATE TABLE IF NOT EXISTS compost_integrations (
            compost_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            PRIMARY KEY(compost_id, source_id)
        );
        CREATE INDEX IF NOT EXISTS idx_compost_integration_source ON compost_integrations(source_id);

        CREATE TABLE IF NOT EXISTS genomes (
            genome_id TEXT NOT NULL,
            version TEXT NOT NULL,
            manifest_json TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(genome_id, version)
        );

        -- ΩE Tables
        CREATE TABLE IF NOT EXISTS graph_edges (
            edge_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            project_id TEXT,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            weight REAL NOT NULL,
            confidence REAL NOT NULL,
            scope TEXT NOT NULL,
            provenance_refs_json TEXT NOT NULL,
            valid_from TEXT,
            valid_to TEXT,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_tenant ON graph_edges(tenant_id, project_id);

        CREATE TABLE IF NOT EXISTS active_graph_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            hive_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            cosmos_version TEXT NOT NULL,
            node_ids_json TEXT NOT NULL,
            edge_ids_json TEXT NOT NULL,
            node_types_json TEXT NOT NULL DEFAULT '[]',
            edge_types_json TEXT NOT NULL DEFAULT '[]',
            frontier_json TEXT NOT NULL,
            event_count INTEGER NOT NULL,
            budget_ledger_json TEXT NOT NULL,
            random_seed INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ags_task ON active_graph_snapshots(task_id);
        CREATE INDEX IF NOT EXISTS idx_ags_hive ON active_graph_snapshots(hive_id);

        CREATE TABLE IF NOT EXISTS hypotheses (
            hypothesis_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            project_id TEXT,
            family_id TEXT NOT NULL,
            statement TEXT NOT NULL,
            assumptions_json TEXT NOT NULL,
            evidence_for_json TEXT NOT NULL,
            evidence_against_json TEXT NOT NULL,
            predictions_json TEXT NOT NULL,
            confidence REAL NOT NULL,
            novelty REAL NOT NULL,
            allocated_budget INTEGER NOT NULL,
            spent_budget INTEGER NOT NULL,
            status TEXT NOT NULL,
            parent_ids_json TEXT NOT NULL,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_hypotheses_task ON hypotheses(task_id);
        CREATE INDEX IF NOT EXISTS idx_hypotheses_family ON hypotheses(family_id);
        CREATE INDEX IF NOT EXISTS idx_hypotheses_tenant ON hypotheses(tenant_id, project_id);

        CREATE TABLE IF NOT EXISTS hypothesis_evidence (
            evidence_id TEXT PRIMARY KEY,
            hypothesis_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            description TEXT NOT NULL,
            weight REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(hypothesis_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_hyp_evidence_hyp ON hypothesis_evidence(hypothesis_id);

        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            hypothesis_id TEXT NOT NULL,
            experiment_input_json TEXT NOT NULL,
            expected_output_json TEXT NOT NULL,
            tolerance_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(hypothesis_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_predictions_hyp ON predictions(hypothesis_id);

        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            competing_hypothesis_ids_json TEXT NOT NULL,
            operation TEXT NOT NULL,
            input_json TEXT NOT NULL,
            expected_information_gain REAL NOT NULL,
            estimated_cost INTEGER NOT NULL,
            risk REAL NOT NULL,
            result_ref TEXT,
            evidence_status TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_experiments_task ON experiments(task_id);
        CREATE INDEX IF NOT EXISTS idx_experiments_tenant ON experiments(tenant_id);

        CREATE TABLE IF NOT EXISTS concept_candidates (
            concept_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            project_id TEXT,
            name TEXT NOT NULL,
            definition_ref TEXT NOT NULL,
            source_subgraph_refs_json TEXT NOT NULL,
            positive_examples_json TEXT NOT NULL,
            negative_examples_json TEXT NOT NULL,
            train_task_ids_json TEXT NOT NULL,
            holdout_manifest_ref TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            state TEXT NOT NULL,
            rollback_target TEXT,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_concept_candidates_tenant ON concept_candidates(tenant_id, project_id);
        CREATE INDEX IF NOT EXISTS idx_concept_candidates_state ON concept_candidates(state);

        CREATE TABLE IF NOT EXISTS concept_evaluations (
            concept_id TEXT PRIMARY KEY,
            baseline_run_id TEXT NOT NULL,
            treatment_run_id TEXT NOT NULL,
            ablation_run_id TEXT NOT NULL,
            quality_delta REAL NOT NULL,
            cost_delta REAL NOT NULL,
            transfer_delta REAL NOT NULL,
            accepted INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(concept_id) REFERENCES concept_candidates(concept_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS benchmark_runs (
            run_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            project_id TEXT,
            git_revision TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            seed INTEGER NOT NULL,
            latency_ms INTEGER NOT NULL,
            cost REAL NOT NULL,
            quality REAL NOT NULL,
            status TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'baseline',
            concept_id TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_benchmark_runs_task ON benchmark_runs(task_id);
        CREATE INDEX IF NOT EXISTS idx_benchmark_runs_tenant ON benchmark_runs(tenant_id, project_id);
        """
        with self._lock:
            self.connection.executescript(schema)
            self._migrate_source_scope_uniqueness()
            self._ensure_skill_scope_columns()
            self._ensure_snapshot_tenant_column()
            self._ensure_omega_e_tables()
            self.connection.execute(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
                ("schema_version", "1.0"),
            )

    def _migrate_source_scope_uniqueness(self) -> None:
        """Allow one immutable payload to be imported under distinct scopes.

        An early prototype made ``(tenant_id, content_hash)`` globally unique.
        That can return a project-A source to an import requested for project B.
        Source records are logical provenance records, so scope is part of their
        identity even when their archive blob is deduplicated.
        """
        row = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'sources'"
        ).fetchone()
        definition = row[0] if row else ""
        if "UNIQUE(tenant_id, content_hash)" not in definition.replace("\n", " "):
            return
        self.connection.executescript(
            """
            ALTER TABLE sources RENAME TO sources_legacy_scope;
            CREATE TABLE sources (
                source_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                title TEXT NOT NULL,
                access_json TEXT NOT NULL,
                status TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                deleted_at TEXT
            );
            INSERT INTO sources(source_id, tenant_id, artifact_id, content_hash, title, access_json, status, imported_at, deleted_at)
                SELECT source_id, tenant_id, artifact_id, content_hash, title, access_json, status, imported_at, deleted_at
                FROM sources_legacy_scope;
            DROP TABLE sources_legacy_scope;
            CREATE INDEX IF NOT EXISTS idx_source_dedupe ON sources(tenant_id, content_hash, deleted_at);
            """
        )

    def _ensure_skill_scope_columns(self) -> None:
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(skills)")}
        if "tenant_id" not in columns:
            self.connection.execute("ALTER TABLE skills ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'local'")
        if "access_json" not in columns:
            self.connection.execute(
                "ALTER TABLE skills ADD COLUMN access_json TEXT NOT NULL DEFAULT '{\"schema_version\":\"1.0\",\"tenant_id\":\"local\",\"project_id\":null,\"visibility\":\"tenant\",\"retention\":\"standard\"}'"
            )
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_skill_tenant ON skills(tenant_id, state)")

    def _ensure_snapshot_tenant_column(self) -> None:
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(snapshots)")}
        if "tenant_id" not in columns:
            self.connection.execute("ALTER TABLE snapshots ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'local'")
            self.connection.execute(
                "UPDATE snapshots SET tenant_id = COALESCE((SELECT tenant_id FROM artifacts WHERE artifacts.artifact_id = snapshots.artifact_id), tenant_id)"
            )
        # Earlier prototype versions keyed sequence globally. Aggregate IDs are
        # usually UUIDs but access isolation must not rely on that accident.
        self.connection.execute("DROP INDEX IF EXISTS idx_snapshot_sequence")
        self.connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshot_sequence ON snapshots(tenant_id, aggregate_type, aggregate_id, sequence)"
        )

    def _ensure_omega_e_tables(self) -> None:
        """Ensure ΩE tables exist (idempotent migration)."""
        # Tables are created in _migrate's main schema block, but this ensures
        # any missing tables/indices are added for upgrades.
        tables = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row["name"] for row in tables}

        if "graph_edges" not in table_names:
            self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                edge_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                project_id TEXT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL,
                confidence REAL NOT NULL,
                scope TEXT NOT NULL,
                provenance_refs_json TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_tenant ON graph_edges(tenant_id, project_id);
            """)

        if "active_graph_snapshots" not in table_names:
            self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS active_graph_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                hive_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                cosmos_version TEXT NOT NULL,
                node_ids_json TEXT NOT NULL,
                edge_ids_json TEXT NOT NULL,
                node_types_json TEXT NOT NULL DEFAULT '[]',
                edge_types_json TEXT NOT NULL DEFAULT '[]',
                frontier_json TEXT NOT NULL,
                event_count INTEGER NOT NULL,
                budget_ledger_json TEXT NOT NULL,
                random_seed INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ags_task ON active_graph_snapshots(task_id);
            CREATE INDEX IF NOT EXISTS idx_ags_hive ON active_graph_snapshots(hive_id);
            """)
        else:
            snapshot_columns = {
                row["name"] for row in self.connection.execute("PRAGMA table_info(active_graph_snapshots)")
            }
            if "node_types_json" not in snapshot_columns:
                self.connection.execute(
                    "ALTER TABLE active_graph_snapshots ADD COLUMN node_types_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "edge_types_json" not in snapshot_columns:
                self.connection.execute(
                    "ALTER TABLE active_graph_snapshots ADD COLUMN edge_types_json TEXT NOT NULL DEFAULT '[]'"
                )

        if "hypotheses" not in table_names:
            self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                project_id TEXT,
                family_id TEXT NOT NULL,
                statement TEXT NOT NULL,
                assumptions_json TEXT NOT NULL,
                evidence_for_json TEXT NOT NULL,
                evidence_against_json TEXT NOT NULL,
                predictions_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                novelty REAL NOT NULL,
                allocated_budget INTEGER NOT NULL,
                spent_budget INTEGER NOT NULL,
                status TEXT NOT NULL,
                parent_ids_json TEXT NOT NULL,
                version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_hypotheses_task ON hypotheses(task_id);
            CREATE INDEX IF NOT EXISTS idx_hypotheses_family ON hypotheses(family_id);
            CREATE INDEX IF NOT EXISTS idx_hypotheses_tenant ON hypotheses(tenant_id, project_id);
            """)

        if "hypothesis_evidence" not in table_names:
            self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS hypothesis_evidence (
                evidence_id TEXT PRIMARY KEY,
                hypothesis_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                description TEXT NOT NULL,
                weight REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(hypothesis_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_hyp_evidence_hyp ON hypothesis_evidence(hypothesis_id);
            """)

        if "predictions" not in table_names:
            self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                hypothesis_id TEXT NOT NULL,
                experiment_input_json TEXT NOT NULL,
                expected_output_json TEXT NOT NULL,
                tolerance_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(hypothesis_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_hyp ON predictions(hypothesis_id);
            """)

        if "experiments" not in table_names:
            self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                competing_hypothesis_ids_json TEXT NOT NULL,
                operation TEXT NOT NULL,
                input_json TEXT NOT NULL,
                expected_information_gain REAL NOT NULL,
                estimated_cost INTEGER NOT NULL,
                risk REAL NOT NULL,
                result_ref TEXT,
                evidence_status TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_experiments_task ON experiments(task_id);
            CREATE INDEX IF NOT EXISTS idx_experiments_tenant ON experiments(tenant_id);
            """)

        if "concept_candidates" not in table_names:
            self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS concept_candidates (
                concept_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                project_id TEXT,
                name TEXT NOT NULL,
                definition_ref TEXT NOT NULL,
                source_subgraph_refs_json TEXT NOT NULL,
                positive_examples_json TEXT NOT NULL,
                negative_examples_json TEXT NOT NULL,
                train_task_ids_json TEXT NOT NULL,
                holdout_manifest_ref TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                state TEXT NOT NULL,
                rollback_target TEXT,
                version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_concept_candidates_tenant ON concept_candidates(tenant_id, project_id);
            CREATE INDEX IF NOT EXISTS idx_concept_candidates_state ON concept_candidates(state);
            """)

        if "concept_evaluations" not in table_names:
            self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS concept_evaluations (
                concept_id TEXT PRIMARY KEY,
                baseline_run_id TEXT NOT NULL,
                treatment_run_id TEXT NOT NULL,
                ablation_run_id TEXT NOT NULL,
                quality_delta REAL NOT NULL,
                cost_delta REAL NOT NULL,
                transfer_delta REAL NOT NULL,
                accepted INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(concept_id) REFERENCES concept_candidates(concept_id) ON DELETE CASCADE
            );
            """)

        if "benchmark_runs" not in table_names:
            self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                project_id TEXT,
                git_revision TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                seed INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                cost REAL NOT NULL,
                quality REAL NOT NULL,
                status TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'baseline',
                concept_id TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_benchmark_runs_task ON benchmark_runs(task_id);
            CREATE INDEX IF NOT EXISTS idx_benchmark_runs_tenant ON benchmark_runs(tenant_id, project_id);
            """)
        else:
            benchmark_columns = {
                row["name"] for row in self.connection.execute("PRAGMA table_info(benchmark_runs)")
            }
            if "mode" not in benchmark_columns:
                self.connection.execute(
                    "ALTER TABLE benchmark_runs ADD COLUMN mode TEXT NOT NULL DEFAULT 'baseline'"
                )
            if "concept_id" not in benchmark_columns:
                self.connection.execute("ALTER TABLE benchmark_runs ADD COLUMN concept_id TEXT")
