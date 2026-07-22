"""Destructive, explicit and verifiable reset operations for test spaces."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import server.database as database

from .graph_repository import GraphRepository, encode, utcnow
from .runtime_reset import RuntimeResetRegistry, register_default_runtime_resetters
from .semantic_field import SemanticFieldService
from .universe import UniverseService


class ResetScope(str, Enum):
    FULL_TEST_STATE = "FULL_TEST_STATE"
    DERIVED_SEMANTIC_SPACE = "DERIVED_SEMANTIC_SPACE"
    DIALOGUE_STATE = "DIALOGUE_STATE"
    REASONING_TRACES = "REASONING_TRACES"
    EXPERIMENT_STATE = "EXPERIMENT_STATE"


class ResetMode(str, Enum):
    FRESH_SCHEMA = "FRESH_SCHEMA"
    CLEAR_DATA = "CLEAR_DATA"


COUNTER_TABLES = (
    "knowledge_sources",
    "graph_events",
    "semantic_clouds",
    "universe_entities",
    "dialogue_turns",
    "swarm_runs",
    "latent_dimensions",
    "experiment_runs",
)

DERIVED_TABLES = {
    "event_participant_clouds",
    "field_transitions",
    "contextual_cloud_projection_contributions",
    "contextual_cloud_projections",
    "semantic_field_force_traces",
    "field_source_contributions",
    "cloud_dimension_projections",
    "semantic_cloud_current_projections",
    "semantic_cloud_projections",
    "semantic_cloud_projection_revisions",
    "field_revisions",
    "semantic_clouds",
    "dimension_relations",
    "dimension_aliases",
    "projections",
    "dimension_clouds",
    "shadow_retrieval_runs",
    "dimension_evaluations",
    "dimension_lineage",
    "dimension_history",
    "latent_dimensions",
    "cloud_memberships",
    "entity_clouds",
    "universe_training_events",
    "universe_transitions",
    "word_usages",
    "word_forms",
    "lexemes",
    "universe_occurrences",
    "universe_entities",
    "candidate_event_observations",
    "nectar_packets",
    "bee_steps",
    "bee_missions",
    "swarm_runs",
}

DIALOGUE_TABLES = {
    "gap_release_candidate_scores",
    "gap_release_diagnostics",
    "query_interpretation_hypotheses",
    "event_binding_frame_question_profiles",
    "event_binding_frame_participants",
    "event_binding_frames",
    "dialogue_context_states",
    "training_episodes",
    "binding_configurations",
    "candidate_bindings",
    "query_operator_experiences",
    "query_operator_occurrences",
    "nectar_packets",
    "bee_steps",
    "bee_missions",
    "candidate_event_observations",
    "swarm_runs",
    "dialogue_turns",
    "query_graphs",
    "hives",
}

TRACE_TABLES = {
    "gap_release_candidate_scores",
    "gap_release_diagnostics",
    "query_interpretation_hypotheses",
    "shadow_retrieval_runs",
    "candidate_event_observations",
    "nectar_packets",
    "bee_steps",
    "bee_missions",
    "swarm_runs",
}

EXPERIMENT_TABLES = {"experiment_metrics", "experiment_runs"}


@dataclass(frozen=True)
class ResetReport:
    reset: bool
    scope: str
    mode: str
    database_generation_id: str
    before: dict[str, int]
    after: dict[str, int]
    universes_recreated: bool
    rebuild_available: bool
    field_revision: int
    runtime_caches: dict[str, dict[str, Any]]
    invariants: dict[str, bool]
    audit_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TestingResetService:
    """One reset implementation shared by API, CLI and experiments."""

    __test__ = False

    def __init__(self, repository: GraphRepository | None = None) -> None:
        self.repository = repository or GraphRepository()
        register_default_runtime_resetters()

    def reset(
        self,
        scope: ResetScope | str,
        mode: ResetMode | str,
        *,
        requested_by: str = "",
    ) -> dict[str, Any]:
        resolved_scope = ResetScope(scope)
        resolved_mode = ResetMode(mode)
        if resolved_mode is ResetMode.FRESH_SCHEMA and resolved_scope is not ResetScope.FULL_TEST_STATE:
            raise ValueError("FRESH_SCHEMA is valid only for FULL_TEST_STATE")

        before = self._counts()
        generation_id = uuid.uuid4().hex
        if resolved_mode is ResetMode.FRESH_SCHEMA:
            self._fresh_schema(generation_id)
        else:
            self._clear_data(resolved_scope, generation_id)

        universes_recreated = resolved_scope in {
            ResetScope.FULL_TEST_STATE,
            ResetScope.DERIVED_SEMANTIC_SPACE,
        }
        if universes_recreated:
            UniverseService(self.repository)

        runtime = RuntimeResetRegistry.reset_all()
        after = self._counts()
        field_revision = self._field_revision()
        invariants = self._invariants(resolved_scope, after, runtime, field_revision)
        audit_id = f"reset-audit-{uuid.uuid4().hex}"
        report = ResetReport(
            reset=all(invariants.values()),
            scope=resolved_scope.value,
            mode=resolved_mode.value,
            database_generation_id=generation_id,
            before=before,
            after=after,
            universes_recreated=universes_recreated,
            rebuild_available=resolved_scope is ResetScope.DERIVED_SEMANTIC_SPACE,
            field_revision=field_revision,
            runtime_caches=runtime,
            invariants=invariants,
            audit_id=audit_id,
        )
        self._write_audit(report, requested_by=requested_by)
        return report.to_dict()

    def rebuild_derived_space(self) -> dict[str, Any]:
        """Rebuild micro-universes and field from active confirmed evidence."""

        universes = UniverseService(self.repository)
        field = SemanticFieldService(self.repository)
        with self.repository.transaction() as conn:
            source_ids = [
                str(row[0])
                for row in conn.execute(
                    "SELECT id FROM knowledge_sources WHERE status='CONFIRMED' ORDER BY created_at,id"
                ).fetchall()
            ]
        updates = []
        for source_id in source_ids:
            updates.append(universes.ingest_source(source_id))
        field_update = field.rebuild_from_event_revision()
        runtime = RuntimeResetRegistry.reset_all()
        return {
            "rebuilt": True,
            "confirmed_source_count": len(source_ids),
            "universe_updates": updates,
            "field_update": field_update,
            "runtime_caches": runtime,
            "counts": self._counts(),
        }

    def _fresh_schema(self, generation_id: str) -> None:
        database.close_current_connection()
        path = Path(database.get_db_path())
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        database.init_db()
        self.repository = GraphRepository()
        with self.repository.transaction() as conn:
            self._set_generation(conn, generation_id)

    def _clear_data(self, scope: ResetScope, generation_id: str) -> None:
        with self.repository.transaction() as conn:
            tables = self._tables(conn)
            if scope is ResetScope.FULL_TEST_STATE:
                selected = tables - {"graph_meta", "testing_reset_audit"}
            elif scope is ResetScope.DERIVED_SEMANTIC_SPACE:
                selected = tables & DERIVED_TABLES
            elif scope is ResetScope.DIALOGUE_STATE:
                selected = tables & DIALOGUE_TABLES
            elif scope is ResetScope.REASONING_TRACES:
                selected = tables & TRACE_TABLES
            else:
                selected = tables & EXPERIMENT_TABLES

            conn.execute("PRAGMA defer_foreign_keys = ON")
            for table in self._delete_order(conn, selected):
                conn.execute(f'DELETE FROM "{table.replace(chr(34), chr(34) * 2)}"')
            if "sqlite_sequence" in {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }:
                sequence_tables = sorted(selected)
                if sequence_tables:
                    placeholders = ",".join("?" for _ in sequence_tables)
                    conn.execute(
                        f"DELETE FROM sqlite_sequence WHERE name IN ({placeholders})",
                        sequence_tables,
                    )
            self._set_generation(conn, generation_id)
            if scope in {ResetScope.FULL_TEST_STATE, ResetScope.DERIVED_SEMANTIC_SPACE}:
                # Invalidate cached spatial indexes even before the process hooks run.
                for key in ("projection_revision", "transition_revision"):
                    conn.execute(
                        """INSERT INTO graph_meta(key,value) VALUES(?, '1')
                           ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER)+1""",
                        (key,),
                    )

    @staticmethod
    def _set_generation(conn: sqlite3.Connection, generation_id: str) -> None:
        conn.execute(
            """INSERT INTO graph_meta(key,value) VALUES('database_generation_id',?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (generation_id,),
        )

    def _counts(self) -> dict[str, int]:
        with self.repository.transaction() as conn:
            tables = self._tables(conn)
            return {
                table: int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                if table in tables
                else 0
                for table in COUNTER_TABLES
            }

    def _field_revision(self) -> int:
        with self.repository.transaction() as conn:
            if "field_revisions" not in self._tables(conn):
                return 0
            return int(conn.execute("SELECT COALESCE(MAX(revision),0) FROM field_revisions").fetchone()[0])

    def _invariants(
        self,
        scope: ResetScope,
        after: dict[str, int],
        runtime: dict[str, dict[str, Any]],
        field_revision: int,
    ) -> dict[str, bool]:
        with self.repository.transaction() as conn:
            tables = self._tables(conn)
            version = conn.execute(
                "SELECT value FROM graph_meta WHERE key='schema_version'"
            ).fetchone()
            universe_count = int(conn.execute("SELECT COUNT(*) FROM universes").fetchone()[0])
        full = scope is ResetScope.FULL_TEST_STATE
        derived = scope in {ResetScope.FULL_TEST_STATE, ResetScope.DERIVED_SEMANTIC_SPACE}
        dialogue = scope in {ResetScope.FULL_TEST_STATE, ResetScope.DIALOGUE_STATE}
        return {
            "schema_valid": bool(version) and bool(tables),
            "graph_empty": (not full) or after["knowledge_sources"] == 0 and after["graph_events"] == 0,
            "field_empty": (not derived) or after["semantic_clouds"] == 0 and field_revision == 0,
            "universes_empty": (not derived) or after["universe_entities"] == 0,
            "universe_registry_present": (not derived) or universe_count > 0,
            "dialogue_empty": (not dialogue) or after["dialogue_turns"] == 0,
            "runtime_caches_empty": all(item.get("reset") is True for item in runtime.values()),
        }

    def _write_audit(self, report: ResetReport, *, requested_by: str) -> None:
        with self.repository.transaction() as conn:
            conn.execute(
                """INSERT INTO testing_reset_audit
                   (id,scope,mode,database_generation_id,report_json,requested_by,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    report.audit_id,
                    report.scope,
                    report.mode,
                    report.database_generation_id,
                    encode(report.to_dict()),
                    requested_by,
                    utcnow(),
                ),
            )

    @staticmethod
    def _tables(conn: sqlite3.Connection) -> set[str]:
        return {
            str(row[0])
            for row in conn.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='table' AND name NOT LIKE 'sqlite_%'"""
            ).fetchall()
        }

    @staticmethod
    def _delete_order(conn: sqlite3.Connection, selected: Iterable[str]) -> list[str]:
        selected_set = set(selected)
        parents: dict[str, set[str]] = {}
        for table in selected_set:
            escaped = table.replace(chr(34), chr(34) * 2)
            parents[table] = {
                str(row[2])
                for row in conn.execute(f'PRAGMA foreign_key_list("{escaped}")').fetchall()
                if str(row[2]) in selected_set
            }

        memo: dict[str, int] = {}

        def depth(table: str, stack: set[str]) -> int:
            if table in memo:
                return memo[table]
            if table in stack:
                return 0
            value = 1 + max(
                (depth(parent, stack | {table}) for parent in parents.get(table, set())),
                default=0,
            )
            memo[table] = value
            return value

        return sorted(selected_set, key=lambda table: (-depth(table, set()), table))


def _main() -> int:
    parser = argparse.ArgumentParser(description="Reset SuperAI test state")
    parser.add_argument("--scope", default="full", choices=("full", "derived", "dialogue", "traces", "experiment"))
    parser.add_argument("--mode", default="fresh-schema", choices=("fresh-schema", "clear-data"))
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--requested-by", default=f"cli:{os.getpid()}")
    args = parser.parse_args()
    expected = os.getenv("SUPERAI_TEST_RESET_CONFIRMATION", "RESET TEST SPACE")
    if args.confirm != expected:
        parser.error("confirmation string does not match")
    scopes = {
        "full": ResetScope.FULL_TEST_STATE,
        "derived": ResetScope.DERIVED_SEMANTIC_SPACE,
        "dialogue": ResetScope.DIALOGUE_STATE,
        "traces": ResetScope.REASONING_TRACES,
        "experiment": ResetScope.EXPERIMENT_STATE,
    }
    modes = {
        "fresh-schema": ResetMode.FRESH_SCHEMA,
        "clear-data": ResetMode.CLEAR_DATA,
    }
    report = TestingResetService().reset(
        scopes[args.scope],
        modes[args.mode],
        requested_by=args.requested_by,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["reset"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
