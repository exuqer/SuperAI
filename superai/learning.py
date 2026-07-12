"""Conservative experience recycling, skill lifecycle and optional genome records.

Nothing in this module publishes user-derived data or changes an active route
automatically. Every promotion is an explicit, inspectable state transition.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .contracts import (
    AccessScope,
    AnswerEnvelope,
    CompostRecord,
    ExecutionPlan,
    GenomeManifest,
    SkillManifest,
    SkillState,
    TaskContract,
    TaskState,
    new_id,
    utcnow,
)
from .cosmos import Cosmos
from .database import SqliteDatabase, json_dumps, json_loads
from .storage import ObjectStore


class LearningSafetyError(ValueError):
    pass


class ExperienceCompiler:
    def __init__(self, database: SqliteDatabase, store: ObjectStore, cosmos: Cosmos) -> None:
        self.database = database
        self.store = store
        self.cosmos = cosmos

    def decompose_task(self, task_id: str, tenant_id: str, *, project_id: Optional[str] = None) -> CompostRecord:
        """Create a private, reconstructable derived artifact from a completed task."""
        row = self.database.one(
            "SELECT * FROM tasks WHERE task_id = ? AND tenant_id = ?", (task_id, tenant_id)
        )
        if row is None:
            raise KeyError("task not found")
        if row["status"] != TaskState.SUCCEEDED.value or not row["answer_json"]:
            raise LearningSafetyError("only successful trace-backed tasks may be decomposed")
        contract = TaskContract.model_validate(json_loads(row["contract_json"]))
        if contract.project_id != project_id:
            raise LearningSafetyError("task belongs to another project")
        answer = AnswerEnvelope.model_validate(json_loads(row["answer_json"]))
        all_pass = all(report.verdict.value == "pass" for report in answer.critic_reports)
        scope = AccessScope(
            tenant_id=tenant_id,
            project_id=contract.project_id,
            visibility="project" if contract.project_id else "tenant",
        )
        payload = {
            "type": "task_outcome",
            "normalized_content": _normalise_answer(answer.answer),
            "source_trace_refs": [row["trace_id"]],
            "source_artifact_refs": [source.artifact_id for source in answer.sources],
            "applicability_scope": {"task_class": _task_class(contract.goal), "source_policy": contract.source_policy},
            "access_scope": scope.model_dump(mode="json"),
            "confidence": 0.7 if all_pass else 0.3,
            "verification_status": "reviewed" if all_pass else "hypothesis",
            "reconstruction_pointer": {"hive_id": row["hive_id"], "task_id": task_id},
            "compiler_version": "1.0",
        }
        artifact = self.store.put_json(
            payload,
            tenant_id=tenant_id,
            schema_name="Compost",
            access_scope=scope,
            idempotency_key="compost:" + task_id,
        )
        record = CompostRecord(
            tenant_id=tenant_id,
            artifact_ref=artifact,
            trace_id=row["trace_id"],
            access_scope=scope,
            status="candidate",
        )
        self.database.execute(
            "INSERT OR IGNORE INTO composts(compost_id, tenant_id, artifact_id, trace_id, access_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.compost_id,
                tenant_id,
                artifact.artifact_id,
                row["trace_id"],
                json_dumps(scope),
                record.status,
                record.created_at.isoformat(),
            ),
        )
        # An idempotent artifact write can return a previous artifact; locate
        # the existing compost record to make the API idempotent too.
        existing = self.database.one(
            "SELECT * FROM composts WHERE tenant_id = ? AND trace_id = ? ORDER BY created_at LIMIT 1",
            (tenant_id, row["trace_id"]),
        )
        result = self._row_to_compost(existing, tenant_id, project_id=project_id)
        for source in answer.sources:
            self.database.execute(
                "INSERT OR IGNORE INTO compost_dependencies(compost_id, source_artifact_id) VALUES (?, ?)",
                (result.compost_id, source.artifact_id),
            )
        return result

    def validate_compost(self, compost_id: str, tenant_id: str, *, project_id: Optional[str] = None) -> CompostRecord:
        record = self.compost(compost_id, tenant_id, project_id=project_id)
        payload = self.store.get_json(
            record.artifact_ref.artifact_id,
            tenant_id,
            project_id=record.access_scope.project_id,
        )
        required = {"type", "source_trace_refs", "access_scope", "verification_status", "reconstruction_pointer"}
        if not required.issubset(payload):
            raise LearningSafetyError("compost is missing required provenance fields")
        if payload["access_scope"].get("tenant_id") != tenant_id:
            raise LearningSafetyError("compost scope changed during validation")
        self.database.execute("UPDATE composts SET status = ? WHERE compost_id = ?", ("validated", compost_id))
        return self.compost(compost_id, tenant_id, project_id=project_id)

    def integrate_compost(self, compost_id: str, tenant_id: str, *, project_id: Optional[str] = None) -> CompostRecord:
        """Explicit integration; a hypothesis remains quarantined, never upgraded."""
        record = self.compost(compost_id, tenant_id, project_id=project_id)
        if record.status != "validated":
            raise LearningSafetyError("only validated compost can be integrated")
        payload = self.store.get_json(
            record.artifact_ref.artifact_id,
            tenant_id,
            project_id=record.access_scope.project_id,
        )
        scope = AccessScope.model_validate(payload["access_scope"])
        imported = self.cosmos.import_text(
            title="compost-%s.md" % compost_id,
            text="Experience-derived conclusion: " + payload["normalized_content"],
            tenant_id=tenant_id,
            access_scope=scope,
            sectors=["Experience"],
            trusted=payload["verification_status"] in ("reviewed", "verified"),
        )
        with self.database.transaction() as connection:
            connection.execute("UPDATE composts SET status = ? WHERE compost_id = ?", ("integrated", compost_id))
            connection.execute(
                "INSERT OR IGNORE INTO compost_integrations(compost_id, source_id) VALUES (?, ?)",
                (compost_id, imported.source_id),
            )
        return self.compost(compost_id, tenant_id, project_id=project_id)

    def compile_candidate(
        self,
        *,
        tenant_id: str,
        train_task_ids: Sequence[str],
        holdout_task_ids: Sequence[str],
        minimum_traces: int = 2,
    ) -> SkillManifest:
        train = list(dict.fromkeys(train_task_ids))
        holdout = list(dict.fromkeys(holdout_task_ids))
        if set(train) & set(holdout):
            raise LearningSafetyError("train and holdout sets must be disjoint")
        if not holdout:
            raise LearningSafetyError("a separate holdout set is required before skill compilation")
        if len(train) < minimum_traces:
            raise LearningSafetyError("insufficient successful traces to compile a skill")
        task_rows = []
        for task_id in train:
            row = self.database.one("SELECT * FROM tasks WHERE task_id = ? AND tenant_id = ?", (task_id, tenant_id))
            if row is None or row["status"] != TaskState.SUCCEEDED.value:
                raise LearningSafetyError("every training task must be a successful owned trace")
            task_rows.append(row)
        for task_id in holdout:
            # Ownership/existence is checked without loading the holdout trace,
            # answer or plan into the compiler's training path.
            row = self.database.one(
                "SELECT task_id FROM tasks WHERE task_id = ? AND tenant_id = ?", (task_id, tenant_id)
            )
            if row is None:
                raise LearningSafetyError("every holdout reference must belong to the same tenant")
        conversations = {TaskContract.model_validate(json_loads(row["contract_json"])).conversation_id for row in task_rows}
        if len(conversations) < min(2, len(train)):
            raise LearningSafetyError("training traces lack input diversity")
        project_ids = {TaskContract.model_validate(json_loads(row["contract_json"])).project_id for row in task_rows}
        if len(project_ids) != 1:
            raise LearningSafetyError("a private skill cannot mix project scopes")
        project_id = next(iter(project_ids))
        groups: Dict[tuple[str, ...], list[tuple[Dict[str, Any], ExecutionPlan]]] = defaultdict(list)
        for row in task_rows:
            plan_row = self.database.one(
                "SELECT plan_json FROM plans WHERE task_id = ? ORDER BY revision DESC, created_at DESC LIMIT 1", (row["task_id"],)
            )
            if not plan_row:
                raise LearningSafetyError("successful task has no saved execution plan")
            plan = ExecutionPlan.model_validate(json_loads(plan_row["plan_json"]))
            groups[tuple(step.operation for step in plan.steps)].append((row, plan))
        signature, group = max(groups.items(), key=lambda item: len(item[1]))
        if len(group) < minimum_traces:
            raise LearningSafetyError("no repeated procedure graph found")
        exemplar = group[0][1]
        first_contract = TaskContract.model_validate(json_loads(group[0][0]["contract_json"]))
        skill = SkillManifest(
            tenant_id=tenant_id,
            access_scope=AccessScope(
                tenant_id=tenant_id,
                project_id=project_id,
                visibility="project" if project_id else "tenant",
            ),
            task_class=_task_class(first_contract.goal),
            procedure=exemplar.steps,
            preconditions=["TaskContract source policy is compatible", "input/output schemas match"],
            train_task_ids=[row["task_id"] for row, _ in group],
            holdout_task_ids=holdout,
            provenance_trace_ids=[row["trace_id"] for row, _ in group],
            metrics={"training_trace_count": float(len(group)), "operation_count": float(len(signature))},
        )
        self._save_skill(skill)
        return skill

    def validate_skill(
        self,
        skill_id: str,
        version: str,
        *,
        tenant_id: str,
        project_id: Optional[str] = None,
        quality_delta: float,
        latency_delta: float,
        resource_delta: float,
        risk_penalty: float,
    ) -> SkillManifest:
        skill = self.skill(skill_id, version, tenant_id=tenant_id, project_id=project_id)
        if skill.state != SkillState.CANDIDATE:
            raise LearningSafetyError("only a candidate skill can be validated")
        if set(skill.train_task_ids) & set(skill.holdout_task_ids):
            raise LearningSafetyError("holdout leakage detected")
        utility = quality_delta - 0.2 * latency_delta - 0.2 * resource_delta - risk_penalty
        skill.metrics.update(
            {
                "quality_delta": quality_delta,
                "latency_delta": latency_delta,
                "resource_delta": resource_delta,
                "risk_penalty": risk_penalty,
                "utility": utility,
            }
        )
        if quality_delta < 0 or utility < 0:
            raise LearningSafetyError("candidate does not improve the baseline")
        skill.state = SkillState.VALIDATED
        self._save_skill(skill)
        return skill

    def shadow_skill(self, skill_id: str, version: str, *, tenant_id: str, project_id: Optional[str] = None) -> SkillManifest:
        skill = self.skill(skill_id, version, tenant_id=tenant_id, project_id=project_id)
        if skill.state != SkillState.VALIDATED:
            raise LearningSafetyError("only validated skills may enter shadow mode")
        skill.state = SkillState.SHADOW
        self._save_skill(skill)
        return skill

    def activate_skill(self, skill_id: str, version: str, *, tenant_id: str, project_id: Optional[str] = None) -> SkillManifest:
        skill = self.skill(skill_id, version, tenant_id=tenant_id, project_id=project_id)
        if skill.state != SkillState.SHADOW:
            raise LearningSafetyError("only a shadow skill may be activated")
        current = self.database.one(
            "SELECT manifest_json FROM skills WHERE tenant_id = ? AND state = ? ORDER BY created_at DESC LIMIT 1",
            (tenant_id, SkillState.ACTIVE.value),
        )
        if current:
            prior = SkillManifest.model_validate(json_loads(current["manifest_json"]))
            skill.rollback_version = prior.version
            prior.state = SkillState.DEPRECATED
            self._save_skill(prior)
        skill.state = SkillState.ACTIVE
        self._save_skill(skill)
        return skill

    def skill(
        self,
        skill_id: str,
        version: str,
        *,
        tenant_id: str,
        project_id: Optional[str] = None,
    ) -> SkillManifest:
        row = self.database.one(
            "SELECT manifest_json FROM skills WHERE skill_id = ? AND version = ?", (skill_id, version)
        )
        if row is None:
            raise KeyError("skill not found")
        skill = SkillManifest.model_validate(json_loads(row["manifest_json"]))
        if not _scope_allows(skill.access_scope, tenant_id, project_id):
            raise LearningSafetyError("skill belongs to another access scope")
        return skill

    def skills(self, *, tenant_id: str, project_id: Optional[str] = None) -> list[SkillManifest]:
        result: list[SkillManifest] = []
        for row in self.database.all("SELECT manifest_json FROM skills ORDER BY created_at DESC"):
            skill = SkillManifest.model_validate(json_loads(row["manifest_json"]))
            if _scope_allows(skill.access_scope, tenant_id, project_id):
                result.append(skill)
        return result

    def compost(self, compost_id: str, tenant_id: str, *, project_id: Optional[str] = None) -> CompostRecord:
        row = self.database.one("SELECT * FROM composts WHERE compost_id = ? AND tenant_id = ?", (compost_id, tenant_id))
        if row is None:
            raise KeyError("compost not found")
        return self._row_to_compost(row, tenant_id, project_id=project_id)

    def consolidate(self, tenant_id: str) -> Dict[str, int]:
        """A budgetable maintenance pass; it makes no quality promotion itself."""
        rows = self.database.all(
            "SELECT c.compost_id, a.content_hash FROM composts c JOIN artifacts a ON a.artifact_id = c.artifact_id WHERE c.tenant_id = ?",
            (tenant_id,),
        )
        duplicate_count = len(rows) - len({row["content_hash"] for row in rows})
        return {
            "compost_records": len(rows),
            "duplicate_derived_artifacts": max(0, duplicate_count),
            "skills_checked": len(self.skills(tenant_id=tenant_id)),
        }

    def _save_skill(self, skill: SkillManifest) -> None:
        self.database.execute(
            "INSERT OR REPLACE INTO skills(skill_id, version, tenant_id, access_json, state, manifest_json, rollback_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                skill.skill_id,
                skill.version,
                skill.tenant_id,
                json_dumps(skill.access_scope),
                skill.state.value,
                json_dumps(skill),
                skill.rollback_version,
                skill.created_at.isoformat(),
            ),
        )

    def _row_to_compost(
        self, row: Dict[str, Any], tenant_id: str, *, project_id: Optional[str] = None
    ) -> CompostRecord:
        scope = AccessScope.model_validate(json_loads(row["access_json"]))
        if not _scope_allows(scope, tenant_id, project_id):
            raise LearningSafetyError("compost belongs to another access scope")
        return CompostRecord(
            compost_id=row["compost_id"],
            tenant_id=row["tenant_id"],
            # Deleted derivatives remain auditable as metadata but their bytes
            # cannot be read through the normal object-store path.
            artifact_ref=self.store.get_metadata(
                row["artifact_id"],
                tenant_id,
                project_id=scope.project_id,
                include_deleted=row["status"] == "deleted",
            ),
            trace_id=row["trace_id"],
            access_scope=scope,
            status=row["status"],
            created_at=row["created_at"],
        )


class GenomeRegistry:
    """Stores reproducible component descriptions, never live Hive/session state."""

    def __init__(self, database: SqliteDatabase) -> None:
        self.database = database

    def register(self, genome: GenomeManifest) -> GenomeManifest:
        material = genome.model_dump(mode="json", exclude={"content_hash"})
        content_hash = hashlib.sha256(json_dumps(material).encode("utf-8")).hexdigest()
        genome.content_hash = content_hash
        self.database.execute(
            "INSERT OR REPLACE INTO genomes(genome_id, version, manifest_json, content_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (genome.genome_id, genome.version, json_dumps(genome), content_hash, genome.created_at.isoformat()),
        )
        return genome

    def materialize(self, genome_id: str, version: str) -> GenomeManifest:
        row = self.database.one("SELECT manifest_json, content_hash FROM genomes WHERE genome_id = ? AND version = ?", (genome_id, version))
        if row is None:
            raise KeyError("genome not found")
        genome = GenomeManifest.model_validate(json_loads(row["manifest_json"]))
        material = genome.model_dump(mode="json", exclude={"content_hash"})
        actual = hashlib.sha256(json_dumps(material).encode("utf-8")).hexdigest()
        if actual != row["content_hash"]:
            raise LearningSafetyError("genome patch or manifest checksum is corrupted")
        return genome


class EvolutionEngine:
    """Only compares externally-evaluated deterministic candidates; it runs no code."""

    @staticmethod
    def pareto_frontier(candidates: Sequence[Dict[str, float]]) -> list[Dict[str, float]]:
        required = {"quality", "latency", "memory", "risk"}
        if any(not required.issubset(candidate) for candidate in candidates):
            raise ValueError("candidate requires quality, latency, memory and risk metrics")
        frontier: list[Dict[str, float]] = []
        for candidate in candidates:
            dominated = False
            for other in candidates:
                if other is candidate:
                    continue
                no_worse = (
                    other["quality"] >= candidate["quality"]
                    and other["latency"] <= candidate["latency"]
                    and other["memory"] <= candidate["memory"]
                    and other["risk"] <= candidate["risk"]
                )
                strictly_better = (
                    other["quality"] > candidate["quality"]
                    or other["latency"] < candidate["latency"]
                    or other["memory"] < candidate["memory"]
                    or other["risk"] < candidate["risk"]
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                frontier.append(dict(candidate))
        return frontier


def _normalise_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _task_class(goal: str) -> str:
    # Stable enough to group a prototype corpus, deliberately not an inferred
    # truth about task semantics.
    terms = re.findall(r"[\w-]{3,}", goal.lower())
    return ":".join(terms[:3]) or "generic-text-task"


def _scope_allows(scope: AccessScope, tenant_id: str, project_id: Optional[str]) -> bool:
    return scope.visibility == "global" or (
        scope.tenant_id == tenant_id and (scope.visibility != "project" or scope.project_id == project_id)
    )
