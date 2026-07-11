"""Composition root and the first observable end-to-end task use case."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from .contracts import (
    AnswerEnvelope,
    ErrorEnvelope,
    HiveView,
    TaskContract,
    TaskState,
    TaskSubmission,
    TaskView,
    WorkItem,
    new_id,
    utcnow,
)
from .cosmos import Cosmos
from .database import SqliteDatabase, json_dumps, json_loads
from .execution import Atlas, CriticSystem, ExecutionEngine, Planner, TextCodec
from .hive import HiveManager
from .learning import ExperienceCompiler, GenomeRegistry
from .observability import TraceRecorder
from .runtime import CommandRuntime, RuntimeContext
from .storage import AccessDenied, ArtifactNotFound, ObjectStore


@dataclass
class ServiceConfig:
    data_dir: Path

    @classmethod
    def from_environment(cls) -> "ServiceConfig":
        value = os.environ.get("SUPERAI_DATA_DIR", ".superai")
        return cls(data_dir=Path(value).expanduser().resolve())


class SuperAIService:
    """Wires explicit components without turning routes into domain services."""

    def __init__(self, config: Optional[ServiceConfig] = None) -> None:
        self.config = config or ServiceConfig.from_environment()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.database = SqliteDatabase(self.config.data_dir / "superai.sqlite3")
        self.store = ObjectStore(self.config.data_dir / "objects", self.database)
        self.store.reconcile_orphans()
        self.traces = TraceRecorder(self.database)
        self.runtime = CommandRuntime(self.database, self.traces)
        self.hives = HiveManager(self.database, self.store, self.traces)
        self.cosmos = Cosmos(self.database, self.store)
        self.atlas = Atlas(self.database)
        self.atlas.register_builtin_capabilities()
        self.planner = Planner(self.database, self.atlas)
        self.execution = ExecutionEngine(
            cosmos=self.cosmos,
            hives=self.hives,
            planner=self.planner,
            critics=CriticSystem(),
            codec=TextCodec(),
            traces=self.traces,
        )
        self.learning = ExperienceCompiler(self.database, self.store, self.cosmos)
        self.genomes = GenomeRegistry(self.database)
        self.runtime.register("execute_task", self._execute_task)
        self.runtime.set_state_listener(self._on_work_state)
        self.runtime.recover_unfinished()
        self.runtime.start_worker()

    def close(self) -> None:
        self.runtime.stop_worker()
        self.database.close()

    def submit_task(self, submission: TaskSubmission, *, idempotency_key: Optional[str] = None, execute_now: bool = True) -> TaskView:
        # Idempotency is an operation property supplied by the caller. Equal
        # conversational messages are legitimate separate turns, so they must
        # not silently collapse without an explicit key.
        key = idempotency_key or new_id("op")
        trace_id = new_id("trace")
        contract = TaskContract(
            tenant_id=submission.tenant_id,
            user_id=submission.user_id,
            conversation_id=submission.conversation_id or new_id("conv"),
            project_id=submission.project_id,
            goal=submission.message,
            budget=submission.budget,
            source_policy=submission.source_policy,
            success_criteria=["Ответ соответствует контракту и содержит допустимое происхождение."],
        )
        now = utcnow()
        item = WorkItem(
            task_id=contract.task_id,
            trace_id=trace_id,
            handler="execute_task",
            payload={"contract": contract.model_dump(mode="json")},
            idempotency_key=key,
            budget=contract.budget,
            tenant_id=contract.tenant_id,
            deadline_at=now + timedelta(milliseconds=contract.budget.time_ms),
        )
        existing_task_id: Optional[str] = None
        inserted = False
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT task_id FROM work_items WHERE tenant_id = ? AND idempotency_key = ?",
                (submission.tenant_id, key),
            ).fetchone()
            if existing:
                existing_task_id = existing["task_id"]
            else:
                connection.execute(
                    "INSERT INTO tasks(task_id, tenant_id, trace_id, hive_id, status, contract_json, answer_json, error_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, NULL, ?, ?, NULL, NULL, ?, ?)",
                    (
                        contract.task_id,
                        contract.tenant_id,
                        trace_id,
                        TaskState.QUEUED.value,
                        json_dumps(contract),
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
                _, inserted = self.runtime.enqueue_in_transaction(connection, item)
        if existing_task_id:
            return self.task(existing_task_id, submission.tenant_id)
        if inserted:
            self.runtime.notify_enqueued(item)
        if execute_now:
            self.runtime.run_until_idle()
        return self.task(contract.task_id, contract.tenant_id)

    def task(
        self,
        task_id: str,
        tenant_id: str,
        project_id: Optional[str] = None,
        *,
        enforce_project: bool = False,
    ) -> TaskView:
        row = self.database.one("SELECT * FROM tasks WHERE task_id = ? AND tenant_id = ?", (task_id, tenant_id))
        if row is None:
            raise KeyError("task not found")
        contract = TaskContract.model_validate(json_loads(row["contract_json"]))
        if enforce_project and contract.project_id != project_id:
            raise AccessDenied("task belongs to another project")
        return TaskView(
            task_id=row["task_id"],
            trace_id=row["trace_id"],
            hive_id=row["hive_id"],
            status=row["status"],
            contract=contract,
            answer=AnswerEnvelope.model_validate(json_loads(row["answer_json"])) if row["answer_json"] else None,
            error=ErrorEnvelope.model_validate(json_loads(row["error_json"])) if row["error_json"] else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def cancel_task(self, task_id: str, tenant_id: str, project_id: Optional[str] = None) -> TaskView:
        self.task(task_id, tenant_id, project_id, enforce_project=True)
        if not self.runtime.cancel(task_id, tenant_id):
            # A completed task has nothing to cancel but its state is still an
            # accurate answer to a repeated cancel request.
            return self.task(task_id, tenant_id, project_id, enforce_project=True)
        return self.task(task_id, tenant_id, project_id, enforce_project=True)

    def trace(self, trace_id: str, tenant_id: str, project_id: Optional[str] = None) -> Dict[str, Any]:
        task = self.database.one("SELECT task_id FROM tasks WHERE trace_id = ? AND tenant_id = ?", (trace_id, tenant_id))
        if task is None:
            raise KeyError("trace not found")
        self.task(task["task_id"], tenant_id, project_id, enforce_project=True)
        return self.traces.trace(trace_id)

    def hive(self, hive_id: str, tenant_id: str, project_id: Optional[str] = None) -> HiveView:
        return self.hives.get(hive_id, tenant_id, project_id, enforce_project=True)

    def health(self) -> Dict[str, Any]:
        counts = self.database.all("SELECT status, COUNT(*) AS count FROM work_items GROUP BY status")
        return {
            "status": "ok",
            "runtime": "sqlite-in-process",
            "data_dir": str(self.config.data_dir),
            "work_items": {row["status"]: row["count"] for row in counts},
        }

    def meta(self) -> Dict[str, Any]:
        return {
            "service": "superai",
            "api_version": "v1",
            "schema_version": "1.0",
            "runtime": "modular-monolith",
            "capabilities": [item.capability_id for item in self.atlas.manifests()],
        }

    def _execute_task(self, item: WorkItem, context: RuntimeContext) -> AnswerEnvelope:
        contract = TaskContract.model_validate(item.payload["contract"])
        committed = self.database.one(
            "SELECT hive_id, answer_json FROM tasks WHERE task_id = ? AND tenant_id = ?",
            (contract.task_id, contract.tenant_id),
        )
        if committed and committed["answer_json"]:
            # The handler may be delivered again after a crash between its
            # materialized state/outbox commit and runtime acknowledgement.
            # A committed answer is the durable idempotency marker.
            return AnswerEnvelope.model_validate(json_loads(committed["answer_json"]))
        analysis_span = self.traces.start_span(
            trace_id=item.trace_id,
            component="RequestAnalyzer",
            operation="BUILD_TASK_CONTRACT",
            input_summary={"task_id": contract.task_id, "goal_length": len(contract.goal)},
            budget_before=contract.budget,
        )
        context.checkpoint(1)
        if committed and committed["hive_id"]:
            hive = self.hives.get(committed["hive_id"], contract.tenant_id)
            decision, alternatives = "continue", []
        else:
            hive, decision, alternatives = self.hives.select_or_create(contract, item.trace_id)
        self.traces.finish_span(
            analysis_span,
            output_summary={"hive_id": hive.hive_id, "hive_decision": decision, "candidates": len(alternatives)},
            budget_after=contract.budget,
        )
        self.database.execute(
            "UPDATE tasks SET hive_id = ?, contract_json = ?, updated_at = ? WHERE task_id = ?",
            (hive.hive_id, json_dumps(contract), utcnow().isoformat(), contract.task_id),
        )
        self.hives.add_entry(
            hive.hive_id,
            contract.tenant_id,
            store_name="WorkingContextStore",
            content_type="user_message",
            content={"message": contract.goal, "task_id": contract.task_id},
            relevance=0.9,
            protected=False,
            trace_id=item.trace_id,
            idempotency_key="task:%s:message" % contract.task_id,
        )
        answer = self.execution.execute(contract, hive.hive_id, item.trace_id, context)
        self.database.execute(
            "UPDATE tasks SET answer_json = ?, updated_at = ? WHERE task_id = ?",
            (json_dumps(answer), utcnow().isoformat(), contract.task_id),
        )
        return answer

    def _on_work_state(self, item: WorkItem) -> None:
        # The task aggregate mirrors durable work state; its answer is written
        # by the handler before ``succeeded`` is committed.
        error = json_dumps(item.last_error) if item.last_error else None
        self.database.execute(
            "UPDATE tasks SET status = ?, error_json = ?, updated_at = ? WHERE task_id = ?",
            (item.status.value, error, utcnow().isoformat(), item.task_id),
        )
