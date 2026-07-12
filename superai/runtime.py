"""Durable in-process command runtime with cooperative cancellation.

Delivery is at-least-once. A unique tenant/idempotency key, durable work item
state and a transactional outbox make a replay after a process crash safe for
handlers that put their externally-visible mutation behind the same key.
"""

from __future__ import annotations

import random
import threading
from datetime import timedelta
from typing import Any, Callable, Dict, Optional

from .contracts import DomainEvent, ErrorEnvelope, SpanStatus, TaskState, WorkItem, utcnow
from .database import SqliteDatabase, json_dumps, json_loads
from .observability import TraceRecorder, redact


class RuntimeErrorBase(Exception):
    code = "runtime_error"
    retryable = False


class RetryableError(RuntimeErrorBase):
    code = "transient_failure"
    retryable = True


class BudgetExceeded(RuntimeErrorBase):
    code = "budget_exceeded"


class Cancelled(RuntimeErrorBase):
    code = "cancelled"


Handler = Callable[[WorkItem, "RuntimeContext"], Any]
StateListener = Callable[[WorkItem], None]


class RuntimeContext:
    def __init__(self, runtime: "CommandRuntime", item: WorkItem) -> None:
        self.runtime = runtime
        self.item = item
        self.steps = 0
        self.events = 0

    def checkpoint(self, steps: int = 1) -> None:
        self.steps += steps
        if self.steps > self.item.budget.step_limit:
            raise BudgetExceeded("step budget exhausted")
        self.events += 1
        if self.events > self.item.budget.event_limit:
            raise BudgetExceeded("event budget exhausted")
        if self.item.deadline_at and utcnow() >= self.item.deadline_at:
            raise BudgetExceeded("task deadline exceeded")
        if self.runtime.is_cancelled(self.item.command_id):
            raise Cancelled("command cancelled by user")

    def emit(self, kind: str, payload: Dict[str, Any]) -> None:
        self.checkpoint(0)
        self.runtime.traces.record_event(
            DomainEvent(
                id="evt-envelope-" + self.item.command_id,
                task_id=self.item.task_id,
                trace_id=self.item.trace_id,
                tenant_id=self.item.tenant_id,
                kind=kind,
                producer="runtime",
                payload=payload,
                causation_id=self.item.command_id,
                correlation_id=self.item.command_id,
            )
        )


class CommandRuntime:
    def __init__(
        self,
        database: SqliteDatabase,
        traces: TraceRecorder,
        *,
        retry_base_seconds: float = 0.05,
    ) -> None:
        self.database = database
        self.traces = traces
        self.retry_base_seconds = retry_base_seconds
        self.handlers: Dict[str, Handler] = {}
        self._state_listener: Optional[StateListener] = None
        self._wakeup = threading.Event()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

    def register(self, name: str, handler: Handler) -> None:
        if name in self.handlers:
            raise ValueError("handler already registered: %s" % name)
        self.handlers[name] = handler

    def set_state_listener(self, listener: StateListener) -> None:
        self._state_listener = listener

    def start_worker(self, *, idle_wait_seconds: float = 0.05) -> None:
        """Run exactly one durable queue worker for the local monolith."""
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                processed = self.run_once()
                if processed is None:
                    self._wakeup.wait(idle_wait_seconds)
                    self._wakeup.clear()

        self._worker = threading.Thread(target=loop, name="superai-runtime", daemon=True)
        self._worker.start()

    def stop_worker(self, timeout_seconds: float = 2.0) -> None:
        self._stop.set()
        self._wakeup.set()
        if self._worker and self._worker is not threading.current_thread():
            self._worker.join(timeout_seconds)
        self._worker = None

    def enqueue(self, item: WorkItem) -> WorkItem:
        """Persist once; a duplicate request returns the original work item."""
        with self.database.transaction() as connection:
            result, inserted = self.enqueue_in_transaction(connection, item)
        if inserted:
            self.notify_enqueued(result)
        return result

    def enqueue_in_transaction(self, connection: Any, item: WorkItem) -> tuple[WorkItem, bool]:
        """Append work and CommandQueued to a caller-owned aggregate transaction."""
        existing = connection.execute(
            "SELECT * FROM work_items WHERE tenant_id = ? AND idempotency_key = ?",
            (item.tenant_id, item.idempotency_key),
        ).fetchone()
        if existing:
            return self._row_to_item(dict(existing)), False
        now = utcnow()
        connection.execute(
            "INSERT INTO work_items(command_id, task_id, trace_id, handler, payload_json, status, attempt, max_attempts, priority, scheduled_at, idempotency_key, budget_json, deadline_at, last_error_json, tenant_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item.command_id,
                item.task_id,
                item.trace_id,
                item.handler,
                json_dumps(item.payload),
                item.status.value,
                item.attempt,
                item.max_attempts,
                item.priority,
                item.scheduled_at.isoformat(),
                item.idempotency_key,
                json_dumps(item.budget),
                item.deadline_at.isoformat() if item.deadline_at else None,
                None,
                item.tenant_id,
                now.isoformat(),
                now.isoformat(),
            ),
        )
        event = DomainEvent(
            id="evt-envelope-" + item.command_id,
            task_id=item.task_id,
            trace_id=item.trace_id,
            tenant_id=item.tenant_id,
            kind="CommandQueued",
            producer="runtime",
            payload={"command_id": item.command_id, "handler": item.handler},
            causation_id=item.command_id,
            correlation_id=item.command_id,
        )
        self._insert_event(connection, event)
        return item, True

    def notify_enqueued(self, item: WorkItem) -> None:
        """Wake the worker only after the enclosing transaction committed."""
        self._notify(item)
        self._wakeup.set()

    def recover_unfinished(self) -> int:
        """Only incomplete work becomes runnable after a restart."""
        now = utcnow().isoformat()
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE work_items SET status = ?, scheduled_at = ?, updated_at = ? WHERE status = ?",
                (TaskState.QUEUED.value, now, now, TaskState.RUNNING.value),
            )
            return cursor.rowcount

    def cancel(self, task_id: str, tenant_id: str) -> bool:
        now = utcnow().isoformat()
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM work_items WHERE task_id = ? AND tenant_id = ? AND status IN (?, ?)",
                (task_id, tenant_id, TaskState.QUEUED.value, TaskState.RUNNING.value),
            ).fetchall()
            if not rows:
                return False
            connection.execute(
                "UPDATE work_items SET status = ?, updated_at = ? WHERE task_id = ? AND tenant_id = ? AND status IN (?, ?)",
                (
                    TaskState.CANCELLED.value,
                    now,
                    task_id,
                    tenant_id,
                    TaskState.QUEUED.value,
                    TaskState.RUNNING.value,
                ),
            )
            for row in rows:
                event = DomainEvent(
                    id="evt-envelope-cancel-" + row["command_id"],
                    task_id=task_id,
                    trace_id=row["trace_id"],
                    tenant_id=tenant_id,
                    kind="CommandCancellationRequested",
                    producer="runtime",
                    payload={"command_id": row["command_id"]},
                    causation_id=row["command_id"],
                    correlation_id=row["command_id"],
                )
                self._insert_event(connection, event)
        for row in rows:
            item = self._row_to_item(dict(row))
            item.status = TaskState.CANCELLED
            self._notify(item)
        self._wakeup.set()
        return True

    def is_cancelled(self, command_id: str) -> bool:
        row = self.database.one("SELECT status FROM work_items WHERE command_id = ?", (command_id,))
        return row is not None and row["status"] == TaskState.CANCELLED.value

    def run_once(self) -> Optional[WorkItem]:
        item = self._claim_next()
        if item is None:
            return None
        if item.handler not in self.handlers:
            return self._finish_failure(
                item,
                ErrorEnvelope(code="handler_not_found", message="No handler registered: %s" % item.handler),
                terminal=True,
            )
        context = RuntimeContext(self, item)
        span = self.traces.start_span(
            trace_id=item.trace_id,
            component="CommandRuntime",
            operation=item.handler,
            input_summary={"command_id": item.command_id, "attempt": item.attempt},
            budget_before=item.budget,
        )
        try:
            context.checkpoint(1)
            result = self.handlers[item.handler](item, context)
            context.checkpoint(0)
        except Cancelled as exc:
            error = ErrorEnvelope(code=exc.code, message=str(exc))
            self.traces.finish_span(span, status=SpanStatus.CANCELLED, error=error)
            return self._set_state(item, TaskState.CANCELLED, error=error)
        except BudgetExceeded as exc:
            error = ErrorEnvelope(code=exc.code, message=str(exc))
            self.traces.fail_span(span, error)
            return self._finish_failure(item, error, terminal=True)
        except RuntimeErrorBase as exc:
            error = ErrorEnvelope(code=exc.code, message=str(exc), retryable=exc.retryable)
            self.traces.fail_span(span, error)
            return self._finish_failure(item, error, terminal=not exc.retryable)
        except Exception as exc:  # errors cross a normalized boundary
            error = ErrorEnvelope(code=getattr(exc, "code", "handler_error"), message=str(exc), retryable=False)
            self.traces.fail_span(span, error)
            return self._finish_failure(item, error, terminal=True)
        self.traces.finish_span(
            span,
            output_summary={"result_type": type(result).__name__, "steps": context.steps},
            budget_after=item.budget,
        )
        return self._set_state(item, TaskState.SUCCEEDED)

    def run_until_idle(self, limit: int = 100) -> int:
        processed = 0
        while processed < limit:
            if self.run_once() is None:
                break
            processed += 1
        return processed

    def work_item(self, command_id: str) -> Optional[WorkItem]:
        row = self.database.one("SELECT * FROM work_items WHERE command_id = ?", (command_id,))
        return self._row_to_item(row) if row else None

    def dead_letters(self, tenant_id: str) -> list[WorkItem]:
        return [
            self._row_to_item(row)
            for row in self.database.all(
                "SELECT * FROM work_items WHERE tenant_id = ? AND status = ? ORDER BY updated_at DESC",
                (tenant_id, TaskState.DEAD_LETTER.value),
            )
        ]

    def _claim_next(self) -> Optional[WorkItem]:
        now = utcnow()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM work_items WHERE status = ? AND scheduled_at <= ? ORDER BY priority ASC, scheduled_at ASC LIMIT 1",
                (TaskState.QUEUED.value, now.isoformat()),
            ).fetchone()
            if row is None:
                return None
            item = self._row_to_item(dict(row))
            if item.deadline_at and now >= item.deadline_at:
                # It is made running here so terminal transition remains visible.
                next_state = TaskState.RUNNING.value
            else:
                next_state = TaskState.RUNNING.value
            cursor = connection.execute(
                "UPDATE work_items SET status = ?, attempt = ?, updated_at = ? WHERE command_id = ? AND status = ?",
                (next_state, item.attempt + 1, now.isoformat(), item.command_id, TaskState.QUEUED.value),
            )
            if cursor.rowcount != 1:
                return None
            item.status = TaskState.RUNNING
            item.attempt += 1
        self._notify(item)
        return item

    def _finish_failure(self, item: WorkItem, error: ErrorEnvelope, *, terminal: bool) -> WorkItem:
        if not terminal and item.attempt < item.max_attempts:
            # Bounded backoff with jitter avoids a hot retry loop. It stays
            # deterministic enough for the local diagnostic flow.
            delay = self.retry_base_seconds * (2 ** max(0, item.attempt - 1))
            delay += random.uniform(0, delay / 4 if delay else 0)
            item.scheduled_at = utcnow() + timedelta(seconds=delay)
            return self._set_state(item, TaskState.QUEUED, error=error)
        # Budget/contract failures are terminal task failures; an unexpected
        # handler or exhausted retry needs an operator-visible dead letter.
        state = TaskState.FAILED if error.code in {"budget_exceeded", "cancelled", "insufficient_evidence"} else TaskState.DEAD_LETTER
        return self._set_state(item, state, error=error)

    def _set_state(self, item: WorkItem, state: TaskState, *, error: Optional[ErrorEnvelope] = None) -> WorkItem:
        item.status = state
        item.last_error = error
        now = utcnow()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE work_items SET status = ?, scheduled_at = ?, last_error_json = ?, updated_at = ? WHERE command_id = ?",
                (
                    state.value,
                    item.scheduled_at.isoformat(),
                    json_dumps(error) if error else None,
                    now.isoformat(),
                    item.command_id,
                ),
            )
            event = DomainEvent(
                id="evt-envelope-state-" + item.command_id + "-" + state.value + "-" + str(item.attempt),
                task_id=item.task_id,
                trace_id=item.trace_id,
                tenant_id=item.tenant_id,
                kind="Command" + state.value.title().replace("_", ""),
                producer="runtime",
                payload={"command_id": item.command_id, "status": state.value, "error": error.model_dump(mode="json") if error else None},
                causation_id=item.command_id,
                correlation_id=item.command_id,
            )
            self._insert_event(connection, event)
        self._notify(item)
        if state == TaskState.QUEUED:
            self._wakeup.set()
        return item

    def _notify(self, item: WorkItem) -> None:
        if self._state_listener:
            self._state_listener(item)

    def _insert_event(self, connection: Any, event: DomainEvent) -> None:
        sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM domain_events WHERE trace_id = ?", (event.trace_id,)
        ).fetchone()[0]
        connection.execute(
            "INSERT OR IGNORE INTO domain_events(event_id, trace_id, task_id, sequence, kind, producer, causation_id, payload_json, occurred_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.trace_id,
                event.task_id,
                sequence,
                event.kind,
                event.producer,
                event.causation_id,
                json_dumps(redact(event.payload)),
                event.occurred_at.isoformat(),
            ),
        )
        connection.execute(
            "INSERT OR IGNORE INTO outbox(event_id, topic, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (
                event.event_id,
                event.kind,
                json_dumps(redact(event.model_dump(mode="json"))),
                event.occurred_at.isoformat(),
            ),
        )

    @staticmethod
    def _row_to_item(row: Dict[str, Any]) -> WorkItem:
        return WorkItem(
            command_id=row["command_id"],
            task_id=row["task_id"],
            trace_id=row["trace_id"],
            handler=row["handler"],
            payload=json_loads(row["payload_json"], {}),
            status=row["status"],
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            priority=row["priority"],
            scheduled_at=row["scheduled_at"],
            idempotency_key=row["idempotency_key"],
            budget=json_loads(row["budget_json"], {}),
            deadline_at=row["deadline_at"],
            last_error=json_loads(row["last_error_json"]) if row.get("last_error_json") else None,
            tenant_id=row["tenant_id"],
        )
