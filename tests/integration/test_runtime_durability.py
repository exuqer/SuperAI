from __future__ import annotations

import threading

from superai.contracts import TaskState, WorkItem
from superai.database import SqliteDatabase
from superai.observability import TraceRecorder
from superai.runtime import CommandRuntime, RetryableError


def test_retry_is_bounded_and_reuses_the_same_durable_work_item(tmp_path) -> None:
    database = SqliteDatabase(tmp_path / "runtime.sqlite3")
    runtime = CommandRuntime(database, TraceRecorder(database), retry_base_seconds=0)
    calls = 0

    def handler(_item, _context):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableError("temporary source failure")
        return {"ok": True}

    runtime.register("test", handler)
    queued = runtime.enqueue(
        WorkItem(
            task_id="task-retry",
            trace_id="trace-retry",
            handler="test",
            payload={},
            idempotency_key="retry-key",
            tenant_id="tenant-a",
            max_attempts=2,
        )
    )

    first = runtime.run_once()
    assert first is not None
    assert first.status is TaskState.QUEUED
    assert first.attempt == 1

    second = runtime.run_once()
    assert second is not None
    assert second.command_id == queued.command_id
    assert second.status is TaskState.SUCCEEDED
    assert second.attempt == 2
    assert calls == 2
    assert [event["kind"] for event in runtime.traces.trace("trace-retry")["events"]] == [
        "CommandQueued",
        "CommandQueued",
        "CommandSucceeded",
    ]
    database.close()


def test_restart_recovers_running_item_but_never_replays_succeeded_work(tmp_path) -> None:
    database = SqliteDatabase(tmp_path / "runtime.sqlite3")
    original = CommandRuntime(database, TraceRecorder(database))
    item = original.enqueue(
        WorkItem(
            task_id="task-recovery",
            trace_id="trace-recovery",
            handler="test",
            payload={},
            idempotency_key="recover-key",
            tenant_id="tenant-a",
        )
    )
    claimed = original._claim_next()
    assert claimed is not None and claimed.status is TaskState.RUNNING

    recovered = CommandRuntime(database, TraceRecorder(database))
    assert recovered.recover_unfinished() == 1
    recovered.register("test", lambda _item, _context: {"recovered": True})
    completed = recovered.run_once()
    assert completed is not None and completed.status is TaskState.SUCCEEDED

    # A second restart sees no completed work as runnable.
    assert recovered.recover_unfinished() == 0
    assert recovered.run_once() is None
    durable = recovered.work_item(item.command_id)
    assert durable is not None and durable.status is TaskState.SUCCEEDED
    database.close()


def test_cancellation_of_running_work_is_observed_at_the_next_checkpoint(tmp_path) -> None:
    database = SqliteDatabase(tmp_path / "runtime.sqlite3")
    runtime = CommandRuntime(database, TraceRecorder(database))
    started = threading.Event()
    release = threading.Event()

    def handler(_item, context):
        started.set()
        assert release.wait(1)
        context.checkpoint()
        return {"should_not": "complete"}

    runtime.register("blocking", handler)
    item = runtime.enqueue(
        WorkItem(
            task_id="task-cancel",
            trace_id="trace-cancel",
            handler="blocking",
            payload={},
            idempotency_key="cancel-key",
            tenant_id="tenant-a",
        )
    )
    worker = threading.Thread(target=runtime.run_once)
    worker.start()
    assert started.wait(1)
    assert runtime.cancel(item.task_id, "tenant-a")
    release.set()
    worker.join(1)

    final = runtime.work_item(item.command_id)
    assert final is not None and final.status is TaskState.CANCELLED
    assert runtime.traces.trace(item.trace_id)["events"][-1]["kind"] == "CommandCancelled"
    database.close()
