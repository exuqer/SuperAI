"""Live HTTP contract tests for the deterministic task-to-trace slice."""

from __future__ import annotations

from collections.abc import Iterator
from time import sleep

import pytest
from fastapi.testclient import TestClient

from superai.api import create_app
from superai.contracts import ErrorEnvelope, HiveView, TaskState, TaskView, TraceSpan
from superai.service import ServiceConfig


@pytest.fixture()
def client(tmp_path) -> Iterator[TestClient]:
    """Give each test a fresh durable service, including its FastAPI lifespan."""
    app = create_app(ServiceConfig(data_dir=tmp_path / "superai-data"))
    with TestClient(app) as test_client:
        yield test_client


def _submission(*, tenant_id: str = "tenant-live-test") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "message": "Сформируй ответ из разрешённого контекста.",
        "tenant_id": tenant_id,
        "conversation_id": "conv-live-test",
        "project_id": "project-live-test",
        "budget": {"time_ms": 10_000, "step_limit": 32, "memory_bytes": 8_192, "event_limit": 64},
    }


def _wait_for_terminal_task(client: TestClient, view: TaskView, *, tenant_id: str) -> TaskView:
    """Accept either an already-completed local task or the documented 202/poll flow."""
    terminal = {TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED, TaskState.DEAD_LETTER}
    current = view
    for _ in range(50):
        if current.status in terminal:
            return current
        sleep(0.01)
        project_id = current.contract.project_id if current.contract else None
        response = client.get(
            f"/api/v1/tasks/{current.task_id}",
            params={"project_id": project_id} if project_id else None,
            headers={"X-Tenant-Id": tenant_id},
        )
        assert response.status_code == 200
        current = TaskView.model_validate(response.json())
    pytest.fail(f"task {view.task_id} did not reach a terminal state")


def test_live_task_can_be_retrieved_with_its_complete_trace(client: TestClient) -> None:
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    submitted = client.post("/api/v1/tasks", json=_submission())

    assert submitted.status_code == 202
    submitted_view = TaskView.model_validate(submitted.json())
    submitted_view = _wait_for_terminal_task(client, submitted_view, tenant_id="tenant-live-test")
    assert submitted_view.status is TaskState.SUCCEEDED
    assert submitted_view.contract is not None
    assert submitted_view.contract.conversation_id == "conv-live-test"
    assert submitted_view.contract.project_id == "project-live-test"
    assert submitted_view.hive_id
    assert submitted_view.answer is not None
    assert submitted_view.answer.task_id == submitted_view.task_id
    assert submitted_view.answer.trace_id == submitted_view.trace_id
    assert submitted_view.answer.answer.strip()

    retrieved = client.get(
        f"/api/v1/tasks/{submitted_view.task_id}",
        params={"project_id": "project-live-test"},
        headers={"X-Tenant-Id": "tenant-live-test"},
    )
    assert retrieved.status_code == 200
    retrieved_view = TaskView.model_validate(retrieved.json())
    assert retrieved_view.task_id == submitted_view.task_id
    assert retrieved_view.trace_id == submitted_view.trace_id
    assert retrieved_view.status is TaskState.SUCCEEDED

    trace_response = client.get(
        f"/api/v1/traces/{submitted_view.trace_id}",
        params={"project_id": "project-live-test"},
        headers={"X-Tenant-Id": "tenant-live-test"},
    )
    assert trace_response.status_code == 200
    trace = trace_response.json()
    assert trace["trace_id"] == submitted_view.trace_id
    spans = [TraceSpan.model_validate(item) for item in trace["spans"]]
    assert spans
    assert {span.trace_id for span in spans} == {submitted_view.trace_id}
    assert {span.status.value for span in spans} == {"succeeded"}
    assert {span.operation for span in spans} >= {
        "execute_task",
        "BUILD_TASK_CONTRACT",
        "RETRIEVE_CLAIMS",
        "BUILD_PLAN",
        "FORMAT_TEXT",
        "VERIFY",
    }

    event_kinds = [event["kind"] for event in trace["events"]]
    assert event_kinds[0] == "CommandQueued"
    assert event_kinds[-1] == "CommandSucceeded"
    assert {event["sequence"] for event in trace["events"]} == set(range(1, len(trace["events"]) + 1))
    assert {"id", "event_id", "tenant_id", "task_id", "trace_id", "schema_version", "correlation_id", "payload"} <= set(trace["events"][0])

    hive_response = client.get(
        f"/api/v1/hives/{submitted_view.hive_id}",
        params={"project_id": "project-live-test"},
        headers={"X-Tenant-Id": "tenant-live-test"},
    )
    assert hive_response.status_code == 200
    hive = HiveView.model_validate(hive_response.json())
    assert hive.hive_id == submitted_view.hive_id
    assert hive.contract.task_id == submitted_view.task_id


def test_post_tasks_reuses_one_task_and_trace_for_the_same_idempotency_key(client: TestClient) -> None:
    headers = {"Idempotency-Key": "live-api-idempotency-key"}

    first = client.post("/api/v1/tasks", headers=headers, json=_submission())
    assert first.status_code == 202
    first_view = _wait_for_terminal_task(client, TaskView.model_validate(first.json()), tenant_id="tenant-live-test")
    first_trace = client.get(
        f"/api/v1/traces/{first_view.trace_id}", params={"project_id": "project-live-test"}, headers={"X-Tenant-Id": "tenant-live-test"}
    )
    assert first_trace.status_code == 200

    replay = client.post("/api/v1/tasks", headers=headers, json=_submission())
    assert replay.status_code == 202
    replay_view = _wait_for_terminal_task(client, TaskView.model_validate(replay.json()), tenant_id="tenant-live-test")
    replay_trace = client.get(
        f"/api/v1/traces/{replay_view.trace_id}", params={"project_id": "project-live-test"}, headers={"X-Tenant-Id": "tenant-live-test"}
    )
    assert replay_trace.status_code == 200

    assert replay_view.task_id == first_view.task_id
    assert replay_view.trace_id == first_view.trace_id
    assert replay_view.hive_id == first_view.hive_id
    assert replay_view.status is TaskState.SUCCEEDED
    assert len(replay_trace.json()["spans"]) == len(first_trace.json()["spans"])
    assert len(replay_trace.json()["events"]) == len(first_trace.json()["events"])


def test_idempotency_and_trace_access_are_scoped_to_the_tenant(client: TestClient) -> None:
    headers = {"Idempotency-Key": "shared-key-across-tenants"}

    tenant_a = _wait_for_terminal_task(
        client,
        TaskView.model_validate(client.post("/api/v1/tasks", headers=headers, json=_submission(tenant_id="tenant-a")).json()),
        tenant_id="tenant-a",
    )
    tenant_b = _wait_for_terminal_task(
        client,
        TaskView.model_validate(client.post("/api/v1/tasks", headers=headers, json=_submission(tenant_id="tenant-b")).json()),
        tenant_id="tenant-b",
    )

    assert tenant_a.task_id != tenant_b.task_id
    assert tenant_a.trace_id != tenant_b.trace_id

    denied = client.get(f"/api/v1/traces/{tenant_a.trace_id}", params={"project_id": "project-live-test"}, headers={"X-Tenant-Id": "tenant-b"})
    assert denied.status_code == 404
    error = ErrorEnvelope.model_validate(denied.json())
    assert error.code == "not_found"


def test_authenticated_tenant_header_is_authoritative_for_task_creation(client: TestClient) -> None:
    response = client.post(
        "/api/v1/tasks",
        headers={"X-Tenant-Id": "header-tenant"},
        json=_submission(tenant_id="untrusted-body-tenant"),
    )
    view = TaskView.model_validate(response.json())

    assert response.status_code == 202
    assert view.contract is not None
    assert view.contract.tenant_id == "header-tenant"
    denied = client.get(f"/api/v1/tasks/{view.task_id}", params={"project_id": "project-live-test"}, headers={"X-Tenant-Id": "untrusted-body-tenant"})
    assert denied.status_code == 404
