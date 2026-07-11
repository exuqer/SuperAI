"""Contract-boundary tests for the versioned public models.

These tests intentionally exercise Pydantic at the same boundary used by the
HTTP adapter.  They keep the MVP's compatibility rule explicit: additions in
schema major ``1`` are accepted, while another major must fail before it can
enter the runtime.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from superai.contracts import CommandEnvelope, DomainEvent, TaskContract, TaskSubmission


def test_task_submission_accepts_v1_additions_and_ignores_unknown_boundary_fields() -> None:
    submission = TaskSubmission.model_validate(
        {
            "schema_version": "1.7",
            "message": "Собери детерминированный ответ.",
            "tenant_id": "tenant-contract-test",
            "conversation_id": "conv-contract-test",
            "budget": {"schema_version": "1.3", "time_ms": 1_500, "event_limit": 12},
            "future_transport_field": {"safe_to_ignore": True},
        }
    )

    assert submission.schema_version == "1.7"
    assert submission.budget.schema_version == "1.3"
    assert submission.budget.time_ms == 1_500
    assert submission.budget.event_limit == 12
    assert "future_transport_field" not in submission.model_dump()


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"schema_version": "2.0", "message": "incompatible"}, "schema_version"),
        ({"message": "bad budget", "budget": {"time_ms": 0}}, "budget.time_ms"),
        ({"message": ""}, "message"),
    ],
)
def test_task_submission_rejects_incompatible_or_invalid_schema_payloads(payload: dict[str, object], field: str) -> None:
    with pytest.raises(ValidationError) as raised:
        TaskSubmission.model_validate(payload)

    locations = {".".join(str(part) for part in error["loc"]) for error in raised.value.errors()}
    assert field in locations


def test_task_contract_and_event_envelopes_preserve_required_correlation_metadata() -> None:
    contract = TaskContract.model_validate(
        {
            "schema_version": "1.0",
            "task_id": "task-contract-test",
            "tenant_id": "tenant-contract-test",
            "conversation_id": "conv-contract-test",
            "goal": "Проверить обязательные поля контракта.",
            "budget": {"time_ms": 2_000, "step_limit": 10, "memory_bytes": 4_096, "event_limit": 25},
        }
    )
    command = CommandEnvelope(
        id="cmd-envelope-test",
        task_id=contract.task_id,
        trace_id="trace-contract-test",
        tenant_id=contract.tenant_id,
        kind="ExecuteTask",
        producer="contract-test",
        idempotency_key="contract-test-key",
        causation_id="request-contract-test",
        correlation_id="conversation-contract-test",
        payload={"contract_id": contract.task_id},
    )
    event = DomainEvent(
        id="evt-envelope-test",
        task_id=contract.task_id,
        trace_id=command.trace_id,
        tenant_id=contract.tenant_id,
        kind="TaskAccepted",
        producer="contract-test",
        causation_id=command.id,
        correlation_id=command.correlation_id,
    )

    assert contract.goal
    assert contract.budget.time_ms == 2_000
    assert command.schema_version == "1.0"
    assert command.causation_id == "request-contract-test"
    assert command.correlation_id == "conversation-contract-test"
    assert event.event_id.startswith("evt_")
    assert event.task_id == command.task_id
    assert event.trace_id == command.trace_id
    assert event.causation_id == command.id
