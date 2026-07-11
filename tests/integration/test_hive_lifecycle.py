from __future__ import annotations

import pytest

from superai.contracts import HiveState, TaskContract
from superai.service import ServiceConfig, SuperAIService
from superai.storage import AccessDenied


@pytest.fixture
def service(tmp_path):
    instance = SuperAIService(ServiceConfig(data_dir=tmp_path / "superai-data"))
    try:
        yield instance
    finally:
        instance.close()


def _contract(goal: str) -> TaskContract:
    return TaskContract(
        tenant_id="tenant-a",
        project_id="game-a",
        conversation_id="conversation-collider-bread",
        goal=goal,
    )


def test_collider_bread_collider_uses_separate_hives_and_restores_prior_decision(
    service: SuperAIService,
) -> None:
    collider_contract = _contract("Collider2D персонаж проходит сквозь стену")
    collider, decision, _ = service.hives.select_or_create(collider_contract, "trace-collider")

    assert decision == "create"
    service.hives.add_entry(
        collider.hive_id,
        "tenant-a",
        store_name="EvidenceStore",
        content_type="confirmed_decision",
        content={"decision": "Disable Is Trigger", "symptom": "passes through wall"},
        relevance=1.0,
        protected=True,
        trace_id="trace-collider",
    )
    frozen_collider = service.hives.freeze(collider.hive_id, "tenant-a", "trace-collider")

    assert frozen_collider.state == HiveState.FROZEN
    assert frozen_collider.snapshot_id is not None

    # Restore is a materialization, not merely a lifecycle-flag change.
    service.database.execute("DELETE FROM hive_entries WHERE hive_id = ?", (collider.hive_id,))
    service.database.execute(
        "UPDATE hives SET state_json = ? WHERE hive_id = ?",
        ('{"goals": []}', collider.hive_id),
    )
    restored_after_fault = service.hives.restore(collider.hive_id, "tenant-a", "trace-collider-repair")
    assert restored_after_fault.state == HiveState.ACTIVE
    assert any(entry.content.get("decision") == "Disable Is Trigger" for entry in restored_after_fault.entries)
    assert restored_after_fault.state_data["goals"]

    service.hives.freeze(collider.hive_id, "tenant-a", "trace-collider-refreeze")

    bread_contract = _contract("Как испечь хлеб без дрожжей")
    bread, decision, _ = service.hives.select_or_create(bread_contract, "trace-bread")

    assert decision == "create"
    assert bread.hive_id != collider.hive_id
    assert bread.state == HiveState.ACTIVE
    assert all("Trigger" not in str(entry.content) for entry in bread.entries)

    service.hives.freeze(bread.hive_id, "tenant-a", "trace-bread")
    return_contract = _contract("Вернемся к Collider2D и проверке стены")
    restored, decision, alternatives = service.hives.select_or_create(return_contract, "trace-return")

    assert decision == "restore"
    assert restored.hive_id == collider.hive_id
    assert restored.state == HiveState.ACTIVE
    assert any(entry.content.get("decision") == "Disable Is Trigger" for entry in restored.entries)
    assert any(candidate["hive_id"] == collider.hive_id for candidate in alternatives)

    with pytest.raises(AccessDenied):
        service.hive(collider.hive_id, "tenant-a", project_id="another-project")
    with pytest.raises(AccessDenied):
        service.hive(collider.hive_id, "tenant-a")

    events = service.traces.trace("trace-return")["events"]
    assert any(event["kind"] == "HiveRestored" for event in events)
