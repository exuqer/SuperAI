from __future__ import annotations

from superai.contracts import TaskState, TaskSubmission
from superai.database import json_loads
from superai.service import ServiceConfig, SuperAIService


def test_source_backed_task_persists_omega_snapshot_and_hypotheses(tmp_path) -> None:
    service = SuperAIService(ServiceConfig(tmp_path / "superai-data"))
    service.runtime.stop_worker()
    try:
        service.cosmos.import_text(
            title="greeting-training",
            text="Привет! Как у тебя дела?",
            tenant_id="tenant-omega",
        )
        task = service.submit_task(
            TaskSubmission(
                message="Привет!",
                tenant_id="tenant-omega",
                conversation_id="omega-conversation",
            ),
            execute_now=True,
        )

        assert task.status is TaskState.SUCCEEDED
        assert task.answer is not None
        assert task.answer.answer == "Привет!"

        snapshot = service.database.one(
            "SELECT * FROM active_graph_snapshots WHERE task_id = ?",
            (task.task_id,),
        )
        assert snapshot is not None
        assert json_loads(snapshot["node_types_json"]) == ["claim", "concept"]
        assert json_loads(snapshot["edge_types_json"]) == ["semantic"]

        hypotheses = service.database.all(
            "SELECT * FROM hypotheses WHERE task_id = ? ORDER BY created_at",
            (task.task_id,),
        )
        assert hypotheses
        assert all(json_loads(hypothesis["predictions_json"]) for hypothesis in hypotheses)
        assert any(hypothesis["status"] == "selected" for hypothesis in hypotheses)

        trace = service.trace(task.trace_id, "tenant-omega")
        assert "HypothesisBoardCompleted" in [event["kind"] for event in trace["events"]]
    finally:
        service.close()
