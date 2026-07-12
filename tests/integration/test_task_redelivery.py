from superai.contracts import TaskState, TaskSubmission
from superai.service import ServiceConfig, SuperAIService


def test_ack_loss_redelivery_reuses_committed_task_materialization(tmp_path) -> None:
    service = SuperAIService(ServiceConfig(tmp_path / "superai-data"))
    service.runtime.stop_worker()
    try:
        service.cosmos.import_text(
            title="runtime-context",
            text="Объясни ограниченный runtime.",
            tenant_id="local",
        )
        task = service.submit_task(
            TaskSubmission(message="Объясни ограниченный runtime", conversation_id="redelivery-conversation"),
            idempotency_key="redelivery-key",
            execute_now=True,
        )
        assert task.status is TaskState.SUCCEEDED
        before_entries = service.database.one(
            "SELECT COUNT(*) AS count FROM hive_entries WHERE hive_id = ?", (task.hive_id,)
        )["count"]
        before_plans = service.database.one(
            "SELECT COUNT(*) AS count FROM plans WHERE task_id = ?", (task.task_id,)
        )["count"]
        command = service.database.one("SELECT command_id FROM work_items WHERE task_id = ?", (task.task_id,))

        # Model a crash after handler-side materialization but before runtime
        # acknowledgement. A restart requeues only the unfinished work item.
        service.database.execute(
            "UPDATE work_items SET status = ? WHERE command_id = ?", (TaskState.RUNNING.value, command["command_id"])
        )
        assert service.runtime.recover_unfinished() == 1
        service.runtime.run_once()

        after_entries = service.database.one(
            "SELECT COUNT(*) AS count FROM hive_entries WHERE hive_id = ?", (task.hive_id,)
        )["count"]
        after_plans = service.database.one(
            "SELECT COUNT(*) AS count FROM plans WHERE task_id = ?", (task.task_id,)
        )["count"]
        assert after_entries == before_entries
        assert after_plans == before_plans
        assert service.task(task.task_id, "local").status is TaskState.SUCCEEDED
    finally:
        service.close()
