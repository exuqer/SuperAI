import pytest

from superai.contracts import TaskSubmission
from superai.learning import LearningSafetyError
from superai.service import ServiceConfig, SuperAIService


def _completed(service: SuperAIService, message: str, conversation_id: str):
    task = service.submit_task(
        TaskSubmission(message=message, conversation_id=conversation_id, tenant_id="tenant-a"),
        execute_now=True,
    )
    assert task.answer is not None
    return task


def test_compost_revocation_cascades_to_integrated_cosmos_claims(tmp_path) -> None:
    service = SuperAIService(ServiceConfig(tmp_path / "superai-data"))
    service.runtime.stop_worker()
    try:
        source = service.cosmos.import_text(
            title="evidence",
            text="Collider2D requires a Rigidbody2D for physical collision handling.",
            tenant_id="tenant-a",
        )
        task = _completed(service, "Collider2D collision handling", "compost-conversation")
        compost = service.learning.decompose_task(task.task_id, "tenant-a")
        service.learning.validate_compost(compost.compost_id, "tenant-a")
        service.learning.integrate_compost(compost.compost_id, "tenant-a")
        assert len(service.cosmos.list_claims(tenant_id="tenant-a")) >= 2

        service.cosmos.delete_source(source.source_id, "tenant-a")

        assert service.cosmos.list_claims(tenant_id="tenant-a") == []
        assert service.learning.compost(compost.compost_id, "tenant-a").status == "deleted"
    finally:
        service.close()


def test_skill_requires_owned_disjoint_holdout_and_stays_tenant_scoped(tmp_path) -> None:
    service = SuperAIService(ServiceConfig(tmp_path / "superai-data"))
    service.runtime.stop_worker()
    try:
        service.cosmos.import_text(
            title="report-context",
            text="Сформируй краткий отчёт из разрешённого контекста.",
            tenant_id="tenant-a",
        )
        first = _completed(service, "Сформируй краткий отчёт", "skill-conversation-a")
        second = _completed(service, "Сформируй краткий отчёт", "skill-conversation-b")
        holdout = _completed(service, "Сформируй краткий отчёт", "skill-conversation-holdout")

        with pytest.raises(LearningSafetyError):
            service.learning.compile_candidate(
                tenant_id="tenant-a",
                train_task_ids=[first.task_id, second.task_id],
                holdout_task_ids=[first.task_id],
            )

        skill = service.learning.compile_candidate(
            tenant_id="tenant-a",
            train_task_ids=[first.task_id, second.task_id],
            holdout_task_ids=[holdout.task_id],
        )
        validated = service.learning.validate_skill(
            skill.skill_id,
            skill.version,
            tenant_id="tenant-a",
            quality_delta=0.1,
            latency_delta=0.0,
            resource_delta=0.0,
            risk_penalty=0.0,
        )
        shadow = service.learning.shadow_skill(validated.skill_id, validated.version, tenant_id="tenant-a")
        active = service.learning.activate_skill(shadow.skill_id, shadow.version, tenant_id="tenant-a")
        assert active.state.value == "active"
        assert service.learning.skills(tenant_id="tenant-b") == []
        with pytest.raises(LearningSafetyError):
            service.learning.skill(active.skill_id, active.version, tenant_id="tenant-b")
    finally:
        service.close()
