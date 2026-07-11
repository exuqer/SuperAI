from __future__ import annotations

import pytest

from superai.contracts import AccessScope, TaskContract
from superai.service import ServiceConfig, SuperAIService
from superai.storage import AccessDenied, ArtifactNotFound


@pytest.fixture
def service(tmp_path):
    instance = SuperAIService(ServiceConfig(data_dir=tmp_path / "superai-data"))
    try:
        yield instance
    finally:
        instance.close()


def _contract(*, tenant_id: str, project_id: str | None, goal: str) -> TaskContract:
    return TaskContract(
        tenant_id=tenant_id,
        project_id=project_id,
        conversation_id="cosmos-provenance",
        goal=goal,
    )


def test_import_retrieval_preserves_exact_provenance_and_project_boundary(service: SuperAIService) -> None:
    source_text = "Collider2D collision requires Is Trigger to be disabled for walls."
    imported = service.cosmos.import_text(
        title="Collider notes",
        text=source_text,
        tenant_id="tenant-a",
        access_scope=AccessScope(
            tenant_id="tenant-a",
            project_id="game-a",
            visibility="project",
        ),
        sectors=["Programming", "Unity"],
    )

    allowed = service.cosmos.retrieve(
        _contract(
            tenant_id="tenant-a",
            project_id="game-a",
            goal="Collider2D collision diagnostics",
        )
    )

    assert imported.status == "integrated"
    assert imported.imported_claims == 1
    assert len(allowed.claims) == 1
    retrieved = allowed.claims[0]
    assert retrieved.claim.source_id == imported.source_id
    assert retrieved.claim.source_artifact_id == imported.artifact.artifact_id
    assert retrieved.claim.source_fragment == source_text
    assert retrieved.source.artifact_id == imported.artifact.artifact_id
    assert retrieved.source.content_hash == imported.artifact.content_hash
    assert service.store.get_bytes(retrieved.source.artifact_id, "tenant-a", project_id="game-a").decode("utf-8") == source_text
    assert retrieved.claim.access_scope.visibility == "project"
    assert retrieved.claim.access_scope.project_id == "game-a"
    assert retrieved.claim.sectors == ["Programming", "Unity"]

    other_project = service.cosmos.retrieve(
        _contract(
            tenant_id="tenant-a",
            project_id="game-b",
            goal="Collider2D collision diagnostics",
        )
    )
    other_tenant = service.cosmos.retrieve(
        _contract(
            tenant_id="tenant-b",
            project_id="game-a",
            goal="Collider2D collision diagnostics",
        )
    )

    assert other_project.claims == []
    assert other_tenant.claims == []
    assert other_project.gaps
    assert other_tenant.gaps
    with pytest.raises(AccessDenied):
        service.store.get_bytes(retrieved.source.artifact_id, "tenant-a", project_id="game-b")


def test_import_is_idempotent_quarantine_is_not_retrieved_and_deleted_source_hides_claims(
    service: SuperAIService,
) -> None:
    text = "Rigidbody2D must be paired with Collider2D for physical collisions."
    first = service.cosmos.import_text(title="Physics", text=text, tenant_id="tenant-a")
    duplicate = service.cosmos.import_text(title="Physics copy", text=text, tenant_id="tenant-a")

    assert duplicate.duplicate is True
    assert duplicate.source_id == first.source_id
    assert duplicate.artifact.artifact_id == first.artifact.artifact_id

    quarantined = service.cosmos.import_text(
        title="Unverified note",
        text="Collider2D undocumented toggle changes wall collision behavior.",
        tenant_id="tenant-a",
        trusted=False,
    )
    assert quarantined.status == "quarantined"
    assert quarantined.imported_claims == 0

    contract = _contract(
        tenant_id="tenant-a",
        project_id=None,
        goal="Rigidbody2D Collider2D physical collisions",
    )
    assert service.cosmos.retrieve(contract).claims

    removed = service.cosmos.delete_source(first.source_id, "tenant-a")
    after_deletion = service.cosmos.retrieve(contract)

    assert removed == first.imported_claims
    assert after_deletion.claims == []
    with pytest.raises(ArtifactNotFound):
        service.store.get_metadata(first.artifact.artifact_id, "tenant-a")


def test_identical_content_under_two_project_scopes_is_not_scope_deduplicated(service: SuperAIService) -> None:
    text = "Таблица очистки требует удалить пустые значения до расчёта среднего."
    first = service.cosmos.import_text(
        title="Rules A",
        text=text,
        tenant_id="tenant-a",
        access_scope=AccessScope(tenant_id="tenant-a", project_id="project-a", visibility="project"),
    )
    second = service.cosmos.import_text(
        title="Rules B",
        text=text,
        tenant_id="tenant-a",
        access_scope=AccessScope(tenant_id="tenant-a", project_id="project-b", visibility="project"),
    )

    assert not first.duplicate
    assert not second.duplicate
    assert first.source_id != second.source_id
    assert first.artifact.artifact_id != second.artifact.artifact_id
    assert len(service.cosmos.retrieve(_contract(tenant_id="tenant-a", project_id="project-a", goal="очистка таблицы")).claims) == 1
    assert len(service.cosmos.retrieve(_contract(tenant_id="tenant-a", project_id="project-b", goal="очистка таблицы")).claims) == 1
