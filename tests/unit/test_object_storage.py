from __future__ import annotations

import pytest

from superai.service import ServiceConfig, SuperAIService
from superai.storage import IntegrityError


@pytest.fixture
def service(tmp_path):
    instance = SuperAIService(ServiceConfig(data_dir=tmp_path / "superai-data"))
    try:
        yield instance
    finally:
        instance.close()


def test_content_addressed_dedupe_and_idempotent_write_reuse_artifact(service: SuperAIService) -> None:
    first = service.store.put_bytes(
        b"same immutable payload",
        tenant_id="tenant-a",
        media_type="text/plain",
        schema_name="TestArtifact",
    )
    second = service.store.put_bytes(
        b"same immutable payload",
        tenant_id="tenant-a",
        media_type="text/plain",
        schema_name="TestArtifact",
    )

    assert first.artifact_id != second.artifact_id
    assert first.content_hash == second.content_hash
    assert service.store.get_bytes(first.artifact_id, "tenant-a") == b"same immutable payload"
    assert service.store.get_bytes(second.artifact_id, "tenant-a") == b"same immutable payload"
    assert service.database.one("SELECT COUNT(*) AS count FROM blobs WHERE content_hash = ?", (first.content_hash,))["count"] == 1

    keyed_first = service.store.put_bytes(
        b"idempotent payload",
        tenant_id="tenant-a",
        media_type="text/plain",
        schema_name="TestArtifact",
        idempotency_key="write-1",
    )
    keyed_repeat = service.store.put_bytes(
        b"a later payload must not replace the first write",
        tenant_id="tenant-a",
        media_type="text/plain",
        schema_name="TestArtifact",
        idempotency_key="write-1",
    )

    assert keyed_repeat.artifact_id == keyed_first.artifact_id
    assert service.store.get_bytes(keyed_repeat.artifact_id, "tenant-a") == b"idempotent payload"


def test_read_detects_tampered_content_hash(service: SuperAIService) -> None:
    artifact = service.store.put_bytes(
        b"untampered data",
        tenant_id="tenant-a",
        media_type="application/octet-stream",
        schema_name="BinaryInput",
    )
    service.store._blob_path(artifact.content_hash).write_bytes(b"tampered data")

    with pytest.raises(IntegrityError, match="checksum mismatch"):
        service.store.get_bytes(artifact.artifact_id, "tenant-a")


def test_gc_preserves_idempotency_roots_and_removes_revoked_source_bytes(service: SuperAIService) -> None:
    keyed = service.store.put_bytes(
        b"must survive because an idempotency mapping references it",
        tenant_id="tenant-a",
        media_type="text/plain",
        schema_name="TestArtifact",
        idempotency_key="survive-gc",
    )
    source = service.cosmos.import_text(
        title="Delete me",
        text="This source is intentionally revoked before garbage collection.",
        tenant_id="tenant-a",
    )
    revoked_path = service.store._blob_path(source.artifact.content_hash)
    assert revoked_path.exists()

    service.cosmos.delete_source(source.source_id, "tenant-a")
    result = service.store.garbage_collect([], grace=False)

    assert service.store.get_bytes(keyed.artifact_id, "tenant-a").startswith(b"must survive")
    replay = service.store.put_bytes(
        b"different bytes must not replace keyed artifact",
        tenant_id="tenant-a",
        media_type="text/plain",
        schema_name="TestArtifact",
        idempotency_key="survive-gc",
    )
    assert replay.artifact_id == keyed.artifact_id
    assert result["deleted_blobs"] >= 1
    assert not revoked_path.exists()
