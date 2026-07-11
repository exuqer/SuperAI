"""Content-addressed object storage, archive and aggregate snapshots."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .contracts import AccessScope, ArtifactRef, ModuleSnapshot, new_id, utcnow
from .database import SqliteDatabase, json_dumps, json_loads


class ArtifactNotFound(KeyError):
    pass


class AccessDenied(PermissionError):
    pass


class IntegrityError(RuntimeError):
    pass


class ObjectStore:
    """Metadata in SQLite, immutable bytes addressed by SHA-256 on disk."""

    def __init__(self, root: Path, database: SqliteDatabase) -> None:
        self.root = root
        self.database = database
        self.blob_root = root / "blobs"
        self.tmp_root = root / "tmp"
        self._blob_lock = threading.RLock()
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def reconcile_orphans(self) -> int:
        """Remove published files with no committed blob metadata after a crash."""
        committed = {row["content_hash"] for row in self.database.all("SELECT content_hash FROM blobs")}
        removed = 0
        with self._blob_lock:
            for path in self.blob_root.glob("*/*/*"):
                if path.is_file() and path.name not in committed:
                    path.unlink()
                    removed += 1
        return removed

    def put_bytes(
        self,
        data: bytes,
        *,
        tenant_id: str,
        media_type: str,
        schema_name: str,
        access_scope: Optional[AccessScope] = None,
        artifact_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> ArtifactRef:
        scope = access_scope or AccessScope(tenant_id=tenant_id)
        if scope.tenant_id != tenant_id and scope.visibility != "global":
            raise AccessDenied("artifact scope tenant must match artifact owner")
        if idempotency_key:
            previous = self.database.one(
                "SELECT artifact_id FROM artifact_writes WHERE tenant_id = ? AND idempotency_key = ?",
                (tenant_id, idempotency_key),
            )
            if previous:
                return self.get_metadata(previous["artifact_id"], tenant_id, project_id=scope.project_id)

        digest = self._publish_blob(data)

        now = utcnow()
        ref = ArtifactRef(
            artifact_id=artifact_id or new_id("art"),
            content_hash=digest,
            media_type=media_type,
            schema_name=schema_name,
            size=len(data),
            tenant_id=tenant_id,
            created_at=now,
            access_scope=scope,
        )
        with self.database.transaction() as connection:
            self._insert_artifact(connection, ref)
            if idempotency_key:
                connection.execute(
                    "INSERT INTO artifact_writes(tenant_id, idempotency_key, artifact_id, created_at) VALUES (?, ?, ?, ?)",
                    (tenant_id, idempotency_key, ref.artifact_id, now.isoformat()),
                )
        return ref

    def put_json(
        self,
        value: Any,
        *,
        tenant_id: str,
        schema_name: str,
        access_scope: Optional[AccessScope] = None,
        artifact_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> ArtifactRef:
        return self.put_bytes(
            json_dumps(value).encode("utf-8"),
            tenant_id=tenant_id,
            media_type="application/json",
            schema_name=schema_name,
            access_scope=access_scope,
            artifact_id=artifact_id,
            idempotency_key=idempotency_key,
        )

    def get_metadata(
        self,
        artifact_id: str,
        tenant_id: str,
        *,
        project_id: Optional[str] = None,
        include_deleted: bool = False,
    ) -> ArtifactRef:
        clause = "artifact_id = ?" if include_deleted else "artifact_id = ? AND deleted_at IS NULL"
        row = self.database.one("SELECT * FROM artifacts WHERE " + clause, (artifact_id,))
        if row is None:
            raise ArtifactNotFound(artifact_id)
        scope = AccessScope.model_validate(json_loads(row["access_json"]))
        self._check_access(scope, tenant_id, project_id)
        return ArtifactRef(
            artifact_id=row["artifact_id"],
            content_hash=row["content_hash"],
            media_type=row["media_type"],
            schema_name=row["schema_name"],
            schema_version=row["schema_version"],
            size=row["size"],
            tenant_id=row["tenant_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            access_scope=scope,
        )

    def get_bytes(
        self,
        artifact_id: str,
        tenant_id: str,
        *,
        project_id: Optional[str] = None,
        verify: bool = True,
    ) -> bytes:
        ref = self.get_metadata(artifact_id, tenant_id, project_id=project_id)
        path = self._blob_path(ref.content_hash)
        if not path.exists():
            raise IntegrityError("blob missing for artifact %s" % artifact_id)
        data = path.read_bytes()
        if verify and hashlib.sha256(data).hexdigest() != ref.content_hash:
            raise IntegrityError("checksum mismatch for artifact %s" % artifact_id)
        return data

    def get_json(self, artifact_id: str, tenant_id: str, *, project_id: Optional[str] = None) -> Any:
        return json_loads(self.get_bytes(artifact_id, tenant_id, project_id=project_id).decode("utf-8"))

    def create_snapshot(
        self,
        *,
        aggregate_type: str,
        aggregate_id: str,
        sequence: int,
        state: Any,
        tenant_id: str,
        access_scope: Optional[AccessScope] = None,
        after_insert: Optional[Callable[[Any, ModuleSnapshot], None]] = None,
    ) -> ModuleSnapshot:
        data = json_dumps(state).encode("utf-8")
        scope = access_scope or AccessScope(tenant_id=tenant_id)
        digest = self._publish_blob(data)
        now = utcnow()
        ref = ArtifactRef(
            artifact_id=new_id("art"),
            content_hash=digest,
            media_type="application/json",
            schema_name="ModuleSnapshotState",
            size=len(data),
            tenant_id=tenant_id,
            created_at=now,
            access_scope=scope,
        )
        snapshot = ModuleSnapshot(
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            sequence=sequence,
            artifact_ref=ref,
            state_hash=ref.content_hash,
        )
        # Artifact metadata and snapshot pointer share one SQLite commit. A
        # crash before it leaves at most an unreferenced content-addressed file,
        # which reconciliation/GC can safely remove; it never publishes a
        # half-snapshot to readers.
        with self.database.transaction() as connection:
            self._insert_artifact(connection, ref)
            connection.execute(
                "INSERT INTO snapshots(snapshot_id, aggregate_type, aggregate_id, tenant_id, sequence, artifact_id, state_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot.snapshot_id,
                    aggregate_type,
                    aggregate_id,
                    tenant_id,
                    sequence,
                    ref.artifact_id,
                    snapshot.state_hash,
                    snapshot.created_at.isoformat(),
                ),
            )
            if after_insert:
                after_insert(connection, snapshot)
        return snapshot

    def latest_snapshot(
        self,
        aggregate_type: str,
        aggregate_id: str,
        tenant_id: str,
        *,
        project_id: Optional[str] = None,
    ) -> Optional[ModuleSnapshot]:
        row = self.database.one(
            "SELECT * FROM snapshots WHERE aggregate_type = ? AND aggregate_id = ? AND tenant_id = ? ORDER BY sequence DESC LIMIT 1",
            (aggregate_type, aggregate_id, tenant_id),
        )
        if row is None:
            return None
        ref = self.get_metadata(row["artifact_id"], tenant_id, project_id=project_id)
        return ModuleSnapshot(
            snapshot_id=row["snapshot_id"],
            aggregate_type=row["aggregate_type"],
            aggregate_id=row["aggregate_id"],
            sequence=row["sequence"],
            artifact_ref=ref,
            state_hash=row["state_hash"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def restore_snapshot(self, snapshot: ModuleSnapshot, tenant_id: str, *, project_id: Optional[str] = None) -> Any:
        data = self.get_bytes(snapshot.artifact_ref.artifact_id, tenant_id, project_id=project_id)
        if hashlib.sha256(data).hexdigest() != snapshot.state_hash:
            raise IntegrityError("snapshot state hash mismatch")
        return json_loads(data.decode("utf-8"))

    def garbage_collect(self, root_artifact_ids: Iterable[str], *, grace: bool = True) -> dict[str, int]:
        """Mark/sweep unused logical artifacts without trusting ref-counts.

        Callers pass roots from live aggregates; archive/snapshot/source roots are
        always included. The grace mode only marks records, making deletion
        reversible until a later sweep.
        """
        marked = 0
        deleted_blobs = 0
        now = utcnow().isoformat()
        pending_unlink: list[Path] = []
        with self.database.transaction() as connection:
            # Resolve roots in the same write transaction as marking. These
            # tables are durable references, not advisory cache pointers.
            roots = set(root_artifact_ids)
            roots.update(row["artifact_id"] for row in connection.execute("SELECT artifact_id FROM snapshots"))
            roots.update(
                row["artifact_id"]
                for row in connection.execute("SELECT artifact_id FROM sources WHERE deleted_at IS NULL")
            )
            roots.update(row["artifact_id"] for row in connection.execute("SELECT artifact_id FROM artifact_writes"))
            roots.update(
                row["artifact_id"] for row in connection.execute("SELECT artifact_id FROM composts WHERE status != 'deleted'")
            )
            for row in connection.execute("SELECT metadata_json FROM hive_entries"):
                reference = json_loads(row["metadata_json"], {}).get("source_ref")
                if reference:
                    roots.add(reference)
            for row in connection.execute("SELECT answer_json FROM tasks WHERE answer_json IS NOT NULL"):
                answer = json_loads(row["answer_json"], {})
                roots.update(source.get("artifact_id") for source in answer.get("sources", []) if source.get("artifact_id"))
            # During the grace period an aggregate may gain a reference again;
            # make the logical tombstone reversible while bytes still exist.
            if grace and roots:
                placeholders = ",".join("?" for _ in roots)
                connection.execute(
                    "UPDATE artifacts SET deleted_at = NULL WHERE artifact_id IN (%s)" % placeholders,
                    tuple(roots),
                )
            rows = connection.execute("SELECT artifact_id, content_hash FROM artifacts WHERE deleted_at IS NULL").fetchall()
            for row in rows:
                if row["artifact_id"] not in roots:
                    marked += 1
                    connection.execute("UPDATE artifacts SET deleted_at = ? WHERE artifact_id = ?", (now, row["artifact_id"]))
            if not grace:
                hashes = connection.execute("SELECT content_hash FROM blobs").fetchall()
                for blob in hashes:
                    active = connection.execute(
                        "SELECT 1 FROM artifacts WHERE content_hash = ? AND deleted_at IS NULL LIMIT 1",
                        (blob["content_hash"],),
                    ).fetchone()
                    if active is None:
                        connection.execute("DELETE FROM blobs WHERE content_hash = ?", (blob["content_hash"],))
                        pending_unlink.append(self._blob_path(blob["content_hash"]))
                        deleted_blobs += 1
        # Never unlink a blob while the metadata transaction may still roll
        # back. The in-process blob lock prevents a concurrent writer from
        # publishing a reference between this final check and unlink.
        if not grace:
            with self._blob_lock:
                for path in pending_unlink:
                    digest = path.name
                    if self.database.one("SELECT 1 FROM blobs WHERE content_hash = ?", (digest,)) is None and path.exists():
                        path.unlink()
        return {"marked_artifacts": marked, "deleted_blobs": deleted_blobs}

    def _blob_path(self, content_hash: str) -> Path:
        return self.blob_root / content_hash[:2] / content_hash[2:4] / content_hash

    def _publish_blob(self, data: bytes) -> str:
        """Durably publish bytes before their metadata becomes reachable."""
        digest = hashlib.sha256(data).hexdigest()
        target = self._blob_path(digest)
        with self._blob_lock:
            if target.exists():
                return digest
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix="artifact-", dir=str(self.tmp_root))
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                # Another writer can win the race; equal content makes that safe.
                if not target.exists():
                    os.replace(temp_name, target)
                    try:
                        directory_fd = os.open(str(target.parent), os.O_RDONLY)
                        try:
                            os.fsync(directory_fd)
                        finally:
                            os.close(directory_fd)
                    except OSError:
                        # Some filesystems do not allow directory fsync; the
                        # file itself was still fsynced before publication.
                        pass
                else:
                    os.unlink(temp_name)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
        return digest

    @staticmethod
    def _insert_artifact(connection: Any, ref: ArtifactRef) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO blobs(content_hash, size, created_at) VALUES (?, ?, ?)",
            (ref.content_hash, ref.size, ref.created_at.isoformat()),
        )
        connection.execute(
            "INSERT INTO artifacts(artifact_id, content_hash, media_type, schema_name, schema_version, size, tenant_id, access_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ref.artifact_id,
                ref.content_hash,
                ref.media_type,
                ref.schema_name,
                ref.schema_version,
                ref.size,
                ref.tenant_id,
                json_dumps(ref.access_scope),
                ref.created_at.isoformat(),
            ),
        )

    @staticmethod
    def _check_access(scope: AccessScope, requester_tenant_id: str, requester_project_id: Optional[str]) -> None:
        if scope.visibility != "global" and scope.tenant_id != requester_tenant_id:
            raise AccessDenied("artifact belongs to a different tenant")
        if scope.visibility == "project" and scope.project_id != requester_project_id:
            raise AccessDenied("artifact belongs to a different project")
