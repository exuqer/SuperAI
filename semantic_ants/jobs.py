from __future__ import annotations

from dataclasses import dataclass, field
import traceback
from threading import RLock, Thread
import time
from typing import Any, Callable
from uuid import uuid4


@dataclass
class Job:
    job_id: str
    kind: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "traceback": self.traceback,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = RLock()

    def submit(self, kind: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Job:
        job = Job(job_id=uuid4().hex[:16], kind=kind)
        with self._lock:
            self._jobs[job.job_id] = job
        thread = Thread(target=self._run, args=(job.job_id, fn, args, kwargs), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)

    def shutdown(self) -> None:
        return None

    def _run(self, job_id: str, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = time.time()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - exercised through API error paths
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = str(exc)
                job.traceback = traceback.format_exc()
                job.finished_at = time.time()
            return
        with self._lock:
            job = self._jobs[job_id]
            job.status = "completed"
            job.result = result
            job.finished_at = time.time()
