from __future__ import annotations

import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable


@dataclass
class Job:
    job_id: str
    name: str
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
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "traceback": self.traceback,
        }


class JobRegistry:
    def __init__(self, max_workers: int = 2, max_jobs: int = 100) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="semantic-ants-job")
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()
        self._max_jobs = max_jobs

    def submit(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:16], name=name)
        with self._lock:
            self._jobs[job.job_id] = job
            self._trim_locked()
        future = self._executor.submit(self._run, job.job_id, fn, *args, **kwargs)
        future.add_done_callback(self._consume_future)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _run(self, job_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = time.time()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - preserved for API diagnostics
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.finished_at = time.time()
                job.error = str(exc)
                job.traceback = traceback.format_exc()
            raise
        with self._lock:
            job = self._jobs[job_id]
            job.status = "completed"
            job.finished_at = time.time()
            job.result = result
        return result

    def _consume_future(self, future: Future[Any]) -> None:
        try:
            future.result()
        except Exception:
            return

    def _trim_locked(self) -> None:
        if len(self._jobs) <= self._max_jobs:
            return
        removable = sorted(self._jobs.values(), key=lambda job: job.created_at)
        for job in removable[: len(self._jobs) - self._max_jobs]:
            if job.status in {"completed", "failed"}:
                self._jobs.pop(job.job_id, None)
