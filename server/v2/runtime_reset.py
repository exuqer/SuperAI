"""Process-local reset hooks used after destructive test-memory operations."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any


ResetCallback = Callable[[], Any]


class RuntimeResetRegistry:
    """Registry of process-local caches that must not survive a database reset."""

    _lock = RLock()
    _callbacks: dict[str, ResetCallback] = {}

    @classmethod
    def register(cls, owner: str, callback: ResetCallback) -> None:
        name = str(owner).strip()
        if not name:
            raise ValueError("runtime reset owner must not be empty")
        with cls._lock:
            cls._callbacks[name] = callback

    @classmethod
    def unregister(cls, owner: str) -> None:
        with cls._lock:
            cls._callbacks.pop(str(owner), None)

    @classmethod
    def reset_all(cls) -> dict[str, dict[str, Any]]:
        with cls._lock:
            callbacks = list(sorted(cls._callbacks.items()))
        report: dict[str, dict[str, Any]] = {}
        for owner, callback in callbacks:
            try:
                result = callback()
                report[owner] = {
                    "reset": True,
                    "detail": result if isinstance(result, dict) else {"result": result},
                }
            except Exception as error:  # reset report must expose every failed owner
                report[owner] = {
                    "reset": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        return report


_DEFAULTS_REGISTERED = False


def register_default_runtime_resetters() -> None:
    """Register reset hooks lazily to avoid import cycles during app startup."""

    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return

    import server.database as database
    from server.v2.acceleration import runtime as acceleration_runtime

    RuntimeResetRegistry.register(
        "sqlite_thread_local",
        lambda: (database.close_current_connection() or {"connection_closed": True}),
    )
    RuntimeResetRegistry.register(
        "acceleration_indexes",
        acceleration_runtime.reset_runtime_state,
    )
    _DEFAULTS_REGISTERED = True


__all__ = ["RuntimeResetRegistry", "register_default_runtime_resetters"]
