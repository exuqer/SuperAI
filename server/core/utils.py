"""Common utility functions."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone


def clamp(value: float) -> float:
    """Clamp value to [0, 1] range."""
    return max(0.0, min(1.0, float(value)))


def utcnow() -> str:
    """Get current UTC time as ISO format string."""
    return datetime.now(timezone.utc).isoformat()


def radius_for(mass: float, density: float = 1.0) -> float:
    """Calculate radius for a cloud based on mass and density."""
    min_radius = 6.0
    max_radius = 250.0
    base_radius = 14.0
    radius_factor = 10.0
    return min(max_radius, max(min_radius, base_radius + radius_factor * math.sqrt(max(0.01, mass * density))))


def stable_position(namespace: str, index: int = 0) -> tuple[float, float]:
    """Generate stable pseudo-random position from namespace and index."""
    value = int(hashlib.sha256(f"{namespace}:{index}".encode()).hexdigest()[:12], 16)
    return 100.0 + float(value % 1400), 100.0 + float((value // 1400) % 800)


def get_settings():
    """Lazy import of settings to avoid circular imports."""
    from server.core.settings import settings
    return settings
