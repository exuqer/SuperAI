"""JSON serialization utilities."""

from __future__ import annotations

import json
from typing import Any



def encode(value: Any) -> str:
    """Encode a value to JSON string with consistent formatting."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode(value: str | None, default: Any = None) -> Any:
    """Decode a JSON string, returning default if empty or None."""
    if not value:
        return {} if default is None else default
    return json.loads(value)


def decode_list(value: str | None, default: Any = None) -> list[Any]:
    """Decode a JSON string as list."""
    if not value:
        return [] if default is None else default
    result = json.loads(value)
    return result if isinstance(result, list) else []


def decode_dict(value: str | None, default: Any = None) -> dict[str, Any]:
    """Decode a JSON string as dict."""
    if not value:
        return {} if default is None else default
    result = json.loads(value)
    return result if isinstance(result, dict) else {}
