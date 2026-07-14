"""Canonical hive schema installer."""

from server.v2.schema import ensure_schema

ensure_hive_schema = ensure_schema

__all__ = ["ensure_schema", "ensure_hive_schema"]
