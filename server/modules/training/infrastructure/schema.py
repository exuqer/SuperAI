"""Canonical training schema installer."""

from server.v2.schema import ensure_schema

ensure_training_schema = ensure_schema

__all__ = ["ensure_schema", "ensure_training_schema"]
