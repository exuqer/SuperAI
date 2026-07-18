"""Canonical schema entry point for SuperAI V2.7."""

from .graph_schema import SCHEMA_VERSION, ensure_graph_schema


def ensure_schema(conn):
    """Create the fresh event-graph schema without old projections."""
    return ensure_graph_schema(conn)


__all__ = ["SCHEMA_VERSION", "ensure_graph_schema", "ensure_schema"]
