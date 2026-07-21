"""Request-local state for the dialogue pipeline.

The service itself is long lived, while every query must be causally isolated.
This small immutable envelope makes ownership explicit and gives traces a stable
execution identity without retaining mutable collections between turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from .graph_repository import stable_id, utcnow


@dataclass(frozen=True)
class QueryExecutionContext:
    execution_id: str
    conversation_id: str
    turn_id: str
    query_graph_id: str
    created_at: str
    hypothesis_contexts: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    diagnostics: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(
        default_factory=dict
    )

    @classmethod
    def create(
        cls,
        *,
        conversation_id: str,
        turn_id: str,
        query_graph_id: str,
    ) -> "QueryExecutionContext":
        return cls(
            execution_id=stable_id(
                "query-execution", conversation_id, turn_id, query_graph_id
            ),
            conversation_id=conversation_id,
            turn_id=turn_id,
            query_graph_id=query_graph_id,
            created_at=utcnow(),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "query_graph_id": self.query_graph_id,
            "created_at": self.created_at,
        }


def diagnostic_owner_valid(
    diagnostic: Mapping[str, Any],
    *,
    execution: QueryExecutionContext,
    hypothesis_id: str,
    gap_ids: Sequence[str],
    event_ids: Sequence[str],
) -> bool:
    """Validate a selected diagnostic before it enters a public trace."""
    return bool(
        diagnostic.get("execution_id") == execution.execution_id
        and diagnostic.get("query_graph_id") == execution.query_graph_id
        and diagnostic.get("hypothesis_id") == hypothesis_id
        and diagnostic.get("gap_id") in set(gap_ids)
        and diagnostic.get("event_id") in set(event_ids)
    )
