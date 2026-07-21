"""DialogueContextState — prevents UNRESOLVED contamination and manages context inheritance.

Only RESOLVED turns with valid BindingConfigurations become context sources.
UNRESOLVED turns are tracked but never inherited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .graph_repository import stable_id, utcnow


@dataclass
class DialogueContextState:
    """Tracks which turns are valid context sources for the next query."""

    conversation_id: str

    last_turn_id: Optional[str] = None
    last_resolved_turn_id: Optional[str] = None
    last_valid_binding_configuration_id: Optional[str] = None
    active_event_binding_frame_id: Optional[str] = None

    unresolved_turn_ids: List[str] = field(default_factory=list)
    context_strength: float = 0.0

    created_at: str = ""
    updated_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "last_turn_id": self.last_turn_id,
            "last_resolved_turn_id": self.last_resolved_turn_id,
            "last_valid_binding_configuration_id": self.last_valid_binding_configuration_id,
            "active_event_binding_frame_id": self.active_event_binding_frame_id,
            "unresolved_turn_ids": list(self.unresolved_turn_ids),
            "context_strength": self.context_strength,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def create(cls, conversation_id: str) -> DialogueContextState:
        now = utcnow()
        return cls(
            conversation_id=conversation_id,
            created_at=now,
            updated_at=now,
        )

    def mark_resolved(
        self,
        turn_id: str,
        binding_configuration_id: str,
        frame_id: Optional[str] = None,
    ) -> None:
        """Mark a turn as resolved and update context state."""
        self.last_turn_id = turn_id
        self.last_resolved_turn_id = turn_id
        self.last_valid_binding_configuration_id = binding_configuration_id
        if frame_id:
            self.active_event_binding_frame_id = frame_id
        self.context_strength = min(1.0, self.context_strength + 0.15)
        self.updated_at = utcnow()

    def mark_unresolved(self, turn_id: str) -> None:
        """Mark a turn as unresolved — it will NOT become a context source."""
        self.last_turn_id = turn_id
        if turn_id not in self.unresolved_turn_ids:
            self.unresolved_turn_ids.append(turn_id)
        # Keep only last 10 unresolved turns
        self.unresolved_turn_ids = self.unresolved_turn_ids[-10:]
        self.context_strength = max(0.0, self.context_strength - 0.05)
        self.updated_at = utcnow()

    def is_unresolved(self, turn_id: str) -> bool:
        """Check if a turn was unresolved."""
        return turn_id in self.unresolved_turn_ids

    def can_inherit_from(self, turn_id: str) -> bool:
        """Check if a turn can be used as context source."""
        if self.is_unresolved(turn_id):
            return False
        if turn_id == self.last_resolved_turn_id:
            return True
        return False

    def is_currently_resolved_last_turn(self) -> bool:
        """Check if the last turn was resolved (i.e., context is clean)."""
        if not self.last_turn_id:
            return False
        return not self.is_unresolved(self.last_turn_id) and self.last_turn_id == self.last_resolved_turn_id

    def get_context_source_turn_id(self) -> Optional[str]:
        """Get the turn ID that should be used as context source."""
        return self.last_resolved_turn_id

    def get_active_frame_id(self) -> Optional[str]:
        """Get the active EventBindingFrame ID."""
        return self.active_event_binding_frame_id

    def clear_frame(self) -> None:
        """Clear the active frame (e.g., on NEW_QUERY)."""
        self.active_event_binding_frame_id = None
        self.context_strength = 0.0
        self.updated_at = utcnow()


class DialogueContextManager:
    """Manages dialogue context state persistence and retrieval."""

    @staticmethod
    def load(conn: Any, conversation_id: str) -> DialogueContextState:
        """Load context state from database."""
        row = conn.execute(
            """SELECT * FROM dialogue_context_states
               WHERE conversation_id=?""",
            (conversation_id,),
        ).fetchone()
        if not row:
            return DialogueContextState.create(conversation_id)
        return DialogueContextState(
            conversation_id=str(row["conversation_id"]),
            last_turn_id=row["last_turn_id"],
            last_resolved_turn_id=row["last_resolved_turn_id"],
            last_valid_binding_configuration_id=row["last_valid_binding_configuration_id"],
            active_event_binding_frame_id=row["active_event_binding_frame_id"],
            unresolved_turn_ids=list(
                __import__("json").loads(row["unresolved_turn_ids_json"] or "[]")
            ),
            context_strength=float(row["context_strength"] or 0.0),
            created_at=str(row["created_at"] or ""),
            updated_at=str(row["updated_at"] or ""),
        )

    @staticmethod
    def save(conn: Any, state: DialogueContextState) -> None:
        """Persist context state to database."""
        import json
        now = utcnow()
        conn.execute(
            """INSERT OR REPLACE INTO dialogue_context_states
               (conversation_id, last_turn_id, last_resolved_turn_id,
                last_valid_binding_configuration_id,
                active_event_binding_frame_id,
                unresolved_turn_ids_json, context_strength,
                state_json, created_at, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                state.conversation_id,
                state.last_turn_id,
                state.last_resolved_turn_id,
                state.last_valid_binding_configuration_id,
                state.active_event_binding_frame_id,
                json.dumps(state.unresolved_turn_ids, ensure_ascii=False),
                state.context_strength,
                "{}",
                state.created_at or now,
                now,
            ),
        )

    @staticmethod
    def should_inherit_context(
        state: DialogueContextState,
        previous_answer_status: Optional[str] = None,
    ) -> bool:
        """Determine if context should be inherited for the next query."""
        if previous_answer_status is None:
            return state.context_strength > 0.0
        # Only inherit from RESOLVED or PARTIALLY_RESOLVED turns
        if previous_answer_status in {"RESOLVED", "PARTIALLY_RESOLVED"}:
            return True
        # UNRESOLVED, AMBIGUOUS, BUILD_FAILED — do not inherit
        return False