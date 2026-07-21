"""EventBindingFrame — stable dialogue projection of a selected event.

The frame survives multiple short turns (Кому? Кто? Что?) and tracks
how participants relate to questions and bindings across the dialogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .graph_repository import stable_id, utcnow


class FrameStatus(str, Enum):
    ACTIVE = "ACTIVE"
    WEAK = "WEAK"
    CLOSED = "CLOSED"


class ParticipantOrigin(str, Enum):
    EXPLICIT_ROOT_QUERY = "EXPLICIT_ROOT_QUERY"
    RESOLVED_ROOT_GAP = "RESOLVED_ROOT_GAP"
    RESOLVED_LATER_GAP = "RESOLVED_LATER_GAP"
    INFERRED_EVENT_PARTICIPANT = "INFERRED_EVENT_PARTICIPANT"


@dataclass(frozen=True)
class ObservedQuestionProfile:
    """A single observation of a question being asked about this participant."""

    question_family_key: str
    question_surface: str
    morphology_signature: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    observation_count: int = 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "question_family_key": self.question_family_key,
            "question_surface": self.question_surface,
            "morphology_signature": dict(self.morphology_signature),
            "confidence": self.confidence,
            "observation_count": self.observation_count,
        }


@dataclass(frozen=True)
class EventBindingFrameParticipant:
    """A participant within an EventBindingFrame with full lineage tracking."""

    frame_participant_id: str
    frame_id: str

    participant_node_id: str
    concept_id: str
    resolved_lemma: str
    canonical_surface: str
    local_slot_ids: Sequence[str] = ()

    morphology_profile: Mapping[str, Any] = field(default_factory=dict)

    origin: ParticipantOrigin = ParticipantOrigin.INFERRED_EVENT_PARTICIPANT

    lineage_root_gap_id: Optional[str] = None
    latest_source_gap_id: Optional[str] = None
    latest_source_binding_id: Optional[str] = None
    source_query_graph_ids: Sequence[str] = ()

    observed_question_profiles: Sequence[ObservedQuestionProfile] = ()
    compatible_question_profiles: Mapping[str, float] = field(default_factory=dict)

    binding_confidence: float = 0.0
    replaceable: bool = True
    last_released_turn: Optional[int] = None
    last_selected_turn: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "frame_participant_id": self.frame_participant_id,
            "frame_id": self.frame_id,
            "participant_node_id": self.participant_node_id,
            "concept_id": self.concept_id,
            "resolved_lemma": self.resolved_lemma,
            "canonical_surface": self.canonical_surface,
            "local_slot_ids": list(self.local_slot_ids),
            "morphology_profile": dict(self.morphology_profile),
            "origin": self.origin.value,
            "lineage_root_gap_id": self.lineage_root_gap_id,
            "latest_source_gap_id": self.latest_source_gap_id,
            "latest_source_binding_id": self.latest_source_binding_id,
            "source_query_graph_ids": list(self.source_query_graph_ids),
            "observed_question_profiles": [
                profile.as_dict() for profile in self.observed_question_profiles
            ],
            "compatible_question_profiles": dict(self.compatible_question_profiles),
            "binding_confidence": self.binding_confidence,
            "replaceable": self.replaceable,
            "last_released_turn": self.last_released_turn,
            "last_selected_turn": self.last_selected_turn,
        }


@dataclass(frozen=True)
class EventBindingFrame:
    """Stable dialogue projection of a selected event.

    Does NOT replace Event or QueryGraph.
    Event stores the fact.
    QueryGraph stores the current query.
    EventBindingFrame stores how participants relate to questions and bindings.
    """

    frame_id: str
    conversation_id: str
    root_query_graph_id: str
    latest_query_graph_id: str
    event_id: str
    predicate_concept_id: str

    status: FrameStatus = FrameStatus.ACTIVE
    confidence: float = 0.0
    created_at: str = ""
    updated_at: str = ""

    participants: Sequence[EventBindingFrameParticipant] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "conversation_id": self.conversation_id,
            "root_query_graph_id": self.root_query_graph_id,
            "latest_query_graph_id": self.latest_query_graph_id,
            "event_id": self.event_id,
            "predicate_concept_id": self.predicate_concept_id,
            "status": self.status.value,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "participants": [
                participant.as_dict() for participant in self.participants
            ],
        }

    def mark_released(self, participant_node_id: str, turn_index: int) -> "EventBindingFrame":
        """Mark a frame participant as released at the given turn index."""
        updated = list(self.participants)
        for i, p in enumerate(updated):
            if p.participant_node_id == participant_node_id:
                from dataclasses import replace as dc_replace
                updated[i] = dc_replace(p, last_released_turn=turn_index)
                break
        from dataclasses import replace as dc_replace
        return dc_replace(self, participants=tuple(updated))


class EventBindingFrameBuilder:
    """Constructs and updates EventBindingFrames from binding configurations."""

    @staticmethod
    def create_from_configuration(
        conversation_id: str,
        query_graph_id: str,
        event_id: str,
        predicate_concept_id: str,
        bindings: Sequence[Any],  # CandidateBinding
        event_participants: Sequence[Any],  # ParticipantNode
        gap_surfaces: Mapping[str, str],  # gap_node_id -> question surface
        turn_index: int,
    ) -> EventBindingFrame:
        """Create a new frame from a validated BindingConfiguration."""
        now = utcnow()
        frame_id = stable_id("event-binding-frame", conversation_id, event_id, query_graph_id)

        binding_by_participant: Dict[str, Any] = {}
        for binding in bindings:
            binding_by_participant[binding.resolved_node_id] = binding

        frame_participants: List[EventBindingFrameParticipant] = []
        for participant in event_participants:
            participant_node_id = participant.id
            binding = binding_by_participant.get(participant_node_id)

            # Determine origin
            if binding is not None:
                origin = ParticipantOrigin.RESOLVED_ROOT_GAP
            else:
                origin = ParticipantOrigin.INFERRED_EVENT_PARTICIPANT

            # Build observed question profiles
            observed: List[ObservedQuestionProfile] = []
            if binding is not None:
                from .question_family import resolve_question_family
                gap_id = binding.gap_node_id
                surface = gap_surfaces.get(gap_id, "")
                family_key = resolve_question_family(surface) or ""
                if family_key:
                    observed.append(ObservedQuestionProfile(
                        question_family_key=family_key,
                        question_surface=surface,
                        confidence=binding.total_score,
                        observation_count=1,
                    ))

            # Build compatible question profiles from morphology
            compatible: Dict[str, float] = {}
            mention = participant.mention
            animacy = str(mention.features.get("animacy") or "")
            if animacy == "anim":
                compatible["кто"] = 0.85
                compatible["что"] = 0.15
            elif animacy == "inan":
                compatible["кто"] = 0.10
                compatible["что"] = 0.90
            else:
                compatible["кто"] = 0.50
                compatible["что"] = 0.50

            # Add location family if participant has a preposition
            if mention.preposition:
                compatible["где"] = 0.70

            frame_participant = EventBindingFrameParticipant(
                frame_participant_id=stable_id(
                    "frame-participant", frame_id, participant_node_id,
                ),
                frame_id=frame_id,
                participant_node_id=participant_node_id,
                concept_id=mention.entity_id or stable_id("entity", mention.head_lemma),
                resolved_lemma=mention.head_lemma,
                canonical_surface=mention.surface,
                local_slot_ids=tuple(
                    hypothesis.local_slot_id
                    for hypothesis in participant.slot_hypotheses
                ),
                morphology_profile={
                    "case": str(mention.features.get("case") or ""),
                    "number": str(mention.features.get("number") or ""),
                    "gender": str(mention.features.get("gender") or ""),
                    "animacy": animacy,
                    "preposition": mention.preposition,
                },
                origin=origin,
                lineage_root_gap_id=binding.gap_node_id if binding else None,
                latest_source_gap_id=binding.gap_node_id if binding else None,
                latest_source_binding_id=binding.id if binding else None,
                source_query_graph_ids=(query_graph_id,),
                observed_question_profiles=tuple(observed),
                compatible_question_profiles=compatible,
                binding_confidence=binding.total_score if binding else 0.0,
                replaceable=True,  # All event participants can be query targets
                last_selected_turn=turn_index if binding else None,
            )
            frame_participants.append(frame_participant)

        return EventBindingFrame(
            frame_id=frame_id,
            conversation_id=conversation_id,
            root_query_graph_id=query_graph_id,
            latest_query_graph_id=query_graph_id,
            event_id=event_id,
            predicate_concept_id=predicate_concept_id,
            status=FrameStatus.ACTIVE,
            confidence=sum(p.binding_confidence for p in frame_participants) / max(1, len(frame_participants)),
            created_at=now,
            updated_at=now,
            participants=tuple(frame_participants),
        )

    @staticmethod
    def update_for_gap_release(
        frame: EventBindingFrame,
        released_participant_node_id: str,
        new_gap_id: str,
        new_gap_surface: str,
        new_query_graph_id: str,
        turn_index: int,
    ) -> EventBindingFrame:
        """Update frame after a GAP rotation releases a participant."""
        from .question_family import resolve_question_family

        family_key = resolve_question_family(new_gap_surface) or ""
        now = utcnow()

        updated_participants: List[EventBindingFrameParticipant] = []
        for participant in frame.participants:
            if participant.participant_node_id == released_participant_node_id:
                # Add new question profile observation
                new_observed = list(participant.observed_question_profiles)
                if family_key:
                    new_observed.append(ObservedQuestionProfile(
                        question_family_key=family_key,
                        question_surface=new_gap_surface,
                        confidence=0.90,
                        observation_count=1,
                    ))

                updated = EventBindingFrameParticipant(
                    frame_participant_id=participant.frame_participant_id,
                    frame_id=participant.frame_id,
                    participant_node_id=participant.participant_node_id,
                    concept_id=participant.concept_id,
                    resolved_lemma=participant.resolved_lemma,
                    canonical_surface=participant.canonical_surface,
                    local_slot_ids=participant.local_slot_ids,
                    morphology_profile=participant.morphology_profile,
                    origin=participant.origin,
                    lineage_root_gap_id=participant.lineage_root_gap_id,
                    latest_source_gap_id=new_gap_id,
                    latest_source_binding_id=participant.latest_source_binding_id,
                    source_query_graph_ids=tuple(list(participant.source_query_graph_ids) + [new_query_graph_id]),
                    observed_question_profiles=tuple(new_observed),
                    compatible_question_profiles=participant.compatible_question_profiles,
                    binding_confidence=participant.binding_confidence,
                    replaceable=participant.replaceable,
                    last_released_turn=turn_index,
                    last_selected_turn=participant.last_selected_turn,
                )
                updated_participants.append(updated)
            else:
                updated_participants.append(participant)

        return EventBindingFrame(
            frame_id=frame.frame_id,
            conversation_id=frame.conversation_id,
            root_query_graph_id=frame.root_query_graph_id,
            latest_query_graph_id=new_query_graph_id,
            event_id=frame.event_id,
            predicate_concept_id=frame.predicate_concept_id,
            status=frame.status,
            confidence=frame.confidence,
            created_at=frame.created_at,
            updated_at=now,
            participants=tuple(updated_participants),
        )

    @staticmethod
    def update_for_new_binding(
        frame: EventBindingFrame,
        new_binding: Any,  # CandidateBinding
        new_gap_surface: str,
        new_query_graph_id: str,
        turn_index: int,
    ) -> EventBindingFrame:
        """Update frame when a new binding is selected for a participant."""
        from .question_family import resolve_question_family

        family_key = resolve_question_family(new_gap_surface) or ""
        now = utcnow()

        updated_participants: List[EventBindingFrameParticipant] = []
        for participant in frame.participants:
            if participant.participant_node_id == new_binding.resolved_node_id:
                new_observed = list(participant.observed_question_profiles)
                if family_key:
                    new_observed.append(ObservedQuestionProfile(
                        question_family_key=family_key,
                        question_surface=new_gap_surface,
                        confidence=new_binding.total_score,
                        observation_count=1,
                    ))

                updated = EventBindingFrameParticipant(
                    frame_participant_id=participant.frame_participant_id,
                    frame_id=participant.frame_id,
                    participant_node_id=participant.participant_node_id,
                    concept_id=participant.concept_id,
                    resolved_lemma=participant.resolved_lemma,
                    canonical_surface=participant.canonical_surface,
                    local_slot_ids=participant.local_slot_ids,
                    morphology_profile=participant.morphology_profile,
                    origin=(
                        ParticipantOrigin.RESOLVED_LATER_GAP
                        if participant.origin == ParticipantOrigin.INFERRED_EVENT_PARTICIPANT
                        else participant.origin
                    ),
                    lineage_root_gap_id=participant.lineage_root_gap_id or new_binding.gap_node_id,
                    latest_source_gap_id=new_binding.gap_node_id,
                    latest_source_binding_id=new_binding.id,
                    source_query_graph_ids=tuple(list(participant.source_query_graph_ids) + [new_query_graph_id]),
                    observed_question_profiles=tuple(new_observed),
                    compatible_question_profiles=participant.compatible_question_profiles,
                    binding_confidence=new_binding.total_score,
                    replaceable=participant.replaceable,
                    last_selected_turn=turn_index,
                )
                updated_participants.append(updated)
            else:
                updated_participants.append(participant)

        return EventBindingFrame(
            frame_id=frame.frame_id,
            conversation_id=frame.conversation_id,
            root_query_graph_id=frame.root_query_graph_id,
            latest_query_graph_id=new_query_graph_id,
            event_id=frame.event_id,
            predicate_concept_id=frame.predicate_concept_id,
            status=frame.status,
            confidence=frame.confidence,
            created_at=frame.created_at,
            updated_at=now,
            participants=tuple(updated_participants),
        )