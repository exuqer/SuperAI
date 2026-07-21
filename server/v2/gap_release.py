"""GAP release scoring and diagnostic for role-free event binding frames.

Implements the select_participant_to_release algorithm that uses
EventBindingFrame, QuestionFamily, animacy, morphology, and lineage
instead of flat case-matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .event_binding_frame import (
    EventBindingFrame,
    EventBindingFrameParticipant,
)
from .question_family import (
    AnimacyCompatibility,
    check_animacy_compatibility,
    resolve_question_family,
)


class ReleaseDecision(str, Enum):
    RELEASED = "RELEASED"
    AMBIGUOUS = "AMBIGUOUS"
    NO_COMPATIBLE_PARTICIPANT = "NO_COMPATIBLE_PARTICIPANT"


# Configurable scoring weights
DEFAULT_RELEASE_WEIGHTS = {
    "exact_surface_match": 1.00,
    "question_family_match": 0.95,
    "root_gap_lineage_match": 0.90,
    "local_slot_score": 0.75,
    "animacy_score": 0.65,
    "case_score": 0.50,
    "morphology_score": 0.25,
    "frame_confidence": 0.15,
    "recency_score": 0.10,
    "explicit_current_penalty": -2.00,
    "animacy_conflict": -1.50,
    "hard_slot_conflict": -1.00,
}

DEFAULT_MIN_RELEASE_SCORE = 0.25
DEFAULT_MIN_RELEASE_MARGIN = 0.12


@dataclass(frozen=True)
class GapReleaseCandidate:
    """A single candidate for GAP release with full scoring breakdown."""

    participant_node_id: str
    concept_id: str
    resolved_surface: str

    exact_surface_match: float = 0.0
    question_family_match: float = 0.0
    root_gap_lineage_match: float = 0.0
    latest_gap_lineage_match: float = 0.0
    local_slot_score: float = 0.0
    animacy_score: float = 0.0
    case_score: float = 0.0
    morphology_score: float = 0.0
    frame_confidence: float = 0.0
    recency_score: float = 0.0

    explicit_current_penalty: float = 0.0
    animacy_conflict: float = 0.0
    hard_slot_conflict: float = 0.0

    final_score: float = 0.0
    rank: int = 0
    accepted: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "participant_node_id": self.participant_node_id,
            "concept_id": self.concept_id,
            "resolved_surface": self.resolved_surface,
            "exact_surface_match": self.exact_surface_match,
            "question_family_match": self.question_family_match,
            "root_gap_lineage_match": self.root_gap_lineage_match,
            "latest_gap_lineage_match": self.latest_gap_lineage_match,
            "local_slot_score": self.local_slot_score,
            "animacy_score": self.animacy_score,
            "case_score": self.case_score,
            "morphology_score": self.morphology_score,
            "frame_confidence": self.frame_confidence,
            "recency_score": self.recency_score,
            "explicit_current_penalty": self.explicit_current_penalty,
            "animacy_conflict": self.animacy_conflict,
            "hard_slot_conflict": self.hard_slot_conflict,
            "final_score": self.final_score,
            "rank": self.rank,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class GapReleaseDiagnostic:
    """Full diagnostic for a GAP release decision."""

    query_graph_id: str
    frame_id: Optional[str]
    event_id: Optional[str]
    question_family_key: Optional[str]
    execution_id: str = ""
    hypothesis_id: str = ""
    gap_id: str = ""
    status: str = "SELECTED"

    candidates: Sequence[GapReleaseCandidate] = ()

    selected_participant_node_id: Optional[str] = None
    selected_score: float = 0.0
    second_score: float = 0.0
    release_margin: float = 0.0

    decision: ReleaseDecision = ReleaseDecision.NO_COMPATIBLE_PARTICIPANT
    decision_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query_graph_id": self.query_graph_id,
            "execution_id": self.execution_id,
            "hypothesis_id": self.hypothesis_id,
            "gap_id": self.gap_id,
            "status": self.status,
            "frame_id": self.frame_id,
            "event_id": self.event_id,
            "question_family_key": self.question_family_key,
            "candidates": [c.as_dict() for c in self.candidates],
            "selected_participant_node_id": self.selected_participant_node_id,
            "selected_score": self.selected_score,
            "second_score": self.second_score,
            "release_margin": self.release_margin,
            "decision": self.decision.value,
            "decision_reason": self.decision_reason,
        }


class GapReleaseSelector:
    """Selects which participant to release for a GAP rotation."""

    def __init__(
        self,
        weights: Optional[Mapping[str, float]] = None,
        min_release_score: float = DEFAULT_MIN_RELEASE_SCORE,
        min_release_margin: float = DEFAULT_MIN_RELEASE_MARGIN,
    ) -> None:
        self.weights = dict(weights or DEFAULT_RELEASE_WEIGHTS)
        self.min_release_score = min_release_score
        self.min_release_margin = min_release_margin

    def select_participant_to_release(
        self,
        current_question_surface: str,
        active_frame: EventBindingFrame,
        current_explicit_node_ids: set[str],
        previous_query_result: Optional[Mapping[str, Any]] = None,
        event_anchor_id: Optional[str] = None,
        query_graph_id: str = "",
    ) -> Tuple[Optional[str], GapReleaseDiagnostic]:
        """Select the best participant to release for the current question.

        Returns (participant_node_id or None, diagnostic).
        """
        question_family_key = resolve_question_family(current_question_surface)

        # Build candidates from frame participants
        candidates: List[GapReleaseCandidate] = []
        for participant in active_frame.participants:
            # Skip explicit current nodes
            if participant.participant_node_id in current_explicit_node_ids:
                continue

            # Skip non-replaceable participants
            if not participant.replaceable:
                continue

            candidate = self._score_candidate(
                participant=participant,
                question_surface=current_question_surface,
                question_family_key=question_family_key,
                current_explicit_node_ids=current_explicit_node_ids,
            )
            candidates.append(candidate)

        if not candidates:
            return None, GapReleaseDiagnostic(
                query_graph_id=query_graph_id,
                frame_id=active_frame.frame_id,
                event_id=active_frame.event_id,
                question_family_key=question_family_key,
                decision=ReleaseDecision.NO_COMPATIBLE_PARTICIPANT,
                decision_reason="NO_CANDIDATES_IN_FRAME",
            )

        # Sort by final score descending
        candidates.sort(key=lambda c: c.final_score, reverse=True)

        # Assign ranks
        ranked: List[GapReleaseCandidate] = []
        for rank, candidate in enumerate(candidates, start=1):
            ranked.append(GapReleaseCandidate(
                participant_node_id=candidate.participant_node_id,
                concept_id=candidate.concept_id,
                resolved_surface=candidate.resolved_surface,
                exact_surface_match=candidate.exact_surface_match,
                question_family_match=candidate.question_family_match,
                root_gap_lineage_match=candidate.root_gap_lineage_match,
                latest_gap_lineage_match=candidate.latest_gap_lineage_match,
                local_slot_score=candidate.local_slot_score,
                animacy_score=candidate.animacy_score,
                case_score=candidate.case_score,
                morphology_score=candidate.morphology_score,
                frame_confidence=candidate.frame_confidence,
                recency_score=candidate.recency_score,
                explicit_current_penalty=candidate.explicit_current_penalty,
                animacy_conflict=candidate.animacy_conflict,
                hard_slot_conflict=candidate.hard_slot_conflict,
                final_score=candidate.final_score,
                rank=rank,
                accepted=rank == 1,
            ))

        best = ranked[0]
        second_best = ranked[1] if len(ranked) > 1 else None
        second_score = second_best.final_score if second_best else 0.0
        release_margin = best.final_score - second_score

        # Check thresholds
        if best.final_score < self.min_release_score:
            return None, GapReleaseDiagnostic(
                query_graph_id=query_graph_id,
                frame_id=active_frame.frame_id,
                event_id=active_frame.event_id,
                question_family_key=question_family_key,
                candidates=tuple(ranked),
                selected_score=best.final_score,
                second_score=second_score,
                release_margin=release_margin,
                decision=ReleaseDecision.AMBIGUOUS,
                decision_reason="BEST_SCORE_BELOW_MINIMUM",
            )

        if release_margin < self.min_release_margin:
            return None, GapReleaseDiagnostic(
                query_graph_id=query_graph_id,
                frame_id=active_frame.frame_id,
                event_id=active_frame.event_id,
                question_family_key=question_family_key,
                candidates=tuple(ranked),
                selected_score=best.final_score,
                second_score=second_score,
                release_margin=release_margin,
                decision=ReleaseDecision.AMBIGUOUS,
                decision_reason="RELEASE_MARGIN_TOO_LOW",
            )

        return best.participant_node_id, GapReleaseDiagnostic(
            query_graph_id=query_graph_id,
            frame_id=active_frame.frame_id,
            event_id=active_frame.event_id,
            question_family_key=question_family_key,
            candidates=tuple(ranked),
            selected_participant_node_id=best.participant_node_id,
            selected_score=best.final_score,
            second_score=second_score,
            release_margin=release_margin,
            decision=ReleaseDecision.RELEASED,
            decision_reason="CLEAR_WINNER",
        )

    def _score_candidate(
        self,
        participant: EventBindingFrameParticipant,
        question_surface: str,
        question_family_key: Optional[str],
        current_explicit_node_ids: set[str],
    ) -> GapReleaseCandidate:
        """Score a single frame participant for release."""
        w = self.weights
        morph = participant.morphology_profile

        # 1. Exact surface match
        exact_surface_match = 0.0
        for profile in participant.observed_question_profiles:
            if profile.question_surface.casefold() == question_surface.casefold():
                exact_surface_match = 1.0
                break

        # 2. Question family match
        question_family_match = 0.0
        if question_family_key:
            for profile in participant.observed_question_profiles:
                if profile.question_family_key == question_family_key:
                    question_family_match = 1.0
                    break
            if question_family_match == 0.0:
                # Check compatible profiles
                compat = participant.compatible_question_profiles.get(
                    question_family_key, 0.0
                )
                question_family_match = compat

        # 3. Root GAP lineage match
        root_gap_lineage_match = 0.0
        if question_family_key and participant.lineage_root_gap_id:
            # Check if root gap was from the same family
            for profile in participant.observed_question_profiles:
                if profile.question_family_key == question_family_key:
                    root_gap_lineage_match = 1.0
                    break

        # 4. Latest GAP lineage match
        latest_gap_lineage_match = 0.0
        if question_family_key and participant.latest_source_gap_id:
            for profile in participant.observed_question_profiles:
                if profile.question_family_key == question_family_key:
                    latest_gap_lineage_match = 1.0
                    break

        # 5. Local slot score (from compatible profiles)
        local_slot_score = 0.0
        if question_family_key:
            local_slot_score = participant.compatible_question_profiles.get(
                question_family_key, 0.0
            )

        # 6. Animacy score
        animacy_score = 0.0
        animacy_conflict = 0.0
        if question_family_key:
            candidate_animacy = morph.get("animacy")
            compatibility = check_animacy_compatibility(
                question_family_key, candidate_animacy,
            )
            if compatibility == AnimacyCompatibility.EXACT:
                animacy_score = 1.0
            elif compatibility == AnimacyCompatibility.COMPATIBLE:
                animacy_score = 0.5
            elif compatibility == AnimacyCompatibility.CONFLICTING:
                animacy_score = 0.0
                animacy_conflict = 1.0
            else:
                animacy_score = 0.3

        # 7. Case score
        case_score = 0.0
        participant_case = str(morph.get("case") or "")
        # Check if the question surface implies a case
        # This is a simplified version; full implementation would use morphology
        if participant_case:
            # Basic case compatibility: if participant has a case, it's somewhat relevant
            case_score = 0.5

        # 8. Morphology score
        morphology_score = 0.0
        number = str(morph.get("number") or "")
        gender = str(morph.get("gender") or "")
        if number or gender:
            morphology_score = 0.3  # Base morphology presence

        # 9. Frame confidence
        frame_confidence = participant.binding_confidence

        # 10. Recency score
        recency_score = 0.0
        if participant.last_selected_turn is not None:
            recency_score = 0.8  # Recently selected
        elif participant.last_released_turn is not None:
            recency_score = 0.4  # Recently released

        # Penalties
        explicit_current_penalty = (
            1.0 if participant.participant_node_id in current_explicit_node_ids
            else 0.0
        )

        hard_slot_conflict = 0.0
        # Check if there's a hard slot conflict (e.g., incompatible local slot)
        if question_family_key:
            compat = participant.compatible_question_profiles.get(
                question_family_key, 0.5
            )
            if compat < 0.1:
                hard_slot_conflict = 1.0

        # Calculate final score
        final_score = (
            w.get("exact_surface_match", 1.0) * exact_surface_match
            + w.get("question_family_match", 0.95) * question_family_match
            + w.get("root_gap_lineage_match", 0.90) * root_gap_lineage_match
            + w.get("local_slot_score", 0.75) * local_slot_score
            + w.get("animacy_score", 0.65) * animacy_score
            + w.get("case_score", 0.50) * case_score
            + w.get("morphology_score", 0.25) * morphology_score
            + w.get("frame_confidence", 0.15) * frame_confidence
            + w.get("recency_score", 0.10) * recency_score
            + w.get("explicit_current_penalty", -2.0) * explicit_current_penalty
            + w.get("animacy_conflict", -1.5) * animacy_conflict
            + w.get("hard_slot_conflict", -1.0) * hard_slot_conflict
        )

        return GapReleaseCandidate(
            participant_node_id=participant.participant_node_id,
            concept_id=participant.concept_id,
            resolved_surface=participant.canonical_surface,
            exact_surface_match=exact_surface_match,
            question_family_match=question_family_match,
            root_gap_lineage_match=root_gap_lineage_match,
            latest_gap_lineage_match=latest_gap_lineage_match,
            local_slot_score=local_slot_score,
            animacy_score=animacy_score,
            case_score=case_score,
            morphology_score=morphology_score,
            frame_confidence=frame_confidence,
            recency_score=recency_score,
            explicit_current_penalty=explicit_current_penalty,
            animacy_conflict=animacy_conflict,
            hard_slot_conflict=hard_slot_conflict,
            final_score=final_score,
        )
