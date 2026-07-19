"""QuestionFamily profiles and animacy compatibility for role-free GAP binding.

QuestionFamily groups interrogative word-forms (кто/кому/кем) without
assigning semantic roles.  Animacy is a linguistic signal, not a role.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

from .graph_repository import stable_id


class AnimacyCompatibility(str, Enum):
    EXACT = "EXACT"
    COMPATIBLE = "COMPATIBLE"
    UNKNOWN = "UNKNOWN"
    CONFLICTING = "CONFLICTING"


class QuestionFamilyStatus(str, Enum):
    SHADOW = "SHADOW"
    PROBATION = "PROBATION"
    ACTIVE = "ACTIVE"


# Canonical Russian interrogative families.
# Each family_key is the nominative lemma.
FAMILY_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "кто": {
        "family_key": "кто",
        "canonical_lemma": "кто",
        "operator_type": "PARTICIPANT",
        "surfaces": {"кто", "кого", "кому", "кем", "ком", "кто́"},
        "animacy_preference": "anim",
    },
    "что": {
        "family_key": "что",
        "canonical_lemma": "что",
        "operator_type": "PARTICIPANT",
        "surfaces": {"что", "чего", "чему", "чем", "чём", "что́"},
        "animacy_preference": "inan",
    },
    "где": {
        "family_key": "где",
        "canonical_lemma": "где",
        "operator_type": "LOCATION",
        "surfaces": {"где"},
        "animacy_preference": None,
    },
    "когда": {
        "family_key": "когда",
        "canonical_lemma": "когда",
        "operator_type": "TIME",
        "surfaces": {"когда"},
        "animacy_preference": None,
    },
    "куда": {
        "family_key": "куда",
        "canonical_lemma": "куда",
        "operator_type": "DIRECTION",
        "surfaces": {"куда"},
        "animacy_preference": None,
    },
    "откуда": {
        "family_key": "откуда",
        "canonical_lemma": "откуда",
        "operator_type": "SOURCE",
        "surfaces": {"откуда"},
        "animacy_preference": None,
    },
    "зачем": {
        "family_key": "зачем",
        "canonical_lemma": "зачем",
        "operator_type": "PURPOSE",
        "surfaces": {"зачем"},
        "animacy_preference": None,
    },
    "почему": {
        "family_key": "почему",
        "canonical_lemma": "почему",
        "operator_type": "CAUSE",
        "surfaces": {"почему"},
        "animacy_preference": None,
    },
    "как": {
        "family_key": "как",
        "canonical_lemma": "как",
        "operator_type": "MANNER",
        "surfaces": {"как"},
        "animacy_preference": None,
    },
    "сколько": {
        "family_key": "сколько",
        "canonical_lemma": "сколько",
        "operator_type": "QUANTITY",
        "surfaces": {"сколько", "скольких", "скольким", "сколькими"},
        "animacy_preference": None,
    },
    "какой": {
        "family_key": "какой",
        "canonical_lemma": "какой",
        "operator_type": "ATTRIBUTE",
        "surfaces": {
            "какой", "какая", "какое", "какие",
            "какого", "какой", "какому", "каким", "каком",
        },
        "animacy_preference": None,
    },
    "чей": {
        "family_key": "чей",
        "canonical_lemma": "чей",
        "operator_type": "POSSESSOR",
        "surfaces": {"чей", "чья", "чьё", "чьи", "чьего", "чьей", "чьему", "чьим", "чьём"},
        "animacy_preference": "anim",
    },
}


def resolve_question_family(surface: str) -> Optional[str]:
    """Map an interrogative surface form to its family key."""
    normalized = str(surface).strip().casefold()
    for family_key, definition in FAMILY_DEFINITIONS.items():
        if normalized in definition["surfaces"]:
            return family_key
    return None


def question_family_animacy_preference(family_key: str) -> Optional[str]:
    """Return the preferred animacy for a question family, if any."""
    definition = FAMILY_DEFINITIONS.get(family_key)
    if definition is None:
        return None
    return definition.get("animacy_preference")


def check_animacy_compatibility(
    family_key: str,
    candidate_animacy: Optional[str],
) -> AnimacyCompatibility:
    """Evaluate whether a candidate's animacy is compatible with a question family."""
    preference = question_family_animacy_preference(family_key)
    if preference is None:
        return AnimacyCompatibility.UNKNOWN
    if candidate_animacy is None:
        return AnimacyCompatibility.UNKNOWN
    normalized_candidate = str(candidate_animacy).casefold()
    if normalized_candidate == preference:
        return AnimacyCompatibility.EXACT
    # "кто" prefers anim, "что" prefers inan
    if preference == "anim" and normalized_candidate == "inan":
        return AnimacyCompatibility.CONFLICTING
    if preference == "inan" and normalized_candidate == "anim":
        return AnimacyCompatibility.CONFLICTING
    return AnimacyCompatibility.COMPATIBLE


@dataclass(frozen=True)
class QuestionFamilyProfile:
    """Learned profile for a question family, built from validated bindings."""

    family_key: str
    canonical_lemma: str
    operator_type: str

    observed_surfaces: Sequence[str] = ()
    morphology_distributions: Mapping[str, float] = field(default_factory=dict)

    animacy_preference: Optional[str] = None
    animacy_confidence: float = 0.0
    animacy_evidence_count: int = 0

    compatible_local_slots: Mapping[str, float] = field(default_factory=dict)
    compatible_answer_clouds: Sequence[str] = ()
    compatible_event_relations: Sequence[str] = ()

    support_count: int = 0
    contradiction_count: int = 0
    confidence: float = 0.0
    status: QuestionFamilyStatus = QuestionFamilyStatus.SHADOW

    def as_dict(self) -> Dict[str, Any]:
        return {
            "family_key": self.family_key,
            "canonical_lemma": self.canonical_lemma,
            "operator_type": self.operator_type,
            "observed_surfaces": list(self.observed_surfaces),
            "morphology_distributions": dict(self.morphology_distributions),
            "animacy_preference": self.animacy_preference,
            "animacy_confidence": self.animacy_confidence,
            "animacy_evidence_count": self.animacy_evidence_count,
            "compatible_local_slots": dict(self.compatible_local_slots),
            "compatible_answer_clouds": list(self.compatible_answer_clouds),
            "compatible_event_relations": list(self.compatible_event_relations),
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "confidence": self.confidence,
            "status": self.status.value,
        }


class QuestionFamilyRegistry:
    """Central registry for question family profiles."""

    def __init__(self) -> None:
        self._profiles: Dict[str, QuestionFamilyProfile] = {}

    def get(self, family_key: str) -> Optional[QuestionFamilyProfile]:
        return self._profiles.get(family_key)

    def get_or_create(self, family_key: str) -> QuestionFamilyProfile:
        if family_key in self._profiles:
            return self._profiles[family_key]
        definition = FAMILY_DEFINITIONS.get(family_key, {})
        profile = QuestionFamilyProfile(
            family_key=family_key,
            canonical_lemma=definition.get("canonical_lemma", family_key),
            operator_type=definition.get("operator_type", "UNKNOWN"),
            observed_surfaces=tuple(sorted(definition.get("surfaces", {family_key}))),
            animacy_preference=definition.get("animacy_preference"),
        )
        self._profiles[family_key] = profile
        return profile

    def resolve_surface(self, surface: str) -> Optional[QuestionFamilyProfile]:
        family_key = resolve_question_family(surface)
        if family_key is None:
            return None
        return self.get_or_create(family_key)

    def all_profiles(self) -> Sequence[QuestionFamilyProfile]:
        return tuple(self._profiles.values())


# Singleton registry
question_family_registry = QuestionFamilyRegistry()