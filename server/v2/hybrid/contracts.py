"""Small, serialisable contracts for the deterministic hybrid workspace."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _id(prefix: str, *parts: object) -> str:
    value = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(item) for item in value]
    return value


@dataclass(frozen=True)
class Gap:
    gap_id: str
    source_query_id: str
    expected_type: str = "unknown"
    expected_relation: Optional[str] = None
    known_elements: Sequence[Any] = ()
    surface_projection: str = ""
    constraints: Sequence[Mapping[str, Any]] = ()
    exclusions: Sequence[str] = ()
    status: str = "OPEN"
    expected_signature: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "source_query_id": self.source_query_id,
            "expected_type": self.expected_type,
            "expected_relation": self.expected_relation,
            "known_elements": plain(self.known_elements),
            "surface_projection": self.surface_projection,
            "constraints": plain(self.constraints),
            "exclusions": list(self.exclusions),
            "status": self.status,
            "expected_signature": plain(self.expected_signature),
        }


class CandidateRelation(str, Enum):
    DUPLICATE = "DUPLICATE"
    COMPATIBLE_SET_MEMBER = "COMPATIBLE_SET_MEMBER"
    ALTERNATIVE = "ALTERNATIVE"
    CONTRADICTORY = "CONTRADICTORY"
    UNRELATED = "UNRELATED"
    UNKNOWN = "UNKNOWN"


class AnswerCardinality(str, Enum):
    NONE = "NONE"
    SINGLE = "SINGLE"
    MULTIPLE = "MULTIPLE"
    ALTERNATIVES = "ALTERNATIVES"
    CONFLICTING = "CONFLICTING"


class AnswerStatus(str, Enum):
    STABLE_SINGLE = "STABLE_SINGLE"
    STABLE_SET = "STABLE_SET"
    AMBIGUOUS_SELECTION = "AMBIGUOUS_SELECTION"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ENUMERATION_EXHAUSTED = "ENUMERATION_EXHAUSTED"


class HypothesisType(str, Enum):
    SINGLE_FILL = "SINGLE_FILL"
    SET_FILL = "SET_FILL"
    ALTERNATIVE_FILL = "ALTERNATIVE_FILL"
    JOINT_MULTI_GAP_FILL = "JOINT_MULTI_GAP_FILL"


class EnumerationPolicy(str, Enum):
    FIRST_THEN_CONTINUE = "FIRST_THEN_CONTINUE"
    RETURN_ALL = "RETURN_ALL"
    TOP_K = "TOP_K"
    SINGLE_BEST = "SINGLE_BEST"


class EnumerationStatus(str, Enum):
    ACTIVE = "ACTIVE"
    EXHAUSTED = "EXHAUSTED"
    INVALIDATED = "INVALIDATED"
    SUPERSEDED = "SUPERSEDED"


@dataclass(frozen=True)
class CandidateCompatibility:
    left_candidate_id: str
    right_candidate_id: str
    relation: CandidateRelation
    reason: str = ""
    confidence: float = 0.0
    structural_match: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {key: plain(value) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class GapFillSet:
    gap_id: str
    cardinality: AnswerCardinality
    members: Sequence[Mapping[str, Any]] = ()
    independent_source_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {key: plain(value) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class EnumerationState:
    enumeration_id: str
    source_query_id: str
    source_gap_id: str
    query_signature: Mapping[str, Any] = field(default_factory=dict)
    all_candidate_ids: Sequence[str] = ()
    delivered_candidate_ids: Sequence[str] = ()
    remaining_candidate_ids: Sequence[str] = ()
    excluded_candidate_ids: Sequence[str] = ()
    ordering: Sequence[str] = ()
    policy: str = EnumerationPolicy.FIRST_THEN_CONTINUE.value
    exhausted: bool = False
    field_revision: Optional[int] = None
    graph_revision: Optional[int] = None
    all_element_ids: Sequence[str] = ()
    delivered_element_ids: Sequence[str] = ()
    remaining_element_ids: Sequence[str] = ()
    status: str = EnumerationStatus.ACTIVE.value

    def as_dict(self) -> Dict[str, Any]:
        return {key: plain(value) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class QueryFrame:
    query_id: str
    session_id: str
    raw_text: str
    normalized_text: str
    query_type: str
    explicit_predicate: Optional[str] = None
    surface_focus: Optional[str] = None
    known_elements: Sequence[Any] = ()
    gaps: Sequence[Gap] = ()
    constraints: Sequence[Mapping[str, Any]] = ()
    negations: Sequence[str] = ()
    exclusions: Sequence[str] = ()
    temporal_scope: Optional[Mapping[str, Any]] = None
    continuation_of: Optional[str] = None
    confidence: float = 0.0
    inherited_elements: Sequence[Mapping[str, Any]] = ()
    reconstructed_query: Optional[str] = None
    unresolved_context: bool = False
    context_inheritance: Mapping[str, Any] = field(default_factory=lambda: {
        "mode": "BLOCK",
        "source_query_id": None,
        "inherited_elements": [],
    })
    predicate_hypotheses: Sequence[Mapping[str, Any]] = ()
    constraint_groups: Sequence[Mapping[str, Any]] = ()
    trace: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "session_id": self.session_id,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "query_type": self.query_type,
            "explicit_predicate": self.explicit_predicate,
            "surface_focus": self.surface_focus,
            "known_elements": plain(self.known_elements),
            "gaps": [gap.as_dict() for gap in self.gaps],
            "constraints": plain(self.constraints),
            "negations": list(self.negations),
            "exclusions": list(self.exclusions),
            "temporal_scope": plain(self.temporal_scope),
            "continuation_of": self.continuation_of,
            "confidence": clamp(self.confidence),
            "inherited_elements": plain(self.inherited_elements),
            "reconstructed_query": self.reconstructed_query,
            "unresolved_context": self.unresolved_context,
            "context_inheritance": plain(self.context_inheritance),
            "predicate_hypotheses": plain(self.predicate_hypotheses),
            "constraint_groups": plain(self.constraint_groups),
            "trace": plain(self.trace),
        }


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_type: str
    source_id: str
    event_id: Optional[str] = None
    scene_id: Optional[str] = None
    supports: Sequence[str] = ()
    strength: float = 0.0
    trust_status: str = "OBSERVED_UNTRUSTED"
    retrieval_path: Sequence[str] = ()
    conflicts: Sequence[str] = ()
    independent_source_key: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "event_id": self.event_id,
            "scene_id": self.scene_id,
            "supports": list(self.supports),
            "strength": clamp(self.strength),
            "trust_status": self.trust_status,
            "retrieval_path": list(self.retrieval_path),
            "conflicts": list(self.conflicts),
            "independent_source_key": self.independent_source_key,
        }


@dataclass(frozen=True)
class GraphEvidence(Evidence):
    """Evidence grounded in a persisted graph record, never a field hit."""


@dataclass(frozen=True)
class IndexTrace:
    trace_id: str
    index_name: str
    element_id: str
    route: Sequence[str] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"trace_id": self.trace_id, "index_name": self.index_name, "element_id": self.element_id, "route": list(self.route)}


@dataclass(frozen=True)
class BeeDiscovery:
    discovery_id: str
    bee_id: str
    element_id: str
    mode: str
    utility: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"discovery_id": self.discovery_id, "bee_id": self.bee_id, "element_id": self.element_id, "mode": self.mode, "utility": clamp(self.utility)}


@dataclass(frozen=True)
class SpatialSupport:
    support_id: str
    cloud_id: str
    region_id: Optional[str] = None
    score: float = 0.0
    relation_alignment: float = 0.0
    field_revision: int = 0
    retrieval_path: Sequence[str] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"support_id": self.support_id, "cloud_id": self.cloud_id, "region_id": self.region_id, "score": clamp(self.score), "relation_alignment": clamp(self.relation_alignment), "field_revision": self.field_revision, "retrieval_path": list(self.retrieval_path)}


@dataclass(frozen=True)
class RetrievalHit:
    hit_id: str
    element_id: str
    element_type: str
    source_id: str
    match_score: float
    matched_features: Sequence[str] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)
    provenance: Sequence[Mapping[str, Any]] = ()
    conflicts: Sequence[str] = ()
    retrieval_path: Sequence[str] = ()
    origin: str = "GRAPH"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hit_id": self.hit_id,
            "element_id": self.element_id,
            "element_type": self.element_type,
            "source_id": self.source_id,
            "match_score": clamp(self.match_score),
            "matched_features": list(self.matched_features),
            "payload": plain(self.payload),
            "provenance": plain(self.provenance),
            "conflicts": list(self.conflicts),
            "retrieval_path": list(self.retrieval_path),
            "origin": self.origin,
        }


@dataclass(frozen=True)
class ActivationResult:
    activations: Mapping[str, float]
    paths: Mapping[str, Sequence[str]] = field(default_factory=dict)
    hits: Sequence[RetrievalHit] = ()
    steps: int = 0
    visited: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "activations": {key: clamp(value) for key, value in self.activations.items()},
            "paths": plain(self.paths),
            "hits": [item.as_dict() for item in self.hits],
            "steps": self.steps,
            "visited": self.visited,
        }


@dataclass(frozen=True)
class WorkspaceBudget:
    max_anchors: int = 12
    max_context_elements: int = 32
    max_entities: int = 48
    max_events: int = 64
    max_scenes: int = 16
    max_clouds: int = 48
    max_candidates_per_gap: int = 16
    max_hypotheses: int = 8
    max_conflicts: int = 16
    max_evidence_per_hypothesis: int = 12
    max_total_elements: int = 256

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


@dataclass
class WorkspaceElement:
    element_id: str
    element_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    activation: float = 0.0
    workspace_functions: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    retrieval_paths: List[str] = field(default_factory=list)
    provenance: List[Mapping[str, Any]] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    conflict_ids: List[str] = field(default_factory=list)

    def merge(self, other: "WorkspaceElement") -> None:
        self.activation = max(self.activation, other.activation)
        for field_name in ("workspace_functions", "source_ids", "retrieval_paths", "evidence_ids", "conflict_ids"):
            target = getattr(self, field_name)
            for item in getattr(other, field_name):
                if item not in target:
                    target.append(item)
        for item in other.provenance:
            if item not in self.provenance:
                self.provenance.append(item)
        self.payload = {**other.payload, **self.payload}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "payload": plain(self.payload),
            "activation": clamp(self.activation),
            "workspace_functions": list(self.workspace_functions),
            "source_ids": list(self.source_ids),
            "retrieval_paths": list(self.retrieval_paths),
            "provenance": plain(self.provenance),
            "evidence_ids": list(self.evidence_ids),
            "conflict_ids": list(self.conflict_ids),
        }


@dataclass
class Candidate:
    candidate_id: str
    gap_id: str
    element_id: str
    activation: float = 0.0
    query_match: float = 0.0
    event_fit: float = 0.0
    context_fit: float = 0.0
    type_fit: float = 0.0
    gap_fit: float = 0.0
    provenance_score: float = 0.0
    conflict_score: float = 0.0
    exclusion_score: float = 0.0
    redundancy_score: float = 0.0
    semantic_distance: float = 0.0
    field_fit: float = 0.0
    mutual_support: float = 0.0
    constraint_fit: float = 1.0
    constraint_violations: List[str] = field(default_factory=list)
    score: float = 0.0
    status: str = "ACTIVE"
    evidence_ids: List[str] = field(default_factory=list)
    graph_evidence_ids: List[str] = field(default_factory=list)
    spatial_support_ids: List[str] = field(default_factory=list)
    independent_source_keys: List[str] = field(default_factory=list)
    provenance: List[Mapping[str, Any]] = field(default_factory=list)
    event_id: Optional[str] = None
    supporting_event_ids: List[str] = field(default_factory=list)
    surface: Optional[str] = None
    lemma: Optional[str] = None
    configuration_id: Optional[str] = None
    origin: Mapping[str, Any] = field(default_factory=dict)
    structural_signature: Mapping[str, Any] = field(default_factory=dict)
    polarity: str = "POSITIVE"
    temporal_scope: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {key: plain(value) for key, value in self.__dict__.items()}


@dataclass
class Hypothesis:
    hypothesis_id: str
    fills: Dict[str, Sequence[str]] = field(default_factory=dict)
    gap_fill_sets: Dict[str, GapFillSet] = field(default_factory=dict)
    candidate_ids: List[str] = field(default_factory=list)
    supporting_events: List[str] = field(default_factory=list)
    supporting_scenes: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    conflict_ids: List[str] = field(default_factory=list)
    score: float = 0.0
    status: str = "ACTIVE"
    provenance: List[Mapping[str, Any]] = field(default_factory=list)
    configuration_ids: List[str] = field(default_factory=list)
    graph_evidence_ids: List[str] = field(default_factory=list)
    spatial_support_ids: List[str] = field(default_factory=list)
    active_cloud_ids: List[str] = field(default_factory=list)
    field_region_id: Optional[str] = None
    evidential_score: float = 0.0
    spatial_score: float = 0.0
    joint_score: float = 0.0
    hypothesis_type: str = HypothesisType.SINGLE_FILL.value

    def as_dict(self) -> Dict[str, Any]:
        return {key: plain(value) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class Conflict:
    conflict_id: str
    subject_id: str
    competing_ids: Sequence[str]
    reason: str
    severity: float = 0.0
    provenance: Sequence[Mapping[str, Any]] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {key: plain(value) for key, value in self.__dict__.items()}


@dataclass
class EventCandidateConfiguration:
    configuration_id: str
    query_id: str
    event_id: str
    gap_bindings: Dict[str, str] = field(default_factory=dict)
    known_member_bindings: List[Mapping[str, Any]] = field(default_factory=list)
    predicate_binding: Optional[Mapping[str, Any]] = None
    relation_matches: List[Mapping[str, Any]] = field(default_factory=list)
    component_matches: List[Mapping[str, Any]] = field(default_factory=list)
    temporal_match: Optional[Mapping[str, Any]] = None
    constraint_evaluations: List[Mapping[str, Any]] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    conflict_ids: List[str] = field(default_factory=list)
    score: float = 0.0
    status: str = "ACTIVE"
    rejection_reasons: List[str] = field(default_factory=list)
    graph_fit: float = 0.0
    field_fit: float = 0.0
    gradient_alignment: float = 0.0
    cloud_overlap: float = 0.0
    semantic_region_id: Optional[str] = None
    known_bindings_by_gap: Mapping[str, Any] = field(default_factory=dict)
    relation_evaluations: List[Mapping[str, Any]] = field(default_factory=list)
    predicate_evaluation: Mapping[str, Any] = field(default_factory=dict)
    temporal_evaluation: Mapping[str, Any] = field(default_factory=dict)
    polarity_evaluation: Mapping[str, Any] = field(default_factory=dict)
    state_evaluation: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {key: plain(value) for key, value in self.__dict__.items()}


@dataclass
class BoundedAssociativeWorkspace:
    workspace_id: str
    query_id: str
    session_id: str
    query_type: str = "canonical_query"
    budget: WorkspaceBudget = field(default_factory=WorkspaceBudget)
    anchors: List[WorkspaceElement] = field(default_factory=list)
    active_context: List[WorkspaceElement] = field(default_factory=list)
    entities: List[WorkspaceElement] = field(default_factory=list)
    events: List[WorkspaceElement] = field(default_factory=list)
    scenes: List[WorkspaceElement] = field(default_factory=list)
    clouds: List[WorkspaceElement] = field(default_factory=list)
    field_region: Mapping[str, Any] = field(default_factory=dict)
    local_gradients: List[Mapping[str, Any]] = field(default_factory=list)
    graph_evidence: List[Evidence] = field(default_factory=list)
    spatial_support: List[SpatialSupport] = field(default_factory=list)
    gaps: List[Gap] = field(default_factory=list)
    constraints: List[Mapping[str, Any]] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)
    temporal_scope: Optional[Mapping[str, Any]] = None
    explicit_predicate: Optional[str] = None
    predicate_hypotheses: List[Mapping[str, Any]] = field(default_factory=list)
    candidates: List[Candidate] = field(default_factory=list)
    configurations: List[EventCandidateConfiguration] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    resonance_state: Dict[str, Any] = field(default_factory=lambda: {"iteration": 0, "stability": 0.0, "leader_id": None})
    snapshots: List[Mapping[str, Any]] = field(default_factory=list)
    status: str = "BUILDING"
    evictions: List[Mapping[str, Any]] = field(default_factory=list)
    candidate_relations: List[CandidateCompatibility] = field(default_factory=list)
    candidate_groups: List[Mapping[str, Any]] = field(default_factory=list)
    enumeration_policy: str = EnumerationPolicy.FIRST_THEN_CONTINUE.value
    enumeration_state: Optional[EnumerationState] = None

    def _section(self, element_type: str) -> List[WorkspaceElement]:
        return {
            "entity": self.entities,
            "entities": self.entities,
            "event": self.events,
            "events": self.events,
            "scene": self.scenes,
            "scenes": self.scenes,
            "cloud": self.clouds,
            "clouds": self.clouds,
            "context": self.active_context,
            "active_context": self.active_context,
            "anchor": self.anchors,
            "anchors": self.anchors,
        }.get(element_type, self.entities)

    def add_element(self, element: WorkspaceElement, section: Optional[str] = None) -> WorkspaceElement:
        target = self._section(section or element.element_type)
        existing = next((item for item in target if item.element_id == element.element_id), None)
        if existing is not None:
            existing.merge(element)
            return existing
        target.append(element)
        self._enforce_budget()
        return element

    def add_evidence(self, evidence: Evidence) -> None:
        if not any(item.evidence_id == evidence.evidence_id for item in self.evidence):
            self.evidence.append(evidence)

    def add_candidate(self, candidate: Candidate) -> None:
        existing = next((item for item in self.candidates if item.candidate_id == candidate.candidate_id), None)
        if existing is None:
            self.candidates.append(candidate)
        else:
            existing.evidence_ids = list(dict.fromkeys(existing.evidence_ids + candidate.evidence_ids))
            existing.graph_evidence_ids = list(dict.fromkeys(existing.graph_evidence_ids + candidate.graph_evidence_ids))
            existing.spatial_support_ids = list(dict.fromkeys(existing.spatial_support_ids + candidate.spatial_support_ids))
            existing.independent_source_keys = list(dict.fromkeys(existing.independent_source_keys + candidate.independent_source_keys))
            existing.supporting_event_ids = list(dict.fromkeys(existing.supporting_event_ids + candidate.supporting_event_ids))
            if candidate.configuration_id and not existing.configuration_id:
                existing.configuration_id = candidate.configuration_id
            existing.origin = {**candidate.origin, **existing.origin}
            for value in candidate.provenance:
                if value not in existing.provenance:
                    existing.provenance.append(value)
            if candidate.score > existing.score:
                existing.activation = candidate.activation
                existing.query_match = candidate.query_match
                existing.event_fit = candidate.event_fit
                existing.context_fit = candidate.context_fit
                existing.type_fit = candidate.type_fit
                existing.gap_fit = candidate.gap_fit
                existing.provenance_score = candidate.provenance_score
                existing.conflict_score = candidate.conflict_score
                existing.exclusion_score = candidate.exclusion_score
                existing.semantic_distance = candidate.semantic_distance
                existing.constraint_fit = candidate.constraint_fit
                existing.constraint_violations = list(candidate.constraint_violations)
                existing.score = candidate.score
                existing.event_id = candidate.event_id
                existing.surface = candidate.surface
                existing.lemma = candidate.lemma
        self.candidates.sort(key=lambda item: (-item.score, item.candidate_id))
        by_gap: Dict[str, List[Candidate]] = {}
        for item in self.candidates:
            by_gap.setdefault(item.gap_id, []).append(item)
        self.candidates = [item for gap in sorted(by_gap) for item in by_gap[gap][:self.budget.max_candidates_per_gap]]

    def total_elements(self) -> int:
        return sum(len(getattr(self, name)) for name in ("anchors", "active_context", "entities", "events", "scenes", "clouds"))

    def _enforce_budget(self) -> None:
        limits = {"anchors": self.budget.max_anchors, "active_context": self.budget.max_context_elements, "entities": self.budget.max_entities, "events": self.budget.max_events, "scenes": self.budget.max_scenes, "clouds": self.budget.max_clouds}
        for name, limit in limits.items():
            section = getattr(self, name)
            if len(section) > limit:
                protected = set(item.element_id for item in self.anchors) | {gap.gap_id for gap in self.gaps}
                section.sort(key=lambda item: (item.element_id in protected, item.activation, bool(item.provenance), item.element_id))
                while len(section) > limit:
                    removed = section.pop(0)
                    self.evictions.append({"element_id": removed.element_id, "reason": "SECTION_BUDGET", "activation": removed.activation})
        while self.total_elements() > self.budget.max_total_elements:
            sections = [self.active_context, self.entities, self.events, self.scenes, self.clouds]
            candidates = [item for section in sections for item in section if item.element_id not in {a.element_id for a in self.anchors}]
            if not candidates:
                break
            removed = min(candidates, key=lambda item: (item.activation, bool(item.provenance), item.element_id))
            for section in sections:
                if removed in section:
                    section.remove(removed)
                    break
            self.evictions.append({"element_id": removed.element_id, "reason": "TOTAL_BUDGET", "activation": removed.activation})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id, "query_id": self.query_id, "session_id": self.session_id,
            "query_type": self.query_type,
            "anchors": [item.as_dict() for item in self.anchors],
            "active_context": [item.as_dict() for item in self.active_context],
            "entities": [item.as_dict() for item in self.entities],
            "events": [item.as_dict() for item in self.events],
            "scenes": [item.as_dict() for item in self.scenes],
            "field_region": plain(self.field_region),
            "active_clouds": [item.as_dict() for item in self.clouds],
            "local_gradients": plain(self.local_gradients),
            "graph_evidence": [item.as_dict() for item in self.graph_evidence],
            "spatial_support": [item.as_dict() for item in self.spatial_support],
            "gaps": [item.as_dict() for item in self.gaps],
            "constraints": plain(self.constraints), "exclusions": list(self.exclusions),
            "temporal_scope": plain(self.temporal_scope),
            "explicit_predicate": self.explicit_predicate,
            "predicate_hypotheses": plain(self.predicate_hypotheses),
            "candidates": [item.as_dict() for item in self.candidates],
            "configurations": [item.as_dict() for item in self.configurations],
            "hypotheses": [item.as_dict() for item in self.hypotheses],
            "conflicts": [item.as_dict() for item in self.conflicts],
            "evidence": [item.as_dict() for item in self.evidence],
            "resonance_state": plain(self.resonance_state), "snapshots": plain(self.snapshots),
            "candidate_relations": [item.as_dict() for item in self.candidate_relations],
            "candidate_groups": plain(self.candidate_groups),
            "enumeration_policy": self.enumeration_policy,
            "enumeration_state": self.enumeration_state.as_dict() if self.enumeration_state else None,
            "budget": self.budget.as_dict(), "evictions": plain(self.evictions), "status": self.status,
        }


@dataclass(frozen=True)
class BeeTask:
    bee_task_id: str
    task_type: str
    gap_id: Optional[str] = None
    anchors: Sequence[str] = ()
    excluded_elements: Sequence[str] = ()
    search_scope: str = "session_and_global"
    max_steps: int = 4
    energy_budget: int = 32
    required_evidence: bool = True
    bee_mode: str = "GRAPH_BEE"

    def as_dict(self) -> Dict[str, Any]:
        return {key: plain(value) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class BeeResult:
    bee_id: str
    task_id: str
    status: str
    result_element_id: Optional[str] = None
    result_element_type: Optional[str] = None
    path: Sequence[str] = ()
    score: float = 0.0
    energy_spent: int = 0
    provenance: Sequence[Mapping[str, Any]] = ()
    conflicts: Sequence[str] = ()
    reason: str = ""
    bee_mode: str = "GRAPH_BEE"
    payload: Mapping[str, Any] = field(default_factory=dict)
    graph_evidence_ids: Sequence[str] = ()
    spatial_support_ids: Sequence[str] = ()
    independent_source_key: Optional[str] = None
    route: Sequence[str] = ()
    actual_cost: int = 0
    utility: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {key: plain(value) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class AnswerStructure:
    answer_type: str
    status: str
    filled_gaps: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    evidence_ids: Sequence[str] = ()
    rejected_candidates: Sequence[Mapping[str, Any]] = ()
    uncertainties: Sequence[str] = ()
    generation_constraints: Mapping[str, Any] = field(default_factory=lambda: {"language": "ru", "do_not_invent": True, "state_uncertainty": True})
    provenance: Mapping[str, Any] = field(default_factory=dict)
    epistemic_mode: str = "UNKNOWN"
    graph_support: float = 0.0
    field_support: float = 0.0
    graph_evidence: Sequence[str] = ()
    spatial_support: Sequence[str] = ()
    independent_source_count: int = 0
    answer_cardinality: str = AnswerCardinality.NONE.value
    set_members: Sequence[Mapping[str, Any]] = ()
    delivered_members: Sequence[Mapping[str, Any]] = ()
    remaining_members: Sequence[Mapping[str, Any]] = ()
    alternative_fills: Mapping[str, Any] = field(default_factory=dict)
    ambiguity_reason: Optional[str] = None
    conflict_reason: Optional[str] = None
    enumeration_state: Optional[EnumerationState] = None

    def as_dict(self) -> Dict[str, Any]:
        return {key: plain(value) for key, value in self.__dict__.items()}
