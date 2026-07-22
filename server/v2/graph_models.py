"""Role-free event graph contracts used by the SuperAI V3.0 pipeline.

The computational model deliberately knows only graph nodes, structural
attachments, observations and gaps.  Human-readable labels may be attached to
learned clusters for analytics, but are never consulted by these contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


EVENT_SCHEMA_VERSION = "3.0.0"
SLOT_MODEL_VERSION = "3.0.0"
CONSTRUCTION_MODEL_VERSION = "3.0.0"
SEMANTIC_CLUSTER_VERSION = "3.0.0"
QUERY_GRAPH_VERSION = "3.0.0"
GENERATION_VERSION = "3.0.0"
# The graph database is intentionally recreated when this contract changes.
# Query-operator occurrences became an explicit learned layer in 2.9, so
# there is no legacy merge path for databases without this evidence.
MIGRATION_VERSION = "fresh-v3.0-spatial-reset"


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class NodeType(StringEnum):
    EVENT = "EVENT"
    MENTION = "MENTION"
    ENTITY_REFERENCE = "ENTITY_REFERENCE"
    VALUE = "VALUE"
    RELATION_INSTANCE = "RELATION_INSTANCE"
    GAP = "GAP"
    CONSTRUCTION = "CONSTRUCTION"


class StructuralEdge(StringEnum):
    EVENT_HAS_PARTICIPANT = "EVENT_HAS_PARTICIPANT"
    MENTION_HAS_COMPONENT = "MENTION_HAS_COMPONENT"
    VALUE_ATTACHED_TO_NODE = "VALUE_ATTACHED_TO_NODE"
    COREFERS_TO = "COREFERS_TO"
    EXCLUDES = "EXCLUDES"
    CONTINUES = "CONTINUES"
    SUPPORTED_BY = "SUPPORTED_BY"
    CONTRADICTS = "CONTRADICTS"


class GapKind(StringEnum):
    EVENT_ATTACHMENT = "EVENT_ATTACHMENT"
    NODE_COMPONENT = "NODE_COMPONENT"
    RELATION_VALUE = "RELATION_VALUE"
    EVENT_PROPERTY = "EVENT_PROPERTY"
    BOOLEAN_RESULT = "BOOLEAN_RESULT"
    QUANTITY_VALUE = "QUANTITY_VALUE"
    WHOLE_EVENT = "WHOLE_EVENT"


class SlotStatus(StringEnum):
    CANDIDATE = "CANDIDATE"
    LOCAL = "LOCAL"
    STABLE = "STABLE"
    GENERALIZED = "GENERALIZED"
    WEAKENED = "WEAKENED"
    DEPRECATED = "DEPRECATED"


class GraphStatus(StringEnum):
    READY = "READY"
    AMBIGUOUS = "AMBIGUOUS"
    INCOMPLETE = "INCOMPLETE"
    CONFLICTED = "CONFLICTED"


class BindingStatus(StringEnum):
    CANDIDATE = "CANDIDATE"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    SELECTED = "SELECTED"


class AnswerStatus(StringEnum):
    RESOLVED = "RESOLVED"
    PARTIALLY_RESOLVED = "PARTIALLY_RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    AMBIGUOUS_BINDING = "AMBIGUOUS_BINDING"
    CONFLICTED = "CONFLICTED"
    BUILD_FAILED = "BUILD_FAILED"


_ALLOWED_OBSERVATION_NAMESPACES = frozenset({
    "morph",
    "position",
    "agreement",
    "disagreement",
    "construction",
    "entity_cluster",
    "polarity",
    "distance",
    "preposition",
    "question",
    "shape",
    "voice",
    "gap_kind",
    "context",
})


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=str)
    return value


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def assert_role_free_keys(keys: Iterable[str]) -> None:
    """Accept only namespaces representing observable or learned evidence."""
    for key in keys:
        namespace = str(key).split(":", 1)[0].casefold()
        if namespace not in _ALLOWED_OBSERVATION_NAMESPACES:
            raise ValueError(
                f"unsupported observation namespace: {namespace}"
            )


@dataclass(frozen=True)
class ModelVersions:
    event_schema: str = EVENT_SCHEMA_VERSION
    slot_model: str = SLOT_MODEL_VERSION
    construction_model: str = CONSTRUCTION_MODEL_VERSION
    semantic_cluster: str = SEMANTIC_CLUSTER_VERSION
    query_graph: str = QUERY_GRAPH_VERSION
    generation: str = GENERATION_VERSION
    migration: str = MIGRATION_VERSION

    def as_dict(self) -> Dict[str, str]:
        return {
            "event_schema_version": self.event_schema,
            "slot_model_version": self.slot_model,
            "construction_model_version": self.construction_model,
            "semantic_cluster_version": self.semantic_cluster,
            "query_graph_version": self.query_graph,
            "generation_version": self.generation,
            "migration_version": self.migration,
        }


@dataclass(frozen=True)
class ObservationSignature:
    values: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_role_free_keys(self.values)
        normalized = {
            str(key): _clamp(float(value))
            for key, value in self.values.items()
            if float(value) > 0.0
        }
        object.__setattr__(self, "values", MappingProxyType(normalized))

    def as_dict(self) -> Dict[str, float]:
        return dict(self.values)


@dataclass(frozen=True)
class MorphHypothesis:
    id: str
    lemma: str
    part_of_speech: str
    features: Mapping[str, Any]
    morph_score: float
    selected: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "lemma": self.lemma,
            "part_of_speech": self.part_of_speech,
            "features": _plain(self.features),
            "morph_score": _clamp(self.morph_score),
            "selected": bool(self.selected),
        }


@dataclass(frozen=True)
class MentionComponent:
    id: str
    lemma: str
    surface: str
    token_index: int
    attachment_signature: ObservationSignature
    required: bool = True
    grammatical_features: Mapping[str, Any] = field(default_factory=dict)
    evidence: Sequence[str] = ()
    confidence: float = 0.82

    def as_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.id,
            "lemma": self.lemma,
            "surface": self.surface,
            "token_index": self.token_index,
            "attachment_signature": self.attachment_signature.as_dict(),
            "required": bool(self.required),
            "grammatical_features": _plain(self.grammatical_features),
            "evidence": list(self.evidence),
            "confidence": _clamp(self.confidence),
        }


@dataclass(frozen=True)
class MentionNode:
    id: str
    head_lemma: str
    head_surface: str
    surface: str
    token_start: int
    token_end: int
    token_indices: Sequence[int]
    features: Mapping[str, Any]
    components: Sequence[MentionComponent] = ()
    preposition: str = ""
    entity_id: Optional[str] = None
    semantic_cluster_ids: Sequence[str] = ()
    origin: str = "EXPLICIT_CURRENT"
    source_query_graph_id: Optional[str] = None
    source_gap_id: Optional[str] = None
    source_binding_id: Optional[str] = None
    replaceable: bool = False
    context_confidence: float = 1.0

    @property
    def qualified_key(self) -> str:
        component_lemmas = sorted(component.lemma for component in self.components)
        return "|".join((self.head_lemma, *component_lemmas))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.id,
            "node_type": NodeType.MENTION.value,
            "head": {
                "lemma": self.head_lemma,
                "surface": self.head_surface,
            },
            "surface": self.surface,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_indices": list(self.token_indices),
            "features": _plain(self.features),
            "components": [component.as_dict() for component in self.components],
            "preposition": self.preposition,
            "entity_id": self.entity_id,
            "semantic_cluster_ids": list(self.semantic_cluster_ids),
            "origin": self.origin,
            "source_query_graph_id": self.source_query_graph_id,
            "source_gap_id": self.source_gap_id,
            "source_binding_id": self.source_binding_id,
            "replaceable": self.replaceable,
            "context_confidence": _clamp(self.context_confidence),
            "qualified_key": self.qualified_key,
        }


@dataclass(frozen=True)
class PredicateNode:
    lemma: str
    surface: str
    concept_id: str
    token_index: Optional[int]
    features: Mapping[str, Any]
    origin: str = "CURRENT_EXPLICIT"
    source_token_index: Optional[int] = None
    inherited_from_query_graph_id: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lemma": self.lemma,
            "surface": self.surface,
            "concept_id": self.concept_id,
            "token_index": self.token_index,
            "features": _plain(self.features),
            "origin": self.origin,
            "source_token_index": self.source_token_index,
            "inherited_from_query_graph_id": self.inherited_from_query_graph_id,
        }


@dataclass(frozen=True)
class PredicatePerspectiveRelation:
    """Evidence-backed mapping between two predicate-local slot spaces.

    It intentionally has no named semantic roles.  A relation is usable only
    after its observed slot permutation has enough support; a dialogue anchor
    remains a separate, local form of evidence.
    """

    source_predicate_concept_id: str
    target_predicate_concept_id: str
    slot_permutation: Mapping[str, str]
    evidence_count: int
    confidence: float
    context_support: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_predicate_concept_id": self.source_predicate_concept_id,
            "target_predicate_concept_id": self.target_predicate_concept_id,
            "slot_permutation": _plain(self.slot_permutation),
            "evidence_count": int(self.evidence_count),
            "confidence": _clamp(self.confidence),
            "context_support": _plain(self.context_support),
        }


@dataclass(frozen=True)
class SlotHypothesis:
    local_slot_id: str
    compatibility: float
    evidence: Sequence[str] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "local_slot_id": self.local_slot_id,
            "compatibility": _clamp(self.compatibility),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ParticipantNode:
    id: str
    mention: MentionNode
    observation_signature: ObservationSignature
    slot_hypotheses: Sequence[SlotHypothesis] = ()
    confidence: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "participant_id": self.id,
            "node_type": NodeType.ENTITY_REFERENCE.value,
            "mention": self.mention.as_dict(),
            "observation_signature": self.observation_signature.as_dict(),
            "slot_hypotheses": [
                hypothesis.as_dict() for hypothesis in self.slot_hypotheses
            ],
            "confidence": _clamp(self.confidence),
        }


@dataclass(frozen=True)
class EventNode:
    id: str
    predicate: PredicateNode
    participants: Sequence[ParticipantNode]
    properties: Sequence[Mapping[str, Any]] = ()
    construction_id: Optional[str] = None
    polarity: str = "POSITIVE"
    actuality: str = "ACTUAL"
    confidence: float = 0.0
    raw_text: str = ""
    source_surface: str = ""
    token_start: int = 0
    token_end: int = 0
    sentence_index: int = 0
    versions: ModelVersions = field(default_factory=ModelVersions)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.id,
            "node_type": NodeType.EVENT.value,
            "predicate": self.predicate.as_dict(),
            "participants": [
                participant.as_dict() for participant in self.participants
            ],
            "properties": _plain(self.properties),
            "construction_id": self.construction_id,
            "polarity": self.polarity,
            "actuality": self.actuality,
            "confidence": _clamp(self.confidence),
            "raw_text": self.raw_text,
            "source_surface": self.source_surface,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "sentence_index": self.sentence_index,
            "versions": self.versions.as_dict(),
        }


@dataclass(frozen=True)
class GapNode:
    id: str
    gap_kind: GapKind
    question_signature: ObservationSignature
    surface: str
    token_indices: Sequence[int]
    attached_to_node_id: Optional[str] = None
    compatible_slot_hypotheses: Mapping[str, float] = field(default_factory=dict)
    # A gap is requested only when it is the value the user asked for.  Past
    # tense agreement may also reveal an omitted participant; that participant
    # is represented as an implicit, non-requested gap.
    requested: bool = True
    required: bool = True
    coordination_group_id: Optional[str] = None
    morphology_hypotheses: Mapping[str, float] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compatible_slot_hypotheses",
            MappingProxyType({
                str(key): _clamp(value)
                for key, value in self.compatible_slot_hypotheses.items()
            }),
        )
        object.__setattr__(
            self,
            "morphology_hypotheses",
            MappingProxyType({
                str(key): _clamp(value)
                for key, value in self.morphology_hypotheses.items()
            }),
        )
        object.__setattr__(self, "evidence", MappingProxyType(_plain(self.evidence)))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.id,
            "node_type": NodeType.GAP.value,
            "gap_kind": self.gap_kind.value,
            "surface": self.surface,
            "token_indices": list(self.token_indices),
            "attached_to_node_id": self.attached_to_node_id,
            "question_signature": self.question_signature.as_dict(),
            "compatible_slot_hypotheses": dict(
                self.compatible_slot_hypotheses
            ),
            "requested": self.requested,
            "required": self.required,
            "coordination_group_id": self.coordination_group_id,
            "morphology_hypotheses": dict(self.morphology_hypotheses),
            "evidence": _plain(self.evidence),
        }


@dataclass(frozen=True)
class QueryGraph:
    id: str
    predicate: Optional[PredicateNode]
    known_nodes: Sequence[MentionNode]
    gap_node: GapNode
    target_gaps: Sequence[GapNode] = ()
    question_operators: Sequence[Mapping[str, Any]] = ()
    required_edges: Sequence[Mapping[str, Any]] = ()
    exclusions: Sequence[Mapping[str, Any]] = ()
    status: GraphStatus = GraphStatus.READY
    continuation_of: Optional[str] = None
    construction_ids: Sequence[str] = ()
    implicit_gaps: Sequence[GapNode] = ()
    trace: Mapping[str, Any] = field(default_factory=dict)
    versions: ModelVersions = field(default_factory=ModelVersions)

    def as_dict(self) -> Dict[str, Any]:
        target_gaps = tuple(self.target_gaps) or (self.gap_node,)
        pattern: Dict[str, Any] = {
            "predicate": self.predicate.as_dict() if self.predicate else None,
            "known_nodes": [node.as_dict() for node in self.known_nodes],
            "target_gaps": [gap.as_dict() for gap in target_gaps],
            "implicit_gaps": [gap.as_dict() for gap in self.implicit_gaps],
            "required_edges": _plain(self.required_edges),
            # Legacy readers still consume this field.  It is meaningful only
            # for a one-gap request.
            "gap_node": self.gap_node.as_dict(),
        }
        if len(target_gaps) == 1:
            pattern["target_gap"] = target_gaps[0].as_dict()
        return {
            "query_graph_id": self.id,
            "question_operators": _plain(self.question_operators),
            "event_pattern": pattern,
            "exclusions": _plain(self.exclusions),
            "status": self.status.value,
            "continuation_of": self.continuation_of,
            "construction_ids": list(self.construction_ids),
            "trace": _plain(self.trace),
            "versions": self.versions.as_dict(),
        }

    @property
    def target_gap(self) -> GapNode:
        """Compatibility accessor for callers that only support one GAP."""
        return self.gap_node


@dataclass(frozen=True)
class CandidateBinding:
    id: str
    query_graph_id: str
    event_id: str
    gap_node_id: str
    resolved_node_id: str
    resolved_concept_id: str
    resolved_lemma: str
    resolved_surface: str
    resolved_features: Mapping[str, Any]
    structural_score: float
    signature_score: float
    evidence_score: float
    total_score: float
    status: BindingStatus = BindingStatus.CANDIDATE
    failed_constraint: Optional[str] = None
    evidence: Sequence[Mapping[str, Any]] = ()
    slot_compatibility_state: str = "fallback"
    # Selection answers "was this candidate returned?"; support answers
    # "how well is the returned value evidenced?".  They intentionally do
    # not collapse into the historical BindingStatus enum.
    selection_status: str = ""
    support_status: str = ""
    configuration_id: str = ""

    def __post_init__(self) -> None:
        if self.slot_compatibility_state not in {
            "compatible",
            "below_threshold",
            "fallback",
            "rejected",
        }:
            raise ValueError(
                "slot_compatibility_state must be compatible, "
                "below_threshold, fallback, or rejected"
            )
        if self.support_status and self.support_status not in {
            "SUPPORTED", "WEAK_SUPPORTED", "BELOW_THRESHOLD",
            "FALLBACK_ONLY", "CONFLICTING", "REJECTED",
        }:
            raise ValueError("unsupported binding support_status")

    def _support_status(self) -> str:
        if self.support_status:
            return self.support_status
        return {
            "compatible": "SUPPORTED",
            "below_threshold": "WEAK_SUPPORTED",
            "fallback": "FALLBACK_ONLY",
            "rejected": "REJECTED",
        }[self.slot_compatibility_state]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "binding_id": self.id,
            "query_graph_id": self.query_graph_id,
            "event_id": self.event_id,
            "gap_node_id": self.gap_node_id,
            "resolved_node_id": self.resolved_node_id,
            "resolved_concept_id": self.resolved_concept_id,
            "resolved_lemma": self.resolved_lemma,
            "resolved_surface": self.resolved_surface,
            "resolved_features": _plain(self.resolved_features),
            "scores": {
                "structural": _clamp(self.structural_score),
                "signature": _clamp(self.signature_score),
                "evidence": _clamp(self.evidence_score),
                "total": _clamp(self.total_score),
            },
            "status": self.status.value,
            "selection_status": self.selection_status or self.status.value,
            "support_status": self._support_status(),
            "configuration_id": self.configuration_id or None,
            "failed_constraint": self.failed_constraint,
            "slot_compatibility_state": self.slot_compatibility_state,
            "evidence": _plain(self.evidence),
        }


@dataclass(frozen=True)
class BindingConfiguration:
    """Atomic result for one or more requested GAPs in one event."""

    id: str
    query_graph_id: str
    event_id: str
    bindings: Sequence[CandidateBinding]
    all_required_gaps_bound: bool
    distinct_node_count: int
    configuration_score: float
    graph_validation: Mapping[str, Any]
    status: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "configuration_id": self.id,
            "query_graph_id": self.query_graph_id,
            "event_id": self.event_id,
            "bindings_by_gap": {
                binding.gap_node_id: binding.as_dict()
                for binding in self.bindings
            },
            "all_required_gaps_bound": self.all_required_gaps_bound,
            "distinct_node_count": int(self.distinct_node_count),
            "configuration_score": _clamp(self.configuration_score),
            "graph_validation": _plain(self.graph_validation),
            "status": self.status,
        }


@dataclass(frozen=True)
class LocalSlot:
    id: str
    predicate_concept_id: str
    centroid_signature: ObservationSignature
    support_count: int
    contradiction_count: int
    domain_diversity: int
    confidence: float
    status: SlotStatus
    display_label: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "local_slot_id": self.id,
            "predicate_concept_id": self.predicate_concept_id,
            "centroid_signature": self.centroid_signature.as_dict(),
            "support_count": int(self.support_count),
            "contradiction_count": int(self.contradiction_count),
            "domain_diversity": int(self.domain_diversity),
            "confidence": _clamp(self.confidence),
            "status": self.status.value,
            "display_label": self.display_label,
            "display_label_is_non_computational": True,
        }


@dataclass(frozen=True)
class SlotSet:
    id: str
    predicate_concept_id: str
    local_slot_ids: Sequence[str]
    support_count: int
    confidence: float
    status: SlotStatus

    def as_dict(self) -> Dict[str, Any]:
        return {
            "slot_set_id": self.id,
            "predicate_concept_id": self.predicate_concept_id,
            "local_slot_ids": list(self.local_slot_ids),
            "support_count": int(self.support_count),
            "confidence": _clamp(self.confidence),
            "status": self.status.value,
        }


@dataclass(frozen=True)
class SlotPrototype:
    id: str
    member_slot_ids: Sequence[str]
    centroid_signature: ObservationSignature
    support_count: int
    domain_diversity: int
    confidence: float
    display_label: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prototype_id": self.id,
            "member_slot_ids": list(self.member_slot_ids),
            "centroid_signature": self.centroid_signature.as_dict(),
            "support_count": int(self.support_count),
            "domain_diversity": int(self.domain_diversity),
            "confidence": _clamp(self.confidence),
            "display_label": self.display_label,
            "display_label_is_non_computational": True,
        }


@dataclass(frozen=True)
class ConstructionCluster:
    id: str
    structural_signature: ObservationSignature
    gap_kind: Optional[GapKind]
    compatible_slot_prototypes: Mapping[str, float]
    support_count: int
    contradiction_count: int
    domain_diversity: int
    confidence: float
    status: SlotStatus

    def as_dict(self) -> Dict[str, Any]:
        return {
            "construction_id": self.id,
            "structural_signature": self.structural_signature.as_dict(),
            "gap_placement": {
                "gap_kind": self.gap_kind.value if self.gap_kind else None,
                "compatible_slot_prototypes": dict(
                    self.compatible_slot_prototypes
                ),
            },
            "support_count": int(self.support_count),
            "contradiction_count": int(self.contradiction_count),
            "domain_diversity": int(self.domain_diversity),
            "confidence": _clamp(self.confidence),
            "status": self.status.value,
        }


def mention_node_from_dict(value: Mapping[str, Any]) -> MentionNode:
    head = value.get("head") or {}
    return MentionNode(
        id=str(value.get("node_id") or value.get("mention_id") or ""),
        head_lemma=str(head.get("lemma") or value.get("head_lemma") or ""),
        head_surface=str(head.get("surface") or value.get("head_surface") or ""),
        surface=str(value.get("surface") or ""),
        token_start=int(value.get("token_start") or 0),
        token_end=int(value.get("token_end") or 0),
        token_indices=tuple(value.get("token_indices") or ()),
        features=dict(value.get("features") or {}),
        components=tuple(
            MentionComponent(
                id=str(item.get("component_id") or ""),
                lemma=str(item.get("lemma") or ""),
                surface=str(item.get("surface") or ""),
                token_index=int(item.get("token_index") or 0),
                attachment_signature=ObservationSignature(
                    item.get("attachment_signature") or {}
                ),
                required=bool(item.get("required", True)),
                grammatical_features=dict(item.get("grammatical_features") or {}),
                evidence=tuple(item.get("evidence") or ()),
                confidence=float(item.get("confidence") or 0.82),
            )
            for item in value.get("components") or ()
        ),
        preposition=str(value.get("preposition") or ""),
        entity_id=value.get("entity_id"),
        semantic_cluster_ids=tuple(value.get("semantic_cluster_ids") or ()),
        origin=str(value.get("origin") or "EXPLICIT_CURRENT"),
        source_query_graph_id=value.get("source_query_graph_id"),
        source_gap_id=value.get("source_gap_id"),
        source_binding_id=value.get("source_binding_id"),
        replaceable=bool(value.get("replaceable", False)),
        context_confidence=float(value.get("context_confidence") or 1.0),
    )


def query_graph_from_dict(value: Mapping[str, Any]) -> QueryGraph:
    pattern = value.get("event_pattern") or {}
    predicate_value = pattern.get("predicate")
    predicate = (
        PredicateNode(
            lemma=str(predicate_value.get("lemma") or ""),
            surface=str(predicate_value.get("surface") or ""),
            concept_id=str(predicate_value.get("concept_id") or ""),
            token_index=predicate_value.get("token_index"),
            features=dict(predicate_value.get("features") or {}),
            origin=str(predicate_value.get("origin") or "CURRENT"),
            source_token_index=predicate_value.get("source_token_index"),
            inherited_from_query_graph_id=predicate_value.get(
                "inherited_from_query_graph_id"
            ),
        )
        if predicate_value else None
    )
    gap_values = pattern.get("target_gaps") or [
        pattern.get("target_gap") or pattern.get("gap_node") or {}
    ]
    def parse_gap(gap_value: Mapping[str, Any]) -> GapNode:
        return GapNode(
        id=str(gap_value.get("node_id") or ""),
        gap_kind=GapKind(
            str(gap_value.get("gap_kind") or GapKind.WHOLE_EVENT.value)
        ),
        question_signature=ObservationSignature(
            gap_value.get("question_signature") or {}
        ),
        surface=str(gap_value.get("surface") or ""),
        token_indices=tuple(gap_value.get("token_indices") or ()),
        attached_to_node_id=gap_value.get("attached_to_node_id"),
        compatible_slot_hypotheses=dict(
            gap_value.get("compatible_slot_hypotheses") or {}
        ),
            requested=bool(gap_value.get("requested", True)),
            required=bool(gap_value.get("required", True)),
            coordination_group_id=gap_value.get("coordination_group_id"),
            morphology_hypotheses=dict(
                gap_value.get("morphology_hypotheses") or {}
            ),
            evidence=dict(gap_value.get("evidence") or {}),
        )
    target_gaps = tuple(parse_gap(item) for item in gap_values)
    gap = target_gaps[0]
    implicit_gaps = tuple(
        GapNode(
            id=str(item.get("node_id") or ""),
            gap_kind=GapKind(str(item.get("gap_kind") or GapKind.EVENT_ATTACHMENT.value)),
            question_signature=ObservationSignature(
                item.get("question_signature") or {}
            ),
            surface=str(item.get("surface") or ""),
            token_indices=tuple(item.get("token_indices") or ()),
            attached_to_node_id=item.get("attached_to_node_id"),
            compatible_slot_hypotheses=dict(
                item.get("compatible_slot_hypotheses") or {}
            ),
            requested=bool(item.get("requested", False)),
            required=bool(item.get("required", False)),
            coordination_group_id=item.get("coordination_group_id"),
            morphology_hypotheses=dict(
                item.get("morphology_hypotheses") or {}
            ),
            evidence=dict(item.get("evidence") or {}),
        )
        for item in pattern.get("implicit_gaps") or ()
    )
    return QueryGraph(
        id=str(value.get("query_graph_id") or ""),
        predicate=predicate,
        known_nodes=tuple(
            mention_node_from_dict(item)
            for item in pattern.get("known_nodes") or ()
        ),
        gap_node=gap,
        target_gaps=target_gaps,
        question_operators=tuple(value.get("question_operators") or ()),
        required_edges=tuple(pattern.get("required_edges") or ()),
        exclusions=tuple(value.get("exclusions") or ()),
        status=GraphStatus(str(value.get("status") or GraphStatus.READY.value)),
        continuation_of=value.get("continuation_of"),
        construction_ids=tuple(value.get("construction_ids") or ()),
        implicit_gaps=implicit_gaps,
        trace=dict(value.get("trace") or {}),
    )


def candidate_binding_from_dict(
    value: Optional[Mapping[str, Any]],
) -> Optional[CandidateBinding]:
    if not value:
        return None
    scores = value.get("scores") or {}
    return CandidateBinding(
        id=str(value.get("binding_id") or ""),
        query_graph_id=str(value.get("query_graph_id") or ""),
        event_id=str(value.get("event_id") or ""),
        gap_node_id=str(value.get("gap_node_id") or ""),
        resolved_node_id=str(value.get("resolved_node_id") or ""),
        resolved_concept_id=str(value.get("resolved_concept_id") or ""),
        resolved_lemma=str(value.get("resolved_lemma") or ""),
        resolved_surface=str(value.get("resolved_surface") or ""),
        resolved_features=dict(value.get("resolved_features") or {}),
        structural_score=float(scores.get("structural") or 0.0),
        signature_score=float(scores.get("signature") or 0.0),
        evidence_score=float(scores.get("evidence") or 0.0),
        total_score=float(scores.get("total") or 0.0),
        status=BindingStatus(
            str(value.get("status") or BindingStatus.CANDIDATE.value)
        ),
        failed_constraint=value.get("failed_constraint"),
        evidence=tuple(value.get("evidence") or ()),
        slot_compatibility_state=str(
            value.get("slot_compatibility_state") or "fallback"
        ),
        selection_status=str(value.get("selection_status") or ""),
        support_status=str(value.get("support_status") or ""),
        configuration_id=str(value.get("configuration_id") or ""),
    )
