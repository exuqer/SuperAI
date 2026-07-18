"""Data structures for the shared phrase-aware language analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional


INTERPRETATION_VERSION = "dialogue-v2.5"


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class DialogueActType(StringEnum):
    GREETING = "GREETING"
    SMALL_TALK = "SMALL_TALK"
    ASSERTION = "ASSERTION"
    QUESTION = "QUESTION"
    REQUEST = "REQUEST"
    COMMAND = "COMMAND"
    CONFIRMATION = "CONFIRMATION"
    DENIAL = "DENIAL"
    CORRECTION = "CORRECTION"
    CLARIFICATION_REQUEST = "CLARIFICATION_REQUEST"
    DEFINITION = "DEFINITION"
    ASSUMPTION = "ASSUMPTION"
    HYPOTHESIS = "HYPOTHESIS"
    CONDITION = "CONDITION"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    DESIRE = "DESIRE"
    PLAN = "PLAN"
    QUOTE = "QUOTE"
    REPORTED_SPEECH = "REPORTED_SPEECH"
    EXAMPLE = "EXAMPLE"


class ClauseMode(StringEnum):
    ASSERTION = "ASSERTION"
    QUESTION = "QUESTION"
    COMMAND = "COMMAND"
    REQUEST = "REQUEST"
    DEFINITION = "DEFINITION"
    ASSUMPTION = "ASSUMPTION"
    HYPOTHESIS = "HYPOTHESIS"
    CONDITION = "CONDITION"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    DESIRE = "DESIRE"
    PLAN = "PLAN"
    QUOTE = "QUOTE"
    REPORTED_SPEECH = "REPORTED_SPEECH"
    EXAMPLE = "EXAMPLE"


class Actuality(StringEnum):
    ACTUAL = "ACTUAL"
    POSSIBLE = "POSSIBLE"
    HYPOTHETICAL = "HYPOTHETICAL"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    FICTIONAL = "FICTIONAL"
    UNKNOWN = "UNKNOWN"


class EvidenceStatus(StringEnum):
    OBSERVED = "OBSERVED"
    STATED = "STATED"
    INFERRED = "INFERRED"
    DISPUTED = "DISPUTED"
    REJECTED = "REJECTED"


class Polarity(StringEnum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class CompletionStatus(StringEnum):
    COMPLETED = "COMPLETED"
    ONGOING = "ONGOING"
    PLANNED = "PLANNED"
    INTERRUPTED = "INTERRUPTED"
    NOT_STARTED = "NOT_STARTED"
    UNKNOWN = "UNKNOWN"


class Modality(StringEnum):
    CAN = "CAN"
    MUST = "MUST"
    MAY = "MAY"
    SHOULD = "SHOULD"
    WANT = "WANT"
    INTEND = "INTEND"
    TRY = "TRY"
    BELIEVE = "BELIEVE"
    KNOW = "KNOW"


class ClauseRelationType(StringEnum):
    SEQUENCE = "SEQUENCE"
    SIMULTANEOUS = "SIMULTANEOUS"
    CAUSE = "CAUSE"
    RESULT = "RESULT"
    CONDITION = "CONDITION"
    CONCESSION = "CONCESSION"
    PURPOSE = "PURPOSE"
    CONTRAST = "CONTRAST"
    EXPLANATION = "EXPLANATION"
    ENUMERATION = "ENUMERATION"
    ALTERNATIVE = "ALTERNATIVE"
    QUOTE_CONTENT = "QUOTE_CONTENT"
    REPORTED_CONTENT = "REPORTED_CONTENT"


class HypothesisStatus(StringEnum):
    EPHEMERAL = "EPHEMERAL"
    PROVISIONAL = "PROVISIONAL"
    CONFIRMED = "CONFIRMED"
    QUARANTINED = "QUARANTINED"
    SUPERSEDED = "SUPERSEDED"
    RETRACTED = "RETRACTED"


class InterpretationStatus(StringEnum):
    STABLE = "STABLE"
    AMBIGUOUS = "AMBIGUOUS"
    INCOMPLETE = "INCOMPLETE"
    CONFLICTED = "CONFLICTED"


class CommitmentStatus(StringEnum):
    ACTIVE = "ACTIVE"
    CONFIRMED_IN_DIALOGUE = "CONFIRMED_IN_DIALOGUE"
    DISPUTED = "DISPUTED"
    SUPERSEDED = "SUPERSEDED"
    RETRACTED = "RETRACTED"


class ResponseType(StringEnum):
    DIRECT = "DIRECT"
    FULL = "FULL"
    CLARIFICATION = "CLARIFICATION"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICT = "CONFLICT"
    CONFIRMATION = "CONFIRMATION"
    CORRECTION_ACK = "CORRECTION_ACK"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): _freeze(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_value(item) for item in value), key=str)
    return value


@dataclass(frozen=True)
class EvidencePacket:
    id: str
    origin: str
    target_hypothesis_id: str
    value: Any
    support: float
    evidence_type: str
    independent_group: str
    scope_type: str
    scope_id: str
    penalty: float = 0.0
    source_token_start: Optional[int] = None
    source_token_end: Optional[int] = None
    source_object_id: Optional[str] = None
    parser_version: str = INTERPRETATION_VERSION

    def __post_init__(self) -> None:
        support = float(self.support)
        penalty = float(self.penalty)
        if not 0.0 <= support <= 1.0:
            raise ValueError("evidence support must be between 0 and 1")
        if not 0.0 <= penalty <= 1.0:
            raise ValueError("evidence penalty must be between 0 and 1")
        if (
            self.source_token_start is not None
            and self.source_token_end is not None
            and self.source_token_end < self.source_token_start
        ):
            raise ValueError("evidence source span is reversed")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "penalty", penalty)
        object.__setattr__(self, "value", _freeze(self.value))

    @property
    def dedupe_key(self) -> tuple[Any, ...]:
        return (
            self.origin,
            self.scope_type,
            self.scope_id,
            self.target_hypothesis_id,
            self.evidence_type,
            self.source_token_start,
            self.source_token_end,
            self.parser_version,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "origin": self.origin,
            "target_hypothesis_id": self.target_hypothesis_id,
            "value": _value(self.value),
            "support": float(self.support),
            "penalty": float(self.penalty),
            "evidence_type": self.evidence_type,
            "independent_group": self.independent_group,
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "source_token_start": self.source_token_start,
            "source_token_end": self.source_token_end,
            "source_object_id": self.source_object_id,
            "parser_version": self.parser_version,
        }


@dataclass
class InterpretationHypothesis:
    id: str
    scope_type: str
    scope_id: str
    hypothesis_type: str
    value: Any
    status: HypothesisStatus = HypothesisStatus.EPHEMERAL
    support_by_group: Dict[str, float] = field(default_factory=dict)
    penalties: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_slots: List[str] = field(default_factory=list)
    stability_cycles: int = 0
    leader_margin: float = 0.0
    selected: bool = False
    parser_version: str = INTERPRETATION_VERSION
    support: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "hypothesis_type": self.hypothesis_type,
            "value": _value(self.value),
            "status": _value(self.status),
            "support_by_group": dict(self.support_by_group),
            "support": float(self.support),
            "penalties": list(self.penalties),
            "constraints": list(self.constraints),
            "unresolved_slots": list(self.unresolved_slots),
            "stability_cycles": int(self.stability_cycles),
            "leader_margin": float(self.leader_margin),
            "selected": bool(self.selected),
            "parser_version": self.parser_version,
        }


@dataclass
class DialogueAct:
    id: str
    utterance_id: str
    act_type: DialogueActType
    token_start: int
    token_end: int
    target_act_id: Optional[str] = None
    addressee: Optional[str] = None
    confidence: float = 0.5
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "utterance_id": self.utterance_id,
            "act_type": _value(self.act_type),
            "token_start": int(self.token_start),
            "token_end": int(self.token_end),
            "target_act_id": self.target_act_id,
            "addressee": self.addressee,
            "confidence": float(self.confidence),
            "evidence": list(self.evidence),
            "alternatives": list(self.alternatives),
        }


@dataclass
class Clause:
    id: str
    utterance_id: str
    sentence_index: int
    token_start: int
    token_end: int
    parent_clause_id: Optional[str] = None
    clause_type: str = "MAIN"
    relation_to_parent: Optional[ClauseRelationType] = None
    predicate_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    mode: ClauseMode = ClauseMode.ASSERTION
    actuality: Actuality = Actuality.ACTUAL
    evidence_status: EvidenceStatus = EvidenceStatus.STATED
    polarity: Polarity = Polarity.POSITIVE
    negation_scope: Optional[Dict[str, Any]] = None
    modality: Optional[Modality] = None
    completion_status: CompletionStatus = CompletionStatus.UNKNOWN
    temporal_anchor: Optional[Dict[str, Any]] = None
    speaker: str = "user"
    quoted_speaker: Optional[str] = None
    surface: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    alternative_boundaries: List[Dict[str, Any]] = field(default_factory=list)
    participants: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "utterance_id": self.utterance_id,
            "sentence_index": int(self.sentence_index),
            "parent_clause_id": self.parent_clause_id,
            "token_start": int(self.token_start),
            "token_end": int(self.token_end),
            "clause_type": self.clause_type,
            "relation_to_parent": _value(self.relation_to_parent),
            "predicate_hypotheses": list(self.predicate_hypotheses),
            "mode": _value(self.mode),
            "actuality": _value(self.actuality),
            "evidence_status": _value(self.evidence_status),
            "polarity": _value(self.polarity),
            "negation_scope": dict(self.negation_scope) if self.negation_scope else None,
            "modality": _value(self.modality),
            "completion_status": _value(self.completion_status),
            "temporal_anchor": dict(self.temporal_anchor) if self.temporal_anchor else None,
            "speaker": self.speaker,
            "quoted_speaker": self.quoted_speaker,
            "surface": self.surface,
            "evidence": list(self.evidence),
            "alternative_boundaries": list(self.alternative_boundaries),
            "participants": list(self.participants),
        }


@dataclass
class ClauseRelation:
    id: str
    source_clause_id: str
    target_clause_id: str
    relation_type: ClauseRelationType
    confidence: float
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_clause_id": self.source_clause_id,
            "target_clause_id": self.target_clause_id,
            "relation_type": _value(self.relation_type),
            "confidence": float(self.confidence),
            "evidence": list(self.evidence),
        }


@dataclass
class UtteranceEnvelope:
    id: str
    conversation_id: str
    turn_index: int
    speaker_role: str
    raw_text: str
    normalized_text: str
    received_at: str
    language: str = "ru"
    source_type: str = "dialogue"
    parser_version: str = INTERPRETATION_VERSION
    interpretation_status: InterpretationStatus = InterpretationStatus.INCOMPLETE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "turn_index": int(self.turn_index),
            "speaker_role": self.speaker_role,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "received_at": self.received_at,
            "language": self.language,
            "source_type": self.source_type,
            "parser_version": self.parser_version,
            "interpretation_status": _value(self.interpretation_status),
        }


@dataclass
class SpeakerCommitment:
    id: str
    conversation_id: str
    speaker_role: str
    source_utterance_id: str
    source_clause_id: str
    interpretation_id: str
    status: CommitmentStatus
    supersedes_commitment_id: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    content: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "speaker_role": self.speaker_role,
            "source_utterance_id": self.source_utterance_id,
            "source_clause_id": self.source_clause_id,
            "interpretation_id": self.interpretation_id,
            "status": _value(self.status),
            "supersedes_commitment_id": self.supersedes_commitment_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "content": dict(self.content),
        }


@dataclass
class DialogueState:
    conversation_id: str
    active_topic: Optional[Dict[str, Any]] = None
    focus_stack: List[Dict[str, Any]] = field(default_factory=list)
    entity_candidates: List[Dict[str, Any]] = field(default_factory=list)
    pending_questions: List[Dict[str, Any]] = field(default_factory=list)
    expected_response: Optional[Dict[str, Any]] = None
    speaker_commitments: List[Dict[str, Any]] = field(default_factory=list)
    shared_confirmations: List[Dict[str, Any]] = field(default_factory=list)
    temporal_anchor: Optional[Dict[str, Any]] = None
    active_world: str = "actual"
    active_quote_source: Optional[str] = None
    exclusions: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    unresolved_references: List[Dict[str, Any]] = field(default_factory=list)
    last_query_frame: Optional[Dict[str, Any]] = None
    last_answer: Optional[Dict[str, Any]] = None
    topic_history: List[Dict[str, Any]] = field(default_factory=list)
    pending_clarification: Optional[Dict[str, Any]] = None
    version: str = INTERPRETATION_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "active_topic": (
                dict(self.active_topic) if self.active_topic else None
            ),
            "focus_stack": list(self.focus_stack),
            "entity_candidates": list(self.entity_candidates),
            "pending_questions": list(self.pending_questions),
            "expected_response": (
                dict(self.expected_response) if self.expected_response else None
            ),
            "speaker_commitments": list(self.speaker_commitments),
            "shared_confirmations": list(self.shared_confirmations),
            "temporal_anchor": (
                dict(self.temporal_anchor) if self.temporal_anchor else None
            ),
            "active_world": self.active_world,
            "active_quote_source": self.active_quote_source,
            "exclusions": {
                key: list(value) for key, value in self.exclusions.items()
            },
            "unresolved_references": list(self.unresolved_references),
            "last_query_frame": (
                dict(self.last_query_frame) if self.last_query_frame else None
            ),
            "last_answer": dict(self.last_answer) if self.last_answer else None,
            "topic_history": list(self.topic_history),
            "pending_clarification": (
                dict(self.pending_clarification)
                if self.pending_clarification else None
            ),
            "version": self.version,
        }


@dataclass
class ResponsePlan:
    response_type: ResponseType
    target_act_id: Optional[str] = None
    focus_role: Optional[str] = None
    content_slots: Dict[str, Any] = field(default_factory=dict)
    source_evidence: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty: Optional[Dict[str, Any]] = None
    attribution: Optional[Dict[str, Any]] = None
    surface_constraints: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "response_type": _value(self.response_type),
            "target_act_id": self.target_act_id,
            "focus_role": self.focus_role,
            "content_slots": dict(self.content_slots),
            "source_evidence": list(self.source_evidence),
            "uncertainty": dict(self.uncertainty) if self.uncertainty else None,
            "attribution": dict(self.attribution) if self.attribution else None,
            "surface_constraints": dict(self.surface_constraints),
        }


@dataclass
class MorphAnalysis:
    lemma: str
    pos: str
    features: Dict[str, Any]
    confidence: float
    selected: bool = False
    evidence: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lemma": self.lemma,
            "pos": self.pos,
            **self.features,
            "confidence": self.confidence,
            "selected": self.selected,
            "evidence": list(self.evidence),
        }


@dataclass
class ParsedToken:
    index: int
    surface: str
    normalized: str
    lemma: str
    pos: str
    features: Dict[str, Any]
    lexeme_cloud_id: Optional[int] = None
    word_form_cloud_id: Optional[int] = None
    parser_annotation: str = "unknown"
    analyses: List[MorphAnalysis] = field(default_factory=list)

    @property
    def grammatical_case(self) -> Optional[str]:
        return self.features.get("case")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "surface": self.surface,
            "normalized": self.normalized,
            "lemma": self.lemma,
            "part_of_speech": self.pos,
            "grammatical_features": dict(self.features),
            "lexeme_cloud_id": self.lexeme_cloud_id,
            "word_form_cloud_id": self.word_form_cloud_id,
            "parser_annotation": self.parser_annotation,
            "morphological_analyses": [
                analysis.as_dict() for analysis in self.analyses
            ],
        }


@dataclass
class Phrase:
    id: str
    phrase_type: str
    token_start: int
    token_end: int
    head_token_index: int
    token_indices: List[int]
    surface: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.phrase_type,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "head_token_index": self.head_token_index,
            "tokens": list(self.token_indices),
            "surface": self.surface,
            **self.metadata,
        }


@dataclass
class PhraseGraph:
    phrases: List[Phrase]
    dependencies: List[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phrases": [phrase.as_dict() for phrase in self.phrases],
            "dependencies": list(self.dependencies),
        }


@dataclass
class QuestionOperator:
    operator_type: str
    surface: str
    token_indices: List[int]
    question_lemma: str
    grammatical_features: Dict[str, Any]
    type_constraint_token_index: Optional[int] = None
    compatible_slot_hypotheses: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "operator_type": self.operator_type,
            "surface": self.surface,
            "token_indices": list(self.token_indices),
            "question_lemma": self.question_lemma,
            "grammatical_features": dict(self.grammatical_features),
            "type_constraint_token_index": self.type_constraint_token_index,
            "compatible_slot_hypotheses": list(self.compatible_slot_hypotheses),
        }


@dataclass
class LanguageAnalysis:
    tokens: List[ParsedToken]
    mentions: List[Any]
    phrase_graph: PhraseGraph
    predicate: Optional[ParsedToken]
    question_operator: Optional[QuestionOperator]
    relation_phrases: List[Dict[str, Any]]
    diagnostics: List[Dict[str, Any]]
    utterance: Optional[UtteranceEnvelope] = None
    dialogue_acts: List[DialogueAct] = field(default_factory=list)
    clauses: List[Clause] = field(default_factory=list)
    clause_relations: List[ClauseRelation] = field(default_factory=list)
    hypotheses: List[InterpretationHypothesis] = field(default_factory=list)
    evidence_packets: List[EvidencePacket] = field(default_factory=list)
    interpretation_status: InterpretationStatus = InterpretationStatus.INCOMPLETE
    interpretation_trace: Dict[str, Any] = field(default_factory=dict)
    interpretation_version: str = INTERPRETATION_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tokens": [token.as_dict() for token in self.tokens],
            "entity_mentions": [
                mention.as_dict(self.tokens) for mention in self.mentions
            ],
            "phrase_graph": self.phrase_graph.as_dict(),
            "predicate": self.predicate.as_dict() if self.predicate else None,
            "question_operator": (
                self.question_operator.as_dict()
                if self.question_operator
                else None
            ),
            "relation_phrases": list(self.relation_phrases),
            "diagnostics": list(self.diagnostics),
            "utterance": self.utterance.as_dict() if self.utterance else None,
            "dialogue_acts": [act.as_dict() for act in self.dialogue_acts],
            "clauses": [clause.as_dict() for clause in self.clauses],
            "clause_relations": [
                relation.as_dict() for relation in self.clause_relations
            ],
            "interpretation_hypotheses": [
                hypothesis.as_dict() for hypothesis in self.hypotheses
            ],
            "interpretation_evidence": [
                packet.as_dict() for packet in self.evidence_packets
            ],
            "interpretation_status": _value(self.interpretation_status),
            "interpretation_trace": dict(self.interpretation_trace),
            "interpretation_version": self.interpretation_version,
        }
