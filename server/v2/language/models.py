"""Data structures for the shared phrase-aware language analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    grammatical_role: str = "unknown"
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
            "scene_role": self.grammatical_role,
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
    requested_slot_hypotheses: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "operator_type": self.operator_type,
            "surface": self.surface,
            "token_indices": list(self.token_indices),
            "question_lemma": self.question_lemma,
            "grammatical_features": dict(self.grammatical_features),
            "type_constraint_token_index": self.type_constraint_token_index,
            "requested_slot_hypotheses": list(self.requested_slot_hypotheses),
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
        }
