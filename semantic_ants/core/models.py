from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConceptNode:
    """Узел смыслового графа."""

    uri: str
    label: str
    language: str
    source: str = "local"
    layer: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "label": self.label,
            "language": self.language,
            "source": self.source,
            "layer": self.layer,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SemanticEdge:
    """Направленная смысловая связь."""

    start: str
    end: str
    relation: str
    weight: float = 1.0
    source: str = "local"
    surface_text: str | None = None
    layer: int = 1
    distance: float = 1.0
    edge_type: str = "semantic"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return edge_key(self.start, self.relation, self.end)

    def reversed(self, relation_prefix: str = "Reverse") -> "SemanticEdge":
        return SemanticEdge(
            start=self.end,
            end=self.start,
            relation=f"{relation_prefix}:{self.relation}",
            weight=max(self.weight * 0.7, 0.01),
            source=self.source,
            surface_text=self.surface_text,
            layer=self.layer,
            distance=self.distance,
            edge_type=self.edge_type,
            metadata={**self.metadata, "reversed": True},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "relation": self.relation,
            "weight": self.weight,
            "source": self.source,
            "surface_text": self.surface_text,
            "layer": self.layer,
            "distance": self.distance,
            "edge_type": self.edge_type,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AntStep:
    """Один переход муравья по графу."""

    start: str
    end: str
    relation: str
    edge_weight: float
    pheromone: float
    score: float
    source: str = "local"
    layer: int = 1
    distance: float = 1.0
    remaining_strength: int | None = None
    edge_type: str = "semantic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "relation": self.relation,
            "edge_weight": self.edge_weight,
            "pheromone": self.pheromone,
            "score": self.score,
            "source": self.source,
            "layer": self.layer,
            "distance": self.distance,
            "remaining_strength": self.remaining_strength,
            "edge_type": self.edge_type,
        }


@dataclass(frozen=True)
class AntRoute:
    """Маршрут муравья от слова к цепочке смыслов."""

    ant_id: int
    start: str
    steps: list[AntStep]
    total_score: float

    @property
    def concepts(self) -> list[str]:
        values = [self.start]
        values.extend(step.end for step in self.steps)
        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "ant_id": self.ant_id,
            "start": self.start,
            "concepts": self.concepts,
            "total_score": self.total_score,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class SemanticResult:
    """Результат анализа фразы."""

    result_id: str
    input_text: str
    lang: str
    tokens: list[str]
    activated_concepts: list[dict[str, Any]]
    routes: list[AntRoute]
    summary: str
    response: str
    sources: list[str]
    session_id: str | None = None
    context_turns: list[dict[str, Any]] = field(default_factory=list)
    semantic_vector: dict[str, Any] = field(default_factory=dict)
    signal_trace: list[dict[str, Any]] = field(default_factory=list)
    response_source: str = ""
    response_lang: str = ""
    response_candidates: list[str] = field(default_factory=list)
    canonical_concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "input_text": self.input_text,
            "lang": self.lang,
            "tokens": self.tokens,
            "activated_concepts": self.activated_concepts,
            "routes": [route.to_dict() for route in self.routes],
            "summary": self.summary,
            "response": self.response,
            "sources": self.sources,
            "session_id": self.session_id,
            "context_turns": self.context_turns,
            "semantic_vector": self.semantic_vector,
            "signal_trace": self.signal_trace,
            "response_source": self.response_source,
            "response_lang": self.response_lang,
            "response_candidates": self.response_candidates,
            "canonical_concepts": self.canonical_concepts,
        }


def edge_key(start: str, relation: str, end: str) -> str:
    return f"{start}|{relation}|{end}"


def split_edge_key(value: str) -> tuple[str, str, str]:
    parts = value.split("|", 2)
    if len(parts) != 3:
        raise ValueError(f"Некорректный ключ ребра: {value}")
    return parts[0], parts[1], parts[2]
