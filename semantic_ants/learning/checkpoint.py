from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semantic_ants.core.models import SemanticEdge, edge_key, split_edge_key
from semantic_ants.core.normalization import detect_language, text_to_concept_uri, tokenize
from semantic_ants.learning.canonical import CanonicalResolver, canonical_concept_uri


DEFAULT_CHECKPOINT_NAME = "model.bin"


def default_checkpoint_path(state_dir: Path | str = ".semantic_ants") -> Path:
    return Path(state_dir) / "checkpoints" / DEFAULT_CHECKPOINT_NAME


@dataclass
class Checkpoint:
    """Обучаемый слой поверх внешнего графа знаний."""

    version: int = 5
    pheromones: dict[str, float] = field(default_factory=dict)
    concept_pheromones: dict[str, float] = field(default_factory=dict)
    suppressed_concepts: dict[str, float] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    canonical_concepts: dict[str, dict[str, Any]] = field(default_factory=dict)
    concept_redirects: dict[str, str] = field(default_factory=dict)
    surface_forms: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    custom_edges: list[dict[str, Any]] = field(default_factory=list)
    response_memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    negative_memory: list[dict[str, Any]] = field(default_factory=list)
    experiences: list[dict[str, Any]] = field(default_factory=list)
    learned_bridges: list[dict[str, Any]] = field(default_factory=list)
    accepted_answers: list[dict[str, Any]] = field(default_factory=list)
    route_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    chat_sessions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    mini_generator: dict[str, Any] = field(default_factory=dict)
    last_result_id: str | None = None
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    examples_seen: int = 0
    seed: int = 42

    @property
    def canonical_resolver(self) -> CanonicalResolver:
        return CanonicalResolver(self)

    def canonical_uri(self, concept_uri: str) -> str:
        return self.canonical_resolver.canonical_uri(concept_uri)

    def register_canonical_concept(
        self,
        concept_uri: str,
        *,
        label: str | None = None,
        aliases: list[str] | None = None,
        lang: str | None = None,
        source_uri: str | None = None,
        quality: float | None = None,
    ) -> str:
        return self.canonical_resolver.register_concept(
            concept_uri,
            label=label,
            aliases=aliases,
            lang=lang,
            source_uri=source_uri,
            quality=quality,
        )

    def register_surface_form(self, concept_uri: str, surface: str, lang: str | None = None) -> None:
        self.canonical_resolver.register_surface(concept_uri, surface, lang=lang)

    def pheromone_for(self, edge: SemanticEdge) -> float:
        canonical_key = edge_key(self.canonical_uri(edge.start), edge.relation, self.canonical_uri(edge.end))
        return max(float(self.pheromones.get(canonical_key, self.pheromones.get(edge.key, 1.0))), 0.01)

    def concept_pheromone_for(self, concept_uri: str) -> float:
        concept_uri = self.canonical_uri(concept_uri)
        return max(float(self.concept_pheromones.get(concept_uri, 1.0)), 0.01)

    def penalty_for(self, concept_uri: str) -> float:
        concept_uri = self.canonical_uri(concept_uri)
        penalty = max(float(self.suppressed_concepts.get(concept_uri, 0.0)), 0.0)
        return 1.0 / (1.0 + penalty)

    def reinforce_edge(self, start: str, relation: str, end: str, amount: float = 0.2) -> None:
        start = self.canonical_uri(start)
        end = self.canonical_uri(end)
        key = edge_key(start, relation, end)
        self.pheromones[key] = min(float(self.pheromones.get(key, 1.0)) + amount, 25.0)
        stat = self.route_stats.setdefault(key, {"positive": 0, "negative": 0})
        stat["positive"] = int(stat.get("positive", 0)) + 1
        self.reinforce_concept(start, amount=amount * 0.25)
        self.reinforce_concept(end, amount=amount * 0.5)

    def penalize_edge(self, start: str, relation: str, end: str, amount: float = 0.2) -> None:
        start = self.canonical_uri(start)
        end = self.canonical_uri(end)
        key = edge_key(start, relation, end)
        self.pheromones[key] = max(float(self.pheromones.get(key, 1.0)) - amount, 0.05)
        stat = self.route_stats.setdefault(key, {"positive": 0, "negative": 0})
        stat["negative"] = int(stat.get("negative", 0)) + 1
        self.penalize_concept(end, amount=amount * 0.5)

    def reinforce_concept(self, concept_uri: str, amount: float = 0.2) -> None:
        if not concept_uri:
            return
        concept_uri = self.canonical_uri(concept_uri)
        self.concept_pheromones[concept_uri] = min(
            float(self.concept_pheromones.get(concept_uri, 1.0)) + amount,
            25.0,
        )
        if concept_uri in self.suppressed_concepts:
            self.suppressed_concepts[concept_uri] = max(
                float(self.suppressed_concepts.get(concept_uri, 0.0)) - amount,
                0.0,
            )

    def penalize_concept(self, concept_uri: str, amount: float = 0.2) -> None:
        if not concept_uri:
            return
        concept_uri = self.canonical_uri(concept_uri)
        self.concept_pheromones[concept_uri] = max(
            float(self.concept_pheromones.get(concept_uri, 1.0)) - amount,
            0.05,
        )

    def suppress_concept(self, concept_uri: str, amount: float = 0.5) -> None:
        concept_uri = self.canonical_uri(concept_uri)
        self.suppressed_concepts[concept_uri] = min(
            float(self.suppressed_concepts.get(concept_uri, 0.0)) + amount,
            25.0,
        )
        self.penalize_concept(concept_uri, amount=amount * 0.25)

    def remember_response(
        self,
        concepts: list[str],
        response: str,
        amount: float = 1.0,
        lang: str | None = None,
        source_lang: str | None = None,
    ) -> None:
        if not concepts or not response:
            return
        clean_response = " ".join(response.split())
        if not clean_response:
            return
        concepts = [self.canonical_uri(concept) for concept in concepts if concept]
        key = concept_set_key(concepts)
        response_lang = lang if lang in {"ru", "en"} else _detect_lang_from_concepts(concepts) or detect_language(clean_response)
        item = self.response_memory.setdefault(
            key,
            {
                "concepts": list(dict.fromkeys(concepts)),
                "answer_concepts": [],
                "lang": response_lang,
                "source_lang": source_lang or _detect_lang_from_concepts(concepts) or detect_language(clean_response),
                "answer": "",
                "response": "",
                "weight": 0.0,
            },
        )
        item["concepts"] = list(dict.fromkeys(concepts))
        item["answer_concepts"] = _text_to_concepts(clean_response, lang=response_lang, checkpoint=self)
        item["lang"] = response_lang
        item["source_lang"] = source_lang or _detect_lang_from_concepts(concepts) or detect_language(clean_response)
        item["answer"] = clean_response
        item["response"] = clean_response
        item["weight"] = float(item.get("weight", 0.0)) + amount
        for concept in concepts:
            self.register_canonical_concept(concept, aliases=[concept.split("/")[-1]], lang=response_lang, source_uri=concept)
        for concept in item["answer_concepts"]:
            self.register_canonical_concept(concept, source_uri=concept, lang=response_lang)

    def remember_concept_label(self, concept_uri: str, label: str) -> None:
        clean_label = " ".join(label.split())
        if not concept_uri or not clean_label:
            return
        concept_uri = self.canonical_uri(concept_uri)
        self.register_canonical_concept(concept_uri, label=clean_label)
        labels = self.metadata.setdefault("concept_labels", {})
        if isinstance(labels, dict):
            labels[concept_uri] = clean_label
        definitions = self.metadata.setdefault("concept_definitions", {})
        if isinstance(definitions, dict):
            raw = definitions.get(concept_uri, {})
            info = dict(raw) if isinstance(raw, dict) else {}
            info["label"] = clean_label
            definitions[concept_uri] = info

    def remember_accepted_answer(
        self,
        stimulus: str,
        semantic_prompt: str,
        concepts: list[str],
        answer: str,
        reward: float = 1.0,
        limit: int = 500,
        lang: str | None = None,
        source_lang: str | None = None,
    ) -> dict[str, Any] | None:
        if not answer:
            return None
        clean_answer = " ".join(answer.split())
        if not clean_answer:
            return None
        concepts = [self.canonical_uri(concept) for concept in concepts if concept]
        answer_lang = lang if lang in {"ru", "en"} else _detect_lang_from_concepts(concepts) or detect_language(answer)
        source_language = source_lang or _detect_lang_from_concepts(concepts) or detect_language(stimulus or semantic_prompt or answer)
        answer_concepts = _text_to_concepts(clean_answer, lang=answer_lang, checkpoint=self)
        item = {
            "stimulus": stimulus,
            "semantic_prompt": semantic_prompt,
            "concepts": list(dict.fromkeys(concepts)),
            "answer_concepts": answer_concepts,
            "answer": clean_answer,
            "lang": answer_lang,
            "source_lang": source_language,
            "reward": float(reward),
            "created_at": time.time(),
        }
        for concept in concepts:
            self.register_canonical_concept(concept, source_uri=concept, lang=source_language)
        for concept in answer_concepts:
            self.register_canonical_concept(concept, source_uri=concept, lang=answer_lang)
        for existing in self.accepted_answers:
            if (
                str(existing.get("stimulus", "")).strip().lower() == stimulus.strip().lower()
                and " ".join(str(existing.get("answer", "")).split()) == clean_answer
            ):
                existing["semantic_prompt"] = semantic_prompt
                existing["concepts"] = item["concepts"]
                existing["answer_concepts"] = answer_concepts
                existing["answer"] = clean_answer
                existing["lang"] = answer_lang
                existing["source_lang"] = source_language
                existing["reward"] = max(float(existing.get("reward", 0.0)), float(reward))
                existing["created_at"] = time.time()
                self.accepted_answers.sort(key=lambda value: float(value.get("reward", 0.0)), reverse=True)
                self.remember_response(
                    item["concepts"],
                    clean_answer,
                    amount=max(float(reward), 0.1),
                    lang=answer_lang,
                    source_lang=source_language,
                )
                return existing
        self.accepted_answers.append(item)
        self.accepted_answers.sort(key=lambda value: float(value.get("reward", 0.0)), reverse=True)
        del self.accepted_answers[limit:]
        self.remember_response(
            item["concepts"],
            answer,
            amount=max(float(reward), 0.1),
            lang=answer_lang,
            source_lang=source_language,
        )
        return item

    def remember_negative(
        self,
        stimulus: str,
        semantic_prompt: str,
        concepts: list[str],
        answer: str,
        reason: str = "",
        limit: int = 500,
        lang: str | None = None,
        source_lang: str | None = None,
    ) -> dict[str, Any]:
        concepts = [self.canonical_uri(concept) for concept in concepts if concept]
        response_lang = lang if lang in {"ru", "en"} else _detect_lang_from_concepts(concepts) or detect_language(answer or stimulus or semantic_prompt)
        source_language = source_lang or _detect_lang_from_concepts(concepts) or detect_language(stimulus or semantic_prompt or answer)
        clean_answer = " ".join(answer.split())
        item = {
            "stimulus": stimulus,
            "semantic_prompt": semantic_prompt,
            "concepts": list(dict.fromkeys(concepts)),
            "answer_concepts": _text_to_concepts(clean_answer, lang=response_lang, checkpoint=self) if clean_answer else [],
            "answer": clean_answer,
            "lang": response_lang,
            "source_lang": source_language,
            "reason": reason,
            "created_at": time.time(),
        }
        for concept in concepts:
            self.register_canonical_concept(concept, source_uri=concept, lang=source_language)
        for concept in item["answer_concepts"]:
            self.register_canonical_concept(concept, source_uri=concept, lang=response_lang)
        self.negative_memory.append(item)
        del self.negative_memory[:-limit]
        return item

    def remember_experience(self, experience: dict[str, Any], limit: int = 2000) -> None:
        self.experiences.append({**experience, "stored_at": time.time()})
        del self.experiences[:-limit]

    def remember_result(self, result: dict[str, Any], limit: int = 50) -> None:
        result_id = str(result["result_id"])
        self.last_result_id = result_id
        self.results[result_id] = result
        while len(self.results) > limit:
            oldest = next(iter(self.results))
            del self.results[oldest]

    def session_history(self, session_id: str | None, limit: int = 8) -> list[dict[str, Any]]:
        if not session_id:
            return []
        history = self.chat_sessions.get(session_id, [])
        return [dict(item) for item in history[-limit:]]

    def reset_chat_session(self, session_id: str | None) -> None:
        if not session_id:
            return
        self.chat_sessions[session_id] = []

    def remember_chat_turn(
        self,
        session_id: str | None,
        role: str,
        text: str,
        result_id: str,
        concepts: list[str] | None = None,
        limit: int = 80,
    ) -> None:
        if not session_id or not text:
            return
        turns = self.chat_sessions.setdefault(session_id, [])
        turns.append(
            {
                "role": role,
                "text": text,
                "result_id": result_id,
                "concepts": list(dict.fromkeys(self.canonical_uri(concept) for concept in (concepts or []) if concept)),
                "created_at": time.time(),
            }
        )
        del turns[:-limit]

    def learned_edges(self) -> list[SemanticEdge]:
        edges: list[SemanticEdge] = []
        seen: set[str] = set()
        for raw in [*self.custom_edges, *self.learned_bridges]:
            try:
                edge = SemanticEdge(
                    start=self.canonical_uri(str(raw["start"])),
                    end=self.canonical_uri(str(raw["end"])),
                    relation=str(raw.get("relation", "LearnedRelatedTo")),
                    weight=float(raw.get("weight", 1.0)),
                    source="learned",
                    surface_text=raw.get("surface_text"),
                    layer=int(raw.get("layer", 1)),
                    distance=float(raw.get("distance", 1.0)),
                    edge_type=str(raw.get("edge_type", "semantic")),
                    metadata=dict(raw.get("metadata", {})),
                )
                if edge.key in seen:
                    continue
                seen.add(edge.key)
                edges.append(edge)
            except (KeyError, TypeError, ValueError):
                continue
        return edges

    def add_custom_edge(
        self,
        start: str,
        end: str,
        relation: str = "LearnedRelatedTo",
        weight: float = 1.0,
        layer: int = 1,
        distance: float = 1.0,
        edge_type: str = "semantic",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        start = self.canonical_uri(start)
        end = self.canonical_uri(end)
        self.register_canonical_concept(start, source_uri=start)
        self.register_canonical_concept(end, source_uri=end)
        candidate = {
            "start": start,
            "end": end,
            "relation": relation,
            "weight": weight,
            "layer": layer,
            "distance": distance,
            "edge_type": edge_type,
        }
        if metadata:
            candidate["metadata"] = dict(metadata)
        for existing in self.custom_edges:
            if (
                existing.get("start") == start
                and existing.get("end") == end
                and existing.get("relation") == relation
            ):
                existing["weight"] = max(float(existing.get("weight", 1.0)), weight)
                existing.setdefault("layer", layer)
                existing.setdefault("distance", distance)
                existing.setdefault("edge_type", edge_type)
                if metadata:
                    existing["metadata"] = {**dict(existing.get("metadata", {})), **dict(metadata)}
                return
        self.custom_edges.append(candidate)

    def add_learned_bridge(
        self,
        start: str,
        relation: str,
        end: str,
        weight: float = 0.3,
        confirmed: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        start = self.canonical_uri(start)
        end = self.canonical_uri(end)
        self.register_canonical_concept(start, source_uri=start)
        self.register_canonical_concept(end, source_uri=end)
        bridge = {
            "start": start,
            "relation": relation,
            "end": end,
            "weight": weight,
            "confirmed": confirmed,
            "layer": int((metadata or {}).get("layer", 1)),
            "distance": float((metadata or {}).get("distance", 1.0)),
            "edge_type": str((metadata or {}).get("edge_type", "semantic")),
            "metadata": dict(metadata or {}),
        }
        for existing in self.learned_bridges:
            if (
                existing.get("start") == start
                and existing.get("relation") == relation
                and existing.get("end") == end
            ):
                if existing.get("confirmed"):
                    return False
                existing["weight"] = max(float(existing.get("weight", 0.0)), weight)
                existing["confirmed"] = bool(existing.get("confirmed")) or confirmed
                existing["metadata"] = {**dict(existing.get("metadata", {})), **bridge["metadata"]}
                return False
        self.learned_bridges.append(bridge)
        self.add_custom_edge(
            start,
            end,
            relation=relation,
            weight=weight,
            layer=int(bridge.get("layer", 1)),
            distance=float(bridge.get("distance", 1.0)),
            edge_type=str(bridge.get("edge_type", "semantic")),
            metadata=dict(metadata or {}),
        )
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "pheromones": self.pheromones,
            "concept_pheromones": self.concept_pheromones,
            "suppressed_concepts": self.suppressed_concepts,
            "aliases": self.aliases,
            "canonical_concepts": self.canonical_concepts,
            "concept_redirects": self.concept_redirects,
            "surface_forms": self.surface_forms,
            "custom_edges": self.custom_edges,
            "response_memory": self.response_memory,
            "negative_memory": self.negative_memory,
            "experiences": self.experiences,
            "learned_bridges": self.learned_bridges,
            "accepted_answers": self.accepted_answers,
            "speech_patterns": self.accepted_answers,
            "route_stats": self.route_stats,
            "chat_sessions": self.chat_sessions,
            "metadata": self.metadata,
            "mini_generator": self.mini_generator,
            "last_result_id": self.last_result_id,
            "results": self.results,
            "examples_seen": self.examples_seen,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        accepted_answers = _normalize_patterns(
            data.get("speech_patterns", data.get("accepted_answers", [])),
            fallback_lang=str(data.get("lang", "auto")),
            checkpoint_data=data,
        )
        response_memory = _normalize_response_memory(data.get("response_memory", {}), checkpoint_data=data)
        negative_memory = _normalize_negative_memory(data.get("negative_memory", []), checkpoint_data=data)
        checkpoint = cls(
            version=max(int(data.get("version", 1)), 5),
            pheromones=dict(data.get("pheromones", {})),
            concept_pheromones=dict(data.get("concept_pheromones", {})),
            suppressed_concepts=dict(data.get("suppressed_concepts", {})),
            aliases=dict(data.get("aliases", {})),
            canonical_concepts=dict(data.get("canonical_concepts", {})),
            concept_redirects=dict(data.get("concept_redirects", {})),
            surface_forms=dict(data.get("surface_forms", {})),
            custom_edges=list(data.get("custom_edges", [])),
            response_memory=response_memory,
            negative_memory=negative_memory,
            experiences=list(data.get("experiences", [])),
            learned_bridges=list(data.get("learned_bridges", [])),
            accepted_answers=accepted_answers,
            route_stats=dict(data.get("route_stats", {})),
            chat_sessions={
                str(key): list(value)
                for key, value in dict(data.get("chat_sessions", {})).items()
                if isinstance(value, list)
            },
            metadata=dict(data.get("metadata", {})),
            mini_generator=dict(data.get("mini_generator", {})),
            last_result_id=data.get("last_result_id"),
            results=dict(data.get("results", {})),
            examples_seen=int(data.get("examples_seen", 0)),
            seed=int(data.get("seed", 42)),
        )
        if int(data.get("version", 1)) < 5 or not checkpoint.canonical_concepts:
            migrate_checkpoint(checkpoint)
        return checkpoint


class CheckpointStore:
    """Читает и сохраняет checkpoint в быстром бинарном формате."""

    def __init__(self, path: Path | str = default_checkpoint_path()) -> None:
        self.path = Path(path)
        self.loaded_path: Path | None = None

    def load(self) -> Checkpoint:
        self.loaded_path = None
        if not self.path.exists():
            return Checkpoint()
        self.loaded_path = self.path
        return self._load_path(self.path)

    def save(self, checkpoint: Checkpoint) -> None:
        self._save_path(self.path, checkpoint)
        self.loaded_path = self.path

    def export(self, destination: Path | str) -> None:
        target = Path(destination)
        checkpoint = self.load()
        self._save_path(target, checkpoint)

    def _load_path(self, path: Path) -> Checkpoint:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, Checkpoint):
            return _upgrade_checkpoint_object(payload)
        if isinstance(payload, dict):
            return Checkpoint.from_dict(payload)
        raise ValueError(f"Некорректный checkpoint: {path}")

    def _save_path(self, path: Path, checkpoint: Checkpoint) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        with tmp.open("wb") as handle:
            pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _replace_with_retry(tmp, path)


def concept_set_key(concepts: list[str]) -> str:
    return "|".join(sorted(dict.fromkeys(concepts)))


def _text_to_concepts(text: str, lang: str | None, checkpoint: Checkpoint, limit: int = 8) -> list[str]:
    selected_lang = lang if lang in {"ru", "en"} else detect_language(text)
    stop_words = set(_common_words(checkpoint, selected_lang))
    concepts: list[str] = []
    for token in tokenize(text):
        if token in stop_words:
            continue
        uri = checkpoint.aliases.get(token) or _token_to_uri(token, selected_lang)
        if uri not in concepts:
            canonical = checkpoint.canonical_uri(uri) if uri else uri
            if canonical:
                checkpoint.register_surface_form(canonical, token, lang=selected_lang)
                checkpoint.register_canonical_concept(canonical, aliases=[token], lang=selected_lang, source_uri=uri)
            concepts.append(canonical)
        if len(concepts) >= limit:
            break
    return concepts


def _token_to_uri(token: str, lang: str) -> str:
    try:
        return text_to_concept_uri(token, lang=lang)
    except ValueError:
        return ""


def _common_words(checkpoint: Checkpoint, lang: str) -> list[str]:
    common_words = checkpoint.metadata.get("common_words", {})
    if isinstance(common_words, dict):
        values = common_words.get(lang, [])
        if isinstance(values, list):
            return [str(value).lower() for value in values]
    return []


def _detect_lang_from_concepts(concepts: list[str]) -> str | None:
    for concept in concepts:
        parts = str(concept).split("/", 3)
        if len(parts) > 2 and parts[1] == "c" and parts[2] in {"ru", "en"}:
            return parts[2]
    return None


def _normalize_lang(
    value: Any,
    *,
    concept_values: list[str] | None = None,
    text_values: tuple[str, ...] = (),
    fallback: str = "auto",
) -> str:
    candidate = str(value or "")
    if candidate in {"ru", "en"}:
        return candidate
    if concept_values:
        detected = _detect_lang_from_concepts(concept_values)
        if detected in {"ru", "en"}:
            return detected
    for text in text_values:
        detected = detect_language(str(text))
        if detected in {"ru", "en"}:
            return detected
    return fallback


def _normalize_patterns(
    values: Any,
    fallback_lang: str,
    checkpoint_data: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    patterns: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        concepts = [str(value) for value in item.get("concepts", []) if value]
        lang_hint = str(item.get("lang") or fallback_lang)
        raw_answer = _clean_answer(item)
        answer_concepts = item.get("answer_concepts")
        if not isinstance(answer_concepts, list) or not answer_concepts:
            answer_concepts = _text_to_concepts(raw_answer, lang=lang_hint, checkpoint=_checkpoint_from_data(checkpoint_data))
        answer_concept_values = [str(value) for value in answer_concepts if value]
        lang = _normalize_lang(
            item.get("lang"),
            concept_values=answer_concept_values,
            text_values=(raw_answer,),
            fallback=fallback_lang if fallback_lang in {"ru", "en"} else "auto",
        )
        source_lang = _normalize_lang(
            item.get("source_lang"),
            concept_values=concepts,
            text_values=(str(item.get("stimulus", "")), str(item.get("semantic_prompt", "")), raw_answer),
            fallback=fallback_lang if fallback_lang in {"ru", "en"} else "auto",
        )
        pattern = {
            "stimulus": str(item.get("stimulus", "")),
            "semantic_prompt": str(item.get("semantic_prompt", "")),
            "concepts": list(dict.fromkeys(CanonicalResolver(None).canonical_uri(value) for value in concepts if value)),
            "answer_concepts": list(dict.fromkeys(CanonicalResolver(None).canonical_uri(value) for value in answer_concept_values if value)),
            "answer": raw_answer,
            "lang": lang,
            "source_lang": source_lang,
            "reward": float(item.get("reward", 0.0)),
            "created_at": float(item.get("created_at", time.time())),
        }
        patterns.append(pattern)
    patterns.sort(key=lambda value: float(value.get("reward", 0.0)), reverse=True)
    return patterns


def _normalize_response_memory(values: Any, checkpoint_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(values, dict):
        return {}
    checkpoint = _checkpoint_from_data(checkpoint_data)
    normalized: dict[str, dict[str, Any]] = {}
    for key, item in values.items():
        if not isinstance(item, dict):
            continue
        concepts = [str(value) for value in str(key).split("|") if value]
        lang_hint = str(item.get("lang") or _detect_lang_from_concepts(concepts) or "auto")
        raw_answer = _clean_answer(item)
        answer_concepts = item.get("answer_concepts")
        if not isinstance(answer_concepts, list) or not answer_concepts:
            answer_concepts = _text_to_concepts(raw_answer, lang=lang_hint, checkpoint=checkpoint)
        answer_concept_values = [str(value) for value in answer_concepts if value]
        lang = _normalize_lang(
            item.get("lang"),
            concept_values=answer_concept_values,
            text_values=(raw_answer,),
            fallback="auto",
        )
        source_lang = _normalize_lang(
            item.get("source_lang"),
            concept_values=concepts,
            text_values=(str(item.get("stimulus", "")), str(item.get("semantic_prompt", "")), raw_answer),
            fallback="auto",
        )
        normalized[str(key)] = {
            "concepts": list(dict.fromkeys(checkpoint.canonical_uri(value) for value in concepts if value)),
            "answer_concepts": list(dict.fromkeys(checkpoint.canonical_uri(value) for value in answer_concept_values if value)),
            "answer": raw_answer,
            "response": raw_answer,
            "lang": lang,
            "source_lang": source_lang,
            "weight": float(item.get("weight", 0.0)),
        }
    return normalized


def _normalize_negative_memory(values: Any, checkpoint_data: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    checkpoint = _checkpoint_from_data(checkpoint_data)
    normalized: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        concepts = [str(value) for value in item.get("concepts", []) if value]
        lang_hint = str(item.get("lang") or _detect_lang_from_concepts(concepts) or "auto")
        raw_answer = _clean_answer(item)
        answer_concepts = item.get("answer_concepts")
        if not isinstance(answer_concepts, list):
            answer_concepts = _text_to_concepts(raw_answer, lang=lang_hint, checkpoint=checkpoint)
        answer_concept_values = [str(value) for value in answer_concepts if value]
        lang = _normalize_lang(
            item.get("lang"),
            concept_values=answer_concept_values,
            text_values=(raw_answer,),
            fallback="auto",
        )
        source_lang = _normalize_lang(
            item.get("source_lang"),
            concept_values=concepts,
            text_values=(str(item.get("stimulus", "")), str(item.get("semantic_prompt", "")), raw_answer),
            fallback="auto",
        )
        normalized.append(
            {
                "stimulus": str(item.get("stimulus", "")),
                "semantic_prompt": str(item.get("semantic_prompt", "")),
                "concepts": list(dict.fromkeys(checkpoint.canonical_uri(value) for value in concepts if value)),
                "answer_concepts": list(dict.fromkeys(checkpoint.canonical_uri(value) for value in answer_concept_values if value)),
                "answer": raw_answer,
                "lang": lang,
                "source_lang": source_lang,
                "reason": str(item.get("reason", "")),
                "created_at": float(item.get("created_at", time.time())),
            }
        )
    return normalized


def _checkpoint_from_data(data: dict[str, Any]) -> Checkpoint:
    return Checkpoint(
        version=max(int(data.get("version", 1)), 5),
        pheromones=dict(data.get("pheromones", {})),
        concept_pheromones=dict(data.get("concept_pheromones", {})),
        suppressed_concepts=dict(data.get("suppressed_concepts", {})),
        aliases=dict(data.get("aliases", {})),
        canonical_concepts=dict(data.get("canonical_concepts", {})),
        concept_redirects=dict(data.get("concept_redirects", {})),
        surface_forms=dict(data.get("surface_forms", {})),
        custom_edges=list(data.get("custom_edges", [])),
        response_memory={},
        negative_memory=[],
        experiences=[],
        learned_bridges=[],
        accepted_answers=[],
        route_stats=dict(data.get("route_stats", {})),
        chat_sessions={},
        metadata=dict(data.get("metadata", {})),
        mini_generator=dict(data.get("mini_generator", {})),
        last_result_id=data.get("last_result_id"),
        results={},
        examples_seen=int(data.get("examples_seen", 0)),
        seed=int(data.get("seed", 42)),
    )


def _clean_answer(item: dict[str, Any]) -> str:
    for key in ("answer", "response", "accepted_answer", "target_response", "expected_answer"):
        value = str(item.get(key, ""))
        clean = " ".join(value.split())
        if clean:
            return clean
    return ""


def _replace_with_retry(tmp: Path, target: Path, attempts: int = 20) -> None:
    for index in range(attempts):
        try:
            tmp.replace(target)
            return
        except PermissionError:
            if index == attempts - 1:
                raise
            time.sleep(0.05 * (index + 1))


def _upgrade_checkpoint_object(checkpoint: Checkpoint) -> Checkpoint:
    defaults: dict[str, Any] = {
        "canonical_concepts": {},
        "concept_redirects": {},
        "surface_forms": {},
    }
    for key, value in defaults.items():
        if not hasattr(checkpoint, key):
            setattr(checkpoint, key, value)
    if int(getattr(checkpoint, "version", 1)) < 5 or not getattr(checkpoint, "canonical_concepts", {}):
        migrate_checkpoint(checkpoint)
    return checkpoint


def migrate_checkpoint(checkpoint: Checkpoint) -> dict[str, Any]:
    resolver = checkpoint.canonical_resolver
    concepts_before = len(checkpoint.canonical_concepts)
    redirects_before = len(checkpoint.concept_redirects)
    edges_before = len(checkpoint.custom_edges) + len(checkpoint.learned_bridges)

    merged_pheromones: dict[str, float] = {}
    for key, value in checkpoint.pheromones.items():
        start, relation, end = _split_edge_key(key)
        canonical_start = resolver.canonical_uri(start)
        canonical_end = resolver.canonical_uri(end)
        canonical_key = edge_key(canonical_start, relation, canonical_end)
        merged_pheromones[canonical_key] = merged_pheromones.get(canonical_key, 0.0) + float(value)
        if canonical_key != key:
            checkpoint.concept_redirects[key] = canonical_key
    checkpoint.pheromones = merged_pheromones

    merged_concept_pheromones: dict[str, float] = {}
    for uri, value in checkpoint.concept_pheromones.items():
        canonical = resolver.canonical_uri(uri)
        merged_concept_pheromones[canonical] = merged_concept_pheromones.get(canonical, 0.0) + float(value)
        if canonical != uri:
            checkpoint.concept_redirects[uri] = canonical
    checkpoint.concept_pheromones = merged_concept_pheromones

    merged_suppressed: dict[str, float] = {}
    for uri, value in checkpoint.suppressed_concepts.items():
        canonical = resolver.canonical_uri(uri)
        merged_suppressed[canonical] = merged_suppressed.get(canonical, 0.0) + float(value)
        if canonical != uri:
            checkpoint.concept_redirects[uri] = canonical
    checkpoint.suppressed_concepts = merged_suppressed

    canonical_concepts: dict[str, dict[str, Any]] = {}
    for uri, info in list(checkpoint.canonical_concepts.items()):
        canonical = resolver.canonical_uri(uri)
        payload = dict(info) if isinstance(info, dict) else {}
        payload["uri"] = canonical
        canonical_concepts.setdefault(canonical, {}).update(payload)
        if canonical != uri:
            checkpoint.concept_redirects[uri] = canonical
    checkpoint.canonical_concepts = canonical_concepts

    surface_forms: dict[str, dict[str, list[str]]] = {}
    for canonical, by_lang in list(checkpoint.surface_forms.items()):
        canonical_key = resolver.canonical_uri(canonical)
        bucket = surface_forms.setdefault(canonical_key, {})
        if not isinstance(by_lang, dict):
            continue
        for lang, values in by_lang.items():
            if not isinstance(values, list):
                continue
            bucket.setdefault(lang, [])
            for value in values:
                clean = " ".join(str(value).split())
                if clean and clean not in bucket[lang]:
                    bucket[lang].append(clean)
    checkpoint.surface_forms = surface_forms

    new_aliases: dict[str, str] = {}
    for token, uri in checkpoint.aliases.items():
        canonical = resolver.canonical_uri(uri)
        new_aliases[token] = canonical
        if canonical != uri:
            checkpoint.concept_redirects[uri] = canonical
    checkpoint.aliases = new_aliases

    def _canonicalize_list(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            concepts = [resolver.canonical_uri(str(value)) for value in item.get("concepts", []) if value]
            answer_concepts = [resolver.canonical_uri(str(value)) for value in item.get("answer_concepts", []) if value]
            item = {
                **item,
                "concepts": list(dict.fromkeys(concepts)),
                "answer_concepts": list(dict.fromkeys(answer_concepts)),
            }
            output.append(item)
        return output

    checkpoint.accepted_answers = _normalize_patterns(
        checkpoint.accepted_answers,
        fallback_lang="auto",
        checkpoint_data=checkpoint.to_dict(),
    )
    checkpoint.response_memory = _normalize_response_memory(checkpoint.response_memory, checkpoint.to_dict())
    checkpoint.negative_memory = _normalize_negative_memory(checkpoint.negative_memory, checkpoint.to_dict())
    checkpoint.custom_edges = _canonicalize_edges(checkpoint.custom_edges, resolver)
    checkpoint.learned_bridges = _canonicalize_edges(checkpoint.learned_bridges, resolver)
    checkpoint.version = 5

    return {
        "version": checkpoint.version,
        "canonical_concepts_before": concepts_before,
        "canonical_concepts_after": len(checkpoint.canonical_concepts),
        "redirects_before": redirects_before,
        "redirects_after": len(checkpoint.concept_redirects),
        "edges_before": edges_before,
        "edges_after": len(checkpoint.custom_edges) + len(checkpoint.learned_bridges),
    }


def _canonicalize_edges(values: list[dict[str, Any]], resolver: CanonicalResolver) -> list[dict[str, Any]]:
    canonicalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in values:
        if not isinstance(item, dict):
            continue
        start = resolver.canonical_uri(str(item.get("start", "")))
        end = resolver.canonical_uri(str(item.get("end", "")))
        relation = str(item.get("relation", "LearnedRelatedTo"))
        key = (start, relation, end)
        if key in seen:
            continue
        seen.add(key)
        canonicalized.append({**item, "start": start, "end": end})
    return canonicalized


def _split_edge_key(value: str) -> tuple[str, str, str]:
    return split_edge_key(value)
