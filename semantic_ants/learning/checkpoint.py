from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semantic_ants.core.models import SemanticEdge, edge_key
from semantic_ants.core.normalization import detect_language, text_to_concept_uri, tokenize


@dataclass
class Checkpoint:
    """Обучаемый слой поверх внешнего графа знаний."""

    version: int = 4
    pheromones: dict[str, float] = field(default_factory=dict)
    concept_pheromones: dict[str, float] = field(default_factory=dict)
    suppressed_concepts: dict[str, float] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
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

    def pheromone_for(self, edge: SemanticEdge) -> float:
        return max(float(self.pheromones.get(edge.key, 1.0)), 0.01)

    def concept_pheromone_for(self, concept_uri: str) -> float:
        return max(float(self.concept_pheromones.get(concept_uri, 1.0)), 0.01)

    def penalty_for(self, concept_uri: str) -> float:
        penalty = max(float(self.suppressed_concepts.get(concept_uri, 0.0)), 0.0)
        return 1.0 / (1.0 + penalty)

    def reinforce_edge(self, start: str, relation: str, end: str, amount: float = 0.2) -> None:
        key = edge_key(start, relation, end)
        self.pheromones[key] = min(float(self.pheromones.get(key, 1.0)) + amount, 25.0)
        stat = self.route_stats.setdefault(key, {"positive": 0, "negative": 0})
        stat["positive"] = int(stat.get("positive", 0)) + 1
        self.reinforce_concept(start, amount=amount * 0.25)
        self.reinforce_concept(end, amount=amount * 0.5)

    def penalize_edge(self, start: str, relation: str, end: str, amount: float = 0.2) -> None:
        key = edge_key(start, relation, end)
        self.pheromones[key] = max(float(self.pheromones.get(key, 1.0)) - amount, 0.05)
        stat = self.route_stats.setdefault(key, {"positive": 0, "negative": 0})
        stat["negative"] = int(stat.get("negative", 0)) + 1
        self.penalize_concept(end, amount=amount * 0.5)

    def reinforce_concept(self, concept_uri: str, amount: float = 0.2) -> None:
        if not concept_uri:
            return
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
        self.concept_pheromones[concept_uri] = max(
            float(self.concept_pheromones.get(concept_uri, 1.0)) - amount,
            0.05,
        )

    def suppress_concept(self, concept_uri: str, amount: float = 0.5) -> None:
        self.suppressed_concepts[concept_uri] = min(
            float(self.suppressed_concepts.get(concept_uri, 0.0)) + amount,
            25.0,
        )
        self.penalize_concept(concept_uri, amount=amount * 0.25)

    def remember_response(self, concepts: list[str], response: str, amount: float = 1.0) -> None:
        if not concepts or not response:
            return
        clean_response = " ".join(response.split())
        if not clean_response:
            return
        key = concept_set_key(concepts)
        lang = _detect_lang_from_concepts(concepts) or detect_language(clean_response)
        item = self.response_memory.setdefault(
            key,
            {
                "concepts": list(dict.fromkeys(concepts)),
                "answer_concepts": [],
                "lang": lang,
                "answer": "",
                "response": "",
                "weight": 0.0,
            },
        )
        item["concepts"] = list(dict.fromkeys(concepts))
        item["answer_concepts"] = _text_to_concepts(clean_response, lang=lang, checkpoint=self)
        item["lang"] = lang
        item["answer"] = clean_response
        item["response"] = clean_response
        item["weight"] = float(item.get("weight", 0.0)) + amount

    def remember_concept_label(self, concept_uri: str, label: str) -> None:
        clean_label = " ".join(label.split())
        if not concept_uri or not clean_label:
            return
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
    ) -> dict[str, Any] | None:
        if not answer:
            return None
        clean_answer = " ".join(answer.split())
        if not clean_answer:
            return None
        lang = _detect_lang_from_concepts(concepts) or detect_language(answer)
        answer_concepts = _text_to_concepts(clean_answer, lang=lang, checkpoint=self)
        item = {
            "stimulus": stimulus,
            "semantic_prompt": semantic_prompt,
            "concepts": list(dict.fromkeys(concepts)),
            "answer_concepts": answer_concepts,
            "answer": clean_answer,
            "lang": lang,
            "reward": float(reward),
            "created_at": time.time(),
        }
        for existing in self.accepted_answers:
            if (
                str(existing.get("stimulus", "")).strip().lower() == stimulus.strip().lower()
                and " ".join(str(existing.get("answer", "")).split()) == clean_answer
            ):
                existing["semantic_prompt"] = semantic_prompt
                existing["concepts"] = item["concepts"]
                existing["answer_concepts"] = answer_concepts
                existing["answer"] = clean_answer
                existing["lang"] = lang
                existing["reward"] = max(float(existing.get("reward", 0.0)), float(reward))
                existing["created_at"] = time.time()
                self.accepted_answers.sort(key=lambda value: float(value.get("reward", 0.0)), reverse=True)
                self.remember_response(item["concepts"], clean_answer, amount=max(float(reward), 0.1))
                return existing
        self.accepted_answers.append(item)
        self.accepted_answers.sort(key=lambda value: float(value.get("reward", 0.0)), reverse=True)
        del self.accepted_answers[limit:]
        self.remember_response(item["concepts"], answer, amount=max(float(reward), 0.1))
        return item

    def remember_negative(
        self,
        stimulus: str,
        semantic_prompt: str,
        concepts: list[str],
        answer: str,
        reason: str = "",
        limit: int = 500,
    ) -> dict[str, Any]:
        lang = _detect_lang_from_concepts(concepts) or detect_language(answer or stimulus or semantic_prompt)
        clean_answer = " ".join(answer.split())
        item = {
            "stimulus": stimulus,
            "semantic_prompt": semantic_prompt,
            "concepts": list(dict.fromkeys(concepts)),
            "answer_concepts": _text_to_concepts(clean_answer, lang=lang, checkpoint=self) if clean_answer else [],
            "answer": clean_answer,
            "lang": lang,
            "reason": reason,
            "created_at": time.time(),
        }
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
                "concepts": list(dict.fromkeys(concepts or [])),
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
                    start=str(raw["start"]),
                    end=str(raw["end"]),
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
        return cls(
            version=max(int(data.get("version", 1)), 4),
            pheromones=dict(data.get("pheromones", {})),
            concept_pheromones=dict(data.get("concept_pheromones", {})),
            suppressed_concepts=dict(data.get("suppressed_concepts", {})),
            aliases=dict(data.get("aliases", {})),
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


class CheckpointStore:
    """Читает и сохраняет checkpoint в JSON."""

    def __init__(self, path: Path | str = ".semantic_ants/checkpoints/model.json") -> None:
        self.path = Path(path)

    def load(self) -> Checkpoint:
        if not self.path.exists():
            return Checkpoint()
        with self.path.open("r", encoding="utf-8") as handle:
            return Checkpoint.from_dict(json.load(handle))

    def save(self, checkpoint: Checkpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(checkpoint.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        _replace_with_retry(tmp, self.path)

    def export(self, destination: Path | str) -> None:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            shutil.copyfile(self.path, target)
        else:
            self.save(Checkpoint())
            shutil.copyfile(self.path, target)


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
            concepts.append(uri)
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
        lang = str(item.get("lang") or fallback_lang)
        concepts = [str(value) for value in item.get("concepts", []) if value]
        raw_answer = _clean_answer(item)
        answer_concepts = item.get("answer_concepts")
        if not isinstance(answer_concepts, list) or not answer_concepts:
            answer_concepts = _text_to_concepts(raw_answer, lang=lang, checkpoint=_checkpoint_from_data(checkpoint_data))
        pattern = {
            "stimulus": str(item.get("stimulus", "")),
            "semantic_prompt": str(item.get("semantic_prompt", "")),
            "concepts": list(dict.fromkeys(concepts)),
            "answer_concepts": [str(value) for value in answer_concepts if value],
            "answer": raw_answer,
            "lang": lang if lang in {"ru", "en"} else detect_language(" ".join(answer_concepts or concepts)),
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
        lang = str(item.get("lang") or _detect_lang_from_concepts(concepts) or "auto")
        raw_answer = _clean_answer(item)
        answer_concepts = item.get("answer_concepts")
        if not isinstance(answer_concepts, list) or not answer_concepts:
            answer_concepts = _text_to_concepts(raw_answer, lang=lang, checkpoint=checkpoint)
        normalized[str(key)] = {
            "concepts": list(dict.fromkeys(concepts)),
            "answer_concepts": [str(value) for value in answer_concepts if value],
            "answer": raw_answer,
            "response": raw_answer,
            "lang": lang if lang in {"ru", "en"} else _detect_lang_from_concepts(concepts) or detect_language(str(key)),
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
        lang = str(item.get("lang") or _detect_lang_from_concepts(concepts) or "auto")
        raw_answer = _clean_answer(item)
        answer_concepts = item.get("answer_concepts")
        if not isinstance(answer_concepts, list):
            answer_concepts = _text_to_concepts(raw_answer, lang=lang, checkpoint=checkpoint)
        normalized.append(
            {
                "stimulus": str(item.get("stimulus", "")),
                "semantic_prompt": str(item.get("semantic_prompt", "")),
                "concepts": list(dict.fromkeys(concepts)),
                "answer_concepts": [str(value) for value in answer_concepts if value],
                "answer": raw_answer,
                "lang": lang if lang in {"ru", "en"} else _detect_lang_from_concepts(concepts) or detect_language(str(item.get("answer", ""))),
                "reason": str(item.get("reason", "")),
                "created_at": float(item.get("created_at", time.time())),
            }
        )
    return normalized


def _checkpoint_from_data(data: dict[str, Any]) -> Checkpoint:
    return Checkpoint(
        version=max(int(data.get("version", 1)), 4),
        pheromones=dict(data.get("pheromones", {})),
        concept_pheromones=dict(data.get("concept_pheromones", {})),
        suppressed_concepts=dict(data.get("suppressed_concepts", {})),
        aliases=dict(data.get("aliases", {})),
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
