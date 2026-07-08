from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from functools import lru_cache
from typing import Any

from semantic_ants.core.models import SemanticEdge, edge_key
from semantic_ants.core.normalization import detect_language, text_to_concept_uri, tokenize
from semantic_ants.learning.checkpoint import Checkpoint

try:
    import pymorphy3
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pymorphy3 = None


RESONANCE_FLAG = "resonance_experiment"
DEFAULT_SESSION = "default"
DEFAULT_LANG = "ru"


def clean_resonance_checkpoint(seed: int = 42) -> Checkpoint:
    return Checkpoint(
        version=7,
        seed=seed,
        metadata={
            RESONANCE_FLAG: True,
            "resonance_created_at": time.time(),
            "concept_labels": {},
            "concept_definitions": {},
        },
    )


def seed_resonance(checkpoint: Checkpoint, force: bool = False, session_id: str = DEFAULT_SESSION) -> dict[str, Any]:
    if checkpoint.metadata.get("resonance_seeded") and not force:
        return resonance_memory(checkpoint) | {"changed": False}

    if force:
        keep_seed = checkpoint.seed
        checkpoint.__dict__.update(clean_resonance_checkpoint(seed=keep_seed).__dict__)

    checkpoint.version = 7
    checkpoint.metadata[RESONANCE_FLAG] = True
    before_edges = len(checkpoint.custom_edges)
    before_forms = sum(len(values) for values in checkpoint.morph_forms.values())

    for plane_id, label in {
        "semantic:nature": "Плоскость природы",
        "language:ru": "Русская языковая плоскость",
        "morphology:ru": "Русские словоформы",
        "syntax:ru": "Русский синтаксис",
        "dev:filesystem": "Файловое дерево",
        f"session:{session_id}": "Контекст сессии",
    }.items():
        _upsert_plane(checkpoint, plane_id, label)

    for area_id, (plane_id, label) in {
        "area:semantic:nature/tree": ("semantic:nature", "Зона понятия дерева"),
        "area:language:ru/tree_subject": ("language:ru", "Дерево как подлежащее"),
        "area:morphology:ru/tree_forms": ("morphology:ru", "Формы слова дерево"),
        "area:dev:filesystem/tree": ("dev:filesystem", "Дерево файлов"),
    }.items():
        _upsert_area(checkpoint, area_id, plane_id, label)

    seed_forms = [
        ("дерево", "NOUN", "дерево", {"case": "nomn", "number": "sing", "gender": "neut"}, "subject"),
        ("дерево", "NOUN", "деревья", {"case": "nomn", "number": "plur", "gender": "neut"}, "subject"),
        ("дерево", "NOUN", "деревом", {"case": "ablt", "number": "sing", "gender": "neut"}, "instrument"),
        ("дерево", "NOUN", "деревьями", {"case": "ablt", "number": "plur", "gender": "neut"}, "instrument"),
        ("стол", "NOUN", "стол", {"case": "nomn", "number": "sing", "gender": "masc"}, "subject"),
        ("стол", "NOUN", "столы", {"case": "nomn", "number": "plur", "gender": "masc"}, "subject"),
        ("стол", "NOUN", "столом", {"case": "ablt", "number": "sing", "gender": "masc"}, "instrument"),
        ("стол", "NOUN", "столами", {"case": "ablt", "number": "plur", "gender": "masc"}, "instrument"),
        ("машина", "NOUN", "машина", {"case": "nomn", "number": "sing", "gender": "femn"}, "subject"),
        ("машина", "NOUN", "машины", {"case": "nomn", "number": "plur", "gender": "femn"}, "subject"),
        ("машина", "NOUN", "машиной", {"case": "ablt", "number": "sing", "gender": "femn"}, "instrument"),
        ("машина", "NOUN", "машинами", {"case": "ablt", "number": "plur", "gender": "femn"}, "instrument"),
        ("программист", "NOUN", "программист", {"case": "nomn", "number": "sing", "gender": "masc"}, "subject"),
        ("код", "NOUN", "код", {"case": "accs", "number": "sing", "gender": "masc"}, "object"),
        ("компьютер", "NOUN", "компьютере", {"case": "loct", "number": "sing", "gender": "masc"}, "instrument"),
        ("писать", "VERB", "пишет", {"tense": "pres", "person": "3", "number": "sing"}, "predicate"),
        ("расти", "VERB", "растет", {"tense": "pres", "person": "3", "number": "sing"}, "predicate"),
        ("расти", "VERB", "растут", {"tense": "pres", "person": "3", "number": "plur"}, "predicate"),
        ("делать", "VERB", "делает", {"tense": "pres", "person": "3", "number": "sing"}, "predicate"),
    ]
    for lemma, pos, surface, gram, role in seed_forms:
        _register_form(
            checkpoint,
            lang=DEFAULT_LANG,
            lemma=lemma,
            pos=pos,
            surface=surface,
            gram=gram,
            role=role,
            plane_id="morphology:ru",
            reward=3.0,
            tentative=False,
        )

    for role, grams in {
        "subject": {"case": "nomn"},
        "predicate": {"tense": "pres"},
        "object": {"case": "accs"},
        "instrument": {"case": "loct"},
        "location": {"case": "loct"},
    }.items():
        role_uri = _role_uri(role)
        _remember_label(checkpoint, role_uri, role)
        for key, value in grams.items():
            _add_plane_edge(
                checkpoint,
                role_uri,
                _gram_uri(value),
                "RequiresGram",
                plane_id="syntax:ru",
                distance=0.12,
                weight=3.0,
                edge_type="syntax",
                metadata={"gram_key": key, "role": role},
            )

    _register_sentence(
        checkpoint,
        "деревья растут",
        [
            {"lemma": "дерево", "role": "subject", "pos": "NOUN", "gram": {"case": "nomn", "number": "plur"}},
            {"lemma": "расти", "role": "predicate", "pos": "VERB", "gram": {"tense": "pres", "person": "3", "number": "plur"}},
        ],
        plane_id="language:ru",
        reward=3.0,
    )
    _register_sentence(
        checkpoint,
        "программист пишет код",
        [
            {"lemma": "программист", "role": "subject", "pos": "NOUN", "gram": {"case": "nomn", "number": "sing"}},
            {"lemma": "писать", "role": "predicate", "pos": "VERB", "gram": {"tense": "pres", "person": "3", "number": "sing"}},
            {"lemma": "код", "role": "object", "pos": "NOUN", "gram": {"case": "accs", "number": "sing"}},
        ],
        plane_id="language:ru",
        reward=3.0,
    )
    _register_sentence(
        checkpoint,
        "программист пишет код на компьютере",
        [
            {"lemma": "программист", "role": "subject", "pos": "NOUN", "gram": {"case": "nomn", "number": "sing"}},
            {"lemma": "писать", "role": "predicate", "pos": "VERB", "gram": {"tense": "pres", "person": "3", "number": "sing"}},
            {"lemma": "код", "role": "object", "pos": "NOUN", "gram": {"case": "accs", "number": "sing"}},
            {
                "lemma": "компьютер",
                "role": "instrument",
                "pos": "NOUN",
                "gram": {"case": "loct", "number": "sing"},
                "preposition": "на",
            },
        ],
        plane_id="language:ru",
        reward=3.0,
    )

    tree = _concept_uri("дерево")
    for surface, target, plane_id, distance in [
        ("растение", _concept_uri("растение"), "semantic:nature", 0.18),
        ("файловое дерево", _concept_uri("файловое дерево"), "dev:filesystem", 0.22),
        ("структура данных", _concept_uri("структура данных"), "dev:data_structures", 0.28),
        ("подлежащее", _role_uri("subject"), "language:ru", 0.14),
        ("деревья", _form_uri("деревья"), "morphology:ru", 0.08),
        ("файл", _concept_uri("файл"), "dev:filesystem", 0.18),
        ("папка", _concept_uri("папка"), "dev:filesystem", 0.18),
        ("корень", _concept_uri("корень"), "dev:filesystem", 0.18),
    ]:
        _remember_label(checkpoint, target, surface)
        _add_plane_edge(
            checkpoint,
            tree,
            target,
            "PlaneRelated",
            plane_id=plane_id,
            distance=distance,
            weight=2.7,
            edge_type="plane",
            metadata={"seed_bridge": True, "area_id": _area_for_plane(plane_id)},
        )
        _add_area_membership(checkpoint, _area_for_plane(plane_id), tree, plane_id, weight=2.7)
        _add_area_membership(checkpoint, _area_for_plane(plane_id), target, plane_id, weight=2.4)

    checkpoint.metadata["resonance_seeded"] = True
    checkpoint.metadata["resonance_seeded_at"] = time.time()
    return resonance_memory(checkpoint) | {
        "changed": True,
        "added_edges": len(checkpoint.custom_edges) - before_edges,
        "added_forms": sum(len(values) for values in checkpoint.morph_forms.values()) - before_forms,
    }


def train_resonance_form(checkpoint: Checkpoint, payload: dict[str, Any]) -> dict[str, Any]:
    lang = _lang(payload)
    form = _register_form(
        checkpoint,
        lang=lang,
        lemma=str(payload.get("lemma", "")).strip(),
        pos=str(payload.get("pos") or "NOUN"),
        surface=str(payload.get("surface", "")).strip(),
        gram=_gram(payload.get("gram")),
        role=str(payload.get("role") or ""),
        plane_id=str(payload.get("plane_id") or f"morphology:{lang}"),
        reward=float(payload.get("reward") or 1.0),
        tentative=False,
    )
    return {"trained": True, "form": form, **resonance_memory(checkpoint)}


def train_resonance_sentence(checkpoint: Checkpoint, payload: dict[str, Any]) -> dict[str, Any]:
    sentence = " ".join(str(payload.get("sentence") or "").split())
    if not sentence:
        raise ValueError("sentence is required")
    lang = _lang(payload)
    raw_slots = payload.get("slots")
    slots = raw_slots if isinstance(raw_slots, list) and raw_slots else _slots_from_text(sentence, lang, None)
    item = _register_sentence(
        checkpoint,
        sentence,
        [dict(slot) for slot in slots if isinstance(slot, dict)],
        plane_id=str(payload.get("plane_id") or "language:ru"),
        reward=float(payload.get("reward") or 1.0),
        lang=lang,
    )
    return {"trained": True, "sentence": item, **resonance_memory(checkpoint)}


def train_resonance_qa(checkpoint: Checkpoint, payload: dict[str, Any]) -> dict[str, Any]:
    if not checkpoint.metadata.get("resonance_seeded"):
        seed_resonance(checkpoint, session_id=str(payload.get("session_id") or DEFAULT_SESSION))

    question = " ".join(str(payload.get("question") or "").split())
    expected_answer = " ".join(str(payload.get("expected_answer") or "").split())
    if not question:
        raise ValueError("question is required")
    if not expected_answer:
        raise ValueError("expected_answer is required")

    lang = _lang({"lang": payload.get("lang"), "text": f"{question} {expected_answer}"})
    session_id = str(payload.get("session_id") or DEFAULT_SESSION)
    session = checkpoint.session_contexts.setdefault(session_id, {"session_id": session_id})
    reward = max(float(payload.get("reward") or 1.0), 0.1)
    epochs = max(int(payload.get("epochs") or 1), 1)
    annotations = [dict(item) for item in payload.get("annotations", []) if isinstance(item, dict)]
    explicit_plane = str(payload.get("plane_id") or "").strip()
    active_plane = explicit_plane or _infer_plane(f"{question} {expected_answer}", session)
    _upsert_plane(checkpoint, active_plane, active_plane)

    answer_slots = _qa_slots_from_answer(expected_answer, lang, annotations)
    question_concepts = _question_concepts(checkpoint, question, lang)
    created_planes: set[str] = set()
    trained_forms: list[dict[str, Any]] = []
    used_planes: set[str] = {active_plane}
    skeleton: dict[str, Any] = {}
    last_epoch_slots: list[dict[str, Any]] = []

    for _ in range(epochs):
        epoch_slots: list[dict[str, Any]] = []
        for slot in answer_slots:
            slot = dict(slot)
            context_planes = _slot_context_planes(
                checkpoint,
                slot,
                lang=lang,
                active_plane=active_plane,
                created_planes=created_planes,
            )
            used_planes.update(context_planes)
            slot["planes"] = context_planes
            form = _register_form(
                checkpoint,
                lang=lang,
                lemma=str(slot.get("lemma") or "").strip(),
                pos=str(slot.get("pos") or "NOUN"),
                surface=str(slot.get("surface") or slot.get("token") or "").strip(),
                gram=_gram(slot.get("gram")),
                role=str(slot.get("role") or ""),
                plane_id=f"morphology:{lang}",
                reward=reward,
                tentative=False,
                concept_uri=str(slot.get("concept") or ""),
            )
            trained_forms.append(dict(form))
            slot["concept"] = form.get("concept")
            slot["form_uri"] = form.get("form_uri")
            _attach_slot_to_context_planes(checkpoint, form, slot, context_planes, reward=reward)
            epoch_slots.append(slot)

        skeleton = _register_sentence(
            checkpoint,
            expected_answer,
            epoch_slots,
            plane_id=active_plane,
            reward=reward,
            lang=lang,
        )
        last_epoch_slots = epoch_slots
        _link_question_to_answer(
            checkpoint,
            question_concepts,
            epoch_slots,
            plane_id=active_plane,
            reward=reward,
        )
    _bump_context(checkpoint, session_id, active_plane, amount=0.25 * reward)

    _remember_qa_template(
        checkpoint,
        question=question,
        question_concepts=question_concepts,
        answer_slots=last_epoch_slots or answer_slots,
        lang=lang,
        plane_id=active_plane,
        reward=reward,
        sentence_end="." if expected_answer.rstrip().endswith((".", "!", "?")) else "",
    )
    if question_concepts:
        checkpoint.remember_accepted_answer(
            question,
            question,
            question_concepts,
            expected_answer,
            reward=reward,
            lang=lang,
            source_lang=lang,
        )

    session["active_plane"] = active_plane
    session["active_planes"] = sorted(used_planes)
    session["active_concepts"] = [
        str(slot.get("concept") or "") for slot in last_epoch_slots if slot.get("concept")
    ]
    session["last_training_question"] = question
    session["last_training_answer"] = expected_answer
    session["updated_at"] = time.time()

    return {
        **resonance_memory(checkpoint),
        "trained": True,
        "mode": "question_answer",
        "question": question,
        "expected_answer": expected_answer,
        "lang": lang,
        "epochs": epochs,
        "active_plane": active_plane,
        "planes": sorted(used_planes),
        "created_planes": sorted(created_planes),
        "question_tokens": tokenize(question),
        "answer_tokens": [str(slot.get("token") or "") for slot in answer_slots],
        "lemmas": [str(slot.get("lemma") or "") for slot in answer_slots],
        "slots": last_epoch_slots or answer_slots,
        "forms": _dedupe_forms(trained_forms),
        "sentence": skeleton,
    }


def generate_resonance(checkpoint: Checkpoint, payload: dict[str, Any]) -> dict[str, Any]:
    if not checkpoint.metadata.get("resonance_seeded"):
        seed_resonance(checkpoint, session_id=str(payload.get("session_id") or DEFAULT_SESSION))

    lang = _lang(payload)
    session_id = str(payload.get("session_id") or DEFAULT_SESSION)
    creativity = min(max(float(payload.get("creativity") or 0.35), 0.0), 1.0)
    text = " ".join(str(payload.get("text") or "").split())
    explicit_plane = str(payload.get("plane_id") or "")
    raw_slots = payload.get("slots")
    slots = [dict(slot) for slot in raw_slots if isinstance(slot, dict)] if isinstance(raw_slots, list) else []
    if not slots:
        slots = _slots_from_qa_memory(
            checkpoint,
            text=text,
            lang=lang,
            session_id=session_id,
            plane_id=explicit_plane,
            creativity=creativity,
        )
    if not slots:
        slots = _slots_from_text(text, lang, explicit_plane)

    plane_id = explicit_plane or _infer_plane(text, checkpoint.session_contexts.get(session_id, {}))
    context_key = f"session:{session_id}"
    _upsert_plane(checkpoint, context_key, f"Контекст {session_id}")

    selected_tokens: list[dict[str, Any]] = []
    active_concepts: list[str] = []
    signal_trace: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    ordered_slots = _order_slots(checkpoint, slots)
    for index, slot in enumerate(ordered_slots):
        token = _resolve_slot(checkpoint, slot, lang=lang, plane_id=plane_id, session_id=session_id)
        selected_tokens.append(token)
        if token.get("concept"):
            active_concepts.append(str(token["concept"]))
        route = _route_for_token(index, token, plane_id)
        routes.append(route)
        signal_trace.extend(route["steps"])

    sentence = _render_sentence(selected_tokens)
    result_id = _result_id(text or sentence)
    result = {
        "result_id": result_id,
        "input_text": text,
        "lang": lang,
        "tokens": [str(slot.get("lemma", "")) for slot in ordered_slots if slot.get("lemma")],
        "activated_concepts": _activated_concepts(checkpoint, selected_tokens, plane_id),
        "routes": routes,
        "summary": f"plane={plane_id}; forms={len(selected_tokens)}",
        "response": sentence,
        "sources": ["resonance"],
        "session_id": session_id,
        "context_turns": checkpoint.session_history(session_id),
        "semantic_vector": {
            "mode": "resonance",
            "input_text": text,
            "active_plane": plane_id,
            "session_plane": context_key,
            "tokens": selected_tokens,
            "tentative_forms": [token for token in selected_tokens if token.get("tentative")],
            "creativity": creativity,
        },
        "signal_trace": signal_trace,
        "response_source": "resonance",
        "response_lang": lang,
        "response_candidates": [sentence] if sentence else [],
        "canonical_concepts": list(checkpoint.canonical_concepts.keys()),
    }
    _remember_resonance_result(checkpoint, result, selected_tokens, plane_id, session_id, text)
    return result


def apply_resonance_feedback(checkpoint: Checkpoint, payload: dict[str, Any]) -> dict[str, Any]:
    result_id = str(payload.get("result_id") or checkpoint.last_result_id or "")
    if not result_id:
        raise ValueError("result_id is required")
    result = checkpoint.results.get(result_id)
    if not isinstance(result, dict):
        raise ValueError("result not found")

    score = int(payload.get("score") or 0)
    session_id = str(payload.get("session_id") or result.get("session_id") or DEFAULT_SESSION)
    plane_id = str((result.get("semantic_vector") or {}).get("active_plane") or "language:ru")
    tokens = [token for token in (result.get("semantic_vector") or {}).get("tokens", []) if isinstance(token, dict)]
    changed_edges = 0
    if score >= 4:
        for route in result.get("routes", []) or []:
            for step in route.get("steps", []) or []:
                checkpoint.reinforce_edge(
                    str(step.get("start", "")),
                    str(step.get("relation", "")),
                    str(step.get("end", "")),
                    amount=0.7,
                    context_plane=plane_id,
                )
                _shorten_distance(checkpoint, plane_id, str(step.get("start", "")), str(step.get("end", "")), ratio=0.72)
                changed_edges += 1
        for token in tokens:
            _commit_tentative_form(checkpoint, token)
        _bump_context(checkpoint, session_id, plane_id, amount=0.5)
    elif score <= 2:
        template_key = _qa_template_key(str(result.get("input_text") or ""), str(result.get("lang") or DEFAULT_LANG), plane_id)
        for route in result.get("routes", []) or []:
            for step in route.get("steps", []) or []:
                checkpoint.penalize_edge(
                    str(step.get("start", "")),
                    str(step.get("relation", "")),
                    str(step.get("end", "")),
                    amount=0.5,
                    context_plane=plane_id,
                )
                _lengthen_distance(checkpoint, plane_id, str(step.get("start", "")), str(step.get("end", "")), ratio=1.45)
                changed_edges += 1
        for token in tokens:
            checkpoint.suppress_concept(str(token.get("form_uri") or token.get("concept") or ""), amount=0.8)
        _penalize_qa_template(checkpoint, template_key, amount=1.0 + (2 - score) * 0.6)
        _bump_context(checkpoint, session_id, plane_id, amount=-0.3)

    corrected = payload.get("corrected_tokens")
    if isinstance(corrected, list):
        for item in corrected:
            if isinstance(item, dict):
                train_resonance_form(checkpoint, item)
    corrected_sentence = " ".join(str(payload.get("corrected_sentence") or "").split())
    if corrected_sentence:
        train_resonance_sentence(checkpoint, {"sentence": corrected_sentence, "lang": result.get("lang", DEFAULT_LANG)})

    feedback = {
        "result_id": result_id,
        "session_id": session_id,
        "score": score,
        "plane_id": plane_id,
        "changed_edges": changed_edges,
        "created_at": time.time(),
    }
    checkpoint.resonance_feedback.append(feedback)
    del checkpoint.resonance_feedback[:-500]
    return {"ok": True, **feedback}


def resonance_memory(checkpoint: Checkpoint) -> dict[str, Any]:
    return {
        "version": checkpoint.version,
        "resonance_experiment": bool(checkpoint.metadata.get(RESONANCE_FLAG)),
        "seeded": bool(checkpoint.metadata.get("resonance_seeded")),
        "planes": len(checkpoint.planes),
        "areas": len(checkpoint.areas),
        "area_memberships": len(checkpoint.area_memberships),
        "area_bridges": len(checkpoint.area_bridges),
        "morph_forms": sum(len(values) for values in checkpoint.morph_forms.values()),
        "morph_patterns": len(checkpoint.morph_patterns),
        "syntax_skeletons": len(checkpoint.syntax_skeletons),
        "session_contexts": len(checkpoint.session_contexts),
        "feedback": len(checkpoint.resonance_feedback),
        "custom_edges": len(checkpoint.custom_edges),
        "accepted_answers": len(checkpoint.accepted_answers),
        "learned_bridges": len(checkpoint.learned_bridges),
    }


def resonance_planes(checkpoint: Checkpoint) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "plane_id": plane_id,
                **dict(info),
                "pheromone": float(checkpoint.plane_pheromones.get(plane_id, 1.0)),
            }
            for plane_id, info in checkpoint.planes.items()
        ],
        key=lambda item: str(item["plane_id"]),
    )


def resonance_areas(checkpoint: Checkpoint) -> list[dict[str, Any]]:
    memberships: dict[str, int] = Counter()
    for key in checkpoint.area_memberships:
        try:
            payload = json.loads(key)
            memberships[str(payload.get("area_id", ""))] += 1
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return sorted(
        [
            {
                "area_id": area_id,
                **dict(info),
                "members": memberships.get(area_id, 0),
            }
            for area_id, info in checkpoint.areas.items()
        ],
        key=lambda item: str(item["area_id"]),
    )


def resonance_session_context(checkpoint: Checkpoint, session_id: str = DEFAULT_SESSION) -> dict[str, Any]:
    return dict(checkpoint.session_contexts.get(session_id, {"session_id": session_id}))


def effective_distance(checkpoint: Checkpoint, start: str, end: str, plane_id: str) -> float:
    start = checkpoint.canonical_uri(start)
    end = checkpoint.canonical_uri(end)
    if start == end:
        return 0.03
    direct = checkpoint.plane_distance_overrides.get(_distance_key(plane_id, start, end))
    if direct is not None:
        return float(direct)
    reverse = checkpoint.plane_distance_overrides.get(_distance_key(plane_id, end, start))
    if reverse is not None:
        return float(reverse)
    best = 9.0
    for edge in checkpoint.learned_edges():
        if {edge.start, edge.end} == {start, end}:
            edge_plane = str(edge.metadata.get("plane_id") or edge.context_plane or "")
            if edge_plane == plane_id:
                best = min(best, float(edge.distance))
            else:
                best = min(best, float(edge.distance) * 4.0)
    return best


def _register_form(
    checkpoint: Checkpoint,
    *,
    lang: str,
    lemma: str,
    pos: str,
    surface: str,
    gram: dict[str, str],
    role: str = "",
    plane_id: str = "morphology:ru",
    reward: float = 1.0,
    tentative: bool = False,
    pattern_id: str = "",
    concept_uri: str = "",
    area_id: str | None = None,
) -> dict[str, Any]:
    if not lemma:
        raise ValueError("lemma is required")
    if not surface:
        raise ValueError("surface is required")
    concept = _concept_from_hint(checkpoint, concept_uri, lemma, lang)
    form_uri = _form_uri(surface, lang, lemma=lemma, concept_uri=concept)
    _remember_label(checkpoint, concept, lemma)
    _remember_label(checkpoint, form_uri, surface)
    checkpoint.register_surface_form(concept, lemma, lang=lang)
    checkpoint.register_surface_form(form_uri, surface, lang=lang)
    key = _form_key(lang, lemma)
    forms = checkpoint.morph_forms.setdefault(key, [])
    item = {
        "lang": lang,
        "lemma": lemma,
        "concept": concept,
        "pos": pos,
        "surface": surface,
        "form_uri": form_uri,
        "gram": dict(gram),
        "role": role,
        "plane_id": plane_id,
        "pheromone": max(float(reward), 0.1),
        "tentative": tentative,
        "pattern_id": pattern_id,
    }
    for existing in forms:
        if str(existing.get("surface", "")).casefold() == surface.casefold() and dict(existing.get("gram", {})) == dict(gram):
            existing.update(
                {
                    **item,
                    "pheromone": min(
                        max(float(existing.get("pheromone", 1.0)), item["pheromone"]) + max(float(reward), 0.1),
                        25.0,
                    ),
                    "tentative": bool(existing.get("tentative", tentative)) and tentative,
                }
            )
            item = existing
            break
    else:
        forms.append(item)
    if concept != form_uri:
        _add_plane_edge(
            checkpoint,
            concept,
            form_uri,
            "HasForm",
            plane_id=plane_id,
            distance=0.08 if not tentative else 0.65,
            weight=2.5 if not tentative else 0.9,
            edge_type="morphology",
                metadata={"lemma": lemma, "surface": surface, "tentative": tentative, "area_id": _area_for_plane(plane_id)},
        )
    for gram_value in gram.values():
        if gram_value:
            _add_plane_edge(
                checkpoint,
                form_uri,
                _gram_uri(gram_value, lang),
                "RealizesGram",
                plane_id=plane_id,
                distance=0.1,
                weight=2.0,
                edge_type="morphology",
                metadata={"lemma": lemma, "surface": surface, "area_id": _area_for_plane(plane_id)},
            )
    if role:
        _add_plane_edge(
            checkpoint,
            form_uri,
            _role_uri(role, lang),
            "RealizesRole",
            plane_id="language:ru",
            distance=0.14,
            weight=2.0,
            edge_type="syntax",
            metadata={"role": role, "surface": surface, "area_id": "area:language:ru/tree_subject"},
        )
    if not tentative:
        _learn_pattern(checkpoint, lang, lemma, surface, pos, gram, reward=reward)
    membership_area_id = area_id or _auto_area_id(plane_id, lemma, role, concept)
    _upsert_area(checkpoint, membership_area_id, plane_id, _auto_area_label(membership_area_id, lemma, role))
    _add_area_membership(checkpoint, membership_area_id, form_uri, plane_id, weight=max(reward, 0.1))
    _add_area_membership(checkpoint, membership_area_id, concept, plane_id, weight=max(reward * 0.7, 0.1))
    return item


def _register_sentence(
    checkpoint: Checkpoint,
    sentence: str,
    slots: list[dict[str, Any]],
    *,
    plane_id: str,
    reward: float,
    lang: str = DEFAULT_LANG,
) -> dict[str, Any]:
    skeleton_id = f"skel:{hashlib.sha1(sentence.encode('utf-8')).hexdigest()[:10]}"
    item = checkpoint.syntax_skeletons.setdefault(
        skeleton_id,
        {
            "id": skeleton_id,
            "sentence": sentence,
            "roles": [str(slot.get("role", "")) for slot in slots],
            "slots": slots,
            "plane_id": plane_id,
            "pheromone": 0.0,
        },
    )
    item["pheromone"] = float(item.get("pheromone", 0.0)) + reward
    previous_uri = ""
    for index, slot in enumerate(slots):
        lemma = str(slot.get("lemma", "")).strip()
        role = str(slot.get("role", "")).strip()
        pos = str(slot.get("pos") or ("VERB" if role == "predicate" else "NOUN"))
        gram = _gram(slot.get("gram"))
        surface = str(slot.get("surface") or "").strip()
        if not surface:
            surface = _surface_from_existing(checkpoint, lemma, lang, gram) or lemma
        concept_uri = str(slot.get("concept") or slot.get("concept_uri") or "")
        form = _register_form(
            checkpoint,
            lang=lang,
            lemma=lemma,
            pos=pos,
            surface=surface,
            gram=gram,
            role=role,
            plane_id=f"morphology:{lang}",
            reward=reward,
            tentative=False,
            concept_uri=concept_uri,
        )
        form_uri = str(form.get("form_uri", ""))
        if previous_uri and form_uri:
            _add_plane_edge(
                checkpoint,
                previous_uri,
                form_uri,
                "NextSlot",
                plane_id=plane_id,
                distance=0.18,
                weight=2.4,
                edge_type="syntax",
                metadata={"skeleton_id": skeleton_id, "index": index},
            )
        previous_uri = form_uri
    return item


def _qa_slots_from_answer(answer: str, lang: str, annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_tokens = _raw_tokens(answer)
    tokens = [token.casefold() for token in raw_tokens]
    by_index: dict[int, dict[str, Any]] = {}
    by_token: dict[str, list[dict[str, Any]]] = {}
    for item in annotations:
        index = item.get("index")
        if index is not None:
            try:
                by_index[int(index)] = item
            except (TypeError, ValueError):
                pass
        token = str(item.get("token") or item.get("surface") or "").strip().casefold()
        if token:
            by_token.setdefault(token, []).append(item)

    analyses = [_analyze_training_token(token, lang) for token in tokens]
    predicate_index = next(
        (index for index, item in enumerate(analyses) if str(item.get("pos")) == "VERB"),
        -1,
    )
    slots: list[dict[str, Any]] = []
    pending_preposition = ""
    for index, token in enumerate(tokens):
        normalized = token.casefold()
        if normalized in _RU_PREPOSITIONS:
            pending_preposition = normalized
            continue
        annotation = by_index.get(index)
        if annotation is None:
            annotation_pool = by_token.get(normalized, [])
            annotation = annotation_pool.pop(0) if annotation_pool else {}
        analysis = analyses[index]
        role = str(annotation.get("role") or "").strip() or _infer_training_role(
            index=index,
            predicate_index=predicate_index,
            pos=str(analysis.get("pos") or "NOUN"),
            preposition=pending_preposition,
        )
        lemma = str(annotation.get("lemma") or analysis.get("lemma") or normalized).strip().casefold()
        pos = str(annotation.get("pos") or analysis.get("pos") or ("VERB" if role == "predicate" else "NOUN"))
        gram = dict(analysis.get("gram", {}))
        gram.update(_annotation_gram(annotation.get("gram")))
        gram = _required_gram({"gram": gram}, role)
        preposition = str(annotation.get("preposition") or pending_preposition or "").strip()
        slot = {
            "id": str(annotation.get("id") or f"answer:{index}"),
            "index": index,
            "token": str(annotation.get("token") or raw_tokens[index] or normalized).strip(),
            "surface": str(annotation.get("surface") or raw_tokens[index] or normalized).strip(),
            "lemma": lemma,
            "role": role,
            "pos": pos,
            "gram": gram,
            "preposition": preposition,
            "concept": str(annotation.get("concept") or annotation.get("concept_uri") or "").strip(),
            "planes": _planes_from_annotation(annotation),
        }
        slots.append(slot)
        pending_preposition = ""
    return slots


def _question_concepts(checkpoint: Checkpoint, question: str, lang: str) -> list[str]:
    concepts: list[str] = []
    for token in tokenize(question):
        normalized = token.casefold()
        if normalized in _QA_STOP_WORDS or normalized in _RU_PREPOSITIONS:
            continue
        analysis = _analyze_training_token(normalized, lang)
        lemma = str(analysis.get("lemma") or normalized)
        if not lemma or lemma in _QA_STOP_WORDS:
            continue
        concept = _concept_uri(lemma, lang)
        _remember_label(checkpoint, concept, lemma)
        concepts.append(checkpoint.canonical_uri(concept))
    return list(dict.fromkeys(concepts))


def _slot_context_planes(
    checkpoint: Checkpoint,
    slot: dict[str, Any],
    *,
    lang: str,
    active_plane: str,
    created_planes: set[str],
) -> list[str]:
    explicit = [plane for plane in slot.get("planes", []) if plane]
    if explicit:
        planes = explicit
    else:
        inferred = _infer_plane(
            " ".join(str(slot.get(key) or "") for key in ("token", "lemma", "surface")),
            {},
        )
        planes = [active_plane]
        if inferred != active_plane:
            planes.append(inferred)
        if _is_new_concept(checkpoint, slot, lang):
            learned_plane = _learned_plane_id(str(slot.get("lemma") or slot.get("token") or "concept"), lang)
            created_planes.add(learned_plane)
            planes.append(learned_plane)
    unique_planes = list(dict.fromkeys(plane for plane in planes if plane))
    for plane_id in unique_planes:
        _upsert_plane(checkpoint, plane_id, plane_id)
        _upsert_area(checkpoint, _area_for_plane(plane_id), plane_id, _area_label_for_plane(plane_id))
    return unique_planes


def _attach_slot_to_context_planes(
    checkpoint: Checkpoint,
    form: dict[str, Any],
    slot: dict[str, Any],
    planes: list[str],
    *,
    reward: float,
) -> None:
    concept = str(form.get("concept") or slot.get("concept") or "")
    form_uri = str(form.get("form_uri") or "")
    role = str(slot.get("role") or form.get("role") or "")
    gram = _gram(slot.get("gram"))
    for plane_id in planes:
        area_id = _auto_area_id(plane_id, str(slot.get("lemma") or slot.get("token") or concept), role, concept)
        _upsert_area(checkpoint, area_id, plane_id, _auto_area_label(area_id, str(slot.get("lemma") or ""), role))
        _add_area_membership(checkpoint, area_id, concept, plane_id, weight=max(reward, 0.1))
        _add_area_membership(checkpoint, area_id, form_uri, plane_id, weight=max(reward * 0.8, 0.1))
        if concept and form_uri and concept != form_uri:
            _add_plane_edge(
                checkpoint,
                concept,
                form_uri,
                "ContextHasForm",
                plane_id=plane_id,
                distance=max(0.05, 0.16 / reward),
                weight=max(1.3 * reward, 0.1),
                edge_type="morphology",
                metadata={"role": role, "area_id": area_id, "training_mode": "question_answer"},
            )
        if role:
            _add_plane_edge(
                checkpoint,
                concept,
                _role_uri(role),
                "ContextRole",
                plane_id=plane_id,
                distance=max(0.06, 0.22 / reward),
                weight=max(1.1 * reward, 0.1),
                edge_type="syntax",
                metadata={"role": role, "area_id": area_id, "training_mode": "question_answer"},
            )
        for gram_value in gram.values():
            if not gram_value or not form_uri:
                continue
            _add_plane_edge(
                checkpoint,
                form_uri,
                _gram_uri(gram_value),
                "ContextGram",
                plane_id=plane_id,
                distance=max(0.06, 0.2 / reward),
                weight=max(0.9 * reward, 0.1),
                edge_type="morphology",
                metadata={"gram_value": gram_value, "area_id": area_id, "training_mode": "question_answer"},
            )


def _link_question_to_answer(
    checkpoint: Checkpoint,
    question_concepts: list[str],
    answer_slots: list[dict[str, Any]],
    *,
    plane_id: str,
    reward: float,
) -> None:
    answer_concepts = [
        str(slot.get("concept") or "")
        for slot in answer_slots
        if slot.get("concept")
    ]
    if not question_concepts:
        question_concepts = answer_concepts[:1]
    for question_concept in question_concepts:
        for answer_concept in answer_concepts:
            if not question_concept or not answer_concept:
                continue
            _add_plane_edge(
                checkpoint,
                question_concept,
                answer_concept,
                "QuestionActivates",
                plane_id=plane_id,
                distance=max(0.07, 0.26 / reward),
                weight=max(1.4 * reward, 0.1),
                edge_type="semantic",
                metadata={"training_mode": "question_answer", "area_id": _area_for_plane(plane_id)},
            )
            _shorten_distance(checkpoint, plane_id, question_concept, answer_concept, ratio=0.9)


def _dedupe_forms(forms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    values: list[dict[str, Any]] = []
    for form in forms:
        key = (
            str(form.get("lang") or ""),
            str(form.get("lemma") or ""),
            str(form.get("surface") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        values.append(form)
    return values


def _raw_tokens(text: str) -> list[str]:
    return [match.group(0) for match in re.finditer(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-'][0-9A-Za-zА-Яа-яЁё]+)?", text)]


def _remember_qa_template(
    checkpoint: Checkpoint,
    *,
    question: str,
    question_concepts: list[str],
    answer_slots: list[dict[str, Any]],
    lang: str,
    plane_id: str,
    reward: float,
    sentence_end: str = "",
) -> None:
    memory = checkpoint.metadata.setdefault("resonance_qa_memory", [])
    if not isinstance(memory, list):
        checkpoint.metadata["resonance_qa_memory"] = []
        memory = checkpoint.metadata["resonance_qa_memory"]

    item = {
        "question": question,
        "template_key": _qa_template_key(question, lang, plane_id),
        "question_tokens": tokenize(question),
        "question_concepts": list(dict.fromkeys(question_concepts)),
        "answer": _render_sentence(answer_slots),
        "sentence_end": sentence_end,
        "answer_slots": [dict(slot) for slot in answer_slots],
        "answer_variants": [
            {
                "answer": _render_sentence(answer_slots),
                "sentence_end": sentence_end,
                "answer_slots": [dict(slot) for slot in answer_slots],
                "reward": reward,
                "created_at": time.time(),
            }
        ],
        "lang": lang,
        "plane_id": plane_id,
        "reward": reward,
        "created_at": time.time(),
    }

    for existing in memory:
        if not isinstance(existing, dict):
            continue
        if str(existing.get("question", "")).casefold() != question.casefold():
            continue
        variants = existing.setdefault("answer_variants", [])
        if not isinstance(variants, list):
            variants = []
            existing["answer_variants"] = variants
        answer_text = str(item.get("answer") or "")
        variant = item["answer_variants"][0]
        matched = False
        for existing_variant in variants:
            if not isinstance(existing_variant, dict):
                continue
            if str(existing_variant.get("answer", "")).casefold() == answer_text.casefold():
                existing_variant.update(variant)
                matched = True
                break
        if not matched:
            variants.append(variant)
        if float(item.get("reward", 0.0)) >= float(existing.get("reward", 0.0)):
            existing.update({k: v for k, v in item.items() if k != "answer_variants"})
        break
    else:
        memory.append(item)
    del memory[:-200]


def _slots_from_qa_memory(
    checkpoint: Checkpoint,
    *,
    text: str,
    lang: str,
    session_id: str,
    plane_id: str,
    creativity: float = 0.35,
) -> list[dict[str, Any]]:
    memory = checkpoint.metadata.get("resonance_qa_memory", [])
    if not isinstance(memory, list) or not memory:
        return []

    question_tokens = set(tokenize(text))
    question_concepts = set(_question_concepts(checkpoint, text, lang))
    session = checkpoint.session_contexts.get(session_id, {})
    active_plane = plane_id or _infer_plane(text, session)

    scored_items: list[tuple[float, dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]] = []
    for item in memory:
        if not isinstance(item, dict):
            continue
        template_key = str(item.get("template_key") or _qa_template_key(str(item.get("question") or ""), str(item.get("lang") or lang), str(item.get("plane_id") or active_plane)))
        penalty = float(_qa_penalty(checkpoint, template_key))
        if penalty > 0:
            continue
        item_concepts = {str(value) for value in item.get("question_concepts", []) if value}
        item_tokens = {str(value) for value in item.get("question_tokens", []) if value}
        base_score = float(len(question_concepts & item_concepts) * 4 + len(question_tokens & item_tokens))
        if base_score <= 0:
            continue
        if str(item.get("lang") or "") == lang:
            base_score += 1.0
        if str(item.get("plane_id") or "") == active_plane:
            base_score += 1.0
        variants = item.get("answer_variants")
        variant_items: list[dict[str, Any]] = []
        if isinstance(variants, list) and variants:
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                slots = [dict(slot) for slot in variant.get("answer_slots", []) if isinstance(slot, dict)]
                if slots:
                    variant_items.append(variant)
                    scored_items.append((base_score, item, variant, slots))
        if not variant_items:
            slots = [dict(slot) for slot in item.get("answer_slots", []) if isinstance(slot, dict)]
            if slots:
                scored_items.append((base_score, item, None, slots))

    if not scored_items:
        return []

    scored_items.sort(key=lambda value: value[0], reverse=True)
    best_score = float(scored_items[0][0])
    if best_score < 1.0:
        return []
    top_items = [entry for entry in scored_items if entry[0] >= max(best_score * 0.8, best_score - 1.0)]
    slot_sets = [entry[3] for entry in top_items]
    if len(slot_sets) > 1 and creativity >= 0.25:
        slots = _blend_qa_slots(slot_sets, text, session_id, plane_id, checkpoint)
    else:
        slots = slot_sets[0]
    if not slots:
        return []
    best_entry = top_items[_qa_variant_index(text, session_id, plane_id, len(top_items), checkpoint, salt="answer")]
    best_item = best_entry[1]
    best_variant = best_entry[2]
    sentence_end = str((best_variant or best_item).get("sentence_end") or "")
    if sentence_end:
        slots[-1]["sentence_end"] = sentence_end
    return slots


def _qa_variant_index(
    text: str,
    session_id: str,
    plane_id: str,
    variant_count: int,
    checkpoint: Checkpoint,
    salt: str = "",
) -> int:
    if variant_count <= 1:
        return 0
    history = checkpoint.session_history(session_id)
    assistant_turns = sum(1 for turn in history if isinstance(turn, dict) and str(turn.get("role", "")) == "assistant")
    user_turns = sum(1 for turn in history if isinstance(turn, dict) and str(turn.get("role", "")) == "user")
    payload = f"{' '.join(tokenize(text)).casefold()}|{session_id}|{plane_id}|{salt}|{variant_count}"
    base = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12], 16) % variant_count
    return (base + assistant_turns + user_turns) % variant_count


def _blend_qa_slots(
    slot_sets: list[list[dict[str, Any]]],
    text: str,
    session_id: str,
    plane_id: str,
    checkpoint: Checkpoint,
) -> list[dict[str, Any]]:
    max_length = max((len(slots) for slots in slot_sets), default=0)
    if max_length <= 0:
        return []
    blended: list[dict[str, Any]] = []
    for index in range(max_length):
        options = [slots[index] if index < len(slots) else slots[-1] for slots in slot_sets if slots]
        if not options:
            continue
        chosen = options[
            _qa_variant_index(
                f"{text}:{index}",
                session_id,
                plane_id,
                len(options),
                checkpoint,
                salt=f"{index}:{len(options)}",
            )
        ]
        blended.append(dict(chosen))
    if blended:
        sentence_end = next((str(slot.get("sentence_end") or "") for slot in reversed(blended) if slot.get("sentence_end")), "")
        if sentence_end:
            blended[-1]["sentence_end"] = sentence_end
    return blended


def _resolve_slot(
    checkpoint: Checkpoint,
    slot: dict[str, Any],
    *,
    lang: str,
    plane_id: str,
    session_id: str,
) -> dict[str, Any]:
    lemma = str(slot.get("lemma", "")).strip().lower()
    if not lemma:
        lemma = "неизвестное"
    role = str(slot.get("role") or "subject")
    pos = str(slot.get("pos") or ("VERB" if role == "predicate" else "NOUN"))
    gram = _required_gram(slot, role)
    concept = _concept_from_hint(checkpoint, str(slot.get("concept") or slot.get("concept_uri") or ""), lemma, lang)
    exact = _best_exact_form(checkpoint, lang, lemma, gram, role)
    if exact:
        return {
            **exact,
            "slot_id": str(slot.get("id") or role),
            "role": role,
            "source": "exact",
            "transform_status": "known_form",
            "tentative": bool(exact.get("tentative")),
            "preposition": str(slot.get("preposition") or ""),
            "sentence_end": str(slot.get("sentence_end") or ""),
            "resonance_score": _form_score(checkpoint, exact, plane_id, session_id),
        }
    generated = _analogical_form(checkpoint, lang, lemma, pos, gram)
    form = _register_form(
        checkpoint,
        lang=lang,
        lemma=lemma,
        pos=pos,
        surface=generated["surface"],
        gram=gram,
        role=role,
        plane_id=f"session:{session_id}",
        reward=0.35,
        tentative=True,
        pattern_id=str(generated.get("pattern_id", "")),
        concept_uri=concept,
    )
    session = checkpoint.session_contexts.setdefault(session_id, {"session_id": session_id})
    tentative_forms = session.setdefault("tentative_forms", [])
    if isinstance(tentative_forms, list):
        tentative_forms.append(form)
        del tentative_forms[:-50]
    return {
        **form,
        "slot_id": str(slot.get("id") or role),
        "role": role,
        "source": "analogy",
        "transform_status": "analogical_transfer",
        "tentative": True,
        "preposition": str(slot.get("preposition") or ""),
        "sentence_end": str(slot.get("sentence_end") or ""),
        "resonance_score": _form_score(checkpoint, form, plane_id, session_id),
    }


def _best_exact_form(
    checkpoint: Checkpoint,
    lang: str,
    lemma: str,
    required_gram: dict[str, str],
    role: str,
) -> dict[str, Any] | None:
    candidates = []
    for form in checkpoint.morph_forms.get(_form_key(lang, lemma), []):
        gram = dict(form.get("gram", {}))
        if not _gram_matches(gram, required_gram):
            continue
        role_bonus = 0.4 if not role or form.get("role") == role else 0.0
        candidates.append((float(form.get("pheromone", 1.0)) + role_bonus, form))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return dict(candidates[0][1])


def _analogical_form(checkpoint: Checkpoint, lang: str, lemma: str, pos: str, gram: dict[str, str]) -> dict[str, Any]:
    scored = []
    for pattern_id, pattern in checkpoint.morph_patterns.items():
        if str(pattern.get("lang")) != lang or str(pattern.get("pos")) != pos:
            continue
        if not _gram_matches(dict(pattern.get("gram", {})), gram):
            continue
        suffix = str(pattern.get("from_suffix", ""))
        suffix_score = len(suffix) if suffix and lemma.endswith(suffix) else (0.5 if suffix == "" else 0.0)
        if suffix_score <= 0:
            continue
        scored.append((float(pattern.get("pheromone", 1.0)) + suffix_score, pattern_id, pattern))
    if not scored:
        fallback_surface = lemma + ("ы" if pos == "NOUN" and gram.get("number") == "plur" else "")
        return {"surface": fallback_surface, "pattern_id": "fallback"}
    scored.sort(key=lambda item: item[0], reverse=True)
    _, pattern_id, pattern = scored[0]
    from_suffix = str(pattern.get("from_suffix", ""))
    to_suffix = str(pattern.get("to_suffix", ""))
    stem = lemma[: -len(from_suffix)] if from_suffix and lemma.endswith(from_suffix) else lemma
    return {"surface": stem + to_suffix, "pattern_id": pattern_id}


def _learn_pattern(
    checkpoint: Checkpoint,
    lang: str,
    lemma: str,
    surface: str,
    pos: str,
    gram: dict[str, str],
    *,
    reward: float,
) -> None:
    prefix = _shared_prefix(lemma, surface)
    from_suffix = lemma[len(prefix) :]
    to_suffix = surface[len(prefix) :]
    pattern_id = f"{lang}:{pos}:{_gram_signature(gram)}:{from_suffix}>{to_suffix}"
    pattern = checkpoint.morph_patterns.setdefault(
        pattern_id,
        {
            "id": pattern_id,
            "lang": lang,
            "pos": pos,
            "gram": dict(gram),
            "from_suffix": from_suffix,
            "to_suffix": to_suffix,
            "pheromone": 0.0,
            "examples": [],
        },
    )
    pattern["pheromone"] = float(pattern.get("pheromone", 0.0)) + max(reward, 0.1)
    examples = pattern.setdefault("examples", [])
    if isinstance(examples, list):
        example = {"lemma": lemma, "surface": surface}
        if example not in examples:
            examples.append(example)


def _route_for_token(index: int, token: dict[str, Any], plane_id: str) -> dict[str, Any]:
    concept = str(token.get("concept") or "")
    form_uri = str(token.get("form_uri") or "")
    role_uri = _role_uri(str(token.get("role") or "subject"))
    steps = [
        _step(index, concept, "HasForm", form_uri, 2, plane_id, float(token.get("resonance_score", 1.0) or 1.0)),
        _step(index, form_uri, "RealizesRole", role_uri, 3, plane_id, float(token.get("resonance_score", 1.0) or 1.0) * 0.8),
    ]
    return {
        "ant_id": index,
        "start": concept,
        "concepts": [value for value in [concept, form_uri, role_uri] if value],
        "total_score": sum(float(step["score"]) for step in steps),
        "steps": steps,
    }


def _step(index: int, start: str, relation: str, end: str, layer: int, plane_id: str, score: float) -> dict[str, Any]:
    return {
        "ant_id": index,
        "step_index": len(relation) + index,
        "start": start,
        "end": end,
        "relation": relation,
        "edge_weight": max(score, 0.1),
        "pheromone": max(score, 0.1),
        "score": round(max(score, 0.0001), 4),
        "source": "resonance",
        "layer": layer,
        "from_layer": layer - 1,
        "to_layer": layer,
        "context_plane": plane_id,
        "layer_pheromone": max(score, 0.1),
        "projection_shift": 1.0,
        "distance": max(1.0 / max(score, 0.1), 0.01),
        "remaining_strength": None,
        "edge_type": "morphology" if layer == 2 else "syntax",
    }


def _remember_resonance_result(
    checkpoint: Checkpoint,
    result: dict[str, Any],
    tokens: list[dict[str, Any]],
    plane_id: str,
    session_id: str,
    text: str,
) -> None:
    checkpoint.remember_result(result)
    checkpoint.remember_chat_turn(session_id, "user", text or "[structured resonance]", result["result_id"])
    checkpoint.remember_chat_turn(
        session_id,
        "assistant",
        str(result.get("response", "")),
        result["result_id"],
        [str(token.get("concept", "")) for token in tokens if token.get("concept")],
    )
    session = checkpoint.session_contexts.setdefault(session_id, {"session_id": session_id})
    session["active_plane"] = plane_id
    session["last_result_id"] = result["result_id"]
    session["updated_at"] = time.time()
    session["active_concepts"] = [str(token.get("concept", "")) for token in tokens if token.get("concept")]
    _bump_context(checkpoint, session_id, plane_id, amount=0.2)


def _activated_concepts(checkpoint: Checkpoint, tokens: list[dict[str, Any]], plane_id: str) -> list[dict[str, Any]]:
    values = []
    seen = set()
    for token in tokens:
        for uri_key in ("concept", "form_uri"):
            uri = str(token.get(uri_key) or "")
            if not uri or uri in seen:
                continue
            seen.add(uri)
            values.append(
                {
                    "uri": uri,
                    "label": _label(checkpoint, uri),
                    "language": "ru",
                    "layer": 2 if uri_key == "form_uri" else 1,
                    "layers": [1, 2, 3],
                    "active_layers": [2, 3],
                    "score": round(float(token.get("resonance_score", 1.0) or 1.0), 4),
                    "sources": ["resonance", plane_id],
                }
            )
    return values


def _slots_from_text(text: str, lang: str, plane_id: str | None) -> list[dict[str, Any]]:
    tokens = tokenize(text)
    token_set = set(tokens)
    if {"программист", "код", "компьютер"} <= token_set or "компьютере" in token_set:
        return [
            {"id": "subject", "lemma": "программист", "role": "subject", "pos": "NOUN", "gram": {"case": "nomn", "number": "sing"}},
            {"id": "predicate", "lemma": "писать", "role": "predicate", "pos": "VERB", "gram": {"tense": "pres", "person": "3", "number": "sing"}},
            {"id": "object", "lemma": "код", "role": "object", "pos": "NOUN", "gram": {"case": "accs", "number": "sing"}},
            {
                "id": "instrument",
                "lemma": "компьютер",
                "role": "instrument",
                "pos": "NOUN",
                "gram": {"case": "loct", "number": "sing"},
                "preposition": "на",
            },
        ]
    if "деревья" in token_set or ("дерево" in token_set and any(value in token_set for value in ("растут", "расти", "растет"))):
        return [
            {"id": "subject", "lemma": "дерево", "role": "subject", "pos": "NOUN", "gram": {"case": "nomn", "number": "plur"}},
            {"id": "predicate", "lemma": "расти", "role": "predicate", "pos": "VERB", "gram": {"tense": "pres", "person": "3", "number": "plur"}},
        ]
    lemma = next((token for token in tokens if token not in _CONTROL_TOKENS), "дерево")
    number = "plur" if any(token in _PLURAL_HINTS for token in tokens) else "sing"
    role = "subject" if any(token in {"subject", "подлежащее"} for token in tokens) or not tokens else "subject"
    return [{"id": role, "lemma": lemma, "role": role, "pos": "NOUN", "gram": {"case": "nomn", "number": number}}]


def _infer_plane(text: str, session: dict[str, Any]) -> str:
    tokens = set(tokenize(text))
    if tokens & {"файл", "файлы", "папка", "папки", "корень", "filesystem", "операционная", "система"}:
        return "dev:filesystem"
    if tokens & {"растение", "растения", "лес", "ветка", "природа"}:
        return "semantic:nature"
    if tokens & {"subject", "подлежащее", "множественное", "число", "plur", "форма", "словоформа"}:
        return "language:ru"
    previous = str(session.get("active_plane") or "")
    return previous or "language:ru"


def _order_slots(checkpoint: Checkpoint, slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"subject": 0, "predicate": 1, "object": 2, "instrument": 3, "location": 4, "modifier": 5}
    return sorted(slots, key=lambda slot: order.get(str(slot.get("role") or ""), 99))


def _render_sentence(tokens: list[dict[str, Any]]) -> str:
    values = []
    sentence_end = ""
    for token in tokens:
        surface = str(token.get("surface") or token.get("lemma") or "").strip()
        preposition = str(token.get("preposition") or "").strip()
        if preposition and not surface.startswith(preposition + " "):
            surface = f"{preposition} {surface}"
        if surface:
            values.append(surface)
            sentence_end = str(token.get("sentence_end") or sentence_end)
    sentence = " ".join(values)
    if sentence and sentence_end and not sentence.endswith(sentence_end):
        sentence = f"{sentence}{sentence_end}"
    return sentence


def _required_gram(slot: dict[str, Any], role: str) -> dict[str, str]:
    gram = _gram(slot.get("gram"))
    if role == "subject":
        gram["case"] = "nomn"
    elif role == "object":
        gram["case"] = "accs"
    elif role in {"instrument", "location"}:
        gram["case"] = "loct"
    elif role == "predicate":
        gram.setdefault("tense", "pres")
        gram.setdefault("person", "3")
    return gram


def _form_score(checkpoint: Checkpoint, form: dict[str, Any], plane_id: str, session_id: str) -> float:
    concept = str(form.get("concept") or "")
    form_uri = str(form.get("form_uri") or "")
    distance = effective_distance(checkpoint, concept, form_uri, plane_id)
    plane_bonus = float(checkpoint.plane_pheromones.get(plane_id, 1.0))
    context_bonus = float(checkpoint.context_plane_pheromones.get(_context_key(session_id, plane_id), 1.0))
    suppression = float(checkpoint.suppressed_concepts.get(form_uri, 0.0))
    return (float(form.get("pheromone", 1.0)) * plane_bonus * context_bonus) / max(distance * (1.0 + suppression), 0.01)


def _analyze_training_token(token: str, lang: str) -> dict[str, Any]:
    normalized = token.casefold()
    if lang == "ru":
        parsed = _parse_ru(normalized)
        if parsed is not None:
            tag = getattr(parsed, "tag", None)
            pos = _normalize_pos(str(getattr(tag, "POS", "") or ""))
            gram = {
                key: value
                for key, value in {
                    "case": _normalize_gram_value(str(getattr(tag, "case", "") or "")),
                    "number": _normalize_gram_value(str(getattr(tag, "number", "") or "")),
                    "gender": _normalize_gram_value(str(getattr(tag, "gender", "") or "")),
                    "tense": _normalize_gram_value(str(getattr(tag, "tense", "") or "")),
                    "person": _normalize_gram_value(str(getattr(tag, "person", "") or "")),
                }.items()
                if value
            }
            return {
                "lemma": str(getattr(parsed, "normal_form", "") or normalized).casefold(),
                "pos": pos or _fallback_pos(normalized),
                "gram": gram,
            }
    return {
        "lemma": _RU_LEMMA_OVERRIDES.get(normalized, normalized),
        "pos": _fallback_pos(normalized),
        "gram": dict(_RU_GRAM_OVERRIDES.get(normalized, {})),
    }


def _parse_ru(token: str) -> Any:
    parsed = _morph().parse(token)
    return parsed[0] if parsed else None


@lru_cache(maxsize=1)
def _morph() -> Any:
    if pymorphy3 is None:
        return _FallbackMorphAnalyzer()
    return pymorphy3.MorphAnalyzer()


class _FallbackMorphAnalyzer:
    def parse(self, token: str) -> list[Any]:
        return []


def _normalize_pos(pos: str) -> str:
    if pos in {"VERB", "INFN"}:
        return "VERB"
    if pos in {"NOUN", "NPRO"}:
        return "NOUN"
    if pos in {"ADJF", "ADJS", "COMP"}:
        return "ADJ"
    return pos or ""


def _normalize_gram_value(value: str) -> str:
    if value in {"1per", "2per", "3per"}:
        return value[0]
    return value


def _fallback_pos(token: str) -> str:
    if token in _RU_VERB_LEMMAS or token in _RU_VERB_SURFACES:
        return "VERB"
    return "NOUN"


def _infer_training_role(*, index: int, predicate_index: int, pos: str, preposition: str) -> str:
    if pos == "VERB":
        return "predicate"
    if preposition in {"на", "в", "о", "об", "при"}:
        return "instrument"
    if predicate_index >= 0 and index < predicate_index:
        return "subject"
    if predicate_index >= 0 and index > predicate_index:
        return "object"
    return "subject"


def _annotation_gram(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return _gram(parsed)
    return _gram(value)


def _planes_from_annotation(annotation: dict[str, Any]) -> list[str]:
    raw = annotation.get("planes")
    if raw is None:
        raw = annotation.get("plane_ids")
    if raw is None:
        raw = annotation.get("plane_id")
    if raw is None:
        raw = annotation.get("plane")
    if isinstance(raw, str):
        parts = raw.replace(";", ",").split(",")
    elif isinstance(raw, list):
        parts = [str(item) for item in raw]
    else:
        parts = []
    return list(dict.fromkeys(part.strip() for part in parts if part.strip()))


def _is_new_concept(checkpoint: Checkpoint, slot: dict[str, Any], lang: str) -> bool:
    lemma = str(slot.get("lemma") or slot.get("token") or "").strip()
    if not lemma:
        return False
    concept = _concept_from_hint(checkpoint, str(slot.get("concept") or ""), lemma, lang)
    if checkpoint.morph_forms.get(_form_key(lang, lemma)):
        return False
    for edge in checkpoint.learned_edges():
        if concept in {edge.start, edge.end}:
            return False
    return True


def _learned_plane_id(label: str, lang: str) -> str:
    slug = "_".join(tokenize(label)) or re.sub(r"[^a-z0-9_]+", "_", label.casefold()).strip("_") or "concept"
    return f"semantic:learned:{lang}:{slug}"


def _area_label_for_plane(plane_id: str) -> str:
    if plane_id.startswith("semantic:learned:"):
        return f"Новая область {plane_id.rsplit(':', 1)[-1]}"
    return f"Область {plane_id}"


def _concept_from_hint(checkpoint: Checkpoint, hint: str, lemma: str, lang: str) -> str:
    value = str(hint or "").strip()
    if value.startswith("/"):
        return checkpoint.canonical_uri(value)
    if value:
        return _concept_uri(value, lang)
    return _concept_uri(lemma, lang)


def _add_plane_edge(
    checkpoint: Checkpoint,
    start: str,
    end: str,
    relation: str,
    *,
    plane_id: str,
    distance: float,
    weight: float,
    edge_type: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    _upsert_plane(checkpoint, plane_id, plane_id)
    checkpoint.add_custom_edge(
        start,
        end,
        relation=relation,
        weight=weight,
        layer=_layer_for_edge(edge_type),
        from_layer=1,
        to_layer=_layer_for_edge(edge_type),
        context_plane=plane_id,
        distance=distance,
        edge_type=edge_type,
        metadata={**(metadata or {}), "plane_id": plane_id},
    )
    start = checkpoint.canonical_uri(start)
    end = checkpoint.canonical_uri(end)
    checkpoint.reinforce_edge(start, relation, end, amount=max(weight * 0.2, 0.1), context_plane=plane_id)
    checkpoint.plane_distance_overrides[_distance_key(plane_id, start, end)] = min(
        float(checkpoint.plane_distance_overrides.get(_distance_key(plane_id, start, end), distance)),
        distance,
    )


def _upsert_plane(checkpoint: Checkpoint, plane_id: str, label: str) -> None:
    checkpoint.planes.setdefault(
        plane_id,
        {
            "plane_id": plane_id,
            "label": label,
            "created_at": time.time(),
        },
    )
    checkpoint.plane_pheromones.setdefault(plane_id, 1.0)


def _upsert_area(checkpoint: Checkpoint, area_id: str, plane_id: str, label: str) -> None:
    checkpoint.areas.setdefault(
        area_id,
        {
            "area_id": area_id,
            "plane_id": plane_id,
            "label": label,
            "created_at": time.time(),
        },
    )


def _add_area_membership(checkpoint: Checkpoint, area_id: str, node_uri: str, plane_id: str, weight: float) -> None:
    checkpoint.area_memberships[_area_key(area_id, node_uri, plane_id)] = max(
        float(checkpoint.area_memberships.get(_area_key(area_id, node_uri, plane_id), 0.0)),
        weight,
    )


def _remember_label(checkpoint: Checkpoint, uri: str, label: str) -> None:
    checkpoint.register_canonical_concept(uri, label=label, aliases=[label], lang=detect_language(label), source_uri=uri)
    labels = checkpoint.metadata.setdefault("concept_labels", {})
    if isinstance(labels, dict):
        labels[checkpoint.canonical_uri(uri)] = label


def _qa_template_key(question: str, lang: str, plane_id: str) -> str:
    normalized = " ".join(tokenize(question))
    payload = f"{lang}|{plane_id}|{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _qa_penalty(checkpoint: Checkpoint, template_key: str) -> float:
    penalties = checkpoint.metadata.get("resonance_qa_penalties", {})
    if not isinstance(penalties, dict):
        return 0.0
    return max(float(penalties.get(template_key, 0.0)), 0.0)


def _penalize_qa_template(checkpoint: Checkpoint, template_key: str, amount: float) -> None:
    penalties = checkpoint.metadata.setdefault("resonance_qa_penalties", {})
    if not isinstance(penalties, dict):
        checkpoint.metadata["resonance_qa_penalties"] = {}
        penalties = checkpoint.metadata["resonance_qa_penalties"]
    penalties[template_key] = max(float(penalties.get(template_key, 0.0)) + max(amount, 0.1), 0.0)


def _surface_from_existing(checkpoint: Checkpoint, lemma: str, lang: str, gram: dict[str, str]) -> str:
    form = _best_exact_form(checkpoint, lang, lemma, gram, "")
    return str(form.get("surface", "")) if form else ""


def _commit_tentative_form(checkpoint: Checkpoint, token: dict[str, Any]) -> None:
    lang = str(token.get("lang") or DEFAULT_LANG)
    lemma = str(token.get("lemma") or "")
    surface = str(token.get("surface") or "")
    for form in checkpoint.morph_forms.get(_form_key(lang, lemma), []):
        if form.get("surface") == surface:
            form["tentative"] = False
            form["pheromone"] = float(form.get("pheromone", 1.0)) + 1.0


def _shorten_distance(checkpoint: Checkpoint, plane_id: str, start: str, end: str, ratio: float) -> None:
    if not start or not end:
        return
    key = _distance_key(plane_id, checkpoint.canonical_uri(start), checkpoint.canonical_uri(end))
    checkpoint.plane_distance_overrides[key] = max(float(checkpoint.plane_distance_overrides.get(key, 1.0)) * ratio, 0.03)


def _lengthen_distance(checkpoint: Checkpoint, plane_id: str, start: str, end: str, ratio: float) -> None:
    if not start or not end:
        return
    key = _distance_key(plane_id, checkpoint.canonical_uri(start), checkpoint.canonical_uri(end))
    checkpoint.plane_distance_overrides[key] = min(float(checkpoint.plane_distance_overrides.get(key, 1.0)) * ratio, 12.0)


def _bump_context(checkpoint: Checkpoint, session_id: str, plane_id: str, amount: float) -> None:
    key = _context_key(session_id, plane_id)
    checkpoint.context_plane_pheromones[key] = max(float(checkpoint.context_plane_pheromones.get(key, 1.0)) + amount, 0.05)


def _gram_matches(candidate: dict[str, Any], required: dict[str, str]) -> bool:
    return all(str(candidate.get(key) or "") == str(value) for key, value in required.items() if value)


def _gram(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if item not in {None, ""}}


def _lang(payload: dict[str, Any]) -> str:
    lang = str(payload.get("lang") or "auto")
    if lang in {"ru", "en"}:
        return lang
    text = str(payload.get("text") or payload.get("sentence") or "")
    return detect_language(text) if text else DEFAULT_LANG


def _concept_uri(text: str, lang: str = DEFAULT_LANG) -> str:
    return Checkpoint().canonical_uri(text_to_concept_uri(text, lang))


def _form_uri(surface: str, lang: str = DEFAULT_LANG, *, lemma: str = "", concept_uri: str = "") -> str:
    if lemma and " ".join(tokenize(surface)).casefold() == " ".join(tokenize(lemma)).casefold():
        return concept_uri or _concept_uri(lemma, lang)
    slug = "_".join(tokenize(surface)) or "empty"
    return f"/morph/{lang}/form/{slug}"


def _gram_uri(value: str, lang: str = DEFAULT_LANG) -> str:
    return f"/gram/{lang}/{value}"


def _role_uri(role: str, lang: str = DEFAULT_LANG) -> str:
    return f"/role/{lang}/{role}"


def _form_key(lang: str, lemma: str) -> str:
    return f"{lang}:{lemma.casefold()}"


def _distance_key(plane_id: str, start: str, end: str) -> str:
    return json.dumps({"plane_id": plane_id, "start": start, "end": end}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _context_key(session_id: str, plane_id: str) -> str:
    return json.dumps({"session_id": session_id, "plane_id": plane_id}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _area_key(area_id: str, node_uri: str, plane_id: str) -> str:
    return json.dumps({"area_id": area_id, "node_uri": node_uri, "plane_id": plane_id}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _gram_signature(gram: dict[str, str]) -> str:
    return ",".join(f"{key}={gram[key]}" for key in sorted(gram))


def _shared_prefix(left: str, right: str) -> str:
    index = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        index += 1
    return left[:index]


def _layer_for_edge(edge_type: str) -> int:
    if edge_type == "syntax":
        return 3
    if edge_type == "plane":
        return 4
    return 2


def _area_for_plane(plane_id: str) -> str:
    if plane_id == "semantic:nature":
        return "area:semantic:nature/tree"
    if plane_id == "language:ru":
        return "area:language:ru/tree_subject"
    if plane_id == "dev:filesystem":
        return "area:dev:filesystem/tree"
    if plane_id.startswith("semantic:learned:"):
        return f"area:{_safe_plane_slug(plane_id)}/concepts"
    if plane_id.startswith("morphology:"):
        return "area:morphology:ru/tree_forms"
    return "area:morphology:ru/tree_forms"


def _safe_plane_slug(plane_id: str) -> str:
    return re.sub(r"[^0-9a-zа-яё_:-]+", "_", plane_id.casefold()).strip("_") or "plane"


def _auto_area_id(plane_id: str, lemma: str, role: str, concept: str) -> str:
    anchor = concept or lemma or role or "context"
    slug = "_".join(tokenize(anchor)) or re.sub(r"[^0-9a-zа-яё_]+", "_", anchor.casefold()).strip("_") or "context"
    return f"area:{_safe_plane_slug(plane_id)}/{slug}"


def _auto_area_label(area_id: str, lemma: str, role: str) -> str:
    if lemma:
        return lemma
    if role:
        return role
    return area_id.rsplit("/", 1)[-1].replace("_", " ")


def _label(checkpoint: Checkpoint, uri: str) -> str:
    labels = checkpoint.metadata.get("concept_labels", {})
    if isinstance(labels, dict) and labels.get(uri):
        return str(labels[uri])
    canonical = checkpoint.canonical_uri(uri)
    if isinstance(labels, dict) and labels.get(canonical):
        return str(labels[canonical])
    return uri.rsplit("/", 1)[-1].replace("_", " ")


def _result_id(text: str) -> str:
    return hashlib.sha256(f"{time.time_ns()}:{text}".encode("utf-8")).hexdigest()[:16]


_PLURAL_HINTS = {"plur", "plural", "множественное", "много", "несколько"}
_CONTROL_TOKENS = _PLURAL_HINTS | {"subject", "подлежащее", "форма", "словоформа", "число"}
_RU_PREPOSITIONS = {"в", "во", "на", "с", "со", "к", "ко", "по", "о", "об", "при", "из", "за", "для", "у"}
_QA_STOP_WORDS = {
    "а",
    "в",
    "и",
    "или",
    "на",
    "с",
    "что",
    "кто",
    "где",
    "как",
    "зачем",
    "почему",
    "какой",
    "какая",
    "какое",
    "какие",
    "делает",
    "такое",
    "это",
    "the",
    "a",
    "an",
    "and",
    "to",
    "of",
}
_RU_VERB_SURFACES = {"пишет", "растет", "растёт", "растут", "делает"}
_RU_VERB_LEMMAS = {"писать", "расти", "делать"}
_RU_LEMMA_OVERRIDES = {
    "деревья": "дерево",
    "деревом": "дерево",
    "деревьями": "дерево",
    "столы": "стол",
    "столом": "стол",
    "столами": "стол",
    "машины": "машина",
    "машиной": "машина",
    "машинами": "машина",
    "пишет": "писать",
    "растет": "расти",
    "растёт": "расти",
    "растут": "расти",
    "делает": "делать",
    "компьютере": "компьютер",
}
_RU_GRAM_OVERRIDES = {
    "деревья": {"case": "nomn", "number": "plur", "gender": "neut"},
    "деревом": {"case": "ablt", "number": "sing", "gender": "neut"},
    "деревьями": {"case": "ablt", "number": "plur", "gender": "neut"},
    "пишет": {"tense": "pres", "person": "3", "number": "sing"},
    "растет": {"tense": "pres", "person": "3", "number": "sing"},
    "растёт": {"tense": "pres", "person": "3", "number": "sing"},
    "растут": {"tense": "pres", "person": "3", "number": "plur"},
    "делает": {"tense": "pres", "person": "3", "number": "sing"},
    "компьютере": {"case": "loct", "number": "sing", "gender": "masc"},
}
