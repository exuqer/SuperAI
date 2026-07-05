from __future__ import annotations

import hashlib
from typing import Any

from semantic_ants.core.normalization import detect_language
from semantic_ants.learning.checkpoint import Checkpoint


class SenseSentenceBuilder:
    def build_candidates(self, semantic_vector: dict[str, Any], checkpoint: Checkpoint, count: int = 3) -> list[str]:
        lang = _vector_lang(semantic_vector)
        items = [item for item in semantic_vector.get("items", []) if isinstance(item, dict)]
        if _is_sparse_unknown_vector(items):
            return [_fallback(lang)]
        subject = _surface_for_item(_best_item(items, checkpoint, lang), checkpoint, lang)
        definition = _meaning_for_subject(items, checkpoint, lang)
        definition_candidate = _variant_definition(lang, subject, definition)
        items = _ordered_surfaces(items, checkpoint, lang, subject)
        domain = _surface_for_uri(_domain_uri(semantic_vector), checkpoint, lang)
        vectors = _nonempty([subject, *items[:5], domain])
        if not vectors:
            return [_fallback(lang)]

        candidates = _unique_nonempty(
            [
                definition_candidate,
                _variant_is_a(lang, subject, items, domain),
                _variant_source(lang, subject, items, domain),
                _variant_related(lang, subject, items, domain),
                _variant_image(lang, subject, items, domain),
            ]
        )
        if not candidates:
            candidates = [_fallback(lang)]
        return candidates[: max(count, 1)]

    def render_pattern(
        self,
        concepts: list[str],
        checkpoint: Checkpoint,
        lang: str | None = None,
        reward: float = 0.0,
        count: int = 3,
    ) -> list[str]:
        selected_lang = lang if lang in {"ru", "en"} else _concept_lang(concepts) or "auto"
        surfaces = _concept_surfaces(concepts, checkpoint, selected_lang)
        if not surfaces:
            return []
        seed = int(abs(reward) * 10) % 4
        subject = surfaces[0]
        rest = _nonempty(surfaces[1:])
        domain = _domain_surface_from_concepts(concepts, checkpoint, selected_lang)
        candidates = _unique_nonempty(
            [
                _phrase_is_a(selected_lang, subject, rest, domain, seed),
                _phrase_source(selected_lang, subject, rest, domain, seed + 1),
                _phrase_related(selected_lang, subject, rest, domain, seed + 2),
                _phrase_image(selected_lang, subject, rest, domain, seed + 3),
            ]
        )
        return candidates[: max(count, 1)]


def build_vector_candidates(semantic_vector: dict[str, Any], checkpoint: Checkpoint, count: int = 3) -> list[str]:
    return SenseSentenceBuilder().build_candidates(semantic_vector, checkpoint, count=count)


def render_concept_pattern(
    concepts: list[str],
    checkpoint: Checkpoint,
    lang: str | None = None,
    reward: float = 0.0,
    count: int = 3,
) -> list[str]:
    return SenseSentenceBuilder().render_pattern(concepts, checkpoint, lang=lang, reward=reward, count=count)


def render_uri(uri: str, checkpoint: Checkpoint, lang: str | None = None) -> str:
    if not uri:
        return ""
    selected_lang = lang if lang in {"ru", "en"} else None
    localized = _definition_label(uri, checkpoint, selected_lang)
    if localized:
        return localized
    if uri.startswith("/c/"):
        parts = uri.split("/", 3)
        if len(parts) > 3:
            uri_lang = parts[2]
            token = parts[3].replace("_", " ")
            if selected_lang is None or uri_lang == selected_lang:
                return token
            if uri_lang in {"ru", "en"}:
                return _surface_from_aliases(uri, checkpoint, selected_lang) or token
            return token
    return _surface_from_aliases(uri, checkpoint, selected_lang) or uri.rstrip("/").split("/")[-1].replace("_", " ")


def render_concepts(concepts: list[str], checkpoint: Checkpoint, lang: str | None = None) -> list[str]:
    selected_lang = lang if lang in {"ru", "en"} else _concept_lang(concepts) or "auto"
    return _concept_surfaces(concepts, checkpoint, selected_lang)


def _vector_lang(semantic_vector: dict[str, Any]) -> str:
    lang = str(semantic_vector.get("lang", "auto"))
    if lang in {"ru", "en"}:
        return lang
    text = str(semantic_vector.get("input_text", ""))
    return detect_language(text)


def _best_item(items: list[dict[str, Any]], checkpoint: Checkpoint, lang: str) -> dict[str, Any] | None:
    if not items:
        return None
    preferred = [item for item in items if _item_matches_lang(item, lang)]
    ordered = preferred or items
    ordered = [item for item in ordered if _item_score(item) > 0.05]
    return ordered[0] if ordered else items[0]


def _ordered_surfaces(
    items: list[dict[str, Any]],
    checkpoint: Checkpoint,
    lang: str,
    subject: str,
) -> list[str]:
    ordered = sorted(items, key=lambda item: _item_score(item), reverse=True)
    values: list[str] = []
    for item in ordered:
        surface = _surface_for_item(item, checkpoint, lang)
        if not surface or surface == subject:
            continue
        if not _looks_like_content(surface, lang):
            continue
        values.append(surface)
    return _nonempty(values)


def _surface_for_item(item: dict[str, Any] | None, checkpoint: Checkpoint, lang: str) -> str:
    if not item:
        return ""
    uri = str(item.get("uri", ""))
    if uri.startswith("/c/"):
        return render_uri(uri, checkpoint, lang)
    label = str(item.get("label") or "")
    if label:
        return label
    return render_uri(uri, checkpoint, lang)


def _surface_for_uri(uri: str, checkpoint: Checkpoint, lang: str) -> str:
    return render_uri(uri, checkpoint, lang)


def _surface_from_aliases(uri: str, checkpoint: Checkpoint, lang: str | None) -> str:
    if not lang:
        return ""
    candidates: list[str] = []
    for word, mapped in checkpoint.aliases.items():
        if mapped != uri:
            continue
        if _word_lang(word) == lang:
            return word.replace("_", " ")
        candidates.append(word.replace("_", " "))
    if candidates:
        return candidates[0]
    return ""


def _definition_label(uri: str, checkpoint: Checkpoint, lang: str | None) -> str:
    definitions = checkpoint.metadata.get("concept_definitions", {})
    if not isinstance(definitions, dict):
        return ""
    info = definitions.get(uri)
    if not isinstance(info, dict):
        return ""
    label = str(info.get("label", ""))
    if not label:
        return ""
    if lang == "en" and uri.startswith("/c/en/"):
        return uri.split("/", 3)[-1].replace("_", " ")
    return label


def _concept_surfaces(concepts: list[str], checkpoint: Checkpoint, lang: str) -> list[str]:
    values: list[str] = []
    for uri in concepts:
        surface = _surface_for_uri(uri, checkpoint, lang)
        if surface and surface not in values:
            values.append(surface)
    return values


def _concept_lang(concepts: list[str]) -> str | None:
    for uri in concepts:
        parts = str(uri).split("/", 3)
        if len(parts) > 2 and parts[1] == "c" and parts[2] in {"ru", "en"}:
            return parts[2]
    return None


def _domain_uri(semantic_vector: dict[str, Any]) -> str:
    domain = semantic_vector.get("top_domain")
    if isinstance(domain, dict):
        return str(domain.get("uri", ""))
    return ""


def _domain_surface_from_concepts(concepts: list[str], checkpoint: Checkpoint, lang: str) -> str:
    for uri in concepts:
        if uri.startswith("/m/top/") or uri.startswith("/m/basic/category/"):
            surface = _surface_for_uri(uri, checkpoint, lang)
            if surface:
                return surface
    return ""


def _variant_is_a(lang: str, subject: str, items: list[str], domain: str) -> str:
    if not subject:
        return ""
    predicate = _choose_anchor(items, lang, preferred=("star", "звезда", "object", "предмет", "nature", "природа"))
    if not predicate:
        predicate = domain or _choose_anchor(items, lang)
    if not predicate:
        return ""
    if lang == "ru":
        return f"{subject} — это {predicate}."
    return f"{subject} is a {predicate}."


def _variant_source(lang: str, subject: str, items: list[str], domain: str) -> str:
    if not subject:
        return ""
    primary = _choose_anchor(items, lang, preferred=("light", "свет", "heat", "тепло"))
    secondary = _choose_anchor(items, lang, preferred=("heat", "тепло", "light", "свет"))
    parts = _unique_nonempty(_nonempty([primary, secondary]))
    if not parts:
        return ""
    joined = _join_list(parts[:2], lang)
    if lang == "ru":
        return f"{subject} дарит {joined}."
    return f"{subject} gives {joined}."


def _variant_related(lang: str, subject: str, items: list[str], domain: str) -> str:
    if not subject:
        return ""
    terms = _unique_nonempty([value for value in items[:3] if value != subject])
    if not terms and domain:
        terms = [domain]
    if not terms:
        return ""
    joined = _join_list(terms[:3], lang)
    if lang == "ru":
        return f"{subject} связано с {joined}."
    return f"{subject} connects with {joined}."


def _variant_image(lang: str, subject: str, items: list[str], domain: str) -> str:
    if not subject:
        return ""
    terms = _unique_nonempty([value for value in items[:2] if value != subject])
    if not terms and domain:
        terms = [domain]
    if not terms:
        return ""
    if lang == "ru":
        return f"Если смотреть образно, {subject} выглядит как {terms[0]}."
    return f"Seen loosely, {subject} feels like {terms[0]}."


def _variant_definition(lang: str, subject: str, meaning: str) -> str:
    if not subject or not meaning:
        return ""
    if lang == "ru":
        return f"{subject} — это {meaning}."
    return f"{subject} is {meaning}."


def _phrase_is_a(lang: str, subject: str, rest: list[str], domain: str, seed: int) -> str:
    predicate = _choose_anchor(rest, lang, preferred=("star", "звезда", "light", "свет", "nature", "природа", "object", "предмет"))
    if not predicate:
        predicate = domain or (rest[0] if rest else "")
    if not subject or not predicate:
        return ""
    if lang == "ru":
        variants = [
            f"{subject} — это {predicate}.",
            f"{subject} — {predicate}.",
        ]
    else:
        variants = [
            f"{subject} is a {predicate}.",
            f"{subject} is {predicate}.",
        ]
    return variants[seed % len(variants)]


def _phrase_source(lang: str, subject: str, rest: list[str], domain: str, seed: int) -> str:
    if not subject:
        return ""
    anchors = _unique_nonempty([value for value in rest if value not in {subject, domain}])
    if not anchors:
        anchors = [domain] if domain else []
    if not anchors:
        return ""
    pairs = _join_list(anchors[:2], lang)
    if lang == "ru":
        variants = [
            f"{subject} дарит {pairs}.",
            f"{subject} приносит {pairs}.",
        ]
    else:
        variants = [
            f"{subject} gives {pairs}.",
            f"{subject} brings {pairs}.",
        ]
    return variants[seed % len(variants)]


def _phrase_related(lang: str, subject: str, rest: list[str], domain: str, seed: int) -> str:
    terms = _unique_nonempty([value for value in rest[:3] if value != subject])
    if not terms and domain:
        terms = [domain]
    if not subject or not terms:
        return ""
    joined = _join_list(terms[:3], lang)
    if lang == "ru":
        variants = [
            f"{subject} связано с {joined}.",
            f"{subject} держится рядом с {joined}.",
        ]
    else:
        variants = [
            f"{subject} connects with {joined}.",
            f"{subject} stays near {joined}.",
        ]
    return variants[seed % len(variants)]


def _phrase_image(lang: str, subject: str, rest: list[str], domain: str, seed: int) -> str:
    terms = _unique_nonempty([value for value in rest[:2] if value != subject])
    if not terms and domain:
        terms = [domain]
    if not subject or not terms:
        return ""
    if lang == "ru":
        variants = [
            f"Если смотреть образно, {subject} будто несет {terms[0]}.",
            f"Образно говоря, {subject} похож на {terms[0]}.",
        ]
    else:
        variants = [
            f"Seen loosely, {subject} carries {terms[0]}.",
            f"Loosely speaking, {subject} feels like {terms[0]}.",
        ]
    return variants[seed % len(variants)]


def _choose_anchor(values: list[str], lang: str, preferred: tuple[str, ...] = ()) -> str:
    for token in preferred:
        for value in values:
            if value == token:
                return value
    if values:
        return values[0]
    return ""


def _join_list(values: list[str], lang: str) -> str:
    items = _nonempty(values)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} и {items[1]}" if lang == "ru" else f"{items[0]} and {items[1]}"
    tail = items[-1]
    head = ", ".join(items[:-1])
    return f"{head} и {tail}" if lang == "ru" else f"{head}, and {tail}"


def _fallback(lang: str) -> str:
    return "Смысл пока слишком разрежен для уверенного ответа." if lang == "ru" else "The meaning is still too sparse for a confident answer."


def _meaning_for_subject(items: list[dict[str, Any]], checkpoint: Checkpoint, lang: str) -> str:
    item = _best_item(items, checkpoint, lang)
    if not item:
        return ""
    uri = str(item.get("uri", ""))
    definitions = checkpoint.metadata.get("concept_definitions", {})
    if not isinstance(definitions, dict):
        return ""
    info = definitions.get(uri)
    if not isinstance(info, dict):
        return ""
    return " ".join(str(info.get("meaning", "")).split())


def _word_lang(word: str) -> str:
    return "ru" if any("а" <= char.lower() <= "я" or char.lower() == "ё" for char in word) else "en"


def _item_score(item: dict[str, Any]) -> float:
    try:
        return float(item.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _item_matches_lang(item: dict[str, Any], lang: str) -> bool:
    uri = str(item.get("uri", ""))
    if uri.startswith(f"/c/{lang}/"):
        return True
    item_lang = str(item.get("language", ""))
    if item_lang == lang:
        return True
    return False


def _looks_like_content(surface: str, lang: str) -> bool:
    clean = surface.strip().lower()
    if not clean:
        return False
    if clean in {"unknown context", "unknown_context"}:
        return False
    if lang == "ru" and clean in {"и", "в", "не", "на", "это", "как", "что", "а", "я"}:
        return False
    if lang == "en" and clean in {"the", "a", "an", "and", "or", "to", "of", "in", "is", "are"}:
        return False
    return True


def _is_sparse_unknown_vector(items: list[dict[str, Any]]) -> bool:
    if not items:
        return True
    known_signal = False
    for item in items:
        uri = str(item.get("uri", ""))
        sources = set(map(str, item.get("sources", []))) if isinstance(item.get("sources", []), list) else set()
        if uri.endswith("/unknown_context"):
            continue
        if not sources or sources <= {"input", "fallback"}:
            continue
        known_signal = True
    return not known_signal and any(str(item.get("uri", "")).endswith("/unknown_context") for item in items)


def _unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = " ".join(str(value).split())
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _nonempty(values: list[str]) -> list[str]:
    return [" ".join(str(value).split()) for value in values if " ".join(str(value).split())]
