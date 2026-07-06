from __future__ import annotations

from typing import Any, TYPE_CHECKING

from semantic_ants.core.normalization import detect_language, tokenize
from semantic_ants.learning.canonical import canonical_concept_uri
from semantic_ants.learning.checkpoint import Checkpoint

if TYPE_CHECKING:
    from semantic_ants.generation.torch_dialogue import TorchDialogueNavigator

LANGUAGE_URI_SURFACES = {
    "/m/language/ru": {"ru": "русский язык", "en": "Russian language"},
    "/m/language/en": {"ru": "английский язык", "en": "English language"},
}


class SenseSentenceBuilder:
    def build_candidates(self, semantic_vector: dict[str, Any], checkpoint: Checkpoint, count: int = 3) -> list[str]:
        return build_vector_candidates(semantic_vector, checkpoint, count=count)

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


def build_vector_candidates(
    semantic_vector: dict[str, Any],
    checkpoint: Checkpoint,
    count: int = 3,
    navigator: Any | None = None,
    model_dir: str | None = None,
) -> list[str]:
    return select_vector_response(
        semantic_vector,
        checkpoint,
        count=count,
        navigator=navigator,
        model_dir=model_dir,
    )["candidates"]


def select_vector_response(
    semantic_vector: dict[str, Any],
    checkpoint: Checkpoint,
    count: int = 3,
    navigator: Any | None = None,
    model_dir: str | None = None,
) -> dict[str, Any]:
    lang = _vector_lang(semantic_vector)
    items = [item for item in semantic_vector.get("items", []) if isinstance(item, dict)]
    if _is_sparse_unknown_vector(items):
        return {"response": "", "candidates": [], "source": "semantic_fallback", "lang": lang}
    subject = _surface_for_item(_best_item(items, checkpoint, lang), checkpoint, lang)
    definition = _meaning_for_subject(items, checkpoint, lang)
    definition_candidate = _variant_definition(lang, subject, definition)
    language_candidate = _variant_language(lang, subject, _language_surface_from_items(items, checkpoint, lang))
    items = _ordered_surfaces(items, checkpoint, lang, subject)
    domain = _surface_for_uri(_domain_uri(semantic_vector), checkpoint, lang)
    vectors = _nonempty([subject, *items[:5], domain]) or _fallback_basis(items, checkpoint, lang)

    exact_memory = _exact_memory_candidates(semantic_vector, checkpoint, lang, count=count)
    contextual = _contextual_candidates(semantic_vector, lang, count=count)
    definition_candidates = _unique_nonempty(
        [
            definition_candidate,
            language_candidate,
            _variant_is_a(lang, subject, items, domain),
            _variant_source(lang, subject, items, domain),
            _variant_related(lang, subject, items, domain),
            _variant_image(lang, subject, items, domain),
        ]
    )
    translation_candidates = _translation_candidates(vectors, checkpoint, lang, count=count)
    torch_candidates = _torch_candidates(
        semantic_vector,
        checkpoint,
        navigator=navigator,
        model_dir=model_dir,
        lang=lang,
    )
    fallback_candidates = _fallback_candidates(vectors, lang, count=count)
    candidates = _unique_nonempty(
        [
            *exact_memory,
            *contextual,
            *definition_candidates,
            *translation_candidates,
            *torch_candidates,
            *fallback_candidates,
        ]
    )
    response = candidates[0] if candidates else ""
    return {
        "response": response,
        "candidates": candidates[: max(count, 1)],
        "source": _candidate_source(response, exact_memory, contextual, definition_candidates, translation_candidates, torch_candidates),
        "lang": lang,
    }


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
    if uri in LANGUAGE_URI_SURFACES:
        surfaces = LANGUAGE_URI_SURFACES[uri]
        return surfaces.get(selected_lang or "", "") or surfaces["en"]
    aliased = _surface_from_aliases(uri, checkpoint, selected_lang)
    if aliased:
        return aliased
    translated = _cross_language_surface(uri, checkpoint, selected_lang)
    if translated:
        return translated
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
                return aliased or token
            return token
    return aliased or uri.rstrip("/").split("/")[-1].replace("_", " ")


def render_concepts(concepts: list[str], checkpoint: Checkpoint, lang: str | None = None) -> list[str]:
    selected_lang = lang if lang in {"ru", "en"} else _concept_lang(concepts) or "auto"
    return _concept_surfaces(concepts, checkpoint, selected_lang)


def _vector_lang(semantic_vector: dict[str, Any]) -> str:
    lang = str(
        semantic_vector.get("response_lang")
        or semantic_vector.get("target_lang")
        or semantic_vector.get("answer_lang")
        or semantic_vector.get("lang", "auto")
    )
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
    content = [
        item
        for item in ordered
        if _item_layer(item) != 0 and _looks_like_content(_surface_for_item(item, checkpoint, lang), lang)
    ]
    if content:
        return content[0]
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
    surface = render_uri(uri, checkpoint, lang)
    if surface:
        return surface
    label = str(item.get("label") or "")
    if label:
        return label
    return ""


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


def _cross_language_surface(uri: str, checkpoint: Checkpoint, lang: str | None) -> str:
    if lang not in {"ru", "en"} or not uri.startswith("/c/"):
        return ""
    token = uri.split("/", 3)[-1].replace("_", " ")
    basic_concepts = checkpoint.metadata.get("basic_concepts", {})
    if not isinstance(basic_concepts, dict):
        return ""
    for info in basic_concepts.values():
        if not isinstance(info, dict):
            continue
        aliases = info.get("aliases", {})
        if not isinstance(aliases, dict):
            continue
        source_words = aliases.get("ru" if lang == "en" else "en", [])
        target_words = aliases.get(lang, [])
        if not isinstance(source_words, list) or not isinstance(target_words, list):
            continue
        if token in {str(word).replace("_", " ") for word in source_words} and target_words:
            return str(target_words[0]).replace("_", " ")
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


def _variant_language(lang: str, subject: str, language: str) -> str:
    if not subject or not language:
        return ""
    if lang == "ru":
        return f"Понятие «{subject}» связано с языком: {language}."
    return f'The concept "{subject}" is linked to language: {language}.'


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


def _meaning_for_subject(items: list[dict[str, Any]], checkpoint: Checkpoint, lang: str) -> str:
    best = _best_item(items, checkpoint, lang)
    if not best:
        return ""
    best_surface = _surface_for_item(best, checkpoint, lang)
    for item in [best, *items]:
        if best_surface and _surface_for_item(item, checkpoint, lang) not in {best_surface, ""}:
            continue
        meaning = _definition_text(str(item.get("uri", "")), checkpoint)
        if meaning:
            return meaning
    return ""


def _definition_text(uri: str, checkpoint: Checkpoint) -> str:
    definitions = checkpoint.metadata.get("concept_definitions", {})
    if not isinstance(definitions, dict):
        return ""
    info = definitions.get(uri)
    if not isinstance(info, dict):
        return ""
    return " ".join(str(info.get("meaning") or info.get("definition") or "").split())


def _language_surface_from_items(items: list[dict[str, Any]], checkpoint: Checkpoint, lang: str) -> str:
    for item in items:
        uri = str(item.get("uri", ""))
        if uri in LANGUAGE_URI_SURFACES:
            return _surface_for_uri(uri, checkpoint, lang)
    return ""


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
    surface = str(item.get("label") or uri.rsplit("/", 1)[-1].replace("_", " "))
    if item_lang in {"", "concept", "unknown"} and lang in {"ru", "en"} and surface:
        return _word_lang(surface) == lang
    return False


def _item_layer(item: dict[str, Any]) -> int:
    try:
        return int(item.get("layer", 1))
    except (TypeError, ValueError):
        return 1


def _looks_like_content(surface: str, lang: str) -> bool:
    clean = surface.strip().lower()
    if not clean:
        return False
    if clean in {"unknown context", "unknown_context"}:
        return False
    ru_noise = {"и", "в", "не", "на", "это", "как", "что", "а", "я"}
    en_noise = {"the", "a", "an", "and", "or", "to", "of", "in", "is", "are", "meaning"}
    tokens = tokenize(clean)
    if lang == "ru" and (clean in ru_noise or all(token in ru_noise for token in tokens)):
        return False
    if lang == "ru" and any(token in ru_noise for token in tokens) and any(token in en_noise for token in tokens):
        return False
    if lang == "en" and (clean in en_noise or all(token in en_noise for token in tokens)):
        return False
    if lang == "en" and any(token in en_noise for token in tokens) and any(_word_lang(token) == "ru" for token in tokens):
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


def _fallback_basis(items: list[str], checkpoint: Checkpoint, lang: str) -> list[str]:
    values = [value for value in items if value]
    if values:
        return values
    return _memory_basis(checkpoint, lang)


def _memory_basis(checkpoint: Checkpoint, lang: str) -> list[str]:
    candidates: list[str] = []
    for item in checkpoint.accepted_answers[:10]:
        if not isinstance(item, dict):
            continue
        answer = " ".join(str(item.get("answer", "")).split())
        if answer:
            candidates.append(answer)
    for item in checkpoint.response_memory.values():
        if not isinstance(item, dict):
            continue
        answer = " ".join(str(item.get("answer", "")).split())
        if answer:
            candidates.append(answer)
    if candidates:
        return _unique_nonempty(candidates[:10])
    if lang == "ru":
        return ["Это связано с известным понятием."]
    return ["It is related to a known concept."]


def _exact_memory_candidates(semantic_vector: dict[str, Any], checkpoint: Checkpoint, lang: str, count: int) -> list[str]:
    raw_input_text = str(semantic_vector.get("input_text", ""))
    if _is_follow_up_text(raw_input_text) and not _is_translation_request(raw_input_text):
        return []
    input_text = _memory_text_key(raw_input_text)
    concept_key = _concept_key([item.get("uri") for item in semantic_vector.get("items", []) if isinstance(item, dict)])
    candidates: list[str] = []
    for item in checkpoint.accepted_answers:
        if not isinstance(item, dict):
            continue
        stimulus = _memory_text_key(str(item.get("stimulus", "")))
        answer = " ".join(str(item.get("answer", "")).split())
        if answer and not _memory_lang_matches(item, lang, raw_input_text):
            continue
        if answer and stimulus and stimulus == input_text:
            candidates.append(answer)
            continue
        memory_key = _concept_key(item.get("concepts", []))
        if answer and concept_key and memory_key == concept_key:
            candidates.append(answer)
    for item in checkpoint.response_memory.values():
        if not isinstance(item, dict):
            continue
        answer = " ".join(str(item.get("answer", "")).split())
        if answer and not _memory_lang_matches(item, lang, raw_input_text):
            continue
        memory_key = _concept_key(item.get("concepts", []))
        if answer and concept_key and memory_key == concept_key:
            candidates.append(answer)
    return _unique_nonempty(candidates)[:count]


def _contextual_candidates(semantic_vector: dict[str, Any], lang: str, count: int) -> list[str]:
    if not _is_follow_up_text(str(semantic_vector.get("input_text", ""))):
        return []
    history = semantic_vector.get("chat_history", [])
    if not isinstance(history, list) or not history:
        return []
    previous_user = next((turn for turn in reversed(history) if isinstance(turn, dict) and str(turn.get("role", "")) == "user"), None)
    if not isinstance(previous_user, dict):
        return []
    previous_text = " ".join(str(previous_user.get("text", "")).split())
    if not previous_text:
        return []
    if lang == "ru":
        return [f"Ты спрашивал про {previous_text}. Могу продолжить.", "Могу связать это с предыдущим вопросом."][:count]
    return [f"You asked about {previous_text}. I can continue.", "I can connect this with the previous question."][:count]


def _memory_text_key(value: str) -> str:
    tokens = tokenize(value)
    if tokens:
        return " ".join(tokens).casefold()
    return " ".join(str(value).split()).casefold()


def _memory_lang_matches(item: dict[str, Any], lang: str, input_text: str) -> bool:
    item_lang = str(item.get("lang") or "")
    if item_lang not in {"ru", "en"} or item_lang == lang:
        return True
    return _is_translation_request(input_text)


def _is_translation_request(value: str) -> bool:
    normalized = " ".join(token.replace("_", " ") for token in tokenize(value)).casefold()
    return any(
        cue in normalized
        for cue in (
            "переведи",
            "перевод",
            "как будет",
            "как сказать",
            "по английски",
            "на английском",
            "по русски",
            "на русском",
            "translate",
            "translation",
            "how do you say",
            "how to say",
            "in english",
            "in russian",
        )
    )


def _is_follow_up_text(value: str) -> bool:
    tokens = tokenize(value)
    if not tokens:
        return False
    normalized = " ".join(tokens).casefold()
    if normalized in {
        "это",
        "что это",
        "а это",
        "а он",
        "а она",
        "а они",
        "подробнее",
        "расскажи подробнее",
        "ещё",
        "еще",
        "продолжай",
        "почему",
        "как именно",
        "what about it",
        "tell me more",
        "more",
        "continue",
        "why",
    }:
        return True
    follow_up_terms = {"это", "он", "она", "они", "подробнее", "ещё", "еще", "more", "it", "this", "that"}
    return len(tokens) <= 3 and all(token.casefold() in follow_up_terms for token in tokens)


def _translation_candidates(items: list[str], checkpoint: Checkpoint, lang: str, count: int) -> list[str]:
    candidates: list[str] = []
    for uri in items:
        surface = render_uri(uri, checkpoint, lang)
        if surface and surface not in candidates:
            candidates.append(surface)
    return candidates[:count]


def _torch_candidates(
    semantic_vector: dict[str, Any],
    checkpoint: Checkpoint,
    *,
    navigator: Any | None,
    model_dir: str | None,
    lang: str,
) -> list[str]:
    if navigator is None:
        from semantic_ants.generation.torch_dialogue import TorchDialogueNavigator

        navigator = TorchDialogueNavigator()
    prompt = _semantic_prompt_from_vector(semantic_vector, lang)
    return navigator.generate(prompt, checkpoint, model_dir=model_dir, fallback="", count=1, lang=lang)


def _semantic_prompt_from_vector(semantic_vector: dict[str, Any], lang: str) -> str:
    items = [item for item in semantic_vector.get("items", []) if isinstance(item, dict)]
    labels = [str(item.get("label", "")) for item in items[:8] if item.get("label")]
    history = semantic_vector.get("chat_history", [])
    history_text = "\n".join(
        f'{turn.get("role", "user")}: {turn.get("text", "")}' for turn in history[-6:] if isinstance(turn, dict)
    )
    return "\n".join(
        [
            f"lang: {lang}",
            f"input_text: {semantic_vector.get('input_text', '')}",
            f"labels: {', '.join(labels)}",
            history_text,
        ]
    )


def _fallback_candidates(items: list[str], lang: str, count: int) -> list[str]:
    values = [value for value in items if value]
    if values:
        return values[:count]
    if lang == "ru":
        return ["Это связано с известным понятием."]
    return ["It is related to a known concept."]


def _candidate_source(
    response: str,
    exact_memory: list[str],
    contextual: list[str],
    definition_candidates: list[str],
    translation: list[str],
    torch_candidates: list[str],
) -> str:
    if response and response in exact_memory:
        return "exact_memory"
    if response and response in contextual:
        return "contextual_follow_up"
    if response and response in definition_candidates:
        return "concept_definition"
    if response and response in translation:
        return "translation_evidence"
    if response and response in torch_candidates:
        return "torch_candidate"
    return "semantic_fallback"


def _concept_key(values: Any) -> str:
    if not isinstance(values, (list, tuple)):
        return ""
    normalized = []
    for value in values:
        if not value:
            continue
        text = str(value)
        if text.startswith("/m/top/") or text.startswith("/m/concept/"):
            normalized.append(text)
        elif text.startswith("/c/"):
            normalized.append(canonical_concept_uri(text.split("/", 3)[-1]))
        else:
            normalized.append(canonical_concept_uri(text))
    return "|".join(sorted(dict.fromkeys(normalized)))
