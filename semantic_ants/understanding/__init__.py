from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Any

from semantic_ants.core.normalization import detect_language, normalize_text, text_to_concept_uri
from semantic_ants.learning.checkpoint import Checkpoint

try:
    import pymorphy3
except ModuleNotFoundError as exc:  # pragma: no cover - dependency is declared in pyproject
    raise RuntimeError("Install pymorphy3 with the project dependencies") from exc


TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-'][0-9A-Za-zА-Яа-яЁё]+)?")

RU_EXTRA_STOP_WORDS = {
    "а",
    "без",
    "бы",
    "был",
    "была",
    "были",
    "в",
    "вам",
    "вас",
    "весь",
    "во",
    "вот",
    "все",
    "всех",
    "всю",
    "вся",
    "где",
    "да",
    "для",
    "до",
    "ей",
    "ею",
    "если",
    "еще",
    "ж",
    "же",
    "за",
    "и",
    "из",
    "или",
    "им",
    "их",
    "к",
    "как",
    "кого",
    "когда",
    "кто",
    "ли",
    "либо",
    "между",
    "меня",
    "мне",
    "много",
    "может",
    "мой",
    "мы",
    "на",
    "над",
    "нам",
    "нас",
    "не",
    "него",
    "нее",
    "него",
    "ней",
    "нею",
    "ни",
    "но",
    "ну",
    "о",
    "об",
    "оба",
    "обо",
    "от",
    "по",
    "под",
    "пока",
    "при",
    "про",
    "раз",
    "с",
    "сам",
    "себе",
    "себя",
    "сво",
    "та",
    "там",
    "тебя",
    "тоже",
    "тут",
    "ты",
    "у",
    "уж",
    "ужо",
    "что",
    "чтоб",
    "чтобы",
    "эта",
    "это",
    "этот",
    "я",
    "эй",
}

EN_EXTRA_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "go",
    "if",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "so",
    "than",
    "that",
    "the",
    "their",
    "this",
    "to",
    "too",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class UnderstandingToken:
    raw_token: str
    lemma: str
    search_token: str
    concept_uri: str | None
    match_status: str
    is_stop_word: bool
    morphology: dict[str, str | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnderstandingSummary:
    total_tokens: int
    working_tokens: int
    stop_words: int
    matched: int
    candidates: int
    partial_root_matches: int
    edit_distance_matches: int
    search_tokens: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnderstandingResult:
    input_text: str
    lang: str
    session_id: str | None
    turn_id: str | None
    tokens: list[UnderstandingToken]
    summary: UnderstandingSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "lang": self.lang,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "tokens": [token.to_dict() for token in self.tokens],
            "summary": self.summary.to_dict(),
        }


def understand_text(
    text: str,
    *,
    lang: str = "auto",
    checkpoint: Checkpoint,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> UnderstandingResult:
    input_text = str(text)
    selected_lang = _select_lang(input_text, lang)
    alias_map = _collect_aliases(checkpoint)
    known_tokens = tuple(alias_map.keys())
    stop_words = _stop_words(checkpoint, selected_lang)
    tokens: list[UnderstandingToken] = []
    search_tokens: list[str] = []
    counts = {
        "matched": 0,
        "candidates": 0,
        "partial_root_matches": 0,
        "edit_distance_matches": 0,
        "stop_words": 0,
    }

    for raw_token in _raw_tokens(input_text):
        token = _build_token(raw_token, selected_lang, alias_map, known_tokens, stop_words)
        tokens.append(token)
        if token.is_stop_word:
            counts["stop_words"] += 1
            continue
        search_tokens.append(token.search_token)
        if token.match_status == "candidate":
            counts["candidates"] += 1
        elif token.match_status == "partial_root_match":
            counts["partial_root_matches"] += 1
            counts["matched"] += 1
        elif token.match_status == "edit_distance_match":
            counts["edit_distance_matches"] += 1
            counts["matched"] += 1
        else:
            counts["matched"] += 1

    summary = UnderstandingSummary(
        total_tokens=len(tokens),
        working_tokens=len(search_tokens),
        stop_words=counts["stop_words"],
        matched=counts["matched"],
        candidates=counts["candidates"],
        partial_root_matches=counts["partial_root_matches"],
        edit_distance_matches=counts["edit_distance_matches"],
        search_tokens=search_tokens,
    )
    return UnderstandingResult(
        input_text=input_text,
        lang=selected_lang,
        session_id=session_id,
        turn_id=turn_id,
        tokens=tokens,
        summary=summary,
    )


def _build_token(
    raw_token: str,
    lang: str,
    alias_map: dict[str, str],
    known_tokens: tuple[str, ...],
    stop_words: set[str],
) -> UnderstandingToken:
    normalized_raw = _normalize_token(raw_token)
    lemma = normalized_raw
    morphology = _empty_morphology()
    candidate_forms: list[tuple[str, str]] = []

    if lang == "ru":
        parsed = _parse_ru(raw_token)
        if parsed:
            morph_lemma = _normalize_token(parsed.normal_form or normalized_raw)
            if _is_proper_name(raw_token, parsed):
                lemma = normalized_raw
                candidate_forms = [(normalized_raw, "found_as_raw")]
            else:
                candidate_forms = _ru_candidate_forms(normalized_raw, morph_lemma)
                lemma = candidate_forms[0][0] if candidate_forms else morph_lemma
            morphology = _morphology_from_parse(parsed)
    elif lang == "en":
        candidate_forms = _en_candidate_forms(normalized_raw)
        lemma = candidate_forms[0][0] if candidate_forms else normalized_raw
    else:
        candidate_forms = [(normalized_raw, "found_as_raw")]

    is_stop_word = normalized_raw in stop_words or lemma in stop_words
    if is_stop_word:
        return UnderstandingToken(
            raw_token=raw_token,
            lemma=lemma,
            search_token="",
            concept_uri=None,
            match_status="ignored_stop_word",
            is_stop_word=True,
            morphology=morphology,
        )

    exact_match = _exact_match(candidate_forms, alias_map)
    if exact_match is not None:
        search_token, concept_uri, status = exact_match
    else:
        search_token, concept_uri, status = _heuristic_match(
            normalized_raw=normalized_raw,
            lemma=lemma,
            lang=lang,
            alias_map=alias_map,
            known_tokens=known_tokens,
        )

    return UnderstandingToken(
        raw_token=raw_token,
        lemma=lemma,
        search_token=search_token,
        concept_uri=concept_uri,
        match_status=status,
        is_stop_word=False,
        morphology=morphology,
    )


def _exact_match(
    candidate_forms: list[tuple[str, str]],
    alias_map: dict[str, str],
) -> tuple[str, str, str] | None:
    for token, status in candidate_forms:
        uri = alias_map.get(token)
        if uri:
            return token, uri, status
    return None


def _heuristic_match(
    *,
    normalized_raw: str,
    lemma: str,
    lang: str,
    alias_map: dict[str, str],
    known_tokens: tuple[str, ...],
) -> tuple[str, str | None, str]:
    search_token = lemma or normalized_raw
    candidate_token = _partial_root_match(search_token, known_tokens)
    if candidate_token:
        return candidate_token, alias_map.get(candidate_token), "partial_root_match"

    if lang == "en" or len(search_token) >= 5:
        edit_token = _edit_distance_match(search_token, known_tokens)
        if edit_token:
            return edit_token, alias_map.get(edit_token), "edit_distance_match"

    concept_uri = _build_concept_uri(search_token, lang)
    return search_token, concept_uri, "candidate"


def _ru_candidate_forms(normalized_raw: str, lemma: str) -> list[tuple[str, str]]:
    forms: list[tuple[str, str]] = []
    seen: set[str] = set()
    for token in _ru_root_variants(lemma) + [lemma, normalized_raw]:
        if not token or token in seen:
            continue
        seen.add(token)
        status = "found_as_alias" if token not in {lemma, normalized_raw} else ("found_as_lemma" if token == lemma else "found_as_raw")
        forms.append((token, status))
    return forms


def _en_candidate_forms(normalized_raw: str) -> list[tuple[str, str]]:
    forms: list[tuple[str, str]] = []
    seen: set[str] = set()
    for token in [normalized_raw]:
        if token and token not in seen:
            seen.add(token)
            forms.append((token, "found_as_raw"))
    return forms


def _ru_root_variants(token: str) -> list[str]:
    suffixes = (
        "чик",
        "очек",
        "очк",
        "еньк",
        "ышк",
        "ушк",
        "ик",
    )
    variants: list[str] = []
    queue = [token]
    seen = {token}
    for _ in range(2):
        next_queue: list[str] = []
        for current in queue:
            for suffix in suffixes:
                if not current.endswith(suffix):
                    continue
                root = current[: -len(suffix)]
                if len(root) < 3 or root in seen:
                    continue
                seen.add(root)
                variants.append(root)
                next_queue.append(root)
        queue = next_queue
    return variants


def _partial_root_match(token: str, known_tokens: tuple[str, ...]) -> str | None:
    cleaned = _normalized_lookup_key(token)
    if len(cleaned) < 3:
        return None
    best_key = None
    best_score = 0
    for candidate in known_tokens:
        if candidate and candidate[0] != cleaned[0]:
            continue
        score = _shared_prefix_score(cleaned, candidate)
        if score >= 3 and score > best_score:
            best_key = candidate
            best_score = score
    return best_key


def _edit_distance_match(token: str, known_tokens: tuple[str, ...]) -> str | None:
    cleaned = _normalized_lookup_key(token)
    if len(cleaned) < 2:
        return None
    best_key = None
    best_distance = 10**9
    for candidate in known_tokens:
        candidate_key = _normalized_lookup_key(candidate)
        if abs(len(candidate_key) - len(cleaned)) > 2:
            continue
        distance = _levenshtein(cleaned, candidate_key)
        max_distance = 1 if max(len(cleaned), len(candidate_key)) <= 4 else 2
        if distance <= max_distance and distance < best_distance:
            best_key = candidate
            best_distance = distance
    return best_key


def _collect_aliases(checkpoint: Checkpoint) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for token, uri in checkpoint.aliases.items():
        _register_alias(aliases, token, uri)

    labels = checkpoint.metadata.get("concept_labels", {})
    if isinstance(labels, dict):
        for uri, label in labels.items():
            _register_alias(aliases, str(label), str(uri))

    definitions = checkpoint.metadata.get("concept_definitions", {})
    if isinstance(definitions, dict):
        for uri, info in definitions.items():
            if not isinstance(info, dict):
                continue
            _register_alias(aliases, str(info.get("label", "")), str(uri))

    return aliases


def _register_alias(aliases: dict[str, str], token: str, uri: str) -> None:
    key = _normalized_lookup_key(token)
    if key and uri and key not in aliases:
        aliases[key] = uri


def _stop_words(checkpoint: Checkpoint, lang: str) -> set[str]:
    words = _common_words(checkpoint, lang)
    if lang == "ru":
        words.update(RU_EXTRA_STOP_WORDS)
    elif lang == "en":
        words.update(EN_EXTRA_STOP_WORDS)
    return words


def _common_words(checkpoint: Checkpoint, lang: str) -> set[str]:
    values: set[str] = set()
    common_words = checkpoint.metadata.get("common_words", {})
    if isinstance(common_words, dict):
        lang_words = common_words.get(lang, [])
        if isinstance(lang_words, list):
            values.update(_normalized_lookup_key(str(value)) for value in lang_words if value)
    return values


def _raw_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).replace("’", "'")
    return [match.group(0) for match in TOKEN_RE.finditer(normalized)]


def _normalize_token(text: str) -> str:
    return _normalized_lookup_key(text)


def _normalized_lookup_key(text: str) -> str:
    normalized = normalize_text(text)
    return normalized.replace("-", "_").replace("'", "").replace("’", "")


def _build_concept_uri(token: str, lang: str) -> str | None:
    try:
        return text_to_concept_uri(token, lang=lang)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _morph() -> "pymorphy3.MorphAnalyzer":
    return pymorphy3.MorphAnalyzer()


def _parse_ru(token: str):
    parsed = _morph().parse(token)
    return parsed[0] if parsed else None


def _morphology_from_parse(parsed: Any) -> dict[str, str | None]:
    tag = getattr(parsed, "tag", None)
    return {
        "POS": str(getattr(tag, "POS", "") or "") or None,
        "case": str(getattr(tag, "case", "") or "") or None,
        "number": str(getattr(tag, "number", "") or "") or None,
        "gender": str(getattr(tag, "gender", "") or "") or None,
        "tense": str(getattr(tag, "tense", "") or "") or None,
        "person": str(getattr(tag, "person", "") or "") or None,
    }


def _is_proper_name(raw_token: str, parsed: Any) -> bool:
    tag = getattr(parsed, "tag", None)
    return bool(raw_token[:1].isupper() and tag and "Name" in tag)


def _empty_morphology() -> dict[str, str | None]:
    return {"POS": None, "case": None, "number": None, "gender": None, "tense": None, "person": None}


def _select_lang(text: str, lang: str) -> str:
    if lang in {"ru", "en"}:
        return lang
    return detect_language(text)


def _shared_prefix_score(left: str, right: str) -> int:
    score = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        score += 1
    return score


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (left_char != right_char)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]
