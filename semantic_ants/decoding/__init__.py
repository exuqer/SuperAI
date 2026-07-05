from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from semantic_ants.core.normalization import detect_language, normalize_text, text_to_concept_uri, tokenize

try:
    import pymorphy3
except ModuleNotFoundError as exc:  # pragma: no cover - dependency is declared in pyproject
    raise RuntimeError("Install pymorphy3 with the project dependencies") from exc


@dataclass(frozen=True)
class DecodeToken:
    input_token: str
    normalized_token: str
    role: str
    surface: str
    concept_uri: str | None
    transform_status: str
    morphology: dict[str, str | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecodeSummary:
    total_tokens: int
    used_tokens: int
    objects: int
    fallbacks: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecodeResult:
    input_text: str
    input_tokens: list[str]
    lang: str
    sentence: str
    pattern: str
    session_id: str | None
    turn_id: str | None
    tokens: list[DecodeToken]
    summary: DecodeSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "input_tokens": self.input_tokens,
            "lang": self.lang,
            "sentence": self.sentence,
            "pattern": self.pattern,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "tokens": [token.to_dict() for token in self.tokens],
            "summary": self.summary.to_dict(),
        }


def decode_words(
    text: str,
    *,
    tokens: list[str] | None = None,
    lang: str = "auto",
    session_id: str | None = None,
    turn_id: str | None = None,
) -> DecodeResult:
    input_text = str(text or "")
    input_tokens = _select_tokens(input_text, tokens)
    selected_lang = _select_lang(input_text, input_tokens, lang)
    if not input_tokens:
        return DecodeResult(
            input_text=input_text,
            input_tokens=[],
            lang=selected_lang,
            sentence="",
            pattern="empty",
            session_id=session_id,
            turn_id=turn_id,
            tokens=[],
            summary=DecodeSummary(total_tokens=0, used_tokens=0, objects=0, fallbacks=0),
        )

    if selected_lang == "ru":
        decoded_tokens = _decode_ru(input_tokens)
    elif selected_lang == "en":
        decoded_tokens = _decode_en(input_tokens)
    else:
        decoded_tokens = _decode_surface(input_tokens, selected_lang)

    sentence = _join_sentence(
        decoded_tokens["subject"],
        decoded_tokens["verb"],
        decoded_tokens["objects"],
        selected_lang,
        decoded_tokens.get("modifiers", []),
    )
    output_tokens = decoded_tokens["tokens"]
    summary = DecodeSummary(
        total_tokens=len(input_tokens),
        used_tokens=len(output_tokens),
        objects=len(decoded_tokens["objects"]),
        fallbacks=sum(1 for token in output_tokens if token.transform_status == "fallback"),
    )
    return DecodeResult(
        input_text=input_text,
        input_tokens=input_tokens,
        lang=selected_lang,
        sentence=sentence,
        pattern="svo",
        session_id=session_id,
        turn_id=turn_id,
        tokens=output_tokens,
        summary=summary,
    )


def _decode_ru(input_tokens: list[str]) -> dict[str, Any]:
    analyses = [_ru_analysis(token) for token in input_tokens]
    verb_index = _select_ru_verb_index(analyses)
    subject_index = _select_ru_subject_index(analyses, verb_index)

    subject = _ru_token(input_tokens[subject_index], "subject", {"nomn", "sing"}, "inflected")
    verb = (
        _ru_token(input_tokens[verb_index], "verb", {"3per", "sing", "pres"}, "inflected")
        if verb_index is not None
        else None
    )
    object_indexes = _select_ru_object_indexes(analyses, subject_index, verb_index)
    modifier_indexes = _select_ru_modifier_indexes(analyses, subject_index, verb_index, object_indexes)

    if verb is not None and _is_ru_change_state_verb(analyses[verb_index]):
        objects = [_ru_complement_token(input_tokens[index], subject) for index in object_indexes]
    else:
        objects = [_ru_token(input_tokens[index], "object", {"accs", "sing"}, "inflected") for index in object_indexes]
    modifiers = [_ru_token(input_tokens[index], "modifier", {"ablt", "sing"}, "inflected") for index in modifier_indexes]
    tokens = [*modifiers, subject, *([verb] if verb else []), *objects]
    return {
        "subject": subject,
        "verb": verb,
        "objects": objects,
        "modifiers": modifiers,
        "tokens": tokens,
    }


def _decode_en(input_tokens: list[str]) -> dict[str, Any]:
    subject_input = input_tokens[0]
    verb_input = input_tokens[1] if len(input_tokens) > 1 else ""
    object_inputs = input_tokens[2:]

    subject = _surface_token(subject_input, "subject", "surface")
    verb = _en_verb_token(verb_input, "verb") if verb_input else None
    objects = [_surface_token(token, "object", "surface") for token in object_inputs]
    tokens = [subject, *([verb] if verb else []), *objects]
    return {
        "subject": subject,
        "verb": verb,
        "objects": objects,
        "modifiers": [],
        "tokens": tokens,
    }


def _decode_surface(input_tokens: list[str], lang: str) -> dict[str, Any]:
    subject = _surface_token(input_tokens[0], "subject", "surface", lang=lang)
    verb = _surface_token(input_tokens[1], "verb", "surface", lang=lang) if len(input_tokens) > 1 else None
    objects = [_surface_token(token, "object", "surface", lang=lang) for token in input_tokens[2:]]
    tokens = [subject, *([verb] if verb else []), *objects]
    return {
        "subject": subject,
        "verb": verb,
        "objects": objects,
        "modifiers": [],
        "tokens": tokens,
    }


def _ru_token(input_token: str, role: str, grammemes: set[str], success_status: str) -> DecodeToken:
    raw = str(input_token).strip()
    normalized = _normalized_token(raw)
    parsed = _parse_ru(raw, grammemes)
    surface = normalized
    morphology = _morphology_from_parse(parsed) if parsed else _empty_morphology()
    status = "fallback"
    if parsed:
        inflected = parsed.inflect(grammemes)
        if inflected and inflected.word:
            surface = _normalized_token(inflected.word)
            morphology = _morphology_from_parse(inflected)
            status = success_status
        else:
            surface = _normalized_token(parsed.word or parsed.normal_form or normalized)
    concept_uri = _concept_uri(surface or normalized, "ru")
    return DecodeToken(
        input_token=raw,
        normalized_token=normalized,
        role=role,
        surface=surface,
        concept_uri=concept_uri,
        transform_status=status,
        morphology=morphology,
    )


def _ru_complement_token(input_token: str, subject: DecodeToken) -> DecodeToken:
    grammemes = {"ablt"}
    if subject.morphology.get("number"):
        grammemes.add(subject.morphology["number"])
    if subject.morphology.get("gender"):
        grammemes.add(subject.morphology["gender"])
    return _ru_token(input_token, "complement", grammemes, "inflected")


def _surface_token(input_token: str, role: str, success_status: str, lang: str = "en") -> DecodeToken:
    raw = str(input_token).strip()
    normalized = _normalized_token(raw)
    concept_uri = _concept_uri(normalized, lang)
    return DecodeToken(
        input_token=raw,
        normalized_token=normalized,
        role=role,
        surface=normalized,
        concept_uri=concept_uri,
        transform_status=success_status,
        morphology=_empty_morphology(),
    )


def _en_verb_token(input_token: str, role: str) -> DecodeToken:
    raw = str(input_token).strip()
    normalized = _normalized_token(raw)
    surface = _en_third_person_singular(normalized)
    status = "inflected" if surface != normalized else "fallback"
    return DecodeToken(
        input_token=raw,
        normalized_token=normalized,
        role=role,
        surface=surface,
        concept_uri=_concept_uri(surface, "en"),
        transform_status=status,
        morphology=_empty_morphology(),
    )


def _join_sentence(
    subject: DecodeToken,
    verb: DecodeToken | None,
    objects: list[DecodeToken],
    lang: str,
    modifiers: list[DecodeToken] | None = None,
) -> str:
    parts = [token.surface for token in modifiers or []]
    parts.append(subject.surface)
    if verb is not None and verb.surface:
        parts.append(verb.surface)
    if objects:
        parts.append(_join_objects([token.surface for token in objects], lang))
    return " ".join(part for part in parts if part).strip()


def _join_objects(values: list[str], lang: str) -> str:
    items = [value for value in values if value]
    if not items:
        return ""
    conjunction = "и" if lang == "ru" else "and"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def _select_tokens(text: str, tokens: list[str] | None) -> list[str]:
    if tokens is not None:
        values = [str(token).strip() for token in tokens if str(token).strip()]
        if values:
            return values
    return tokenize(text)


def _select_lang(text: str, tokens: list[str], lang: str) -> str:
    if lang in {"ru", "en"}:
        return lang
    sample = " ".join(tokens) if tokens else text
    detected = detect_language(sample)
    return detected if detected in {"ru", "en"} else "en"


def _normalized_token(text: str) -> str:
    return normalize_text(text).replace("-", "_").replace("’", "'")


def _concept_uri(token: str, lang: str) -> str | None:
    if not token:
        return None
    try:
        return text_to_concept_uri(token, lang=lang)
    except ValueError:
        return None


def _en_third_person_singular(token: str) -> str:
    irregular = {
        "be": "is",
        "do": "does",
        "go": "goes",
        "have": "has",
    }
    if token in irregular:
        return irregular[token]
    if token.endswith("y") and len(token) > 1 and token[-2] not in "aeiou":
        return token[:-1] + "ies"
    if token.endswith(("s", "x", "z", "ch", "sh", "o")):
        return token + "es"
    return token + "s"


@lru_cache(maxsize=1)
def _morph() -> "pymorphy3.MorphAnalyzer":
    return pymorphy3.MorphAnalyzer()


_RU_VERB_POS = {"INFN", "VERB"}
_RU_NOMINAL_POS = {"NOUN", "NPRO"}
_RU_ADJECTIVE_POS = {"ADJF", "ADJS", "PRTF", "PRTS"}
_RU_TEMPORAL_NOUNS = {
    "весна",
    "лето",
    "осень",
    "зима",
    "день",
    "ночь",
    "утро",
    "вечер",
    "время",
}
_RU_CHANGE_STATE_VERBS = {"становиться", "стать", "быть", "оказаться"}


def _ru_analysis(token: str) -> dict[str, Any]:
    parses = _morph().parse(token)
    parsed = parses[0] if parses else None
    normal_form = str(getattr(parsed, "normal_form", "") or _normalized_token(token))
    return {
        "token": token,
        "parses": parses,
        "parse": parsed,
        "normal_form": normal_form,
    }


def _select_ru_verb_index(analyses: list[dict[str, Any]]) -> int | None:
    for index, analysis in enumerate(analyses):
        if _is_ru_verb(analysis):
            return index
    return 1 if len(analyses) > 1 else None


def _select_ru_subject_index(analyses: list[dict[str, Any]], verb_index: int | None) -> int:
    if not analyses:
        return 0
    candidates = [index for index, analysis in enumerate(analyses) if index != verb_index and _is_ru_nominal(analysis)]
    if not candidates:
        return 0 if verb_index != 0 else min(1, len(analyses) - 1)

    meaningful = [index for index in candidates if not _is_ru_temporal(analyses[index])] or candidates
    before_verb = [index for index in meaningful if verb_index is None or index < verb_index]
    animate_before_verb = [index for index in before_verb if _is_ru_animate(analyses[index])]
    if animate_before_verb:
        return animate_before_verb[0]

    if verb_index is not None and _is_ru_transitive_verb(analyses[verb_index]):
        animate_anywhere = [index for index in meaningful if _is_ru_animate(analyses[index])]
        if animate_anywhere:
            return animate_anywhere[0]

    if before_verb:
        return before_verb[-1]
    return meaningful[0]


def _select_ru_object_indexes(
    analyses: list[dict[str, Any]],
    subject_index: int,
    verb_index: int | None,
) -> list[int]:
    if verb_index is not None and _is_ru_change_state_verb(analyses[verb_index]):
        return [
            index
            for index, analysis in enumerate(analyses)
            if index not in {subject_index, verb_index}
            and not _is_ru_temporal(analysis)
            and (_is_ru_adjective(analysis) or _is_ru_nominal(analysis))
        ]
    return [
        index
        for index, analysis in enumerate(analyses)
        if index not in {subject_index, verb_index}
        and not _is_ru_temporal(analysis)
        and (_is_ru_nominal(analysis) or _is_ru_adjective(analysis))
    ]


def _select_ru_modifier_indexes(
    analyses: list[dict[str, Any]],
    subject_index: int,
    verb_index: int | None,
    object_indexes: list[int],
) -> list[int]:
    used = {subject_index, *object_indexes}
    if verb_index is not None:
        used.add(verb_index)
    return [index for index, analysis in enumerate(analyses) if index not in used and _is_ru_temporal(analysis)]


def _is_ru_verb(analysis: dict[str, Any]) -> bool:
    return any(str(getattr(parse.tag, "POS", "") or "") in _RU_VERB_POS for parse in analysis["parses"])


def _is_ru_transitive_verb(analysis: dict[str, Any]) -> bool:
    return any("tran" in parse.tag for parse in analysis["parses"] if str(getattr(parse.tag, "POS", "") or "") in _RU_VERB_POS)


def _is_ru_change_state_verb(analysis: dict[str, Any]) -> bool:
    return analysis["normal_form"] in _RU_CHANGE_STATE_VERBS


def _is_ru_nominal(analysis: dict[str, Any]) -> bool:
    return any(str(getattr(parse.tag, "POS", "") or "") in _RU_NOMINAL_POS for parse in analysis["parses"])


def _is_ru_adjective(analysis: dict[str, Any]) -> bool:
    return any(str(getattr(parse.tag, "POS", "") or "") in _RU_ADJECTIVE_POS for parse in analysis["parses"])


def _is_ru_animate(analysis: dict[str, Any]) -> bool:
    return any("anim" in parse.tag for parse in analysis["parses"] if str(getattr(parse.tag, "POS", "") or "") in _RU_NOMINAL_POS)


def _is_ru_temporal(analysis: dict[str, Any]) -> bool:
    return analysis["normal_form"] in _RU_TEMPORAL_NOUNS


def _parse_ru(token: str, preferred_grammemes: set[str] | None = None):
    parsed = _morph().parse(token)
    if not parsed:
        return None
    if preferred_grammemes:
        if {"3per", "sing", "pres"}.issubset(preferred_grammemes):
            for item in parsed:
                if str(getattr(item.tag, "POS", "") or "") == "INFN" and item.inflect(preferred_grammemes):
                    return item
        for item in parsed:
            if preferred_grammemes.issubset(set(item.tag.grammemes)):
                return item
        for item in parsed:
            if item.inflect(preferred_grammemes):
                return item
    return parsed[0]


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


def _empty_morphology() -> dict[str, str | None]:
    return {"POS": None, "case": None, "number": None, "gender": None, "tense": None, "person": None}
