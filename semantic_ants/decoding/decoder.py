from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from semantic_ants.ants import AntColony, AntConfig
from semantic_ants.core.graph import SemanticGraph
from semantic_ants.core.models import ConceptNode, SemanticEdge
from semantic_ants.core.normalization import detect_language, normalize_text, text_to_concept_uri, tokenize
from semantic_ants.learning.checkpoint import Checkpoint

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


@dataclass(frozen=True)
class _RUAnalysis:
    index: int
    input_token: str
    normalized_token: str
    concept_uri: str
    parse: Any | None
    normal_form: str
    is_verb: bool
    is_nominal: bool
    is_adjective: bool
    is_animate: bool
    is_temporal: bool

    @property
    def pos(self) -> str | None:
        if self.parse is None:
            return None
        return str(getattr(self.parse.tag, "POS", "") or "") or None


@dataclass(frozen=True)
class _RolePlan:
    subject: int
    verb: int | None
    objects: tuple[int, ...] = ()
    instruments: tuple[int, ...] = ()
    locations: tuple[int, ...] = ()
    complements: tuple[int, ...] = ()
    modifiers: tuple[int, ...] = ()
    score: float = 0.0
    source: str = "heuristic"


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
_RU_DEVICE_HINTS = {
    "компьютер",
    "ноутбук",
    "монитор",
    "экран",
    "телефон",
    "планшет",
}
_RU_RELATIONS = {
    "subject": {"CanDo", "AgentOf", "Performs", "SubjectOf", "HasAgent"},
    "object": {"TakesObject", "ObjectOf", "HasObject", "Creates", "Produces", "Writes"},
    "instrument": {"UsesInstrument", "Instrument", "Uses", "With"},
    "location": {"AtLocation", "LocatedIn", "InLocation", "InTopDomain"},
    "complement": {"HasProperty", "PropertyOf", "Becomes", "Becoming"},
}


def decode_words(
    text: str,
    *,
    tokens: list[str] | None = None,
    lang: str = "auto",
    session_id: str | None = None,
    turn_id: str | None = None,
    checkpoint: Checkpoint | None = None,
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
        analyses = [_analyze_ru_token(index, token) for index, token in enumerate(input_tokens)]
        baseline = _decode_ru_heuristic(analyses)
        learned = _decode_ru_learned(analyses, checkpoint) if checkpoint is not None else None
        plan = _choose_plan(baseline, learned, analyses)
        decoded = _materialize_ru_plan(plan, analyses, checkpoint)
    elif selected_lang == "en":
        decoded = _decode_en(input_tokens)
    else:
        decoded = _decode_surface(input_tokens, selected_lang)

    sentence = _join_sentence(
        decoded["subject"],
        decoded["verb"],
        decoded["objects"],
        selected_lang,
        decoded.get("modifiers", []),
        decoded.get("instruments", []),
        decoded.get("locations", []),
        decoded.get("complements", []),
    )
    output_tokens = decoded["tokens"]
    summary = DecodeSummary(
        total_tokens=len(input_tokens),
        used_tokens=len(output_tokens),
        objects=len(decoded["objects"]),
        fallbacks=sum(1 for token in output_tokens if token.transform_status == "fallback"),
    )
    return DecodeResult(
        input_text=input_text,
        input_tokens=input_tokens,
        lang=selected_lang,
        sentence=sentence,
        pattern=decoded.get("pattern", "svo"),
        session_id=session_id,
        turn_id=turn_id,
        tokens=output_tokens,
        summary=summary,
    )


def _choose_plan(baseline: _RolePlan, learned: _RolePlan | None, analyses: list[_RUAnalysis]) -> _RolePlan:
    if learned is None:
        return baseline
    if learned.score >= baseline.score + 0.2:
        return learned
    return baseline


def _decode_ru_heuristic(analyses: list[_RUAnalysis]) -> _RolePlan:
    if not analyses:
        return _RolePlan(subject=0, verb=None, score=0.0, source="heuristic")

    verb_index = _select_ru_verb_index(analyses)
    subject_index = _select_ru_subject_index(analyses, verb_index)
    object_indexes = _select_ru_object_indexes(analyses, subject_index, verb_index)
    modifier_indexes = _select_ru_modifier_indexes(analyses, subject_index, verb_index, object_indexes)
    verb = analyses[verb_index] if verb_index is not None else None

    complements: list[int] = []
    instruments: list[int] = []
    locations: list[int] = []
    for index in object_indexes:
        analysis = analyses[index]
        if verb and _is_change_state_verb(verb) and (analysis.is_adjective or analysis.is_nominal):
            complements.append(index)
            continue
        if _is_instrument_candidate(analysis, verb):
            instruments.append(index)
            continue
        if _is_location_candidate(analysis, verb):
            locations.append(index)
            continue
    objects = [index for index in object_indexes if index not in complements and index not in instruments and index not in locations]
    score = 1.0 + float(len(objects) + len(instruments) + len(locations) + len(complements)) * 0.2
    return _RolePlan(
        subject=subject_index,
        verb=verb_index,
        objects=tuple(objects),
        instruments=tuple(instruments),
        locations=tuple(locations),
        complements=tuple(complements),
        modifiers=tuple(modifier_indexes),
        score=score,
        source="heuristic",
    )


def _decode_ru_learned(analyses: list[_RUAnalysis], checkpoint: Checkpoint | None) -> _RolePlan | None:
    if not analyses:
        return None
    verb_candidates = _top_indices(
        analyses,
        key=lambda analysis: _verb_score(analysis, analyses, checkpoint),
        count=3,
    )
    best_plan: _RolePlan | None = None
    for verb_candidate in verb_candidates:
        subject_candidates = _top_indices(
            [analysis for analysis in analyses if analysis.index != verb_candidate.index],
            key=lambda analysis: _subject_score(analysis, verb_candidate, analyses, checkpoint),
            count=3,
        )
        if not subject_candidates:
            continue
        for subject in subject_candidates:
            plan = _build_learned_plan(analyses, checkpoint, verb_candidate.index, subject.index)
            if best_plan is None or plan.score > best_plan.score:
                best_plan = plan
    return best_plan


def _build_learned_plan(
    analyses: list[_RUAnalysis],
    checkpoint: Checkpoint | None,
    verb_index: int,
    subject_index: int,
) -> _RolePlan:
    verb = analyses[verb_index]
    selected_roles: dict[str, list[int]] = {"objects": [], "instruments": [], "locations": [], "complements": [], "modifiers": []}
    subject = analyses[subject_index]
    for analysis in analyses:
        if analysis.index in {verb_index, subject_index}:
            continue
        role = _best_role_for_token(analysis, verb, subject, checkpoint)
        if role == "modifier":
            selected_roles["modifiers"].append(analysis.index)
        elif role == "complement":
            selected_roles["complements"].append(analysis.index)
        elif role == "instrument":
            selected_roles["instruments"].append(analysis.index)
        elif role == "location":
            selected_roles["locations"].append(analysis.index)
        else:
            selected_roles["objects"].append(analysis.index)

    route_bonus = _ant_bonus_for_plan(analyses, checkpoint, verb_index, subject_index, selected_roles)
    direct_score = _plan_direct_score(analyses, checkpoint, verb_index, subject_index, selected_roles)
    score = direct_score + route_bonus
    return _RolePlan(
        subject=subject_index,
        verb=verb_index,
        objects=tuple(selected_roles["objects"]),
        instruments=tuple(selected_roles["instruments"]),
        locations=tuple(selected_roles["locations"]),
        complements=tuple(selected_roles["complements"]),
        modifiers=tuple(selected_roles["modifiers"]),
        score=score,
        source="learned",
    )


def _plan_direct_score(
    analyses: list[_RUAnalysis],
    checkpoint: Checkpoint | None,
    verb_index: int,
    subject_index: int,
    selected_roles: dict[str, list[int]],
) -> float:
    verb = analyses[verb_index]
    subject = analyses[subject_index]
    score = _verb_score(verb, analyses, checkpoint)
    score += _subject_score(subject, verb, analyses, checkpoint)
    for index in selected_roles["objects"]:
        score += _object_score(analyses[index], verb, subject, checkpoint)
    for index in selected_roles["instruments"]:
        score += _instrument_score(analyses[index], verb, checkpoint)
    for index in selected_roles["locations"]:
        score += _location_score(analyses[index], verb, checkpoint)
    for index in selected_roles["complements"]:
        score += _complement_score(analyses[index], verb, subject, checkpoint)
    for index in selected_roles["modifiers"]:
        score += _modifier_score(analyses[index], verb, checkpoint)
    score += _verb_subject_relation_bonus(subject, verb, checkpoint)
    return score


def _ant_bonus_for_plan(
    analyses: list[_RUAnalysis],
    checkpoint: Checkpoint | None,
    verb_index: int,
    subject_index: int,
    selected_roles: dict[str, list[int]],
) -> float:
    local_checkpoint = checkpoint or Checkpoint()
    graph = SemanticGraph()
    selected_indexes = [subject_index, verb_index]
    for role in ("objects", "instruments", "locations", "complements", "modifiers"):
        selected_indexes.extend(selected_roles[role])
    selected_indexes = list(dict.fromkeys(selected_indexes))
    role_nodes = {
        "subject": "/m/decode/role/subject",
        "object": "/m/decode/role/object",
        "instrument": "/m/decode/role/instrument",
        "location": "/m/decode/role/location",
        "complement": "/m/decode/role/complement",
        "modifier": "/m/decode/role/modifier",
        "verb": "/m/decode/role/verb",
    }
    for role_uri in role_nodes.values():
        graph.add_node(ConceptNode(uri=role_uri, label=role_uri.rsplit("/", 1)[-1], language="unknown", source="decode"))
    for index in selected_indexes:
        analysis = analyses[index]
        graph.add_node(
            ConceptNode(
                uri=analysis.concept_uri,
                label=analysis.normalized_token,
                language="ru",
                source="decode",
            )
        )
    graph.add_edge(
        SemanticEdge(
            start=analyses[verb_index].concept_uri,
            end=role_nodes["subject"],
            relation="DecodeRole",
            weight=max(_subject_score(analyses[subject_index], analyses[verb_index], analyses, checkpoint), 0.1),
            source="decode",
        )
    )
    graph.add_edge(
        SemanticEdge(
            start=role_nodes["subject"],
            end=analyses[subject_index].concept_uri,
            relation="DecodeToken",
            weight=max(_subject_score(analyses[subject_index], analyses[verb_index], analyses, checkpoint), 0.1),
            source="decode",
        )
    )
    for role_name, indexes in selected_roles.items():
        for index in indexes:
            analysis = analyses[index]
            weight = max(_role_weight_for_plan_token(role_name, analysis, analyses[verb_index], analyses, checkpoint), 0.1)
            graph.add_edge(
                SemanticEdge(
                    start=analyses[verb_index].concept_uri,
                    end=role_nodes[role_name[:-1] if role_name.endswith("s") else role_name],
                    relation="DecodeRole",
                    weight=max(weight * 0.5, 0.1),
                    source="decode",
                )
            )
            graph.add_edge(
                SemanticEdge(
                    start=role_nodes[role_name[:-1] if role_name.endswith("s") else role_name],
                    end=analysis.concept_uri,
                    relation="DecodeToken",
                    weight=weight,
                    source="decode",
                )
            )
    colony = AntColony(
        AntConfig(
            ant_count=max(8, len(selected_indexes) * 4),
            max_depth=2,
            seed=local_checkpoint.seed,
            exploration=0.05,
        )
    )
    routes = colony.search(
        graph,
        [analyses[verb_index].concept_uri, analyses[subject_index].concept_uri],
        local_checkpoint,
        context_key="decode:" + ":".join(analysis.concept_uri for analysis in analyses),
    )
    return sum(route.total_score for route in routes[:6])


def _materialize_ru_plan(
    plan: _RolePlan,
    analyses: list[_RUAnalysis],
    checkpoint: Checkpoint | None,
) -> dict[str, Any]:
    subject = _ru_token(analyses[plan.subject], "subject", {"nomn", "sing"}, "inflected", checkpoint)
    verb = (
        _ru_token(analyses[plan.verb], "verb", {"3per", "sing", "pres"}, "inflected", checkpoint)
        if plan.verb is not None
        else None
    )
    objects = [
        _ru_token(analyses[index], "object", {"accs", "sing"}, "inflected", checkpoint)
        for index in plan.objects
    ]
    complements = [
        _ru_complement_token(analyses[index], subject, checkpoint)
        for index in plan.complements
    ]
    instruments = [_ru_instrument_token(analyses[index], checkpoint) for index in plan.instruments]
    locations = [
        _ru_prepositional_token(analyses[index], "location", "в", checkpoint)
        for index in plan.locations
    ]
    modifiers = [
        _ru_token(analyses[index], "modifier", {"ablt", "sing"}, "inflected", checkpoint)
        for index in plan.modifiers
    ]
    tokens = [*modifiers, subject, *([verb] if verb else []), *objects, *complements, *instruments, *locations]
    pattern = "svo"
    if verb is None:
        pattern = "s"
    elif instruments:
        pattern = "svoi"
    elif complements:
        pattern = "svoc"
    elif modifiers:
        pattern = "svm"
    return {
        "subject": subject,
        "verb": verb,
        "objects": objects,
        "complements": complements,
        "instruments": instruments,
        "locations": locations,
        "modifiers": modifiers,
        "tokens": tokens,
        "pattern": pattern,
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
        "complements": [],
        "instruments": [],
        "locations": [],
        "modifiers": [],
        "tokens": tokens,
        "pattern": "svo",
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
        "complements": [],
        "instruments": [],
        "locations": [],
        "modifiers": [],
        "tokens": tokens,
        "pattern": "svo",
    }


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


def _ru_token(
    analysis: _RUAnalysis,
    role: str,
    grammemes: set[str],
    success_status: str,
    checkpoint: Checkpoint | None,
) -> DecodeToken:
    raw = analysis.input_token
    normalized = analysis.normalized_token
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
    if role in {"instrument", "location"}:
        surface = _surface_with_preposition(analysis, role, checkpoint, surface)
    concept_uri = _concept_uri(normalized, "ru")
    return DecodeToken(
        input_token=raw,
        normalized_token=normalized,
        role=role,
        surface=surface,
        concept_uri=concept_uri,
        transform_status=status,
        morphology=morphology,
    )


def _ru_prepositional_token(
    analysis: _RUAnalysis,
    role: str,
    preposition: str,
    checkpoint: Checkpoint | None,
) -> DecodeToken:
    raw = analysis.input_token
    normalized = analysis.normalized_token
    parsed = _parse_ru(raw, {"loct", "sing"}) or _parse_ru(raw, {"loct"}) or _parse_ru(raw, {"sing"})
    surface_word = normalized
    morphology = _morphology_from_parse(parsed) if parsed else _empty_morphology()
    status = "fallback"
    if parsed:
        inflected = parsed.inflect({"loct", "sing"})
        if inflected and inflected.word:
            surface_word = _normalized_token(inflected.word)
            morphology = _morphology_from_parse(inflected)
            status = "inflected"
        else:
            surface_word = _normalized_token(parsed.word or parsed.normal_form or normalized)
    surface = f"{preposition} {surface_word}".strip()
    concept_uri = _concept_uri(normalized, "ru")
    return DecodeToken(
        input_token=raw,
        normalized_token=normalized,
        role=role,
        surface=surface,
        concept_uri=concept_uri,
        transform_status=status,
        morphology=morphology,
    )


def _ru_instrument_token(analysis: _RUAnalysis, checkpoint: Checkpoint | None) -> DecodeToken:
    if analysis.normal_form in _RU_DEVICE_HINTS:
        return _ru_prepositional_token(analysis, "instrument", "на", checkpoint)
    raw = analysis.input_token
    normalized = analysis.normalized_token
    parsed = _parse_ru(raw, {"ablt", "sing"}) or _parse_ru(raw, {"ablt"}) or _parse_ru(raw, {"sing"})
    surface_word = normalized
    morphology = _morphology_from_parse(parsed) if parsed else _empty_morphology()
    status = "fallback"
    if parsed:
        inflected = parsed.inflect({"ablt", "sing"})
        if inflected and inflected.word:
            surface_word = _normalized_token(inflected.word)
            morphology = _morphology_from_parse(inflected)
            status = "inflected"
        else:
            surface_word = _normalized_token(parsed.word or parsed.normal_form or normalized)
    return DecodeToken(
        input_token=raw,
        normalized_token=normalized,
        role="instrument",
        surface=f"с {surface_word}".strip(),
        concept_uri=_concept_uri(normalized, "ru"),
        transform_status=status,
        morphology=morphology,
    )


def _ru_complement_token(analysis: _RUAnalysis, subject: DecodeToken, checkpoint: Checkpoint | None) -> DecodeToken:
    grammemes = {"ablt"}
    if subject.morphology.get("number"):
        grammemes.add(subject.morphology["number"])
    if subject.morphology.get("gender"):
        grammemes.add(subject.morphology["gender"])
    return _ru_token(analysis, "complement", grammemes, "inflected", checkpoint)


def _surface_with_preposition(
    analysis: _RUAnalysis,
    role: str,
    checkpoint: Checkpoint | None,
    surface: str,
) -> str:
    if role == "instrument":
        preposition = "на" if analysis.normal_form in _RU_DEVICE_HINTS or not analysis.is_animate else "с"
    elif role == "location":
        preposition = "в"
    else:
        preposition = ""
    if not preposition:
        return surface
    return f"{preposition} {surface}".strip()


def _join_sentence(
    subject: DecodeToken,
    verb: DecodeToken | None,
    objects: list[DecodeToken],
    lang: str,
    modifiers: list[DecodeToken] | None = None,
    instruments: list[DecodeToken] | None = None,
    locations: list[DecodeToken] | None = None,
    complements: list[DecodeToken] | None = None,
) -> str:
    parts = [token.surface for token in modifiers or []]
    parts.append(subject.surface)
    if verb is not None and verb.surface:
        parts.append(verb.surface)
    if objects:
        parts.append(_join_objects([token.surface for token in objects], lang))
    if complements:
        parts.append(_join_objects([token.surface for token in complements], lang))
    if instruments:
        parts.append(_join_objects([token.surface for token in instruments], lang))
    if locations:
        parts.append(_join_objects([token.surface for token in locations], lang))
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


def _analyze_ru_token(index: int, token: str) -> _RUAnalysis:
    raw = str(token).strip()
    normalized = _normalized_token(raw)
    parses = _morph().parse(raw)
    parse = _best_ru_parse(parses, None)
    if parse is None and parses:
        parse = parses[0]
    normal_form = str(getattr(parse, "normal_form", "") or normalized)
    pos = str(getattr(getattr(parse, "tag", None), "POS", "") or "") if parse else ""
    concept_uri = _concept_uri(normalized, "ru") or ""
    return _RUAnalysis(
        index=index,
        input_token=raw,
        normalized_token=normalized,
        concept_uri=concept_uri,
        parse=parse,
        normal_form=normal_form,
        is_verb=pos in _RU_VERB_POS,
        is_nominal=pos in _RU_NOMINAL_POS,
        is_adjective=pos in _RU_ADJECTIVE_POS,
        is_animate=bool(parse and "anim" in parse.tag and pos in _RU_NOMINAL_POS),
        is_temporal=normal_form in _RU_TEMPORAL_NOUNS,
    )


def _best_ru_parse(parses: list[Any], preferred_grammemes: set[str] | None) -> Any | None:
    if not parses:
        return None
    if preferred_grammemes:
        if {"3per", "sing", "pres"}.issubset(preferred_grammemes):
            for item in parses:
                if str(getattr(item.tag, "POS", "") or "") == "INFN" and item.inflect(preferred_grammemes):
                    return item
        for item in parses:
            if preferred_grammemes.issubset(set(item.tag.grammemes)):
                return item
        for item in parses:
            if item.inflect(preferred_grammemes):
                return item
    return parses[0]


def _parse_ru(token: str, preferred_grammemes: set[str] | None = None):
    parses = _morph().parse(token)
    return _best_ru_parse(parses, preferred_grammemes)


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


@lru_cache(maxsize=1)
def _morph() -> "pymorphy3.MorphAnalyzer":
    return pymorphy3.MorphAnalyzer()


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


def _select_ru_verb_index(analyses: list[_RUAnalysis]) -> int | None:
    if len(analyses) < 2:
        return None
    scored = sorted(analyses, key=lambda analysis: _verb_score(analysis, analyses, None), reverse=True)
    if not scored:
        return None
    if scored[0].is_verb or scored[0].normal_form in _RU_CHANGE_STATE_VERBS:
        return scored[0].index
    for analysis in scored:
        if analysis.is_verb:
            return analysis.index
    return 1 if len(analyses) > 1 else 0


def _select_ru_subject_index(analyses: list[_RUAnalysis], verb_index: int | None) -> int:
    if not analyses:
        return 0
    candidates = [analysis for analysis in analyses if analysis.index != verb_index and analysis.is_nominal]
    if not candidates:
        return 0 if verb_index != 0 else min(1, len(analyses) - 1)
    before_verb = [analysis for analysis in candidates if verb_index is None or analysis.index < verb_index]
    animate_before = [analysis for analysis in before_verb if analysis.is_animate]
    if animate_before:
        return animate_before[0].index
    if verb_index is not None and _is_transitive_verb(analyses[verb_index]):
        animate_anywhere = [analysis for analysis in candidates if analysis.is_animate]
        if animate_anywhere:
            return animate_anywhere[0].index
    if before_verb:
        return before_verb[-1].index
    return candidates[0].index


def _select_ru_object_indexes(
    analyses: list[_RUAnalysis],
    subject_index: int,
    verb_index: int | None,
) -> list[int]:
    return [
        analysis.index
        for analysis in analyses
        if analysis.index not in {subject_index, verb_index}
        and not analysis.is_temporal
        and (analysis.is_nominal or analysis.is_adjective)
    ]


def _select_ru_modifier_indexes(
    analyses: list[_RUAnalysis],
    subject_index: int,
    verb_index: int | None,
    object_indexes: list[int],
) -> list[int]:
    used = {subject_index, *object_indexes}
    if verb_index is not None:
        used.add(verb_index)
    return [analysis.index for analysis in analyses if analysis.index not in used and analysis.is_temporal]


def _decode_ru_learned_score(
    analyses: list[_RUAnalysis],
    checkpoint: Checkpoint | None,
    verb: _RUAnalysis,
    subject: _RUAnalysis,
    role: str,
    candidate: _RUAnalysis,
) -> float:
    if role == "subject":
        return _subject_score(candidate, verb, analyses, checkpoint)
    if role == "object":
        return _object_score(candidate, verb, subject, checkpoint)
    if role == "instrument":
        return _instrument_score(candidate, verb, checkpoint)
    if role == "location":
        return _location_score(candidate, verb, checkpoint)
    if role == "complement":
        return _complement_score(candidate, verb, subject, checkpoint)
    if role == "modifier":
        return _modifier_score(candidate, verb, checkpoint)
    return 0.0


def _best_role_for_token(
    candidate: _RUAnalysis,
    verb: _RUAnalysis,
    subject: _RUAnalysis,
    checkpoint: Checkpoint | None,
) -> str:
    role_scores = {
        "modifier": _modifier_score(candidate, verb, checkpoint),
        "complement": _complement_score(candidate, verb, subject, checkpoint),
        "instrument": _instrument_score(candidate, verb, checkpoint),
        "location": _location_score(candidate, verb, checkpoint),
        "object": _object_score(candidate, verb, subject, checkpoint),
    }
    best_role, best_score = "object", role_scores["object"]
    for role, score in role_scores.items():
        if score > best_score:
            best_role, best_score = role, score
    return best_role


def _role_weight_for_plan_token(
    role_name: str,
    candidate: _RUAnalysis,
    verb: _RUAnalysis,
    subject: _RUAnalysis,
    checkpoint: Checkpoint | None,
) -> float:
    role = role_name[:-1] if role_name.endswith("s") else role_name
    return _decode_ru_learned_score([], checkpoint, verb, subject, role, candidate)


def _top_indices(items: list[_RUAnalysis], key: Any, count: int) -> list[_RUAnalysis]:
    ordered = sorted(items, key=key, reverse=True)
    return ordered[: max(count, 1)]


def _verb_score(analysis: _RUAnalysis, analyses: list[_RUAnalysis], checkpoint: Checkpoint | None) -> float:
    score = 0.0
    if analysis.is_verb:
        score += 2.5
    if analysis.pos == "INFN":
        score += 1.8
    if analysis.normal_form in _RU_CHANGE_STATE_VERBS:
        score += 1.2
    if analysis.index == 1:
        score += 0.25
    if checkpoint is not None:
        score += min(checkpoint.concept_pheromone_for(analysis.concept_uri), 5.0) * 0.08
        for other in analyses:
            if other.index == analysis.index:
                continue
            score += _relation_support(analysis.concept_uri, other.concept_uri, "object", checkpoint) * 0.15
            score += _relation_support(analysis.concept_uri, other.concept_uri, "instrument", checkpoint) * 0.1
    return score


def _subject_score(
    candidate: _RUAnalysis,
    verb: _RUAnalysis,
    analyses: list[_RUAnalysis],
    checkpoint: Checkpoint | None,
) -> float:
    score = 0.0
    if candidate.is_nominal:
        score += 1.8
    if candidate.is_animate:
        score += 0.8
    if candidate.index < verb.index:
        score += 0.45
    if candidate.is_temporal:
        score -= 1.5
    if checkpoint is not None:
        score += min(checkpoint.concept_pheromone_for(candidate.concept_uri), 5.0) * 0.08
        score += _relation_support(candidate.concept_uri, verb.concept_uri, "subject", checkpoint) * 2.2
        score += _relation_support(verb.concept_uri, candidate.concept_uri, "subject", checkpoint) * 0.7
        if _is_transitive_verb(verb):
            score += 0.25 if candidate.is_animate else -0.1
    return score


def _object_score(
    candidate: _RUAnalysis,
    verb: _RUAnalysis,
    subject: _RUAnalysis,
    checkpoint: Checkpoint | None,
) -> float:
    score = 0.0
    if candidate.is_nominal or candidate.is_adjective:
        score += 1.2
    if candidate.index > verb.index:
        score += 0.35
    if candidate.is_temporal:
        score -= 1.4
    if candidate.index == subject.index:
        score -= 5.0
    if checkpoint is not None:
        score += min(checkpoint.concept_pheromone_for(candidate.concept_uri), 5.0) * 0.08
        score += _relation_support(verb.concept_uri, candidate.concept_uri, "object", checkpoint) * 2.6
        score += _relation_support(candidate.concept_uri, verb.concept_uri, "object", checkpoint) * 0.5
    return score


def _instrument_score(candidate: _RUAnalysis, verb: _RUAnalysis, checkpoint: Checkpoint | None) -> float:
    score = 0.0
    if candidate.is_nominal:
        score += 1.0
    if candidate.normal_form in _RU_DEVICE_HINTS:
        score += 0.9
    if candidate.index > verb.index:
        score += 0.15
    if checkpoint is not None:
        score += min(checkpoint.concept_pheromone_for(candidate.concept_uri), 5.0) * 0.07
        score += _relation_support(verb.concept_uri, candidate.concept_uri, "instrument", checkpoint) * 2.7
        score += _relation_support(candidate.concept_uri, verb.concept_uri, "instrument", checkpoint) * 0.4
    return score


def _location_score(candidate: _RUAnalysis, verb: _RUAnalysis, checkpoint: Checkpoint | None) -> float:
    score = 0.0
    if candidate.is_nominal:
        score += 0.8
    if candidate.index > verb.index:
        score += 0.12
    if checkpoint is not None:
        score += min(checkpoint.concept_pheromone_for(candidate.concept_uri), 5.0) * 0.06
        score += _relation_support(verb.concept_uri, candidate.concept_uri, "location", checkpoint) * 2.5
        score += _relation_support(candidate.concept_uri, verb.concept_uri, "location", checkpoint) * 0.4
    return score


def _complement_score(
    candidate: _RUAnalysis,
    verb: _RUAnalysis,
    subject: _RUAnalysis,
    checkpoint: Checkpoint | None,
) -> float:
    score = 0.0
    if candidate.is_adjective:
        score += 1.6
    elif candidate.is_nominal:
        score += 0.8
    if _is_change_state_verb(verb):
        score += 1.1
    if checkpoint is not None:
        score += min(checkpoint.concept_pheromone_for(candidate.concept_uri), 5.0) * 0.07
        score += _relation_support(verb.concept_uri, candidate.concept_uri, "complement", checkpoint) * 2.5
        score += _relation_support(subject.concept_uri, candidate.concept_uri, "complement", checkpoint) * 0.3
    return score


def _modifier_score(candidate: _RUAnalysis, verb: _RUAnalysis, checkpoint: Checkpoint | None) -> float:
    score = 0.0
    if candidate.is_temporal:
        score += 2.0
    if candidate.index < verb.index:
        score += 0.15
    if checkpoint is not None:
        score += min(checkpoint.concept_pheromone_for(candidate.concept_uri), 5.0) * 0.05
    return score


def _relation_support(
    start_uri: str,
    end_uri: str,
    role: str,
    checkpoint: Checkpoint,
) -> float:
    if not start_uri or not end_uri:
        return 0.0
    supported = 0.0
    for edge in checkpoint.learned_edges():
        if edge.start == start_uri and edge.end == end_uri and edge.relation in _RU_RELATIONS[role]:
            supported = max(supported, _edge_strength(edge, checkpoint))
        if edge.start == end_uri and edge.end == start_uri and edge.relation in _RU_RELATIONS[role]:
            supported = max(supported, _edge_strength(edge, checkpoint) * 0.7)
    return supported


def _edge_strength(edge: SemanticEdge, checkpoint: Checkpoint) -> float:
    pheromone = checkpoint.pheromone_for(edge)
    concept_pheromone = checkpoint.concept_pheromone_for(edge.end)
    penalty = checkpoint.penalty_for(edge.end)
    return max(edge.weight, 0.01) * pheromone * concept_pheromone * penalty / max(float(edge.distance), 0.25)


def _verb_subject_relation_bonus(subject: _RUAnalysis, verb: _RUAnalysis, checkpoint: Checkpoint | None) -> float:
    if checkpoint is None:
        return 0.0
    return _relation_support(subject.concept_uri, verb.concept_uri, "subject", checkpoint) * 1.5


def _is_change_state_verb(verb: _RUAnalysis) -> bool:
    return verb.normal_form in _RU_CHANGE_STATE_VERBS


def _is_transitive_verb(verb: _RUAnalysis) -> bool:
    return verb.pos in {"INFN", "VERB"} and verb.normal_form not in {"быть"}


def _is_instrument_candidate(candidate: _RUAnalysis, verb: _RUAnalysis | None) -> bool:
    if candidate.is_temporal or not candidate.is_nominal:
        return False
    if candidate.normal_form in _RU_DEVICE_HINTS:
        return True
    if verb is not None and _is_change_state_verb(verb):
        return False
    return False


def _is_location_candidate(candidate: _RUAnalysis, verb: _RUAnalysis | None) -> bool:
    if candidate.is_temporal or not candidate.is_nominal:
        return False
    if verb is not None and _is_change_state_verb(verb):
        return False
    return False
