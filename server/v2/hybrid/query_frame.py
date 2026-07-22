"""Deterministic query-frame construction and conservative context inheritance."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

from .contracts import Gap, QueryFrame, clamp

TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)
QUESTION_WORDS = {
    "кто": "кто", "что": "что", "чем": "чем", "кого": "кого", "кому": "кому",
    "какой": "какой", "какая": "какая", "какое": "какое", "какие": "какие",
    "где": "где", "куда": "куда", "откуда": "откуда", "когда": "когда",
    "сколько": "сколько", "почему": "почему", "как": "как", "чей": "чей",
    "чья": "чья", "чьё": "чьё", "чьи": "чьи",
}
QUESTION_FEATURE_CONSTRAINTS = {
    "кто": (("case", "nomn"),),
    "кому": (("case", "datv"),),
    "чем": (("case", "ablt"),),
    "где": (("case", "loct"),),
    "куда": (("case", "accs"),),
    "откуда": (("case", "gent"),),
}
STOPWORDS = {
    "а", "и", "или", "но", "же", "ли", "это", "этот", "эта", "эти", "так",
    "этого", "этому", "этим", "него", "нему", "ним", "неё", "нее", "ней",
    "был", "была", "было", "были", "будет", "будут", "есть", "быть", "в", "во", "на", "из", "с", "со", "у",
    "к", "ко", "от", "для", "по", "за", "под", "над", "при", "про", "о",
    "об", "обо", "не", "ни", "что", "кроме", "без", "ещё", "еще", "другой",
    "другая", "другое", "другие", "потом", "тогда", "внутри", "после", "до",
}
NEGATION_WORDS = {"не", "ни", "никто", "ничего", "никогда", "нет"}
CONTINUATION_WORDS = {"ещё", "еще", "другой", "другая", "другое", "другие", "кроме", "потом", "тогда"}
RELATION_WORDS = {"внутри", "снаружи", "около", "рядом", "между", "после", "до", "под", "над", "в", "на"}


def _normalise(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.casefold().replace("ё", "е")))


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(_normalise(text))


def _analysis_tokens(analysis: Any) -> list[Mapping[str, Any]]:
    if isinstance(analysis, Mapping):
        return [item for item in analysis.get("tokens") or () if isinstance(item, Mapping)]
    return []


def _analysis_query_graph(analysis: Any) -> Mapping[str, Any]:
    if not isinstance(analysis, Mapping):
        return {}
    graph = analysis.get("query_graph") or analysis.get("event_pattern") or {}
    if not isinstance(graph, Mapping):
        return {}
    pattern = graph.get("event_pattern") if isinstance(graph.get("event_pattern"), Mapping) else graph
    return pattern if isinstance(pattern, Mapping) else {}


def _graph_known_elements(analysis: Any) -> list[str]:
    pattern = _analysis_query_graph(analysis)
    values: list[str] = []
    for node in pattern.get("known_nodes") or ():
        if not isinstance(node, Mapping):
            continue
        head = node.get("head") if isinstance(node.get("head"), Mapping) else {}
        value = head.get("lemma") or node.get("lemma") or node.get("surface")
        if value and str(value) not in values:
            values.append(str(value))
    return values


def _graph_predicate(analysis: Any) -> Optional[str]:
    pattern = _analysis_query_graph(analysis)
    predicate = pattern.get("predicate") if isinstance(pattern.get("predicate"), Mapping) else {}
    value = predicate.get("lemma") or predicate.get("surface")
    return str(value) if value else None


def _graph_gap_constraints(analysis: Any, index: int) -> tuple[Mapping[str, Any], ...]:
    pattern = _analysis_query_graph(analysis)
    gaps = pattern.get("target_gaps") or ()
    if not isinstance(gaps, Sequence) or isinstance(gaps, (str, bytes)) or index >= len(gaps):
        return ()
    gap = gaps[index]
    if not isinstance(gap, Mapping):
        return ()
    hypotheses = gap.get("morphology_hypotheses") or {}
    if not isinstance(hypotheses, Mapping):
        return ()
    constraints: list[Mapping[str, Any]] = []
    for key, confidence in sorted(hypotheses.items()):
        namespace, separator, value = str(key).partition(":")
        if namespace != "case" or not separator or not value:
            continue
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            continue
        if score >= 0.5:
            constraints.append({"type": "grammatical_feature", "feature": namespace, "value": value, "confidence": score})
    return tuple(constraints)


def _surface_gap_constraints(focus: str) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {"type": "grammatical_feature", "feature": feature, "value": value, "confidence": 0.75}
        for feature, value in QUESTION_FEATURE_CONSTRAINTS.get(focus, ())
    )


def _predicate(tokens: Sequence[str], analysis: Any = None) -> Optional[str]:
    graph_predicate = _graph_predicate(analysis)
    if graph_predicate:
        return graph_predicate
    for item in _analysis_tokens(analysis):
        pos = str(item.get("pos") or item.get("part_of_speech") or "").upper()
        lemma = str(item.get("lemma") or item.get("normalized") or "")
        if pos in {"VERB", "INFN", "PRTF", "PRTS", "VERB_FORM"} and lemma:
            return lemma
    # A deliberately broad grammatical heuristic. It is not a domain lexicon.
    for token in tokens:
        if token in QUESTION_WORDS or token in STOPWORDS:
            continue
        if token.endswith(("ть", "ет", "ит", "ут", "ют", "ал", "ил", "ел", "ла", "ли", "ло", "ли")) or (len(token) > 4 and token.endswith("л")):
            return token
    return None


def _known_elements(tokens: Sequence[str], question_tokens: Sequence[str], predicate: Optional[str]) -> list[str]:
    result: list[str] = []
    for token in tokens:
        if token in question_tokens or token in STOPWORDS or token == predicate or len(token) < 2:
            continue
        if token not in result:
            result.append(token)
    return result


def _query_type(focuses: Sequence[str], tokens: Sequence[str], predicate: Optional[str]) -> str:
    if len(focuses) > 1:
        return "multi_gap_question"
    if not focuses:
        if set(tokens) & CONTINUATION_WORDS:
            return "continuation_question"
        return "statement_or_open_query"
    if any(item in RELATION_WORDS for item in tokens):
        return "continuation_relation_question" if not predicate else "relation_question"
    if focuses[0] in {"кто", "что", "кого", "кому", "чем"}:
        return "entity_question" if not predicate else "event_attachment_question"
    return "attribute_question"


def build_query_frame(
    text: str,
    session_context: Optional[Mapping[str, Any]] = None,
    *,
    session_id: str = "",
    analysis: Any = None,
) -> QueryFrame:
    raw_text = str(text or "").strip()
    normalized = _normalise(raw_text)
    context = session_context or {}
    query_id = str(context.get("query_id") or "")
    if not query_id:
        import hashlib
        query_id = f"query_{hashlib.sha256(f'{session_id}|{normalized}'.encode()).hexdigest()[:20]}"
    tokens = _tokens(raw_text)
    question_tokens = [token for token in tokens if token in QUESTION_WORDS]
    focuses = [QUESTION_WORDS[token] for token in question_tokens]
    predicate = _predicate(tokens, analysis)
    graph_known = _graph_known_elements(analysis)
    known = graph_known or _known_elements(tokens, question_tokens, predicate)
    negations = [token for token in tokens if token in NEGATION_WORDS]
    excluded_values = [token for token in tokens[tokens.index("кроме") + 1:]] if "кроме" in tokens else []
    if "без" in tokens:
        excluded_values += tokens[tokens.index("без") + 1:]
    gaps = []
    for index, focus in enumerate(focuses):
        relation = next((token for token in tokens if token in RELATION_WORDS), None)
        gap_id = f"gap_{index + 1}_{query_id[-8:]}"
        gaps.append(Gap(
            gap_id=gap_id,
            source_query_id=query_id,
            expected_type="quantity" if focus == "сколько" else "entity",
            expected_relation=relation or ("predicate_attachment" if predicate else None),
            known_elements=tuple(known),
            surface_projection=focus,
            constraints=(
                tuple({"type": "negation", "value": item} for item in negations)
                + (_graph_gap_constraints(analysis, index) or _surface_gap_constraints(focus))
            ),
            exclusions=tuple(excluded_values),
        ))
    if not gaps and ("?" in raw_text or not predicate):
        gaps.append(Gap(
            gap_id=f"gap_1_{query_id[-8:]}", source_query_id=query_id,
            expected_type="unknown", known_elements=tuple(known),
            surface_projection="",
        ))
    continuation = bool(set(tokens) & CONTINUATION_WORDS)
    return QueryFrame(
        query_id=query_id,
        session_id=session_id or str(context.get("session_id") or ""),
        raw_text=raw_text,
        normalized_text=normalized,
        query_type=_query_type(focuses, tokens, predicate),
        explicit_predicate=predicate,
        surface_focus=focuses[0] if focuses else None,
        known_elements=tuple(known),
        gaps=tuple(gaps),
        constraints=tuple({"type": "negation", "value": item} for item in negations),
        negations=tuple(negations),
        exclusions=tuple(dict.fromkeys(excluded_values)),
        temporal_scope=context.get("temporal_scope"),
        continuation_of=(
            (
                (str((analysis or {}).get("query_graph", {}).get("continuation_of") or "") if isinstance(analysis, Mapping) else "")
                or str(context.get("last_query_id") or "")
            )
            or None
        ) if continuation or _graph_known_elements(analysis) else None,
        confidence=clamp(0.92 if focuses else 0.35),
    )


def inherit_context(
    query_frame: QueryFrame,
    dialogue_history: Optional[Mapping[str, Any] | Sequence[Mapping[str, Any]]] = None,
) -> QueryFrame:
    if isinstance(dialogue_history, Mapping):
        context = dialogue_history
    else:
        values = list(dialogue_history or ())
        context = values[-1] if values else {}
    previous = context.get("last_query_frame") or context.get("query_frame") or {}
    if hasattr(previous, "as_dict"):
        previous = previous.as_dict()
    if not isinstance(previous, Mapping):
        previous = {}
    active = context.get("active_context") or context.get("current_focus") or {}
    previous_answer = context.get("last_answer") or context.get("answer") or {}
    if hasattr(active, "as_dict"):
        active = active.as_dict()
    continuation = query_frame.continuation_of or bool(
        set(_tokens(query_frame.raw_text)) & CONTINUATION_WORDS
    )
    if not continuation and query_frame.known_elements:
        return query_frame
    inherited = list(query_frame.known_elements)
    previous_known = list(previous.get("known_elements") or ())
    if isinstance(active, Mapping):
        for key in ("element_id", "lemma", "value", "surface"):
            if active.get(key):
                previous_known.append(str(active[key]))
    previous_values = []
    if isinstance(previous_answer, Mapping):
        previous_values.extend(str(previous_answer.get(key)) for key in ("resolved_value", "surface_answer") if previous_answer.get(key))
        previous_values.extend(str(item) for item in previous_answer.get("resolved_values") or ())
        previous_values.extend(str(item) for item in (previous_answer.get("filled_gaps") or {}).values())
    inherited_exclusions = list(query_frame.exclusions)
    if set(_tokens(query_frame.raw_text)) & {"ещё", "еще", "кроме", "другой", "другая", "другое", "другие"}:
        for value in previous_values:
            if value and value not in query_frame.exclusions:
                inherited_exclusions.append(value)
    for item in previous_known:
        if item and item not in inherited:
            inherited.append(str(item))
    predicate = query_frame.explicit_predicate or (str(previous.get("explicit_predicate") or "") or None)
    unresolved = bool(continuation and not inherited and not predicate)
    gaps = tuple(
        Gap(
            **{**gap.as_dict(), "known_elements": tuple(inherited), "exclusions": tuple(dict.fromkeys(list(gap.exclusions) + inherited_exclusions + list(previous.get("exclusions") or ())))}
        )
        for gap in query_frame.gaps
    )
    inherited_records = tuple({"element": item, "source": "previous_answer_context"} for item in inherited if item not in query_frame.known_elements)
    reconstructed = query_frame.raw_text
    if inherited and (not query_frame.known_elements or query_frame.query_type == "continuation_relation_question"):
        reconstructed = f"{query_frame.raw_text} [{', '.join(inherited)}]"
    return QueryFrame(
        **{**query_frame.as_dict(), "explicit_predicate": predicate, "known_elements": tuple(inherited), "gaps": gaps,
           "continuation_of": query_frame.continuation_of or str(previous.get("query_id") or context.get("last_query_id") or "") or None,
           "exclusions": tuple(dict.fromkeys(inherited_exclusions)), "inherited_elements": inherited_records, "reconstructed_query": reconstructed,
           "unresolved_context": unresolved, "confidence": clamp(query_frame.confidence * (0.92 if inherited else 0.4))}
    )
