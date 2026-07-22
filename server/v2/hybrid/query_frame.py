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
CURRENT_WORDS = {"сейчас", "теперь", "ныне"}
HISTORICAL_WORDS = {"сначала", "раньше", "ранее", "прежде"}


def _normalise(text: Any) -> str:
    return " ".join(TOKEN_RE.findall(str(text or "").casefold().replace("ё", "е")))


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
        if _normalise(node.get("preposition")) in {"после", "до"}:
            continue
        value = head.get("lemma") or node.get("lemma") or node.get("surface")
        if value and str(value) not in values:
            values.append(str(value))
    return values


def _reference(node: Mapping[str, Any]) -> Mapping[str, Any]:
    head = node.get("head") if isinstance(node.get("head"), Mapping) else {}
    lemma = str(head.get("lemma") or node.get("lemma") or node.get("surface") or "")
    return {
        "node_id": str(node.get("node_id") or node.get("id") or ""),
        "concept_id": str(node.get("entity_id") or ""),
        "lemma_hypotheses": list((node.get("features") or {}).get("morphology_alternatives") or ()),
        "surface": str(node.get("surface") or head.get("surface") or lemma),
        "lemma": lemma,
        "morphology": dict(node.get("features") or {}),
        "relation_attachment": node.get("preposition") or None,
        "origin": str(node.get("origin") or "CURRENT_QUERY"),
        "confidence": float(node.get("context_confidence") or 0.91),
    }


def _reference_lemma(value: Any) -> str:
    if isinstance(value, Mapping):
        return _normalise(value.get("lemma") or value.get("surface"))
    return _normalise(value)


def _predicate_hypotheses_from_analysis(
    predicate_data: Mapping[str, Any],
    language_analysis: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Preserve all morphology-backed predicate lemmas without lexical hardcode."""
    values: dict[str, dict[str, Any]] = {}

    def add(lemma: Any, confidence: Any, *, source: str, selected: bool = False) -> None:
        normalized = _normalise(lemma)
        if not normalized:
            return
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0
        current = values.get(normalized)
        candidate = {
            "type": "morphological_predicate",
            "predicate": normalized,
            "lemma": normalized,
            "confidence": clamp(score),
            "selected": bool(selected),
            "source": source,
        }
        if current is None or candidate["confidence"] > current["confidence"]:
            values[normalized] = candidate
        elif selected:
            current["selected"] = True

    predicate_analysis = (
        language_analysis.get("predicate")
        if isinstance(language_analysis.get("predicate"), Mapping)
        else {}
    )
    for item in predicate_analysis.get("morphological_analyses") or ():
        if not isinstance(item, Mapping):
            continue
        if str(item.get("pos") or "").upper() not in {"VERB", "INFN", "PRTF", "PRTS", "GRND"}:
            continue
        add(
            item.get("lemma"),
            item.get("confidence", 0.0),
            source="LANGUAGE_MORPHOLOGY",
            selected=bool(item.get("selected")),
        )
    for clause in language_analysis.get("clauses") or ():
        if not isinstance(clause, Mapping):
            continue
        for item in clause.get("predicate_hypotheses") or ():
            if isinstance(item, Mapping):
                add(
                    item.get("lemma"),
                    item.get("confidence", 0.0),
                    source="CLAUSE_PREDICATE_HYPOTHESIS",
                    selected=bool(item.get("selected")),
                )
    add(
        predicate_data.get("lemma") or predicate_data.get("surface"),
        0.92 if not values else 0.55,
        source="QUERY_GRAPH_PREDICATE",
        selected=not values,
    )
    return tuple(
        sorted(values.values(), key=lambda item: (-float(item["confidence"]), str(item["predicate"])))
    )


def _operational_gap_kind(
    gap_data: Mapping[str, Any],
    operator: Mapping[str, Any] | None,
) -> str:
    """Keep unseen operators broad until successful evidence supports specialization."""
    kind = str(gap_data.get("gap_kind") or "ENTITY")
    evidence = gap_data.get("evidence") if isinstance(gap_data.get("evidence"), Mapping) else {}
    profile = evidence.get("learned_gap_profile") if isinstance(evidence.get("learned_gap_profile"), Mapping) else {}
    profile_status = str(profile.get("profile_status") or "UNSEEN").upper()
    support_count = int(profile.get("support_count") or 0)
    operator_type = str((operator or {}).get("operator_type") or "")
    if (
        kind == "EVENT_PROPERTY"
        and operator_type == "EVENT_ATTACHMENT"
        and (profile_status in {"UNSEEN", "SHADOW"} or support_count < 3)
    ):
        return "EVENT_ATTACHMENT"
    return kind


def from_query_graph(
    query_graph: Mapping[str, Any],
    language_analysis: Mapping[str, Any] | None = None,
    dialogue_context: Mapping[str, Any] | None = None,
) -> QueryFrame:
    """Project the canonical query graph into the operational query frame."""
    graph = dict(query_graph or {})
    pattern = graph.get("event_pattern") if isinstance(graph.get("event_pattern"), Mapping) else graph
    analysis = dict(language_analysis or {})
    context = dict(dialogue_context or {})
    utterance = analysis.get("utterance") if isinstance(analysis.get("utterance"), Mapping) else {}
    raw_text = str(utterance.get("raw_text") or context.get("raw_text") or "")
    query_id = str(graph.get("query_graph_id") or graph.get("id") or "")
    if not query_id:
        import uuid
        query_id = f"query_{uuid.uuid4().hex}"
    operators = graph.get("question_operators") or pattern.get("question_operators") or ()
    focuses = [
        _normalise(item.get("question_lemma") or item.get("surface"))
        for item in operators if isinstance(item, Mapping)
    ]
    predicate_data = pattern.get("predicate") if isinstance(pattern.get("predicate"), Mapping) else {}
    predicate_hypotheses = _predicate_hypotheses_from_analysis(predicate_data, analysis)
    unique_predicates = {str(item.get("predicate") or "") for item in predicate_hypotheses if item.get("predicate")}
    predicate = next(iter(unique_predicates)) if len(unique_predicates) == 1 else None
    nodes = [_reference(node) for node in pattern.get("known_nodes") or () if isinstance(node, Mapping)]
    target_gaps = pattern.get("target_gaps") or ([pattern["target_gap"]] if isinstance(pattern.get("target_gap"), Mapping) else [])
    type_map = {
        "EVENT_PREDICATE": "predicate", "NODE_COMPONENT": "component",
        "EVENT_PROPERTY": "property", "EVENT_ATTACHMENT": "attachment",
        "ENTITY": "entity", "QUANTITY": "quantity", "LOCATION": "location",
        "TIME": "time", "STATE": "state",
    }
    gaps: list[Gap] = []
    for index, gap_data in enumerate(target_gaps):
        if not isinstance(gap_data, Mapping):
            continue
        constraints = ()
        operator = operators[index] if index < len(operators) and isinstance(operators[index], Mapping) else None
        kind = _operational_gap_kind(gap_data, operator)
        gaps.append(Gap(
            gap_id=str(gap_data.get("node_id") or f"gap_{index + 1}_{query_id[-8:]}"),
            source_query_id=query_id,
            expected_type=type_map.get(kind, "entity"),
            expected_relation=next((str(item.get("relation_attachment")) for item in nodes if item.get("relation_attachment")), None),
            known_elements=tuple(nodes),
            surface_projection=str(gap_data.get("surface") or (focuses[index] if index < len(focuses) else "")),
            constraints=constraints,
            exclusions=tuple(graph.get("exclusions") or ()),
        ))
    continuation_of = str(graph.get("continuation_of") or "") or None
    return QueryFrame(
        query_id=query_id,
        session_id=str(context.get("session_id") or utterance.get("conversation_id") or ""),
        raw_text=raw_text,
        normalized_text=_normalise(raw_text),
        query_type=("associative_question" if any(marker in _normalise(raw_text) for marker in ("семантически близк", "похоже", "ассоциац")) else ("multi_gap_question" if len(gaps) > 1 else ("continuation_question" if continuation_of else "canonical_query"))),
        explicit_predicate=predicate,
        surface_focus=focuses[0] if focuses else None,
        known_elements=tuple(nodes),
        gaps=tuple(gaps),
        exclusions=tuple(graph.get("exclusions") or ()),
        continuation_of=continuation_of,
        confidence=0.92 if gaps else 0.35,
        context_inheritance={"mode": "ALLOW" if continuation_of else "BLOCK", "source_query_id": continuation_of, "inherited_elements": []},
        predicate_hypotheses=(predicate_hypotheses or ({"type": "unknown_predicate_question", "confidence": 0.81},)),
        trace={
            "source": "CANONICAL_QUERY_GRAPH",
            "query_graph_id": query_id,
            "gap_count": len(gaps),
            "parser_mode": "CANONICAL",
            "predicate_resolution_status": "RESOLVED" if predicate else ("AMBIGUOUS" if predicate_hypotheses else "UNKNOWN"),
            "predicate_hypothesis_count": len(predicate_hypotheses),
        },
    )


def _graph_predicate(analysis: Any) -> Optional[str]:
    pattern = _analysis_query_graph(analysis)
    predicate = pattern.get("predicate") if isinstance(pattern.get("predicate"), Mapping) else {}
    root_graph = analysis.get("query_graph") if isinstance(analysis, Mapping) and isinstance(analysis.get("query_graph"), Mapping) else analysis
    operators = pattern.get("question_operators") or (root_graph.get("question_operators") if isinstance(root_graph, Mapping) else ()) or ()
    if operators and isinstance(operators[0], Mapping):
        surface = _normalise(operators[0].get("surface") or operators[0].get("question_lemma"))
        token_index = int(operators[0].get("token_indices", [0])[0] or 0)
        predicate_index = int(predicate.get("token_index") or 0)
        if surface == "что" and predicate_index == token_index + 1:
            return None
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
            constraints.append({"type": "GRAMMATICAL_FEATURE", "feature": namespace, "value": value,
                                "scope": "CANDIDATE", "hardness": "SOFT", "confidence": score})
    return tuple(constraints)


def _surface_gap_constraints(focus: str) -> tuple[Mapping[str, Any], ...]:
    return ()


def _constraint_groups(focuses: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
    return tuple({"group_id": f"gap_hint_{index}", "mode": "GAP_HINT", "surface": focus} for index, focus in enumerate(focuses))


def _predicate(tokens: Sequence[str], analysis: Any = None) -> Optional[str]:
    graph_predicate = _graph_predicate(analysis)
    if graph_predicate:
        return graph_predicate
    if _unknown_predicate_query(analysis) or (tokens and tokens[0] == "что" and len(tokens) > 1 and tokens[1] not in STOPWORDS):
        return None
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


def _unknown_predicate_query(analysis: Any) -> bool:
    if not isinstance(analysis, Mapping):
        return False
    graph = analysis.get("query_graph") if isinstance(analysis.get("query_graph"), Mapping) else analysis
    pattern = graph.get("event_pattern") if isinstance(graph, Mapping) and isinstance(graph.get("event_pattern"), Mapping) else {}
    predicate = pattern.get("predicate") if isinstance(pattern.get("predicate"), Mapping) else {}
    operators = graph.get("question_operators") if isinstance(graph, Mapping) else ()
    if not operators or not isinstance(operators[0], Mapping):
        return False
    surface = _normalise(operators[0].get("surface") or operators[0].get("question_lemma"))
    question_index = int((operators[0].get("token_indices") or [0])[0] or 0)
    return surface == "что" and int(predicate.get("token_index") or 0) == question_index + 1


def _known_elements(tokens: Sequence[str], question_tokens: Sequence[str], predicate: Optional[str]) -> list[str]:
    result: list[str] = []
    for index, token in enumerate(tokens):
        if token in question_tokens or token in STOPWORDS or token == predicate or len(token) < 2:
            continue
        if index and tokens[index - 1] in {"после", "до"}:
            continue
        if token not in result:
            result.append(token)
    return result


def _query_type(focuses: Sequence[str], tokens: Sequence[str], predicate: Optional[str]) -> str:
    if focuses and focuses[0] == "что" and predicate is None and len(tokens) > 1 and tokens[1] not in STOPWORDS:
        return "event_predicate_question"
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
    graph = analysis.get("query_graph") if isinstance(analysis, Mapping) else None
    if isinstance(graph, Mapping):
        language = analysis.get("language_analysis") if isinstance(analysis.get("language_analysis"), Mapping) else None
        if language is None:
            language = graph.get("trace", {}).get("language_analysis") if isinstance(graph.get("trace"), Mapping) else None
        return from_query_graph(graph, language if isinstance(language, Mapping) else None, session_context)
    raw_text = str(text or "").strip()
    normalized = _normalise(raw_text)
    context = session_context or {}
    query_id = str(context.get("query_id") or "")
    if not query_id:
        import uuid
        query_id = f"query_{uuid.uuid4().hex}"
    tokens = _tokens(raw_text)
    question_tokens = [token for token in tokens if token in QUESTION_WORDS]
    focuses = [QUESTION_WORDS[token] for token in question_tokens]
    predicate = _predicate(tokens, analysis)
    unknown_predicate = bool(focuses and focuses[0] == "что" and len(tokens) > 1 and tokens[1] not in STOPWORDS) or _unknown_predicate_query(analysis)
    graph_known = _graph_known_elements(analysis)
    known = graph_known or _known_elements(tokens, question_tokens, predicate)
    negations = [token for token in tokens if token in NEGATION_WORDS]
    excluded_values = [token for token in tokens[tokens.index("кроме") + 1:]] if "кроме" in tokens else []
    if "без" in tokens:
        excluded_values += tokens[tokens.index("без") + 1:]
    gaps = []
    for index, focus in enumerate(focuses):
        relation = next((token for token in tokens if token in RELATION_WORDS and token not in {"после", "до"}), None)
        gap_id = f"gap_{index + 1}_{query_id[-8:]}"
        gaps.append(Gap(
            gap_id=gap_id,
            source_query_id=query_id,
            expected_type=("predicate" if unknown_predicate and focus == "что" else "quantity" if focus == "сколько" else "entity"),
            expected_relation=relation or ("predicate_attachment" if predicate else None),
            known_elements=tuple(known),
            surface_projection=focus,
            constraints=(
                tuple({"type": "NEGATION", "value": item, "scope": "EVENT", "hardness": "HARD"} for item in negations)
                + (({"type": "RELATION", "value": relation, "scope": "KNOWN_MEMBER", "hardness": "HARD", "confidence": 0.93},) if relation else ())
                + (() if unknown_predicate else (_graph_gap_constraints(analysis, index) or _surface_gap_constraints(focus)))
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
    temporal_scope = context.get("temporal_scope")
    if not temporal_scope:
        if set(tokens) & HISTORICAL_WORDS:
            temporal_scope = {"kind": "HISTORICAL", "anchors": sorted(set(tokens) & HISTORICAL_WORDS)}
        elif "после" in tokens or "до" in tokens:
            temporal_scope = {
                "kind": "RELATIVE",
                "relation": "после" if "после" in tokens else "до",
                "anchors": [tokens[index + 1] for index, value in enumerate(tokens[:-1]) if value in {"после", "до"}],
            }
        elif set(tokens) & CURRENT_WORDS:
            temporal_scope = {"kind": "CURRENT", "anchors": sorted(set(tokens) & CURRENT_WORDS)}
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
            constraints=tuple({"type": "NEGATION", "value": item, "scope": "EVENT", "hardness": "HARD"} for item in negations),
        negations=tuple(negations),
        exclusions=tuple(dict.fromkeys(excluded_values)),
        temporal_scope=temporal_scope,
        continuation_of=(
            (str((analysis or {}).get("query_graph", {}).get("continuation_of") or "") if isinstance(analysis, Mapping) else "")
            or str(context.get("last_query_id") or "")
        ) or None if continuation else None,
        confidence=0.20,
        context_inheritance={
            "mode": "ALLOW" if continuation else "BLOCK",
            "source_query_id": None,
            "inherited_elements": [],
        },
        predicate_hypotheses=tuple(
            [{"type": "unknown_predicate_question", "confidence": 0.81}]
            if unknown_predicate else ([{"type": "literal_predicate", "predicate": predicate, "confidence": 0.92}]
            if predicate else [{"type": "unknown_predicate_question", "confidence": 0.81}])
        ),
        constraint_groups=_constraint_groups(focuses),
        trace={
            "query_type_reason": "question_focus_and_relation_anchors",
            "parser_mode": "DEGRADED_UNVERIFIED",
            "epistemic_ceiling": "UNVERIFIED_ASSOCIATION",
            "known_elements": list(known),
            "gap_count": len(gaps),
            "predicate": predicate,
        },
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
        provenance = previous_answer.get("provenance") or {}
        if isinstance(provenance, Mapping):
            previous_values.extend(str(item) for item in (provenance.get("surface_forms") or {}).values())
    inherited_exclusions = list(query_frame.exclusions)
    if set(_tokens(query_frame.raw_text)) & {"ещё", "еще", "кроме", "другой", "другая", "другое", "другие"}:
        for value in previous_values:
            if value and value not in query_frame.exclusions:
                inherited_exclusions.append(value)
            normalized_value = _normalise(value) if value else ""
            if normalized_value and normalized_value not in query_frame.exclusions:
                inherited_exclusions.append(normalized_value)
    inherited_keys = {_reference_lemma(item) for item in inherited}
    for item in previous_known:
        key = _reference_lemma(item)
        if key and key not in inherited_keys:
            inherited.append(item if isinstance(item, Mapping) else str(item))
            inherited_keys.add(key)
    predicate = query_frame.explicit_predicate or (str(previous.get("explicit_predicate") or "") or None)
    unresolved = bool(continuation and not inherited and not predicate)
    inheritance_mode = "UNRESOLVED" if unresolved else ("ALLOW" if continuation else "BLOCK")
    def unique_references(values: Sequence[Any]) -> tuple[Any, ...]:
        result: list[Any] = []
        seen: set[str] = set()
        for value in values:
            key = _reference_lemma(value) or _normalise(str(value))
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            result.append(value)
        return tuple(result)

    gaps = tuple(
        Gap(
            **{**gap.as_dict(), "known_elements": tuple(inherited), "exclusions": unique_references(list(gap.exclusions) + inherited_exclusions + list(previous.get("exclusions") or ()))}
        )
        for gap in query_frame.gaps
    )
    original_keys = {_reference_lemma(item) for item in query_frame.known_elements}
    inherited_records = tuple({"element": item, "source": "previous_answer_context"} for item in inherited if _reference_lemma(item) not in original_keys)
    reconstructed = query_frame.raw_text
    if inherited and (not query_frame.known_elements or query_frame.query_type == "continuation_relation_question"):
        reconstructed = f"{query_frame.raw_text} [{', '.join(_reference_lemma(item) for item in inherited)}]"
    return QueryFrame(
        **{**query_frame.as_dict(), "explicit_predicate": predicate, "known_elements": tuple(inherited), "gaps": gaps,
           "continuation_of": query_frame.continuation_of or str(previous.get("query_id") or context.get("last_query_id") or "") or None,
           "exclusions": unique_references(inherited_exclusions), "inherited_elements": inherited_records, "reconstructed_query": reconstructed,
           "unresolved_context": unresolved, "confidence": clamp(query_frame.confidence * (0.92 if inherited else 0.4)),
           "context_inheritance": {
               "mode": inheritance_mode,
               "source_query_id": str(previous.get("query_id") or context.get("last_query_id") or "") or None,
               "inherited_elements": inherited_records,
           },
           "trace": {
               **dict(query_frame.trace),
               "context_inheritance": inheritance_mode,
               "inherited_count": len(inherited_records),
               "unresolved": unresolved,
           }}
    )
