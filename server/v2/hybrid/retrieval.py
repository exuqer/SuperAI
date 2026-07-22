"""Reproducible index retrieval. Results are evidence-bearing hits, never answers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from .contracts import QueryFrame, RetrievalHit, clamp


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("ё", "е").split())


def _terms(value: Any) -> set[str]:
    return {item for item in _norm(value).replace("/", " ").replace(",", " ").split() if item}


def _known_lemma(value: Any) -> str:
    if isinstance(value, Mapping):
        return _norm(value.get("lemma") or value.get("surface"))
    return _norm(value)


def _lemma_match(left: str, right: str) -> bool:
    return bool(left and right and left == right)


def _predicate_lemmas(frame: QueryFrame) -> tuple[str, ...]:
    values: list[str] = []
    explicit = _norm(frame.explicit_predicate)
    if explicit:
        values.append(explicit)
    for item in frame.predicate_hypotheses or ():
        if not isinstance(item, Mapping):
            continue
        value = _norm(item.get("predicate") or item.get("lemma"))
        if value and value not in values:
            values.append(value)
    return tuple(values)


def _records(indexes: Any) -> list[Mapping[str, Any]]:
    if indexes is None:
        return []
    if isinstance(indexes, Mapping):
        for key in ("records", "events", "elements", "observations", "index"):
            if key in indexes and isinstance(indexes[key], Iterable) and not isinstance(indexes[key], (str, bytes, Mapping)):
                return [item for item in indexes[key] if isinstance(item, Mapping)]
        if all(isinstance(value, Mapping) for value in indexes.values()):
            return [{"element_id": str(key), **dict(value)} for key, value in indexes.items()]
        return [indexes]
    return [item for item in indexes if isinstance(item, Mapping)]


def _score(frame: QueryFrame, record: Mapping[str, Any]) -> tuple[float, list[str]]:
    features: list[str] = []
    score = 0.0
    predicates = _predicate_lemmas(frame)
    record_predicate = _norm(record.get("predicate") or record.get("predicate_lemma") or record.get("action"))
    matched_predicate = next((value for value in predicates if _lemma_match(value, record_predicate)), "")
    if matched_predicate:
        score += 0.28
        features.extend(("predicate", f"predicate_hypothesis:{matched_predicate}"))
    record_terms = set().union(*(_terms(record.get(key)) for key in ("text", "raw_text", "surface", "lemma", "value", "predicate", "predicate_lemma")))
    participant_terms = set()
    for participant in record.get("participants") or ():
        if isinstance(participant, Mapping):
            participant_terms |= set().union(*(_terms(participant.get(key)) for key in ("surface", "lemma", "value", "head_lemma")))
    record_terms |= participant_terms
    known_matches = {
        known
        for known in {_known_lemma(item) for item in frame.known_elements}
        if any(_lemma_match(known, term) for term in record_terms)
    }
    if known_matches:
        score += min(0.34, 0.14 * len(known_matches))
        features.append("known_participant")
    if frame.surface_focus and _norm(frame.surface_focus) in record_terms:
        score += 0.05
        features.append("surface_focus")
    if frame.session_id and str(record.get("session_id") or record.get("conversation_id") or "") == frame.session_id:
        score += 0.18
        features.append("session_context")
    if frame.temporal_scope and record.get("temporal_scope"):
        score += 0.08
        features.append("temporal_scope")
    if record.get("provenance") or record.get("source_id"):
        score += 0.10
        features.append("provenance")
    if frame.negations and str(record.get("polarity") or "POSITIVE").upper() in {"NEGATIVE", "NEGATED"}:
        score += 0.06
        features.append("negation")
    if record.get("conflict_ids") or record.get("conflicts"):
        features.append("conflict")
    return clamp(score), features


def _matches_all_known_elements(frame: QueryFrame, record: Mapping[str, Any]) -> bool:
    if not frame.known_elements:
        return True
    record_terms = set().union(*(_terms(record.get(key)) for key in ("text", "raw_text", "surface", "lemma", "value", "predicate", "predicate_lemma")))
    for participant in record.get("participants") or ():
        if isinstance(participant, Mapping):
            record_terms |= set().union(*(_terms(participant.get(key)) for key in ("surface", "lemma", "value", "head_lemma", "head_surface")))
    return all(any(_lemma_match(_known_lemma(known), term) for term in record_terms) for known in frame.known_elements)


class DirectRetriever:
    def retrieve(self, query_frame: QueryFrame, indexes: Any = None, *, limit: int = 128) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for ordinal, record in enumerate(_records(indexes)):
            if not _matches_all_known_elements(query_frame, record):
                continue
            score, features = _score(query_frame, record)
            if score <= 0.0 or not ({"predicate", "known_participant", "session_context", "temporal_scope"} & set(features)):
                continue
            element_id = str(record.get("element_id") or record.get("id") or f"element_{ordinal}")
            source_id = str(record.get("source_id") or record.get("observation_id") or element_id)
            path = tuple(record.get("retrieval_path") or ("query_anchor", "index", element_id))
            provenance = tuple(record.get("provenance") or ({"source_id": source_id, "source_type": record.get("source_type", "observation")},))
            hits.append(RetrievalHit(
                hit_id=f"hit_{query_frame.query_id[-12:]}_{ordinal}",
                element_id=element_id,
                element_type=str(record.get("element_type") or record.get("type") or ("event" if record.get("predicate") or record.get("predicate_lemma") else "entity")),
                source_id=source_id,
                match_score=score,
                matched_features=tuple(features),
                payload=dict(record),
                provenance=provenance,
                conflicts=tuple(str(item) for item in record.get("conflicts") or record.get("conflict_ids") or ()),
                retrieval_path=path,
                origin=str(record.get("retrieval_origin") or "GRAPH"),
            ))
        hits.sort(key=lambda item: (-item.match_score, item.element_id, item.hit_id))
        return hits[:max(1, int(limit))]


DirectRetrievalService = DirectRetriever


def retrieve_direct(query_frame: QueryFrame, indexes: Any = None, *, limit: int = 128) -> list[RetrievalHit]:
    return DirectRetriever().retrieve(query_frame, indexes, limit=limit)


def _predicate_lemmas_from_query_graph(query_graph: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(query_graph, Mapping):
        return ()
    pattern = query_graph.get("event_pattern") if isinstance(query_graph.get("event_pattern"), Mapping) else query_graph
    predicate = pattern.get("predicate") if isinstance(pattern.get("predicate"), Mapping) else {}
    values: list[str] = []

    def add(value: Any) -> None:
        lemma = _norm(value)
        if lemma and lemma not in values:
            values.append(lemma)

    trace = query_graph.get("trace") if isinstance(query_graph.get("trace"), Mapping) else {}
    language = trace.get("language_analysis") if isinstance(trace.get("language_analysis"), Mapping) else {}
    language_predicate = language.get("predicate") if isinstance(language.get("predicate"), Mapping) else {}
    for item in language_predicate.get("morphological_analyses") or ():
        if isinstance(item, Mapping) and str(item.get("pos") or "").upper() in {"VERB", "INFN", "PRTF", "PRTS", "GRND"}:
            add(item.get("lemma"))
    for clause in language.get("clauses") or ():
        if not isinstance(clause, Mapping):
            continue
        for item in clause.get("predicate_hypotheses") or ():
            if isinstance(item, Mapping):
                add(item.get("lemma"))
    add(predicate.get("lemma") or predicate.get("surface"))

    operators = query_graph.get("question_operators") or ()
    if operators and isinstance(operators[0], Mapping):
        surface = _norm(operators[0].get("surface") or operators[0].get("question_lemma"))
        question_index = int((operators[0].get("token_indices") or [0])[0] or 0)
        predicate_index = int(predicate.get("token_index") or 0)
        if surface == "что" and predicate_index == question_index + 1:
            return ()
    return tuple(values)


def _known_constraints_from_query_graph(query_graph: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    if not isinstance(query_graph, Mapping):
        return []
    pattern = query_graph.get("event_pattern") if isinstance(query_graph.get("event_pattern"), Mapping) else query_graph
    known_nodes = pattern.get("known_nodes") if isinstance(pattern, Mapping) else ()
    constraints: list[tuple[str, str]] = []
    for node in known_nodes or ():
        if not isinstance(node, Mapping):
            continue
        head = node.get("head") if isinstance(node.get("head"), Mapping) else {}
        entity_id = str(node.get("entity_id") or "")
        lemma = _norm(head.get("lemma") or node.get("lemma") or node.get("surface"))
        if entity_id or lemma:
            constraints.append((entity_id, lemma))
    return constraints


def _decode_event_rows(conn: Any, rows: Iterable[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in ("predicate_features_json", "properties_json", "metadata_json"):
            value = item.get(key)
            if isinstance(value, str):
                try:
                    item[key.removesuffix("_json")] = json.loads(value)
                except json.JSONDecodeError:
                    item[key.removesuffix("_json")] = {} if key != "properties_json" else []
        participants = conn.execute(
            """SELECT p.id,m.head_lemma,m.head_surface,m.surface,m.entity_id,
                      m.features_json,m.components_json,m.preposition,
                      p.observation_signature_json,p.confidence
               FROM graph_participants p JOIN graph_mentions m ON m.id=p.mention_id
               WHERE p.event_id=? ORDER BY p.ordinal_hint,p.id""",
            (row["id"],),
        ).fetchall()
        decoded_participants: list[dict[str, Any]] = []
        for participant in participants:
            value = dict(participant)
            for key, fallback in (("features_json", {}), ("components_json", []), ("observation_signature_json", {})):
                try:
                    value[key.removesuffix("_json")] = json.loads(value.get(key) or "{}")
                except json.JSONDecodeError:
                    value[key.removesuffix("_json")] = fallback
            decoded_participants.append(value)
        item["element_id"] = str(row["id"])
        item["element_type"] = "event"
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        item["session_id"] = str(metadata.get("conversation_id") or metadata.get("session_id") or "")
        item["participants"] = decoded_participants
        item["retrieval_path"] = ["query_anchor", "event_index", str(row["id"])]
        item["retrieval_origin"] = "GRAPH"
        item["provenance"] = [{
            "source_id": str(row["source_id"]),
            "source_type": str(row["source_type"]),
            "trust_status": str(row["status"]),
            "event_id": str(row["id"]),
            "independent_source_key": str(row["independent_key"] or row["source_id"]),
        }]
        result.append(item)
    return result


def graph_rows_from_connection(
    conn: Any,
    *,
    session_id: str = "",
    query_graph: Mapping[str, Any] | None = None,
    limit: int = 128,
) -> list[dict[str, Any]]:
    """Adapt confirmed events for every morphology-backed predicate hypothesis."""
    predicate_lemmas = _predicate_lemmas_from_query_graph(query_graph)
    known_constraints = _known_constraints_from_query_graph(query_graph)
    if not predicate_lemmas and not known_constraints:
        return []
    clauses: list[str] = []
    params: list[Any] = []
    if predicate_lemmas:
        placeholders = ",".join("?" for _ in predicate_lemmas)
        clauses.append(f"e.predicate_lemma IN ({placeholders})")
        params.extend(predicate_lemmas)
    for entity_id, lemma in known_constraints:
        clauses.append(
            """EXISTS (
                   SELECT 1 FROM graph_participants p
                   JOIN graph_mentions m ON m.id=p.mention_id
                   WHERE p.event_id=e.id
                     AND (m.entity_id=? OR m.head_lemma=?)
               )"""
        )
        params.extend((entity_id, lemma))
    rows = conn.execute(
        f"""SELECT e.*,s.raw_text,s.source_type,s.status,s.independent_key,s.confidence AS source_confidence,
                   s.metadata_json
            FROM graph_events e JOIN knowledge_sources s ON s.id=e.source_id
            WHERE s.status='CONFIRMED' AND {' AND '.join(clauses)}
            ORDER BY e.created_at,e.id
            LIMIT ?""",
        (*params, max(1, int(limit))),
    ).fetchall()
    return _decode_event_rows(conn, rows)


def graph_rows_all_from_connection(conn: Any, *, limit: int = 1024) -> list[dict[str, Any]]:
    """Return confirmed graph events for field-to-evidence bridging."""
    rows = conn.execute(
        """SELECT e.*,s.raw_text,s.source_type,s.status,s.independent_key,
                  s.confidence AS source_confidence,s.metadata_json
           FROM graph_events e JOIN knowledge_sources s ON s.id=e.source_id
           WHERE s.status='CONFIRMED'
           ORDER BY e.created_at,e.id LIMIT ?""",
        (max(1, int(limit)),),
    ).fetchall()
    return _decode_event_rows(conn, rows)


def field_to_graph_hits(
    query_frame: QueryFrame,
    field_hits: Iterable[RetrievalHit],
    indexes: Any,
    *,
    limit: int = 128,
) -> list[RetrievalHit]:
    """Resolve spatially activated concepts back to persisted graph evidence."""
    if not isinstance(indexes, Mapping):
        return []
    records = indexes.get("field_bridge_records") or ()
    if not isinstance(records, Iterable) or isinstance(records, (str, bytes, Mapping)):
        return []
    cloud_concepts = {
        str(hit.payload.get("concept_id") or "")
        for hit in field_hits
        if isinstance(hit, RetrievalHit) and hit.payload.get("concept_id")
    }
    if not cloud_concepts:
        return []
    hits: list[RetrievalHit] = []
    for ordinal, record in enumerate(item for item in records if isinstance(item, Mapping)):
        if not _matches_all_known_elements(query_frame, record):
            continue
        participants = [item for item in record.get("participants") or () if isinstance(item, Mapping)]
        activated = {
            str(item.get("entity_id") or "")
            for item in participants
            if str(item.get("entity_id") or "") in cloud_concepts
        }
        if not activated:
            continue
        score, features = _score(query_frame, record)
        score = clamp(max(score, 0.42) + min(0.18, 0.06 * len(activated)))
        element_id = str(record.get("element_id") or record.get("id") or f"bridge_{ordinal}")
        source_id = str(record.get("source_id") or record.get("observation_id") or element_id)
        provenance = tuple(record.get("provenance") or ({"source_id": source_id, "source_type": "observation"},))
        hits.append(RetrievalHit(
            hit_id=f"bridge_{query_frame.query_id[-12:]}_{ordinal}",
            element_id=element_id,
            element_type="event",
            source_id=source_id,
            match_score=score,
            matched_features=tuple(dict.fromkeys([*features, "FIELD_TO_GRAPH_BRIDGE"])),
            payload={**dict(record), "bridge_cloud_concept_ids": sorted(activated)},
            provenance=provenance,
            conflicts=tuple(str(item) for item in record.get("conflicts") or record.get("conflict_ids") or ()),
            retrieval_path=("field_projection", "concept_to_graph_event", element_id),
            origin="GRAPH_BRIDGE",
        ))
    hits.sort(key=lambda item: (-item.match_score, item.element_id, item.hit_id))
    return hits[:max(1, int(limit))]
