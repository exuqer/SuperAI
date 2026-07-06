from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen

from semantic_ants.learning.checkpoint import Checkpoint


CONCEPTNET_SEED_VERSION = 1
CONCEPTNET_DUMP_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
CONCEPTNET_USER_AGENT = "semantic_ants/1.0 (conceptnet seed)"
CONCEPTNET_LANGS = {"en", "ru"}


@dataclass
class ConceptNetSeedReport:
    concepts: int = 0
    edges: int = 0
    matched_existing: int = 0
    changed: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concepts": self.concepts,
            "edges": self.edges,
            "matched_existing": self.matched_existing,
            "changed": self.changed,
            "errors": self.errors,
        }


def bootstrap_conceptnet_knowledge(
    checkpoint: Checkpoint,
    force: bool = False,
    allow_network: bool = True,
    limit: int = 5000,
) -> ConceptNetSeedReport:
    if not allow_network:
        return ConceptNetSeedReport(changed=False)
    if not force and checkpoint.metadata.get("conceptnet_seed_version") == CONCEPTNET_SEED_VERSION:
        return ConceptNetSeedReport(changed=False)

    report = ConceptNetSeedReport()
    try:
        known_uris = _known_uris(checkpoint)
        existing_index = _existing_index(checkpoint)
        imported: set[str] = set()
        with _dump_stream() as handle:
            for raw_line in handle:
                line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
                edge = _parse_edge(line)
                if edge is None:
                    continue
                if _language_of(edge.start) not in CONCEPTNET_LANGS and _language_of(edge.end) not in CONCEPTNET_LANGS:
                    continue
                start_known = edge.start in known_uris or _normalize(edge.start) in existing_index or edge.start in imported
                end_known = edge.end in known_uris or _normalize(edge.end) in existing_index or edge.end in imported
                if not start_known and not end_known:
                    continue
                if len(imported) < limit:
                    if start_known and not end_known:
                        if _import_concept(checkpoint, edge.end, edge, existing_index):
                            imported.add(edge.end)
                            report.concepts += 1
                            if _normalize(edge.end) in existing_index and existing_index[_normalize(edge.end)] != edge.end:
                                report.matched_existing += 1
                    elif end_known and not start_known:
                        if _import_concept(checkpoint, edge.start, edge, existing_index):
                            imported.add(edge.start)
                            report.concepts += 1
                            if _normalize(edge.start) in existing_index and existing_index[_normalize(edge.start)] != edge.start:
                                report.matched_existing += 1
                if _store_edge(checkpoint, edge):
                    report.edges += 1
                if len(imported) >= limit and report.edges >= limit * 2:
                    break
        checkpoint.metadata["conceptnet_seed_version"] = CONCEPTNET_SEED_VERSION
        checkpoint.metadata["conceptnet_seed_loaded"] = True
        checkpoint.metadata["conceptnet_seed_limit"] = limit
        checkpoint.metadata["conceptnet_seed_languages"] = sorted(CONCEPTNET_LANGS)
        report.changed = True
        return report
    except Exception as exc:
        report.errors.append(str(exc))
        return report


@dataclass(frozen=True)
class ConceptNetEdge:
    edge_uri: str
    relation: str
    start: str
    end: str
    metadata: dict[str, Any]


def _dump_stream():
    request = Request(
        CONCEPTNET_DUMP_URL,
        headers={
            "User-Agent": CONCEPTNET_USER_AGENT,
            "Accept": "application/octet-stream",
        },
    )
    response = urlopen(request, timeout=180)
    return gzip.GzipFile(fileobj=response)


def _parse_edge(line: str) -> ConceptNetEdge | None:
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split("\t", 4)
    if len(parts) < 4:
        return None
    edge_uri, relation_uri, start, end = parts[:4]
    metadata: dict[str, Any] = {}
    if len(parts) > 4 and parts[4]:
        try:
            raw = json.loads(parts[4])
            if isinstance(raw, dict):
                metadata = raw
        except json.JSONDecodeError:
            metadata = {}
    relation = relation_uri.rsplit("/", 1)[-1]
    return ConceptNetEdge(edge_uri=edge_uri, relation=relation, start=start, end=end, metadata=metadata)


def _store_edge(checkpoint: Checkpoint, edge: ConceptNetEdge) -> bool:
    weight = float(edge.metadata.get("weight", 1.0)) if isinstance(edge.metadata, dict) else 1.0
    surface_text = str(edge.metadata.get("surfaceText")) if isinstance(edge.metadata.get("surfaceText"), str) else None
    metadata = dict(edge.metadata) if isinstance(edge.metadata, dict) else {}
    before = len(checkpoint.custom_edges)
    checkpoint.add_custom_edge(
        edge.start,
        edge.end,
        relation=edge.relation,
        weight=weight,
        layer=1,
        distance=1.0,
        edge_type="knowledge",
        metadata={
            **metadata,
            "dataset": "ConceptNet 5.7",
            "source": "conceptnet_dump",
            "edge_uri": edge.edge_uri,
        },
    )
    checkpoint.reinforce_edge(edge.start, edge.relation, edge.end, amount=max(0.1, min(weight, 1.5) * 0.1))
    return len(checkpoint.custom_edges) != before


def _import_concept(
    checkpoint: Checkpoint,
    uri: str,
    edge: ConceptNetEdge,
    existing_index: dict[str, str],
) -> bool:
    if not uri or _language_of(uri) not in CONCEPTNET_LANGS:
        return False
    label = _surface_for_uri(uri)
    normalized = _normalize(label)
    if not label:
        return False
    checkpoint.remember_concept_label(uri, label)
    definitions = _definitions(checkpoint)
    info = definitions.get(uri, {})
    payload = dict(info) if isinstance(info, dict) else {}
    payload.setdefault("label", label)
    payload.setdefault("source", "ConceptNet 5.7")
    payload.setdefault("dataset", "ConceptNet 5.7")
    payload.setdefault("uri", uri)
    payload.setdefault("language", _language_of(uri))
    payload.setdefault("relation", edge.relation)
    payload.setdefault("edge_uri", edge.edge_uri)
    if edge.metadata.get("surfaceText"):
        payload.setdefault("surface_text", edge.metadata["surfaceText"])
    if edge.metadata.get("dataset"):
        payload.setdefault("conceptnet_dataset", edge.metadata["dataset"])
    definitions[uri] = payload
    checkpoint.metadata["concept_definitions"] = definitions

    if normalized and normalized not in checkpoint.aliases:
        checkpoint.aliases[normalized] = uri

    anchor = existing_index.get(normalized)
    if anchor and anchor != uri:
        checkpoint.add_custom_edge(
            uri,
            anchor,
            relation="MatchesConcept",
            weight=1.4,
            layer=1,
            distance=1.0,
            edge_type="semantic",
            metadata={"dataset": "ConceptNet 5.7", "matched_by": "label"},
        )
        checkpoint.reinforce_edge(uri, "MatchesConcept", anchor, amount=0.15)
    existing_index.setdefault(normalized, uri)
    return True


def _known_uris(checkpoint: Checkpoint) -> set[str]:
    values: set[str] = set()
    values.update(str(uri) for uri in checkpoint.aliases.values())
    definitions = checkpoint.metadata.get("concept_definitions", {})
    if isinstance(definitions, dict):
        values.update(str(uri) for uri in definitions.keys())
    for edge in [*checkpoint.custom_edges, *checkpoint.learned_bridges]:
        if isinstance(edge, dict):
            values.add(str(edge.get("start", "")))
            values.add(str(edge.get("end", "")))
    for item in checkpoint.accepted_answers:
        if isinstance(item, dict):
            values.update(str(value) for value in item.get("concepts", []) if value)
    for item in checkpoint.response_memory.values():
        if isinstance(item, dict):
            values.update(str(value) for value in item.get("concepts", []) if value)
    return {value for value in values if value}


def _existing_index(checkpoint: Checkpoint) -> dict[str, str]:
    index: dict[str, str] = {}
    definitions = checkpoint.metadata.get("concept_definitions", {})
    if isinstance(definitions, dict):
        for uri, info in definitions.items():
            if not isinstance(info, dict):
                continue
            label = str(info.get("label") or "").strip()
            if label:
                index.setdefault(_normalize(label), str(uri))
    for alias, uri in checkpoint.aliases.items():
        if alias:
            index.setdefault(_normalize(str(alias)), str(uri))
    return index


def _definitions(checkpoint: Checkpoint) -> dict[str, dict[str, Any]]:
    raw = checkpoint.metadata.get("concept_definitions", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _surface_for_uri(uri: str) -> str:
    if not uri:
        return ""
    parts = uri.split("/", 4)
    if len(parts) < 4:
        return uri.rstrip("/").split("/")[-1].replace("_", " ")
    token = parts[3]
    token = token.split("/", 1)[0]
    return token.replace("_", " ")


def _language_of(uri: str) -> str:
    parts = uri.split("/")
    return parts[2] if len(parts) > 2 and parts[1] == "c" else "unknown"


def _normalize(text: str) -> str:
    value = text.casefold().replace("_", " ").strip()
    value = re.sub(r"[^\w\sа-яё]+", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value
