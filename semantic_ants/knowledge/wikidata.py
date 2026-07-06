from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from semantic_ants.core.normalization import tokenize
from semantic_ants.learning.checkpoint import Checkpoint


WIKIDATA_SEED_VERSION = 1
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_USER_AGENT = "semantic_ants/1.0 (wikidata seed)"

WIKIDATA_CLASS_TERMS = [
    "human",
    "person",
    "city",
    "country",
    "organization",
    "company",
    "software",
    "video game",
    "book",
    "film",
    "song",
    "album",
    "chemical compound",
    "plant",
    "animal",
    "food",
    "tool",
    "vehicle",
    "language",
    "emotion",
    "mathematical object",
    "scientific concept",
    "programming language",
    "medical condition",
    "sport",
    "astronomical object",
    "event",
    "work of art",
    "natural object",
    "physical object",
    "concept",
]


@dataclass
class WikidataSeedReport:
    classes: int = 0
    fetched_items: int = 0
    imported_items: int = 0
    matched_existing: int = 0
    linked_edges: int = 0
    top_domain_edges: int = 0
    changed: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": self.classes,
            "fetched_items": self.fetched_items,
            "imported_items": self.imported_items,
            "matched_existing": self.matched_existing,
            "linked_edges": self.linked_edges,
            "top_domain_edges": self.top_domain_edges,
            "changed": self.changed,
            "errors": self.errors,
        }


def bootstrap_wikidata_knowledge(
    checkpoint: Checkpoint,
    force: bool = False,
    limit: int = 5000,
    allow_network: bool = True,
) -> WikidataSeedReport:
    if not allow_network:
        return WikidataSeedReport(changed=False)
    if not force and checkpoint.metadata.get("wikidata_seed_version") == WIKIDATA_SEED_VERSION:
        return WikidataSeedReport(changed=False)

    report = WikidataSeedReport()
    try:
        class_terms = list(WIKIDATA_CLASS_TERMS)
        class_qids = _resolve_class_qids(class_terms)
        report.classes = len(class_qids)
        items = _collect_items(class_qids, limit=limit)
        report.fetched_items = len(items)
        if not items:
            checkpoint.metadata["wikidata_seed_version"] = WIKIDATA_SEED_VERSION
            checkpoint.metadata["wikidata_seed_loaded"] = True
            return report
        existing_index = _build_existing_index(checkpoint)
        report.imported_items = _apply_items(checkpoint, items, existing_index, report)
        checkpoint.metadata["wikidata_seed_version"] = WIKIDATA_SEED_VERSION
        checkpoint.metadata["wikidata_seed_loaded"] = True
        checkpoint.metadata["wikidata_seed_limit"] = limit
        checkpoint.metadata["wikidata_seed_classes"] = class_terms
        report.changed = True
        return report
    except Exception as exc:
        report.errors.append(str(exc))
        return report


@dataclass(frozen=True)
class WikidataItem:
    qid: str
    label: str
    description: str
    class_qid: str
    class_label: str
    class_description: str = ""

    @property
    def uri(self) -> str:
        return f"/m/wikidata/{self.qid}"

    @property
    def class_uri(self) -> str:
        return f"/m/wikidata/class/{self.class_qid}"


def _resolve_class_qids(class_terms: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for term in class_terms:
        qid, label = _search_wikidata_entity(term, language="en")
        if not qid or qid in seen:
            continue
        seen.add(qid)
        pairs.append((qid, label or term))
    return pairs


def _collect_items(class_qids: list[tuple[str, str]], limit: int = 5000) -> list[WikidataItem]:
    items: list[WikidataItem] = []
    seen: set[str] = set()
    if not class_qids:
        return items
    per_class = max(1, limit // max(len(class_qids), 1) + 20)
    for class_qid, class_label in class_qids:
        for raw in _query_items_for_class(class_qid, per_class=per_class):
            qid = str(raw.get("qid") or "").strip()
            if not qid or qid in seen:
                continue
            label = str(raw.get("label") or "").strip()
            description = str(raw.get("description") or "").strip()
            if not label:
                continue
            seen.add(qid)
            items.append(
                WikidataItem(
                    qid=qid,
                    label=label,
                    description=description,
                    class_qid=class_qid,
                    class_label=class_label,
                    class_description=str(raw.get("class_description") or ""),
                )
            )
            if len(items) >= limit:
                return items
    return items


def _apply_items(
    checkpoint: Checkpoint,
    items: list[WikidataItem],
    existing_index: dict[str, str],
    report: WikidataSeedReport,
) -> int:
    definitions = _definitions(checkpoint)
    added = 0
    domain_counts: dict[str, int] = {}
    for item in items:
        item_label_key = _normalize(item.label)
        item_uri = item.uri
        class_uri = item.class_uri
        domain_uri = _top_domain_for_class(item.class_label)
        domain_counts[domain_uri] = domain_counts.get(domain_uri, 0) + 1

        checkpoint.remember_concept_label(item_uri, item.label)
        definitions[item_uri] = _definition_payload(
            label=item.label,
            description=item.description,
            source="Wikidata",
            wikidata_qid=item.qid,
            wikidata_url=f"https://www.wikidata.org/wiki/{item.qid}",
            wikidata_class_qid=item.class_qid,
            wikidata_class_label=item.class_label,
        )
        if item.description:
            definitions.setdefault(item_uri, {}).setdefault("description", item.description)

        if item_label_key and item_label_key not in checkpoint.aliases:
            checkpoint.aliases[item_label_key] = item_uri

        checkpoint.add_custom_edge(
            item_uri,
            class_uri,
            relation="InstanceOf",
            weight=1.5,
            layer=1,
            distance=1.0,
            edge_type="knowledge",
            metadata={
                "dataset": "Wikidata",
                "wikidata_qid": item.qid,
                "wikidata_class_qid": item.class_qid,
                "wikidata_class_label": item.class_label,
            },
        )
        checkpoint.reinforce_edge(item_uri, "InstanceOf", class_uri, amount=0.2)
        report.linked_edges += 1

        checkpoint.add_custom_edge(
            class_uri,
            domain_uri,
            relation="InTopDomain",
            weight=1.2,
            layer=0,
            distance=1.5,
            edge_type="domain",
            metadata={
                "dataset": "Wikidata",
                "wikidata_class_qid": item.class_qid,
                "wikidata_class_label": item.class_label,
            },
        )
        checkpoint.reinforce_edge(class_uri, "InTopDomain", domain_uri, amount=0.15)
        report.top_domain_edges += 1

        matched_uri = existing_index.get(item_label_key)
        if matched_uri and matched_uri != item_uri:
            checkpoint.add_custom_edge(
                item_uri,
                matched_uri,
                relation="MatchesConcept",
                weight=1.7,
                layer=1,
                distance=1.0,
                edge_type="semantic",
                metadata={
                    "dataset": "Wikidata",
                    "wikidata_qid": item.qid,
                    "matched_by": "label",
                },
            )
            checkpoint.reinforce_edge(item_uri, "MatchesConcept", matched_uri, amount=0.25)
            report.matched_existing += 1
            report.linked_edges += 1
            _merge_definition(definitions, matched_uri, item)
        else:
            checkpoint.remember_concept_label(item_uri, item.label)
            if item_label_key:
                existing_index.setdefault(item_label_key, item_uri)

        for token in _extra_tokens(item.label, item.description):
            anchor = existing_index.get(token)
            if not anchor or anchor == item_uri:
                continue
            checkpoint.add_custom_edge(
                item_uri,
                anchor,
                relation="RelatedTo",
                weight=1.1,
                layer=1,
                distance=1.2,
                edge_type="semantic",
                metadata={
                    "dataset": "Wikidata",
                    "wikidata_qid": item.qid,
                    "match_token": token,
                },
            )
            checkpoint.reinforce_edge(item_uri, "RelatedTo", anchor, amount=0.12)
            report.linked_edges += 1
            break

        added += 1
        if added % 250 == 0:
            checkpoint.reinforce_concept(item_uri, amount=0.05)

    checkpoint.metadata["wikidata_domain_counts"] = domain_counts
    checkpoint.metadata["concept_definitions"] = definitions
    return added


def _search_wikidata_entity(term: str, language: str = "en") -> tuple[str | None, str | None]:
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": language,
        "search": term,
        "limit": 5,
        "type": "item",
    }
    payload = _load_json(WIKIDATA_API_ENDPOINT, params=params)
    for item in payload.get("search", []) if isinstance(payload, dict) else []:
        qid = str(item.get("id") or "").strip()
        label = str(item.get("label") or "").strip()
        if qid.startswith("Q"):
            return qid, label or term
    return None, None


def _query_items_for_class(class_qid: str, per_class: int = 250) -> list[dict[str, Any]]:
    query = f"""
SELECT ?item ?itemLabel ?itemDescription ?classLabel WHERE {{
  ?item wdt:P31/wdt:P279* wd:{class_qid} .
  OPTIONAL {{ ?item wdt:P31 ?class . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "ru,en". }}
}}
ORDER BY ?item
LIMIT {int(per_class)}
"""
    payload = _load_json(
        WIKIDATA_SPARQL_ENDPOINT,
        params={"format": "json", "query": query},
        accept="application/sparql-results+json",
    )
    rows: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return rows
    results = payload.get("results", {})
    bindings = results.get("bindings", []) if isinstance(results, dict) else []
    for row in bindings:
        if not isinstance(row, dict):
            continue
        item_uri = row.get("item", {}).get("value", "") if isinstance(row.get("item"), dict) else ""
        qid = item_uri.rsplit("/", 1)[-1] if item_uri else ""
        label = row.get("itemLabel", {}).get("value", "") if isinstance(row.get("itemLabel"), dict) else ""
        description = row.get("itemDescription", {}).get("value", "") if isinstance(row.get("itemDescription"), dict) else ""
        class_label = row.get("classLabel", {}).get("value", "") if isinstance(row.get("classLabel"), dict) else ""
        rows.append(
            {
                "qid": qid,
                "label": label,
                "description": description,
                "class_label": class_label,
            }
        )
    return rows


def _load_json(url: str, params: dict[str, Any], accept: str = "application/json") -> dict[str, Any]:
    request_url = f"{url}?{urlencode(params)}"
    request = Request(
        request_url,
        headers={
            "Accept": accept,
            "User-Agent": WIKIDATA_USER_AGENT,
        },
    )
    with urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def _build_existing_index(checkpoint: Checkpoint) -> dict[str, str]:
    index: dict[str, str] = {}
    definitions = checkpoint.metadata.get("concept_definitions", {})
    if isinstance(definitions, dict):
        for uri, raw in definitions.items():
            if not isinstance(raw, dict):
                continue
            label = str(raw.get("label") or "").strip()
            if label:
                index.setdefault(_normalize(label), str(uri))
    aliases = checkpoint.aliases
    if isinstance(aliases, dict):
        for alias, uri in aliases.items():
            if alias:
                index.setdefault(_normalize(str(alias)), str(uri))
    return index


def _definitions(checkpoint: Checkpoint) -> dict[str, dict[str, Any]]:
    raw = checkpoint.metadata.get("concept_definitions", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _definition_payload(**kwargs: Any) -> dict[str, Any]:
    payload = {key: value for key, value in kwargs.items() if value is not None and value != ""}
    payload.setdefault("source", "Wikidata")
    payload.setdefault("dataset", "Wikidata")
    return payload


def _merge_definition(definitions: dict[str, dict[str, Any]], uri: str, item: WikidataItem) -> None:
    raw = definitions.get(uri, {})
    existing = dict(raw) if isinstance(raw, dict) else {}
    existing.setdefault("source", "Wikidata")
    existing.setdefault("dataset", "Wikidata")
    existing.setdefault("label", item.label)
    if item.description:
        existing.setdefault("description", item.description)
    existing.setdefault("wikidata_qid", item.qid)
    existing.setdefault("wikidata_url", f"https://www.wikidata.org/wiki/{item.qid}")
    existing.setdefault("wikidata_class_qid", item.class_qid)
    existing.setdefault("wikidata_class_label", item.class_label)
    definitions[uri] = existing


def _top_domain_for_class(class_label: str) -> str:
    label = _normalize(class_label)
    if any(token in label for token in ("human", "person", "organization", "company", "role")):
        return "/m/top/person"
    if any(token in label for token in ("place", "city", "country", "location", "region", "country", "state")):
        return "/m/top/place"
    if any(token in label for token in ("emotion", "feeling", "mood")):
        return "/m/top/emotion"
    if any(token in label for token in ("language", "word", "speech", "communication", "conversation")):
        return "/m/top/language"
    if any(token in label for token in ("number", "mathemat", "count", "quantity")):
        return "/m/top/number"
    if any(token in label for token in ("plant", "animal", "food", "natural", "organism", "astronomical", "earth", "nature")):
        return "/m/top/nature"
    if any(token in label for token in ("mind", "concept", "idea", "scientific", "software", "programming", "technology", "tool", "vehicle", "object", "physical")):
        return "/m/top/object"
    if any(token in label for token in ("event", "action", "process", "sport", "work of art", "film", "book", "song", "album")):
        return "/m/top/action"
    if any(token in label for token in ("perception", "color", "colour", "sense")):
        return "/m/top/perception"
    return "/m/top/object"


def _extra_tokens(*texts: str) -> list[str]:
    tokens: list[str] = []
    for text in texts:
        for token in tokenize(text):
            normalized = _normalize(token)
            if normalized and normalized not in tokens:
                tokens.append(normalized)
    return tokens[:8]


def _normalize(text: str) -> str:
    value = text.casefold().replace("_", " ").strip()
    value = re.sub(r"[^\w\sа-яё]+", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value
