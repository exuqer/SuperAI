from __future__ import annotations

from urllib.parse import unquote

from semantic_ants.core.graph import SemanticGraph
from semantic_ants.core.models import SemanticEdge
from semantic_ants.core.normalization import detect_language, text_to_concept_uri


LANGUAGE_NODES = {
    "ru": "/m/language/ru",
    "en": "/m/language/en",
}

LANGUAGE_LABELS = {
    "ru": "русский язык",
    "en": "english language",
}

RELATION_LIKE_QUERIES = {
    "AnswerNextToken",
    "ContextNeighbor",
    "DescribedByToken",
    "DialogueTurnEquivalent",
    "ExpectedAnswerToken",
    "FrequentWord",
    "HasAlphabet",
    "HasLanguage",
    "HasWord",
    "InLanguage",
    "InTopDomain",
    "IsA",
    "LanguageCode",
    "LayerTarget",
    "LearnedRelatedTo",
    "Means",
    "MeaningHint",
    "RelatedTo",
    "Reverse:InLanguage",
    "Reverse:InTopDomain",
    "TopBridge",
    "TopDomain",
}


def ensure_isolated_language_edge(graph: SemanticGraph, concept_uri: str, lang: str | None = None) -> bool:
    detected_lang, _ = concept_parts(concept_uri)
    lang = lang if lang in LANGUAGE_NODES else detected_lang
    if lang not in LANGUAGE_NODES:
        return False
    if _has_non_input_incident_edge(graph, concept_uri):
        return False
    _add_language_edges(graph, concept_uri, lang)
    return True


def ensure_isolated_concept_edges(
    graph: SemanticGraph,
    concept_uri: str,
    *,
    lang: str | None = None,
    top_domain: str | None = None,
) -> bool:
    if _has_non_input_incident_edge(graph, concept_uri):
        return False
    changed = False
    selected_lang = lang if lang in LANGUAGE_NODES else concept_parts(concept_uri)[0]
    if selected_lang in LANGUAGE_NODES:
        _add_language_edges(graph, concept_uri, selected_lang)
        changed = True
    if top_domain and top_domain.startswith("/m/top/"):
        _add_top_domain_edge(graph, concept_uri, top_domain)
        changed = True
    return changed


def ensure_query_language_edge(graph: SemanticGraph, query: str | None) -> bool:
    concept_uri = concept_uri_from_query(query)
    if not concept_uri:
        return False
    lang, _ = concept_parts(concept_uri)
    if lang not in LANGUAGE_NODES:
        return False
    _add_language_edges(graph, concept_uri, lang)
    return True


def concept_uri_from_query(query: str | None) -> str:
    value = " ".join(str(query or "").strip().split())
    if not value:
        return ""
    if value.casefold() in {item.casefold() for item in RELATION_LIKE_QUERIES}:
        return ""
    if value.startswith("/c/"):
        return value
    if value.startswith("/m/") or "/" in value:
        return ""
    if "|" in value or len(value) > 80:
        return ""
    if not any(char.isalnum() for char in value):
        return ""
    try:
        return text_to_concept_uri(value, detect_language(value))
    except ValueError:
        return ""


def concept_parts(uri: str) -> tuple[str, str]:
    parts = str(uri).split("/", 3)
    if len(parts) != 4 or parts[1] != "c":
        return "", ""
    return parts[2], unquote(parts[3]).replace("_", " ").strip().lower()


def _add_language_edges(graph: SemanticGraph, concept_uri: str, lang: str) -> None:
    language_uri = LANGUAGE_NODES[lang]
    graph.add_edge(
        SemanticEdge(
            start=concept_uri,
            end=language_uri,
            relation="InLanguage",
            weight=1.0,
            source="inferred",
            layer=1,
            distance=1.0,
            edge_type="language",
            metadata={"language": lang},
        ),
        include_reverse=True,
    )
    graph.add_edge(
        SemanticEdge(
            start=language_uri,
            end="/m/top/language",
            relation="InTopDomain",
            weight=1.2,
            source="inferred",
            layer=0,
            distance=1.0,
            edge_type="domain",
            metadata={"top_domain": "language", "language": lang, "label": LANGUAGE_LABELS[lang]},
        ),
        include_reverse=True,
    )


def _add_top_domain_edge(graph: SemanticGraph, concept_uri: str, top_domain: str) -> None:
    graph.add_edge(
        SemanticEdge(
            start=concept_uri,
            end=top_domain,
            relation="InTopDomain",
            weight=1.1,
            source="inferred",
            layer=0,
            distance=1.0,
            edge_type="domain",
            metadata={"top_domain": top_domain.rsplit("/", 1)[-1], "inferred": True},
        ),
        include_reverse=True,
    )


def _has_non_input_incident_edge(graph: SemanticGraph, concept_uri: str) -> bool:
    for edge in graph.edges():
        if edge.start != concept_uri and edge.end != concept_uri:
            continue
        if edge.source == "input" or edge.relation in {"ContextNeighbor", "Reverse:ContextNeighbor"}:
            continue
        return True
    return False
