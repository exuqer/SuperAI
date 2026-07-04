from __future__ import annotations

from collections import Counter
from typing import Any

from semantic_ants.core.graph import SemanticGraph
from semantic_ants.core.models import ConceptNode, SemanticEdge, SemanticResult, edge_key, split_edge_key
from semantic_ants.learning.checkpoint import Checkpoint


def graph_from_checkpoint(checkpoint: Checkpoint) -> SemanticGraph:
    graph = SemanticGraph()
    graph.add_edges(checkpoint.learned_edges(), include_reverse=False)
    return graph


def graph_snapshot(
    graph: SemanticGraph,
    checkpoint: Checkpoint,
    result: SemanticResult | dict[str, Any] | None = None,
    *,
    layer: int | None = None,
    source: str | None = None,
    edge_type: str | None = None,
    relation: str | None = None,
    query: str | None = None,
    min_pheromone: float | None = None,
    only_signal: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    signal = signal_index(result)
    query_value = query.casefold().strip() if query else ""
    edges = []
    degree: Counter[str] = Counter()

    for edge in graph.edges():
        payload = _edge_payload(edge, checkpoint, signal)
        if not _edge_matches(payload, layer, source, edge_type, relation, query_value, min_pheromone, only_signal):
            continue
        edges.append(payload)
        degree[edge.start] += 1
        degree[edge.end] += 1
        if limit is not None and len(edges) >= limit:
            break

    node_ids = {edge["start"] for edge in edges} | {edge["end"] for edge in edges}
    if not only_signal:
        for uri, node in graph.nodes.items():
            payload = _node_payload(uri, node, checkpoint, degree, signal)
            if _node_matches(payload, layer, source, query_value):
                node_ids.add(uri)
                if limit is not None and len(node_ids) >= limit and not edges:
                    break

    nodes = [
        _node_payload(uri, graph.nodes.get(uri), checkpoint, degree, signal)
        for uri in sorted(node_ids)
        if uri in graph.nodes or _label_for_uri(uri, None, checkpoint)
    ]
    nodes.sort(key=lambda item: (int(item.get("layer", 1)), -float(item.get("concept_pheromone", 1.0)), item["label"]))

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "signal_nodes": len(signal["nodes"]),
            "signal_edges": len(signal["edges"]),
        },
    }


def concept_detail(
    checkpoint: Checkpoint,
    uri: str,
    graph: SemanticGraph | None = None,
    result: SemanticResult | dict[str, Any] | None = None,
) -> dict[str, Any]:
    local_graph = graph or graph_from_checkpoint(checkpoint)
    signal = signal_index(result)
    degree: Counter[str] = Counter()
    incoming = []
    outgoing = []

    for edge in local_graph.edges():
        if edge.start == uri or edge.end == uri:
            payload = _edge_payload(edge, checkpoint, signal)
            degree[edge.start] += 1
            degree[edge.end] += 1
            if edge.start == uri:
                outgoing.append(payload)
            if edge.end == uri:
                incoming.append(payload)

    return {
        "node": _node_payload(uri, local_graph.nodes.get(uri), checkpoint, degree, signal),
        "incoming": incoming,
        "outgoing": outgoing,
        "aliases": sorted(alias for alias, target in checkpoint.aliases.items() if target == uri),
    }


def concept_list(
    checkpoint: Checkpoint,
    *,
    query: str | None = None,
    layer: int | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    graph = graph_from_checkpoint(checkpoint)
    snapshot = graph_snapshot(graph, checkpoint, query=query, layer=layer, limit=limit)
    concepts = snapshot["nodes"]
    concepts.sort(key=lambda item: (-float(item.get("concept_pheromone", 1.0)), item["label"]))
    return concepts[:limit]


def trace_interpretation(result: SemanticResult | dict[str, Any]) -> dict[str, Any]:
    raw = result.to_dict() if isinstance(result, SemanticResult) else result
    semantic_vector = dict(raw.get("semantic_vector") or {})
    items = [item for item in semantic_vector.get("items", []) if isinstance(item, dict)]
    top_domain = semantic_vector.get("top_domain") if isinstance(semantic_vector.get("top_domain"), dict) else None
    trace = [step for step in raw.get("signal_trace", []) if isinstance(step, dict)]
    chains: dict[str, list[dict[str, Any]]] = {}
    for step in trace:
        key = str(step.get("ant_id", "unknown"))
        chains.setdefault(key, []).append(step)
    return {
        "summary": {
            "input_text": raw.get("input_text", ""),
            "response": raw.get("response", ""),
            "top_domain": top_domain,
            "activated_count": len(items),
            "signal_steps": len(trace),
        },
        "chains": [
            {
                "ant_id": ant_id,
                "steps": sorted(steps, key=lambda value: int(value.get("step_index", 0))),
                "concept_chain": _chain_from_steps(steps),
            }
            for ant_id, steps in sorted(chains.items(), key=lambda value: value[0])[:8]
        ],
        "active_edge_ids": sorted(signal_index(raw)["edges"]),
    }


def signal_index(result: SemanticResult | dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {"nodes": set(), "edges": set(), "edge_scores": {}, "node_counts": Counter()}
    raw = result.to_dict() if isinstance(result, SemanticResult) else result
    nodes: set[str] = set()
    edges: set[str] = set()
    edge_scores: dict[str, float] = {}
    node_counts: Counter[str] = Counter()

    for route in raw.get("routes", []) or []:
        if not isinstance(route, dict):
            continue
        concepts = [str(value) for value in route.get("concepts", []) if value]
        for concept in concepts:
            nodes.add(concept)
            node_counts[concept] += 1
        for step in route.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            start = str(step.get("start", ""))
            relation = str(step.get("relation", ""))
            end = str(step.get("end", ""))
            if not start or not relation or not end:
                continue
            key = edge_key(start, relation, end)
            edges.add(key)
            nodes.update({start, end})
            node_counts[start] += 1
            node_counts[end] += 1
            edge_scores[key] = max(edge_scores.get(key, 0.0), float(step.get("score", 0.0) or 0.0))

    for step in raw.get("signal_trace", []) or []:
        if not isinstance(step, dict):
            continue
        start = str(step.get("start", ""))
        relation = str(step.get("relation", ""))
        end = str(step.get("end", ""))
        if not start or not relation or not end:
            continue
        key = edge_key(start, relation, end)
        edges.add(key)
        nodes.update({start, end})
        node_counts[start] += 1
        node_counts[end] += 1
        edge_scores[key] = max(edge_scores.get(key, 0.0), float(step.get("score", 0.0) or 0.0))

    return {"nodes": nodes, "edges": edges, "edge_scores": edge_scores, "node_counts": node_counts}


def _edge_payload(edge: SemanticEdge, checkpoint: Checkpoint, signal: dict[str, Any]) -> dict[str, Any]:
    active = edge.key in signal["edges"]
    return {
        "id": edge.key,
        "start": edge.start,
        "end": edge.end,
        "relation": edge.relation,
        "weight": edge.weight,
        "source": edge.source,
        "surface_text": edge.surface_text,
        "layer": edge.layer,
        "distance": edge.distance,
        "edge_type": edge.edge_type,
        "metadata": edge.metadata,
        "pheromone": checkpoint.pheromone_for(edge),
        "route_stats": checkpoint.route_stats.get(edge.key, {}),
        "signal": {
            "active": active,
            "score": signal["edge_scores"].get(edge.key, 0.0),
        },
    }


def _node_payload(
    uri: str,
    node: ConceptNode | None,
    checkpoint: Checkpoint,
    degree: Counter[str],
    signal: dict[str, Any],
) -> dict[str, Any]:
    layer = node.layer if node else (0 if uri.startswith("/m/top/") else 1)
    return {
        "id": uri,
        "uri": uri,
        "label": _label_for_uri(uri, node, checkpoint),
        "language": node.language if node else _language_from_uri(uri),
        "source": node.source if node else "checkpoint",
        "layer": layer,
        "metadata": _metadata_for_uri(uri, node, checkpoint),
        "concept_pheromone": checkpoint.concept_pheromone_for(uri),
        "suppression": float(checkpoint.suppressed_concepts.get(uri, 0.0)),
        "degree": int(degree.get(uri, 0)),
        "signal": {
            "active": uri in signal["nodes"],
            "count": int(signal["node_counts"].get(uri, 0)),
        },
    }


def _edge_matches(
    edge: dict[str, Any],
    layer: int | None,
    source: str | None,
    edge_type: str | None,
    relation: str | None,
    query: str,
    min_pheromone: float | None,
    only_signal: bool,
) -> bool:
    if layer is not None and int(edge.get("layer", 1)) != layer:
        return False
    if source and edge.get("source") != source:
        return False
    if edge_type and edge.get("edge_type") != edge_type:
        return False
    if relation and edge.get("relation") != relation:
        return False
    if min_pheromone is not None and float(edge.get("pheromone", 0.0)) < min_pheromone:
        return False
    if only_signal and not edge.get("signal", {}).get("active"):
        return False
    if query:
        values = [edge.get("start", ""), edge.get("end", ""), edge.get("relation", ""), edge.get("surface_text", "")]
        if not any(query in str(value).casefold() for value in values):
            return False
    return True


def _node_matches(node: dict[str, Any], layer: int | None, source: str | None, query: str) -> bool:
    if layer is not None and int(node.get("layer", 1)) != layer:
        return False
    if source and node.get("source") != source:
        return False
    if query and query not in f"{node.get('uri', '')} {node.get('label', '')}".casefold():
        return False
    return True


def _metadata_for_uri(uri: str, node: ConceptNode | None, checkpoint: Checkpoint) -> dict[str, Any]:
    metadata = dict(node.metadata) if node else {}
    definitions = checkpoint.metadata.get("concept_definitions", {})
    if isinstance(definitions, dict) and isinstance(definitions.get(uri), dict):
        metadata = {**dict(definitions[uri]), **metadata}
    return metadata


def _label_for_uri(uri: str, node: ConceptNode | None, checkpoint: Checkpoint) -> str:
    definitions = checkpoint.metadata.get("concept_definitions", {})
    if isinstance(definitions, dict):
        raw = definitions.get(uri)
        if isinstance(raw, dict) and raw.get("label"):
            return str(raw["label"])
    labels = checkpoint.metadata.get("concept_labels", {})
    if isinstance(labels, dict) and labels.get(uri):
        return str(labels[uri])
    top_domains = checkpoint.metadata.get("top_domains", {})
    if isinstance(top_domains, dict):
        for key, raw in top_domains.items():
            if isinstance(raw, dict) and raw.get("uri") == uri:
                return str(raw.get("label") or key)
    if node and node.label:
        return str(node.label)
    return uri.rstrip("/").split("/")[-1].replace("_", " ")


def _language_from_uri(uri: str) -> str:
    parts = uri.split("/")
    return parts[2] if len(parts) > 2 and parts[1] == "c" else "unknown"


def _chain_from_steps(steps: list[dict[str, Any]]) -> list[str]:
    ordered = sorted(steps, key=lambda value: int(value.get("step_index", 0)))
    if not ordered:
        return []
    chain = [str(ordered[0].get("start", ""))]
    chain.extend(str(step.get("end", "")) for step in ordered)
    return [value for value in chain if value]


def edge_id_parts(edge_id: str) -> tuple[str, str, str]:
    return split_edge_key(edge_id)
