from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from semantic_ants.core.models import ConceptNode, SemanticEdge


class SemanticGraph:
    """Локальный граф смыслов, собранный из источников и обучаемого overlay."""

    def __init__(self) -> None:
        self.nodes: dict[str, ConceptNode] = {}
        self._edges: dict[str, SemanticEdge] = {}
        self._adjacency: dict[str, list[SemanticEdge]] = defaultdict(list)

    def add_node(self, node: ConceptNode) -> None:
        existing = self.nodes.get(node.uri)
        if existing is None:
            self.nodes[node.uri] = _normalize_node_layers(node)
            return
        merged_layers = _merge_layers(existing.layers, node.layers, existing.layer, node.layer)
        merged_active_layers = _merge_layers(existing.active_layers, node.active_layers, existing.layer, node.layer)
        if (
            merged_layers != existing.layers
            or merged_active_layers != existing.active_layers
            or node.metadata
            or node.label != existing.label
            or node.language != existing.language
            or node.source != existing.source
        ):
            self.nodes[node.uri] = ConceptNode(
                uri=existing.uri,
                label=node.label or existing.label,
                language=node.language if node.language != "unknown" else existing.language,
                source=node.source or existing.source,
                layer=min(existing.layer, node.layer),
                layers=merged_layers,
                active_layers=merged_active_layers,
                metadata={**existing.metadata, **node.metadata},
            )

    def add_edge(self, edge: SemanticEdge, include_reverse: bool = False) -> None:
        self._add_single_edge(edge)
        if include_reverse:
            self._add_single_edge(edge.reversed())

    def add_edges(self, edges: Iterable[SemanticEdge], include_reverse: bool = False) -> None:
        for edge in edges:
            self.add_edge(edge, include_reverse=include_reverse)

    def neighbors(self, uri: str) -> list[SemanticEdge]:
        return list(self._adjacency.get(uri, []))

    def edges(self) -> list[SemanticEdge]:
        return list(self._edges.values())

    def _add_single_edge(self, edge: SemanticEdge) -> None:
        if edge.key in self._edges:
            return
        self._edges[edge.key] = edge
        self._adjacency[edge.start].append(edge)
        self._ensure_node(edge.start, edge.source, edge.layer)
        self._ensure_node(edge.end, edge.source, edge.layer)
        self._update_node_layers(edge.start, edge.layer, edge.from_layer, edge.to_layer)
        self._update_node_layers(edge.end, edge.layer, edge.from_layer, edge.to_layer)

    def _ensure_node(self, uri: str, source: str, layer: int = 1) -> None:
        if uri in self.nodes:
            return
        label = uri.rstrip("/").split("/")[-1].replace("_", " ")
        parts = uri.split("/")
        language = parts[2] if len(parts) > 2 and parts[1] == "c" else "unknown"
        self.nodes[uri] = ConceptNode(
            uri=uri,
            label=label,
            language=language,
            source=source,
            layer=layer,
            layers=[layer],
            active_layers=[layer],
        )

    def _update_node_layers(
        self,
        uri: str,
        *layers: int | None,
    ) -> None:
        node = self.nodes.get(uri)
        if node is None:
            return
        merged_layers = _merge_layers(node.layers, [layer for layer in layers if layer is not None], node.layer)
        merged_active_layers = _merge_layers(node.active_layers, [layer for layer in layers if layer is not None], node.layer)
        if merged_layers != node.layers or merged_active_layers != node.active_layers:
            self.nodes[uri] = ConceptNode(
                uri=node.uri,
                label=node.label,
                language=node.language,
                source=node.source,
                layer=min([node.layer, *merged_layers]) if merged_layers else node.layer,
                layers=merged_layers,
                active_layers=merged_active_layers,
                metadata=node.metadata,
            )


def _merge_layers(*groups: list[int] | tuple[int, ...] | int) -> list[int]:
    values: list[int] = []
    for group in groups:
        if isinstance(group, int):
            candidates = [group]
        else:
            candidates = list(group)
        for layer in candidates:
            if layer not in values:
                values.append(layer)
    values.sort()
    return values


def _normalize_node_layers(node: ConceptNode) -> ConceptNode:
    layers = _merge_layers(node.layers, node.active_layers, node.layer)
    return ConceptNode(
        uri=node.uri,
        label=node.label,
        language=node.language,
        source=node.source,
        layer=min(layers) if layers else node.layer,
        layers=layers,
        active_layers=_merge_layers(node.active_layers, node.layers, node.layer),
        metadata=node.metadata,
    )
