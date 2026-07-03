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
        if node.uri not in self.nodes:
            self.nodes[node.uri] = node

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
        self._ensure_node(edge.start, edge.source)
        self._ensure_node(edge.end, edge.source)

    def _ensure_node(self, uri: str, source: str) -> None:
        if uri in self.nodes:
            return
        label = uri.rstrip("/").split("/")[-1].replace("_", " ")
        parts = uri.split("/")
        language = parts[2] if len(parts) > 2 and parts[1] == "c" else "unknown"
        self.nodes[uri] = ConceptNode(uri=uri, label=label, language=language, source=source)
