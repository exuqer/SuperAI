from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from semantic_ants.core.models import ConceptNode, SemanticEdge
from semantic_ants.core.normalization import quote_concept_path
from semantic_ants.providers.cache import JsonCache


class ConceptNetError(RuntimeError):
    """Ошибка доступа к ConceptNet."""


class ConceptNetClient:
    """Минимальный клиент ConceptNet REST API на стандартной библиотеке."""

    def __init__(
        self,
        cache: JsonCache | None = None,
        base_url: str = "https://api.conceptnet.io",
        timeout: float = 10.0,
        allow_network: bool = True,
        min_interval: float = 1.0,
    ) -> None:
        self.cache = cache
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.allow_network = allow_network
        self.min_interval = min_interval
        self._last_request_at = 0.0

    def lookup(self, uri: str, limit: int = 30) -> dict[str, Any]:
        path = quote_concept_path(uri)
        return self._get(f"{path}?{urlencode({'limit': limit})}")

    def query(self, **params: str | int) -> dict[str, Any]:
        return self._get(f"/query?{urlencode(params)}")

    def related(self, uri: str, filter_lang: str | None = None, limit: int = 20) -> dict[str, Any]:
        params: dict[str, str | int] = {"limit": limit}
        if filter_lang:
            params["filter"] = f"/c/{filter_lang}"
        return self._get(f"/related{quote_concept_path(uri)}?{urlencode(params)}")

    def relatedness(self, node1: str, node2: str) -> float:
        data = self._get(
            "/relatedness?"
            + urlencode({"node1": node1, "node2": node2})
        )
        return float(data.get("value", 0.0))

    def edges_for(self, uri: str, limit: int = 30) -> tuple[list[ConceptNode], list[SemanticEdge]]:
        data = self.lookup(uri, limit=limit)
        nodes: dict[str, ConceptNode] = {}
        edges: list[SemanticEdge] = []
        for raw in data.get("edges", []):
            parsed = self._parse_edge(raw)
            if parsed is None:
                continue
            start_node, end_node, edge = parsed
            nodes[start_node.uri] = start_node
            nodes[end_node.uri] = end_node
            edges.append(edge)
        return list(nodes.values()), edges

    def _get(self, endpoint: str) -> dict[str, Any]:
        cache_key = f"{self.base_url}{endpoint}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        if not self.allow_network:
            raise ConceptNetError(f"Нет кэша для запроса ConceptNet: {endpoint}")
        self._respect_rate_limit()
        url = f"{self.base_url}{endpoint}"
        request = Request(url, headers={"User-Agent": "semantic-ants/0.1"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise ConceptNetError(f"ConceptNet недоступен: {exc}") from exc
        data = json.loads(payload)
        if self.cache:
            self.cache.set(cache_key, data)
        return data

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _parse_edge(
        self,
        raw: dict[str, Any],
    ) -> tuple[ConceptNode, ConceptNode, SemanticEdge] | None:
        start = raw.get("start") or {}
        end = raw.get("end") or {}
        rel = raw.get("rel") or {}
        start_uri = start.get("@id") or start.get("term")
        end_uri = end.get("@id") or end.get("term")
        relation = rel.get("label") or rel.get("@id", "/r/RelatedTo").split("/")[-1]
        if not start_uri or not end_uri:
            return None
        dataset = str(raw.get("dataset", "conceptnet"))
        source = "wordnet" if "wordnet" in dataset.lower() else "conceptnet"
        metadata = {
            "dataset": dataset,
            "license": raw.get("license"),
            "edge_id": raw.get("@id"),
            "sources": raw.get("sources", []),
        }
        edge = SemanticEdge(
            start=start_uri,
            end=end_uri,
            relation=relation,
            weight=float(raw.get("weight", 1.0)),
            source=source,
            surface_text=raw.get("surfaceText"),
            metadata=metadata,
        )
        return self._node_from_raw(start, source), self._node_from_raw(end, source), edge

    def _node_from_raw(self, raw: dict[str, Any], source: str) -> ConceptNode:
        uri = raw.get("@id") or raw.get("term")
        label = raw.get("label") or str(uri).rstrip("/").split("/")[-1].replace("_", " ")
        language = raw.get("language") or _language_from_uri(str(uri))
        return ConceptNode(uri=str(uri), label=str(label), language=str(language), source=source)


def _language_from_uri(uri: str) -> str:
    parts = uri.split("/")
    if len(parts) > 2 and parts[1] == "c":
        return parts[2]
    return "unknown"
