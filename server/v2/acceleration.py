"""Optional, deterministic acceleration backends for the V2 graph.

The durable source of truth is SQLite.  This module deliberately owns only
process-local derived state, so losing an index is always safe: it is rebuilt
from SQLite on the next request.  Optional imports happen lazily in order to
keep the base Python 3.9 installation small and usable.
"""

from __future__ import annotations

import heapq
import importlib
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np


ACCELERATION_MODES = {"auto", "native", "python"}


def _as_vector(
    value: Mapping[str, float] | Sequence[float],
    dimensions: Sequence[str],
) -> Optional[np.ndarray]:
    """Convert observable coordinates without turning malformed data into errors."""
    try:
        if isinstance(value, Mapping):
            return np.asarray(
                [float(value.get(key, 0.0)) for key in dimensions],
                dtype=np.float32,
            )
        return np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None


def _normalise(
    vector: Optional[np.ndarray],
    *,
    expected_size: Optional[int] = None,
) -> Optional[np.ndarray]:
    if (
        vector is None
        or vector.size == 0
        or (expected_size is not None and vector.size != expected_size)
        or not np.isfinite(vector).all()
    ):
        return None
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return vector / norm


@dataclass(frozen=True)
class ProjectionMatch:
    """One exact cosine neighbour, ordered by descending score then id."""

    id: str
    score: float


class ProjectionIndex:
    """Exact cosine index with FAISS and NumPy implementations.

    IDs are intentionally retained outside FAISS.  FAISS only sees rows in
    stable ID order, which makes ties reproducible across implementations.
    """

    def __init__(self, runtime: "AccelerationRuntime", dimensions: Iterable[str] = ()) -> None:
        self.runtime = runtime
        self.dimensions = tuple(sorted(str(item) for item in dimensions))
        self._ids: list[str] = []
        self._vectors = np.empty((0, len(self.dimensions)), dtype=np.float32)
        self._faiss_index: Any = None
        self.skipped = 0
        self.build_ms = 0.0

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss_index is not None else "numpy"

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(self, item_id: str, vector: Mapping[str, float] | Sequence[float]) -> bool:
        if not self.dimensions:
            if isinstance(vector, Mapping):
                self.dimensions = tuple(sorted(str(key) for key in vector))
            else:
                self.dimensions = tuple(str(index) for index, _ in enumerate(vector))
            self._vectors = np.empty((0, len(self.dimensions)), dtype=np.float32)
        candidate = _normalise(
            _as_vector(vector, self.dimensions),
            expected_size=len(self.dimensions),
        )
        if candidate is None:
            self.skipped += 1
            return False
        self._ids.append(str(item_id))
        self._vectors = np.vstack((self._vectors, candidate.reshape(1, -1)))
        self._faiss_index = None
        return True

    def rebuild(self, items: Iterable[Any]) -> "ProjectionIndex":
        """Replace the index contents.

        ``items`` accepts ``(id, vector)`` tuples or mappings containing
        ``id`` and ``vector``.  This small permissive surface makes it useful
        in HTTP code and tests without leaking a database row shape.
        """
        started = perf_counter()
        prepared: list[tuple[str, Mapping[str, float] | Sequence[float]]] = []
        for item in items:
            if isinstance(item, Mapping):
                prepared.append((str(item["id"]), item.get("vector", item.get("coordinates", {}))))
            else:
                item_id, vector = item
                prepared.append((str(item_id), vector))
        if not self.dimensions and prepared:
            mapping_dimensions = {
                str(key)
                for _, vector in prepared
                if isinstance(vector, Mapping)
                for key in vector
            }
            if mapping_dimensions:
                self.dimensions = tuple(sorted(mapping_dimensions))
            else:
                sequence_lengths = {
                    len(vector)
                    for _, vector in prepared
                    if not isinstance(vector, Mapping)
                    and hasattr(vector, "__len__")
                }
                if len(sequence_lengths) == 1:
                    self.dimensions = tuple(
                        str(index) for index in range(sequence_lengths.pop())
                    )
        ids: list[str] = []
        vectors: list[np.ndarray] = []
        self._faiss_index = None
        self.skipped = 0
        for item_id, vector in sorted(prepared, key=lambda value: value[0]):
            candidate = _normalise(
                _as_vector(vector, self.dimensions),
                expected_size=len(self.dimensions),
            )
            if candidate is None:
                self.skipped += 1
                continue
            ids.append(item_id)
            vectors.append(candidate)
        self._ids = ids
        self._vectors = (
            np.vstack(vectors).astype(np.float32, copy=False)
            if vectors
            else np.empty((0, len(self.dimensions)), dtype=np.float32)
        )
        if self.runtime.use("faiss") and len(self._ids):
            try:
                index = self.runtime.faiss.IndexFlatIP(self._vectors.shape[1])
                index.add(self._vectors)
                self._faiss_index = index
            except Exception as exc:  # native errors are safe only in auto
                self.runtime.backend_failure("faiss", exc)
                self._faiss_index = None
        self.build_ms = (perf_counter() - started) * 1000.0
        return self

    def search(self, vector: Mapping[str, float] | Sequence[float], limit: int = 10) -> list[ProjectionMatch]:
        query = _normalise(
            _as_vector(vector, self.dimensions),
            expected_size=len(self.dimensions),
        )
        if query is None or not self._ids or limit <= 0:
            return []
        # Ask FAISS for all rows.  Its internal tie ordering is unspecified;
        # sorting all exact scores below preserves the public stable contract.
        if self._faiss_index is not None:
            try:
                scores, indices = self._faiss_index.search(query.reshape(1, -1), len(self._ids))
                pairs = [
                    ProjectionMatch(self._ids[int(index)], float(score))
                    for score, index in zip(scores[0], indices[0]) if int(index) >= 0
                ]
            except Exception as exc:
                self.runtime.backend_failure("faiss", exc)
                pairs = []
        else:
            pairs = []
        if not pairs:
            scores = self._vectors @ query
            pairs = [ProjectionMatch(item_id, float(score)) for item_id, score in zip(self._ids, scores)]
        return sorted(pairs, key=lambda item: (-item.score, item.id))[:limit]


@dataclass(frozen=True)
class Route:
    nodes: tuple[str, ...]
    cost: float


class RouteGraphIndex:
    """A budgeted directed route graph with a rustworkx-compatible fallback."""

    def __init__(self, runtime: "AccelerationRuntime") -> None:
        self.runtime = runtime
        self.adjacency: MutableMapping[str, list[tuple[str, float, str]]] = defaultdict(list)
        self._graph: Any = None
        self._node_indices: dict[str, int] = {}

    @property
    def backend(self) -> str:
        return "rustworkx" if self._graph is not None else "python"

    def rebuild(self, transitions: Iterable[Mapping[str, Any]]) -> "RouteGraphIndex":
        self.adjacency = defaultdict(list)
        rows = sorted(
            (dict(row) for row in transitions),
            key=lambda row: str(row.get("id") or ""),
        )
        for row in rows:
            source = str(row.get("source_id") or row.get("source") or "")
            target = str(row.get("target_id") or row.get("target") or "")
            if not source or not target:
                continue
            weight = max(float(row.get("weight", 0.0)), 1e-6)
            self.adjacency[source].append((target, 1.0 / weight, str(row.get("id") or "")))
        for source in self.adjacency:
            self.adjacency[source].sort(key=lambda item: (item[1], item[0], item[2]))
        self._graph = None
        self._node_indices = {}
        if self.runtime.use("rustworkx"):
            try:
                graph = self.runtime.rustworkx.PyDiGraph()
                nodes = sorted(set(self.adjacency) | {target for values in self.adjacency.values() for target, _, _ in values})
                self._node_indices = {node: graph.add_node(node) for node in nodes}
                for source, values in self.adjacency.items():
                    for target, cost, edge_id in values:
                        graph.add_edge(self._node_indices[source], self._node_indices[target], (cost, edge_id))
                self._graph = graph
            except Exception as exc:
                self.runtime.backend_failure("rustworkx", exc)
        return self

    def expand(self, seeds: Iterable[str], budget: int = 128) -> list[str]:
        """Deterministically expand neighbours without exceeding ``budget``."""
        limit = max(0, int(budget))
        if not limit:
            return []
        initial = sorted({str(seed) for seed in seeds})[:limit]
        seen = set(initial)
        queue = deque(initial)
        result = list(initial)
        while queue and len(result) < limit:
            node = queue.popleft()
            for target, _, _ in self.adjacency.get(node, ()):
                if target not in seen:
                    seen.add(target)
                    result.append(target)
                    queue.append(target)
                    if len(result) >= limit:
                        break
        return result

    def search(self, source: str, target: str, budget: int = 128) -> Optional[Route]:
        """Dijkstra with deterministic tie breaking and a settled-node budget."""
        if source == target:
            return Route((source,), 0.0)
        frontier: list[tuple[float, tuple[str, ...], str]] = [(0.0, (source,), source)]
        best: dict[str, tuple[float, tuple[str, ...]]] = {source: (0.0, (source,))}
        visited = 0
        while frontier and visited < max(0, budget):
            cost, nodes, node = heapq.heappop(frontier)
            if best.get(node) != (cost, nodes):
                continue
            visited += 1
            if node == target:
                return Route(nodes, cost)
            for neighbour, edge_cost, _ in self.adjacency.get(node, ()):
                candidate = (cost + edge_cost, nodes + (neighbour,))
                if neighbour not in best or candidate < best[neighbour]:
                    best[neighbour] = candidate
                    heapq.heappush(frontier, (candidate[0], candidate[1], neighbour))
        return None


@dataclass
class AccelerationRuntime:
    """One process runtime; test/experiment instances may be supplied directly."""

    mode: str = field(default_factory=lambda: os.getenv("SUPERAI_ACCELERATION_MODE", "auto").strip().casefold() or "auto")
    fallback_reasons: list[str] = field(default_factory=list)
    _modules: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _projection_cache: dict[tuple[str, str, tuple[str, ...]], ProjectionIndex] = field(default_factory=dict, init=False, repr=False)
    _route_cache: dict[tuple[str, str], RouteGraphIndex] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in ACCELERATION_MODES:
            raise ValueError("SUPERAI_ACCELERATION_MODE must be auto, native, or python")
        if self.mode == "native":
            # Fail during application/runtime construction instead of halfway
            # through a request, which is the contract of native mode.
            for backend in ("faiss", "rustworkx", "scipy", "sklearn"):
                self._load(backend)

    def _load(self, name: str) -> Any:
        if name in self._modules:
            return self._modules[name]
        if self.mode == "python":
            self._modules[name] = None
            return None
        module_name = {
            "faiss": "faiss",
            "rustworkx": "rustworkx",
            "scipy": "scipy",
            "sklearn": "sklearn",
        }[name]
        try:
            module = importlib.import_module(module_name)
            # The accelerated callers use these public submodules directly.
            # Import them here so availability is checked during native
            # startup and does not depend on unrelated import order.
            if name == "scipy":
                importlib.import_module("scipy.sparse")
            elif name == "sklearn":
                importlib.import_module("sklearn.cluster")
                importlib.import_module("sklearn.decomposition")
        except Exception as exc:
            if self.mode == "native":
                raise RuntimeError("native acceleration requires %s" % module_name) from exc
            self.fallback(name, exc)
            module = None
        self._modules[name] = module
        return module

    @property
    def faiss(self) -> Any:
        return self._load("faiss")

    @property
    def rustworkx(self) -> Any:
        return self._load("rustworkx")

    @property
    def scipy(self) -> Any:
        return self._load("scipy")

    @property
    def sklearn(self) -> Any:
        return self._load("sklearn")

    def use(self, name: str) -> bool:
        return self._load(name) is not None

    def fallback(self, backend: str, reason: BaseException | str) -> None:
        message = "%s: %s" % (backend, str(reason))
        if message not in self.fallback_reasons:
            self.fallback_reasons.append(message)

    def backend_failure(self, backend: str, reason: BaseException) -> None:
        """Fallback only in auto mode; native mode must remain fail-fast."""
        if self.mode == "native":
            raise RuntimeError("%s acceleration backend failed" % backend) from reason
        self.fallback(backend, reason)

    def projection_index(self, database_path: str | Path, revision: str | int, dimensions: Iterable[str]) -> ProjectionIndex:
        key = (str(Path(database_path)), str(revision), tuple(sorted(str(item) for item in dimensions)))
        return self._projection_cache.setdefault(key, ProjectionIndex(self, key[2]))

    def route_index(self, database_path: str | Path, revision: str | int) -> RouteGraphIndex:
        key = (str(Path(database_path)), str(revision))
        return self._route_cache.setdefault(key, RouteGraphIndex(self))

    def diagnostics(self) -> dict[str, Any]:
        return {
            "acceleration_mode": self.mode,
            "vector_backend": "faiss" if self.use("faiss") else "numpy",
            "route_backend": "rustworkx" if self.use("rustworkx") else "python",
            "discovery_backend": "scipy" if self.use("scipy") else "python",
            "backend_fallback_reason": "; ".join(self.fallback_reasons) or None,
        }


runtime = AccelerationRuntime()


__all__ = [
    "ACCELERATION_MODES", "AccelerationRuntime", "ProjectionIndex", "ProjectionMatch",
    "Route", "RouteGraphIndex", "runtime",
]
