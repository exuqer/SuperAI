from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
from threading import RLock
import tempfile
import time
from typing import Any, Callable, Iterator

import numpy as np

from .jobs import JobManager
from .preprocess import iter_valid_dialogue_pairs
from .state import Checkpoint, DEFAULT_VECTOR_DIM, load_checkpoint, save_checkpoint, _normalize_slim_edge_record

PUNCT_TOKENS = {".", ",", "!", "?", ";", ":"}
TERMINAL_TOKENS = {".", "!", "?"}
ROLE_USER_TOKEN = "__user__"
ROLE_ASSISTANT_TOKEN = "__assistant__"
LEGACY_ROLE_USER_TOKEN = "[__user__]"
LEGACY_ROLE_ASSISTANT_TOKEN = "[__assistant__]"
ROLE_TOKENS = {ROLE_USER_TOKEN, ROLE_ASSISTANT_TOKEN}
TOKEN_OR_PUNCT_RE = re.compile(
    r"\[__user__\]|\[__assistant__\]|__user__|__assistant__|[0-9A-Za-zА-Яа-яЁё]+(?:['-][0-9A-Za-zА-Яа-яЁё]+)*|[.,!?;:]",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
DIALOGUE_PAIR_SEPARATORS = ("=>", "->", " -- ", " — ", " – ", " - ")
DIALOGUE_ROLE_RE = re.compile(r"^(user|assistant|system|bot|ai|human)\s*[:\-]\s*(.+)$", re.IGNORECASE)
RAW_DIALOGUE_KEYS = {"question", "answer", "relevance"}
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_PROGRESS_INTERVAL = 500
HIERARCHY_NODE_PREFIX = "hyper:"
ROOT_HYPERNODE_ID = "hyper:__root__"


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r", " ").split()).strip()


def split_sentences(text: str) -> list[str]:
    normalized = str(text or "").replace("\r", "\n").strip()
    if not normalized:
        return []
    chunks = [chunk.strip() for chunk in SENTENCE_SPLIT_RE.split(normalized) if chunk.strip()]
    return chunks or [normalized]


def canonical_token(token: Any) -> str:
    cleaned = normalize_text(str(token or "")).casefold()
    if cleaned == LEGACY_ROLE_USER_TOKEN:
        return ROLE_USER_TOKEN
    if cleaned == LEGACY_ROLE_ASSISTANT_TOKEN:
        return ROLE_ASSISTANT_TOKEN
    return cleaned


def token_id(token: Any) -> str:
    return f"token:{canonical_token(str(token).removeprefix('token:'))}"


def is_role_token(token: str) -> bool:
    return canonical_token(token) in ROLE_TOKENS


def tokenize_with_surfaces(text: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for match in TOKEN_OR_PUNCT_RE.finditer(str(text or "")):
        token = canonical_token(match.group(0))
        if token:
            items.append({"surface": token, "token": token})
    return items


def tokenize(text: str) -> list[str]:
    return [item["token"] for item in tokenize_with_surfaces(text) if item["token"] not in PUNCT_TOKENS and not is_role_token(item["token"])]


def strip_wrapping_quotes(text: str) -> str:
    return str(text or "").strip().strip('\'"«»“”').strip()


def strip_dialogue_markers(text: str) -> str:
    cleaned = strip_wrapping_quotes(text)
    while cleaned and cleaned[0] in "-—–•*":
        cleaned = cleaned[1:].lstrip()
    return cleaned.strip()


def normalize_dialogue_role(role: str) -> str | None:
    cleaned = normalize_text(role).casefold()
    if cleaned in {"user", "human"}:
        return "user"
    if cleaned in {"assistant", "bot", "ai"}:
        return "assistant"
    return None


def looks_like_raw_dialogue_jsonl_record(record: Any) -> bool:
    return isinstance(record, dict) and RAW_DIALOGUE_KEYS.issubset(set(record))


def dialogue_separator_allowed(prompt_part: str, response_part: str) -> bool:
    prompt = strip_wrapping_quotes(prompt_part)
    response = strip_wrapping_quotes(response_part)
    if not prompt or not response:
        return False
    if any(mark in prompt for mark in ("?", "!", ":")):
        return True
    if prompt.startswith(("'", '"', "«", "“")) or prompt.endswith(("'", '"', "»", "”")):
        return True
    prompt_words = tokenize(prompt)
    response_words = tokenize(response)
    return 2 <= len(prompt_words) <= 16 and len(response_words) >= 2 and any(mark in response for mark in (".", "?", "!"))


def zero_vector(dim: int = DEFAULT_VECTOR_DIM) -> list[float]:
    return [0.0] * max(int(dim), 0)


def normalize_vector(vector: Any, dim: int = DEFAULT_VECTOR_DIM) -> list[float]:
    size = max(int(dim), 0)
    if size <= 0:
        return []
    if not isinstance(vector, (list, tuple)):
        return zero_vector(size)
    values = [float(value) for value in vector[:size]]
    if len(values) < size:
        values.extend([0.0] * (size - len(values)))
    return values


def cosine(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(normalize_vector(left), dtype=np.float32)
    right_array = np.asarray(normalize_vector(right), dtype=np.float32)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left_array, right_array) / denominator)


def cosine_similarity_matrix(vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((0,), dtype=np.float32)
    query_norm = float(np.linalg.norm(query_vector))
    vector_norms = np.linalg.norm(vectors, axis=1)
    denominator = vector_norms * query_norm
    similarities = np.zeros((vectors.shape[0],), dtype=np.float32)
    valid = denominator > 0.0
    if np.any(valid):
        similarities[valid] = np.dot(vectors[valid], query_vector) / denominator[valid]
    return similarities


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def sha_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


class _LazyEmbeddingBackend:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._attempted = False
        self._available = False

    @property
    def dim(self) -> int:
        return DEFAULT_VECTOR_DIM

    def ensure(self) -> bool:
        if self._attempted:
            return self._available
        self._attempted = True
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            self._available = False
            self._model = None
            return False
        try:
            model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
            if int(model.get_sentence_embedding_dimension() or 0) != DEFAULT_VECTOR_DIM:
                self._available = False
                self._model = None
                return False
            self._model = model
            self._available = True
            return True
        except Exception:
            self._available = False
            self._model = None
            return False

    def encode(self, text: str) -> list[float]:
        if not self.ensure() or self._model is None:
            return zero_vector()
        raw = self._model.encode(
            [normalize_text(text)],
            device="cpu",
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        values = raw.tolist() if hasattr(raw, "tolist") else raw
        if not values:
            return zero_vector()
        if isinstance(values[0], list):
            return normalize_vector(values[0])
        return normalize_vector(values)

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.ensure() or self._model is None:
            return [zero_vector() for _ in texts]
        raw = self._model.encode(
            [normalize_text(text) for text in texts],
            device="cpu",
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        values = raw.tolist() if hasattr(raw, "tolist") else raw
        return [normalize_vector(item) for item in values] if values else [zero_vector() for _ in texts]


@dataclass
class EngineConfig:
    state_dir: Path = Path(".semantic_ants")
    vector_dim: int = DEFAULT_VECTOR_DIM
    window_size: int = 1
    graph_limit: int = 120
    backpack_limit: int = 48
    session_limit: int = 50
    session_context_turns: int = 4
    session_context_decay: float = 0.5
    result_limit: int = 200


class SemanticEngine:
    def __init__(self, config: EngineConfig | None = None, *, state_dir: Path | None = None) -> None:
        self.config = config or EngineConfig(state_dir=state_dir or Path(".semantic_ants"))
        if state_dir is not None:
            self.config.state_dir = Path(state_dir)
        self.config.vector_dim = DEFAULT_VECTOR_DIM
        self.config.window_size = 1
        self.state_path = self.config.state_dir / "checkpoint.sqlite"
        self._lock = RLock()
        self.jobs = JobManager()
        self._embedding_backend = _LazyEmbeddingBackend()
        self.checkpoint = load_checkpoint(self.state_path, default=Checkpoint())
        self.checkpoint.vector_dim = DEFAULT_VECTOR_DIM
        self._normalize_loaded_state()
        self._normalize_hierarchy_state()
        self._touch_meta("state_dir", str(self.config.state_dir))
        self._touch_meta("embedding_model", EMBEDDING_MODEL_NAME)

    def _normalize_loaded_state(self) -> None:
        normalized_tokens: dict[str, dict[str, Any]] = {}
        for token, record in self.checkpoint.tokens.items():
            cleaned = canonical_token(record.get("token") or token)
            if not cleaned:
                continue
            normalized_tokens[cleaned] = {
                "id": f"token:{cleaned}",
                "type": "token",
                "token": cleaned,
                "label": canonical_token(record.get("label") or record.get("surface") or cleaned) or cleaned,
                "count": int(record.get("count", 0) or 0),
                "vector": normalize_vector(record.get("vector")),
                "created_at": float(record.get("created_at", 0.0) or 0.0),
                "updated_at": float(record.get("updated_at", 0.0) or 0.0),
            }
        self.checkpoint.tokens = normalized_tokens
        root = self._ensure_root_hypernode()
        root["subgraph"] = self._normalize_subgraph_payload(root.get("subgraph"))

    def _normalize_hierarchy_state(self) -> None:
        hypernodes = self._hypernode_store()
        root = self._ensure_root_hypernode()
        normalized: dict[str, dict[str, Any]] = {}
        for node_id, record in hypernodes.items():
            if not isinstance(record, dict):
                continue
            cleaned_id = str(record.get("id") or node_id).strip()
            if not cleaned_id.startswith(HIERARCHY_NODE_PREFIX):
                cleaned_id = self._hierarchy_node_id(record.get("hierarchy") or cleaned_id.removeprefix(HIERARCHY_NODE_PREFIX))
            hierarchy = self._normalize_hierarchy(record.get("hierarchy") or cleaned_id.removeprefix(HIERARCHY_NODE_PREFIX).split("/"))
            if not hierarchy:
                continue
            parent_hierarchy = hierarchy[:-1]
            parent_id = self._hierarchy_node_id(parent_hierarchy) if parent_hierarchy else None
            normalized[cleaned_id] = {
                "id": cleaned_id,
                "type": "hypernode",
                "label": normalize_text(str(record.get("label") or hierarchy[-1])) or hierarchy[-1],
                "hierarchy": hierarchy,
                "parent": str(record.get("parent") or parent_id or "") or None,
                "depth": int(record.get("depth", len(hierarchy)) or len(hierarchy)),
                "count": int(record.get("count", 0) or 0),
                "vector": normalize_vector(record.get("vector")),
                "created_at": float(record.get("created_at", 0.0) or 0.0),
                "updated_at": float(record.get("updated_at", 0.0) or 0.0),
                "subgraph": self._normalize_subgraph_payload(record.get("subgraph")),
            }
        if ROOT_HYPERNODE_ID not in normalized:
            normalized[ROOT_HYPERNODE_ID] = root
        else:
            normalized[ROOT_HYPERNODE_ID]["subgraph"] = self._normalize_subgraph_payload(normalized[ROOT_HYPERNODE_ID].get("subgraph"))
        normalized[ROOT_HYPERNODE_ID]["id"] = ROOT_HYPERNODE_ID
        normalized[ROOT_HYPERNODE_ID]["type"] = "hypernode"
        normalized[ROOT_HYPERNODE_ID]["parent"] = None
        normalized[ROOT_HYPERNODE_ID]["depth"] = 0
        self.checkpoint.meta["hypernodes"] = normalized
        stacks = self._backpack_stack_store()
        cleaned_stacks: dict[str, list[str]] = {}
        for session_id, stack in stacks.items():
            if not isinstance(stack, list):
                continue
            cleaned_stack = [str(node_id) for node_id in stack if self._node_exists(str(node_id))]
            cleaned_stacks[str(session_id)] = cleaned_stack[-self.config.session_limit :]
        self.checkpoint.meta["backpack_stack"] = cleaned_stacks

    def _embedding_available(self) -> bool:
        return bool(self._embedding_backend.ensure())

    def _embedding_vector(self, text: str) -> list[float]:
        normalized = normalize_text(text)
        if not normalized:
            return zero_vector()
        return normalize_vector(self._embedding_backend.encode(normalized)) if self._embedding_backend.ensure() else zero_vector()

    def _embedding_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._embedding_backend.ensure():
            return [normalize_vector(vector) for vector in self._embedding_backend.encode_many(texts)]
        return [zero_vector() for _ in texts]

    def _token_embedding(self, token: str) -> list[float]:
        cleaned = canonical_token(token)
        if not cleaned or cleaned in PUNCT_TOKENS or cleaned in ROLE_TOKENS:
            return zero_vector()
        return self._embedding_vector(cleaned)

    def _hypernode_store(self) -> dict[str, dict[str, Any]]:
        store = self.checkpoint.meta.get("hypernodes")
        if not isinstance(store, dict):
            store = {}
            self.checkpoint.meta["hypernodes"] = store
        return store

    def _ensure_root_hypernode(self) -> dict[str, Any]:
        store = self._hypernode_store()
        record = store.get(ROOT_HYPERNODE_ID)
        if not isinstance(record, dict):
            record = {
                "id": ROOT_HYPERNODE_ID,
                "type": "hypernode",
                "label": "__root__",
                "hierarchy": [],
                "parent": None,
                "depth": 0,
                "count": 0,
                "vector": zero_vector(),
                "created_at": 0.0,
                "updated_at": 0.0,
                "subgraph": {"tokens": {}, "edges": {}},
            }
            store[ROOT_HYPERNODE_ID] = record
        if not isinstance(record.get("subgraph"), dict):
            record["subgraph"] = {"tokens": {}, "edges": {}}
        return record

    def _backpack_stack_store(self) -> dict[str, list[str]]:
        store = self.checkpoint.meta.get("backpack_stack")
        if not isinstance(store, dict):
            store = {}
            self.checkpoint.meta["backpack_stack"] = store
        return store

    def _session_backpack_stack(self, session_id: str) -> list[str]:
        store = self._backpack_stack_store()
        stack = store.get(session_id)
        if not isinstance(stack, list):
            stack = []
            store[session_id] = stack
        cleaned = [str(node_id) for node_id in stack if self._node_exists(str(node_id))]
        if cleaned != stack:
            store[session_id] = cleaned
        return store[session_id]

    def _normalize_hierarchy(self, hierarchy: Any) -> list[str]:
        if isinstance(hierarchy, str):
            parts = [part for part in hierarchy.replace("\\", "/").split("/") if part]
        elif isinstance(hierarchy, (list, tuple)):
            parts = [normalize_text(str(part)) for part in hierarchy if normalize_text(str(part))]
        else:
            parts = []
        return [normalize_text(part) for part in parts if normalize_text(part)]

    def _hierarchy_node_id(self, hierarchy: Any) -> str:
        parts = self._normalize_hierarchy(hierarchy)
        if not parts:
            return HIERARCHY_NODE_PREFIX
        slug = "/".join(part.casefold().replace(" ", "_") for part in parts)
        return f"{HIERARCHY_NODE_PREFIX}{slug}"

    def _node_exists(self, node_id: str) -> bool:
        if str(node_id).startswith("token:"):
            token = canonical_token(str(node_id).removeprefix("token:"))
            return token in self.checkpoint.tokens
        if str(node_id).startswith(HIERARCHY_NODE_PREFIX):
            return str(node_id) in self._hypernode_store()
        return False

    def _canonical_edge_relation(self, relation: Any) -> str:
        cleaned = str(relation or "next").strip()
        return "next" if cleaned == "transition_edge" else cleaned

    def _canonical_edge_node_id(self, node_id: Any) -> str:
        cleaned = str(node_id or "").strip()
        return token_id(cleaned) if cleaned.startswith("token:") else cleaned

    def _parse_edge_key(self, edge_id: Any) -> tuple[str, str, str] | None:
        parts = str(edge_id or "").split("|", 2)
        if len(parts) != 3:
            return None
        source, relation, target = parts
        source = self._canonical_edge_node_id(source)
        relation = self._canonical_edge_relation(relation)
        target = self._canonical_edge_node_id(target)
        if relation not in {"next", "hierarchical_edge"}:
            return None
        if source in {"token:", "hyper:"} or target in {"token:", "hyper:"}:
            return None
        if not source.startswith(("token:", HIERARCHY_NODE_PREFIX)) or not target.startswith(("token:", HIERARCHY_NODE_PREFIX)):
            return None
        return source, relation, target

    def _normalize_edge_key(self, edge_id: Any) -> str | None:
        parsed = self._parse_edge_key(edge_id)
        if parsed is None:
            return None
        source, relation, target = parsed
        return self._edge_key(source, relation, target)

    def _normalize_subgraph_payload(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"tokens": {}, "edges": {}}
        tokens: dict[str, dict[str, Any]] = {}
        raw_tokens = value.get("tokens")
        if isinstance(raw_tokens, dict):
            for key, token_record in raw_tokens.items():
                token = canonical_token(str(key).removeprefix("token:"))
                if not token:
                    continue
                cleaned = dict(token_record or {}) if isinstance(token_record, dict) else {}
                token_id_key = token_id(cleaned.get("id") or key)
                cleaned["id"] = token_id_key
                cleaned["type"] = "token"
                cleaned["token"] = token
                cleaned["label"] = normalize_text(str(cleaned.get("label") or cleaned.get("surface") or token)) or token
                cleaned["count"] = int(cleaned.get("count", 0) or 0)
                tokens[token_id_key] = cleaned
        elif isinstance(raw_tokens, list):
            for token_record in raw_tokens:
                if not isinstance(token_record, dict):
                    continue
                token = canonical_token(token_record.get("token") or token_record.get("label") or token_record.get("surface"))
                if not token:
                    continue
                token_id_key = token_id(token_record.get("id") or token)
                tokens[token_id_key] = {
                    "id": token_id_key,
                    "type": "token",
                    "token": token,
                    "label": normalize_text(str(token_record.get("label") or token_record.get("surface") or token)) or token,
                    "count": int(token_record.get("count", 0) or 0),
                }
        edges: dict[str, dict[str, float]] = {}
        raw_edges = value.get("edges")
        if isinstance(raw_edges, dict):
            for key, edge_record in raw_edges.items():
                if isinstance(edge_record, dict) and ("weight" in edge_record or "pheromone" in edge_record) and str(key).count("|") >= 2:
                    edge_key = self._normalize_edge_key(key)
                    if edge_key is None:
                        continue
                    payload = {
                        "weight": float(edge_record.get("weight", 0.0) or 0.0),
                        "pheromone": float(edge_record.get("pheromone", 0.0) or 0.0),
                    }
                else:
                    normalized = _normalize_slim_edge_record(edge_record)
                    if normalized is None:
                        continue
                    raw_edge_key, payload = normalized
                    edge_key = self._normalize_edge_key(raw_edge_key)
                    if edge_key is None:
                        continue
                existing = edges.get(edge_key)
                if existing is None:
                    edges[edge_key] = payload
                else:
                    existing["weight"] = float(existing.get("weight", 0.0)) + float(payload.get("weight", 0.0))
                    existing["pheromone"] = max(float(existing.get("pheromone", 0.0)), float(payload.get("pheromone", 0.0)))
        elif isinstance(raw_edges, list):
            for edge_record in raw_edges:
                normalized = _normalize_slim_edge_record(edge_record)
                if normalized is None:
                    continue
                raw_edge_key, payload = normalized
                edge_key = self._normalize_edge_key(raw_edge_key)
                if edge_key is None:
                    continue
                existing = edges.get(edge_key)
                if existing is None:
                    edges[edge_key] = payload
                else:
                    existing["weight"] = float(existing.get("weight", 0.0)) + float(payload.get("weight", 0.0))
                    existing["pheromone"] = max(float(existing.get("pheromone", 0.0)), float(payload.get("pheromone", 0.0)))
        if not tokens and isinstance(value.get("nodes"), list):
            for node in value.get("nodes", []) or []:
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("id") or "").strip()
                if node_id.startswith("token:"):
                    token = canonical_token(node_id.removeprefix("token:"))
                    if not token:
                        continue
                    token_id_key = token_id(node_id)
                    tokens[token_id_key] = {
                        "id": token_id_key,
                        "type": "token",
                        "token": token,
                        "label": normalize_text(str(node.get("label") or node.get("surface") or token)) or token,
                        "count": int(node.get("count", 0) or 0),
                    }
        for edge_key in edges:
            parsed = self._parse_edge_key(edge_key)
            if parsed is None:
                continue
            source, _, target = parsed
            for endpoint in (source, target):
                if not endpoint.startswith("token:") or endpoint in tokens:
                    continue
                token = canonical_token(endpoint.removeprefix("token:"))
                if not token:
                    continue
                record = self.checkpoint.tokens.get(token) or {}
                tokens[endpoint] = {
                    "id": endpoint,
                    "type": "token",
                    "token": token,
                    "label": normalize_text(str(record.get("label") or token)) or token,
                    "count": int(record.get("count", 0) or 0),
                }
        return {"tokens": tokens, "edges": edges}

    def _normalize_graph_node_payload(self, value: Any) -> dict[str, Any]:
        subgraph = self._normalize_subgraph_payload(value)
        return self._subgraph_to_graph_payload(ROOT_HYPERNODE_ID, subgraph, label="__root__")

    def _graph_edges_from_subgraph(self, node_id: str, subgraph: dict[str, Any]) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        for edge_id, payload in (subgraph.get("edges") or {}).items():
            parsed = self._parse_edge_key(edge_id)
            if parsed is None:
                continue
            source, relation, target = parsed
            edges.append(
                {
                    "id": self._edge_key(source, relation, target),
                    "source": source,
                    "target": target,
                    "relation": relation or "next",
                    "type": "transition_edge" if (relation or "next") == "next" else (relation or "next"),
                    "weight": float(payload.get("weight", 0.0)),
                    "pheromone": float(payload.get("pheromone", 0.0)),
                    "title": f"{relation or 'next'} | {float(payload.get('weight', 0.0)):.2f}",
                }
            )
        return edges

    def _subgraph_to_graph_payload(self, node_id: str, subgraph: dict[str, Any], *, label: str | None = None) -> dict[str, Any]:
        normalized = self._normalize_subgraph_payload(subgraph)
        nodes: dict[str, dict[str, Any]] = {}
        root_node = self._node_by_id(node_id) or {
            "id": node_id,
            "type": "hypernode",
            "label": label or node_id.removeprefix(HIERARCHY_NODE_PREFIX),
            "shape": "rect",
            "count": int(self._hypernode_store().get(node_id, {}).get("count", 0) or 0),
        }
        nodes[node_id] = {
            **root_node,
            "id": node_id,
            "type": "hypernode",
            "shape": "rect",
            "label": str(root_node.get("label") or label or node_id.removeprefix(HIERARCHY_NODE_PREFIX)),
        }
        for token_id_key, token_record in (normalized.get("tokens") or {}).items():
            if not str(token_id_key).startswith("token:"):
                continue
            token = canonical_token(str(token_record.get("token") or token_id_key.removeprefix("token:")))
            if not token:
                continue
            record = self.checkpoint.tokens.get(token) or {}
            nodes[token_id_key] = {
                "id": token_id_key,
                "type": "token",
                "token": token,
                "label": str(token_record.get("label") or record.get("label") or token),
                "shape": "circle",
                "count": int(record.get("count", token_record.get("count", 0)) or 0),
            }
        edges = self._graph_edges_from_subgraph(node_id, normalized)
        for edge in edges:
            if str(edge.get("source") or "").startswith(HIERARCHY_NODE_PREFIX) and str(edge["source"]) not in nodes:
                source_record = self._hypernode_store().get(str(edge["source"])) or {}
                nodes[str(edge["source"])] = {
                    "id": str(edge["source"]),
                    "type": "hypernode",
                    "label": str(source_record.get("label") or str(edge["source"]).removeprefix(HIERARCHY_NODE_PREFIX)),
                    "shape": "rect",
                    "count": int(source_record.get("count", 0) or 0),
                }
            if str(edge.get("target") or "").startswith(HIERARCHY_NODE_PREFIX) and str(edge["target"]) not in nodes:
                target_record = self._hypernode_store().get(str(edge["target"])) or {}
                nodes[str(edge["target"])] = {
                    "id": str(edge["target"]),
                    "type": "hypernode",
                    "label": str(target_record.get("label") or str(edge["target"]).removeprefix(HIERARCHY_NODE_PREFIX)),
                    "shape": "rect",
                    "count": int(target_record.get("count", 0) or 0),
                }
        node_list = list(nodes.values())
        return {
            "nodes": node_list,
            "edges": edges,
            "stats": {
                "nodes": len(node_list),
                "edges": len(edges),
                "tokens": len([node for node in node_list if str(node.get("type") or "") == "token"]),
            },
            "seed_ids": [node_id],
            "node_ids": [node["id"] for node in node_list],
            "edge_ids": [edge["id"] for edge in edges],
        }

    def _active_graph_record(self, session_id: str) -> tuple[str, dict[str, Any]]:
        focus_id = self._stack_focus_id(session_id)
        if focus_id and str(focus_id).startswith(HIERARCHY_NODE_PREFIX):
            record = self._hypernode_store().get(str(focus_id))
            if isinstance(record, dict):
                return str(focus_id), record
        root = self._ensure_root_hypernode()
        return ROOT_HYPERNODE_ID, root

    def _active_graph_edges(self, session_id: str | None = None) -> list[dict[str, Any]]:
        if session_id is None:
            record = self._ensure_root_hypernode()
        else:
            _, record = self._active_graph_record(session_id)
        return self._graph_edges_from_subgraph(str(record.get("id") or ROOT_HYPERNODE_ID), self._normalize_subgraph_payload(record.get("subgraph")))

    def _total_edge_count(self) -> int:
        total = 0
        for record in self._hypernode_store().values():
            if not isinstance(record, dict):
                continue
            total += len(self._normalize_subgraph_payload(record.get("subgraph")).get("edges", {}))
        return total

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "time": time.time(),
                "tokens": len(self.checkpoint.tokens),
                "edges": self._total_edge_count(),
                "results": len(self.checkpoint.results),
            }

    def config_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state_dir": str(self.config.state_dir),
                "vector_dim": DEFAULT_VECTOR_DIM,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "graph_limit": self.config.graph_limit,
                "backpack_limit": self.config.backpack_limit,
                "session_limit": self.config.session_limit,
                "session_context_turns": self.config.session_context_turns,
                "session_context_decay": self.config.session_context_decay,
                "result_limit": self.config.result_limit,
                "tokens": len(self.checkpoint.tokens),
                "edges": self._total_edge_count(),
                "sessions": len(self.checkpoint.sessions),
                "results": len(self.checkpoint.results),
            }

    def train_text(
        self,
        text: str,
        *,
        session_id: str = "default",
        epochs: int = 1,
        max_records: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        raw_text = str(text or "")
        normalized = normalize_text(raw_text)
        if not normalized:
            raise ValueError("text is required")
        epochs = max(int(epochs or 1), 1)
        max_records_val = max(int(max_records or 0), 0) if max_records is not None else 0
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "start",
                    "kind": "text",
                    "session_id": session_id,
                    "epochs": epochs,
                    "text_length": len(normalized),
                    "max_records": max_records,
                }
            )
        fragments = self._training_fragments(raw_text)
        report = self._empty_train_report(session_id=session_id, epochs=epochs)
        run = {
            "session_id": session_id,
            "kind": "text",
            "epochs": epochs,
            "text_length": len(normalized),
            "max_records": max_records,
            "started_at": time.time(),
            "sequences": 0,
        }
        with self._lock:
            for _ in range(epochs):
                if max_records_val > 0 and int(report.get("dataset_records", 0)) >= max_records_val:
                    break
                self._apply_training_fragments(fragments, report, run, session_id=session_id)
            self._finish_train_report(report, run)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "done",
                    "kind": "text",
                    "session_id": session_id,
                    "epochs": epochs,
                    "text_length": len(normalized),
                    "tokens": report["tokens"],
                    "edges": report["edges"],
                    "sequences": report["source_sequences"],
                    "max_records": max_records,
                }
            )
        return report

    def train_jsonl(
        self,
        path: str | Path,
        *,
        session_id: str = "default",
        epochs: int = 1,
        max_pairs: int | None = None,
        max_records: int | None = None,
        batch_size: int = 5000,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        if dataset_path.is_dir():
            raise IsADirectoryError(dataset_path)
        epochs = max(int(epochs or 1), 1)
        max_records_val = max(int(max_records or 0), 0) if max_records is not None else 0
        dataset_mode = self._dataset_mode(dataset_path)
        if dataset_mode == "raw_jsonl":
            return self._train_raw_dialogue_jsonl(
                dataset_path,
                session_id=session_id,
                epochs=epochs,
                max_pairs=max_pairs,
                max_records=max_records_val,
                batch_size=batch_size,
                progress_callback=progress_callback,
            )
        if dataset_mode == "hierarchy_jsonl":
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "start",
                        "kind": "hierarchy_jsonl",
                        "session_id": session_id,
                        "epochs": epochs,
                        "dataset_path": str(dataset_path),
                        "max_records": max_records,
                    }
                )
            report = self._empty_train_report(session_id=session_id, epochs=epochs)
            run = {
                "session_id": session_id,
                "kind": "hierarchy_jsonl",
                "dataset_path": str(dataset_path),
                "started_at": time.time(),
                "records": 0,
                "sequences": 0,
                "hypernodes": 0,
            }
            with self._lock:
                for epoch_idx in range(epochs):
                    with dataset_path.open("r", encoding="utf-8") as handle:
                        for raw_line in handle:
                            line = raw_line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError as exc:
                                raise ValueError(f"invalid jsonl record in {dataset_path}") from exc
                            if not isinstance(record, dict):
                                continue
                            text = normalize_text(str(record.get("text") or record.get("content") or record.get("body") or ""))
                            hierarchy = self._normalize_hierarchy(record.get("hierarchy"))
                            if not text and not hierarchy:
                                continue
                            records_done = int(run.get("records", 0)) + 1
                            if max_records_val > 0 and records_done > max_records_val:
                                break
                            run["records"] = records_done
                            report["dataset_records"] = int(report.get("dataset_records", 0)) + 1
                            self._train_hierarchy_record(record, report, run, session_id=session_id)
                            if progress_callback is not None and records_done % _PROGRESS_INTERVAL == 0:
                                progress_callback(
                                    {
                                        "phase": "progress",
                                        "kind": "hierarchy_jsonl",
                                        "session_id": session_id,
                                        "dataset_path": str(dataset_path),
                                        "max_records": max_records,
                                        "records": records_done,
                                        "sequences": int(run.get("sequences", 0)),
                                    }
                                )
                    if max_records_val > 0 and int(run.get("records", 0)) >= max_records_val:
                        break
                self._finish_train_report(report, run)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "done",
                        "kind": "hierarchy_jsonl",
                        "session_id": session_id,
                        "epochs": epochs,
                        "dataset_path": str(dataset_path),
                        "max_records": max_records,
                        "dataset_records": report["dataset_records"],
                        "tokens": report["tokens"],
                        "edges": report["edges"],
                        "sequences": report["source_sequences"],
                    }
                )
            return report
        if dataset_mode == "text":
            return self._train_corpus_file(
                dataset_path,
                session_id=session_id,
                epochs=epochs,
                max_records=max_records_val,
                progress_callback=progress_callback,
                dataset_path=str(dataset_path),
                corpus_kind="corpus",
            )

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "start",
                    "kind": "jsonl",
                    "session_id": session_id,
                    "epochs": epochs,
                    "dataset_path": str(dataset_path),
                    "max_records": max_records,
                }
            )
        report = self._empty_train_report(session_id=session_id, epochs=epochs)
        run = {
            "session_id": session_id,
            "kind": "jsonl",
            "dataset_path": str(dataset_path),
            "started_at": time.time(),
            "records": 0,
            "sequences": 0,
        }
        with self._lock:
            for epoch_idx in range(epochs):
                with dataset_path.open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"invalid jsonl record in {dataset_path}") from exc
                        text = self._jsonl_record_to_text(record)
                        if not text:
                            continue
                        records_done = int(run.get("records", 0)) + 1
                        if max_records_val > 0 and records_done > max_records_val:
                            break
                        run["records"] = records_done
                        report["dataset_records"] = int(report.get("dataset_records", 0)) + 1
                        self._apply_training_fragments(self._training_fragments(text), report, run, session_id=session_id)
                        if progress_callback is not None and records_done % _PROGRESS_INTERVAL == 0:
                            progress_callback(
                                {
                                    "phase": "progress",
                                    "kind": "jsonl",
                                    "session_id": session_id,
                                    "dataset_path": str(dataset_path),
                                    "max_records": max_records,
                                    "records": records_done,
                                    "sequences": int(run.get("sequences", 0)),
                                }
                            )
                    if max_records_val > 0 and int(run.get("records", 0)) >= max_records_val:
                        break
            self._finish_train_report(report, run)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "done",
                    "kind": "jsonl",
                    "session_id": session_id,
                    "epochs": epochs,
                    "dataset_path": str(dataset_path),
                    "max_records": max_records,
                    "dataset_records": report["dataset_records"],
                    "tokens": report["tokens"],
                    "edges": report["edges"],
                    "sequences": report["source_sequences"],
                }
            )
        return report

    def _train_corpus_file(
        self,
        corpus_path: Path,
        *,
        session_id: str,
        epochs: int,
        max_records: int = 0,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        dataset_path: str,
        corpus_kind: str,
    ) -> dict[str, Any]:
        if not corpus_path.exists():
            raise FileNotFoundError(corpus_path)
        if corpus_path.is_dir():
            raise IsADirectoryError(corpus_path)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "start",
                    "kind": corpus_kind,
                    "session_id": session_id,
                    "epochs": epochs,
                    "dataset_path": dataset_path,
                    "corpus_path": str(corpus_path),
                    "max_records": max_records,
                }
            )
        report = self._empty_train_report(session_id=session_id, epochs=epochs)
        run = {
            "session_id": session_id,
            "kind": corpus_kind,
            "dataset_path": dataset_path,
            "corpus_path": str(corpus_path),
            "started_at": time.time(),
            "records": 0,
            "sequences": 0,
        }
        with self._lock:
            for _ in range(epochs):
                with corpus_path.open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        text = normalize_text(raw_line)
                        if not text:
                            continue
                        records_done = int(run.get("records", 0)) + 1
                        if max_records > 0 and records_done > max_records:
                            break
                        run["records"] = records_done
                        report["dataset_records"] = int(report.get("dataset_records", 0)) + 1
                        self._apply_training_fragments(self._training_fragments(text), report, run, session_id=session_id)
                        if progress_callback is not None and records_done % _PROGRESS_INTERVAL == 0:
                            progress_callback(
                                {
                                    "phase": "progress",
                                    "kind": corpus_kind,
                                    "session_id": session_id,
                                    "dataset_path": dataset_path,
                                    "max_records": max_records,
                                    "records": records_done,
                                    "sequences": int(run.get("sequences", 0)),
                                }
                            )
                if max_records > 0 and int(run.get("records", 0)) >= max_records:
                    break
            self._finish_train_report(report, run)
        report["dataset_path"] = dataset_path
        report["corpus_path"] = str(corpus_path)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "done",
                    "kind": corpus_kind,
                    "session_id": session_id,
                    "epochs": epochs,
                    "dataset_path": dataset_path,
                    "corpus_path": str(corpus_path),
                    "max_records": max_records,
                    "dataset_records": report["dataset_records"],
                    "tokens": report["tokens"],
                    "edges": report["edges"],
                    "sequences": report["source_sequences"],
                }
            )
        return report

    def _train_raw_dialogue_jsonl(
        self,
        dataset_path: Path,
        *,
        session_id: str,
        epochs: int,
        max_pairs: int | None,
        max_records: int,
        batch_size: int,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        batch_size = max(int(batch_size or 5000), 1)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "start",
                    "kind": "raw_jsonl",
                    "session_id": session_id,
                    "epochs": epochs,
                    "dataset_path": str(dataset_path),
                    "max_records": max_records or None,
                    "batch_size": batch_size,
                }
            )
        effective_limit = None
        for candidate in (max_pairs, max_records):
            if candidate is None:
                continue
            candidate_value = max(int(candidate or 0), 0)
            if candidate_value <= 0:
                continue
            effective_limit = candidate_value if effective_limit is None else min(effective_limit, candidate_value)
        pair_store_path = self._temporary_pair_store_path(dataset_path)
        try:
            accepted_pairs, unique_pairs = self._build_dialogue_pair_store(
                dataset_path,
                pair_store_path,
                max_pairs=effective_limit,
                batch_size=batch_size,
                progress_callback=progress_callback,
                session_id=session_id,
            )
            report = self._empty_train_report(session_id=session_id, epochs=epochs)
            report["dataset_records"] = accepted_pairs
            report["unique_pairs"] = unique_pairs
            report["duplicates_collapsed"] = accepted_pairs - unique_pairs
            report["checkpoint_format"] = "sqlite"
            run = {
                "session_id": session_id,
                "kind": "raw_jsonl",
                "dataset_path": str(dataset_path),
                "started_at": time.time(),
                "records": 0,
                "sequences": 0,
                "unique_pairs": unique_pairs,
                "duplicates_collapsed": accepted_pairs - unique_pairs,
            }
            pending_since_save = 0
            with self._lock:
                for _ in range(epochs):
                    if max_records > 0 and int(run.get("records", 0)) >= max_records:
                        break
                    for prompt, response, weight in self._iter_dialogue_pair_store(pair_store_path):
                        records_done = int(run.get("records", 0)) + weight
                        if max_records > 0 and records_done > max_records:
                            break
                        run["records"] = records_done
                        self._train_dialogue_pair(prompt, response, report, run, subgraph_node_id=None, weight=weight)
                        pending_since_save += weight
                        if pending_since_save >= batch_size:
                            self._trim_collections()
                            self._persist()
                            pending_since_save = 0
                            if progress_callback is not None:
                                progress_callback(
                                    {
                                        "phase": "progress",
                                        "kind": "raw_jsonl",
                                        "session_id": session_id,
                                        "dataset_path": str(dataset_path),
                                        "max_records": max_records or None,
                                        "records": int(run.get("records", 0)),
                                        "sequences": int(run.get("sequences", 0)),
                                        "unique_pairs": report["unique_pairs"],
                                        "duplicates_collapsed": report["duplicates_collapsed"],
                                    }
                                )
                    if max_records > 0 and int(run.get("records", 0)) >= max_records:
                        break
                self._finish_train_report(report, run)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "done",
                        "kind": "raw_jsonl",
                        "session_id": session_id,
                        "epochs": epochs,
                        "dataset_path": str(dataset_path),
                        "max_records": max_records or None,
                        "dataset_records": report["dataset_records"],
                        "unique_pairs": report["unique_pairs"],
                        "duplicates_collapsed": report["duplicates_collapsed"],
                        "checkpoint_format": report["checkpoint_format"],
                        "tokens": report["tokens"],
                        "edges": report["edges"],
                        "sequences": report["source_sequences"],
                    }
                )
            return report
        finally:
            pair_store_path.unlink(missing_ok=True)

    def _temporary_pair_store_path(self, dataset_path: Path) -> Path:
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{time.time_ns()}-{random.randrange(1_000_000):06d}"
        return self.config.state_dir / f"{dataset_path.stem}.{suffix}.pairs.sqlite"

    def _build_dialogue_pair_store(
        self,
        dataset_path: Path,
        store_path: Path,
        *,
        max_pairs: int | None,
        batch_size: int,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        session_id: str,
    ) -> tuple[int, int]:
        if store_path.exists():
            store_path.unlink()
        accepted = 0
        batch: list[tuple[str, str]] = []
        conn = sqlite3.connect(store_path)
        try:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute(
                "CREATE TABLE dialogue_pairs (prompt TEXT NOT NULL, response TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY (prompt, response)) WITHOUT ROWID"
            )
            for prompt, response in iter_valid_dialogue_pairs(dataset_path, max_pairs=max_pairs):
                batch.append((prompt, response))
                accepted += 1
                if len(batch) >= batch_size:
                    self._flush_dialogue_pair_store_batch(conn, batch)
                    batch.clear()
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "phase": "progress",
                                "kind": "raw_jsonl_dedupe",
                                "session_id": session_id,
                                "dataset_path": str(dataset_path),
                                "records": accepted,
                            }
                        )
            if batch:
                self._flush_dialogue_pair_store_batch(conn, batch)
            conn.commit()
            unique = int(conn.execute("SELECT COUNT(*) FROM dialogue_pairs").fetchone()[0] or 0)
            return accepted, unique
        finally:
            conn.close()

    def _flush_dialogue_pair_store_batch(self, conn: sqlite3.Connection, batch: list[tuple[str, str]]) -> None:
        conn.executemany(
            """
            INSERT INTO dialogue_pairs (prompt, response, count)
            VALUES (?, ?, 1)
            ON CONFLICT(prompt, response) DO UPDATE SET count = count + 1
            """,
            batch,
        )
        conn.commit()

    def _iter_dialogue_pair_store(self, store_path: Path) -> Iterator[tuple[str, str, int]]:
        conn = sqlite3.connect(store_path)
        try:
            for prompt, response, count in conn.execute("SELECT prompt, response, count FROM dialogue_pairs"):
                yield str(prompt), str(response), max(int(count or 0), 1)
        finally:
            conn.close()

    def chat(self, text: str, *, session_id: str = "default", backpack_limit: int | None = None) -> dict[str, Any]:
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError("text is required")
        with self._lock:
            query_tokens = tokenize(normalized)
            prompt_tail_token = next((item["token"] for item in reversed(tokenize_with_surfaces(normalized)) if not is_role_token(item["token"])), "")
            stack_vector = self._stack_focus_vector(session_id)
            stack_focus_id = self._stack_focus_id(session_id)
            backpack = self._build_dense_backpack(normalized, session_id=session_id, focus_vector=stack_vector)
            scoring_vector = stack_vector if self._vector_norm(stack_vector) > 0.0 else backpack["query_vector"]
            top_tokens = self._score_tokens(query_tokens, scoring_vector)[:8]
            milestones = self._generate_semantic_plan(scoring_vector, query_tokens=query_tokens)
            focused_response = self._synthesize_hypernode_response(stack_focus_id, query_tokens=query_tokens) if stack_focus_id else None
            if focused_response is None:
                response, response_source = self._synthesize_response(query_tokens, backpack, session_id=session_id, milestones=milestones, prompt_tail_token=prompt_tail_token)
            else:
                response, response_source = focused_response
            result_id = sha_id("result", f"{time.time_ns()}:{normalized}:{response}")
            current_depth = self._stack_depth(session_id)
            total_depth_layers = self._total_depth_layers()
            active_focus_label = self._focus_label(stack_focus_id) if stack_focus_id else normalized
            result = {
                "result_id": result_id,
                "input_text": normalized,
                "session_id": session_id,
                "created_at": time.time(),
                "tokens": query_tokens,
                "known_tokens": [item["token"] for item in top_tokens if item["known"]],
                "top_tokens": top_tokens,
                "response": response,
                "response_source": response_source,
                "summary": self._summary_line(top_tokens, response_source),
                "trace": {
                    "query_tokens": query_tokens,
                    "query_vector_norm": self._vector_norm(backpack["query_vector"]),
                    "focus_vector_norm": self._vector_norm(scoring_vector),
                    "history_vectors": [
                        {
                            "role": item["role"],
                            "weight": item["weight"],
                            "text": item["text"],
                            "vector_norm": self._vector_norm(item["vector"]),
                        }
                        for item in backpack["history_vectors"]
                    ],
                    "token_scores": top_tokens,
                    "backpack_stack": self._stack_snapshot(session_id),
                },
            }
            graph_backpack = self._build_backpack_graph(
                session_id=session_id,
                query=normalized,
                limit=backpack_limit or self.config.backpack_limit,
                seed_ids=[item["node_id"] for item in top_tokens[:4]],
                highlight_result_id=None,
            )
            backpack_layers = self._build_recursive_backpack_layers(
                base_graph=graph_backpack,
                query=normalized,
                limit=backpack_limit or self.config.backpack_limit,
                seed_ids=[item["node_id"] for item in top_tokens[:4]],
                highlight_result_id=None,
            )
            graph_backpack["layers"] = backpack_layers
            graph_backpack["layer_count"] = len(backpack_layers)
            graph_backpack["current_depth"] = current_depth
            graph_backpack["total_depth_layers"] = total_depth_layers
            graph_backpack["active_focus_label"] = active_focus_label
            graph_backpack["stack"] = self._stack_snapshot(session_id)
            result["backpack"] = {
                "current_depth": current_depth,
                "total_depth_layers": total_depth_layers,
                "active_focus_label": active_focus_label,
                "graph_id": graph_backpack.get("graph_id", ""),
                "graph_data": graph_backpack,
                "layers": backpack_layers,
                "stack": self._stack_snapshot(session_id),
                "query_vector": backpack["query_vector"],
                "raw_query_vector": backpack["raw_query_vector"],
                "focus_vector": scoring_vector,
                "history_vectors": backpack["history_vectors"],
            }
            self._store_result(result, session_id=session_id, user_text=normalized)
            graph = self._build_graph(
                query=normalized,
                limit=self.config.graph_limit,
                seed_ids=[item["node_id"] for item in top_tokens[:4]],
                highlight_result_id=result_id,
            )
            self._persist()
            return {
                "result": result,
                "graph": graph,
                "graph_data": graph_backpack,
                "backpack": result["backpack"],
                "current_depth": current_depth,
                "total_depth_layers": total_depth_layers,
                "active_focus_label": active_focus_label,
                "trace": result["trace"],
            }

    def drill_down(self, node_id: str, *, session_id: str = "default", limit: int | None = None) -> dict[str, Any]:
        with self._lock:
            if not self._node_exists(node_id):
                raise KeyError(node_id)
            stack = self._session_backpack_stack(session_id)
            stack.append(str(node_id))
            self._trim_stack(session_id)
            self._persist()
            return self._backpack_snapshot(session_id=session_id, query=None, limit=limit)

    def drill_up(self, *, session_id: str = "default", limit: int | None = None) -> dict[str, Any]:
        with self._lock:
            stack = self._session_backpack_stack(session_id)
            if stack:
                stack.pop()
            self._trim_stack(session_id)
            self._persist()
            return self._backpack_snapshot(session_id=session_id, query=None, limit=limit)

    def _stack_snapshot(self, session_id: str) -> list[dict[str, Any]]:
        stack = self._session_backpack_stack(session_id)
        snapshot: list[dict[str, Any]] = []
        for node_id in stack:
            snapshot.append(
                {
                    "id": node_id,
                    "label": self._focus_label(node_id),
                    "type": "hypernode" if str(node_id).startswith(HIERARCHY_NODE_PREFIX) else "token",
                    "vector_norm": round(self._vector_norm(self._focus_vector_for_node(node_id)), 6),
                }
            )
        return snapshot

    def _stack_depth(self, session_id: str) -> int:
        return len(self._session_backpack_stack(session_id))

    def _stack_focus_id(self, session_id: str) -> str | None:
        stack = self._session_backpack_stack(session_id)
        return stack[-1] if stack else None

    def _stack_focus_vector(self, session_id: str) -> list[float]:
        focus_id = self._stack_focus_id(session_id)
        if focus_id:
            return self._focus_vector_for_node(focus_id)
        return zero_vector()

    def _focus_vector_for_node(self, node_id: str) -> list[float]:
        if str(node_id).startswith("token:"):
            token = canonical_token(str(node_id).removeprefix("token:"))
            record = self.checkpoint.tokens.get(token)
            return normalize_vector(record.get("vector")) if record else zero_vector()
        if str(node_id).startswith(HIERARCHY_NODE_PREFIX):
            record = self._hypernode_store().get(str(node_id))
            return normalize_vector(record.get("vector")) if record else zero_vector()
        return zero_vector()

    def _focus_label(self, node_id: str | None) -> str:
        if not node_id:
            return ""
        if str(node_id).startswith("token:"):
            token = canonical_token(str(node_id).removeprefix("token:"))
            record = self.checkpoint.tokens.get(token)
            return normalize_text(str(record.get("label") or token)) if record else token
        if str(node_id).startswith(HIERARCHY_NODE_PREFIX):
            record = self._hypernode_store().get(str(node_id))
            if record:
                hierarchy = list(record.get("hierarchy") or [])
                fallback = hierarchy[-1] if hierarchy else node_id.removeprefix(HIERARCHY_NODE_PREFIX)
                return normalize_text(str(record.get("label") or fallback))
            return normalize_text(str(node_id.removeprefix(HIERARCHY_NODE_PREFIX)))
        return normalize_text(str(node_id))

    def _total_depth_layers(self) -> int:
        hypernodes = self._hypernode_store()
        if not hypernodes:
            return 1
        return max((int(record.get("depth", 1)) for record in hypernodes.values() if isinstance(record, dict)), default=1)

    def _trim_stack(self, session_id: str) -> None:
        stack = self._session_backpack_stack(session_id)
        if len(stack) > self.config.session_limit:
            self.checkpoint.meta["backpack_stack"][session_id] = stack[-self.config.session_limit :]

    def _backpack_snapshot(
        self,
        *,
        session_id: str,
        query: str | None,
        limit: int | None,
        result_id: str | None = None,
    ) -> dict[str, Any]:
        focus_id = self._stack_focus_id(session_id)
        current_depth = self._stack_depth(session_id)
        total_depth_layers = self._total_depth_layers()
        active_focus_label = self._focus_label(focus_id) if focus_id else normalize_text(query or "")
        seed_ids = [focus_id] if focus_id else []
        graph_data = self._build_backpack_graph(
            session_id=session_id,
            query=query,
            limit=limit or self.config.backpack_limit,
            seed_ids=seed_ids,
            highlight_result_id=result_id,
        )
        graph_data["current_depth"] = current_depth
        graph_data["total_depth_layers"] = total_depth_layers
        graph_data["active_focus_label"] = active_focus_label
        graph_data["stack"] = self._stack_snapshot(session_id)
        return {
            "current_depth": current_depth,
            "total_depth_layers": total_depth_layers,
            "active_focus_label": active_focus_label,
            "graph_id": graph_data.get("graph_id", ""),
            "graph_data": graph_data,
            "stack": self._stack_snapshot(session_id),
        }

    def _build_backpack_graph(
        self,
        *,
        session_id: str,
        query: str | None,
        limit: int,
        seed_ids: list[str],
        highlight_result_id: str | None,
    ) -> dict[str, Any]:
        stack_focus_id = self._stack_focus_id(session_id)
        if stack_focus_id and str(stack_focus_id).startswith(HIERARCHY_NODE_PREFIX):
            return self._build_hypernode_graph(stack_focus_id, limit=limit, highlight_result_id=highlight_result_id)
        focus_seed_ids = [node_id for node_id in ([stack_focus_id] if stack_focus_id else []) + list(seed_ids) if node_id]
        if not focus_seed_ids and query:
            query_vector = self._embedding_vector(query)
            focus_seed_ids = [item["node_id"] for item in self._score_tokens(tokenize(query), query_vector)[:4]]
        return self._build_graph(
            query=query,
            limit=limit,
            seed_ids=focus_seed_ids,
            highlight_result_id=highlight_result_id,
        )

    def _build_hypernode_graph(self, node_id: str, *, limit: int, highlight_result_id: str | None) -> dict[str, Any]:
        record = self._hypernode_store().get(node_id)
        if not record:
            return {"nodes": [], "edges": [], "stats": {"nodes": 0, "edges": 0, "tokens": 0}, "seed_ids": [node_id], "graph_id": node_id}
        graph = self._subgraph_to_graph_payload(node_id, record.get("subgraph"), label=str(record.get("label") or node_id))
        if not graph.get("nodes"):
            graph["nodes"] = [
                {
                    "id": node_id,
                    "type": "hypernode",
                    "label": str(record.get("label") or node_id),
                    "shape": "rect",
                    "count": int(record.get("count", 0)),
                    "score": 0.0,
                    "active": True,
                    "x": 500.0,
                    "y": 350.0,
                }
            ]
            graph["edges"] = []
        positions = self._layout_graph(graph["nodes"], graph["edges"], seed_ids=[node_id])
        nodes = [{**node, **positions.get(node["id"], {"x": 500.0, "y": 350.0}), "active": node["id"] == node_id} for node in graph["nodes"]]
        edges: list[dict[str, Any]] = []
        for edge in graph["edges"]:
            source = positions.get(edge["source"], {"x": 500.0, "y": 350.0})
            target = positions.get(edge["target"], {"x": 500.0, "y": 350.0})
            edges.append(
                {
                    **edge,
                    "x1": source["x"],
                    "y1": source["y"],
                    "x2": target["x"],
                    "y2": target["y"],
                    "active": edge["source"] == node_id or edge["target"] == node_id,
                }
            )
        graph_id = sha_id("graph", f"hyper:{node_id}:{limit}:{record.get('updated_at', 0.0)}")
        payload = {
            "nodes": nodes[: max(1, limit)],
            "edges": edges[: max(limit * 2, 24)],
            "stats": {
                "nodes": len(nodes),
                "edges": len(edges),
                "tokens": max(len(nodes) - 1, 0),
                "seeds": 1,
            },
            "seed_ids": [node_id],
            "node_ids": [node["id"] for node in nodes],
            "edge_ids": [edge["id"] for edge in edges],
            "graph_id": graph_id,
        }
        if highlight_result_id and highlight_result_id in self.checkpoint.results:
            payload["stats"]["result_id"] = highlight_result_id
        return payload

    def graph(self, *, query: str | None = None, limit: int | None = None, result_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            highlight_result_id = result_id if result_id and result_id in self.checkpoint.results else None
            normalized_query = normalize_text(query or "")
            seed_ids: list[str] = []
            if normalized_query:
                query_vector = self._embedding_vector(normalized_query)
                seed_ids = [item["node_id"] for item in self._score_tokens(tokenize(normalized_query), query_vector)[:4]]
            elif highlight_result_id:
                result = self.checkpoint.results[highlight_result_id]
                seed_ids = list(result.get("backpack", {}).get("seed_ids", []))
            return self._build_graph(
                query=normalized_query or None,
                limit=limit or self.config.graph_limit,
                seed_ids=seed_ids,
                highlight_result_id=highlight_result_id,
            )

    def node_detail(self, node_id: str) -> dict[str, Any]:
        with self._lock:
            node = self._node_by_id(node_id)
            if node is None:
                raise KeyError(node_id)
            return {
                "node": node,
                "neighbors": self._neighbors(node_id)[:16],
                "examples": [],
            }

    def sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            items = []
            for session_id, turns in self.checkpoint.sessions.items():
                updated_at = max((float(turn.get("created_at", 0.0)) for turn in turns), default=0.0)
                items.append(
                    {
                        "session_id": session_id,
                        "turns": turns,
                        "turn_count": len(turns),
                        "updated_at": updated_at,
                    }
                )
            return sorted(items, key=lambda item: item["updated_at"], reverse=True)

    def reset_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            self.checkpoint.sessions[session_id] = []
            self._backpack_stack_store().pop(session_id, None)
            self._persist()
        return {"session_id": session_id, "reset": True}

    def feedback(self, *, result_id: str, score: int, corrected_response: str | None = None) -> dict[str, Any]:
        with self._lock:
            result = self.checkpoint.results.get(result_id)
            if not result:
                raise KeyError(result_id)
            amount = (0.12 * max(int(score), 0)) if int(score) > 0 else -0.08
            for item in result.get("top_tokens", [])[:8]:
                node_id = str(item.get("node_id") or "")
                if node_id:
                    self._reinforce_node_neighbors(node_id, amount)
            if corrected_response:
                prompt_text = normalize_text(str(result.get("input_text") or ""))
                response_text = normalize_text(corrected_response)
                if prompt_text and response_text:
                    report = self._empty_train_report(session_id=str(result.get("session_id") or "default"), epochs=1)
                    run = {"session_id": str(result.get("session_id") or "default"), "kind": "feedback", "sequences": 0}
                    self._train_dialogue_pair(prompt_text, response_text, report, run)
            self._persist()
            return {
                "result_id": result_id,
                "score": score,
                "corrected_response": corrected_response,
                "updated": True,
            }

    def train_job(self, *, text: str, session_id: str = "default", epochs: int = 1) -> dict[str, Any]:
        return self.train_text(text, session_id=session_id, epochs=epochs)

    def chat_job(self, *, text: str, session_id: str = "default", backpack_limit: int | None = None) -> dict[str, Any]:
        return self.chat(text, session_id=session_id, backpack_limit=backpack_limit)

    def save(self) -> None:
        with self._lock:
            self._persist()

    def analyze(self, text: str, **kwargs: Any) -> dict[str, Any]:
        return self.chat(text, session_id=str(kwargs.get("session_id") or "default"))["result"]

    def analyze_with_graph(self, text: str, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = self.chat(text, session_id=str(kwargs.get("session_id") or "default"))
        return payload["result"], payload["graph"]

    def _empty_train_report(self, *, session_id: str, epochs: int) -> dict[str, Any]:
        return {
            "trained": True,
            "session_id": session_id,
            "epochs": epochs,
            "source_sequences": 0,
            "source_pairs": 0,
            "dataset_records": 0,
            "unique_pairs": 0,
            "duplicates_collapsed": 0,
            "checkpoint_format": "sqlite",
            "new_tokens": 0,
            "updated_tokens": 0,
            "new_edges": 0,
            "updated_edges": 0,
            "tokens": 0,
            "edges": 0,
        }

    def _finish_train_report(self, report: dict[str, Any], run: dict[str, Any]) -> None:
        self._trim_collections()
        run["finished_at"] = time.time()
        run["tokens"] = len(self.checkpoint.tokens)
        run["edges"] = self._total_edge_count()
        training_runs = self.checkpoint.meta.setdefault("training_runs", [])
        if isinstance(training_runs, list):
            training_runs.append(run)
            self.checkpoint.meta["training_runs"] = training_runs[-50:]
        report["tokens"] = len(self.checkpoint.tokens)
        report["edges"] = self._total_edge_count()
        self._persist()

    def _dataset_mode(self, path: Path) -> str:
        sample_lines: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line:
                    sample_lines.append(line)
                if len(sample_lines) >= 10:
                    break
        if not sample_lines:
            return "text"
        json_records = 0
        raw_records = 0
        hierarchy_records = 0
        for line in sample_lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return "text"
            json_records += 1
            if looks_like_raw_dialogue_jsonl_record(record):
                raw_records += 1
            if isinstance(record, dict) and ("hierarchy" in record or "text" in record):
                hierarchy_records += 1
        if json_records and raw_records == json_records:
            return "raw_jsonl"
        if json_records and hierarchy_records == json_records:
            return "hierarchy_jsonl"
        return "jsonl"

    def _temporary_corpus_path(self, dataset_path: Path) -> Path:
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            prefix=f"{dataset_path.stem}.",
            suffix=".preprocessed.txt",
            dir=self.config.state_dir,
            delete=False,
        )
        handle.close()
        return Path(handle.name)

    def _training_fragments(self, text: str) -> list[dict[str, Any]]:
        raw_text = str(text or "").replace("\r", "\n").strip()
        if not raw_text:
            return []
        fragments: list[dict[str, Any]] = []
        paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", raw_text) if chunk.strip()]
        for paragraph in paragraphs:
            turns = self._dialogue_turns_from_paragraph(paragraph)
            if turns:
                fragments.append({"kind": "dialogue_turns", "turns": turns})
                continue
            for line in [normalize_text(item) for item in paragraph.split("\n") if normalize_text(item)]:
                for sentence in split_sentences(line):
                    if normalize_text(sentence):
                        fragments.append({"kind": "sequence", "text": sentence})
        return fragments

    def _dialogue_turns_from_paragraph(self, paragraph: str) -> list[dict[str, str]]:
        lines = [normalize_text(line) for line in str(paragraph or "").split("\n") if normalize_text(line)]
        turns: list[dict[str, str]] = []
        for line in lines:
            tagged_pair = self._extract_tagged_dialogue_pair(line)
            if tagged_pair is not None:
                prompt, response = tagged_pair
                turns.append({"role": "user", "text": prompt})
                turns.append({"role": "assistant", "text": response})
                continue
            role_match = DIALOGUE_ROLE_RE.match(line)
            if role_match:
                role = normalize_dialogue_role(str(role_match.group(1) or ""))
                content = strip_dialogue_markers(str(role_match.group(2) or ""))
                if role and content:
                    turns.append({"role": role, "text": content})
                    continue
            pair = self._extract_dialogue_pair(line)
            if pair is not None:
                prompt, response = pair
                turns.append({"role": "user", "text": prompt})
                turns.append({"role": "assistant", "text": response})
        if turns:
            return turns
        if len(lines) > 1 and any(self._dialogue_line_looks_like_chain(line) for line in lines):
            utterances: list[str] = []
            for line in lines:
                utterances.extend(self._dialogue_utterances_from_line(line))
            for index, utterance in enumerate(item for item in utterances if item):
                turns.append({"role": "user" if index % 2 == 0 else "assistant", "text": utterance})
        return turns

    def _extract_tagged_dialogue_pair(self, text: str) -> tuple[str, str] | None:
        cleaned = normalize_text(text)
        user_markers = (LEGACY_ROLE_USER_TOKEN, ROLE_USER_TOKEN)
        assistant_markers = (LEGACY_ROLE_ASSISTANT_TOKEN, ROLE_ASSISTANT_TOKEN)
        user_marker = next((marker for marker in user_markers if cleaned.casefold().startswith(marker)), "")
        if not user_marker:
            return None
        assistant_index = -1
        assistant_marker = ""
        lowered = cleaned.casefold()
        for marker in assistant_markers:
            index = lowered.find(marker, len(user_marker))
            if index != -1:
                assistant_index = index
                assistant_marker = marker
                break
        if assistant_index == -1:
            return None
        prompt = strip_dialogue_markers(cleaned[len(user_marker) : assistant_index])
        response = strip_dialogue_markers(cleaned[assistant_index + len(assistant_marker) :])
        if prompt and response:
            return prompt, response
        return None

    def _extract_dialogue_pair(self, text: str) -> tuple[str, str] | None:
        cleaned = normalize_text(text)
        if not cleaned:
            return None
        for separator in DIALOGUE_PAIR_SEPARATORS:
            if separator not in cleaned:
                continue
            if separator in {" - ", " — ", " – "}:
                prompt_part, response_part = cleaned.split(separator, 1)
                if not dialogue_separator_allowed(prompt_part, response_part):
                    continue
            prompt, response = cleaned.split(separator, 1)
            prompt = strip_wrapping_quotes(prompt)
            response = strip_wrapping_quotes(response)
            if prompt and response:
                return prompt, response
        return None

    def _dialogue_line_looks_like_chain(self, text: str) -> bool:
        cleaned = normalize_text(text)
        if not cleaned:
            return False
        if cleaned.startswith(("-", "—", "–")):
            return True
        return any(separator in cleaned for separator in DIALOGUE_PAIR_SEPARATORS)

    def _dialogue_utterances_from_line(self, text: str) -> list[str]:
        cleaned = normalize_text(text)
        if not cleaned:
            return []
        for separator in ("=>", "->"):
            if separator in cleaned:
                prompt, response = cleaned.split(separator, 1)
                return [item for item in (strip_dialogue_markers(prompt), strip_dialogue_markers(response)) if item]
        for separator in (" -- ", " - ", " — ", " – "):
            if separator in cleaned:
                parts = [strip_dialogue_markers(part) for part in cleaned.split(separator)]
                return [part for part in parts if part]
        stripped = strip_dialogue_markers(cleaned)
        return [stripped] if stripped else []

    def _apply_training_fragments(
        self,
        fragments: list[dict[str, Any]],
        report: dict[str, Any],
        run: dict[str, Any],
        *,
        session_id: str,
        subgraph_node_id: str | None = None,
    ) -> None:
        for fragment in fragments:
            if fragment.get("kind") == "dialogue_turns":
                turns = [
                    {"role": normalize_dialogue_role(str(turn.get("role") or "")), "text": normalize_text(str(turn.get("text") or ""))}
                    for turn in (fragment.get("turns") or [])
                ]
                previous: dict[str, str | None] | None = None
                for turn in turns:
                    if not turn["role"] or not turn["text"]:
                        continue
                    if previous and previous.get("role") == "user" and turn["role"] == "assistant":
                        if subgraph_node_id is None:
                            self._train_dialogue_pair(str(previous["text"]), str(turn["text"]), report, run)
                        else:
                            self._train_dialogue_pair(str(previous["text"]), str(turn["text"]), report, run, subgraph_node_id=subgraph_node_id)
                    previous = turn
                continue
            text = normalize_text(str(fragment.get("text") or ""))
            if text:
                if subgraph_node_id is None:
                    self._train_sequence(tokenize_with_surfaces(text), report, run)
                else:
                    self._train_sequence(tokenize_with_surfaces(text), report, run, subgraph_node_id=subgraph_node_id)

    def _train_hierarchy_record(
        self,
        record: dict[str, Any],
        report: dict[str, Any],
        run: dict[str, Any],
        *,
        session_id: str,
    ) -> None:
        hierarchy = self._normalize_hierarchy(record.get("hierarchy"))
        text = normalize_text(str(record.get("text") or record.get("content") or record.get("body") or ""))
        if not hierarchy and not text:
            return
        now = time.time()
        hierarchy_ids: list[str] = []
        for depth in range(1, len(hierarchy) + 1):
            prefix = hierarchy[:depth]
            node_id = self._hierarchy_node_id(prefix)
            parent_id = self._hierarchy_node_id(prefix[:-1]) if depth > 1 else None
            node = self._ensure_hypernode(prefix, parent_id=parent_id, report=report, now=now)
            hierarchy_ids.append(node["id"])
            if parent_id:
                self._attach_child_hypernode(parent_id, node, now=now)
        if text:
            leaf_id = hierarchy_ids[-1] if hierarchy_ids else None
            if leaf_id:
                self._attach_text_to_hypernode(leaf_id, text, report=report, run=run, now=now)
                self._apply_training_fragments(self._training_fragments(text), report, run, session_id=session_id, subgraph_node_id=leaf_id)

    def _ensure_hypernode(
        self,
        hierarchy: list[str],
        *,
        parent_id: str | None,
        report: dict[str, Any],
        now: float,
    ) -> dict[str, Any]:
        hierarchy = self._normalize_hierarchy(hierarchy)
        node_id = self._hierarchy_node_id(hierarchy)
        if node_id == HIERARCHY_NODE_PREFIX:
            raise ValueError("hierarchy is required")
        store = self._hypernode_store()
        record = store.get(node_id)
        label = hierarchy[-1] if hierarchy else node_id.removeprefix(HIERARCHY_NODE_PREFIX)
        if record is None:
            record = {
                "id": node_id,
                "type": "hypernode",
                "label": label,
                "hierarchy": hierarchy,
                "parent": parent_id,
                "depth": len(hierarchy),
                "count": 0,
                "vector": zero_vector(),
                "created_at": now,
                "updated_at": now,
                "subgraph": {"tokens": {}, "edges": {}},
            }
            store[node_id] = record
            report["new_hypernodes"] = int(report.get("new_hypernodes", 0)) + 1
        else:
            report["updated_hypernodes"] = int(report.get("updated_hypernodes", 0)) + 1
            record["label"] = label
            record["hierarchy"] = hierarchy
            record["parent"] = parent_id
            record["depth"] = len(hierarchy)
            if not isinstance(record.get("subgraph"), dict) or not record.get("subgraph", {}).get("edges"):
                record["subgraph"] = {"tokens": {}, "edges": {}}
        record["count"] = int(record.get("count", 0)) + 1
        record["updated_at"] = now
        record["vector"] = self._blend_vectors(record.get("vector"), self._embedding_vector(" / ".join(hierarchy)))
        return record

    def _attach_text_to_hypernode(
        self,
        node_id: str,
        text: str,
        *,
        report: dict[str, Any],
        run: dict[str, Any],
        now: float,
    ) -> None:
        store = self._hypernode_store()
        record = store.get(node_id)
        if record is None:
            return
        record["vector"] = self._blend_vectors(record.get("vector"), self._embedding_vector(text))
        record["count"] = int(record.get("count", 0)) + 1
        record["updated_at"] = now
        run["hypernodes"] = int(run.get("hypernodes", 0)) + 1
        report["new_tokens"] = int(report.get("new_tokens", 0)) + 0

    def _attach_child_hypernode(self, parent_id: str, child_record: dict[str, Any], *, now: float) -> None:
        parent = self._hypernode_store().get(parent_id)
        if not parent:
            return
        child_id = str(child_record.get("id") or "")
        if not child_id:
            return
        child_node = {
            "id": child_id,
            "type": "hypernode",
            "label": normalize_text(str(child_record.get("label") or child_id.removeprefix(HIERARCHY_NODE_PREFIX))),
            "hierarchy": list(child_record.get("hierarchy") or []),
            "parent": parent_id,
            "depth": int(child_record.get("depth", 0) or 0),
            "count": int(child_record.get("count", 0) or 0),
            "shape": "rect",
        }
        parent_node = {
            "id": parent_id,
            "type": "hypernode",
            "label": normalize_text(str(parent.get("label") or parent_id.removeprefix(HIERARCHY_NODE_PREFIX))),
            "hierarchy": list(parent.get("hierarchy") or []),
            "parent": parent.get("parent"),
            "depth": int(parent.get("depth", 0) or 0),
            "count": int(parent.get("count", 0) or 0),
            "shape": "rect",
        }
        subgraph = self._normalize_subgraph_payload(parent.get("subgraph"))
        edge_key = self._edge_key(parent_id, "hierarchical_edge", child_id)
        existing = subgraph["edges"].get(edge_key)
        payload = {"weight": 1.0, "pheromone": 1.0}
        if existing is None:
            subgraph["edges"][edge_key] = payload
        else:
            existing["weight"] = float(existing.get("weight", 0.0)) + 1.0
            existing["pheromone"] = max(float(existing.get("pheromone", 0.0)), 1.0)
        parent["subgraph"] = subgraph
        parent["updated_at"] = now

    def _merge_subgraphs(self, left: Any, right: Any) -> dict[str, Any]:
        left_graph = self._normalize_subgraph_payload(left)
        right_graph = self._normalize_subgraph_payload(right)
        tokens = dict(left_graph.get("tokens") or {})
        for token_id_key, token_record in (right_graph.get("tokens") or {}).items():
            tokens[token_id_key] = {**tokens.get(token_id_key, {}), **token_record}
        edges = dict(left_graph.get("edges") or {})
        for edge_id, edge_record in (right_graph.get("edges") or {}).items():
            existing = edges.get(edge_id)
            if existing is None:
                edges[edge_id] = dict(edge_record)
            else:
                existing["weight"] = float(existing.get("weight", 0.0)) + float(edge_record.get("weight", 0.0))
                existing["pheromone"] = max(float(existing.get("pheromone", 0.0)), float(edge_record.get("pheromone", 0.0)))
        return {"tokens": tokens, "edges": edges}

    def _blend_vectors(self, left: Any, right: Any, *, ratio: float = 0.5) -> list[float]:
        left_array = np.asarray(normalize_vector(left), dtype=np.float32)
        right_array = np.asarray(normalize_vector(right), dtype=np.float32)
        if not np.any(left_array):
            return normalize_vector(right_array.tolist())
        if not np.any(right_array):
            return normalize_vector(left_array.tolist())
        mixed = (left_array * (1.0 - ratio)) + (right_array * ratio)
        return normalize_vector(mixed.tolist())

    def _build_hypernode_subgraph(self, node_id: str, *, label: str, text: str) -> dict[str, Any]:
        tokens = [item["token"] for item in tokenize_with_surfaces(text) if canonical_token(item["token"])]
        token_records: dict[str, dict[str, Any]] = {}
        for token in dict.fromkeys(tokens[:48]):
            token_records[token_id(token)] = {
                "id": token_id(token),
                "type": "token",
                "token": token,
                "label": token,
            }
        edges: dict[str, dict[str, float]] = {}
        for token in tokens[:48]:
            edge_key = self._edge_key(node_id, "hierarchical_edge", token_id(token))
            edges[edge_key] = {"weight": 1.0, "pheromone": 1.0}
        for left, right in zip(tokens, tokens[1:]):
            edge_key = self._edge_key(token_id(left), "next", token_id(right))
            existing = edges.get(edge_key)
            if existing is None:
                edges[edge_key] = {"weight": 1.0, "pheromone": 1.0}
            else:
                existing["weight"] = float(existing.get("weight", 0.0)) + 1.0
                existing["pheromone"] = max(float(existing.get("pheromone", 0.0)), 1.0)
        return {
            "tokens": token_records,
            "edges": edges,
        }

    def _train_dialogue_pair(self, prompt: str, response: str, report: dict[str, Any], run: dict[str, Any], subgraph_node_id: str | None = None, weight: int = 1) -> None:
        prompt_tokens = tokenize_with_surfaces(prompt)
        response_tokens = tokenize_with_surfaces(response)
        while response_tokens and response_tokens[-1]["token"] in PUNCT_TOKENS:
            response_tokens.pop()
        prompt_tail = next((item["token"] for item in reversed(prompt_tokens) if item["token"] not in PUNCT_TOKENS and not is_role_token(item["token"])), "")
        response_start = next((item["token"] for item in response_tokens if item["token"] not in PUNCT_TOKENS and not is_role_token(item["token"])), "")
        chain = [{"surface": ROLE_USER_TOKEN, "token": ROLE_USER_TOKEN}]
        chain.extend(prompt_tokens)
        chain.append({"surface": ROLE_ASSISTANT_TOKEN, "token": ROLE_ASSISTANT_TOKEN})
        chain.extend(response_tokens)
        chain.append({"surface": ".", "token": "."})
        self._train_sequence(chain, report, run, subgraph_node_id=subgraph_node_id, weight=max(int(weight or 1), 1))
        if prompt_tail and response_start:
            self._add_transition_memory(token_id(prompt_tail), token_id(ROLE_ASSISTANT_TOKEN), token_id(response_start), weight=max(int(weight or 1), 1))
        report["source_pairs"] = int(report.get("source_pairs", 0)) + max(int(weight or 1), 1)

    def _train_sequence(self, tokens: list[dict[str, str]], report: dict[str, Any], run: dict[str, Any], subgraph_node_id: str | None = None, weight: int = 1) -> None:
        chain = [item for item in tokens if canonical_token(item.get("token"))]
        if len(chain) < 2:
            return
        now = time.time()
        cleaned_chain = [canonical_token(item["token"]) for item in chain]
        token_counts = Counter(cleaned_chain)
        unique_tokens = list(dict.fromkeys(cleaned_chain))
        new_tokens = [token for token in unique_tokens if token not in self.checkpoint.tokens]
        new_vectors = self._embedding_many(new_tokens) if new_tokens else []
        vector_map = {token: vector for token, vector in zip(new_tokens, new_vectors)}
        for token, count in token_counts.items():
            self._ensure_token(token, report, now, subgraph_node_id=subgraph_node_id, count=int(count) * max(int(weight or 1), 1), vector=vector_map.get(token))
        edge_counts = Counter((left, right) for left, right in zip(cleaned_chain, cleaned_chain[1:]))
        for (source_token, target_token), count in edge_counts.items():
            edge_weight = (5.0 if source_token == ROLE_ASSISTANT_TOKEN else 1.0) * float(count) * max(float(weight or 1), 1.0)
            source = token_id(source_token)
            target = token_id(target_token)
            self._add_edge(source, target, edge_weight, report, now, subgraph_node_id=subgraph_node_id)
        transition_counts = Counter((previous, current, candidate) for previous, current, candidate in zip(cleaned_chain, cleaned_chain[1:], cleaned_chain[2:]))
        for previous, current, candidate in transition_counts:
            self._add_transition_memory(
                token_id(previous),
                token_id(current),
                token_id(candidate),
                weight=int(transition_counts[(previous, current, candidate)]) * max(int(weight or 1), 1),
            )
        report["source_sequences"] = int(report.get("source_sequences", 0)) + max(int(weight or 1), 1)
        run["sequences"] = int(run.get("sequences", 0)) + max(int(weight or 1), 1)

    def _ensure_token(self, token: str, report: dict[str, Any], now: float, subgraph_node_id: str | None = None, *, count: int = 1, vector: list[float] | None = None) -> dict[str, Any]:
        cleaned = canonical_token(token)
        record = self.checkpoint.tokens.get(cleaned)
        if record is None:
            record = {
                "id": f"token:{cleaned}",
                "type": "token",
                "token": cleaned,
                "label": cleaned,
                "count": 0,
                "vector": vector if vector is not None else self._token_embedding(cleaned),
                "created_at": now,
                "updated_at": now,
            }
            self.checkpoint.tokens[cleaned] = record
            report["new_tokens"] = int(report.get("new_tokens", 0)) + 1
        else:
            report["updated_tokens"] = int(report.get("updated_tokens", 0)) + 1
            record["label"] = canonical_token(record.get("label") or cleaned) or cleaned
            if vector is not None and not record.get("vector"):
                record["vector"] = vector
        record["count"] = int(record.get("count", 0)) + max(int(count or 1), 1)
        record["updated_at"] = now
        subgraph = self._subgraph_store(subgraph_node_id or ROOT_HYPERNODE_ID)
        local_record = subgraph["tokens"].setdefault(
            token_id(cleaned),
            {
                "id": token_id(cleaned),
                "type": "token",
                "token": cleaned,
                "label": cleaned,
                "count": 0,
            },
        )
        local_record["label"] = str(local_record.get("label") or cleaned)
        local_record["count"] = int(local_record.get("count", 0) or 0) + max(int(count or 1), 1)
        return record

    def _jsonl_record_to_text(self, record: Any) -> str:
        if record is None:
            return ""
        if isinstance(record, str):
            return normalize_text(record)
        if isinstance(record, list):
            return "\n".join(item for item in (self._jsonl_record_to_text(value) for value in record) if item)
        if not isinstance(record, dict):
            return normalize_text(str(record))
        hierarchy_text = normalize_text(str(record.get("text") or record.get("content") or record.get("body") or ""))
        if hierarchy_text and "hierarchy" in record:
            return hierarchy_text
        pair_candidates = (
            ("prompt", "response"),
            ("question", "answer"),
            ("input", "output"),
            ("instruction", "response"),
            ("source", "target"),
            ("query", "answer"),
        )
        for left_key, right_key in pair_candidates:
            left = normalize_text(str(record.get(left_key) or ""))
            right = normalize_text(str(record.get(right_key) or ""))
            if left and right:
                return f"{left} => {right}"
        direct_text = normalize_text(str(record.get("text") or record.get("content") or record.get("body") or ""))
        if direct_text:
            return direct_text
        messages = record.get("messages")
        if isinstance(messages, list):
            parts: list[str] = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = normalize_text(str(message.get("role") or ""))
                content = normalize_text(str(message.get("content") or message.get("text") or ""))
                if content:
                    parts.append(f"{role}: {content}" if role else content)
            if parts:
                return "\n".join(parts)
        values = [normalize_text(str(value)) for value in record.values() if isinstance(value, (str, int, float, bool))]
        return "\n".join(value for value in values if value).strip()

    def _build_dense_backpack(self, query_text: str, *, session_id: str, focus_vector: list[float] | None = None) -> dict[str, Any]:
        query_vector = self._embedding_vector(query_text)
        combined = np.asarray(query_vector, dtype=np.float32)
        focus_array = np.asarray(normalize_vector(focus_vector), dtype=np.float32)
        if float(np.linalg.norm(focus_array)) > 0.0:
            combined = (combined * 0.55) + (focus_array * 0.45)
        history_vectors: list[dict[str, Any]] = []
        turns = list(self.checkpoint.sessions.get(session_id) or [])
        recent = [turn for turn in turns[-3:] if normalize_text(str(turn.get("text") or ""))]
        for index, turn in enumerate(reversed(recent), start=1):
            weight = float(self.config.session_context_decay) ** index
            vector = self._embedding_vector(str(turn.get("text") or ""))
            combined = combined + np.asarray(vector, dtype=np.float32) * weight
            history_vectors.append(
                {
                    "role": str(turn.get("role") or "message"),
                    "text": normalize_text(str(turn.get("text") or "")),
                    "weight": round(weight, 6),
                    "vector": vector,
                }
            )
        return {
            "query_vector": normalize_vector(combined.tolist()),
            "raw_query_vector": query_vector,
            "focus_vector": normalize_vector(focus_array.tolist()),
            "history_vectors": history_vectors,
        }

    def _score_tokens(self, query_tokens: list[str], query_vector: list[float]) -> list[dict[str, Any]]:
        query_set = {canonical_token(token) for token in query_tokens}
        if not self.checkpoint.tokens:
            return [
                {
                    "node_id": token_id(token),
                    "type": "token",
                    "token": token,
                    "label": token,
                    "count": 0,
                    "score": 0.0,
                    "known": False,
                }
                for token in query_set
            ]
        labels = list(self.checkpoint.tokens.keys())
        vectors = np.asarray([normalize_vector(self.checkpoint.tokens[label].get("vector")) for label in labels], dtype=np.float32)
        similarities = cosine_similarity_matrix(vectors, np.asarray(normalize_vector(query_vector), dtype=np.float32))
        max_count = max((int(record.get("count", 0)) for record in self.checkpoint.tokens.values()), default=1)
        scored: list[dict[str, Any]] = []
        for label, similarity in zip(labels, similarities):
            record = self.checkpoint.tokens[label]
            if label in ROLE_TOKENS or label in PUNCT_TOKENS:
                continue
            score = float(similarity)
            if label in query_set:
                score += 0.45
            score += 0.1 * math.log1p(max(int(record.get("count", 0)), 0)) / math.log1p(max_count)
            scored.append(
                {
                    "node_id": f"token:{label}",
                    "type": "token",
                    "token": label,
                    "label": str(record.get("label") or label),
                    "count": int(record.get("count", 0)),
                    "score": round(score, 6),
                    "known": True,
                }
            )
        for token in query_set:
            if token not in self.checkpoint.tokens:
                scored.append(
                    {
                        "node_id": token_id(token),
                        "type": "token",
                        "token": token,
                        "label": token,
                        "count": 0,
                        "score": 0.0,
                        "known": False,
                    }
                )
        scored.sort(key=lambda item: (float(item["score"]), int(item.get("count", 0))), reverse=True)
        return scored

    def _semantic_milestone_count(self, query_tokens: list[str], query_vector: list[float]) -> int:
        content_count = len([token for token in (canonical_token(item) for item in query_tokens) if token and token not in ROLE_TOKENS and token not in PUNCT_TOKENS])
        length_based = int(clamp(math.ceil(content_count / 3.0) if content_count else 2, 2, 6))
        query_array = np.asarray(normalize_vector(query_vector), dtype=np.float32)
        if float(np.linalg.norm(query_array)) <= 0.0:
            return length_based
        labels = [label for label in self.checkpoint.tokens if label not in ROLE_TOKENS and label not in PUNCT_TOKENS]
        if not labels:
            return length_based
        vectors = np.asarray([normalize_vector(self.checkpoint.tokens[label].get("vector")) for label in labels], dtype=np.float32)
        similarities = cosine_similarity_matrix(vectors, query_array)
        if similarities.size == 0:
            return length_based
        top_similarity = float(np.max(similarities))
        if top_similarity <= 0.0:
            return length_based
        relevance_floor = max(0.0, top_similarity * 0.65)
        relevant_indexes = np.flatnonzero(similarities >= relevance_floor)
        if relevant_indexes.size == 0:
            return length_based
        ordered_indexes = relevant_indexes[np.argsort(similarities[relevant_indexes])[::-1]]
        ordered_indexes = ordered_indexes[: min(len(ordered_indexes), 36)]
        candidate_vectors = vectors[ordered_indexes]
        norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
        normalized_vectors = np.divide(candidate_vectors, norms, out=np.zeros_like(candidate_vectors), where=norms > 0.0)
        pairwise = np.matmul(normalized_vectors, normalized_vectors.T)
        centers: list[int] = []
        for index in range(len(ordered_indexes)):
            if not centers:
                centers.append(index)
                continue
            if float(np.max(pairwise[index, centers])) < 0.55:
                centers.append(index)
            if len(centers) >= 6:
                break
        return int(clamp(max(length_based, len(centers)), 2, 6))

    def _generate_semantic_plan(
        self,
        query_vector: list[float],
        num_milestones: int | None = None,
        query_tokens: list[str] | None = None,
    ) -> list[str]:
        requested = self._semantic_milestone_count(query_tokens or [], query_vector) if num_milestones is None else int(num_milestones or 0)
        requested = int(clamp(requested, 0, 6))
        if requested <= 0:
            return []
        query_array = np.asarray(normalize_vector(query_vector), dtype=np.float32)
        if float(np.linalg.norm(query_array)) <= 0.0:
            return []
        labels = [label for label in self.checkpoint.tokens if label not in ROLE_TOKENS and label not in PUNCT_TOKENS]
        if not labels:
            return []
        vectors = np.asarray([normalize_vector(self.checkpoint.tokens[label].get("vector")) for label in labels], dtype=np.float32)
        similarities = cosine_similarity_matrix(vectors, query_array)
        candidate_indexes = np.flatnonzero(similarities > 0.0)
        if candidate_indexes.size == 0:
            return []
        ordered_indexes = candidate_indexes[np.argsort(similarities[candidate_indexes])[::-1]]
        ordered_indexes = ordered_indexes[: min(len(ordered_indexes), max(24, requested * 12))]
        candidate_labels = [labels[index] for index in ordered_indexes]
        candidate_vectors = vectors[ordered_indexes]
        candidate_similarities = similarities[ordered_indexes]
        norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
        normalized_vectors = np.divide(candidate_vectors, norms, out=np.zeros_like(candidate_vectors), where=norms > 0.0)
        pairwise = np.matmul(normalized_vectors, normalized_vectors.T)
        selected: list[int] = [int(np.argmax(candidate_similarities))]
        while len(selected) < min(requested, len(candidate_labels)):
            redundancy = np.max(pairwise[:, selected], axis=1)
            mmr_scores = candidate_similarities - (0.65 * redundancy)
            mmr_scores[selected] = -np.inf
            diverse_mask = redundancy < 0.92
            if np.any(np.isfinite(mmr_scores) & diverse_mask):
                mmr_scores = np.where(diverse_mask, mmr_scores, -np.inf)
            next_index = int(np.argmax(mmr_scores))
            if not np.isfinite(mmr_scores[next_index]):
                break
            selected.append(next_index)
        return [f"token:{candidate_labels[index]}" for index in selected]

    def _synthesize_hypernode_response(self, node_id: str | None, *, query_tokens: list[str]) -> tuple[str, str] | None:
        if not node_id or not str(node_id).startswith(HIERARCHY_NODE_PREFIX):
            return None
        record = self._hypernode_store().get(str(node_id))
        if not record:
            return None
        graph = self._subgraph_to_graph_payload(str(node_id), record.get("subgraph"), label=str(record.get("label") or node_id))
        child_nodes = [
            node
            for node in graph.get("nodes", [])
            if str(node.get("id") or "").startswith(HIERARCHY_NODE_PREFIX) and str(node.get("id")) != str(node_id)
        ]
        if child_nodes:
            child_nodes.sort(key=lambda node: (int(node.get("depth", 0) or 0), str(node.get("label") or node.get("id") or "")))
            labels = [normalize_text(str(node.get("label") or node.get("id") or "")) for node in child_nodes[:6]]
            labels = [label for label in labels if label]
            if labels:
                response = "\n".join(f"{index}. {label}" for index, label in enumerate(labels, start=1))
                return response, "hypernode_plan"

        token_edges = [
            edge
            for edge in graph.get("edges", [])
            if str(edge.get("relation") or edge.get("type") or "") == "transition_edge"
            and str(edge.get("source") or "").startswith("token:")
            and str(edge.get("target") or "").startswith("token:")
        ]
        if not token_edges:
            return None
        query_set = {canonical_token(token) for token in query_tokens}
        outgoing: dict[str, list[dict[str, Any]]] = {}
        incoming: set[str] = set()
        for edge in token_edges:
            outgoing.setdefault(str(edge["source"]), []).append(edge)
            incoming.add(str(edge["target"]))
        for edges in outgoing.values():
            edges.sort(key=lambda edge: float(edge.get("weight", 0.0)), reverse=True)
        start_candidates = [source for source in outgoing if source not in incoming]
        if not start_candidates:
            start_candidates = list(outgoing)
        start_candidates.sort(
            key=lambda source: (
                canonical_token(source.removeprefix("token:")) in query_set,
                -len(outgoing.get(source, [])),
                source,
            )
        )
        current = start_candidates[0]
        generated = [current]
        visited_edges: set[str] = set()
        for _ in range(32):
            edges = [edge for edge in outgoing.get(current, []) if str(edge.get("id") or "") not in visited_edges]
            if not edges:
                break
            edge = edges[0]
            visited_edges.add(str(edge.get("id") or ""))
            current = str(edge.get("target") or "")
            generated.append(current)
            if self._is_stop_token(current):
                break
        response = self._render_generated_tokens(generated)
        return (self._truncate_response(response), "hypernode_tokens") if response else None

    def _synthesize_response(
        self,
        query_tokens: list[str],
        backpack: dict[str, Any],
        *,
        session_id: str,
        temperature: float = 0.75,
        max_length: int = 40,
        milestones: list[str] | None = None,
        prompt_tail_token: str | None = None,
    ) -> tuple[str, str]:
        current_token = token_id(ROLE_ASSISTANT_TOKEN)
        active_graph_id, active_record = self._active_graph_record(session_id)
        active_graph = self._subgraph_to_graph_payload(active_graph_id, active_record.get("subgraph"), label=str(active_record.get("label") or active_graph_id))
        active_edges = list(active_graph.get("edges", []))
        prompt_tail = next((canonical_token(token) for token in reversed(query_tokens) if canonical_token(token)), "")
        previous_token: str | None = token_id(prompt_tail) if prompt_tail else None
        raw_tail = canonical_token(prompt_tail_token or "")
        if raw_tail and raw_tail not in ROLE_TOKENS:
            raw_tail_id = token_id(raw_tail)
            if raw_tail_id != previous_token and (previous_token is None or not self._transition_targets(previous_token, current_token)):
                previous_token = raw_tail_id
        generated: list[str] = []
        query_token_set = {canonical_token(token) for token in query_tokens}
        out_degrees = self._source_out_degrees(edges=active_edges)
        milestone_ids = [
            token_id(milestone)
            for milestone in (milestones or [])
            if token_id(milestone).removeprefix("token:") in self.checkpoint.tokens
        ]
        query_array = np.asarray(normalize_vector(backpack["focus_vector"] if self._vector_norm(backpack.get("focus_vector", [])) > 0.0 else backpack["query_vector"]), dtype=np.float32)
        for step in range(max_length):
            non_terminal_count = len([token for token in (item.removeprefix("token:") for item in generated) if token and token not in TERMINAL_TOKENS])
            close_ready = non_terminal_count >= 4
            final_phase = False
            next_milestone: str | None = None
            focal_vector = backpack["query_vector"]
            if milestone_ids:
                phase_index = min((step * len(milestone_ids)) // max(max_length, 1), len(milestone_ids) - 1)
                milestone_vector = np.asarray(normalize_vector(self._token_vector(milestone_ids[phase_index].removeprefix("token:"))), dtype=np.float32)
                focal_vector = normalize_vector(((query_array * 0.5) + (milestone_vector * 1.5)).tolist())
                final_phase = phase_index == len(milestone_ids) - 1
                if not final_phase:
                    next_milestone = milestone_ids[phase_index + 1]
            candidates = self._generation_candidates(
                current_token,
                focal_vector,
                active_edges=active_edges,
                query_tokens=query_token_set,
                generated_tokens=generated,
                previous_token=previous_token,
                next_milestone=next_milestone,
                allow_terminals=(final_phase or close_ready) if milestone_ids else None,
                terminal_gravity=5.0 if final_phase else 2.0 if close_ready else None,
                out_degrees=out_degrees,
            )
            if not candidates:
                bridge_token = self._terminal_bridge_token(current_token, edges=active_edges)
                if bridge_token:
                    previous_token = current_token
                    current_token = bridge_token
                    continue
                break
            next_token = self._weighted_random_choice(candidates, temperature=temperature)
            if not next_token:
                break
            generated.append(next_token)
            next_label = canonical_token(next_token.removeprefix("token:"))
            if self._is_stop_token(next_token) and (final_phase or (next_label == "." and close_ready) or not self._terminal_has_continuation(next_token, edges=active_edges)):
                break
            previous_token = current_token
            current_token = next_token
        response = self._render_generated_tokens(generated)
        if response:
            return self._truncate_response(response), "graph"
        return "Нужен обучающий текст.", "fallback"

    def _generation_candidates(
        self,
        current_token: str,
        query_vector: list[float],
        *,
        active_edges: list[dict[str, Any]] | None = None,
        query_tokens: set[str] | list[str] | None = None,
        generated_tokens: list[str] | None = None,
        previous_token: str | None = None,
        next_milestone: str | None = None,
        allow_terminals: bool | None = None,
        terminal_gravity: float | None = None,
        out_degrees: dict[str, int] | None = None,
    ) -> dict[str, float]:
        edge_pool = active_edges if active_edges is not None else self._active_graph_edges()
        edges = [edge for edge in self._outgoing_edges(current_token, edges=edge_pool) if str(edge.get("target") or "").startswith("token:")]
        if not edges:
            return {}
        candidate_ids = [str(edge["target"]) for edge in edges]
        labels = [candidate.removeprefix("token:") for candidate in candidate_ids]
        valid_indexes = [index for index, label in enumerate(labels) if label and not is_role_token(label)]
        if not valid_indexes:
            return {}
        candidate_ids = [candidate_ids[index] for index in valid_indexes]
        labels = [labels[index] for index in valid_indexes]
        candidate_ids_array = np.asarray(candidate_ids, dtype=object)
        labels_array = np.asarray(labels, dtype=object)
        edge_weights = np.asarray([max(float(edge.get("weight", 0.0)), 0.0) for edge in edges], dtype=np.float32)
        base_weights = edge_weights[np.asarray(valid_indexes, dtype=np.intp)]
        vectors = np.asarray([normalize_vector(self._token_vector(label)) for label in labels], dtype=np.float32)
        similarities = cosine_similarity_matrix(vectors, np.asarray(normalize_vector(query_vector), dtype=np.float32))
        reinforcement = np.ones(len(candidate_ids), dtype=np.float32)
        if next_milestone:
            milestone_id = token_id(next_milestone)
            edge_sources = np.asarray([str(edge.get("source") or "") for edge in edge_pool], dtype=object)
            edge_targets = np.asarray([str(edge.get("target") or "") for edge in edge_pool], dtype=object)
            milestone_sources = edge_sources[edge_targets == milestone_id]
            milestone_mask = (candidate_ids_array == milestone_id) | np.isin(candidate_ids_array, milestone_sources)
            reinforcement = reinforcement + milestone_mask.astype(np.float32) * 0.3
        repeat_labels = np.asarray([label.casefold() for label in labels], dtype=object)
        generated = generated_tokens or []
        generated_labels = np.asarray([canonical_token(str(token).removeprefix("token:")) for token in generated], dtype=object)
        repeat_mask = np.isin(repeat_labels, generated_labels)
        reinforcement = np.where(repeat_mask, 0.01, reinforcement)
        scores = base_weights * (1.0 + similarities) * reinforcement
        transition_targets = self._transition_targets(previous_token, current_token) if previous_token else {}
        transition_boosts = np.asarray([3.0 if candidate_id in transition_targets else 1.0 for candidate_id in candidate_ids], dtype=np.float32)
        scores = scores * transition_boosts
        degrees = out_degrees or {}
        if degrees:
            hub_divisors = np.asarray(
                [math.log(float(degrees.get(candidate_id, 0))) if int(degrees.get(candidate_id, 0)) > 15 else 1.0 for candidate_id in candidate_ids],
                dtype=np.float32,
            )
            scores = scores / np.maximum(hub_divisors, 1.0)
        query_set = {canonical_token(token) for token in (query_tokens or [])}
        generated_label_set = {label.casefold() for label in generated_labels.tolist() if label}
        non_terminal_generated = [label for label in generated_labels.tolist() if label and label not in TERMINAL_TOKENS]
        terminal_multiplier = float(terminal_gravity) if terminal_gravity is not None else 5.0
        query_mask = np.isin(labels_array, np.asarray(list(query_set), dtype=object)) if query_set else np.zeros(len(labels), dtype=bool)
        scores = np.where((len(non_terminal_generated) < 6) & query_mask, 0.0, scores)
        terminal_mask = np.isin(labels_array, np.asarray(list(TERMINAL_TOKENS), dtype=object))
        continuation_mask = np.asarray([self._terminal_has_continuation(candidate_ids[index], edges=edge_pool) for index in range(len(candidate_ids))], dtype=bool)
        if allow_terminals is False:
            terminal_block = terminal_mask & ((transition_boosts <= 1.0) | ~continuation_mask)
            scores = np.where(terminal_block, 0.0, scores)
        elif terminal_gravity is not None:
            scores = np.where(terminal_mask, scores * terminal_multiplier, scores)
        else:
            terminal_soft = terminal_mask & (transition_boosts > 1.0) & continuation_mask
            scores = np.where(terminal_soft, scores, scores)
            scores = np.where(terminal_mask & (len(non_terminal_generated) < 4), 0.0, scores)
            scores = np.where(terminal_mask & (len(non_terminal_generated) >= 6), scores * 5.0, scores)
            scores = np.where(terminal_mask & (len(non_terminal_generated) >= 4) & (len(non_terminal_generated) < 6), scores * 2.0, scores)
        return {candidate_id: float(score) for candidate_id, score in zip(candidate_ids, scores) if float(score) > 0.0}

    def _weighted_random_choice(self, candidates: dict[str, float], temperature: float = 0.7) -> str | None:
        filtered = [(token, max(float(score), 0.0)) for token, score in candidates.items() if token and float(score) > 0.0]
        if not filtered:
            return None
        if temperature <= 0.0:
            return max(filtered, key=lambda item: item[1])[0]
        scores = np.asarray([score for _, score in filtered], dtype=np.float64)
        scaled = scores / max(float(temperature), 1e-6)
        scaled = scaled - float(np.max(scaled))
        weights = np.exp(scaled)
        total = float(np.sum(weights))
        if total <= 0.0:
            return max(filtered, key=lambda item: item[1])[0]
        threshold = random.random() * total
        cumulative = 0.0
        for (token, _), weight in zip(filtered, weights):
            cumulative += float(weight)
            if cumulative >= threshold:
                return token
        return filtered[-1][0]

    def _build_graph(
        self,
        *,
        query: str | None,
        limit: int,
        seed_ids: list[str] | None,
        highlight_result_id: str | None,
    ) -> dict[str, Any]:
        limit = max(int(limit or self.config.graph_limit), 1)
        seed_ids = [node_id for node_id in (seed_ids or []) if node_id in self._all_nodes()]
        node_scores: dict[str, float] = {}
        if query:
            query_vector = self._embedding_vector(query)
            for item in self._score_tokens(tokenize(query), query_vector):
                node_scores[item["node_id"]] = float(item["score"])
        if not seed_ids:
            seed_ids = [
                f"token:{token}"
                for token, _ in sorted(
                    self.checkpoint.tokens.items(),
                    key=lambda item: int(item[1].get("count", 0)),
                    reverse=True,
                )
                if token not in ROLE_TOKENS and token not in PUNCT_TOKENS
            ][: max(4, min(limit, 24))]
        selected = self._expand_graph(seed_ids, node_scores=node_scores, limit=limit, highlight_result_id=highlight_result_id)
        positions = self._layout_graph(selected["nodes"], selected["edges"], seed_ids=seed_ids)
        seed_set = set(seed_ids)
        nodes = []
        for node in selected["nodes"]:
            nodes.append({**node, **positions.get(node["id"], {"x": 500.0, "y": 350.0}), "active": node["id"] in seed_set})
        edges = []
        for edge in selected["edges"]:
            source = positions.get(edge["source"], {"x": 500.0, "y": 350.0})
            target = positions.get(edge["target"], {"x": 500.0, "y": 350.0})
            edges.append(
                {
                    **edge,
                    "x1": source["x"],
                    "y1": source["y"],
                    "x2": target["x"],
                    "y2": target["y"],
                    "active": edge["source"] in seed_set or edge["target"] in seed_set,
                }
            )
        stats = {
            "nodes": len(nodes),
            "edges": len(edges),
            "tokens": len(nodes),
            "seeds": len(seed_ids),
        }
        if highlight_result_id and highlight_result_id in self.checkpoint.results:
            stats["result_id"] = highlight_result_id
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": stats,
            "seed_ids": seed_ids,
            "node_ids": [node["id"] for node in nodes],
            "edge_ids": [edge["id"] for edge in edges],
            "graph_id": sha_id("graph", f"{normalize_text(query or '')}:{limit}:{','.join(seed_ids)}:{highlight_result_id or ''}"),
        }

    def _build_recursive_backpack_layers(
        self,
        *,
        base_graph: dict[str, Any],
        query: str | None,
        limit: int,
        seed_ids: list[str],
        highlight_result_id: str | None,
    ) -> list[dict[str, Any]]:
        limit = max(int(limit or self.config.backpack_limit), 1)
        max_depth = min(4, max(2, int(math.ceil(limit / 12.0))))
        layers: list[dict[str, Any]] = []
        current_graph = base_graph
        current_seed_ids = [node_id for node_id in (seed_ids or []) if node_id in self._all_nodes()]
        seen_focuses = {tuple(current_seed_ids)}
        current_limit = limit
        current_query = normalize_text(query or "")
        layers.append(
            self._decorate_backpack_layer(
                current_graph,
                level=0,
                focus_ids=current_seed_ids,
                focus_query=current_query,
            )
        )
        for level in range(1, max_depth):
            next_seed_ids = self._backpack_layer_seed_ids(current_graph, current_seed_ids, max_seeds=3)
            if not next_seed_ids:
                break
            focus_key = tuple(next_seed_ids)
            if focus_key in seen_focuses:
                break
            seen_focuses.add(focus_key)
            current_query = self._backpack_focus_query(next_seed_ids)
            current_limit = max(12, int(round(current_limit * 0.72)))
            current_graph = self._build_graph(
                query=current_query or None,
                limit=current_limit,
                seed_ids=next_seed_ids,
                highlight_result_id=highlight_result_id,
            )
            layers.append(
                self._decorate_backpack_layer(
                    current_graph,
                    level=level,
                    focus_ids=next_seed_ids,
                    focus_query=current_query,
                )
            )
            current_seed_ids = next_seed_ids
        return layers

    def _decorate_backpack_layer(
        self,
        graph: dict[str, Any],
        *,
        level: int,
        focus_ids: list[str],
        focus_query: str,
    ) -> dict[str, Any]:
        payload = {key: value for key, value in graph.items() if key != "layers"}
        payload["level"] = int(level)
        payload["focus_ids"] = [node_id for node_id in focus_ids if node_id]
        payload["focus_query"] = focus_query
        payload["scale"] = round(max(0.72, 1.0 - float(level) * 0.08), 4)
        payload["opacity"] = round(max(0.34, 1.0 - float(level) * 0.17), 4)
        payload["inset"] = int(level * 18)
        return payload

    def _backpack_layer_seed_ids(self, graph: dict[str, Any], current_seed_ids: list[str], *, max_seeds: int = 3) -> list[str]:
        current = {node_id for node_id in current_seed_ids if node_id}
        nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict) and str(node.get("id") or "").startswith("token:")]
        candidates = [node for node in nodes if str(node.get("id")) not in current]
        candidates.sort(key=lambda node: (float(node.get("score", 0.0)), int(node.get("count", 0))), reverse=True)
        if not candidates:
            return []
        limit = max(1, min(int(max_seeds or 1), len(candidates)))
        return [str(node.get("id") or "") for node in candidates[:limit] if str(node.get("id") or "")]

    def _backpack_focus_query(self, node_ids: list[str]) -> str:
        labels: list[str] = []
        for node_id in node_ids:
            node = self._node_by_id(node_id)
            if not node:
                continue
            label = normalize_text(str(node.get("label") or node.get("token") or node_id.removeprefix("token:")))
            if label:
                labels.append(label)
        return " ".join(labels[:4]).strip()

    def _expand_graph(
        self,
        seed_ids: list[str],
        *,
        node_scores: dict[str, float],
        limit: int,
        highlight_result_id: str | None,
    ) -> dict[str, Any]:
        node_by_id = self._all_nodes()
        if not node_by_id:
            return {"nodes": [], "edges": []}
        selected: dict[str, dict[str, Any]] = {}
        queue = [node_id for node_id in seed_ids if node_id in node_by_id]
        for node_id in queue:
            selected[node_id] = self._node_payload(node_id, node_scores=node_scores, highlight_result_id=highlight_result_id)
        adjacency = self._visual_adjacency()
        rank_pool = sorted(node_by_id, key=lambda node_id: (float(node_scores.get(node_id, 0.0)), self._node_rank(node_by_id[node_id])), reverse=True)
        rank_index = 0
        while len(selected) < min(limit, len(node_by_id)):
            grew = False
            while queue and len(selected) < min(limit, len(node_by_id)):
                current = queue.pop(0)
                for edge in adjacency.get(current, [])[:10]:
                    neighbor = str(edge["target"] if edge["source"] == current else edge["source"])
                    if neighbor not in node_by_id or neighbor in selected:
                        continue
                    selected[neighbor] = self._node_payload(neighbor, node_scores=node_scores, highlight_result_id=highlight_result_id)
                    queue.append(neighbor)
                    grew = True
                    if len(selected) >= min(limit, len(node_by_id)):
                        break
            if not grew:
                while rank_index < len(rank_pool):
                    candidate = rank_pool[rank_index]
                    rank_index += 1
                    if candidate in selected:
                        continue
                    selected[candidate] = self._node_payload(candidate, node_scores=node_scores, highlight_result_id=highlight_result_id)
                    queue.append(candidate)
                    grew = True
                    break
            if not grew:
                break
        selected_ids = set(selected)
        selected_edges = [edge for edge in self._all_edges() if edge["source"] in selected_ids and edge["target"] in selected_ids]
        selected_edges.sort(key=lambda edge: float(edge.get("weight", 0.0)), reverse=True)
        nodes = sorted(selected.values(), key=lambda node: (float(node.get("score", 0.0)), int(node.get("count", 0))), reverse=True)
        return {"nodes": nodes, "edges": selected_edges[: max(limit * 2, 24)]}

    def _layout_graph(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], *, seed_ids: list[str]) -> dict[str, dict[str, float]]:
        if not nodes:
            return {}
        width = 1000.0
        height = 700.0
        center_x = width / 2.0
        center_y = height / 2.0
        adjacency = {node["id"]: set() for node in nodes}
        for edge in edges:
            adjacency.setdefault(edge["source"], set()).add(edge["target"])
            adjacency.setdefault(edge["target"], set()).add(edge["source"])
        depth: dict[str, int] = {}
        queue = [node_id for node_id in seed_ids if node_id in adjacency]
        for node_id in queue:
            depth[node_id] = 0
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, set()):
                if neighbor in depth:
                    continue
                depth[neighbor] = depth[current] + 1
                queue.append(neighbor)
        remaining = [node["id"] for node in nodes if node["id"] not in depth]
        max_depth = max(depth.values(), default=0)
        for offset, node_id in enumerate(remaining, start=1):
            depth[node_id] = max_depth + 1 + (offset - 1) // max(1, len(nodes) // 4)
        grouped: dict[int, list[dict[str, Any]]] = {}
        for node in nodes:
            grouped.setdefault(depth.get(node["id"], 0), []).append(node)
        positions: dict[str, dict[str, float]] = {}
        for ring, group in sorted(grouped.items(), key=lambda item: item[0]):
            group = sorted(group, key=lambda item: (-float(item.get("score", 0.0)), item.get("label", "")))
            radius = 70.0 if ring == 0 else 110.0 + ring * 105.0
            count = max(len(group), 1)
            for index, node in enumerate(group):
                angle = ((index / count) * math.tau) + self._stable_fraction(node["id"]) * 0.9 + ring * 0.35
                distance = radius + self._stable_fraction(node["id"]) * 24.0
                x = center_x + math.cos(angle) * distance
                y = center_y + math.sin(angle) * distance
                positions[node["id"]] = {
                    "x": round(clamp(x, 40.0, width - 40.0), 2),
                    "y": round(clamp(y, 40.0, height - 40.0), 2),
                }
        return positions

    def _stable_fraction(self, text: str) -> float:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return (int.from_bytes(digest, "big", signed=False) % 1000) / 1000.0

    def _node_payload(self, node_id: str, *, node_scores: dict[str, float], highlight_result_id: str | None) -> dict[str, Any]:
        node = self._node_by_id(node_id)
        if node is None:
            return {"id": node_id, "label": node_id, "type": "token", "score": 0.0, "shape": "circle"}
        score = float(node_scores.get(node_id, 0.0))
        active = False
        if highlight_result_id and highlight_result_id in self.checkpoint.results:
            result = self.checkpoint.results[highlight_result_id]
            active = node_id in set(result.get("backpack", {}).get("node_ids", []))
        count = int(node.get("count", 0))
        return {
            "id": node_id,
            "type": "token",
            "label": node.get("label") or node.get("token") or node_id,
            "token": node.get("token") or node_id.removeprefix("token:"),
            "count": count,
            "score": round(score, 6),
            "active": active,
            "x": 0.0,
            "y": 0.0,
            "shape": "circle",
            "radius": round(clamp(14.0 + math.log1p(max(count, 0)) * 2.8 + score * 12.0, 12.0, 34.0), 2),
            "title": f"{node.get('label') or node_id} | {count}",
        }

    def _all_nodes(self) -> dict[str, dict[str, Any]]:
        nodes = {
            f"token:{token}": {
                "id": f"token:{token}",
                "type": "token",
                "token": token,
                "label": record.get("label") or token,
                "count": int(record.get("count", 0)),
            }
            for token, record in self.checkpoint.tokens.items()
        }
        for node_id, record in self._hypernode_store().items():
            if not isinstance(record, dict):
                continue
            nodes[str(node_id)] = {
                "id": str(node_id),
                "type": "hypernode",
                "label": record.get("label") or str(node_id).removeprefix(HIERARCHY_NODE_PREFIX),
                "count": int(record.get("count", 0)),
            }
        return nodes

    def _all_edges(self) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        for node_id, record in self._hypernode_store().items():
            if not isinstance(record, dict):
                continue
            subgraph = self._normalize_subgraph_payload(record.get("subgraph"))
            edges.extend(self._graph_edges_from_subgraph(node_id, subgraph))
        return edges

    def _source_out_degrees(self, *, edges: list[dict[str, Any]]) -> dict[str, int]:
        degrees: dict[str, int] = {}
        for edge in edges:
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if source.startswith("token:") and target.startswith("token:"):
                degrees[source] = degrees.get(source, 0) + 1
        return degrees

    def _outgoing_edges(self, source: str, *, edges: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        pool = edges if edges is not None else self._all_edges()
        edges = [edge for edge in pool if edge["source"] == source]
        edges.sort(key=lambda edge: float(edge.get("weight", 0.0)), reverse=True)
        return edges

    def _visual_adjacency(self) -> dict[str, list[dict[str, Any]]]:
        adjacency: dict[str, list[dict[str, Any]]] = {}
        for edge in self._all_edges():
            adjacency.setdefault(str(edge["source"]), []).append(edge)
            adjacency.setdefault(str(edge["target"]), []).append(edge)
        for edges in adjacency.values():
            edges.sort(key=lambda edge: float(edge.get("weight", 0.0)), reverse=True)
        return adjacency

    def _node_by_id(self, node_id: str) -> dict[str, Any] | None:
        if not str(node_id).startswith("token:"):
            if str(node_id).startswith(HIERARCHY_NODE_PREFIX):
                record = self._hypernode_store().get(str(node_id))
                if not record:
                    return None
                return {
                    "id": str(node_id),
                    "type": "hypernode",
                    "label": record.get("label") or node_id.removeprefix(HIERARCHY_NODE_PREFIX),
                    "hierarchy": list(record.get("hierarchy") or []),
                    "parent": record.get("parent"),
                    "depth": int(record.get("depth", 0)),
                    "count": int(record.get("count", 0)),
                    "vector_norm": round(self._vector_norm(record.get("vector") or []), 6),
                    "shape": "rect",
                }
            return None
        token = canonical_token(str(node_id).removeprefix("token:"))
        record = self.checkpoint.tokens.get(token)
        if not record:
            return None
        return {
            "id": f"token:{token}",
            "type": "token",
            "token": token,
            "label": record.get("label") or token,
            "count": int(record.get("count", 0)),
            "vector_norm": round(self._vector_norm(record.get("vector") or []), 6),
        }

    def _neighbors(self, node_id: str) -> list[dict[str, Any]]:
        neighbors = []
        if str(node_id).startswith(HIERARCHY_NODE_PREFIX):
            node = self._node_by_id(node_id)
            if node is None:
                return neighbors
            record = self._hypernode_store().get(node_id) or {}
            graph = self._subgraph_to_graph_payload(node_id, self._normalize_subgraph_payload(record.get("subgraph")), label=str(record.get("label") or node_id))
            for edge in graph.get("edges", []):
                if edge["source"] != node_id and edge["target"] != node_id:
                    continue
                other = edge["target"] if edge["source"] == node_id else edge["source"]
                other_node = self._node_by_id(str(other))
                if other_node is None:
                    continue
                neighbors.append({"node": other_node, "edge": edge})
        else:
            for edge in self._all_edges():
                if edge["source"] != node_id and edge["target"] != node_id:
                    continue
                other = edge["target"] if edge["source"] == node_id else edge["source"]
                node = self._node_by_id(str(other))
                if node is None:
                    continue
                neighbors.append({"node": node, "edge": edge})
        neighbors.sort(key=lambda item: float(item["edge"].get("weight", 0.0)), reverse=True)
        return neighbors

    def _node_rank(self, node: dict[str, Any]) -> float:
        return float(node.get("count", 0))

    def _token_vector(self, token: str) -> list[float]:
        record = self.checkpoint.tokens.get(canonical_token(token))
        if record is None:
            return zero_vector()
        return normalize_vector(record.get("vector"))

    def _is_stop_token(self, value: str) -> bool:
        return canonical_token(str(value).removeprefix("token:")) in TERMINAL_TOKENS

    def _terminal_has_continuation(self, value: str, *, edges: list[dict[str, Any]]) -> bool:
        if not self._is_stop_token(value):
            return False
        for edge in self._outgoing_edges(str(value), edges=edges):
            target = canonical_token(str(edge.get("target") or "").removeprefix("token:"))
            if target and target not in ROLE_TOKENS and target not in TERMINAL_TOKENS:
                return True
        return False

    def _terminal_bridge_token(self, value: str, *, edges: list[dict[str, Any]]) -> str | None:
        for edge in self._outgoing_edges(str(value), edges=edges):
            target = str(edge.get("target") or "")
            if self._is_stop_token(target) and self._terminal_has_continuation(target, edges=edges):
                return target
        return None

    def _render_generated_tokens(self, token_ids: list[str]) -> str:
        parts: list[str] = []
        for item in token_ids:
            token = canonical_token(str(item).removeprefix("token:"))
            if not token or token in ROLE_TOKENS:
                continue
            if token in PUNCT_TOKENS:
                if parts:
                    parts[-1] = parts[-1].rstrip()
                parts.append(token)
                continue
            parts.append((" " if parts else "") + token)
        return "".join(parts).strip()

    def _truncate_response(self, text: str, limit: int = 240) -> str:
        cleaned = normalize_text(text)
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(limit - 3, 0)].rstrip() + "..."

    def _summary_line(self, token_scores: list[dict[str, Any]], source: str) -> str:
        labels = ", ".join(item["label"] for item in token_scores[:3] if item.get("label"))
        return f"source={source}" + (f" | tokens={labels}" if labels else "")

    def _store_result(self, result: dict[str, Any], *, session_id: str, user_text: str) -> None:
        self.checkpoint.results[result["result_id"]] = result
        self.checkpoint.sessions.setdefault(session_id, [])
        turns = self.checkpoint.sessions[session_id]
        turns.append(
            {
                "role": "user",
                "text": user_text,
                "created_at": result["created_at"],
                "result_id": result["result_id"],
                "kind": "message",
            }
        )
        turns.append(
            {
                "role": "assistant",
                "text": result["response"],
                "created_at": result["created_at"],
                "result_id": result["result_id"],
                "kind": "message",
                "source": result["response_source"],
            }
        )
        self._trim_session(session_id)
        self._trim_results()

    def _trim_session(self, session_id: str) -> None:
        turns = self.checkpoint.sessions.get(session_id)
        if turns is not None and len(turns) > self.config.session_limit:
            self.checkpoint.sessions[session_id] = turns[-self.config.session_limit :]

    def _trim_results(self) -> None:
        if len(self.checkpoint.results) <= self.config.result_limit:
            return
        ordered = sorted(self.checkpoint.results.items(), key=lambda item: float(item[1].get("created_at", 0.0)))
        keep = {key for key, _ in ordered[-self.config.result_limit :]}
        self.checkpoint.results = {key: value for key, value in self.checkpoint.results.items() if key in keep}

    def _trim_collections(self) -> None:
        for session_id in list(self.checkpoint.sessions):
            self._trim_session(session_id)
        self._trim_results()

    def _reinforce_node_neighbors(self, node_id: str, amount: float) -> None:
        for record in self._hypernode_store().values():
            if not isinstance(record, dict):
                continue
            subgraph = record.get("subgraph")
            if not isinstance(subgraph, dict):
                continue
            changed = False
            for edge_id, edge in subgraph["edges"].items():
                parsed = self._parse_edge_key(edge_id)
                if parsed is None:
                    continue
                source, _, target = parsed
                if source != node_id and target != node_id:
                    continue
                edge["weight"] = max(0.0, float(edge.get("weight", 0.0)) + amount)
                edge["pheromone"] = max(0.0, float(edge.get("pheromone", 0.0)) + amount)
                changed = True
            if changed:
                record["subgraph"] = subgraph

    def _add_edge(self, source: str, target: str, weight: float, report: dict[str, Any], now: float, subgraph_node_id: str | None = None) -> None:
        scope_id = subgraph_node_id or ROOT_HYPERNODE_ID
        subgraph = self._subgraph_store(scope_id)
        edge_id = self._edge_key(source, "next", target)
        record = subgraph["edges"].get(edge_id)
        if record is None:
            record = {"weight": 0.0, "pheromone": 0.0}
            subgraph["edges"][edge_id] = record
            report["new_edges"] = int(report.get("new_edges", 0)) + 1
        else:
            report["updated_edges"] = int(report.get("updated_edges", 0)) + 1
        record["weight"] = float(record.get("weight", 0.0)) + float(weight)
        record["pheromone"] = float(record.get("pheromone", 0.0)) * 0.92 + float(weight)

    def _edge_key(self, source: str, relation: str, target: str) -> str:
        return f"{self._canonical_edge_node_id(source)}|{self._canonical_edge_relation(relation)}|{self._canonical_edge_node_id(target)}"

    def _edge_id(self, source: str, target: str) -> str:
        return self._edge_key(source, "next", target)

    def _subgraph_store(self, node_id: str) -> dict[str, Any]:
        if not node_id or not str(node_id).startswith(HIERARCHY_NODE_PREFIX):
            node_id = ROOT_HYPERNODE_ID
        store = self._hypernode_store()
        record = store.get(node_id)
        if not isinstance(record, dict):
            if node_id == ROOT_HYPERNODE_ID:
                record = self._ensure_root_hypernode()
            else:
                record = {
                    "id": node_id,
                    "type": "hypernode",
                    "label": node_id.removeprefix(HIERARCHY_NODE_PREFIX),
                    "hierarchy": [],
                    "parent": None,
                    "depth": 0,
                    "count": 0,
                    "vector": zero_vector(),
                    "created_at": 0.0,
                    "updated_at": 0.0,
                    "subgraph": {"tokens": {}, "edges": {}},
                }
                store[node_id] = record
        if not isinstance(record.get("subgraph"), dict):
            record["subgraph"] = {"tokens": {}, "edges": {}}
        return record["subgraph"]

    def _transition_context_id(self, previous_token: str, current_token: str) -> str:
        return f"{token_id(previous_token)}|then|{token_id(current_token)}"

    def _transition_memory(self) -> dict[str, dict[str, int]]:
        memory = self.checkpoint.meta.get("transition_memory")
        if not isinstance(memory, dict):
            memory = {}
            self.checkpoint.meta["transition_memory"] = memory
        return memory

    def _add_transition_memory(self, previous_token: str, current_token: str, candidate_token: str, *, weight: int = 1) -> None:
        memory = self._transition_memory()
        context_id = self._transition_context_id(previous_token, current_token)
        targets = memory.setdefault(context_id, {})
        if not isinstance(targets, dict):
            targets = {}
            memory[context_id] = targets
        candidate_id = token_id(candidate_token)
        targets[candidate_id] = int(targets.get(candidate_id, 0) or 0) + max(int(weight or 1), 1)

    def _transition_targets(self, previous_token: str | None, current_token: str) -> dict[str, int]:
        if not previous_token:
            return {}
        targets = self.checkpoint.meta.get("transition_memory", {}).get(self._transition_context_id(previous_token, current_token), {})
        return targets if isinstance(targets, dict) else {}

    def _vector_norm(self, vector: list[float]) -> float:
        values = np.asarray(normalize_vector(vector), dtype=np.float32)
        return float(np.linalg.norm(values))

    def _touch_meta(self, key: str, value: Any) -> None:
        self.checkpoint.meta[key] = value

    def _persist(self) -> None:
        save_checkpoint(self.state_path, self.checkpoint)


def _load_engine_config(state_dir: Path | None = None) -> EngineConfig:
    return EngineConfig(state_dir=state_dir or Path(".semantic_ants"))
