from __future__ import annotations

from dataclasses import dataclass, field
import json
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any


DEFAULT_VECTOR_DIM = 384
CHECKPOINT_VERSION = 6
CHECKPOINT_SQLITE_SCHEMA_VERSION = 1
SQLITE_SPLIT_META_KEYS = {"hypernodes", "transition_memory"}


def _zero_vector(dim: int = DEFAULT_VECTOR_DIM) -> list[float]:
    return [0.0] * max(int(dim), 0)


def _coerce_vector(value: Any, dim: int = DEFAULT_VECTOR_DIM) -> list[float]:
    size = max(int(dim), 0)
    if size <= 0:
        return []
    if not isinstance(value, (list, tuple)):
        return _zero_vector(size)
    vector = [float(item) for item in value[:size]]
    if len(vector) < size:
        vector.extend([0.0] * (size - len(vector)))
    return vector


def _canonical_token(value: Any) -> str:
    token = str(value or "").strip().casefold()
    if token == "[__user__]":
        return "__user__"
    if token == "[__assistant__]":
        return "__assistant__"
    return token


def _canonical_token_id(value: Any) -> str:
    raw = str(value or "").strip()
    token = raw.removeprefix("token:")
    return f"token:{_canonical_token(token)}"


def _canonical_node_id(value: Any) -> str:
    raw = str(value or "").strip()
    if raw.startswith("token:"):
        return _canonical_token_id(raw)
    return raw


def _canonical_relation(value: Any) -> str:
    relation = str(value or "next").strip()
    return "next" if relation == "transition_edge" else relation


def _canonical_edge_key(source: Any, relation: Any, target: Any) -> str:
    return f"{_canonical_node_id(source)}|{_canonical_relation(relation)}|{_canonical_node_id(target)}"


def _normalize_edge_key(value: Any) -> str | None:
    parts = str(value or "").split("|", 2)
    if len(parts) != 3:
        return None
    source, relation, target = parts
    relation = _canonical_relation(relation)
    if relation not in {"next", "hierarchical_edge"}:
        return None
    source = _canonical_node_id(source)
    target = _canonical_node_id(target)
    if source in {"token:", "hyper:"} or target in {"token:", "hyper:"}:
        return None
    if not source.startswith(("token:", "hyper:")) or not target.startswith(("token:", "hyper:")):
        return None
    return _canonical_edge_key(source, relation, target)


def _normalize_token_record(key: str, record: Any, dim: int = DEFAULT_VECTOR_DIM) -> tuple[str, dict[str, Any]]:
    raw = dict(record or {}) if isinstance(record, dict) else {}
    token = _canonical_token(raw.get("token") or key.removeprefix("token:"))
    label = _canonical_token(raw.get("label") or raw.get("surface") or token)
    normalized = {
        "id": f"token:{token}",
        "type": "token",
        "token": token,
        "label": label or token,
        "count": int(raw.get("count", 0) or 0),
        "vector": _coerce_vector(raw.get("vector"), dim),
        "created_at": float(raw.get("created_at", 0.0) or 0.0),
        "updated_at": float(raw.get("updated_at", 0.0) or 0.0),
    }
    return token, normalized


def _normalize_edge_record(edge: Any) -> dict[str, Any] | None:
    if not isinstance(edge, dict):
        return None
    relation = str(edge.get("relation") or edge.get("type") or "next")
    if relation not in {"next", "hierarchical_edge", "transition_edge"}:
        return None
    if not str(edge.get("source") or "").startswith(("token:", "hyper:")):
        return None
    if not str(edge.get("target") or "").startswith(("token:", "hyper:")):
        return None
    source_raw = str(edge.get("source") or "")
    target_raw = str(edge.get("target") or "")
    if source_raw.startswith("token:"):
        source = _canonical_token_id(source_raw)
    else:
        source = str(source_raw).strip()
    if target_raw.startswith("token:"):
        target = _canonical_token_id(target_raw)
    else:
        target = str(target_raw).strip()
    if source in {"token:", "hyper:"} or target in {"token:", "hyper:"}:
        return None
    if not source.startswith(("token:", "hyper:")) or not target.startswith(("token:", "hyper:")):
        return None
    now = float(edge.get("updated_at", edge.get("created_at", 0.0)) or 0.0)
    edge_id = str(edge.get("id") or "").strip() or f"{source}|{relation}|{target}"
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "relation": relation,
        "type": "transition_edge" if relation == "next" else relation,
        "count": int(edge.get("count", 0) or 0),
        "weight": float(edge.get("weight", 0.0) or 0.0),
        "pheromone": float(edge.get("pheromone", 0.0) or 0.0),
        "created_at": float(edge.get("created_at", now) or now),
        "updated_at": now,
    }


def _normalize_slim_edge_record(edge: Any) -> tuple[str, dict[str, float]] | None:
    if not isinstance(edge, dict):
        return None
    relation = _canonical_relation(edge.get("relation") or edge.get("type") or "next")
    if relation not in {"next", "hierarchical_edge"}:
        return None
    source = str(edge.get("source") or "").strip()
    target = str(edge.get("target") or "").strip()
    if source.startswith("token:"):
        source = _canonical_token_id(source)
    if target.startswith("token:"):
        target = _canonical_token_id(target)
    if not source or not target:
        return None
    key = _canonical_edge_key(source, relation, target)
    return key, {
        "weight": float(edge.get("weight", 0.0) or 0.0),
        "pheromone": float(edge.get("pheromone", 0.0) or 0.0),
    }


def _normalize_slim_edge_map(value: Any) -> dict[str, dict[str, float]]:
    edges: dict[str, dict[str, float]] = {}
    if isinstance(value, dict):
        for key, edge in value.items():
            slim_key: str | None = None
            slim_value: dict[str, float] | None = None
            if isinstance(edge, dict) and ("weight" in edge or "pheromone" in edge) and str(key).count("|") >= 2:
                slim_key = _normalize_edge_key(key)
                if slim_key is None:
                    continue
                slim_value = {
                    "weight": float(edge.get("weight", 0.0) or 0.0),
                    "pheromone": float(edge.get("pheromone", 0.0) or 0.0),
                }
            else:
                normalized = _normalize_slim_edge_record(edge)
                if normalized is None:
                    continue
                slim_key, slim_value = normalized
            existing = edges.get(slim_key)
            if existing is None:
                edges[slim_key] = slim_value
            else:
                existing["weight"] = float(existing.get("weight", 0.0)) + float(slim_value.get("weight", 0.0))
                existing["pheromone"] = max(float(existing.get("pheromone", 0.0)), float(slim_value.get("pheromone", 0.0)))
    elif isinstance(value, list):
        for edge in value:
            normalized = _normalize_slim_edge_record(edge)
            if normalized is None:
                continue
            slim_key, slim_value = normalized
            existing = edges.get(slim_key)
            if existing is None:
                edges[slim_key] = slim_value
            else:
                existing["weight"] = float(existing.get("weight", 0.0)) + float(slim_value.get("weight", 0.0))
                existing["pheromone"] = max(float(existing.get("pheromone", 0.0)), float(slim_value.get("pheromone", 0.0)))
    return edges


def _normalize_local_token_record(key: str, record: Any) -> tuple[str, dict[str, Any]] | None:
    raw = dict(record or {}) if isinstance(record, dict) else {}
    token = _canonical_token(raw.get("token") or str(key).removeprefix("token:"))
    if not token:
        return None
    token_key = _canonical_token_id(raw.get("id") or key or token)
    return token_key, {
        "id": token_key,
        "type": "token",
        "token": token,
        "label": _canonical_token(raw.get("label") or raw.get("surface") or token) or token,
        "count": int(raw.get("count", 0) or 0),
    }


def _normalize_subgraph_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"tokens": {}, "edges": {}}
    tokens: dict[str, dict[str, Any]] = {}
    raw_tokens = value.get("tokens")
    if isinstance(raw_tokens, dict):
        for key, record in raw_tokens.items():
            normalized = _normalize_local_token_record(str(key), record)
            if normalized is None:
                continue
            token_key, token_record = normalized
            tokens[token_key] = token_record
    elif isinstance(raw_tokens, list):
        for record in raw_tokens:
            normalized = _normalize_local_token_record("", record)
            if normalized is None:
                continue
            token_key, token_record = normalized
            tokens[token_key] = token_record
    if not tokens and isinstance(value.get("nodes"), list):
        for node in value.get("nodes", []) or []:
            if not isinstance(node, dict) or not str(node.get("id") or "").startswith("token:"):
                continue
            normalized = _normalize_local_token_record(str(node.get("id") or ""), node)
            if normalized is None:
                continue
            token_key, token_record = normalized
            tokens[token_key] = token_record
    edges = _normalize_slim_edge_map(value.get("edges"))
    for edge_key in edges:
        parts = str(edge_key).split("|", 2)
        if len(parts) != 3:
            continue
        for endpoint in (parts[0], parts[2]):
            if not str(endpoint).startswith("token:") or endpoint in tokens:
                continue
            token = _canonical_token(str(endpoint).removeprefix("token:"))
            if not token:
                continue
            tokens[endpoint] = {
                "id": endpoint,
                "type": "token",
                "token": token,
                "label": token,
                "count": 0,
            }
    return {"tokens": tokens, "edges": edges}


def _normalize_meta_hypernodes(meta: dict[str, Any]) -> dict[str, Any]:
    hypernodes = meta.get("hypernodes")
    if not isinstance(hypernodes, dict):
        return meta
    for key, record in list(hypernodes.items()):
        if not isinstance(record, dict):
            hypernodes.pop(key, None)
            continue
        record["subgraph"] = _normalize_subgraph_payload(record.get("subgraph"))
    return meta


def _merge_root_edges(meta: dict[str, Any], root_edges: Any) -> dict[str, Any]:
    if not isinstance(meta, dict):
        meta = {}
    hypernodes = meta.get("hypernodes")
    if not isinstance(hypernodes, dict):
        hypernodes = {}
        meta["hypernodes"] = hypernodes
    root = hypernodes.get("hyper:__root__")
    if not isinstance(root, dict):
        root = {
            "id": "hyper:__root__",
            "type": "hypernode",
            "label": "__root__",
            "hierarchy": [],
            "parent": None,
            "depth": 0,
            "count": 0,
            "vector": _zero_vector(),
            "created_at": 0.0,
            "updated_at": 0.0,
            "subgraph": {"tokens": {}, "edges": {}},
        }
        hypernodes["hyper:__root__"] = root
    subgraph = root.get("subgraph")
    if not isinstance(subgraph, dict):
        subgraph = {"tokens": {}, "edges": {}}
        root["subgraph"] = subgraph
    edges = _normalize_slim_edge_map(subgraph.get("edges"))
    for key, value in _normalize_slim_edge_map(root_edges).items():
        parts = str(key).split("|", 2)
        if len(parts) != 3:
            continue
        source, relation, target = parts
        relation = _canonical_relation(relation)
        if relation != "next":
            continue
        if not source.startswith("token:") or not target.startswith("token:"):
            continue
        existing = edges.get(key)
        if existing is None:
            edges[key] = value
        else:
            existing["weight"] = float(existing.get("weight", 0.0)) + float(value.get("weight", 0.0))
            existing["pheromone"] = max(float(existing.get("pheromone", 0.0)), float(value.get("pheromone", 0.0)))
    subgraph["edges"] = edges
    if not isinstance(subgraph.get("tokens"), dict):
        subgraph["tokens"] = {}
    return meta


def _normalize_graph_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    nodes = []
    for node in value.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if not node_id.startswith(("token:", "hyper:")):
            continue
        cleaned_node = dict(node)
        if node_id.startswith("token:"):
            cleaned_node["id"] = _canonical_token_id(cleaned_node.get("id"))
            cleaned_node["type"] = "token"
            cleaned_node["shape"] = "circle"
        else:
            cleaned_node["id"] = node_id
            cleaned_node["type"] = str(cleaned_node.get("type") or "hypernode")
            cleaned_node["shape"] = str(cleaned_node.get("shape") or "rect")
        nodes.append(cleaned_node)
    node_ids = {str(node.get("id")) for node in nodes}
    edges = []
    for edge in value.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        relation = str(edge.get("relation") or edge.get("type") or "next")
        if relation not in {"next", "hierarchical_edge", "transition_edge"}:
            continue
        if not str(edge.get("source") or "").startswith(("token:", "hyper:")):
            continue
        if not str(edge.get("target") or "").startswith(("token:", "hyper:")):
            continue
        source = _canonical_token_id(edge.get("source")) if str(edge.get("source") or "").startswith("token:") else str(edge.get("source") or "").strip()
        target = _canonical_token_id(edge.get("target")) if str(edge.get("target") or "").startswith("token:") else str(edge.get("target") or "").strip()
        if source in node_ids and target in node_ids:
            cleaned = dict(edge)
            cleaned["source"] = source
            cleaned["target"] = target
            cleaned["relation"] = relation
            cleaned["type"] = "transition_edge" if relation == "next" else relation
            edges.append(cleaned)
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "tokens": len(nodes),
            "seeds": len(value.get("seed_ids", []) or []),
        },
        "seed_ids": [_canonical_token_id(node_id) for node_id in (value.get("seed_ids", []) or []) if str(node_id).startswith("token:")],
        "node_ids": [node["id"] for node in nodes],
        "edge_ids": [edge.get("id") for edge in edges if edge.get("id")],
    }


def _normalize_result_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    normalized = dict(record)
    normalized.pop("top_sentences", None)
    trace = dict(normalized.get("trace") or {}) if isinstance(normalized.get("trace"), dict) else {}
    for key in ("sentence_scores", "dialogue_match", "dialogue_backpack"):
        trace.pop(key, None)
    normalized["trace"] = trace
    if "graph_data" in normalized:
        normalized["graph_data"] = _normalize_graph_payload(normalized.get("graph_data"))
    if "backpack" in normalized:
        backpack = dict(normalized.get("backpack") or {}) if isinstance(normalized.get("backpack"), dict) else {}
        if "graph_data" in backpack:
            backpack["graph_data"] = _normalize_graph_payload(backpack.get("graph_data"))
        elif "nodes" in backpack or "edges" in backpack:
            backpack = _normalize_graph_payload(backpack)
        normalized["backpack"] = backpack
    return normalized


def _encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _decode_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value).decode("utf-8")
    if not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _vector_to_blob(vector: Any, dim: int = DEFAULT_VECTOR_DIM) -> bytes:
    values = _coerce_vector(vector, dim)
    if not values:
        return b""
    return struct.pack(f"<{len(values)}f", *[float(value) for value in values])


def _vector_from_blob(blob: Any, dim: int = DEFAULT_VECTOR_DIM) -> list[float]:
    size = max(int(dim), 0)
    if size <= 0:
        return []
    if blob in {None, b"", ""}:
        return _zero_vector(size)
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    if not isinstance(blob, (bytes, bytearray)):
        return _zero_vector(size)
    raw = bytes(blob)
    if not raw:
        return _zero_vector(size)
    usable = len(raw) - (len(raw) % 4)
    if usable <= 0:
        return _zero_vector(size)
    values = list(struct.unpack(f"<{usable // 4}f", raw[:usable]))
    if len(values) < size:
        values.extend([0.0] * (size - len(values)))
    return values[:size]


def _write_sqlite_checkpoint(path: Path, checkpoint: "Checkpoint") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    conn = sqlite3.connect(tmp_path)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tokens (token TEXT PRIMARY KEY, record_json TEXT NOT NULL, vector BLOB NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, value_json TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS results (result_id TEXT PRIMARY KEY, value_json TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hypernodes (node_id TEXT PRIMARY KEY, value_json TEXT NOT NULL, vector BLOB NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS subgraph_tokens (node_id TEXT NOT NULL, token_id TEXT NOT NULL, value_json TEXT NOT NULL, PRIMARY KEY (node_id, token_id))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS subgraph_edges (node_id TEXT NOT NULL, edge_id TEXT NOT NULL, value_json TEXT NOT NULL, PRIMARY KEY (node_id, edge_id))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS transition_memory (context_id TEXT NOT NULL, candidate_id TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY (context_id, candidate_id))"
        )
        conn.execute("DELETE FROM meta")
        conn.execute("DELETE FROM tokens")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM results")
        conn.execute("DELETE FROM hypernodes")
        conn.execute("DELETE FROM subgraph_tokens")
        conn.execute("DELETE FROM subgraph_edges")
        conn.execute("DELETE FROM transition_memory")
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("version", str(int(checkpoint.version))),
                ("vector_dim", str(int(checkpoint.vector_dim))),
                ("schema_version", str(CHECKPOINT_SQLITE_SCHEMA_VERSION)),
            ]
            + [
                (str(key), _encode_json(value))
                for key, value in sorted(checkpoint.meta.items(), key=lambda item: str(item[0]))
                if str(key) not in SQLITE_SPLIT_META_KEYS
            ],
        )
        conn.executemany(
            "INSERT INTO tokens (token, record_json, vector) VALUES (?, ?, ?)",
            [
                (
                    str(token),
                    _encode_json({k: v for k, v in record.items() if k != "vector"}),
                    _vector_to_blob(record.get("vector"), int(checkpoint.vector_dim)),
                )
                for token, record in sorted(checkpoint.tokens.items(), key=lambda item: str(item[0]))
            ],
        )
        conn.executemany(
            "INSERT INTO sessions (session_id, value_json) VALUES (?, ?)",
            [(str(session_id), _encode_json(turns)) for session_id, turns in sorted(checkpoint.sessions.items(), key=lambda item: str(item[0]))],
        )
        conn.executemany(
            "INSERT INTO results (result_id, value_json) VALUES (?, ?)",
            [(str(result_id), _encode_json(record)) for result_id, record in sorted(checkpoint.results.items(), key=lambda item: str(item[0]))],
        )
        hypernodes = checkpoint.meta.get("hypernodes")
        if isinstance(hypernodes, dict):
            hypernode_rows = []
            token_rows = []
            edge_rows = []
            for node_id, record in sorted(hypernodes.items(), key=lambda item: str(item[0])):
                if not isinstance(record, dict):
                    continue
                subgraph = record.get("subgraph")
                payload = {k: v for k, v in record.items() if k not in {"vector", "subgraph"}}
                hypernode_rows.append(
                    (
                        str(node_id),
                        _encode_json(payload),
                        _vector_to_blob(record.get("vector"), int(checkpoint.vector_dim)),
                    )
                )
                if isinstance(subgraph, dict):
                    tokens = subgraph.get("tokens")
                    if isinstance(tokens, dict):
                        for token_id, token_record in sorted(tokens.items(), key=lambda item: str(item[0])):
                            token_rows.append((str(node_id), str(token_id), _encode_json(token_record)))
                    edges = subgraph.get("edges")
                    if isinstance(edges, dict):
                        for edge_id, edge_record in sorted(edges.items(), key=lambda item: str(item[0])):
                            edge_rows.append((str(node_id), str(edge_id), _encode_json(edge_record)))
            if hypernode_rows:
                conn.executemany(
                    "INSERT INTO hypernodes (node_id, value_json, vector) VALUES (?, ?, ?)",
                    hypernode_rows,
                )
            if token_rows:
                conn.executemany(
                    "INSERT INTO subgraph_tokens (node_id, token_id, value_json) VALUES (?, ?, ?)",
                    token_rows,
                )
            if edge_rows:
                conn.executemany(
                    "INSERT INTO subgraph_edges (node_id, edge_id, value_json) VALUES (?, ?, ?)",
                    edge_rows,
                )
        transition_memory = checkpoint.meta.get("transition_memory")
        if isinstance(transition_memory, dict):
            rows = []
            for context_id, targets in sorted(transition_memory.items(), key=lambda item: str(item[0])):
                if not isinstance(targets, dict):
                    continue
                for candidate_id, count in sorted(targets.items(), key=lambda item: str(item[0])):
                    rows.append((str(context_id), str(candidate_id), int(count or 0)))
            if rows:
                conn.executemany(
                    "INSERT INTO transition_memory (context_id, candidate_id, count) VALUES (?, ?, ?)",
                    rows,
                )
        conn.commit()
    finally:
        conn.close()
    tmp_path.replace(path)


def _read_sqlite_checkpoint(path: Path) -> Checkpoint:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        meta_rows = conn.execute("SELECT key, value FROM meta").fetchall()
        meta: dict[str, Any] = {}
        version = CHECKPOINT_VERSION
        vector_dim = DEFAULT_VECTOR_DIM
        for row in meta_rows:
            key = str(row["key"])
            value = row["value"]
            if key == "version":
                try:
                    version = int(value)
                except (TypeError, ValueError):
                    version = CHECKPOINT_VERSION
                continue
            if key == "vector_dim":
                try:
                    vector_dim = int(value)
                except (TypeError, ValueError):
                    vector_dim = DEFAULT_VECTOR_DIM
                continue
            if key == "schema_version":
                continue
            meta[key] = _decode_json(value, value)
        tokens: dict[str, dict[str, Any]] = {}
        for row in conn.execute("SELECT token, record_json, vector FROM tokens"):
            record = _decode_json(row["record_json"], {})
            if not isinstance(record, dict):
                record = {}
            token = _canonical_token(row["token"])
            if not token:
                continue
            record["id"] = f"token:{token}"
            record["type"] = "token"
            record["token"] = token
            record["vector"] = _vector_from_blob(row["vector"], vector_dim)
            tokens[token] = _normalize_token_record(token, record, vector_dim)[1]
        sessions = {
            str(row["session_id"]): _decode_json(row["value_json"], [])
            for row in conn.execute("SELECT session_id, value_json FROM sessions")
        }
        results = {
            str(row["result_id"]): _normalize_result_record(_decode_json(row["value_json"], {}))
            for row in conn.execute("SELECT result_id, value_json FROM results")
        }
        hypernodes: dict[str, Any] = {}
        for row in conn.execute("SELECT node_id, value_json, vector FROM hypernodes"):
            payload = _decode_json(row["value_json"], {})
            if not isinstance(payload, dict):
                payload = {}
            node_id = str(row["node_id"])
            payload["id"] = node_id
            payload["vector"] = _vector_from_blob(row["vector"], vector_dim)
            payload["subgraph"] = {"tokens": {}, "edges": {}}
            hypernodes[node_id] = payload
        for row in conn.execute("SELECT node_id, token_id, value_json FROM subgraph_tokens"):
            node_id = str(row["node_id"])
            if node_id not in hypernodes:
                continue
            subgraph = hypernodes[node_id].setdefault("subgraph", {"tokens": {}, "edges": {}})
            tokens_store = subgraph.setdefault("tokens", {})
            token_record = _decode_json(row["value_json"], {})
            if isinstance(token_record, dict):
                tokens_store[str(row["token_id"])] = token_record
        for row in conn.execute("SELECT node_id, edge_id, value_json FROM subgraph_edges"):
            node_id = str(row["node_id"])
            if node_id not in hypernodes:
                continue
            subgraph = hypernodes[node_id].setdefault("subgraph", {"tokens": {}, "edges": {}})
            edges_store = subgraph.setdefault("edges", {})
            edge_record = _decode_json(row["value_json"], {})
            if isinstance(edge_record, dict):
                edges_store[str(row["edge_id"])] = edge_record
        transition_memory: dict[str, dict[str, int]] = {}
        for row in conn.execute("SELECT context_id, candidate_id, count FROM transition_memory"):
            transition_memory.setdefault(str(row["context_id"]), {})[str(row["candidate_id"])] = int(row["count"] or 0)
        meta["hypernodes"] = hypernodes
        meta["transition_memory"] = transition_memory
        return Checkpoint(
            version=version,
            vector_dim=vector_dim,
            tokens=tokens,
            sessions={str(key): list(value or []) for key, value in sessions.items()},
            results=results,
            meta=meta,
        )
    finally:
        conn.close()


@dataclass
class Checkpoint:
    version: int = CHECKPOINT_VERSION
    vector_dim: int = DEFAULT_VECTOR_DIM
    tokens: dict[str, dict[str, Any]] = field(default_factory=dict)
    sessions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "vector_dim": DEFAULT_VECTOR_DIM,
            "tokens": self.tokens,
            "sessions": self.sessions,
            "results": self.results,
            "meta": self.meta,
        }

    @property
    def edges(self) -> dict[str, dict[str, Any]]:
        meta = self.meta
        hypernodes = meta.get("hypernodes")
        if not isinstance(hypernodes, dict):
            hypernodes = {}
            meta["hypernodes"] = hypernodes
        root = hypernodes.get("hyper:__root__")
        if not isinstance(root, dict):
            root = {
                "id": "hyper:__root__",
                "type": "hypernode",
                "label": "__root__",
                "hierarchy": [],
                "parent": None,
                "depth": 0,
                "count": 0,
                "vector": _zero_vector(),
                "created_at": 0.0,
                "updated_at": 0.0,
                "subgraph": {"tokens": {}, "edges": {}},
            }
            hypernodes["hyper:__root__"] = root
        subgraph = root.get("subgraph")
        if not isinstance(subgraph, dict):
            subgraph = {"tokens": {}, "edges": {}}
            root["subgraph"] = subgraph
        tokens = subgraph.get("tokens")
        if not isinstance(tokens, dict):
            subgraph["tokens"] = {}
        edges = subgraph.get("edges")
        if not isinstance(edges, dict):
            subgraph["edges"] = {}
        else:
            subgraph["edges"] = _normalize_slim_edge_map(edges)
        return subgraph["edges"]

    @edges.setter
    def edges(self, value: dict[str, dict[str, Any]]) -> None:
        self.meta = _merge_root_edges(self.meta, value)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        tokens: dict[str, dict[str, Any]] = {}
        for key, value in dict(data.get("tokens") or {}).items():
            token, record = _normalize_token_record(str(key), value, DEFAULT_VECTOR_DIM)
            if token:
                tokens[token] = record

        meta = _normalize_meta_hypernodes(dict(data.get("meta") or {}))
        meta = _merge_root_edges(meta, data.get("edges"))
        meta = _normalize_meta_hypernodes(meta)

        return cls(
            version=CHECKPOINT_VERSION,
            vector_dim=DEFAULT_VECTOR_DIM,
            tokens=tokens,
            sessions={str(key): list(value or []) for key, value in dict(data.get("sessions") or {}).items()},
            results={str(key): _normalize_result_record(value) for key, value in dict(data.get("results") or {}).items()},
            meta=meta,
        )

def load_checkpoint(path: Path, *, default: Checkpoint | None = None) -> Checkpoint:
    sqlite_path = path if path.suffix == ".sqlite" else path.with_suffix(".sqlite")
    if not sqlite_path.exists():
        return default or Checkpoint()
    return _read_sqlite_checkpoint(sqlite_path)


def save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    sqlite_path = path if path.suffix == ".sqlite" else path.with_suffix(".sqlite")
    _write_sqlite_checkpoint(sqlite_path, checkpoint)
    manifest_path = sqlite_path.with_suffix(".json")
    manifest_payload = json.dumps(
        {
            "version": checkpoint.version,
            "vector_dim": checkpoint.vector_dim,
            "checkpoint_format": "sqlite",
            "sqlite_path": str(sqlite_path.name),
            "token_count": len(checkpoint.tokens),
            "timestamp": time.time(),
        },
        ensure_ascii=False,
        indent=2,
    )
    tmp_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_manifest_path.write_text(
        manifest_payload,
        encoding="utf-8",
    )
    tmp_manifest_path.replace(manifest_path)
