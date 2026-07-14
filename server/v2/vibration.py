"""Deterministic, placement-local hive vibration engine."""

from __future__ import annotations

import hashlib
import json
import math
import random
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .repository import V2Repository, encode, utcnow


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class VibrationConfig:
    reasoning_steps: int = 1
    propagation_ticks_per_step: int = 3
    settle_ticks: int = 2
    initial_temperature: float = 1.0
    temperature_decay: float = 0.72
    shake_amplitude: float = 8.0
    query_energy_gain: float = 0.25
    resonance_gain: float = 0.18
    exploration_noise: float = 0.12
    damping: float = 0.82
    gravity_learning_rate: float = 0.18
    activation_learning_rate: float = 0.32
    retention_learning_rate: float = 0.16
    repulsion_strength: float = 0.4
    attraction_strength: float = 0.7
    overcrowding_pressure: float = 0.08
    contradiction_pressure: float = 0.12
    energy_decay: float = 0.94
    activation_decay: float = 0.91
    stored_strength_decay: float = 0.995
    weak_node_threshold: float = 0.28
    eviction_threshold: float = 0.14
    eviction_confirmation_steps: int = 2
    convergence_threshold: float = 0.002
    convergence_patience: int = 2
    max_velocity: float = 18.0
    delta_time: float = 0.15
    min_distance: float = 24.0
    boundary_width: float = 840.0
    boundary_height: float = 560.0
    boundary_mode: str = "CLAMP"
    deterministic: bool = True
    random_seed: int = 0
    level_frequencies: Dict[str, float] = field(default_factory=dict)

    def normalized(self) -> "VibrationConfig":
        values = asdict(self)
        values["reasoning_steps"] = max(0, int(values["reasoning_steps"]))
        values["propagation_ticks_per_step"] = max(1, int(values["propagation_ticks_per_step"]))
        values["settle_ticks"] = max(0, int(values["settle_ticks"]))
        values["eviction_confirmation_steps"] = max(1, int(values["eviction_confirmation_steps"]))
        values["convergence_patience"] = max(1, int(values["convergence_patience"]))
        values["boundary_mode"] = str(values["boundary_mode"]).upper()
        values["level_frequencies"] = dict(values.get("level_frequencies") or {
            "scene": 0.6, "concept": 0.8, "lexeme": 1.0, "word_form": 1.2,
            "morpheme": 1.4, "morph_pattern": 1.4, "character": 1.7,
        })
        if values["boundary_mode"] not in {"CLAMP", "BOUNCE", "WRAP", "SOFT_REPULSION"}:
            values["boundary_mode"] = "CLAMP"
        return VibrationConfig(**values)


@dataclass(frozen=True)
class QueryActivation:
    cloud_ids: Tuple[int, ...] = ()
    terms: Tuple[str, ...] = ()
    roles: Tuple[str, ...] = ()
    relevance: float = 1.0

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> "QueryActivation":
        value = value or {}
        return cls(
            tuple(sorted({int(item) for item in value.get("cloud_ids", []) if item is not None})),
            tuple(str(item).casefold() for item in value.get("terms", [])),
            tuple(str(item) for item in value.get("roles", [])),
            clamp(float(value.get("relevance", 1.0))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cloud_ids": list(self.cloud_ids),
            "terms": list(self.terms),
            "roles": list(self.roles),
            "relevance": self.relevance,
        }


@dataclass
class VibrationResult:
    run: Dict[str, Any]
    initial_state: Dict[str, Any]
    final_state: Dict[str, Any]
    snapshots: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    clusters: List[Dict[str, Any]]
    evicted_nodes: List[int]
    convergence_score: float
    stop_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HiveVibrationEngine:
    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()

    @staticmethod
    def _hash(state: Dict[str, Any]) -> str:
        return hashlib.sha256(encode(state).encode("utf-8")).hexdigest()

    def _load_nodes(self, conn: Any, hive_id: str) -> List[Dict[str, Any]]:
        rows = conn.execute(
            """SELECT hc.id cell_id, hc.hive_placement_id placement_id, hc.dominant_cloud_id cloud_id,
            hc.stored_strength cell_strength, hc.retention cell_retention, hc.local_activation cell_activation,
            p.x, p.y, p.z, p.local_gravity, c.cloud_type, c.canonical_name
            FROM hive_cells hc JOIN cloud_placements p ON p.id = hc.hive_placement_id
            JOIN clouds c ON c.id = hc.dominant_cloud_id WHERE hc.hive_id = ?""",
            (hive_id,),
        ).fetchall()
        nodes: List[Dict[str, Any]] = []
        for row in rows:
            existing = conn.execute(
                "SELECT * FROM hive_node_states WHERE hive_id = ? AND placement_id = ?",
                (hive_id, row["placement_id"]),
            ).fetchone()
            if existing:
                node = dict(existing)
            else:
                node = {
                    "hive_id": hive_id,
                    "placement_id": int(row["placement_id"]),
                    "cloud_id": int(row["cloud_id"]),
                    "node_type": str(row["cloud_type"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": row["z"],
                    "velocity_x": 0.0,
                    "velocity_y": 0.0,
                    "velocity_z": 0.0,
                    "local_activation": clamp(row["cell_activation"]),
                    "local_gravity": clamp(row["local_gravity"]),
                    "stored_strength": clamp(row["cell_strength"]),
                    "local_stability": clamp(row["cell_retention"]),
                    "retention": clamp(row["cell_retention"]),
                    "energy": clamp(row["cell_activation"]),
                    "phase": 0.0,
                    "frequency": 1.0,
                    "temperature_response": 1.0,
                    "age_steps": 0,
                    "activation_count": 0,
                    "last_activated_step": 0,
                    "weakening_steps": 0,
                    "eviction_status": "ACTIVE",
                    "metadata_json": "{}",
                }
                self._upsert_node(conn, node)
            node["label"] = str(row["canonical_name"])
            node["components"] = [
                dict(item)
                for item in conn.execute(
                    """SELECT hcc.*, c.canonical_name FROM hive_cell_components hcc
                JOIN clouds c ON c.id = hcc.cloud_id WHERE hcc.cell_id = ? ORDER BY hcc.composition_share DESC""",
                    (row["cell_id"],),
                )
            ]
            nodes.append(node)
        return nodes

    @staticmethod
    def _upsert_node(conn: Any, node: Dict[str, Any]) -> None:
        columns = [
            "hive_id",
            "placement_id",
            "cloud_id",
            "node_type",
            "x",
            "y",
            "z",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "local_activation",
            "local_gravity",
            "stored_strength",
            "local_stability",
            "retention",
            "energy",
            "phase",
            "frequency",
            "temperature_response",
            "age_steps",
            "activation_count",
            "last_activated_step",
            "weakening_steps",
            "eviction_status",
            "metadata_json",
        ]
        values = [node.get(key) for key in columns]
        marks = ",".join("?" for _ in columns)
        updates = ",".join(f"{key}=excluded.{key}" for key in columns[2:])
        conn.execute(
            f"INSERT INTO hive_node_states ({','.join(columns)}) VALUES ({marks}) ON CONFLICT(hive_id, placement_id) DO UPDATE SET {updates}",
            values,
        )

    @staticmethod
    def _state(nodes: Iterable[Dict[str, Any]], temperature: float = 1.0) -> Dict[str, Any]:
        return {
            "temperature": temperature,
            "nodes": [
                {
                    key: node.get(key)
                    for key in (
                        "placement_id",
                        "cloud_id",
                        "node_type",
                        "x",
                        "y",
                        "z",
                        "velocity_x",
                        "velocity_y",
                        "velocity_z",
                        "local_activation",
                        "local_gravity",
                        "stored_strength",
                        "local_stability",
                        "retention",
                        "energy",
                        "phase",
                        "frequency",
                        "temperature_response",
                        "age_steps",
                        "activation_count",
                        "last_activated_step",
                        "weakening_steps",
                        "eviction_status",
                    )
                }
                for node in sorted(nodes, key=lambda item: int(item["placement_id"]))
            ],
        }

    @staticmethod
    def _query_support(node: Dict[str, Any], query: QueryActivation) -> float:
        if int(node["cloud_id"]) in query.cloud_ids:
            return query.relevance
        for component in node.get("components", []):
            if int(component["cloud_id"]) in query.cloud_ids:
                return query.relevance * float(component["composition_share"])
        return 0.0

    def _snapshot(
        self,
        conn: Any,
        run_id: str,
        hive_id: str,
        step: int,
        phase: str,
        nodes: List[Dict[str, Any]],
        temperature: float,
        delta: Optional[Dict[str, Any]] = None,
        clusters: Optional[List[Dict[str, Any]]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        state = self._state(nodes, temperature)
        item = {
            "id": f"snapshot-{uuid.uuid4().hex}",
            "run_id": run_id,
            "hive_id": hive_id,
            "step": step,
            "phase": phase,
            "state_hash": self._hash(state),
            "state": state,
            "delta": delta or {},
            "clusters": clusters or [],
            "events": events or [],
            "created_at": utcnow(),
        }
        conn.execute(
            "INSERT INTO hive_reasoning_snapshots (id,run_id,hive_id,step,phase,state_hash,state_json,delta_json,clusters_json,events_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                item["id"],
                run_id,
                hive_id,
                step,
                phase,
                item["state_hash"],
                encode(state),
                encode(item["delta"]),
                encode(item["clusters"]),
                encode(item["events"]),
                item["created_at"],
            ),
        )
        return item

    def _clusters(
        self, conn: Any, run_id: str, hive_id: str, step: int, nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        active = [
            node
            for node in nodes
            if node["eviction_status"] != "EVICTED" and node["local_activation"] >= 0.2
        ]
        groups: List[List[Dict[str, Any]]] = []
        remaining = active[:]
        while remaining:
            group = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                for node in remaining[:]:
                    if any(
                        math.hypot(node["x"] - member["x"], node["y"] - member["y"]) <= 180
                        for member in group
                    ):
                        group.append(node)
                        remaining.remove(node)
                        changed = True
            if len(group) >= 2:
                groups.append(group)
        result: List[Dict[str, Any]] = []
        for group in groups:
            member_ids = [int(node["placement_id"]) for node in group]
            cohesion = clamp(
                sum(node["local_activation"] * node["local_gravity"] for node in group) / len(group)
            )
            item = {
                "id": f"cluster-{uuid.uuid4().hex}",
                "run_id": run_id,
                "hive_id": hive_id,
                "reasoning_step": step,
                "member_placement_ids": member_ids,
                "dominant_cloud_ids": sorted({int(node["cloud_id"]) for node in group}),
                "cohesion": cohesion,
                "total_energy": sum(float(node["energy"]) for node in group),
                "average_gravity": sum(float(node["local_gravity"]) for node in group) / len(group),
                "query_relevance": sum(float(node["local_activation"]) for node in group)
                / len(group),
                "status": "ACTIVE",
                "created_at": utcnow(),
            }
            conn.execute(
                "INSERT INTO hive_resonance_clusters (id,run_id,hive_id,reasoning_step,member_placement_ids_json,dominant_cloud_ids_json,cohesion,total_energy,average_gravity,query_relevance,status,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    item["id"],
                    run_id,
                    hive_id,
                    step,
                    encode(member_ids),
                    encode(item["dominant_cloud_ids"]),
                    item["cohesion"],
                    item["total_energy"],
                    item["average_gravity"],
                    item["query_relevance"],
                    item["status"],
                    item["created_at"],
                ),
            )
            result.append(item)
        return result

    def reason(
        self,
        hive_id: Any,
        query: Optional[QueryActivation] = None,
        config: Optional[VibrationConfig] = None,
    ) -> VibrationResult:
        if isinstance(hive_id, dict):
            hive_id = str(hive_id["id"])
        config = (config or VibrationConfig()).normalized()
        query = query or QueryActivation()
        with self.repository.transaction() as conn:
            hive = conn.execute("SELECT * FROM hives WHERE id = ?", (hive_id,)).fetchone()
            if not hive:
                raise KeyError(hive_id)
            seed = int(config.random_seed or hive["random_seed"] or 0)
            rng = random.Random(seed)
            run_id = f"run-{uuid.uuid4().hex}"
            now = utcnow()
            conn.execute(
                "INSERT INTO hive_reasoning_runs (id,hive_id,status,reasoning_steps,query_json,config_json,random_seed,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    hive_id,
                    "COMPLETED",
                    config.reasoning_steps,
                    encode(query.to_dict()),
                    encode(asdict(config)),
                    seed,
                    now,
                ),
            )
            nodes = self._load_nodes(conn, hive_id)
            for node in nodes:
                node["frequency"] = float(config.level_frequencies.get(node.get("node_type", ""), 1.0))
            temperature = float(hive["current_temperature"] or config.initial_temperature)
            initial = self._snapshot(conn, run_id, hive_id, 0, "INITIAL", nodes, temperature)
            snapshots = [initial]
            events: List[Dict[str, Any]] = []
            clusters: List[Dict[str, Any]] = []
            convergence = 1.0
            stable_count = 0
            completed = 0
            for step in range(1, config.reasoning_steps + 1):
                before = self._state(nodes, temperature)
                snapshots.append(
                    self._snapshot(conn, run_id, hive_id, step, "BEFORE_SHAKE", nodes, temperature)
                )
                step_events: List[Dict[str, Any]] = []
                for node in nodes:
                    if node["eviction_status"] == "EVICTED":
                        continue
                    support = self._query_support(node, query)
                    noise = (
                        (rng.random() * 2.0 - 1.0) * config.exploration_noise * temperature
                        if config.deterministic or config.exploration_noise
                        else 0.0
                    )
                    node["energy"] = clamp(
                        node["energy"] + support * config.query_energy_gain + noise * 0.05
                    )
                    node["local_activation"] = clamp(
                        node["local_activation"]
                        + config.activation_learning_rate
                        * (support - (1.0 - config.activation_decay))
                    )
                    node["local_gravity"] = clamp(
                        node["local_gravity"]
                        + config.gravity_learning_rate
                        * (support - config.contradiction_pressure * (1.0 - support))
                    )
                    node["velocity_x"] += (
                        noise + support * config.shake_amplitude * 0.1
                    ) * config.delta_time
                    node["velocity_y"] += (
                        (rng.random() * 2.0 - 1.0)
                        * config.exploration_noise
                        * temperature
                        * config.delta_time
                    )
                    if support > 0:
                        step_events.append(
                            {
                                "event_type": "QUERY_ENERGY",
                                "placement_id": node["placement_id"],
                                "support": support,
                            }
                        )
                    if support > 0:
                        node["activation_count"] += 1
                        node["last_activated_step"] = step
                for _ in range(config.propagation_ticks_per_step + config.settle_ticks):
                    active = [node for node in nodes if node["eviction_status"] != "EVICTED"]
                    forces = {int(node["placement_id"]): [0.0, 0.0] for node in active}
                    for index, left in enumerate(active):
                        for right in active[index + 1 :]:
                            dx, dy = (
                                float(right["x"]) - float(left["x"]),
                                float(right["y"]) - float(left["y"]),
                            )
                            distance = max(config.min_distance, math.hypot(dx, dy))
                            overlap = sum(
                                min(
                                    float(a.get("composition_share", 0)),
                                    float(b.get("composition_share", 0)),
                                )
                                for a in left.get("components", [])
                                for b in right.get("components", [])
                                if int(a["cloud_id"]) == int(b["cloud_id"])
                            )
                            resonance = clamp(
                                (
                                    overlap
                                    + (
                                        1.0
                                        if int(left["cloud_id"]) == int(right["cloud_id"])
                                        else 0.0
                                    )
                                )
                                * 0.5
                                + left["local_activation"] * right["local_activation"] * 0.5
                            )
                            force = (
                                config.attraction_strength * resonance / distance
                                - config.repulsion_strength * (config.min_distance / distance) ** 2
                            )
                            fx, fy = force * dx / distance, force * dy / distance
                            forces[int(left["placement_id"])][0] += fx
                            forces[int(left["placement_id"])][1] += fy
                            forces[int(right["placement_id"])][0] -= fx
                            forces[int(right["placement_id"])][1] -= fy
                    for node in active:
                        fx, fy = forces[int(node["placement_id"])]
                        node["velocity_x"] = max(
                            -config.max_velocity,
                            min(
                                config.max_velocity,
                                (node["velocity_x"] + fx * config.delta_time) * config.damping,
                            ),
                        )
                        node["velocity_y"] = max(
                            -config.max_velocity,
                            min(
                                config.max_velocity,
                                (node["velocity_y"] + fy * config.delta_time) * config.damping,
                            ),
                        )
                        node["x"] += node["velocity_x"] * config.delta_time
                        node["y"] += node["velocity_y"] * config.delta_time
                        if config.boundary_mode == "WRAP":
                            node["x"] %= config.boundary_width
                            node["y"] %= config.boundary_height
                        elif config.boundary_mode == "BOUNCE":
                            if node["x"] < 0 or node["x"] > config.boundary_width:
                                node["velocity_x"] *= -1
                            if node["y"] < 0 or node["y"] > config.boundary_height:
                                node["velocity_y"] *= -1
                            node["x"] = max(0.0, min(config.boundary_width, node["x"]))
                            node["y"] = max(0.0, min(config.boundary_height, node["y"]))
                        else:
                            node["x"] = max(0.0, min(config.boundary_width, node["x"]))
                            node["y"] = max(0.0, min(config.boundary_height, node["y"]))
                        node["energy"] = clamp(node["energy"] * config.energy_decay)
                        node["local_activation"] = clamp(
                            node["local_activation"] * config.activation_decay
                        )
                for node in nodes:
                    support = self._query_support(node, query)
                    node["stored_strength"] = clamp(
                        node["stored_strength"] * config.stored_strength_decay + support * 0.01
                    )
                    node["local_stability"] = clamp(
                        node["local_stability"]
                        + config.retention_learning_rate * (node["local_activation"] - 0.35)
                    )
                    node["retention"] = clamp(
                        node["stored_strength"]
                        * node["local_stability"]
                        * max(0.05, node["local_activation"])
                    )
                    node["age_steps"] += 1
                    if (
                        node["retention"] < config.weak_node_threshold
                        and node["eviction_status"] == "ACTIVE"
                    ):
                        node["eviction_status"] = "WEAKENING"
                        node["weakening_steps"] = 1
                    elif (
                        node["eviction_status"] == "WEAKENING"
                        and node["retention"] >= config.weak_node_threshold
                    ):
                        node["eviction_status"] = "ACTIVE"
                        node["weakening_steps"] = 0
                    elif node["eviction_status"] == "WEAKENING":
                        node["weakening_steps"] += 1
                    if (
                        node["eviction_status"] == "WEAKENING"
                        and node["weakening_steps"] >= config.eviction_confirmation_steps
                    ):
                        node["eviction_status"] = "EVICTION_CANDIDATE"
                    if (
                        node["eviction_status"] == "EVICTION_CANDIDATE"
                        and node["retention"] < config.eviction_threshold
                    ):
                        node["eviction_status"] = "EVICTED"
                        step_events.append(
                            {"event_type": "EVICTED", "placement_id": node["placement_id"]}
                        )
                    if node["eviction_status"] in {"WEAKENING", "EVICTION_CANDIDATE"}:
                        step_events.append(
                            {
                                "event_type": "WEAKENING",
                                "placement_id": node["placement_id"],
                                "retention": node["retention"],
                            }
                        )
                    self._upsert_node(conn, node)
                    conn.execute(
                        "UPDATE cloud_placements SET x=?, y=?, local_activation=?, local_gravity=?, updated_at=? WHERE id=? AND space_id=?",
                        (
                            node["x"],
                            node["y"],
                            node["local_activation"],
                            node["local_gravity"],
                            utcnow(),
                            node["placement_id"],
                            hive["space_id"],
                        ),
                    )
                    conn.execute(
                        "UPDATE hive_cells SET local_activation=?, retention=?, stored_strength=?, updated_at=? WHERE hive_placement_id=?",
                        (
                            node["local_activation"],
                            node["retention"],
                            node["stored_strength"],
                            utcnow(),
                            node["placement_id"],
                        ),
                    )
                current = self._state(nodes, temperature)
                convergence = sum(
                    abs(float(a.get("local_activation", 0)) - float(b.get("local_activation", 0)))
                    + abs(float(a.get("x", 0)) - float(b.get("x", 0))) / 1000
                    for a, b in zip(before["nodes"], current["nodes"])
                ) / max(1, len(nodes))
                temperature *= config.temperature_decay
                completed = step
                clusters = self._clusters(conn, run_id, hive_id, step, nodes)
                step_events.append(
                    {"event_type": "SETTLE", "placement_id": None, "state_delta": convergence}
                )
                snapshots.append(
                    self._snapshot(
                        conn,
                        run_id,
                        hive_id,
                        step,
                        "AFTER_SETTLE",
                        nodes,
                        temperature,
                        {"state_delta": convergence},
                        clusters,
                        step_events,
                    )
                )
                for event in step_events:
                    event_row = {
                        "id": f"event-{uuid.uuid4().hex}",
                        "run_id": run_id,
                        "hive_id": hive_id,
                        "step": step,
                        "phase": "AFTER_SETTLE",
                        "created_at": utcnow(),
                        **event,
                    }
                    conn.execute(
                        "INSERT INTO hive_reasoning_events (id,run_id,hive_id,step,phase,event_type,placement_id,payload_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            event_row["id"],
                            run_id,
                            hive_id,
                            step,
                            "AFTER_SETTLE",
                            event_row["event_type"],
                            event_row.get("placement_id"),
                            encode(event_row),
                            event_row["created_at"],
                        ),
                    )
                    events.append(event_row)
                if convergence <= config.convergence_threshold:
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= config.convergence_patience:
                    break
            stop_reason = (
                "CONVERGED"
                if stable_count >= config.convergence_patience
                else ("NO_STEPS" if completed == 0 else "COMPLETED")
            )
            final = snapshots[-1]
            conn.execute(
                "UPDATE hive_reasoning_runs SET completed_steps=?, stop_reason=?, initial_state_hash=?, final_state_hash=?, completed_at=? WHERE id=?",
                (
                    completed,
                    stop_reason,
                    initial["state_hash"],
                    final["state_hash"],
                    utcnow(),
                    run_id,
                ),
            )
            if completed:
                conn.execute(
                    "UPDATE hives SET reasoning_step=reasoning_step+?, current_temperature=?, total_energy=?, last_reasoned_at=?, status='STABLE', updated_at=? WHERE id=?",
                    (
                        completed,
                        temperature,
                        sum(float(node["energy"]) for node in nodes),
                        utcnow(),
                        utcnow(),
                        hive_id,
                    ),
                )
            run = dict(
                conn.execute("SELECT * FROM hive_reasoning_runs WHERE id=?", (run_id,)).fetchone()
            )
            run["query"] = query.to_dict()
            run["config"] = asdict(config)
            return VibrationResult(
                run,
                initial["state"],
                final["state"],
                snapshots,
                events,
                clusters,
                [
                    int(node["placement_id"])
                    for node in nodes
                    if node["eviction_status"] == "EVICTED"
                ],
                convergence,
                stop_reason,
            )

    def restore(self, hive_id: str, run_id: str, step: int) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            hive = conn.execute("SELECT * FROM hives WHERE id=?", (hive_id,)).fetchone()
            snapshot = conn.execute(
                "SELECT state_json FROM hive_reasoning_snapshots WHERE run_id=? AND hive_id=? AND step=? AND phase=?",
                (run_id, hive_id, step, "INITIAL" if step == 0 else "AFTER_SETTLE"),
            ).fetchone()
            if not hive or not snapshot:
                raise KeyError(run_id)
            state = json.loads(snapshot["state_json"])
            for saved in state.get("nodes", []):
                current = conn.execute(
                    "SELECT * FROM hive_node_states WHERE hive_id=? AND placement_id=?",
                    (hive_id, saved["placement_id"]),
                ).fetchone()
                if not current:
                    continue
                node = dict(current)
                for key, value in saved.items():
                    if key in node:
                        node[key] = value
                self._upsert_node(conn, node)
                conn.execute(
                    "UPDATE cloud_placements SET x=?, y=?, local_activation=?, local_gravity=?, updated_at=? WHERE id=? AND space_id=?",
                    (
                        node["x"],
                        node["y"],
                        node["local_activation"],
                        node["local_gravity"],
                        utcnow(),
                        node["placement_id"],
                        hive["space_id"],
                    ),
                )
                conn.execute(
                    "UPDATE hive_cells SET local_activation=?, retention=?, stored_strength=?, updated_at=? WHERE hive_placement_id=?",
                    (
                        node["local_activation"],
                        node["retention"],
                        node["stored_strength"],
                        utcnow(),
                        node["placement_id"],
                    ),
                )
            conn.execute(
                "UPDATE hives SET reasoning_step=?, current_temperature=?, total_energy=?, status='STABLE', updated_at=? WHERE id=?",
                (
                    step,
                    state.get("temperature", 1.0),
                    sum(float(node.get("energy", 0)) for node in state.get("nodes", [])),
                    utcnow(),
                    hive_id,
                ),
            )
            return {
                "hive_id": hive_id,
                "run_id": run_id,
                "step": step,
                "state": state,
                "restored": True,
            }
