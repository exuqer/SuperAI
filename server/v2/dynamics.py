"""Explainable physical dynamics for the working hive."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .repository import V2Repository, decode, encode, utcnow
from .admission import DynamicsAdmissionService, REASONING_CLASSES


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _vector(x: float, y: float, magnitude: float) -> Tuple[float, float]:
    length = math.hypot(x, y)
    if length <= 1e-9:
        return 0.0, 0.0
    scale = magnitude / length
    return x * scale, y * scale


class DynamicsConsistencyValidator:
    """Checks that exported snapshots have one physically consistent truth."""

    tolerance = 2e-5

    def validate(self, state: "DynamicsState") -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        for node in state.nodes:
            force_x = sum(_number(item.get("x")) for item in node.force_breakdown)
            force_y = sum(_number(item.get("y")) for item in node.force_breakdown)
            if not math.isclose(force_x, node.net_force_x, abs_tol=self.tolerance) or not math.isclose(force_y, node.net_force_y, abs_tol=self.tolerance):
                errors.append({"type": "DynamicsConsistencyError", "cell_id": node.cell_id, "rule": "net_force must equal force breakdown"})
            if not math.isclose(math.hypot(node.net_force_x, node.net_force_y), node.to_dict()["net_force"]["magnitude"], abs_tol=self.tolerance):
                errors.append({"type": "DynamicsConsistencyError", "cell_id": node.cell_id, "rule": "net force magnitude"})
            mass = max(node.local_mass, .05)
            if not math.isclose(node.acceleration_x, node.net_force_x / mass, abs_tol=.01) or not math.isclose(node.acceleration_y, node.net_force_y / mass, abs_tol=.01):
                errors.append({"type": "DynamicsConsistencyError", "cell_id": node.cell_id, "rule": "acceleration must equal net force / mass"})
        return errors


@dataclass
class TemperatureConfig:
    default: float = 0.35
    minimum: float = 0.05
    maximum: float = 1.0
    cooling_rate: float = 0.72
    reheat_on_stagnation: float = 0.12

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class ForceConfig:
    query_force: float = 0.25
    role_force: float = 0.30
    semantic_force: float = 0.20
    scene_force: float = 0.18
    resonance_force: float = 0.16
    competition_force: float = 0.14
    capacity_force: float = 0.12
    thermal_force: float = 0.04
    boundary_force: float = 0.20

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class MotionConfig:
    damping: float = 0.82
    max_velocity: float = 18.0
    max_acceleration: float = 8.0
    minimum_distance: float = 0.04

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class EvictionConfig:
    warning_threshold: float = 0.45
    drifting_threshold: float = 0.62
    boundary_threshold: float = 0.78
    evict_threshold: float = 0.90
    minimum_weak_steps: int = 2

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class ZoneConfig:
    core_radius: float = 0.14
    active_radius: float = 0.28
    candidate_radius: float = 0.42
    eviction_radius: float = 0.54
    outer_radius: float = 0.62

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class DynamicsConfig:
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    forces: ForceConfig = field(default_factory=ForceConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    eviction: EvictionConfig = field(default_factory=EvictionConfig)
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    local_mass_weights: Dict[str, float] = field(default_factory=lambda: {
        "global_mass": 0.20,
        "query_relevance": 0.25,
        "role_compatibility": 0.20,
        "source_confidence": 0.15,
        "semantic_support": 0.10,
        "structural_support": 0.10,
    })
    eviction_weights: Dict[str, float] = field(default_factory=lambda: {
        "activation": 0.22,
        "retention": 0.23,
        "resonance": 0.18,
        "distance": 0.12,
        "capacity": 0.15,
        "contradiction": 0.10,
    })
    random_seed: int = 0
    max_history: int = 128
    max_trajectory: int = 32

    def __post_init__(self) -> None:
        for name, target in (("temperature", TemperatureConfig), ("forces", ForceConfig), ("motion", MotionConfig), ("eviction", EvictionConfig), ("zones", ZoneConfig)):
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, target(**value))

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> "DynamicsConfig":
        value = value or {}
        defaults = cls()

        def section(name: str, target: Any) -> Any:
            raw = value.get(name)
            if not isinstance(raw, dict):
                return target
            for key in asdict(target):
                if key in raw:
                    setattr(target, key, raw[key])
            return target

        section("temperature", defaults.temperature)
        section("forces", defaults.forces)
        section("motion", defaults.motion)
        section("eviction", defaults.eviction)
        section("zones", defaults.zones)
        if isinstance(value.get("local_mass_weights"), dict):
            defaults.local_mass_weights.update(value["local_mass_weights"])
        if isinstance(value.get("eviction_weights"), dict):
            defaults.eviction_weights.update(value["eviction_weights"])
        defaults.random_seed = int(value.get("random_seed", defaults.random_seed) or 0)
        defaults.max_history = max(1, int(value.get("max_history", defaults.max_history)))
        defaults.max_trajectory = max(2, int(value.get("max_trajectory", defaults.max_trajectory)))
        defaults.temperature.minimum = clamp(defaults.temperature.minimum)
        defaults.temperature.maximum = max(defaults.temperature.minimum, clamp(defaults.temperature.maximum))
        defaults.temperature.default = clamp(defaults.temperature.default, defaults.temperature.minimum, defaults.temperature.maximum)
        defaults.temperature.cooling_rate = clamp(defaults.temperature.cooling_rate)
        return defaults

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DynamicsNodeState:
    cell_id: str
    label: str = ""
    node_type: str = "role_candidate"
    role: Optional[str] = None
    position_x: float = 0.5
    position_y: float = 0.5
    previous_position_x: float = 0.5
    previous_position_y: float = 0.5
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0
    global_mass: float = 0.5
    global_mass_raw: float = 0.5
    global_mass_normalized: float = 0.5
    local_mass: float = 0.5
    activation: float = 0.0
    retention: float = 0.5
    resonance: float = 0.0
    local_gravity: float = 0.0
    energy: float = 0.0
    mobility: float = 1.0
    contradiction: float = 0.0
    query_relevance: float = 0.0
    role_compatibility: float = 0.0
    source_confidence: float = 0.0
    semantic_support: float = 0.0
    structural_support: float = 0.0
    net_force_x: float = 0.0
    net_force_y: float = 0.0
    distance_to_core: float = 0.0
    distance_to_target: float = 0.0
    eviction_score: float = 0.0
    eviction_status: str = "ACTIVE"
    zone: str = "ACTIVE_ZONE"
    created_at_step: int = 0
    grace_steps: int = 1
    weak_steps: int = 0
    resolved_role: Optional[str] = None
    competition_group_id: Optional[str] = None
    query_scene_id: Optional[str] = None
    candidate_status: Optional[str] = None
    admission: Dict[str, Any] = field(default_factory=dict)
    force_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def position(self) -> Dict[str, float]:
        return {"x": self.position_x, "y": self.position_y}

    @property
    def velocity(self) -> Dict[str, float]:
        return {"x": self.velocity_x, "y": self.velocity_y}

    @property
    def acceleration(self) -> Dict[str, float]:
        return {"x": self.acceleration_x, "y": self.acceleration_y}

    @property
    def mass(self) -> Dict[str, float]:
        return {"global": self.global_mass_normalized, "global_raw": self.global_mass_raw, "local": self.local_mass}

    @property
    def gravity(self) -> float:
        return self.local_gravity

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["position"] = self.position
        value["previous_position"] = {"x": self.previous_position_x, "y": self.previous_position_y}
        value["velocity"] = self.velocity
        value["acceleration"] = self.acceleration
        value["mass"] = self.mass
        value["gravity"] = self.local_gravity
        value["local_activation"] = self.activation
        value["local_gravity"] = self.local_gravity
        value["net_force"] = {"x": self.net_force_x, "y": self.net_force_y, "magnitude": math.hypot(self.net_force_x, self.net_force_y)}
        for key in ("position_x", "position_y", "previous_position_x", "previous_position_y", "velocity_x", "velocity_y", "acceleration_x", "acceleration_y", "net_force_x", "net_force_y"):
            value.pop(key, None)
        return value


@dataclass
class DynamicsState:
    version: int = 1
    step: int = 0
    status: str = "READY"
    initial_temperature: float = 0.42
    current_temperature: float = 0.42
    minimum_temperature: float = 0.05
    maximum_temperature: float = 1.0
    cooling_rate: float = 0.72
    temperature_state: str = "EXPLORATION"
    temperature_history: List[Dict[str, Any]] = field(default_factory=list)
    capacity_pressure: float = 0.0
    center_of_mass_x: float = 0.5
    center_of_mass_y: float = 0.5
    zones: Dict[str, Any] = field(default_factory=dict)
    anchors: List[Dict[str, Any]] = field(default_factory=list)
    nodes: List[DynamicsNodeState] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    eviction_history: List[Dict[str, Any]] = field(default_factory=list)
    random_seed: int = 0
    semantic_reasoning_step: int = 0
    physical_step: int = 0

    @property
    def temperature(self) -> Dict[str, Any]:
        if self.status in {"IDLE", "NOT_AVAILABLE"}:
            return {"status": "NOT_STARTED", "initial": None, "current": None, "minimum": self.minimum_temperature, "maximum": self.maximum_temperature, "cooling_rate": self.cooling_rate, "history": []}
        return {
            "status": "STARTED",
            "initial": self.initial_temperature,
            "current": self.current_temperature,
            "minimum": self.minimum_temperature,
            "maximum": self.maximum_temperature,
            "cooling_rate": self.cooling_rate,
            "state": self.temperature_state,
            "history": self.temperature_history,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "step": self.step,
            "status": self.status,
            "temperature": self.temperature,
            "capacity_pressure": self.capacity_pressure,
            "center_of_mass": {"x": self.center_of_mass_x, "y": self.center_of_mass_y},
            "zones": self.zones,
            "anchors": self.anchors,
            "nodes": [node.to_dict() for node in self.nodes],
            "history": self.history,
            "eviction_history": self.eviction_history,
            "random_seed": self.random_seed,
            "semantic_reasoning_step": self.semantic_reasoning_step,
            "physical_step": self.physical_step,
        }

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> "DynamicsState":
        value = value or {}
        temperature = value.get("temperature") or {}
        center = value.get("center_of_mass") or {}
        nodes: List[DynamicsNodeState] = []
        for raw in value.get("nodes") or []:
            raw = dict(raw)
            position = raw.pop("position", {}) or {}
            previous = raw.pop("previous_position", {}) or {}
            velocity = raw.pop("velocity", {}) or {}
            acceleration = raw.pop("acceleration", {}) or {}
            mass = raw.pop("mass", {}) or {}
            net = raw.pop("net_force", {}) or {}
            gravity_value = raw.pop("gravity", None)
            raw.update({
                "position_x": raw.pop("position_x", position.get("x", 0.5)),
                "position_y": raw.pop("position_y", position.get("y", 0.5)),
                "previous_position_x": raw.pop("previous_position_x", previous.get("x", position.get("x", 0.5))),
                "previous_position_y": raw.pop("previous_position_y", previous.get("y", position.get("y", 0.5))),
                "velocity_x": raw.pop("velocity_x", velocity.get("x", 0.0)),
                "velocity_y": raw.pop("velocity_y", velocity.get("y", 0.0)),
                "acceleration_x": raw.pop("acceleration_x", acceleration.get("x", 0.0)),
                "acceleration_y": raw.pop("acceleration_y", acceleration.get("y", 0.0)),
                "global_mass": raw.pop("global_mass", mass.get("global", 0.5)),
                "global_mass_raw": raw.pop("global_mass_raw", mass.get("global_raw", mass.get("global", 0.5))),
                "global_mass_normalized": raw.pop("global_mass_normalized", mass.get("global", 0.5)),
                "local_mass": raw.pop("local_mass", mass.get("local", 0.5)),
                "local_gravity": raw.pop("local_gravity", gravity_value if gravity_value is not None else 0.0),
                "net_force_x": raw.pop("net_force_x", net.get("x", 0.0)),
                "net_force_y": raw.pop("net_force_y", net.get("y", 0.0)),
            })
            allowed = {field_name for field_name in DynamicsNodeState.__dataclass_fields__}
            nodes.append(DynamicsNodeState(**{key: raw[key] for key in allowed if key in raw}))
        return cls(
            version=int(value.get("version", 1)), step=int(value.get("step", 0)), status=str(value.get("status", "READY")),
            initial_temperature=_number(temperature.get("initial", value.get("initial_temperature", 0.42)), 0.42),
            current_temperature=_number(temperature.get("current", value.get("current_temperature", 0.42)), 0.42),
            minimum_temperature=_number(temperature.get("minimum", 0.05), 0.05), maximum_temperature=_number(temperature.get("maximum", 1.0), 1.0),
            cooling_rate=_number(temperature.get("cooling_rate", 0.72), 0.72), temperature_state=str(temperature.get("state", "EXPLORATION")),
            temperature_history=list(temperature.get("history") or []), capacity_pressure=_number(value.get("capacity_pressure")),
            center_of_mass_x=_number(center.get("x", 0.5), 0.5), center_of_mass_y=_number(center.get("y", 0.5), 0.5),
            zones=dict(value.get("zones") or {}), anchors=list(value.get("anchors") or []), nodes=nodes,
            history=list(value.get("history") or []), eviction_history=list(value.get("eviction_history") or []), random_seed=int(value.get("random_seed", 0) or 0),
            semantic_reasoning_step=int(value.get("semantic_reasoning_step", value.get("step", 0)) or 0), physical_step=int(value.get("physical_step", value.get("step", 0)) or 0),
        )


class DynamicsEngine:
    """Pure deterministic integrator for normalized hive coordinates."""

    def __init__(self, config: Optional[DynamicsConfig] = None) -> None:
        self.config = config or DynamicsConfig()

    @staticmethod
    def temperature_state(value: float) -> str:
        if value <= 0.10:
            return "FROZEN"
        if value <= 0.30:
            return "STABILIZING"
        if value <= 0.60:
            return "EXPLORATION"
        return "UNSTABLE"

    def local_mass(self, node: DynamicsNodeState) -> float:
        weights = self.config.local_mass_weights
        value = (
            node.global_mass_normalized * weights["global_mass"]
            + node.query_relevance * weights["query_relevance"]
            + node.role_compatibility * weights["role_compatibility"]
            + node.source_confidence * weights["source_confidence"]
            + node.semantic_support * weights["semantic_support"]
            + node.structural_support * weights["structural_support"]
        )
        return clamp(value)

    @staticmethod
    def gravity(node: DynamicsNodeState) -> float:
        return clamp(node.local_mass * (0.25 + node.activation * 0.35) * (0.40 + node.retention * 0.30) * (0.40 + node.resonance * 0.30))

    def _force(self, force_type: str, x: float, y: float, reason: str, source: str = "hive", target: str = "") -> Dict[str, Any]:
        return {"type": force_type, "source": source, "target": target, "x": round(x, 6), "y": round(y, 6), "magnitude": round(math.hypot(x, y), 6), "reason": reason}

    def _anchor(self, state: DynamicsState, node: DynamicsNodeState) -> Optional[Dict[str, Any]]:
        if node.role:
            for anchor in state.anchors:
                if str(anchor.get("role", "")).casefold() == str(node.role).casefold():
                    return anchor
        return None

    def _zone(self, distance: float) -> str:
        z = self.config.zones
        if distance <= z.core_radius:
            return "CORE"
        if distance <= z.active_radius:
            return "ACTIVE_ZONE"
        if distance <= z.candidate_radius:
            return "CANDIDATE_ORBIT"
        if distance <= z.eviction_radius:
            return "WEAK_ZONE"
        if distance <= z.outer_radius:
            return "EVICTION_ZONE"
        return "OUTSIDE"

    def _eviction(self, node: DynamicsNodeState, pressure: float, distance: float) -> float:
        weights = self.config.eviction_weights
        normalized_distance = clamp(distance / max(self.config.zones.outer_radius, 1e-9))
        return clamp(
            (1 - node.activation) * weights["activation"]
            + (1 - node.retention) * weights["retention"]
            + (1 - node.resonance) * weights["resonance"]
            + normalized_distance * weights["distance"]
            + pressure * weights["capacity"]
            + node.contradiction * weights["contradiction"]
        )

    def _status(self, node: DynamicsNodeState, distance: float) -> str:
        e = self.config.eviction
        if node.resolved_role or node.node_type in {"winner", "pinned"}:
            return "PINNED"
        if node.created_at_step + node.grace_steps >= node._step if hasattr(node, "_step") else False:
            return "ACTIVE"
        if node.eviction_score < e.warning_threshold:
            return "RECOVERING" if node.eviction_status == "RECOVERING" else "ACTIVE"
        if node.eviction_score < e.drifting_threshold:
            return "WEAKENING"
        if distance > e.boundary_threshold or node.eviction_score >= e.boundary_threshold:
            return "AT_BOUNDARY" if node.eviction_score >= e.boundary_threshold else "DRIFTING_OUT"
        return "DRIFTING_OUT" if node.weak_steps >= e.minimum_weak_steps else "WEAKENING"

    def step(self, state: DynamicsState, semantic_scores: Optional[Dict[str, Dict[str, Any]]] = None, *, stagnated: bool = False) -> DynamicsState:
        cfg = self.config
        active = [node for node in state.nodes if node.eviction_status != "EVICTED"]
        reasoning = [node for node in active if node.node_type in REASONING_CLASSES]
        if not active:
            state.status = "IDLE"
            return state
        state.step += 1
        state.physical_step += 1
        state.status = "CALCULATING_FORCES"
        rng = random.Random(int(state.random_seed if state.random_seed is not None else cfg.random_seed) + state.step)
        state.capacity_pressure = clamp((len(reasoning) / 24) ** 2)
        center_x = sum(node.position_x for node in active) / len(active) if active else 0.5
        center_y = sum(node.position_y for node in active) / len(active) if active else 0.5
        state.center_of_mass_x, state.center_of_mass_y = center_x, center_y

        for node in active:
            if semantic_scores and node.cell_id in semantic_scores:
                score = semantic_scores[node.cell_id]
                for attr, key in (("activation", "activation"), ("retention", "retention"), ("resonance", "resonance"), ("role_compatibility", "role_compatibility"), ("query_relevance", "query_relevance"), ("source_confidence", "source_confidence"), ("semantic_support", "semantic_support"), ("structural_support", "structural_support"), ("contradiction", "contradiction")):
                    if key in score:
                        setattr(node, attr, clamp(score[key]))
            node.local_mass = self.local_mass(node)
            node.local_gravity = self.gravity(node)
            dx, dy = 0.5 - node.position_x, 0.5 - node.position_y
            breakdown: List[Dict[str, Any]] = []
            qx = qy = 0.0
            if node.query_relevance > 0:
                qx, qy = _vector(dx, dy, node.query_relevance * node.activation * cfg.forces.query_force)
                breakdown.append(self._force("QUERY_FORCE", qx, qy, "притяжение к смысловому ядру запроса", "query-core", node.cell_id))
            rx = ry = 0.0
            anchor = self._anchor(state, node)
            if anchor:
                ax = _number(anchor.get("position", {}).get("x"), 0.5) - node.position_x
                ay = _number(anchor.get("position", {}).get("y"), 0.5) - node.position_y
                rx, ry = _vector(ax, ay, node.role_compatibility * cfg.forces.role_force)
                breakdown.append(self._force("ROLE_FORCE", rx, ry, f"совместимость с ролью {str(node.role).upper()}", str(anchor.get("anchor_id", "role-anchor")), node.cell_id))
                node.distance_to_target = math.hypot(ax, ay)
            else:
                node.distance_to_target = math.hypot(dx, dy)
            sx = sy = scx = scy = rcx = rcy = 0.0
            related = [other for other in active if other is not node and other.query_scene_id == node.query_scene_id and other.node_type not in {"function_operator", "search_hit", "inspection_projection"}]
            for other in related:
                ox, oy = other.position_x - node.position_x, other.position_y - node.position_y
                semantic_strength = min(node.semantic_support, other.semantic_support) * min(node.activation, other.activation) * cfg.forces.semantic_force
                fx, fy = _vector(ox, oy, semantic_strength)
                sx += fx
                sy += fy
                scene_strength = min(node.source_confidence, other.source_confidence) * min(node.role_compatibility, other.role_compatibility) * cfg.forces.scene_force
                fx, fy = _vector(ox, oy, scene_strength)
                scx += fx
                scy += fy
                resonance_strength = other.activation * min(node.resonance, 1.0) * cfg.forces.resonance_force
                fx, fy = _vector(ox, oy, resonance_strength)
                rcx += fx
                rcy += fy
            if node.semantic_support and (sx or sy):
                breakdown.append(self._force("SEMANTIC_FORCE", sx, sy, "смысловая связь с совместимыми узлами", "semantic-context", node.cell_id))
            if node.source_confidence and (scx or scy):
                breakdown.append(self._force("SCENE_FORCE", scx, scy, "сцена поддерживает требуемую роль", "memory-scene", node.cell_id))
            if rcx or rcy:
                breakdown.append(self._force("RESONANCE_FORCE", rcx, rcy, "поддержка активных совместимых соседей", "resonance", node.cell_id))
            cx = cy = 0.0
            group = node.competition_group_id or (f"implicit:{node.role}" if node.role and node.node_type == "role_candidate" else None)
            for other in active:
                other_group = other.competition_group_id or (f"implicit:{other.role}" if other.role and other.node_type == "role_candidate" else None)
                if other is node or not group or group != other_group:
                    continue
                ox, oy = node.position_x - other.position_x, node.position_y - other.position_y
                distance = max(0.025, math.hypot(ox, oy))
                px, py = _vector(ox, oy, cfg.forces.competition_force * (1 - node.activation) / distance)
                cx += px
                cy += py
            if cx or cy:
                breakdown.append(self._force("COMPETITION_FORCE", cx, cy, "конкуренция кандидатов одной роли", "competition", node.cell_id))
            kx = ky = 0.0
            if state.capacity_pressure > 0:
                kx, ky = _vector(dx, dy, state.capacity_pressure * cfg.forces.capacity_force * (1 - node.retention))
                breakdown.append(self._force("CAPACITY_FORCE", kx, ky, "давление заполненной рабочей памяти", "capacity", node.cell_id))
            instability = clamp((1 - node.retention) * .45 + (1 - node.activation) * .30 + (1 - node.resonance) * .30 - node.local_mass * .25)
            thermal_x = (rng.random() * 2 - 1) * state.current_temperature * instability * cfg.forces.thermal_force
            thermal_y = (rng.random() * 2 - 1) * state.current_temperature * instability * cfg.forces.thermal_force
            if thermal_x or thermal_y:
                breakdown.append(self._force("THERMAL_FORCE", thermal_x, thermal_y, "температурное колебание", "temperature", node.cell_id))
            bx = by = 0.0
            distance = math.hypot(node.position_x - 0.5, node.position_y - 0.5)
            if distance > cfg.zones.active_radius:
                bx, by = _vector(0.5 - node.position_x, 0.5 - node.position_y, min(cfg.forces.boundary_force, distance))
                breakdown.append(self._force("BOUNDARY_FORCE", bx, by, "возврат из внешней зоны", "hive-boundary", node.cell_id))
            total_x = qx + rx + sx + scx + rcx + cx + kx + thermal_x + (bx if distance > cfg.zones.active_radius else 0.0)
            total_y = qy + ry + sy + scy + rcy + cy + ky + thermal_y + (by if distance > cfg.zones.active_radius else 0.0)
            damping_x, damping_y = -node.velocity_x * (1 - cfg.motion.damping), -node.velocity_y * (1 - cfg.motion.damping)
            breakdown.append(self._force("DAMPING_FORCE", damping_x, damping_y, "затухание движения", "inertia", node.cell_id))
            total_x += damping_x
            total_y += damping_y
            raw_magnitude = math.hypot(total_x, total_y)
            magnitude = min(cfg.motion.max_acceleration, raw_magnitude)
            if raw_magnitude > cfg.motion.max_acceleration:
                scale = cfg.motion.max_acceleration / raw_magnitude
                for force in breakdown:
                    force["x"] = round(float(force["x"]) * scale, 6)
                    force["y"] = round(float(force["y"]) * scale, 6)
                    force["magnitude"] = round(math.hypot(force["x"], force["y"]), 6)
                total_x = sum(float(force["x"]) for force in breakdown)
                total_y = sum(float(force["y"]) for force in breakdown)
            else:
                total_x, total_y = _vector(total_x, total_y, magnitude)
            node.net_force_x, node.net_force_y = total_x, total_y
            node.force_breakdown = breakdown
            node.previous_position_x, node.previous_position_y = node.position_x, node.position_y
            node.acceleration_x, node.acceleration_y = total_x / max(node.local_mass, 0.05), total_y / max(node.local_mass, 0.05)
            node.velocity_x = node.velocity_x * cfg.motion.damping + node.acceleration_x * node.mobility * 0.01
            node.velocity_y = node.velocity_y * cfg.motion.damping + node.acceleration_y * node.mobility * 0.01
            velocity = min(cfg.motion.max_velocity, math.hypot(node.velocity_x, node.velocity_y))
            node.velocity_x, node.velocity_y = _vector(node.velocity_x, node.velocity_y, velocity)
            node.position_x = clamp(node.position_x + node.velocity_x * 0.01, 0.0, 1.0)
            node.position_y = clamp(node.position_y + node.velocity_y * 0.01, 0.0, 1.0)
            node.distance_to_core = math.hypot(node.position_x - 0.5, node.position_y - 0.5)
            node.zone = self._zone(node.distance_to_core)
            node.eviction_score = self._eviction(node, state.capacity_pressure, node.distance_to_core)
            node.weak_steps = node.weak_steps + 1 if node.eviction_score >= cfg.eviction.warning_threshold else 0
            node._step = state.step
            old_status = node.eviction_status
            node.eviction_status = self._status(node, node.distance_to_core)
            if old_status in {"DRIFTING_OUT", "AT_BOUNDARY"} and node.eviction_score < cfg.eviction.warning_threshold:
                node.eviction_status = "RECOVERING"
            if node.eviction_status == "RECOVERING" and node.activation > 0.5 and node.retention > 0.5:
                node.eviction_status = "ACTIVE"
            if node.eviction_status == "AT_BOUNDARY" and node.eviction_score >= cfg.eviction.evict_threshold and node.weak_steps >= cfg.eviction.minimum_weak_steps:
                node.eviction_status = "EVICTED"
                state.eviction_history.append({"cell_id": node.cell_id, "label": node.label, "step": state.step, "reason": "низкая поддержка и давление рабочей памяти", "final": {"activation": node.activation, "retention": node.retention, "eviction_score": node.eviction_score}})
            node.trajectory.append({"step": state.step, "x": node.position_x, "y": node.position_y})
            node.trajectory = node.trajectory[-cfg.max_trajectory:]

        next_temperature = max(cfg.temperature.minimum, state.current_temperature * cfg.temperature.cooling_rate)
        if stagnated:
            next_temperature = min(cfg.temperature.maximum, state.current_temperature + cfg.temperature.reheat_on_stagnation)
        state.current_temperature = clamp(next_temperature, cfg.temperature.minimum, cfg.temperature.maximum)
        state.temperature_state = self.temperature_state(state.current_temperature)
        state.temperature_history.append({"step": state.step, "value": state.current_temperature})
        state.temperature_history = state.temperature_history[-cfg.max_history:]
        consistency_errors = DynamicsConsistencyValidator().validate(state)
        snapshot = {"step": state.step, "temperature": state.current_temperature, "nodes": [node.to_dict() for node in state.nodes], "consistency_errors": consistency_errors}
        state.history.append(snapshot)
        state.history = state.history[-cfg.max_history:]
        state.status = "STABILIZING" if state.temperature_state == "STABILIZING" else "STABLE"
        state.semantic_reasoning_step += 1 if any(node.query_relevance or node.semantic_support or node.role_compatibility for node in reasoning) else 0
        return state


class HiveDynamicsService:
    """Persistence adapter for the physical state of a query working hive."""

    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()

    @staticmethod
    def _metadata(conn: Any, hive_id: str) -> Dict[str, Any]:
        row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
        if not row:
            raise KeyError(hive_id)
        return decode(row["metadata_json"], {})

    @staticmethod
    def _save(conn: Any, hive_id: str, metadata: Dict[str, Any]) -> None:
        conn.execute("UPDATE hives SET metadata_json=?, updated_at=? WHERE id=?", (encode(metadata), utcnow(), hive_id))

    def _initial(self, conn: Any, hive_id: str, state: Dict[str, Any], config: DynamicsConfig) -> DynamicsState:
        existing = DynamicsState.from_dict(state.get("dynamics"))
        query_scene = state.get("query_scene") or {}
        has_reasoning_input = bool(query_scene)
        if state.get("dynamics") and not (has_reasoning_input and (existing.status == "IDLE" or not existing.nodes)):
            return existing
        if not query_scene:
            return DynamicsState(status="IDLE", initial_temperature=config.temperature.default, current_temperature=config.temperature.default, minimum_temperature=config.temperature.minimum, maximum_temperature=config.temperature.maximum, cooling_rate=config.temperature.cooling_rate, temperature_state="NOT_STARTED", zones=asdict(config.zones), random_seed=config.random_seed)
        slots = query_scene.get("slots") or []
        roles = [str(slot.get("role")) for slot in slots if slot.get("role")]
        anchors = []
        for index, role in enumerate(roles):
            angle = (index / max(1, len(roles))) * math.tau - math.pi / 2
            anchors.append({"anchor_id": f"slot-{role}", "role": role, "position": {"x": 0.5 + math.cos(angle) * 0.18, "y": 0.5 + math.sin(angle) * 0.18}, "attraction_radius": 0.25, "resolved": any(slot.get("role") == role and str(slot.get("status", "")).lower() == "resolved" for slot in slots)})
        cells = [dict(row) for row in conn.execute("""SELECT hc.id, hc.hive_placement_id, hc.dominant_cloud_id, hc.component_class, hc.metadata_json, hc.local_activation, hc.retention, hc.stored_strength, c.mass AS global_mass, c.canonical_name, p.x, p.y FROM hive_cells hc JOIN clouds c ON c.id=hc.dominant_cloud_id JOIN cloud_placements p ON p.id=hc.hive_placement_id WHERE hc.hive_id=? ORDER BY hc.id""", (hive_id,))]
        candidates = {str(item.get("id")): item for item in state.get("candidates") or []}
        nodes = []
        admission_service = DynamicsAdmissionService()
        raw_masses = [_number(cell.get("global_mass"), _number(cell.get("stored_strength"), 0.5)) for cell in cells]
        min_mass, max_mass = min(raw_masses, default=0.0), max(raw_masses, default=0.0)
        for index, cell in enumerate(cells):
            metadata = decode(cell.get("metadata_json"), {})
            candidate = next((item for item in candidates.values() if str(item.get("cell_id", "")) == str(cell["id"]) or str(item.get("placement_id", "")) == str(cell["hive_placement_id"]) or str(item.get("id", "")) == str(metadata.get("candidate_id", ""))), {})
            scores = candidate.get("scores") or {}
            if cell.get("component_class") == "memory_source":
                scores = {"query_relevance": .5, "role_compatibility": .8, "source_confidence": _number(cell.get("local_activation"), .7), "semantic_support": .7, "structural_support": .7, "activation": _number(cell.get("local_activation"), .7), "retention": _number(cell.get("retention"), .7), **scores}
            role = candidate.get("role") or candidate.get("target_role") or next((slot.get("role") for slot in slots if str(slot.get("status", "")).lower() == "empty"), None)
            admission_item = {**cell, "metadata": metadata, "id": cell["id"], "component_class": cell.get("component_class"), "query_session_id": metadata.get("query_session_id"), "query_relevance": scores.get("query_relevance", scores.get("total", 0.0)), "semantic_support": scores.get("semantic_support", scores.get("semantic_confidence", 0.0)), "role_compatibility": scores.get("role_compatibility", 0.0), "selection_status": metadata.get("selection_status")}
            admission = admission_service.admit(admission_item, query_session_id=state.get("active_query_session_id"), selected_memory=cell.get("component_class") == "memory_source")
            if not admission["status"].startswith("ADMITTED"):
                continue
            position_x = clamp(_number(cell.get("x"), 0.35 + (index % 4) * 0.1) / 1000 if abs(_number(cell.get("x"))) > 1.0 else _number(cell.get("x"), 0.5))
            position_y = clamp(_number(cell.get("y"), 0.35 + (index % 3) * 0.1) / 1000 if abs(_number(cell.get("y"))) > 1.0 else _number(cell.get("y"), 0.5))
            nodes.append(DynamicsNodeState(
                cell_id=str(cell["id"]),
                label=str(candidate.get("surface") or candidate.get("lemma") or cell.get("canonical_name") or ""),
                node_type=str(cell.get("component_class") or "role_candidate"),
                role=role,
                position_x=position_x,
                position_y=position_y,
                previous_position_x=position_x,
                previous_position_y=position_y,
                global_mass=clamp((_number(cell.get("global_mass"), _number(cell.get("stored_strength"), 0.5)) - min_mass) / (max_mass - min_mass) if max_mass > min_mass else 0.5),
                global_mass_raw=_number(cell.get("global_mass"), _number(cell.get("stored_strength"), 0.5)),
                global_mass_normalized=clamp((_number(cell.get("global_mass"), _number(cell.get("stored_strength"), 0.5)) - min_mass) / (max_mass - min_mass) if max_mass > min_mass else 0.5),
                activation=clamp(_number(scores.get("activation", cell.get("local_activation")), 0.0)),
                retention=clamp(_number(scores.get("retention", cell.get("retention")), 0.5)),
                resonance=clamp(_number(scores.get("resonance"), 0.0)),
                energy=clamp(_number(cell.get("local_activation"), 0.0)),
                mobility=0.15 if cell.get("component_class") == "memory_source" else 0.55 if cell.get("component_class") == "semantic_bridge" else 1.0,
                query_relevance=clamp(_number(scores.get("query_relevance", scores.get("total", 0.0)))),
                role_compatibility=clamp(_number(scores.get("role_compatibility"))),
                source_confidence=clamp(_number(scores.get("source_confidence"))),
                semantic_support=clamp(_number(scores.get("semantic_support", scores.get("semantic_confidence")))),
                structural_support=clamp(_number(scores.get("structural_support"))),
                competition_group_id=candidate.get("competition_group_id") or (f"{state.get('query_scene_id')}:{role}" if cell.get("component_class") == "role_candidate" and role else None),
                query_scene_id=state.get("query_scene_id"),
                candidate_status=candidate.get("status"),
                admission=admission,
            ))
            nodes[-1].local_mass = DynamicsEngine(config).local_mass(nodes[-1])
            nodes[-1].local_gravity = DynamicsEngine(config).gravity(nodes[-1])
            nodes[-1].trajectory = [{"step": 0, "x": position_x, "y": position_y}]
        if not nodes:
            return DynamicsState(status="IDLE", initial_temperature=config.temperature.default, current_temperature=config.temperature.default, minimum_temperature=config.temperature.minimum, maximum_temperature=config.temperature.maximum, cooling_rate=config.temperature.cooling_rate, temperature_state="NOT_STARTED", zones=asdict(config.zones), random_seed=config.random_seed)
        return DynamicsState(initial_temperature=config.temperature.default, current_temperature=config.temperature.default, minimum_temperature=config.temperature.minimum, maximum_temperature=config.temperature.maximum, cooling_rate=config.temperature.cooling_rate, temperature_state=DynamicsEngine.temperature_state(config.temperature.default), temperature_history=[{"step": 0, "value": config.temperature.default}], zones=asdict(config.zones), anchors=anchors, nodes=nodes, random_seed=config.random_seed)

    def _sync_candidates(self, state: Dict[str, Any], dynamics: DynamicsState) -> None:
        by_cell = {str(node.cell_id): node for node in dynamics.nodes}
        by_candidate = {}
        for node in dynamics.nodes:
            candidate_id = (node.admission or {}).get("candidate_id") or (node.admission or {}).get("metadata", {}).get("candidate_id")
            if candidate_id:
                by_candidate[str(candidate_id)] = node
        for candidate in state.get("candidates") or []:
            node = by_cell.get(str(candidate.get("cell_id"))) or by_candidate.get(str(candidate.get("id")))
            if not node:
                continue
            scores = candidate.setdefault("scores", {})
            scores.update({"activation": node.activation, "retention": node.retention, "resonance": node.resonance, "gravity": node.local_gravity, "eviction_score": node.eviction_score, "energy": node.energy})
            candidate["activation"] = node.activation
            candidate["resonance"] = node.resonance
            if node.eviction_status == "EVICTED":
                candidate["status"] = "evicted"

    def get(self, hive_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = DynamicsConfig.from_dict(config)
        with self.repository.transaction() as conn:
            metadata = self._metadata(conn, hive_id)
            state = dict(metadata.get("query_working_memory") or {})
            dynamics = self._initial(conn, hive_id, state, cfg)
            metadata.setdefault("query_working_memory", state)["dynamics"] = dynamics.to_dict()
            self._save(conn, hive_id, metadata)
            return dynamics.to_dict()

    def step(self, hive_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = DynamicsConfig.from_dict(config)
        with self.repository.transaction() as conn:
            metadata = self._metadata(conn, hive_id)
            state = dict(metadata.get("query_working_memory") or {})
            dynamics = self._initial(conn, hive_id, state, cfg)
            semantics = {}
            for node in dynamics.nodes:
                candidate = next((item for item in state.get("candidates") or [] if str(item.get("surface") or item.get("lemma") or "").casefold() == node.label.casefold()), None)
                if candidate:
                    semantics[node.cell_id] = candidate.get("scores") or {}
            winner = next((item for item in state.get("candidates") or [] if item.get("status") == "winner"), None)
            dynamics = DynamicsEngine(cfg).step(dynamics, semantics, stagnated=bool(state.get("vibration", {}).get("current_step", 0) > 1 and not winner))
            self._sync_candidates(state, dynamics)
            for node in dynamics.nodes:
                conn.execute("UPDATE hive_cells SET local_activation=?, retention=?, stored_strength=?, updated_at=? WHERE id=? AND hive_id=?", (node.activation, node.retention, node.local_mass, utcnow(), node.cell_id, hive_id))
                conn.execute("UPDATE cloud_placements SET x=?, y=?, local_activation=?, local_gravity=?, updated_at=? WHERE id=(SELECT hive_placement_id FROM hive_cells WHERE id=? AND hive_id=?)", (node.position_x, node.position_y, node.activation, node.local_gravity, utcnow(), node.cell_id, hive_id))
            conn.execute("UPDATE hives SET reasoning_step=?, current_temperature=?, total_energy=?, updated_at=? WHERE id=?", (dynamics.step, dynamics.current_temperature, sum(node.energy for node in dynamics.nodes), utcnow(), hive_id))
            metadata.setdefault("query_working_memory", state)["dynamics"] = dynamics.to_dict()
            self._save(conn, hive_id, metadata)
            return dynamics.to_dict()

    def history(self, hive_id: str) -> List[Dict[str, Any]]:
        return self.get(hive_id).get("history", [])

    def node(self, hive_id: str, cell_id: str) -> Dict[str, Any]:
        node = next((item for item in self.get(hive_id).get("nodes", []) if str(item.get("cell_id")) == str(cell_id)), None)
        if not node:
            raise KeyError(cell_id)
        return node

    def reset(self, hive_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            metadata = self._metadata(conn, hive_id)
            state = dict(metadata.get("query_working_memory") or {})
            current = DynamicsState.from_dict(state.get("dynamics"))
            initial_nodes = []
            for node in current.nodes:
                restored = node.to_dict()
                first = (node.trajectory or [{"x": node.position_x, "y": node.position_y}])[0]
                restored["position"] = {"x": first.get("x", node.position_x), "y": first.get("y", node.position_y)}
                restored["previous_position"] = dict(restored["position"])
                restored["velocity"] = {"x": 0.0, "y": 0.0}
                restored["acceleration"] = {"x": 0.0, "y": 0.0}
                restored["net_force"] = {"x": 0.0, "y": 0.0, "magnitude": 0.0}
                restored["eviction_status"] = "ACTIVE"
                restored["weak_steps"] = 0
                restored["eviction_score"] = 0.0
                restored["force_breakdown"] = []
                restored["trajectory"] = [dict(first, step=0)]
                initial_nodes.append(restored)
            reset_state = DynamicsState.from_dict({
                **current.to_dict(),
                "step": 0,
                "status": "READY",
                "temperature": {**current.temperature, "current": current.initial_temperature, "state": DynamicsEngine.temperature_state(current.initial_temperature), "history": [{"step": 0, "value": current.initial_temperature}]},
                "nodes": initial_nodes,
                "history": [],
                "eviction_history": [],
            })
            for node in reset_state.nodes:
                conn.execute("UPDATE hive_cells SET local_activation=?, retention=?, stored_strength=?, updated_at=? WHERE id=? AND hive_id=?", (node.activation, node.retention, node.local_mass, utcnow(), node.cell_id, hive_id))
                conn.execute("UPDATE cloud_placements SET x=?, y=?, local_activation=?, local_gravity=?, updated_at=? WHERE id=(SELECT hive_placement_id FROM hive_cells WHERE id=? AND hive_id=?)", (node.position_x, node.position_y, node.activation, node.local_gravity, utcnow(), node.cell_id, hive_id))
            conn.execute("UPDATE hives SET reasoning_step=0, current_temperature=?, updated_at=? WHERE id=?", (reset_state.current_temperature, utcnow(), hive_id))
            state["dynamics"] = reset_state.to_dict()
            metadata["query_working_memory"] = state
            self._save(conn, hive_id, metadata)
        return self.get(hive_id)

    def evictions(self, hive_id: str) -> List[Dict[str, Any]]:
        return self.get(hive_id).get("eviction_history", [])
