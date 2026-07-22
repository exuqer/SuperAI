"""Bounded, explainable dynamics for the semantic field."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


FORCE_TYPES = (
    "COACTIVATION_ATTRACTION", "RELATION_CONDITIONED_ATTRACTION",
    "CONTEXT_ATTRACTION", "CONTRADICTION_REPULSION", "EXCLUSION_REPULSION",
    "MASS_INERTIA", "STABILITY_DAMPING", "HALO_OVERLAP", "DENSITY_PRESSURE",
    "DIMENSION_ALIGNMENT",
)


def _vector(value: Sequence[float], dimensions: int) -> list[float]:
    return [float(value[index]) if index < len(value) else 0.0 for index in range(dimensions)]


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(item * item for item in value))


def _unit(value: Sequence[float]) -> list[float]:
    length = _norm(value)
    return [item / length for item in value] if length > 1e-12 else [0.0 for _ in value]


@dataclass(frozen=True)
class ForceComponent:
    type: str
    source_cloud_id: str | None
    vector: tuple[float, ...]
    magnitude: float
    evidence_ids: tuple[str, ...] = ()
    payload: Mapping[str, Any] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "source_cloud_id": self.source_cloud_id,
            "vector": list(self.vector),
            "magnitude": self.magnitude,
            "evidence_ids": list(self.evidence_ids),
            "payload": dict(self.payload or {}),
        }


@dataclass(frozen=True)
class DynamicsResult:
    cloud_id: str
    forces: tuple[ForceComponent, ...]
    net_force: tuple[float, ...]
    proposed_displacement: tuple[float, ...]
    limited_displacement: tuple[float, ...]
    stability_penalty: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "cloud_id": self.cloud_id,
            "forces": [item.as_dict() for item in self.forces],
            "net_force": list(self.net_force),
            "proposed_displacement": list(self.proposed_displacement),
            "limited_displacement": list(self.limited_displacement),
            "stability_penalty": self.stability_penalty,
        }


class SemanticFieldDynamics:
    def __init__(self, *, max_displacement: float = 0.05, min_distance: float = 0.08) -> None:
        self.max_displacement = float(max_displacement)
        self.min_distance = float(min_distance)

    def calculate(
        self,
        cloud: Mapping[str, Any],
        neighbours: Iterable[Mapping[str, Any]] = (),
        *,
        dimensions: int | None = None,
        active_measurements: Mapping[str, float] | None = None,
    ) -> DynamicsResult:
        center = list(cloud.get("learned_center") or cloud.get("center") or [])
        dimensions = dimensions or max(1, len(center))
        center = _vector(center, dimensions)
        mass = max(0.01, float(cloud.get("mass") or 0.01))
        stability = min(1.0, max(0.0, float(cloud.get("stability") or 0.0)))
        density = min(1.0, max(0.0, float(cloud.get("density") or 0.0)))
        halo = min(1.0, max(0.0, float(cloud.get("halo") or 0.0)))
        previous_displacement = _vector(cloud.get("previous_displacement") or (), dimensions)
        exclusion_ids = {str(item) for item in cloud.get("exclusion_cloud_ids") or ()}
        contradiction_ids = {str(item) for item in cloud.get("contradiction_cloud_ids") or ()}
        forces: list[ForceComponent] = []
        for neighbour in neighbours:
            other_id = str(neighbour.get("cloud_id") or neighbour.get("id") or "")
            if not other_id or other_id == str(cloud.get("cloud_id") or cloud.get("id")):
                continue
            other = _vector(neighbour.get("learned_center") or neighbour.get("center") or [], dimensions)
            delta = [other[index] - center[index] for index in range(dimensions)]
            distance = max(self.min_distance, _norm(delta))
            direction = _unit(delta)
            relation = str(neighbour.get("relation_attachment") or neighbour.get("relation") or "")
            predicate = str(neighbour.get("predicate_configuration") or "")
            polarity = str(neighbour.get("polarity") or "positive").lower()
            weight = min(1.0, float(neighbour.get("coactivation_weight") or 0.25))
            weight *= min(1.0, math.sqrt(float(neighbour.get("mass") or 0.1) / mass))
            kind = "COACTIVATION_ATTRACTION"
            if other_id in contradiction_ids or polarity in {"negative", "contradiction"}:
                kind, weight = "CONTRADICTION_REPULSION", -weight
            elif other_id in exclusion_ids or polarity in {"excluded", "exclusion"}:
                kind, weight = "EXCLUSION_REPULSION", -weight
            elif relation:
                kind, weight = "RELATION_CONDITIONED_ATTRACTION", weight * 0.65
            elif predicate:
                kind, weight = "CONTEXT_ATTRACTION", weight * 0.5
            magnitude = weight / distance
            forces.append(ForceComponent(kind, other_id, tuple(item * magnitude for item in direction), abs(magnitude), tuple(neighbour.get("evidence_ids") or ()), {"relation": relation, "predicate": predicate, "polarity": polarity}))
            overlap = max(0.0, halo + float(neighbour.get("halo") or 0.0) - distance)
            if overlap:
                repulsion = overlap * 0.05
                forces.append(ForceComponent("HALO_OVERLAP", other_id, tuple(-item * repulsion for item in direction), repulsion, tuple(neighbour.get("evidence_ids") or ()), {"distance": distance}))
            neighbour_density = min(1.0, max(0.0, float(neighbour.get("density") or 0.0)))
            pressure = min(1.0, density + neighbour_density) * max(0.0, self.min_distance * 2.0 - distance) * 0.25
            if pressure:
                forces.append(ForceComponent("DENSITY_PRESSURE", other_id, tuple(-item * pressure for item in direction), pressure, tuple(neighbour.get("evidence_ids") or ()), {"distance": distance, "neighbour_density": neighbour_density}))
        if active_measurements:
            alignment = _vector([float(value) for value in active_measurements.values()], dimensions)
            if _norm(alignment):
                unit = _unit(alignment)
                forces.append(ForceComponent("DIMENSION_ALIGNMENT", None, tuple(item * 0.08 for item in unit), 0.08, (), {"active_dimensions": list(active_measurements)}))
        damping = min(0.95, stability * 0.7)
        forces.append(ForceComponent("MASS_INERTIA", None, tuple(0.0 for _ in range(dimensions)), 1.0 / mass, (), {"mass": mass}))
        if damping:
            forces.append(ForceComponent("STABILITY_DAMPING", None, tuple(-item * damping for item in previous_displacement), damping * _norm(previous_displacement), (), {}))
        net = [sum(force.vector[index] for force in forces) for index in range(dimensions)]
        proposed = [item / max(1.0, mass) for item in net]
        length = _norm(proposed)
        limit = self.max_displacement * (1.0 - 0.45 * stability) / max(1.0, math.sqrt(mass))
        if length > limit > 0:
            limited = [item * limit / length for item in proposed]
        else:
            limited = list(proposed)
        return DynamicsResult(str(cloud.get("cloud_id") or cloud.get("id") or ""), tuple(forces), tuple(net), tuple(proposed), tuple(limited), damping)


__all__ = ["FORCE_TYPES", "ForceComponent", "DynamicsResult", "SemanticFieldDynamics"]
