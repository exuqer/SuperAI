from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class SpaceLevel(str, Enum):
    EVENT = "event_space"
    CONCEPT = "concept_space"
    WORD = "word_space"
    MORPHEME = "morpheme_space"
    SYMBOL = "symbol_space"


@dataclass
class CloudObject:
    object_id: str
    label: str
    dimensions: dict[str, Any]
    core: dict[str, float] = field(default_factory=dict)
    density: float = 1.0
    halo: float = 0.25
    elongation: dict[str, float] = field(default_factory=dict)
    activated_properties: dict[str, float] = field(default_factory=dict)
    context_variations: list[dict[str, Any]] = field(default_factory=list)
    links: dict[str, list[str]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CloudObject":
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})


@dataclass(frozen=True)
class CloudActivation:
    object_id: str
    label: str
    activation: float
    confidence: float
    distance: float
    dimensions: dict[str, Any]
    activated_dimensions: dict[str, float]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultidimensionalSpace:
    level: SpaceLevel
    dimensions: tuple[str, ...] = ()
    weights: dict[str, float] = {}

    def __init__(self) -> None:
        self.objects: dict[str, CloudObject] = {}

    def register(self, cloud: CloudObject) -> CloudObject:
        self.objects[cloud.object_id] = cloud
        return cloud

    def remove(self, object_id: str) -> None:
        self.objects.pop(object_id, None)

    def describe_object(self, object_id: str) -> dict[str, Any]:
        cloud = self.objects[object_id]
        return {
            **cloud.to_dict(),
            "space": self.level.value,
            "known_dimensions": list(self.dimensions),
        }

    def distance(
        self,
        left: str | CloudObject | Mapping[str, Any],
        right: str | CloudObject | Mapping[str, Any],
        active_dimensions: Iterable[str] | None = None,
    ) -> float:
        left_dimensions = self._dimensions(left)
        right_dimensions = self._dimensions(right)
        names = tuple(
            active_dimensions or self.dimensions or set(left_dimensions) | set(right_dimensions)
        )
        weighted_distance = 0.0
        weight_total = 0.0
        for name in names:
            if name not in left_dimensions and name not in right_dimensions:
                continue
            weight = max(0.0, float(self.weights.get(name, 1.0)))
            weighted_distance += weight * self._value_distance(
                left_dimensions.get(name), right_dimensions.get(name)
            )
            weight_total += weight
        return min(1.0, weighted_distance / weight_total) if weight_total else 1.0

    def activate(
        self,
        desired_features: Mapping[str, Any],
        *,
        min_relevance: float = 0.0,
        limit: int = 12,
    ) -> list[CloudActivation]:
        results: list[CloudActivation] = []
        active_dimensions = tuple(key for key in desired_features if key in self.dimensions)
        for cloud in self.objects.values():
            distance = self.distance(desired_features, cloud, active_dimensions or None)
            support = 1.0 - distance
            shape = min(1.0, 0.65 + cloud.density * 0.2 + cloud.halo * 0.15)
            activation = min(1.0, support * shape)
            if activation < min_relevance:
                continue
            per_dimension = {
                name: 1.0
                - self._value_distance(desired_features.get(name), cloud.dimensions.get(name))
                for name in active_dimensions
            }
            results.append(
                CloudActivation(
                    object_id=cloud.object_id,
                    label=cloud.label,
                    activation=round(activation, 6),
                    confidence=round(min(1.0, support * (0.6 + cloud.density * 0.4)), 6),
                    distance=round(distance, 6),
                    dimensions=dict(cloud.dimensions),
                    activated_dimensions=per_dimension,
                    provenance=dict(cloud.provenance),
                )
            )
        return sorted(results, key=lambda item: (-item.activation, item.object_id))[: max(0, limit)]

    def expand(self, object_id: str, *, limit: int = 12) -> list[dict[str, Any]]:
        cloud = self.objects[object_id]
        linked_ids = [item for values in cloud.links.values() for item in values]
        linked = [self.describe_object(item) for item in linked_ids if item in self.objects]
        if len(linked) < limit:
            neighbours = sorted(
                (
                    (self.distance(cloud, candidate), candidate.object_id)
                    for candidate in self.objects.values()
                    if candidate.object_id != object_id and candidate.object_id not in linked_ids
                ),
                key=lambda item: (item[0], item[1]),
            )
            linked.extend(
                self.describe_object(item[1]) for item in neighbours[: limit - len(linked)]
            )
        return linked[:limit]

    def down_project(self, object_id: str) -> list[dict[str, Any]]:
        cloud = self.objects[object_id]
        return [
            {"source_id": object_id, "target_id": target, "relation": relation}
            for relation, targets in cloud.links.items()
            if relation.startswith("down:")
            for target in targets
        ]

    def up_project(self, object_id: str) -> list[dict[str, Any]]:
        cloud = self.objects[object_id]
        return [
            {"source_id": object_id, "target_id": target, "relation": relation}
            for relation, targets in cloud.links.items()
            if relation.startswith("up:")
            for target in targets
        ]

    def validate_candidate(self, candidate: CloudObject | Mapping[str, Any]) -> dict[str, Any]:
        dimensions = self._dimensions(candidate)
        missing = [name for name in self.dimensions if name not in dimensions]
        invalid = [
            name for name, value in dimensions.items() if name in self.dimensions and value is None
        ]
        coverage = (len(self.dimensions) - len(missing)) / max(1, len(self.dimensions))
        return {
            "valid": not invalid and bool(dimensions),
            "coverage": round(coverage, 6),
            "missing_dimensions": missing,
            "invalid_dimensions": invalid,
        }

    def visualize(self) -> dict[str, Any]:
        nodes = []
        edges = []
        for index, cloud in enumerate(
            sorted(self.objects.values(), key=lambda item: item.object_id)
        ):
            angle = index * 2.399963229728653
            radius = 0.14 + 0.036 * math.sqrt(index + 1)
            nodes.append(
                {
                    "id": cloud.object_id,
                    "label": cloud.label,
                    "x": round(0.5 + math.cos(angle) * radius, 6),
                    "y": round(0.5 + math.sin(angle) * radius, 6),
                    "density": cloud.density,
                    "halo": cloud.halo,
                    "dimensions": cloud.dimensions,
                    "activation": max(cloud.activated_properties.values(), default=0.0),
                }
            )
            edges.extend(
                {"source": cloud.object_id, "target": target, "relation": relation}
                for relation, targets in cloud.links.items()
                for target in targets
            )
        return {
            "space": self.level.value,
            "dimensions": list(self.dimensions),
            "nodes": nodes,
            "edges": edges,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "objects": [item.to_dict() for item in self.objects.values()],
        }

    def load(self, value: Mapping[str, Any]) -> None:
        self.objects = {
            item["object_id"]: CloudObject.from_dict(item) for item in value.get("objects", [])
        }

    def _dimensions(self, value: str | CloudObject | Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(value, str):
            return self.objects[value].dimensions
        if isinstance(value, CloudObject):
            return value.dimensions
        nested = value.get("dimensions")
        return nested if isinstance(nested, Mapping) else value

    @classmethod
    def _value_distance(cls, left: Any, right: Any) -> float:
        if left is None and right is None:
            return 0.0
        if left is None or right is None:
            return 1.0
        if isinstance(left, bool) or isinstance(right, bool):
            return 0.0 if left == right else 1.0
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            scale = max(1.0, abs(float(left)), abs(float(right)))
            return min(1.0, abs(float(left) - float(right)) / scale)
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            keys = set(left) | set(right)
            return sum(cls._value_distance(left.get(key), right.get(key)) for key in keys) / max(
                1, len(keys)
            )
        if isinstance(left, (list, tuple, set)) or isinstance(right, (list, tuple, set)):
            left_set = (
                {str(item).casefold() for item in left}
                if isinstance(left, (list, tuple, set))
                else {str(left).casefold()}
            )
            right_set = (
                {str(item).casefold() for item in right}
                if isinstance(right, (list, tuple, set))
                else {str(right).casefold()}
            )
            return 1.0 - len(left_set & right_set) / max(1, len(left_set | right_set))
        return 0.0 if str(left).casefold() == str(right).casefold() else 1.0
