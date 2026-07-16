from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class MemoryLayer(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    ARCHIVE = "archive"


@dataclass(frozen=True)
class ThermoGravityConfig:
    hot_temperature: float = 0.72
    warm_temperature: float = 0.38
    cold_temperature: float = 0.13
    cooling_rate: float = 0.13
    activation_decay: float = 0.11
    sedimentation_rate: float = 0.12
    diffusion_rate: float = 0.16
    reactivation_threshold: float = 0.22
    max_items: int = 96
    max_cold_clusters: int = 12


@dataclass
class MemoryItem:
    item_id: str
    item_type: str
    content: dict[str, Any]
    topics: set[str]
    activation: float = 1.0
    temperature: float = 1.0
    mass: float = 1.0
    depth: float = 0.0
    retention: float = 0.7
    recency: float = 1.0
    topic_relevance: float = 1.0
    entity_overlap: float = 0.0
    unresolved_support: float = 0.0
    user_priority: float = 0.0
    compression_state: str = "RAW"
    pinned: bool = False
    layer: MemoryLayer = MemoryLayer.HOT
    last_access_turn: int = 0
    access_count: int = 1
    links: set[str] = field(default_factory=set)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["topics"] = sorted(self.topics)
        value["links"] = sorted(self.links)
        value["layer"] = self.layer.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryItem":
        data = dict(value)
        data["topics"] = set(data.get("topics", []))
        data["links"] = set(data.get("links", []))
        data["layer"] = MemoryLayer(data.get("layer", MemoryLayer.HOT.value))
        return cls(**data)


@dataclass
class ThematicCluster:
    cluster_id: str
    topics: set[str]
    member_ids: list[str]
    temperature: float
    mass: float
    depth: float
    retention: float
    state: str = "ACTIVE"
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["topics"] = sorted(self.topics)
        return value


class ThermoGravityMemory:
    token_pattern = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)

    def __init__(self, config: ThermoGravityConfig | None = None) -> None:
        self.config = config or ThermoGravityConfig()
        self.turn = 0
        self.items: dict[str, MemoryItem] = {}
        self.clusters: dict[str, ThematicCluster] = {}
        self.events: list[dict[str, Any]] = []

    def remember(
        self,
        item_id: str,
        item_type: str,
        content: Mapping[str, Any],
        *,
        topics: Iterable[str] | None = None,
        mass: float = 1.0,
        retention: float = 0.7,
        pinned: bool = False,
        unresolved_support: float = 0.0,
        user_priority: float = 0.0,
        provenance: Mapping[str, Any] | None = None,
    ) -> MemoryItem:
        resolved_topics = self.normalize_topics(topics or self._topics_from_content(content))
        existing = self.items.get(item_id)
        if existing:
            existing.content.update(dict(content))
            existing.topics.update(resolved_topics)
            existing.activation = 1.0
            existing.temperature = 1.0
            existing.depth = 0.0
            existing.recency = 1.0
            existing.layer = MemoryLayer.HOT
            existing.mass = max(existing.mass, mass)
            existing.retention = max(existing.retention, retention)
            existing.pinned = existing.pinned or pinned
            existing.unresolved_support = max(existing.unresolved_support, unresolved_support)
            existing.user_priority = max(existing.user_priority, user_priority)
            existing.compression_state = "RAW"
            existing.last_access_turn = self.turn
            existing.access_count += 1
            existing.provenance.update(dict(provenance or {}))
            event_type = "HEATED"
            item = existing
        else:
            item = MemoryItem(
                item_id=item_id,
                item_type=item_type,
                content=dict(content),
                topics=resolved_topics,
                mass=max(0.05, mass),
                retention=self._clamp(retention),
                pinned=pinned,
                unresolved_support=self._clamp(unresolved_support),
                user_priority=self._clamp(user_priority),
                last_access_turn=self.turn,
                provenance=dict(provenance or {}),
            )
            self.items[item_id] = item
            event_type = "REMEMBERED"
        self._event(event_type, item_id, {"layer": item.layer.value, "topics": sorted(item.topics)})
        self._rebuild_clusters()
        return item

    def tick(
        self, active_topics: Iterable[str], active_entities: Iterable[str] = ()
    ) -> dict[str, Any]:
        self.turn += 1
        topics = self.normalize_topics(active_topics)
        entities = self.normalize_topics(active_entities)
        transitions: list[dict[str, Any]] = []
        for item in self.items.values():
            previous_layer = item.layer
            overlap = self.overlap(item.topics, topics)
            entity_overlap = self.overlap(self._topics_from_content(item.content), entities)
            item.topic_relevance = overlap
            item.entity_overlap = entity_overlap
            if overlap >= self.config.reactivation_threshold or entity_overlap >= 0.34:
                boost = min(0.9, 0.34 + overlap * 0.48 + entity_overlap * 0.28)
                item.activation = self._clamp(item.activation + boost)
                item.temperature = self._clamp(item.temperature + boost * 0.82)
                item.depth = max(0.0, item.depth - boost * 0.72)
                item.recency = 1.0
                item.last_access_turn = self.turn
                item.access_count += 1
                if previous_layer in {MemoryLayer.COLD, MemoryLayer.ARCHIVE}:
                    self._event("REACTIVATED", item.item_id, {"from": previous_layer.value})
            else:
                protection = max(
                    item.retention,
                    item.user_priority,
                    item.unresolved_support,
                    1.0 if item.pinned else 0.0,
                )
                item.temperature = self._clamp(
                    item.temperature - self.config.cooling_rate * (1.08 - protection * 0.45)
                )
                item.activation = self._clamp(
                    item.activation - self.config.activation_decay * (1.06 - protection * 0.4)
                )
                item.recency = self._clamp(item.recency - 0.12)
                item.depth = self._clamp(
                    item.depth + self.config.sedimentation_rate * (1.0 - protection * 0.45)
                )
            item.layer = self._layer_for(item)
            if item.layer != previous_layer:
                transition = {
                    "item_id": item.item_id,
                    "from": previous_layer.value,
                    "to": item.layer.value,
                }
                transitions.append(transition)
                self._event("LAYER_CHANGED", item.item_id, transition)
        self.diffuse()
        self._rebuild_clusters()
        compressed = self.compress()
        evicted = self.evict()
        return {
            "turn": self.turn,
            "transitions": transitions,
            "compressed": compressed,
            "evicted": evicted,
            "layers": self.layer_counts(),
        }

    def diffuse(self) -> None:
        updates: dict[str, float] = {}
        for item in self.items.values():
            if item.activation <= 0.2:
                continue
            for linked_id in item.links:
                if linked_id in self.items:
                    updates[linked_id] = (
                        updates.get(linked_id, 0.0) + item.activation * self.config.diffusion_rate
                    )
            for candidate in self.items.values():
                if candidate.item_id == item.item_id:
                    continue
                overlap = self.overlap(item.topics, candidate.topics)
                if overlap >= 0.5:
                    updates[candidate.item_id] = (
                        updates.get(candidate.item_id, 0.0)
                        + item.activation * overlap * self.config.diffusion_rate * 0.5
                    )
        for item_id, amount in updates.items():
            item = self.items[item_id]
            before = item.activation
            item.activation = self._clamp(item.activation + min(0.24, amount))
            if item.activation - before >= 0.05:
                self._event("DIFFUSED", item_id, {"delta": round(item.activation - before, 6)})

    def compress(self) -> list[str]:
        compressed: list[str] = []
        cold_clusters = sorted(
            (
                cluster
                for cluster in self.clusters.values()
                if cluster.temperature < self.config.warm_temperature
                and cluster.state != "COMPRESSED"
            ),
            key=lambda cluster: (cluster.temperature, cluster.cluster_id),
        )
        overflow = max(0, len(cold_clusters) - self.config.max_cold_clusters)
        for cluster in cold_clusters[:overflow]:
            members = [
                self.items[item_id] for item_id in cluster.member_ids if item_id in self.items
            ]
            if not members or any(
                item.pinned or item.unresolved_support > 0.45 for item in members
            ):
                continue
            cluster.state = "COMPRESSED"
            cluster.summary = {
                "topics": sorted(cluster.topics),
                "item_count": len(members),
                "entities": sorted({topic for item in members for topic in item.topics})[:16],
                "source_ids": [item.item_id for item in members],
            }
            for item in members:
                item.compression_state = "CLUSTER_SUMMARY"
                item.content = {
                    "summary": item.content.get("text")
                    or item.content.get("label")
                    or item.item_id,
                    "roles": item.content.get("roles", {}),
                }
            compressed.append(cluster.cluster_id)
            self._event("COMPRESSED", cluster.cluster_id, cluster.summary)
        return compressed

    def evict(self) -> list[str]:
        overflow = len(self.items) - self.config.max_items
        if overflow <= 0:
            return []
        ranked = sorted(
            self.items.values(),
            key=lambda item: (
                self.eviction_score(item),
                -item.last_access_turn,
                item.item_id,
            ),
            reverse=True,
        )
        protected = {
            item.item_id
            for item in ranked
            if item.pinned or item.unresolved_support >= 0.45 or item.user_priority >= 0.65
        }
        preferred = [
            item
            for item in ranked
            if item.item_id not in protected
            and (
                item.layer in {MemoryLayer.COLD, MemoryLayer.ARCHIVE}
                or self.eviction_score(item) >= 0.72
            )
        ]
        preferred_ids = {item.item_id for item in preferred}
        fallback_old = [
            item
            for item in ranked
            if item.item_id not in protected
            and item.item_id not in preferred_ids
            and item.last_access_turn < self.turn
        ]
        fallback_old_ids = {item.item_id for item in fallback_old}
        fallback_current = [
            item
            for item in ranked
            if item.item_id not in protected
            and item.item_id not in preferred_ids
            and item.item_id not in fallback_old_ids
        ]
        evicted: list[str] = []
        for item in [*preferred, *fallback_old, *fallback_current]:
            if len(evicted) >= overflow:
                break
            evicted.append(item.item_id)
            self.items.pop(item.item_id, None)
            self._event("EVICTED", item.item_id, {"score": round(self.eviction_score(item), 6)})
        if evicted:
            evicted_ids = set(evicted)
            for item in self.items.values():
                item.links.difference_update(evicted_ids)
        self._rebuild_clusters()
        return evicted

    def reactivate(self, topics: Iterable[str]) -> list[MemoryItem]:
        normalized = self.normalize_topics(topics)
        candidates = sorted(
            (
                item
                for item in self.items.values()
                if self.overlap(item.topics, normalized) >= self.config.reactivation_threshold
            ),
            key=lambda item: (
                -self.overlap(item.topics, normalized),
                -item.retention,
                item.item_id,
            ),
        )
        for item in candidates:
            previous = item.layer
            item.temperature = max(item.temperature, 0.78)
            item.activation = max(item.activation, 0.82)
            item.depth = min(item.depth, 0.18)
            item.layer = MemoryLayer.HOT
            item.last_access_turn = self.turn
            self._event(
                "REACTIVATED", item.item_id, {"from": previous.value, "trigger": sorted(normalized)}
            )
        self._rebuild_clusters()
        return candidates

    def layer_counts(self) -> dict[str, int]:
        return {
            layer.value: sum(1 for item in self.items.values() if item.layer == layer)
            for layer in MemoryLayer
        }

    def eviction_score(self, item: MemoryItem) -> float:
        protection = (
            item.retention * 0.22
            + item.topic_relevance * 0.2
            + item.entity_overlap * 0.12
            + item.unresolved_support * 0.16
            + item.user_priority * 0.14
            + (0.24 if item.pinned else 0.0)
        )
        weakness = (
            (1.0 - item.activation) * 0.24
            + (1.0 - item.temperature) * 0.2
            + item.depth * 0.22
            + (1.0 - item.recency) * 0.16
            + (0.12 if item.compression_state != "RAW" else 0.0)
        )
        return self._clamp(weakness - protection + 0.38)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "config": asdict(self.config),
            "items": [
                item.to_dict()
                for item in sorted(self.items.values(), key=lambda item: item.item_id)
            ],
            "clusters": [
                cluster.to_dict()
                for cluster in sorted(
                    self.clusters.values(), key=lambda cluster: cluster.cluster_id
                )
            ],
            "events": self.events[-256:],
            "layers": self.layer_counts(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ThermoGravityMemory":
        config_data = value.get("config", {})
        memory = cls(ThermoGravityConfig(**config_data) if config_data else None)
        memory.turn = int(value.get("turn", 0))
        memory.items = {
            item["item_id"]: MemoryItem.from_dict(item) for item in value.get("items", [])
        }
        memory.events = list(value.get("events", []))
        memory._rebuild_clusters()
        return memory

    def _layer_for(self, item: MemoryItem) -> MemoryLayer:
        effective = item.temperature * 0.62 + item.activation * 0.23 + item.retention * 0.15
        if effective >= self.config.hot_temperature or item.unresolved_support >= 0.7:
            return MemoryLayer.HOT
        if effective >= self.config.warm_temperature:
            return MemoryLayer.WARM
        if effective >= self.config.cold_temperature:
            return MemoryLayer.COLD
        return MemoryLayer.ARCHIVE

    def _rebuild_clusters(self) -> None:
        groups: list[list[MemoryItem]] = []
        for item in sorted(self.items.values(), key=lambda candidate: candidate.item_id):
            group = next(
                (
                    candidate
                    for candidate in groups
                    if self.overlap(
                        item.topics, {topic for member in candidate for topic in member.topics}
                    )
                    >= 0.2
                ),
                None,
            )
            (group if group is not None else groups.append([]) or groups[-1]).append(item)
        self.clusters = {}
        for index, group in enumerate(groups):
            topics = {topic for item in group for topic in item.topics}
            stable_key = "-".join(sorted(topics)[:3]) or str(index)
            cluster_id = f"topic:{stable_key}"
            self.clusters[cluster_id] = ThematicCluster(
                cluster_id=cluster_id,
                topics=topics,
                member_ids=[item.item_id for item in group],
                temperature=sum(item.temperature for item in group) / len(group),
                mass=sum(item.mass for item in group),
                depth=sum(item.depth for item in group) / len(group),
                retention=sum(item.retention for item in group) / len(group),
                state="COMPRESSED"
                if all(item.compression_state != "RAW" for item in group)
                else "ACTIVE",
            )

    def _event(self, event_type: str, target_id: str, payload: Mapping[str, Any]) -> None:
        self.events.append(
            {
                "turn": self.turn,
                "event_type": event_type,
                "target_id": target_id,
                "payload": dict(payload),
            }
        )
        self.events = self.events[-256:]

    @classmethod
    def _topics_from_content(cls, content: Mapping[str, Any]) -> set[str]:
        def flatten(value: Any) -> list[str]:
            if isinstance(value, Mapping):
                return [
                    item
                    for key, nested in value.items()
                    if key not in {"id", "index", "confidence", "score"}
                    for item in flatten(nested)
                ]
            if isinstance(value, (list, tuple, set)):
                return [item for nested in value for item in flatten(nested)]
            return [str(value)] if isinstance(value, (str, int)) else []

        values = flatten(content)
        return {
            token.casefold()
            for value in values
            for token in cls.token_pattern.findall(value)
            if len(token) > 1
        }

    @classmethod
    def normalize_topics(cls, topics: Iterable[str]) -> set[str]:
        return {
            token.casefold()
            for topic in topics
            for token in cls.token_pattern.findall(str(topic))
            if len(token) > 1
        }

    @staticmethod
    def overlap(left: Iterable[str], right: Iterable[str]) -> float:
        left_set = set(left)
        right_set = set(right)
        if not left_set or not right_set:
            return 0.0
        return len(left_set & right_set) / math.sqrt(len(left_set) * len(right_set))

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
