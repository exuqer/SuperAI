from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from server.spaces import CloudObject, ConceptSpace


class ConceptFactory:
    def build_from_events(
        self,
        events: Iterable[Mapping[str, Any]],
        space: ConceptSpace,
    ) -> list[CloudObject]:
        observations: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"actions": set(), "roles": set(), "themes": set(), "events": [], "count": 0}
        )
        for event in events:
            dimensions = event.get("dimensions", event)
            action = self._value(dimensions.get("action"))
            for role in ("agent", "object", "location", "instrument", "source", "destination"):
                label = self._value(dimensions.get(role))
                if not label:
                    continue
                item = observations[label]
                item["actions"].add(action) if action else None
                item["roles"].add(role)
                item["themes"].update(
                    str(topic).casefold() for topic in event.get("topics", []) if topic
                )
                item["events"].append(str(event.get("object_id") or event.get("id") or ""))
                item["count"] += 1
        concepts: list[CloudObject] = []
        for label, observation in sorted(observations.items()):
            object_id = f"concept:{label}"
            cloud = CloudObject(
                object_id=object_id,
                label=label,
                dimensions={
                    "themes": sorted(observation["themes"] or {label}),
                    "abstractness": 0.2,
                    "actions": sorted(filter(None, observation["actions"])),
                    "properties": [],
                    "similarity": [],
                    "cross_domain": min(1.0, len(observation["roles"]) / 4),
                    "stability": min(1.0, 0.45 + observation["count"] * 0.08),
                    "scene_roles": sorted(observation["roles"]),
                },
                core={"identity": 1.0, "stability": min(1.0, 0.45 + observation["count"] * 0.08)},
                density=min(1.0, 0.55 + observation["count"] * 0.07),
                halo=min(1.0, 0.2 + len(observation["actions"]) * 0.12),
                context_variations=[
                    {"event_id": event_id, "activation": 1.0 / max(1, len(observation["events"]))}
                    for event_id in observation["events"]
                    if event_id
                ],
                links={
                    "up:event_space": [event_id for event_id in observation["events"] if event_id],
                    "down:word_space": [f"word:{label}"],
                },
                provenance={"source": "concept_factory", "observations": observation["count"]},
            )
            space.register(cloud)
            concepts.append(cloud)
        return concepts

    @staticmethod
    def _value(value: Any) -> str:
        if isinstance(value, Mapping):
            value = value.get("lemma") or value.get("normalized") or value.get("surface")
        return str(value or "").casefold()
