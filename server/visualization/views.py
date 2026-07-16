from __future__ import annotations

import math
from typing import Any

from server.hive.hive_state import HiveState
from server.memory import MemoryLayer


class VisualizationSuite:
    view_ids = (
        "global",
        "hive",
        "topics",
        "events",
        "concepts",
        "words",
        "morphemes",
        "symbols",
        "vertical_transition",
        "tick_timeline",
        "retention",
        "explanation",
    )

    def build(self, state: HiveState, view_id: str = "all") -> dict[str, Any]:
        builders = {
            "global": self.global_view,
            "hive": self.hive_view,
            "topics": self.topic_view,
            "events": lambda value: self.space_view(value, "event_space"),
            "concepts": lambda value: self.space_view(value, "concept_space"),
            "words": lambda value: self.space_view(value, "word_space"),
            "morphemes": lambda value: self.space_view(value, "morpheme_space"),
            "symbols": lambda value: self.space_view(value, "symbol_space"),
            "vertical_transition": self.vertical_transition_view,
            "tick_timeline": self.tick_timeline,
            "retention": self.retention_view,
            "explanation": self.explanation_view,
        }
        if view_id == "all":
            return {name: builder(state) for name, builder in builders.items()}
        if view_id not in builders:
            raise KeyError(view_id)
        return {"id": view_id, "view": builders[view_id](state)}

    def global_view(self, state: HiveState) -> dict[str, Any]:
        concepts = state.spaces.concept.visualize()
        scouts = [
            {
                "task_id": task["task_id"],
                "target_space": task["target_space"],
                "fragment": task.get("source_fragment"),
            }
            for task in state.active_tasks
        ]
        return {
            "kind": "cloud_field",
            "clouds": concepts["nodes"],
            "relations": concepts["edges"],
            "scout_routes": scouts,
            "memory_entry_points": [
                {
                    "id": cluster.cluster_id,
                    "topics": sorted(cluster.topics),
                    "temperature": cluster.temperature,
                }
                for cluster in state.memory.clusters.values()
            ],
        }

    def hive_view(self, state: HiveState) -> dict[str, Any]:
        layers = {
            layer.value: [
                {
                    **item.to_dict(),
                    "eviction_score": state.memory.eviction_score(item),
                }
                for item in state.memory.items.values()
                if item.layer == layer
            ]
            for layer in MemoryLayer
        }
        return {
            "kind": "depth_layers",
            "layers": layers,
            "active_factories": state.factories,
            "active_bees": state.current_trace.get("bees", []),
            "eviction_candidates": sorted(
                (
                    {"id": item.item_id, "score": state.memory.eviction_score(item)}
                    for item in state.memory.items.values()
                    if not item.pinned
                ),
                key=lambda item: (-item["score"], item["id"]),
            )[:12],
        }

    def topic_view(self, state: HiveState) -> dict[str, Any]:
        clusters = []
        edges = []
        for index, cluster in enumerate(state.memory.clusters.values()):
            angle = index * 2.399963229728653
            radius = 0.2 + 0.04 * math.sqrt(index + 1)
            clusters.append(
                {
                    **cluster.to_dict(),
                    "x": round(0.5 + math.cos(angle) * radius, 6),
                    "y": round(0.5 + math.sin(angle) * radius, 6),
                }
            )
        for left_index, left in enumerate(clusters):
            for right in clusters[left_index + 1 :]:
                shared = sorted(set(left["topics"]) & set(right["topics"]))
                if shared:
                    edges.append(
                        {
                            "source": left["cluster_id"],
                            "target": right["cluster_id"],
                            "shared_topics": shared,
                        }
                    )
        return {"kind": "topic_islands", "clusters": clusters, "edges": edges}

    @staticmethod
    def space_view(state: HiveState, space_name: str) -> dict[str, Any]:
        view = state.spaces.get(space_name).visualize()
        return {"kind": "graph", **view}

    @staticmethod
    def vertical_transition_view(state: HiveState) -> dict[str, Any]:
        levels = ["event_space", "concept_space", "word_space", "morpheme_space", "symbol_space"]
        return {
            "kind": "vertical_flow",
            "levels": [
                {"id": level, "index": index, "object_count": len(state.spaces.get(level).objects)}
                for index, level in enumerate(levels)
            ],
            "transitions": state.vertical_transitions,
            "word_assembly": state.current_trace.get("word_assembly"),
        }

    @staticmethod
    def tick_timeline(state: HiveState) -> dict[str, Any]:
        return {"kind": "timeline", "ticks": state.reasoning_ticks}

    @staticmethod
    def retention_view(state: HiveState) -> dict[str, Any]:
        return {
            "kind": "retention_table",
            "rows": sorted(
                (
                    {
                        "id": item.item_id,
                        "layer": item.layer.value,
                        "temperature": item.temperature,
                        "activation": item.activation,
                        "mass": item.mass,
                        "depth": item.depth,
                        "retention": item.retention,
                        "compression": item.compression_state,
                        "eviction_score": state.memory.eviction_score(item),
                        "pinned": item.pinned,
                    }
                    for item in state.memory.items.values()
                ),
                key=lambda item: (item["layer"], -item["temperature"], item["id"]),
            ),
            "events": state.memory.events,
        }

    @staticmethod
    def explanation_view(state: HiveState) -> dict[str, Any]:
        trace = state.current_trace
        return {
            "kind": "explanation",
            "answer": state.answer,
            "retrieved": [
                packet
                for packet in state.nectar_packets
                if packet.get("provenance", {}).get("source")
                not in {"morphology_factory", "symbol_factory"}
            ],
            "composed": trace.get("word_assembly"),
            "reactivated": [
                event
                for event in trace.get("memory_events", [])
                if event.get("event_type") == "REACTIVATED"
            ],
            "rejected": trace.get("rejected", []),
            "factory_trace": state.factories,
            "vertical_transitions": state.vertical_transitions,
            "fact_guard": {"lower_levels_created_fact": False},
        }
