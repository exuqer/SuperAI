from __future__ import annotations

from typing import Any, Iterable, Mapping

from server.bees import BeeBudget, BeeTask, ObserverBee, ScoutBee, WorkerBee
from server.factories import (
    ConceptFactory,
    LexicalFactory,
    MorphologyFactory,
    PromptFactory,
    SymbolFactory,
)
from server.generation import Lexicalizer, ResponseBuilder
from server.spaces import CloudObject, SpaceLevel
from server.v2.repository import V2Repository

from .hive_state import HiveState, HiveStateStore


class HiveDispatcher:
    def __init__(self) -> None:
        self.prompt_factory = PromptFactory()
        self.concept_factory = ConceptFactory()
        self.lexical_factory = LexicalFactory()
        self.morphology_factory = MorphologyFactory()
        self.symbol_factory = SymbolFactory()
        self.lexicalizer = Lexicalizer()
        self.response_builder = ResponseBuilder()
        self.scout = ScoutBee()
        self.worker = WorkerBee()
        self.observer = ObserverBee()

    def process(self, state: HiveState, text: str, query_result: Mapping[str, Any]) -> HiveState:
        state.turn += 1
        frame = self.prompt_factory.build(text, query_result.get("query_frame"))
        state.active_tasks = []
        state.nectar_packets = []
        state.vertical_transitions = []
        state.factories = []
        memory_scenes = list(query_result.get("memory_scenes") or [])
        candidate_values = list(query_result.get("candidates") or [])
        memory_tick = state.memory.tick(frame.topics, self._role_values(frame.roles))
        current_event = self._register_current_event(state, frame, query_result)
        memory_events = self._register_memory_events(state, memory_scenes)
        concepts = self.concept_factory.build_from_events(
            [current_event.to_dict(), *(event.to_dict() for event in memory_events)],
            state.spaces.concept,
        )
        state.factories.extend(
            [
                {"factory": "prompt_factory", "status": "completed", "output": frame.to_dict()},
                {
                    "factory": "concept_factory",
                    "status": "completed",
                    "created": [item.object_id for item in concepts],
                },
            ]
        )
        words = self.lexical_factory.register_candidates(candidate_values, state.spaces.word)
        state.factories.append(
            {
                "factory": "lexical_factory",
                "status": "completed",
                "created": [item.object_id for item in words],
            }
        )
        self._link_vertical(state, current_event, concepts, words)
        pruned = self._prune_spaces(
            state,
            {
                current_event.object_id,
                *(item.object_id for item in memory_events),
                *(item.object_id for item in concepts),
                *(item.object_id for item in words),
            },
        )
        packets, budget, observer_allocations = self._search(state, frame)
        state.nectar_packets = [packet.to_dict() for packet in packets]
        answer = self.response_builder.classify(query_result.get("answer"), candidate_values)
        state.answer = answer
        remembered = self._remember_turn(state, frame, query_result, current_event)
        post_remember_evicted = state.memory.evict()
        memory_tick["evicted"] = list(
            dict.fromkeys([*memory_tick.get("evicted", []), *post_remember_evicted])
        )
        memory_tick["layers"] = state.memory.layer_counts()
        tick_index = (
            int(state.reasoning_ticks[-1].get("tick", 0)) + 1 if state.reasoning_ticks else 1
        )
        tick = {
            "tick": tick_index,
            "turn": state.turn,
            "active_spaces": sorted({packet.origin_space for packet in packets}),
            "packet_count": len(packets),
            "budget": budget.to_dict(),
            "memory": memory_tick,
            "remembered": remembered,
            "pruned": pruned,
            "answer": answer,
        }
        state.reasoning_ticks.append(tick)
        state.reasoning_ticks = state.reasoning_ticks[-state.limits["max_reasoning_ticks"] :]
        state.current_trace = {
            "turn": state.turn,
            "prompt": text,
            "prompt_frame": frame.to_dict(),
            "active_spaces": tick["active_spaces"],
            "bee_tasks": state.active_tasks,
            "bees": budget.ledger,
            "observer_allocations": observer_allocations,
            "nectar_packets": state.nectar_packets,
            "vertical_transitions": state.vertical_transitions,
            "factories": state.factories,
            "memory_events": state.memory.events[-32:],
            "answer": answer,
            "rejected": self._rejected(query_result),
            "limits": state.limits,
        }
        return state

    def refresh_answer(self, state: HiveState, query_result: Mapping[str, Any]) -> HiveState:
        nested_hive = (
            query_result.get("hive", {}) if isinstance(query_result.get("hive"), Mapping) else {}
        )
        raw_answer = query_result.get("answer") or nested_hive.get("answer")
        candidates = list(query_result.get("candidates") or nested_hive.get("candidates") or [])
        answer = self.response_builder.classify(raw_answer, candidates)
        state.answer = answer
        state.current_trace["answer"] = answer
        state.current_trace["answer_source"] = {
            "supporting_scenes": raw_answer.get("supporting_scenes", [])
            if isinstance(raw_answer, Mapping)
            else [],
            "winner": query_result.get("winner")
            or next((item for item in candidates if item.get("status") == "winner"), None),
        }
        current_memory = state.memory.items.get(f"event:query:{state.turn}")
        if current_memory and self._is_resolved_answer(raw_answer):
            current_memory.unresolved_support = 0.0
        if state.reasoning_ticks:
            state.reasoning_ticks[-1]["answer"] = answer
        return state

    def compose_form(
        self,
        state: HiveState,
        concept: str,
        features: Mapping[str, Any],
        *,
        root: str | None = None,
    ) -> dict[str, Any]:
        result = self.lexicalizer.realize(
            concept,
            features,
            state.spaces.word,
            state.spaces.morpheme,
            state.spaces.symbol,
            root=root,
        )
        path = list(result.descent_path)
        transitions = (
            [
                {
                    "from": source,
                    "to": target,
                    "direction": "down" if index < path.index("symbol_space") else "up",
                    "reason": "lazy_form_resolution",
                    "fragment": concept,
                }
                for index, (source, target) in enumerate(zip(path, path[1:]))
            ]
            if "symbol_space" in path
            else [
                {
                    "from": source,
                    "to": target,
                    "direction": "down",
                    "reason": "lexical_retrieval",
                    "fragment": concept,
                }
                for source, target in zip(path, path[1:])
            ]
        )
        state.vertical_transitions.extend(transitions)
        state.factories.extend(
            {"factory": item["stage"].casefold(), "status": "completed", "output": item}
            for item in result.trace
        )
        state.current_trace.setdefault("vertical_transitions", []).extend(transitions)
        state.current_trace["word_assembly"] = result.to_dict()
        word_id = f"word:{concept}:{result.surface}"
        state.current_trace["pruned_after_assembly"] = self._prune_spaces(
            state,
            {
                object_id
                for space in state.spaces.all.values()
                for object_id in space.objects
                if word_id in object_id
            },
        )
        return result.to_dict()

    def _search(self, state: HiveState, frame: Any) -> tuple[list[Any], BeeBudget, dict[str, int]]:
        event_features = {
            role: self._role_value(value)
            for role, value in frame.roles.items()
            if role in state.spaces.event.dimensions and self._role_value(value)
        }
        event_features.update({"polarity": frame.polarity, "modality": frame.modality})
        task = BeeTask(
            target_space=SpaceLevel.EVENT,
            task="fill_missing_role" if frame.missing_slots else "retrieve_event",
            desired_features=event_features,
            budget=state.limits["max_bees_per_slot"] * 2,
            min_relevance=0.42,
            source_fragment=frame.source_text,
            max_candidates=state.limits["max_candidates_per_level"],
            max_downward_depth=state.limits["max_downward_depth"],
        )
        state.active_tasks.append(task.to_dict())
        budget = BeeBudget(task.budget)
        packets = self.scout.run(state.spaces.event, task, budget)
        packets.extend(self.worker.run(state.spaces.event, task, packets, budget))
        if not packets or max((packet.confidence for packet in packets), default=0.0) < 0.56:
            concept_task = BeeTask(
                target_space=SpaceLevel.CONCEPT,
                task="find_supporting_concepts",
                desired_features={"themes": frame.topics, "scene_roles": frame.missing_slots},
                budget=max(4, budget.remaining),
                min_relevance=0.3,
                source_fragment=frame.source_text,
                max_candidates=state.limits["max_candidates_per_level"],
                max_downward_depth=state.limits["max_downward_depth"],
            )
            state.active_tasks.append(concept_task.to_dict())
            concept_budget = BeeBudget(concept_task.budget)
            concept_packets = self.scout.run(state.spaces.concept, concept_task, concept_budget)
            packets.extend(concept_packets)
            state.vertical_transitions.append(
                {
                    "from": SpaceLevel.EVENT.value,
                    "to": SpaceLevel.CONCEPT.value,
                    "direction": "down",
                    "reason": "event_confidence_below_threshold",
                    "fragment": frame.source_text,
                    "result_count": len(concept_packets),
                }
            )
            budget.ledger.extend(concept_budget.ledger)
            budget.spent += concept_budget.spent
            budget.total += concept_budget.total
        packets = sorted(packets, key=lambda packet: (-packet.utility, packet.source_id))[
            : state.limits["max_bees_per_slot"]
        ]
        allocations = self.observer.allocate(packets, budget.remaining)
        return packets, budget, allocations

    def _register_current_event(
        self, state: HiveState, frame: Any, query_result: Mapping[str, Any]
    ) -> CloudObject:
        dimensions = {role: self._role_value(value) for role, value in frame.roles.items()}
        dimensions.update(
            {
                "polarity": frame.polarity,
                "modality": frame.modality,
                "dialogue_relevance": 1.0,
                "topic_relevance": 1.0,
            }
        )
        cloud = CloudObject(
            object_id=f"event:query:{state.turn}",
            label=frame.source_text,
            dimensions=dimensions,
            core={role: 1.0 for role in dimensions if role in {"agent", "action", "object"}},
            density=0.9,
            halo=0.35,
            activated_properties={name: 1.0 for name in dimensions},
            provenance={"source": "query", "message_id": query_result.get("message_id")},
            metadata={
                "missing_slots": frame.missing_slots,
                "intent": frame.intent,
                "topics": frame.topics,
                "turn": state.turn,
            },
        )
        return state.spaces.event.register(cloud)

    def _register_memory_events(
        self, state: HiveState, scenes: Iterable[Mapping[str, Any]]
    ) -> list[CloudObject]:
        events: list[CloudObject] = []
        for index, scene in enumerate(scenes):
            roles = scene.get("roles", {}) if isinstance(scene.get("roles"), Mapping) else {}
            scores = scene.get("scores", {}) if isinstance(scene.get("scores"), Mapping) else {}
            dimensions = {role: self._role_value(value) for role, value in roles.items()}
            dimensions.update(
                {
                    "polarity": scene.get("polarity") or "positive",
                    "modality": "fact",
                    "dialogue_relevance": self._as_float(
                        scene.get("context_score") or scores.get("context"), 0.5
                    ),
                    "topic_relevance": self._as_float(
                        scene.get("score") or scene.get("semantic_total"), 0.5
                    ),
                }
            )
            scene_id = str(scene.get("id") or scene.get("scene_id") or index)
            cloud = CloudObject(
                object_id=f"event:memory:{scene_id}",
                label=str(
                    scene.get("source_text")
                    or scene.get("text")
                    or scene.get("scene_label")
                    or scene_id
                ),
                dimensions=dimensions,
                density=float(scene.get("confidence") or 0.75),
                halo=0.3,
                provenance={
                    "source": scene.get("provenance", {}).get("source")
                    if isinstance(scene.get("provenance"), Mapping)
                    else "memory",
                    "scene_id": scene_id,
                },
                metadata={
                    "result_type": scene.get("result_type"),
                    "topics": list(filter(None, dimensions.values())),
                },
            )
            state.spaces.event.register(cloud)
            events.append(cloud)
        return events

    @staticmethod
    def _link_vertical(
        state: HiveState,
        current: CloudObject,
        concepts: list[CloudObject],
        words: list[CloudObject],
    ) -> None:
        concept_ids = [item.object_id for item in concepts]
        current.links["down:concept_space"] = concept_ids
        words_by_concept: dict[str, list[str]] = {}
        for word in words:
            concept = str(word.dimensions.get("concept") or "").casefold()
            words_by_concept.setdefault(concept, []).append(word.object_id)
        for concept in concepts:
            concept.links["up:event_space"] = sorted(
                set(concept.links.get("up:event_space", []) + [current.object_id])
            )
            concept.links["down:word_space"] = words_by_concept.get(
                concept.label.casefold(), concept.links.get("down:word_space", [])
            )

    def _remember_turn(
        self, state: HiveState, frame: Any, result: Mapping[str, Any], event: CloudObject
    ) -> list[str]:
        remembered: list[str] = []
        answer_resolved = self._is_resolved_answer(result.get("answer"))
        unresolved_support = (
            0.0
            if answer_resolved
            else 0.9
            if frame.missing_slots and not result.get("candidates")
            else 0.2
        )
        item = state.memory.remember(
            event.object_id,
            "event",
            {"text": frame.source_text, "roles": frame.roles, "missing_slots": frame.missing_slots},
            topics=frame.topics,
            mass=1.1,
            retention=0.84 if frame.missing_slots else 0.72,
            unresolved_support=unresolved_support,
            user_priority=0.55 if frame.constraints else 0.2,
            pinned=bool(frame.constraints),
            provenance={"message_id": result.get("message_id"), "source": "dialogue"},
        )
        remembered.append(item.item_id)
        for scene in (result.get("memory_scenes") or [])[:16]:
            scene_id = str(scene.get("id") or scene.get("scene_id") or len(remembered))
            memory_item = state.memory.remember(
                f"memory-scene:{scene_id}",
                "memory_scene",
                {
                    "text": scene.get("source_text")
                    or scene.get("text")
                    or scene.get("scene_label"),
                    "roles": scene.get("roles", {}),
                    "result_type": scene.get("result_type"),
                },
                topics=self._scene_topics(scene),
                mass=0.8,
                retention=float(scene.get("retention") or 0.68),
                provenance={"source": "dialogue_memory", "scene_id": scene_id},
            )
            memory_item.links.add(event.object_id)
            item.links.add(memory_item.item_id)
            remembered.append(memory_item.item_id)
        return remembered

    @staticmethod
    def _rejected(result: Mapping[str, Any]) -> list[dict[str, Any]]:
        rejected = []
        for scene in result.get("memory_scenes") or []:
            validation = (
                scene.get("anchor_validation", {})
                if isinstance(scene.get("anchor_validation"), Mapping)
                else {}
            )
            if (
                scene.get("result_type") in {"NO_HIT", "CONFLICT_HIT"}
                or validation.get("status") == "FAILED"
            ):
                rejected.append(
                    {
                        "id": scene.get("id") or scene.get("scene_id"),
                        "label": scene.get("source_text") or scene.get("text"),
                        "reason": scene.get("result_type") or "anchor_validation_failed",
                    }
                )
        return rejected

    @classmethod
    def _scene_topics(cls, scene: Mapping[str, Any]) -> list[str]:
        roles = scene.get("roles", {}) if isinstance(scene.get("roles"), Mapping) else {}
        return [cls._role_value(value) for value in roles.values() if cls._role_value(value)]

    @classmethod
    def _role_values(cls, roles: Mapping[str, Any]) -> list[str]:
        return [cls._role_value(value) for value in roles.values() if cls._role_value(value)]

    @staticmethod
    def _role_value(value: Any) -> str:
        if isinstance(value, Mapping):
            value = (
                value.get("lemma")
                or value.get("normalized")
                or value.get("surface")
                or value.get("value")
            )
        return str(value or "").casefold()

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _is_resolved_answer(answer: Any) -> bool:
        if not isinstance(answer, Mapping):
            return False
        return answer.get("status") in {"RESOLVED", "RESOLVED_GREETING"} or bool(
            answer.get("surface_answer") or answer.get("full_surface_answer")
        )

    @classmethod
    def _prune_spaces(cls, state: HiveState, protected_ids: set[str]) -> dict[str, list[str]]:
        limits = {
            "event_space": state.limits["max_event_objects"],
            "concept_space": state.limits["max_concept_objects"],
            "word_space": state.limits["max_word_objects"],
            "morpheme_space": state.limits["max_morpheme_objects"],
            "symbol_space": state.limits["max_symbol_objects"],
        }
        pruned: dict[str, list[str]] = {}
        for name, space in state.spaces.all.items():
            overflow = len(space.objects) - limits[name]
            if overflow <= 0:
                continue
            removable = sorted(
                (cloud for cloud in space.objects.values() if cloud.object_id not in protected_ids),
                key=lambda cloud: (
                    cls._as_float(cloud.metadata.get("turn"), 0.0),
                    max(cloud.activated_properties.values(), default=0.0),
                    cloud.density,
                    cloud.object_id,
                ),
            )
            removed = [cloud.object_id for cloud in removable[:overflow]]
            for object_id in removed:
                space.remove(object_id)
            if removed:
                pruned[name] = removed
        removed_ids = {object_id for values in pruned.values() for object_id in values}
        if removed_ids:
            for space in state.spaces.all.values():
                for cloud in space.objects.values():
                    cloud.links = {
                        relation: [target for target in targets if target not in removed_ids]
                        for relation, targets in cloud.links.items()
                    }
        return pruned


class MultilevelHiveService:
    def __init__(self, repository: V2Repository | None = None) -> None:
        self.store = HiveStateStore(repository)
        self.dispatcher = HiveDispatcher()

    def process(self, hive_id: str, text: str, result: Mapping[str, Any]) -> dict[str, Any]:
        state = self.dispatcher.process(self.store.load(hive_id), text, result)
        self.store.save(state)
        return self.snapshot_state(state)

    def refresh_answer(self, hive_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
        state = self.dispatcher.refresh_answer(self.store.load(hive_id), result)
        self.store.save(state)
        return self.snapshot_state(state)

    def get(self, hive_id: str) -> dict[str, Any]:
        return self.snapshot_state(self.store.load(hive_id))

    def traces(self, hive_id: str) -> list[dict[str, Any]]:
        return self.store.traces(hive_id)

    def compose_form(
        self, hive_id: str, concept: str, features: Mapping[str, Any], root: str | None = None
    ) -> dict[str, Any]:
        state = self.store.load(hive_id)
        result = self.dispatcher.compose_form(state, concept, features, root=root)
        self.store.save(state)
        return {"result": result, "multilevel": self.snapshot_state(state)}

    @staticmethod
    def snapshot_state(state: HiveState) -> dict[str, Any]:
        return {
            "hive_id": state.hive_id,
            "turn": state.turn,
            "spaces": {
                name: {"object_count": len(space.objects), "dimensions": list(space.dimensions)}
                for name, space in state.spaces.all.items()
            },
            "memory": state.memory.to_dict(),
            "active_tasks": state.active_tasks,
            "nectar_packets": state.nectar_packets,
            "vertical_transitions": state.vertical_transitions,
            "factories": state.factories,
            "reasoning_ticks": state.reasoning_ticks,
            "trace": state.current_trace,
            "answer": state.answer,
            "limits": state.limits,
        }
