from fastapi.testclient import TestClient

from server.bees import BeeBudget, BeeTask, NectarPacket, WorkerBee
from server.factories import ConceptFactory
from server.generation import ResultStatus
from server.hive.hive_dispatcher import HiveDispatcher
from server.hive.hive_state import HiveState
from server.memory import MemoryLayer, ThermoGravityConfig, ThermoGravityMemory
from server.server import app
from server.spaces import CloudObject, EventSpace
from server.v2.hive import V2HiveService
from server.v2.training import TrainingPipelineV2


def test_all_spaces_implement_cloud_contract():
    state = HiveState("contract")
    for name, space in state.spaces.all.items():
        cloud = CloudObject(
            object_id=f"{name}:one",
            label="one",
            dimensions={space.dimensions[0]: "value"},
        )
        space.register(cloud)
        assert space.describe_object(cloud.object_id)["space"] == name
        assert space.distance(cloud, cloud) == 0
        assert space.activate({space.dimensions[0]: "value"})[0].object_id == cloud.object_id
        assert space.expand(cloud.object_id) == []
        assert space.validate_candidate(cloud)["valid"] is True
        assert space.visualize()["nodes"][0]["id"] == cloud.object_id


def test_concept_factory_builds_cloud_with_context_variations():
    events = EventSpace()
    first = events.register(
        CloudObject("event:1", "Кот ест рыбу", {"agent": "кот", "action": "есть", "object": "рыба"})
    )
    second = events.register(
        CloudObject(
            "event:2", "Рыбак ловит рыбу", {"agent": "рыбак", "action": "ловить", "object": "рыба"}
        )
    )
    state = HiveState("concepts")
    concepts = ConceptFactory().build_from_events(
        [first.to_dict(), second.to_dict()], state.spaces.concept
    )
    fish = next(item for item in concepts if item.label == "рыба")
    assert fish.dimensions["actions"] == ["есть", "ловить"]
    assert fish.dimensions["scene_roles"] == ["object"]
    assert len(fish.context_variations) == 2
    assert fish.density > 0.6


def test_thermogravity_memory_cools_topics_and_reactivates_old_island():
    memory = ThermoGravityMemory(ThermoGravityConfig(cooling_rate=0.24, activation_decay=0.2))
    memory.remember(
        "cat",
        "event",
        {"text": "Кот поймал рыбу", "roles": {"agent": "кот", "object": "рыба"}},
        topics=["кот", "рыба"],
    )
    for _ in range(4):
        memory.tick(["морфемы", "буквы"])
    assert memory.items["cat"].layer in {MemoryLayer.WARM, MemoryLayer.COLD, MemoryLayer.ARCHIVE}
    memory.remember(
        "morphology", "event", {"text": "Как хранить морфемы"}, topics=["морфемы", "буквы"]
    )
    assert len(memory.clusters) == 2
    reactivated = memory.reactivate(["кот"])
    assert [item.item_id for item in reactivated] == ["cat"]
    assert memory.items["cat"].layer == MemoryLayer.HOT
    assert any(event["event_type"] == "REACTIVATED" for event in memory.events)


def test_eviction_never_removes_unresolved_or_pinned_items():
    memory = ThermoGravityMemory(
        ThermoGravityConfig(max_items=2, cooling_rate=0.5, activation_decay=0.5)
    )
    memory.remember("unresolved", "slot", {"label": "agent"}, topics=["рыба"], unresolved_support=1)
    memory.remember(
        "decision", "constraint", {"label": "не удалять"}, topics=["решение"], pinned=True
    )
    memory.remember("noise", "candidate", {"label": "шум"}, topics=["шум"], retention=0)
    for _ in range(4):
        memory.tick(["другая", "тема"])
    memory.evict()
    assert "unresolved" in memory.items
    assert "decision" in memory.items
    assert "noise" not in memory.items


def test_eviction_enforces_capacity_and_removes_dangling_links():
    memory = ThermoGravityMemory(ThermoGravityConfig(max_items=3))
    for index in range(10):
        memory.remember(
            f"noise-{index}",
            "candidate",
            {"label": f"шум {index}"},
            topics=[f"шум{index}"],
            retention=0,
        )
    memory.items["noise-0"].links.add("noise-9")

    evicted = memory.evict()

    assert len(memory.items) == 3
    assert len(evicted) == 7
    assert all(not item.links.intersection(evicted) for item in memory.items.values())


def test_lazy_descent_composes_missing_diminutive_plural_and_returns_up():
    state = HiveState("forms")
    result = HiveDispatcher().compose_form(
        state,
        "мячик",
        {"number": "plur", "diminutive": True},
        root="мяч",
    )
    assert result["surface"] == "мячики"
    assert result["status"] == ResultStatus.COMPOSED.value
    assert result["descent_path"] == (
        "concept_space",
        "word_space",
        "morpheme_space",
        "symbol_space",
        "morpheme_space",
        "word_space",
    )
    assert [item["direction"] for item in state.vertical_transitions] == [
        "down",
        "down",
        "down",
        "up",
        "up",
    ]
    assert len(state.spaces.morpheme.objects) == 3
    assert len(state.spaces.symbol.objects) == len("мячики")
    assert "word:мячик:мячики" in state.spaces.word.objects

    cached = HiveDispatcher().compose_form(
        state,
        "мячик",
        {"number": "plur", "diminutive": True},
        root="мяч",
    )
    assert cached["status"] == ResultStatus.RETRIEVED.value
    assert cached["descent_path"] == ("concept_space", "word_space")


def test_missing_fact_does_not_trigger_morpheme_fact_invention():
    state = HiveState("guard")
    result = {
        "message_id": "one",
        "query_frame": {
            "intent": "QUESTION",
            "requested_role": "agent",
            "roles": {"action": {"lemma": "есть"}, "object": {"lemma": "рыба"}},
        },
        "memory_scenes": None,
        "candidates": None,
        "answer": {"status": "UNRESOLVED", "answer_mode": "unknown", "confidence": 0},
    }
    HiveDispatcher().process(state, "Кто ест рыбу?", result)
    assert state.answer["status"] == ResultStatus.UNVERIFIED.value
    assert state.answer["lower_levels_created_fact"] is False
    assert not state.spaces.morpheme.objects
    assert not state.spaces.symbol.objects
    assert all(transition["to"] != "word_space" for transition in state.vertical_transitions)


def test_worker_bee_keeps_query_features_when_evaluating_neighbours():
    space = EventSpace()
    source = space.register(
        CloudObject(
            "event:cat",
            "Кот ест",
            {"agent": "кот", "action": "есть"},
            links={"related": ["event:dog"]},
        )
    )
    space.register(CloudObject("event:dog", "Собака спит", {"agent": "собака", "action": "спать"}))
    task = BeeTask("event_space", "retrieve", {"agent": "кот"}, budget=10, min_relevance=0.5)
    seed = NectarPacket("event_space", source.object_id, (), 0.9, 0.9, 1, {}, "scout", task.task_id)

    packets = WorkerBee().run(space, task, [seed], BeeBudget(10))

    assert all(packet.source_id != "event:dog" for packet in packets)


def test_cold_cluster_compression_targets_overflow_once():
    memory = ThermoGravityMemory(
        ThermoGravityConfig(
            max_cold_clusters=2,
            cooling_rate=0.4,
            activation_decay=0.4,
        )
    )
    for index in range(6):
        memory.remember(
            f"item-{index}",
            "event",
            {"text": f"уникальная тема {index}"},
            topics=[f"тема{index}"],
        )
    for _ in range(4):
        memory.tick(["другая-тема"])

    assert sum(item.compression_state == "CLUSTER_SUMMARY" for item in memory.items.values()) == 4
    assert memory.compress() == []
    compressed = next(
        item for item in memory.items.values() if item.compression_state == "CLUSTER_SUMMARY"
    )
    memory.remember(
        compressed.item_id,
        "event",
        {"text": "новое полное наблюдение"},
        topics=compressed.topics,
    )
    assert compressed.compression_state == "RAW"


def test_dispatcher_preserves_observer_trace_and_bounds_space_size():
    state = HiveState("bounded")
    state.memory = ThermoGravityMemory(ThermoGravityConfig(max_items=5))
    state.limits["max_event_objects"] = 8
    dispatcher = HiveDispatcher()
    for turn in range(14):
        dispatcher.process(
            state,
            f"Сущность {turn} действует",
            {
                "message_id": str(turn),
                "query_frame": {
                    "intent": "STATEMENT",
                    "roles": {
                        "agent": {"lemma": f"сущность{turn}"},
                        "action": {"lemma": "действовать"},
                    },
                },
                "memory_scenes": [],
                "candidates": [],
                "answer": {"status": "UNRESOLVED", "confidence": 0},
            },
        )

    assert len(state.spaces.event.objects) <= 8
    assert len(state.memory.items) <= 5
    assert f"event:query:{state.turn}" in state.spaces.event.objects
    assert "observer_allocations" in state.current_trace
    assert state.reasoning_ticks[-1]["pruned"]
    assert state.reasoning_ticks[-1]["tick"] == 14


def test_resolved_answer_clears_unresolved_memory_protection():
    state = HiveState("resolved")
    dispatcher = HiveDispatcher()
    dispatcher.process(
        state,
        "Кто ест рыбу?",
        {
            "message_id": "question",
            "query_frame": {
                "intent": "QUESTION",
                "requested_role": "agent",
                "roles": {"action": {"lemma": "есть"}, "object": {"lemma": "рыба"}},
            },
            "memory_scenes": [{"id": "nullable", "roles": {}, "scores": None}],
            "candidates": [],
            "answer": {"status": "UNRESOLVED", "confidence": 0},
        },
    )
    memory_item = state.memory.items[f"event:query:{state.turn}"]
    assert memory_item.unresolved_support == 0.9

    dispatcher.refresh_answer(
        state,
        {
            "hive": {
                "answer": {"status": "PENDING", "confidence": 0},
                "candidates": [{"lemma": "кот", "status": "stable", "confidence": 0.9}],
            }
        },
    )
    assert memory_item.unresolved_support == 0.9

    dispatcher.refresh_answer(
        state,
        {
            "hive": {
                "answer": {"status": "RESOLVED", "surface_answer": "Кот.", "confidence": 0.9},
                "candidates": [{"lemma": "кот", "status": "winner", "confidence": 0.9}],
            }
        },
    )

    assert memory_item.unresolved_support == 0
    assert state.current_trace["answer_source"]["winner"]["lemma"] == "кот"


def test_multilevel_api_exposes_memory_views_traces_and_form_assembly():
    TrainingPipelineV2().train("Кот поймал рыбу у реки.")
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]
    query = service.query(hive_id, "Где кот поймал рыбу?")
    assert query["multilevel"]["spaces"]["event_space"]["object_count"] >= 2

    with TestClient(app) as client:
        state = client.get(f"/api/v2/hives/{hive_id}/multilevel")
        assert state.status_code == 200
        assert state.json()["memory"]["layers"]["hot"] >= 1
        views = client.get(f"/api/v2/hives/{hive_id}/multilevel/views")
        assert views.status_code == 200
        assert set(views.json()) >= {
            "global",
            "hive",
            "topics",
            "vertical_transition",
            "explanation",
        }
        traces = client.get(f"/api/v2/hives/{hive_id}/multilevel/traces")
        assert traces.status_code == 200
        assert traces.json()["traces"][0]["trace"]["prompt"] == "Где кот поймал рыбу?"
        composed = client.post(
            f"/api/v2/hives/{hive_id}/multilevel/compose-form",
            json={
                "concept": "мячик",
                "root": "мяч",
                "features": {"number": "plur", "diminutive": True},
            },
        )
        assert composed.status_code == 200
        assert composed.json()["result"]["surface"] == "мячики"
