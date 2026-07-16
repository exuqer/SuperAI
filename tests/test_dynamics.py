import pytest

from server.v2.dynamics import DynamicsConfig, DynamicsEngine, DynamicsNodeState, DynamicsState
from server.v2.hive import V2HiveService
from server.v2.training import TrainingPipelineV2


def make_state(seed=7):
    return DynamicsState(
        initial_temperature=0.4,
        current_temperature=0.4,
        minimum_temperature=0.05,
        maximum_temperature=1.0,
        cooling_rate=0.7,
        random_seed=seed,
        anchors=[{"anchor_id": "slot-location", "role": "location", "position": {"x": 0.8, "y": 0.5}}],
        nodes=[
            DynamicsNodeState(
                cell_id="market",
                label="рынок",
                role="location",
                position_x=0.2,
                position_y=0.5,
                global_mass=0.5,
                activation=0.8,
                retention=0.8,
                resonance=0.7,
                query_relevance=0.8,
                role_compatibility=1.0,
                source_confidence=0.8,
                semantic_support=0.7,
            ),
            DynamicsNodeState(
                cell_id="shop",
                label="магазин",
                role="location",
                position_x=0.3,
                position_y=0.5,
                global_mass=0.4,
                activation=0.5,
                retention=0.5,
                resonance=0.4,
                query_relevance=0.4,
                role_compatibility=0.4,
                source_confidence=0.4,
                semantic_support=0.4,
            ),
        ],
    )


def test_temperature_cools_and_mass_is_separate_from_gravity():
    state = make_state()
    DynamicsEngine(DynamicsConfig.from_dict({"temperature": {"default": 0.4, "cooling_rate": 0.7}})).step(state)
    assert state.current_temperature == pytest.approx(0.28)
    assert state.nodes[0].local_mass != state.nodes[0].local_gravity


def test_seed_reproduces_thermal_force_and_trajectory():
    left, right = make_state(11), make_state(11)
    engine = DynamicsEngine()
    engine.step(left)
    engine.step(right)
    assert left.nodes[0].force_breakdown == right.nodes[0].force_breakdown
    assert left.nodes[0].trajectory == right.nodes[0].trajectory


def test_role_and_competition_forces_are_explainable():
    state = make_state()
    DynamicsEngine().step(state)
    force_types = {item["type"] for item in state.nodes[0].force_breakdown}
    assert "ROLE_FORCE" in force_types
    assert "COMPETITION_FORCE" in force_types
    assert state.nodes[0].distance_to_target < state.nodes[0].distance_to_core + 0.7


def test_grace_period_prevents_first_step_eviction():
    state = make_state()
    node = state.nodes[0]
    node.activation = node.retention = node.resonance = 0.0
    node.grace_steps = 1
    DynamicsEngine().step(state)
    assert node.eviction_status != "EVICTED"
    assert state.history and state.history[0]["nodes"]


def test_memory_source_only_hive_vibrates_for_requested_steps():
    TrainingPipelineV2().train("Лисичка ест ягоду. Лисичка ест грушу.")
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]

    service.query(hive_id, "Лисичка ест ягоду.")
    service.query(hive_id, "А ещё что?")
    service.vibration_run(hive_id, 3)
    repeated = service.query(hive_id, "А ещё что?")

    assert repeated["candidates"] == []
    assert any(cell["component_class"] == "memory_source" for cell in repeated["cells"])

    result = service.vibration_run(hive_id, 3)
    dynamics = result["hive"]["dynamics"]

    assert result["steps_completed"] == 3
    assert len(result["hive"]["vibration"]["history"]) == 3
    assert dynamics["step"] == 3
    assert dynamics["status"] in {"STABLE", "STABILIZING"}
    assert dynamics["nodes"]
    assert dynamics["temperature"]["current"] < dynamics["temperature"]["initial"]
    assert len(dynamics["nodes"][0]["trajectory"]) == 4


def test_query_uses_the_trained_role_of_an_instrumental_object():
    TrainingPipelineV2().train("Рыбак питается овощами. Питаться — это есть.")
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]

    result = service.query(hive_id, "Что ест рыбак?")

    source = next(scene for scene in result["memory_scenes"] if scene["source_text"] == "Рыбак питается овощами.")
    assert source["roles"]["object"]["lemma"] == "овощ"
    assert [candidate["lemma"] for candidate in result["candidates"]] == ["овощ"]
