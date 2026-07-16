from server.v2.hive import V2HiveService
from server.v2.repository import V2Repository
from server.v2.training import TrainingPipelineV2
from server.v2.validation import ModelInvariantValidator


def _service():
    TrainingPipelineV2().train("Кот ест рыбу. Кот ест птицу.")
    service = V2HiveService()
    hive = service.create()
    return service, hive["hive"]["id"]


def test_repeated_query_uses_local_resonance_without_search():
    service, hive_id = _service()
    service.query(hive_id, "Кот ест рыбу")
    result = service.query(hive_id, "Кот ест рыбу")
    assert result["decision"]["decision"] == "LOCAL_HIT"
    assert result["metrics"]["bees"] == 0
    assert result["metrics"]["created_cells"] == 0


def test_partial_query_searches_only_unresolved_component():
    service, hive_id = _service()
    service.query(hive_id, "Кот ест рыбу")
    result = service.query(hive_id, "Кот ест птицу")
    assert result["decision"]["decision"] == "PARTIAL_HIT"
    assert [item["normalized_form"] for item in result["decision"]["unresolved_components"]] == ["птицу"]
    assert "кот" in result["decision"]["external_request"]["excluded_known_components"]


def test_hive_uses_local_placements_and_chat_does_not_change_global_mass():
    service, hive_id = _service()
    repository = V2Repository()
    with repository.transaction() as conn:
        before = {row["id"]: row["mass"] for row in conn.execute("SELECT id, mass FROM clouds")}
    result = service.query(hive_id, "Кот ест рыбу")
    with repository.transaction() as conn:
        after = {row["id"]: row["mass"] for row in conn.execute("SELECT id, mass FROM clouds")}
        for cell in result["cells"]:
            placement = conn.execute("SELECT * FROM cloud_placements WHERE id = ?", (cell["hive_placement_id"],)).fetchone()
            assert placement["space_id"] == result["hive"]["space_id"]
            assert cell["source_placement_id"] != cell["hive_placement_id"]
            assert cell["source_space_id"] is not None
    assert after == before
    assert ModelInvariantValidator().validate()["valid"]


def test_each_question_fetches_a_missing_answer_role_from_global_field():
    TrainingPipelineV2().train("Кот ест рыбу. Рыба живет в пруду.")
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]
    service.query(hive_id, "Кот ест рыбу")

    result = service.query(hive_id, "Где рыба?")

    role_slot = next(
        item
        for item in result["decision"]["unresolved_components"]
        if item.get("component_type") == "answer_role_slot"
    )
    assert role_slot["expected_role"] == "location"
    assert result["decision"]["external_request"]["required_answer_roles"] == ["location"]
    assert result["metrics"]["bees"] > 0
    assert result["metrics"]["created_cells"] > 0
