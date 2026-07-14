from server.v2.hive import V2HiveService
from server.v2.training import TrainingPipelineV2
from server.training import TrainingManager
from server.v2.validation import ModelInvariantValidator


def _hive(*sentences: str):
    pipeline = TrainingPipelineV2()
    for sentence in sentences:
        pipeline.train(sentence)
    return V2HiveService(), V2HiveService().create()


def test_repeated_query_uses_local_resonance_without_search():
    service, hive = _hive("Кот ест рыбу.")
    service.query(hive["hive"]["id"], "Кот ест рыбу")
    result = service.query(hive["hive"]["id"], "Кот ест рыбу")
    assert result["decision"]["decision"] == "LOCAL_HIT"
    assert result["metrics"]["bees"] == 0
    assert result["metrics"]["created_cells"] == 0


def test_partial_query_searches_only_unresolved_component():
    service, hive = _hive("Кот ест рыбу.", "Кот ест птицу.")
    service.query(hive["hive"]["id"], "Кот ест рыбу")
    result = service.query(hive["hive"]["id"], "Кот ест птицу")
    assert result["decision"]["decision"] == "PARTIAL_HIT"
    assert [item["normalized_form"] for item in result["decision"]["unresolved_components"]] == ["птицу"]
    assert "кот" in result["decision"]["external_request"]["excluded_known_components"]


def test_negation_is_a_conflict_and_values_stay_bounded():
    service, hive = _hive("Кот ест рыбу.")
    service.query(hive["hive"]["id"], "Кот ест рыбу")
    result = service.query(hive["hive"]["id"], "Кот не ест рыбу")
    assert result["decision"]["decision"] == "CONFLICT"
    assert ModelInvariantValidator().validate()["valid"]


def test_hive_synchronizes_the_existing_legacy_field():
    TrainingManager().learn("Рыбак ловит рыбу.")

    result = V2HiveService().forage("Рыбак")

    assert result["external_search"]["sources"]
    assert result["cells"]
    assert result["metrics"]["created_cells"] > 0
