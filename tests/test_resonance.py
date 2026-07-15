from server.v2.hive import V2HiveService
from server.v2.training import TrainingPipelineV2


def _service():
    TrainingPipelineV2().train("Рыбу продают на рынке. Кот ест рыбу. Котик любит рыбку.")
    service = V2HiveService()
    return service, service.create()["hive"]["id"]


def test_structural_fragment_uses_probe_without_query_scene_or_auto_import():
    service, hive_id = _service()
    result = service.query(hive_id, "рыб", resonance_scope="LOCAL_THEN_GLOBAL")
    probe = result["resonance_probe"]
    assert result["intent"] == "STRUCTURAL_PROBE"
    assert result["query_scene"] is None
    assert result["display_status"] == "GLOBAL_MATCHES_FOUND"
    assert [item["value"] for item in probe["global_results"][:4]] == ["рыба", "рыбу", "рыбка", "рыбку"]
    assert result["stats"]["working_cells"] == 0
    assert result["vibration"]["enabled"] is False


def test_probe_import_creates_a_lexical_seed():
    service, hive_id = _service()
    result = service.query(hive_id, "рыб", resonance_scope="LOCAL_THEN_GLOBAL")
    probe = result["resonance_probe"]
    imported = service.resonance_import(hive_id, probe["id"], probe["global_results"][0]["id"])
    state = service.query_working_state(hive_id)
    assert imported["imported_cells"][0]["component_class"] == "lexical_seed"
    assert state["stats"]["working_cells"] == 1
    assert state["vibration"]["enabled"] is True


def test_unknown_fragment_completes_without_scene():
    service, hive_id = _service()
    result = service.query(hive_id, "авцч", resonance_scope="LOCAL_THEN_GLOBAL")
    assert result["resonance_probe"]["status"] == "COMPLETED_NO_MATCH"
    assert result["query_scene"] is None
    assert result["vibration"]["enabled"] is False
