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


def test_regular_message_creates_a_dynamic_resonance_session_without_importing_global_memory():
    service, hive_id = _service()
    before = service.get_hive(hive_id)["cells"]
    result = service.query(hive_id, "Может с рынка?")
    session = result["resonance_session"]
    after = service.get_hive(hive_id)["cells"]

    assert session["status"] == "completed"
    assert session["tick"] >= 1
    assert len(session["snapshots"]) >= 2
    assert session["lexical_candidates"]
    assert all(item["temporary"] for item in session["active_concepts"] if item["source"] == "global")
    assert before == after


def test_dynamic_resonance_keeps_energy_bounded_and_records_tick_state():
    service, hive_id = _service()
    session = service.resonance_create(hive_id, "рыб", temperature=.8, max_ticks=4)
    finished = service.resonance_run(hive_id, session["id"])

    assert finished["status"] == "completed"
    assert finished["snapshots"]
    assert all(snapshot["total_energy"] <= 1.000001 for snapshot in finished["snapshots"])
    assert all(0 <= item["activation"] <= 1 for snapshot in finished["snapshots"] for item in snapshot["concepts"])


def test_global_proxy_is_consolidated_only_after_explicit_import():
    service, hive_id = _service()
    session = service.resonance_create(hive_id, "рыб")
    candidate = session["lexical_candidates"][0]

    assert service.get_hive(hive_id)["cells"] == []
    imported = service.import_resonance_concept(session["id"], candidate["conceptId"])

    assert imported["cell"]["component_class"] == "lexical_seed"
    assert len(service.get_hive(hive_id)["cells"]) == 1
