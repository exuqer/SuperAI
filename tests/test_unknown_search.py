from server.v2.hive import V2HiveService
from server.v2.repository import V2Repository
from server.v2.training import TrainingPipelineV2


def test_unknown_derivative_uses_indexed_lexeme_as_temporary_hive_bridge():
    repository = V2Repository()
    TrainingPipelineV2(repository).train("Рыбу продают на рынке.")
    with repository.transaction() as conn:
        before = repository.stats(conn)
    service = V2HiveService(repository)
    hive_id = service.create()["hive"]["id"]

    result = service.query(hive_id, "Где купить рыбки?")
    search = result["unknown_token_searches"][0]

    assert search["surface"] == "рыбки"
    assert search["status"] == "probable_match"
    assert search["lemma_hypotheses"][0]["lemma"] == "рыбка"
    assert search["selected_candidate"]["candidate_lexeme"] == "рыба"
    assert search["selected_candidate"]["scores"]["structural_total"] <= .60
    assert any(item["bee_type"] == "SCENE_BEE" for item in search["evidence"])
    assert {"semantic_bridge", "role_candidate"} <= {cell["component_class"] for cell in result["cells"]}
    cell_classes = {cell["component_class"] for cell in service.get_hive(hive_id)["cells"]}
    assert {"semantic_bridge", "role_candidate"} <= cell_classes
    state = service.query_working_state(hive_id)
    assert state["pipeline"]["memory_search"]["status"] == "ROLE_CANDIDATES_FOUND"
    assert state["answer"]["status"] == "PENDING"
    assert state["stats"]["cells"] == 2
    final = service.vibration_run(hive_id, 3)
    assert final["answer"]["status"] == "RESOLVED"
    assert final["hive"]["pipeline"]["query_scene"]["status"] == "RESOLVED"
    analytics = service.analytics(hive_id)
    assert analytics["primary"]["run"]["id"].startswith("query-vibration-")
    assert len(analytics["primary"]["snapshots"]) >= 2
    assert analytics["primary"]["snapshots"][-1]["candidates"]
    with repository.transaction() as conn:
        after = repository.stats(conn)
    assert after["clouds_total"] == before["clouds_total"]
