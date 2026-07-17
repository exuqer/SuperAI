from server.v2.hive import V2HiveService
from server.v2.training import TrainingPipelineV2


def _snapshot(memory: str, query: str, aggregation: str = "lexeme"):
    TrainingPipelineV2().train(memory)
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]
    service.query(hive_id, query)
    return service.snapshot(hive_id, aggregation=aggregation)


def test_snapshot_builds_static_words_for_all_persistent_scenes():
    snapshot = _snapshot("Кот ест рыбу. Кошечка ест рыбу.", "Кто ест рыбу?")

    assert snapshot["scenes"]
    assert snapshot["words"]
    fish = next(word for word in snapshot["words"] if word["lemma"] == "рыба")
    assert fish["scene_support_count"] == 2
    assert len({item["scene_id"] for item in fish["contributions"]}) == 2
    assert snapshot["diagnostics"]["counts"]["projected_words"] == len(snapshot["words"])


def test_snapshot_keeps_word_forms_separate_when_requested():
    snapshot = _snapshot("Кот ест рыбу.", "Кто ест рыбу?", "word_form")

    assert any(word["lemma"] == "рыбу" for word in snapshot["words"])


def test_location_anchored_object_question_returns_a_complete_scene():
    TrainingPipelineV2().train("Рыбу продают на рынке. Рыбак приносит рыбу на рынок.")
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]
    service.query(hive_id, "Что на рынке?")

    answer = service.vibration_run(hive_id, 3)["answer"]

    assert answer["status"] == "RESOLVED"
    assert answer["answer_mode"] == "contextual_scene"
    assert answer["full_surface_answer"] == "Рыбу продают на рынке."


def test_snapshot_projects_working_cells_without_empty_projection_warning():
    snapshot = _snapshot("Кот ест рыбу. Медведь ест рыбу.", "Кто ест рыбу?")

    counts = snapshot["diagnostics"]["counts"]
    assert snapshot["cells"]
    assert counts["cells_total"] == len(snapshot["cells"])
    assert counts["working_cells_total"] == counts["cells_total"]
    assert counts["projected_cells_total"] == len(snapshot["cells"])
    assert counts["filtered_cells_total"] == 0
    assert counts["projection_error"] is None
    assert all(warning["code"] != "WORKING_CELLS_EMPTY" for warning in snapshot["diagnostics"]["warnings"])
