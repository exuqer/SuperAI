from server.v2.hive import V2HiveService
from server.v2.training import TrainingPipelineV2


def _ask(memory: str, query: str):
    TrainingPipelineV2().train(memory)
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]
    return service, hive_id, service.query(hive_id, query)


def test_query_scene_keeps_empty_agent_slot_and_partial_sale_match():
    _, _, result = _ask("Рыбу продают на рынке.", "Кто покупает рыбу?")
    assert result["query_frame"]["requested_role"] == "agent"
    assert result["query_scene"]["slots"][0]["status"] == "empty"
    sale = next(scene for scene in result["memory_scenes"] if "прода" in scene["source_text"])
    assert sale["result_type"] == "PARTIAL_HIT"
    assert result["answer"]["answer_mode"] == "partial"


def test_vibration_resolves_full_role_hit_into_answer_frame():
    service, hive_id, result = _ask("Покупатель покупает рыбу. Кот ест рыбу.", "Кто покупает рыбу?")
    assert result["query_scene"]["status"] == "INCOMPLETE"
    final = service.vibration_run(hive_id, 3)
    assert final["winner"]["lemma"] == "покупатель"
    assert final["answer"]["answer_mode"] == "exact"
    assert final["answer"]["surface_answer"] == "Покупатель."
    assert final["answer"]["full_surface_answer"] == "Покупатель покупает рыбу."
    assert final["hive"]["query_scene"]["slots"][0]["status"] == "RESOLVED"
    assert final["hive"]["vibration"]["history"]


def test_question_roles_object_location_and_instrument():
    service, hive_id, _ = _ask("Кот ест рыбу на кухне. Рыбак ловит рыбу сетью.", "Что ест кот?")
    object_answer = service.vibration_run(hive_id, 3)["answer"]
    assert object_answer["surface_answer"] == "Рыбу."
    assert object_answer["full_surface_answer"] == "Кот ест рыбу."

    location = service.query(hive_id, "Где кот ест рыбу?")
    assert location["query_frame"]["requested_role"] == "location"
    location_answer = service.vibration_run(hive_id, 3)["answer"]
    assert location_answer["surface_answer"] == "На кухне."
    assert location_answer["full_surface_answer"] == "Кот ест рыбу на кухне."

    instrument = service.query(hive_id, "Чем рыбак ловит рыбу?")
    assert instrument["query_frame"]["requested_role"] == "instrument"
    instrument_answer = service.vibration_run(hive_id, 3)["answer"]
    assert instrument_answer["surface_answer"] == "Сетью."
    assert instrument_answer["full_surface_answer"] == "Рыбак ловит рыбу сетью."


def test_full_answer_puts_selected_agent_into_agent_slot():
    service, hive_id, _ = _ask("Кот ест рыбу.", "Кто там ест на рынке что?")

    answer = service.vibration_run(hive_id, 3)["answer"]

    assert answer["resolved_value"] == "кот"
    assert answer["surface_answer"] == "Кот."
    assert answer["full_surface_answer"] == "Кот ест на рынке."


def test_new_statement_replaces_the_previous_query_scene():
    service, hive_id, _ = _ask("Рыбу продают на рынке.", "Где купить рыбу?")
    service.vibration_run(hive_id, 3)

    result = service.query(hive_id, "Рыбу")

    assert result["query_frame"]["source_text"] == "Рыбу"
    assert result["query_scene"]["source_query"] == "Рыбу"
    assert result["query_scene"]["requested_role"] is None
    assert service.query_working_state(hive_id)["query_scene"]["source_query"] == "Рыбу"


def test_negative_memory_scene_stays_conflict_not_answer():
    service, hive_id, result = _ask("Кот не покупает рыбу.", "Кто покупает рыбу?")
    assert result["memory_scenes"][0]["result_type"] == "CONFLICT_HIT"
    final = service.vibration_run(hive_id, 3)
    assert final["winner"] is None
    assert final["answer"]["answer_mode"] == "partial"


def test_greeting_with_scene_question_uses_location_as_semantic_anchor():
    service, hive_id, result = _ask("Рыбу продают на рынке.", "Привет! Что там на рынке?")

    assert result["query_frame"]["intent"] == "SCENE_QUESTION"
    assert result["query_frame"]["intent_classification"]["greeting"]["surface"] == "Привет"
    assert result["query_frame"]["requested_role"] == "object"
    market = next(scene for scene in result["memory_scenes"] if "рынке" in scene["source_text"])
    assert market["matched_roles"] == ["location"]
    assert result["candidates"][0]["lemma"] == "рыба"
    assert result["candidates"][0]["surface"] == "рыбу"
    assert result["candidates"][0]["grammatical_features"]["case"] == "accs"
    assert result["candidates"][0]["grammatical_features"]["number"] == "sing"
    assert result["candidates"][0]["form_provenance"] == {
        "source_type": "observed_training_form",
        "scene_id": market["id"],
        "scene_text": "Рыбу продают на рынке.",
        "observed_surface": "рыбу",
        "generated": False,
    }

    final = service.vibration_run(hive_id, 3)
    assert final["answer"]["status"] == "RESOLVED"
    assert final["answer"]["surface_answer"] == "Рыбу."
    assert final["answer"]["full_surface_answer"] == "Продают рыбу на рынке."
    assert final["hive"]["full_sentence_plan"]["source_scene_text"] == "Рыбу продают на рынке."
    assert [slot["role"] for slot in final["hive"]["full_sentence_plan"]["slots"]] == ["action", "object", "location"]
    assert all(not slot["requested_features"] for slot in final["hive"]["full_sentence_plan"]["slots"])
    assert final["hive"]["reverse_validation"]["checks"]["action_preserved"] is True
    assert final["hive"]["sentence_plan"]["slots"][0]["source_type"] == "known_word_form"
    assert final["hive"]["sentence_plan"]["slots"][0]["observed_features"]["case"] == "accs"
    assert final["hive"]["morphology_trace"][0]["selection_mode"] == "reuse_observed_training_form"
    stages = [item["stage"] for item in final["hive"]["reasoning_trace"]["stages"]]
    assert stages[:4] == ["INTENT_CLASSIFICATION", "QUERY_FRAME", "MEMORY_SCENE_SEARCH", "CANDIDATE_RANKING"]
    assert "VIBRATION" in stages
    assert "ANSWER_ASSEMBLY" in stages


def test_plain_greeting_is_not_routed_to_structural_resonance():
    service, hive_id, result = _ask("Рыбу продают на рынке.", "Привет!")

    assert result["query_scene"] is None
    assert result["query_frame"]["intent"] == "GREETING"
    assert result["answer"]["status"] == "RESOLVED_GREETING"
    assert result["answer"]["surface_answer"] == "Здравствуйте!"
