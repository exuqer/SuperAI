import uuid

from server.v2.hive import V2HiveService
from server.v2.analytics import HiveAnalyticsService
from server.v2.query_scene import QuerySceneService
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


def test_observed_noun_is_not_rejected_as_agent_by_builtin_vocabulary():
    service, hive_id, result = _ask("Рыба ест траву.", "Кто ест траву?")
    assert [item["lemma"] for item in result["candidates"]] == ["рыба"]
    assert service.vibration_run(hive_id, 3)["answer"]["resolved_value"] == "рыба"


def test_strong_candidates_produce_a_deterministic_answer_after_bounded_vibration():
    service, hive_id, result = _ask("Кот ест рыбу. Кошечка ест рыбу.", "Кто ест рыбу?")

    assert len(result["candidates"]) == 2
    final = service.vibration_run(hive_id, 3)

    assert final["answer"]["status"] == "RESOLVED"
    assert final["answer"]["full_surface_answer"] == "Кот ест рыбу."
    assert [message["role"] for message in final["hive"]["messages"]] == ["user", "assistant"]


def test_bounded_vibration_selects_best_admitted_candidate_with_a_narrow_gap():
    service = QuerySceneService()
    candidates = [
        {
            "lemma": "кот",
            "status": "stable",
            "hard_forbidden": False,
            "stable_steps": 2,
            "scores": {"decision_score": .70, "total": .70},
        },
        {
            "lemma": "кошка",
            "status": "stable",
            "hard_forbidden": False,
            "stable_steps": 2,
            "scores": {"decision_score": .69, "total": .69},
        },
    ]

    winner = service._winner(
        {"candidates": candidates, "vibration": {"current_step": 2}},
        {**service._defaults(), "max_steps": 3},
    )

    assert winner is candidates[0]
    assert winner["selection_reason"] == "лучший устойчивый кандидат выбран после ограниченной вибрации"


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


def test_unmatched_explicit_anchor_stays_visible_but_is_not_answer_candidate():
    service, hive_id, _ = _ask("Кот ест рыбу.", "Кто там ест на рынке что?")

    answer = service.vibration_run(hive_id, 3)["answer"]

    state = service.query_working_state(hive_id)
    assert answer["resolved_value"] is None
    assert state["candidates"] == []
    assert state["memory_scenes"][0]["anchor_validation"]["status"] == "FAILED"
    assert "location" in state["memory_scenes"][0]["anchor_validation"]["failed_roles"]


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
    assert stages[:4] == [
        "INTENT_CLASSIFICATION", "QUERY_FRAME", "CONTEXT_INHERITANCE",
        "QUERY_SCENE_COMPLETION",
    ]
    assert "MEMORY_SCENE_SEARCH" in stages
    assert "CANDIDATE_RANKING" in stages
    assert stages.index("MEMORY_SCENE_SEARCH") + 1 == stages.index("CANDIDATE_RANKING")
    assert "VIBRATION" in stages
    assert "ANSWER_ASSEMBLY" in stages


def test_plain_greeting_is_not_routed_to_structural_resonance():
    service, hive_id, result = _ask("Рыбу продают на рынке.", "Привет!")

    assert result["query_scene"] is None
    assert result["query_frame"]["intent"] == "GREETING"
    assert result["answer"]["status"] == "RESOLVED_GREETING"
    assert result["answer"]["surface_answer"] == "Здравствуйте!"


def test_source_question_prefers_complete_market_scene_and_excludes_service_word_noise():
    memory = (
        "Рыбу продают на рынке. Рыба из моря. Рыбак приносит рыбу на рынок. "
        "Кошечка спит на солнце. В магазине продают мясо и кошачий корм."
    )
    TrainingPipelineV2().train(memory)
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]

    first = service.query(hive_id, "Что там на рынке?")
    assert all("солнце" not in source["label"].casefold() for source in first["external_search"]["sources"])
    assert service.vibration_run(hive_id, 3)["answer"]["full_surface_answer"] == "Продают рыбу на рынке."

    second = service.query(hive_id, "А откуда рыба на рынке?")
    assert second["resolved_mode"] == "NEW_QUERY"
    assert second["query_frame"]["requested_role"] == "source"
    assert second["candidates"][0]["answer_mode"] == "explanation"

    final = service.vibration_run(hive_id, 3)
    assert final["answer"]["status"] == "RESOLVED"
    assert final["answer"]["full_surface_answer"] == "Рыбак приносит рыбу на рынок."
    assert final["answer"]["semantic_total"] > 0.8
    assert final["answer"]["decision_score"] > 0.6

    current = HiveAnalyticsService().get(hive_id)["current"]
    assert current["query_components"][-1] == {
        "term": "откуда",
        "role": "source",
        "word_form_cloud_id": None,
    }
    assert current["snapshot"]["candidates"][0]["scene_label"] == "Рыбак приносит рыбу на рынок."
    assert current["snapshot"]["candidates"][0]["semantic_score"] > 0.8
    assert all("солнце" not in item["scene_label"].casefold() for item in current["snapshot"]["candidates"])


def test_market_dialogue_context_resolves_tam_without_switching_to_store_goods():
    memory = (
        "Рыбу продают на рынке. Удочку продают в рыболовном магазине. "
        "В магазине продают мясо и кошачий корм."
    )
    TrainingPipelineV2().train(memory)
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]

    service.query(hive_id, "Как дела на рынке?")
    follow_up = service.query(hive_id, "Что продают там?")

    assert follow_up["resolved_mode"] == "FOLLOW_UP"
    assert follow_up["query_frame"]["context_resolution"]["status"] == "RESOLVED"
    assert follow_up["query_frame"]["roles"]["location"]["lemma"] == "рынок"
    answer = service.vibration_run(hive_id, 3)["answer"]
    assert answer["surface_answer"] == "Рыбу."
    assert answer["full_surface_answer"] == "Продают рыбу на рынке."


def test_referential_query_without_dialogue_context_does_not_guess_a_location():
    service, hive_id, result = _ask("Рыбу продают на рынке. Удочку продают в магазине.", "Что продают там?")

    assert result["resolved_mode"] == "FOLLOW_UP"
    assert result["query_frame"]["context_resolution"]["status"] == "UNRESOLVED_CONTEXT"
    assert result["candidates"] == []
    assert service.vibration_run(hive_id, 3)["answer"]["status"] == "UNRESOLVED"


def test_resolved_answer_is_kept_for_possessive_follow_up_and_source_ranking():
    memory = (
        "Кот ест рыбу. Рыбу продают на рынке. Рыбак приносит рыбу на рынок. "
        "Рыбак ловит рыбу удочкой. Рыба из моря."
    )
    TrainingPipelineV2().train(memory)
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]

    service.query(hive_id, "Кто ест рыбу?")
    first_answer = service.vibration_run(hive_id, 3)["answer"]
    assert first_answer["full_surface_answer"] == "Кот ест рыбу."

    follow_up = service.query(hive_id, "Откуда у него рыба?")

    assert follow_up["resolved_mode"] == "FOLLOW_UP"
    assert follow_up["query_frame"]["context_resolution"]["status"] == "RESOLVED"
    assert follow_up["query_frame"]["context_resolution"]["role"] == "possessor"
    assert follow_up["query_frame"]["context_resolution"]["value"]["lemma"] == "кот"
    assert follow_up["query_frame"]["roles"]["object"]["lemma"] == "рыба"
    assert all(value.get("lemma") != "он" for value in follow_up["query_frame"]["roles"].values())
    assert follow_up["candidates"][0]["lemma"] == "море"
    assert follow_up["candidates"][0]["scores"]["semantic_total"] > 0.9

    final = service.vibration_run(hive_id, 3)
    assert final["answer"]["surface_answer"] == "Из моря."
    assert final["answer"]["full_surface_answer"] == "Рыба из моря."

    current = HiveAnalyticsService().get(hive_id)["current"]
    assert current["snapshot"]["candidates"][0]["scene_label"] == "Рыба из моря."


def test_ellipsis_follow_up_uses_persistent_dialogue_fact_and_excludes_last_object():
    TrainingPipelineV2().train("Лисичка ест ягоду. Лисичка ест грушу.")
    service = V2HiveService()
    hive_id = service.create(conversation_id=f"ellipsis-dialogue-memory-{uuid.uuid4()}")["hive"]["id"]

    service.query(hive_id, "Лисичка ест ягоду.")
    follow_up = service.query(hive_id, "А ещё что?")

    assert follow_up["resolved_mode"] == "FOLLOW_UP"
    assert follow_up["query_frame"]["reconstructed_query"] == "Что ещё ест Лисичка, кроме ягоду?"
    assert follow_up["query_frame"]["roles"]["agent"]["lemma"] == "лисичка"
    assert follow_up["query_frame"]["roles"]["action"]["lemma"] == "есть"
    assert follow_up["query_frame"]["excluded_roles"]["object"][0]["lemma"] == "ягода"
    assert [candidate["lemma"] for candidate in follow_up["candidates"]] == ["груша"]
    assert any(scene["provenance"]["source"] == "dialogue_memory" for scene in follow_up["memory_scenes"])


def test_resolved_assistant_turn_is_persisted_once_with_session_memory():
    TrainingPipelineV2().train("Белочка ест орех.")
    service = V2HiveService()
    conversation_id = f"assistant-turn-dialogue-memory-{uuid.uuid4()}"
    hive_id = service.create(conversation_id=conversation_id)["hive"]["id"]

    service.query(hive_id, "Что ест белочка?")
    service.vibration_run(hive_id, 3)
    service.vibration_run(hive_id, 3)

    restored = service.create(conversation_id=conversation_id)
    messages = restored["messages"]
    assert restored["hive"]["id"] == hive_id
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[-1]["text"] == "Белочка ест орех."
