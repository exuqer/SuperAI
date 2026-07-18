import uuid

from server.v2.hive import V2HiveService
from server.v2.analytics import HiveAnalyticsService
from server.v2.query_scene import QuerySceneService
from server.v2.repository import V2Repository
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
    assert final["answer"]["answer_mode"] == "multiple"
    assert final["answer"]["resolved_values"] == ["кот", "кошечка"]
    assert final["answer"]["full_surface_answer"] == "Кот и кошечка."
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


def test_how_question_resolves_to_observed_instrument():
    service, hive_id, result = _ask(
        "Рыбак ловит рыбу удочкой.",
        "Как рыбу ловит рыбак?",
    )

    assert result["query_frame"]["requested_role"] == "instrument"
    assert result["candidates"][0]["lemma"] == "удочка"

    answer = service.vibration_run(hive_id, 3)["answer"]
    assert answer["surface_answer"] == "Удочкой."
    assert answer["full_surface_answer"] == "Рыбак ловит рыбу удочкой."


def test_definition_question_uses_entity_type_relation_and_keeps_answer_derived():
    TrainingPipelineV2().train("Робот Искра находится в мастерской.")
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]

    definition = service.query(hive_id, "Кто такой Искра?")

    assert definition["query_frame"]["query_type"] == "definition_question"
    assert definition["query_frame"]["requested_role"] == "entity_type"
    assert definition["query_frame"]["requested_slot"] == "type_or_definition"
    assert definition["query_frame"]["answer_slot_type"] == "relation"
    assert definition["query_frame"]["roles"]["entity"]["entity_id"]
    assert definition["query_frame"]["roles"]["entity"]["resolution_state"] == "KNOWN_ENTITY_ALIAS"

    answer = service.vibration_run(hive_id, 3)["answer"]

    assert answer["full_surface_answer"] == "Искра — робот."
    assert answer["provenance"]["source_type"] == "assistant_derived_answer"
    assert answer["provenance"]["source_relation_ids"]
    assert answer["provenance"]["independent_fact"] is False
    with V2Repository().transaction() as conn:
        stored = conn.execute(
            """SELECT memory_class,source_type,knowledge_status,independent_evidence,
                      eligible_for_fact_retrieval,provenance_json
               FROM hive_dialogue_scenes WHERE hive_id=? ORDER BY created_at DESC LIMIT 1""",
            (hive_id,),
        ).fetchone()
    assert stored["memory_class"] == "ASSISTANT_DERIVED"
    assert stored["source_type"] == "assistant_derived_answer"
    assert stored["knowledge_status"] == "DERIVED"
    assert stored["independent_evidence"] == 0
    assert stored["eligible_for_fact_retrieval"] == 0
    assert "relation_extraction" in stored["provenance_json"]

    action = service.query(hive_id, "Что делает Искра?")
    assert all("dialogue-scene" not in source for candidate in action["candidates"] for source in candidate["sources"])
    assert service.vibration_run(hive_id, 3)["answer"]["surface_answer"] == "Находится в мастерской."


def test_definition_question_supports_drone_alias_and_complete_action_phrase():
    TrainingPipelineV2().train("Дрон Сокол летит к ангару.")
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]

    service.query(hive_id, "Сокол — это кто?")
    assert service.vibration_run(hive_id, 3)["answer"]["full_surface_answer"] == "Сокол — дрон."

    action = service.query(hive_id, "Что делает Сокол?")
    assert action["candidates"][0]["predicate_phrase_completeness"] == 1.0
    assert service.vibration_run(hive_id, 3)["answer"]["surface_answer"] == "Летит к ангару."


def test_derived_short_dialogue_answer_is_not_fact_evidence_for_action_query():
    TrainingPipelineV2().train("Насос качает воду из резервуара.")
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]
    query_scenes = QuerySceneService()
    message_id = f"message-{uuid.uuid4().hex[:12]}"
    frame = query_scenes.parse("Насос качает.")["query_frame"]
    with V2Repository().transaction() as conn:
        conn.execute(
            "INSERT INTO hive_messages(id,hive_id,turn_index,role,text,parsed_json,created_at) VALUES(?,?,?,?,?,?,?)",
            (message_id, hive_id, 1, "assistant", "Насос качает.", "{}", "2026-01-01T00:00:00+00:00"),
        )
        query_scenes._store_dialogue_scene(
            conn, hive_id, message_id, "assistant", "Насос качает.", frame,
        )
        stored = conn.execute(
            "SELECT completion_status,eligible_for_fact_retrieval FROM hive_dialogue_scenes WHERE message_id=?",
            (message_id,),
        ).fetchone()
    assert stored["completion_status"] == "SEMANTICALLY_INCOMPLETE"
    assert stored["eligible_for_fact_retrieval"] == 0

    result = service.query(hive_id, "Что делает насос?")
    assert all(message_id not in source for candidate in result["candidates"] for source in candidate["sources"])
    assert service.vibration_run(hive_id, 3)["answer"]["surface_answer"] == "Качает воду из резервуара."


def test_user_assertion_is_stored_as_independent_dialogue_memory():
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]

    service.query(hive_id, "Инженер проверяет генератор.")

    with V2Repository().transaction() as conn:
        stored = conn.execute(
            """SELECT memory_class,source_type,independent_evidence,
                      eligible_for_fact_retrieval FROM hive_dialogue_scenes
               WHERE hive_id=? ORDER BY created_at DESC LIMIT 1""",
            (hive_id,),
        ).fetchone()
    assert dict(stored) == {
        "memory_class": "USER_ASSERTION",
        "source_type": "user_assertion",
        "independent_evidence": 1,
        "eligible_for_fact_retrieval": 1,
    }


def test_polar_question_returns_supported_contradicted_or_unknown_answer():
    TrainingPipelineV2().train("Кот ест рыбу.")
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]

    supported = service.query(hive_id, "Кот ест рыбу?")
    assert supported["query_frame"]["intent"] == "SCENE_QUESTION"
    assert supported["query_frame"]["query_type"] == "polar_question"
    assert supported["answer"]["status"] == "RESOLVED"
    assert supported["answer"]["resolved_value"] is True
    assert supported["answer"]["full_surface_answer"] == "Да. Кот ест рыбу."
    assert supported["answer"]["evidence_status"] == "SUPPORTED"
    assert service.query_working_state(hive_id)["vibration"]["status"] == "FINISHED"

    contradicted = service.query(hive_id, "Кот не ест рыбу?")
    assert contradicted["answer"]["status"] == "RESOLVED"
    assert contradicted["answer"]["resolved_value"] is False
    assert contradicted["answer"]["full_surface_answer"] == "Нет. Кот ест рыбу."
    assert contradicted["answer"]["evidence_status"] == "CONTRADICTED"

    unknown = service.query(hive_id, "Рыбак ест?")
    assert unknown["answer"]["status"] == "UNRESOLVED"
    assert unknown["answer"]["resolved_value"] is None
    assert unknown["answer"]["surface_answer"] == "В доступной памяти недостаточно данных."
    assert unknown["answer"]["evidence_status"] == "INSUFFICIENT_EVIDENCE"

    with V2Repository().transaction() as conn:
        stored_questions = conn.execute(
            """SELECT COUNT(*) FROM hive_dialogue_scenes
            WHERE hive_id=? AND RTRIM(source_text) LIKE '%?'""",
            (hive_id,),
        ).fetchone()[0]
    assert stored_questions == 0


def test_need_question_requests_purpose_then_uses_it_to_find_instrument():
    TrainingPipelineV2().train(
        "Рыбак ловит рыбу удочкой. Рыбак питается овощами."
    )
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]

    need = service.query(hive_id, "Что нужно рыбаку?")
    assert need["query_frame"]["query_type"] == "need_question"
    assert need["query_frame"]["requested_role"] == "instrument"
    assert need["query_frame"]["roles"]["agent"]["lemma"] == "рыбак"
    assert need["candidates"] == []
    assert need["answer"]["evidence_status"] == "NEEDS_CLARIFICATION"
    assert need["answer"]["surface_answer"] == "Уточните, для какого действия это нужно."

    purpose = service.query(hive_id, "Чтобы ловить рыбу?")
    assert purpose["resolved_mode"] == "FOLLOW_UP"
    assert purpose["query_frame"]["query_type"] == "continuation_role_question"
    assert purpose["query_frame"]["requested_role"] == "instrument"
    assert purpose["query_frame"]["roles"]["agent"]["lemma"] == "рыбак"
    assert purpose["query_frame"]["roles"]["action"]["lemma"] == "ловить"
    assert purpose["query_frame"]["roles"]["object"]["lemma"] == "рыба"
    assert [candidate["lemma"] for candidate in purpose["candidates"]] == ["удочка"]

    answer = service.vibration_run(hive_id, 3)["answer"]
    assert answer["surface_answer"] == "Удочкой."
    assert answer["full_surface_answer"] == "Рыбак ловит рыбу удочкой."


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


def test_immediate_polar_answer_remains_in_dialogue_after_the_next_query():
    TrainingPipelineV2().train("Корова не ест рыбу. Кот ест рыбу.")
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]

    first = service.query(hive_id, "Ест ли корова рыбу?")
    second = service.query(hive_id, "Кто ест рыбу?")

    assert [message["role"] for message in first["messages"]] == ["user", "assistant"]
    assert first["messages"][-1]["text"] == "Нет. Корова не ест рыбу."
    assert [message["role"] for message in second["messages"]] == ["user", "assistant", "user"]


def test_query_frame_valency_constraints_and_multiple_answers_regression():
    TrainingPipelineV2().train(
        "Кот ест рыбу. Медведь ест рыбу. Пингвин ест рыбу. "
        "Выдра живёт у реки. Медведь живёт в лесу. "
        "Выдра питается рыбой. Цапля питается рыбой. "
        "Кот поймал карася. Карась это рыба. "
        "Кот это животное. Медведь это животное. "
        "Кот употребляет рыбу в пищу. Медведь употребляет рыбу в пищу. Робот употребляет рыбу в пищу."
    )
    service = V2HiveService()
    expectations = {
        "Кто ест рыбу?": "Кот, медведь и пингвин.",
        "Где живёт выдра?": "У реки.",
        "Кто живёт в лесу?": "Медведь.",
        "Кто питается рыбой?": "Выдра и цапля.",
        "Какую рыбу поймал кот?": "Карася.",
    }

    for query, expected in expectations.items():
        hive_id = service.create()["hive"]["id"]
        service.query(hive_id, query)
        assert service.vibration_run(hive_id, 3)["answer"]["surface_answer"] == expected

    typed_object = QuerySceneService().parse("Какую рыбу поймал кот?")["query_frame"]
    assert typed_object["missing_role"] == "object"
    assert typed_object["slot_constraints"]["object"]["is_a"]["lemma"] == "рыба"
    assert typed_object["answer_cardinality"] == "single"

    hive_id = service.create()["hive"]["id"]
    typed_agent = service.query(hive_id, "Какие животные употребляют рыбу в пищу?")
    assert typed_agent["query_frame"]["missing_role"] == "agent"
    assert typed_agent["query_frame"]["slot_constraints"]["agent"]["is_a"]["lemma"] == "животное"
    assert typed_agent["query_frame"]["answer_cardinality"] == "multiple"
    assert [candidate["lemma"] for candidate in typed_agent["candidates"]] == ["кот", "медведь"]
    answer = service.vibration_run(hive_id, 3)["answer"]
    assert answer["resolved_values"] == ["кот", "медведь"]
    assert answer["surface_answer"] == answer["full_surface_answer"] == "Кот и медведь."


def test_old_parser_scenes_are_reparsed_before_querying():
    TrainingPipelineV2().train("Рыбак ловит рыбу удочкой.")
    with V2Repository().transaction() as conn:
        scene_id = conn.execute("SELECT cloud_id FROM scenes").fetchone()[0]
        conn.execute("UPDATE scenes SET parser_version='ru-rule-v5' WHERE cloud_id=?", (scene_id,))

    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]
    result = service.query(hive_id, "Как рыбу ловит рыбак?")

    assert [candidate["lemma"] for candidate in result["candidates"]] == ["удочка"]
    with V2Repository().transaction() as conn:
        parser_version = conn.execute("SELECT parser_version FROM scenes WHERE cloud_id=?", (scene_id,)).fetchone()[0]
        role = conn.execute(
            """SELECT sc.grammatical_role FROM scene_components sc
               JOIN lexemes l ON l.cloud_id=sc.lexeme_cloud_id
               WHERE sc.scene_cloud_id=? AND l.lemma='удочка'""",
            (scene_id,),
        ).fetchone()[0]
    assert parser_version == TrainingPipelineV2.parser_version
    assert role == "instrument"


def test_query_activation_tiers_have_fixed_boundaries():
    scenes = [
        {"result_type": "FULL_HIT", "anchor_validation": {"status": "PASSED"}, "matched_roles": ["agent", "action"], "scores": {}},
        {"result_type": "ROLE_HIT", "anchor_validation": {"status": "PASSED"}, "matched_roles": ["agent", "action"], "scores": {}},
        {"result_type": "PARTIAL_HIT", "anchor_validation": {"status": "FAILED"}, "matched_roles": ["agent"], "scores": {}},
        {"result_type": "NO_HIT", "anchor_validation": {"status": "FAILED"}, "matched_roles": [], "scores": {}},
    ]

    QuerySceneService._assign_scene_activation(scenes)

    assert [(scene["physics"]["activation"], scene["physics"]["relevance_tier"]) for scene in scenes] == [
        (1.0, "DIRECT"), (.75, "RELATED"), (.30, "PARTIAL"), (.05, "BACKGROUND"),
    ]


def test_action_question_recovers_predicate_phrase_instead_of_searching_for_doing():
    service, hive_id, result = _ask(
        "Врач осматривает пациента.",
        "Что делает врач?",
    )

    frame = result["query_frame"]
    assert frame["query_type"] == "action_question"
    assert frame["requested_role"] == "action"
    assert frame["requested_slot"] == "predicate_phrase"
    assert frame["answer_slot_type"] == "predicate_phrase"
    assert frame["roles"]["action"]["status"] == "empty"
    assert frame["action_question"]["removed_predicate"]["lemma"] == "делать"
    assert result["memory_scenes"][0]["anchor_validation"]["answer_extraction"] == "predicate"
    assert result["memory_scenes"][0]["anchor_validation"]["placeholder_predicate_removed"] is True

    answer = service.vibration_run(hive_id, 3)["answer"]
    assert answer["surface_answer"] == "Осматривает пациента."
    assert answer["full_surface_answer"] == "Врач осматривает пациента."


def test_action_question_matches_a_named_entity_through_its_type():
    service, hive_id, result = _ask(
        "Дрон Сокол летит над полем.",
        "Что делает дрон?",
    )

    scene = result["memory_scenes"][0]
    assert scene["scores"]["agent_match"] == .88
    assert scene["role_match_details"]["agent"]["match_type"] in {"entity_type", "is_a"}
    assert scene["anchor_validation"]["status"] == "PASSED"
    assert result["query_frame"]["retrieval_stages"][0]["predicate_ignored"] == "делать"

    answer = service.vibration_run(hive_id, 3)["answer"]
    assert answer["surface_answer"] == "Летит над полем."
    assert answer["full_surface_answer"] == "Дрон Сокол летит над полем."


def test_action_question_keeps_doing_when_it_is_the_memory_predicate():
    service, hive_id, result = _ask(
        "Мастер делает деревянный стол.",
        "Что делает мастер?",
    )

    assert result["query_frame"]["action_question"]["removed_predicate"]["lemma"] == "делать"
    assert result["candidates"][0]["predicate_lemma"] == "делать"
    assert service.vibration_run(hive_id, 3)["answer"]["surface_answer"] == (
        "Делает деревянный стол."
    )


def test_action_question_recognizes_supported_question_forms_and_question_operator_override():
    service = QuerySceneService()
    for text in (
        "Что делал врач?",
        "Что будет делать врач?",
        "Что врач делает?",
        "Чем занимается врач?",
        "Что происходит с врачом?",
    ):
        frame = service.parse(text)["query_frame"]
        assert frame["query_type"] == "action_question"
        assert frame["tokens"][0]["semantic_part_of_speech"] == "QUESTION_OPERATOR"
        assert frame["question_operator"]["operator_type"] == "ACTION_QUERY"
