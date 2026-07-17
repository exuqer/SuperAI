from server.v2.repository import V2Repository
from server.v2.training import TrainingPipelineV2
from server.v2.hive import V2HiveService


def test_consume_as_food_holdout_uses_scene_concept_projection():
    TrainingPipelineV2().train(
        "Кот ест рыбу. Медведь ест рыбу. Пингвин ест рыбу. "
        "Выдра питается рыбой. Цапля питается рыбой. "
        "Корова не ест рыбу. Робот употребляет рыбу в пищу. "
        "Кот это животное. Медведь это животное. Пингвин это животное. "
        "Выдра это животное. Цапля это животное. "
        "Есть и питаться - близкие действия, связанные с употреблением пищи."
    )
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]

    result = service.query(hive_id, "Какие животные употребляют рыбу в пищу?")

    frame = result["query_frame"]
    assert frame["question_word"] == "какие"
    assert frame["answer_cardinality"] == "multiple"
    assert frame["conceptual_query_frame"]["action_concept"] == "consume_as_food"
    assert frame["conceptual_query_frame"]["missing_role"] == "consumer"
    assert [candidate["lemma"] for candidate in result["candidates"]] == [
        "кот", "медведь", "пингвин", "выдра", "цапля",
    ]
    robot = next(scene for scene in result["memory_scenes"] if scene["roles"].get("agent", {}).get("lemma") == "робот")
    cow = next(scene for scene in result["memory_scenes"] if scene["roles"].get("agent", {}).get("lemma") == "корова")
    assert robot["decision_reason"] == "anchor_validation_failed"
    assert "slot.is_a" in robot["anchor_validation"]["failed_constraints"]
    assert cow["result_type"] == "CONFLICT_HIT"

    answer = service.vibration_run(hive_id, 3)["answer"]
    assert answer["surface_answer"] == "Кот, медведь, пингвин, выдра и цапля."

    with V2Repository().transaction() as conn:
        projection_count = conn.execute(
            "SELECT COUNT(*) FROM scene_concept_projections WHERE action_concept_id='action-concept-consume-as-food'"
        ).fetchone()[0]
    assert projection_count >= 6
