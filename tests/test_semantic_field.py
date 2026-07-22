from __future__ import annotations

from server.v2.graph_service import GraphDialogueService, GraphTrainingService
from server.v2.semantic_field import SemanticFieldService


def test_confirmed_events_create_revisioned_cloud_projections() -> None:
    training = GraphTrainingService()
    first = training.train("Кошка ест рыбу.", independent_key="field:cat")
    second = training.train("Тигр ест мясо.", independent_key="field:tiger")

    assert first["field_update"]["status"] == "APPLIED"
    assert second["field_update"]["field_revision"] > first["field_update"]["field_revision"]
    snapshot = SemanticFieldService(training.repository).snapshot()
    assert snapshot["field_revision"] == second["field_update"]["field_revision"]
    assert all(len(cloud["center"]) == 3 for cloud in snapshot["clouds"])
    with training.repository.transaction() as conn:
        assert conn.execute(
            """SELECT COUNT(*) FROM graph_events e
               JOIN knowledge_sources s ON s.id=e.source_id
               WHERE s.raw_text LIKE '%тигр ест рыбу%'"""
        ).fetchone()[0] == 0


def test_factual_answer_requires_evidential_graph_support() -> None:
    training = GraphTrainingService()
    training.train("Кот находится возле миски.", independent_key="field:evidence")
    dialogue = GraphDialogueService(training.repository)
    hive_id = dialogue.create(conversation_id="field-evidence")["hive"]["id"]

    result = dialogue.query(hive_id, "Где находится кот?")

    assert result["hybrid"]["query_field_projection"]["anchor_clouds"]
    assert result["hybrid"]["answer_structure"]["epistemic_mode"] == "OBSERVED"
    assert result["hybrid"]["answer_structure"]["graph_support"] > 0


def test_contextual_cloud_projection_is_separate_from_cloud_identity() -> None:
    training = GraphTrainingService()
    training.train("Ключ открывает замок.", independent_key="field:key-lock")
    training.train("Ключ решает задачу.", independent_key="field:key-problem")

    with training.repository.transaction() as conn:
        row = conn.execute(
            """SELECT COUNT(*) FROM contextual_cloud_projections cp
               JOIN semantic_clouds c ON c.id=cp.cloud_id
               JOIN graph_entities e ON e.id=c.concept_id
               WHERE e.canonical_lemma='ключ'"""
        ).fetchone()
    assert row[0] >= 2
