from server.v2.graph_service import GraphTrainingService
from server.v2.semantic_field import SemanticFieldService


def test_retract_restores_active_field_geometry() -> None:
    training = GraphTrainingService()
    first = training.train("Кошка ест рыбу.", independent_key="a")
    before = SemanticFieldService(training.repository).snapshot()
    second = training.train("Тигр ест мясо.", independent_key="b")
    training.retract(second["source_id"])
    after = SemanticFieldService(training.repository).snapshot()
    assert [(item["concept_id"], item["center"]) for item in before["clouds"]] == [
        (item["concept_id"], item["center"]) for item in after["clouds"]
    ]
    assert SemanticFieldService(training.repository).snapshot(field_revision=first["field_update"]["field_revision"])["clouds"] == before["clouds"]


def test_field_hits_are_spatial_support_not_graph_evidence() -> None:
    training = GraphTrainingService()
    training.train("Кошка ест рыбу.", independent_key="a")
    field = SemanticFieldService(training.repository)
    projection = field.project_query(
        type("Frame", (), {"known_elements": ({"lemma": "кошка"},), "gaps": (), "negations": (), "exclusions": (), "temporal_scope": None, "context_inheritance": {}})()
    )
    hits = field.neighbourhood(projection)
    assert all(hit.origin == "FIELD" for hit in hits)
    assert all(hit.payload.get("spatial_support") for hit in hits)
