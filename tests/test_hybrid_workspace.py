from __future__ import annotations

from fastapi.testclient import TestClient

from server.server import create_app
from server.v2.graph_repository import GraphRepository
from server.v2.graph_service import GraphDialogueService, GraphTrainingService
from server.v2.hybrid import HybridDialoguePipeline


def _services() -> tuple[GraphTrainingService, GraphDialogueService]:
    repository = GraphRepository()
    return GraphTrainingService(repository), GraphDialogueService(repository)


def test_hybrid_direct_retrieval_uses_persisted_event_rows():
    training, dialogue = _services()
    training.train("Робот открыл шкаф ключом.", independent_key="hybrid-direct")
    training.train("Инженер отремонтировал панель отвёрткой.", independent_key="hybrid-distractor")

    result = dialogue.query(
        dialogue.create()["hive"]["id"],
        "Кто открыл шкаф ключом?",
    )

    assert result["hybrid"]["retrieval_hits"]
    assert {hit["payload"]["predicate_lemma"] for hit in result["hybrid"]["retrieval_hits"]} == {"открыть"}
    assert result["hybrid"]["answer"]["status"] == "STABLE"
    assert result["hybrid"]["answer_text"] == "Робот"
    assert result["hybrid"]["answer"]["provenance"]["evidence_ids"]


def test_hybrid_multi_gap_hypothesis_uses_a_joint_event():
    training, dialogue = _services()
    training.train("Робот открыл шкаф ключом.", independent_key="hybrid-multi")

    result = dialogue.query(
        dialogue.create()["hive"]["id"],
        "Кто чем открыл шкаф?",
    )

    hybrid = result["hybrid"]
    assert hybrid["answer"]["status"] == "STABLE"
    assert hybrid["answer_text"] == "Робот, ключом"
    hypothesis = hybrid["workspace"]["hypotheses"][0]
    assert len(hypothesis["fills"]) == 2
    assert len(hypothesis["supporting_events"]) == 1


def test_hybrid_continuation_excludes_the_previous_answer():
    training, dialogue = _services()
    training.train("Робот открыл шкаф ключом.", independent_key="hybrid-follow-up")
    hive_id = dialogue.create()["hive"]["id"]

    first = dialogue.query(hive_id, "Кто открыл шкаф ключом?")
    follow_up = dialogue.query(hive_id, "Ещё кто?")

    assert first["hybrid"]["answer"]["status"] == "STABLE"
    assert "робот" in follow_up["hybrid"]["query_frame"]["exclusions"]
    assert follow_up["hybrid"]["answer"]["status"] != "STABLE"
    assert all(
        candidate["surface"].casefold() != "робот"
        for candidate in follow_up["hybrid"]["workspace"]["candidates"]
        if candidate["status"] == "ACTIVE"
    )


def test_hybrid_rejects_unrelated_event_for_explicit_relation_gap():
    indexes = [{
        "element_id": "event_open",
        "element_type": "event",
        "source_id": "observation_open",
        "predicate_lemma": "открыть",
        "participants": [
            {"entity_id": "robot", "surface": "робот", "lemma": "робот", "features": {"case": "nomn"}},
            {"entity_id": "cabinet", "surface": "шкаф", "lemma": "шкаф", "features": {"case": "accs"}},
            {"entity_id": "key", "surface": "ключом", "lemma": "ключ", "features": {"case": "ablt"}},
        ],
        "provenance": [{"source_id": "observation_open", "source_type": "observation"}],
    }]

    result = HybridDialoguePipeline().run("Что было внутри шкафа?", indexes=indexes)

    assert result["answer"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert all(
        candidate["status"] != "ACTIVE"
        for candidate in result["workspace"]["candidates"]
    )


def test_hybrid_standalone_query_frame_uses_grammatical_gap_constraints():
    indexes = [{
        "element_id": "event_open",
        "element_type": "event",
        "source_id": "observation_open",
        "predicate_lemma": "открыть",
        "participants": [
            {"entity_id": "robot", "surface": "робот", "lemma": "робот", "features": {"case": "nomn"}},
            {"entity_id": "cabinet", "surface": "шкаф", "lemma": "шкаф", "features": {"case": "accs"}},
            {"entity_id": "key", "surface": "ключом", "lemma": "ключ", "features": {"case": "ablt"}},
        ],
        "provenance": [{"source_id": "observation_open", "source_type": "observation"}],
    }]

    result = HybridDialoguePipeline().run("Кто открыл шкаф ключом?", indexes=indexes)

    assert result["answer"]["status"] == "STABLE"
    assert result["answer_text"] == "робот"


def test_hybrid_primary_replaces_the_legacy_surface_answer():
    training, dialogue = _services()
    training.train("Робот открыл шкаф ключом.", independent_key="hybrid-primary")
    dialogue.workspace_mode = "hybrid_primary"

    result = dialogue.query(
        dialogue.create()["hive"]["id"],
        "Кто открыл шкаф ключом?",
    )

    assert result["answer"]["hybrid_primary"] is True
    assert result["answer"]["surface"] == "Робот."


def test_hive_query_http_contract_exposes_hybrid_workspace():
    training, _ = _services()
    training.train("Робот открыл шкаф ключом.", independent_key="hybrid-http")

    with TestClient(create_app()) as client:
        created = client.post("/api/v2/hives", json={"max_cells": 24, "conversation_id": "hybrid-http"})
        hive_id = created.json()["hive"]["id"]
        response = client.post(
            f"/api/v2/hives/{hive_id}/query",
            json={"text": "Кто открыл шкаф ключом?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["hybrid"]["answer"]["status"] == "STABLE"
    assert payload["hybrid"]["trace"]["stages"]
