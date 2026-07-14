import server.database as database
from fastapi.testclient import TestClient

from server.server import app
from server.v2.morphology import MorphologyService, compare_forms
from server.v2.training import TrainingPipelineV2


def test_compare_forms_and_unknown_plural_are_explainable():
    difference = compare_forms("мяч", "мячи", {"number": "plur"})
    assert difference.stable == "мяч"
    assert difference.added == "и"
    assert difference.operation == "PLURAL"

    TrainingPipelineV2().train("Мяч.")
    with database.get_connection() as conn:
        lexeme_id = conn.execute("SELECT cloud_id FROM lexemes WHERE lemma='мяч'").fetchone()[0]
        candidates = MorphologyService().resolve_word_form(conn, lexeme_id, {"number": "plur"})
    assert candidates[0].text == "мячи"
    assert candidates[0].temporary is True
    assert "PLURAL" in candidates[0].applied_patterns


def test_training_persists_morphology_space_and_patterns():
    TrainingPipelineV2().train("Рыба. Рыбу. Мяч. Мячи.")
    with database.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM spaces WHERE space_type='morphology_space'").fetchone()[0] >= 2
        assert conn.execute("SELECT COUNT(*) FROM word_form_features").fetchone()[0] >= 4
        assert conn.execute("SELECT COUNT(*) FROM morph_pattern_data").fetchone()[0] >= 1
        assert conn.execute("SELECT COUNT(*) FROM cloud_compositions WHERE relation_type='known_word_form'").fetchone()[0] >= 4


def test_scene_cell_expansion_and_surface_validation_via_api():
    with TestClient(app) as client:
        assert client.post("/api/v2/training/learn", json={"text": "Мяч. Мячи."}).status_code == 200
        hive_id = client.post("/api/v2/hives", json={}).json()["hive"]["id"]
        query = client.post(f"/api/v2/hives/{hive_id}/query", json={"text": "Мяч"}).json()

        expanded = client.post(
            f"/api/v2/hives/{hive_id}/cells/{query['cells'][0]['id']}/expand",
            json={"target_level": "word_form", "reason": "test", "max_candidates": 5},
        )
        assert expanded.status_code == 200
        candidates = expanded.json()["candidates"]
        assert [item["candidate_text"] for item in candidates] == ["мяч"]
        assert candidates[0]["character_sequence"] == ["м", "я", "ч"]

        generated = client.post(f"/api/v2/hives/{hive_id}/generate", json={"sentence_plan": {
            "slots": [{"role": "subject", "lexeme": "мяч", "requested_features": {"number": "plur"}}]
        }})
        assert generated.status_code == 200
        assert generated.json()["selected_surface"] == "Мячи."
        assert client.post(f"/api/v2/hives/{hive_id}/validate-surface", json={"surface": "Мячи."}).json()["valid"]
        assert not client.post(f"/api/v2/hives/{hive_id}/validate-surface", json={"surface": "Мячы."}).json()["valid"]
