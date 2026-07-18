from __future__ import annotations

from fastapi.testclient import TestClient

from server.v2.graph_repository import GraphRepository
from server.v2.graph_service import GraphTrainingService
from server.v2.universe import UniverseService


def test_confirmed_training_creates_entities_occurrences_and_dynamic_fields():
    repository = GraphRepository()
    training = GraphTrainingService(repository)
    for index, text in enumerate((
        "Яблоко растет в саду.",
        "Помидор растет в саду.",
        "Яблоко созревает в саду.",
        "Помидор созревает в саду.",
    )):
        result = training.train(text, independent_key=f"universe-{index}")
        assert result["universe_update"]["ingested"] is True

    universes = UniverseService(repository)
    listing = universes.list_universes()["universes"]
    words = next(item for item in listing if item["id"] == "words")
    assert words["entity_count"] >= 2
    assert words["occurrence_count"] >= 4

    space = universes.base_space("words")
    apple = next(item for item in space["entities"] if item["label"].casefold() == "яблоко")
    profile = universes.profile(apple["id"])
    assert profile["entity"]["id"] == apple["id"]
    assert len(profile["occurrence_distribution"]) == 2
    assert universes.dimensions("words")["dimensions"]


def test_words_are_lexemes_while_forms_and_usages_preserve_inflection():
    repository = GraphRepository()
    training = GraphTrainingService(repository)
    examples = (
        "Яблоко лежит на столе.",
        "Стол стоит здесь.",
        "Нет яблока и стола.",
        "Яблоки лежат на столе.",
        "Яблоко лежало на столе.",
        "Мяч упал.",
        "Яблоко упало.",
        "Листья упали.",
        "Красный помидор лежит здесь.",
        "Красное яблоко лежит здесь.",
        "Красные яблоки лежат здесь.",
        "Мальчик разрезал яблоко.",
        "Девочка разрезала помидор.",
        "Девочка нарезала помидор.",
    )
    for index, text in enumerate(examples):
        training.train(text, independent_key=f"forms-{index}")

    with repository.transaction() as conn:
        lexemes = {
            str(row["canonical_lemma"]): str(row["lexeme_entity_id"])
            for row in conn.execute(
                "SELECT canonical_lemma,lexeme_entity_id FROM lexemes"
            ).fetchall()
        }
        forms_by_lemma = {
            lemma: {
                str(row["normalized_surface"])
                for row in conn.execute(
                    """SELECT normalized_surface FROM word_forms
                       WHERE lexeme_entity_id=?""",
                    (lexeme_id,),
                ).fetchall()
            }
            for lemma, lexeme_id in lexemes.items()
        }

    assert {"яблоко", "яблока", "яблоки"} <= forms_by_lemma["яблоко"]
    assert {"стол", "стола", "столе"} <= forms_by_lemma["стол"]
    assert {"лежит", "лежат", "лежало"} <= forms_by_lemma["лежать"]
    assert {"упал", "упало", "упали"} <= forms_by_lemma["упасть"]
    assert {"красный", "красное", "красные"} <= forms_by_lemma["красный"]
    assert {"разрезал", "разрезала"} <= forms_by_lemma["разрезать"]
    assert "нарезала" in forms_by_lemma["нарезать"]
    assert lexemes["разрезать"] != lexemes["нарезать"]

    universes = UniverseService(repository)
    words = universes.base_space("words")["entities"]
    assert [item["label"] for item in words].count("яблоко") == 1
    apple = next(item for item in words if item["label"] == "яблоко")
    profile = universes.profile(apple["id"])
    assert profile["canonical_lemma"] == "яблоко"
    assert {item["surface"] for item in profile["word_forms"]} >= {
        "яблоко", "яблока", "яблоки",
    }
    assert profile["usage_count"] >= 4
    assert any(item["id"] == "word_forms" for item in universes.list_universes()["universes"])


def test_universe_http_contract_is_independent_from_graph_roles():
    from server.server import create_app

    with TestClient(create_app()) as client:
        assert client.post(
            "/api/v2/training/learn",
            json={"text": "Борис настроил датчик.", "independent_key": "universe-api"},
        ).status_code == 200
        listing = client.get("/api/universes")
        assert listing.status_code == 200
        assert {item["id"] for item in listing.json()["universes"]} >= {
            "symbols", "morphemes", "words", "usages", "clauses", "events", "scenes",
        }
        space = client.get("/api/universes/words/base-space")
        assert space.status_code == 200
        assert space.json()["entities"]
        dimensions = client.get("/api/universes/words/dimensions").json()["dimensions"]
        assert dimensions
        detail = client.get(f"/api/dimensions/{dimensions[0]['id']}")
        assert detail.status_code == 200
        assert detail.json()["representation_type"] in {
            "axis", "subspace", "cloud", "multi_core", "manifold", "field",
        }
