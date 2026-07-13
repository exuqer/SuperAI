"""Tests for the relation-free concept field."""

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import database
from server.database import get_concepts, init_db
from server.physics import ConceptState, PhysicsConfig, concept_radius, run_physics_step, run_simulation
from server.tokenizer import normalize_text, split_sentences, tokenize
from server.training import TrainingManager
from server import training as training_module
from server.server import app


@pytest.fixture
def isolated_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "state.sqlite")
    monkeypatch.setattr(training_module, "_training_manager", None)
    init_db()
    return tmp_path / "state.sqlite"


def test_tokenize_russian_and_latin():
    tokens = tokenize("Привет, как дела? Hello world!")
    assert tokens == ["привет", "как", "дела", "hello", "world"]


def test_tokenize_ignores_punctuation():
    tokens = tokenize("Привет! Как дела? Всё отлично.")
    assert all(token.isalpha() for token in tokens)


def test_split_sentences():
    sentences = split_sentences("Привет. Как дела? Всё отлично!")
    assert len(sentences) == 3
    assert "Привет" in sentences[0]
    assert "Как дела" in sentences[1]
    assert "Всё отлично" in sentences[2]


def test_normalize_text():
    assert normalize_text("  Привет   мир  ") == "Привет мир"
    assert normalize_text("\n\r\tПривет\nмир\t") == "Привет мир"
    assert normalize_text("") == ""


def test_concept_state_and_radius():
    concept = ConceptState(id=1, token="тест", position=[100, 100], mass=1.0)
    assert concept.distance_to(ConceptState(id=2, token="два", position=[200, 200])) == pytest.approx(141.42, abs=0.1)
    assert concept.radius == pytest.approx(concept_radius(1.0))


def test_global_field_moves_objects_without_context_pairs():
    config = PhysicsConfig(steps=1)
    concepts = [
        ConceptState(id=1, token="один", position=[300, 500], mass=1.0),
        ConceptState(id=2, token="два", position=[500, 500], mass=1.0),
    ]
    before = [concept.position[:] for concept in concepts]
    run_physics_step(concepts, [], config)
    assert concepts[0].position[0] > before[0][0]
    assert concepts[1].position[0] < before[1][0]


def test_training_creates_unique_concepts_and_persists_only_field(isolated_database):
    manager = TrainingManager(PhysicsConfig(steps=4))
    result = manager.learn("Кот кот ест рыбу")
    assert result["success"] is True
    assert {concept["token"] for concept in result["concepts"]} == {"кот", "ест", "рыбу"}
    assert all(concept["mass"] == pytest.approx(1.0) for concept in result["concepts"])
    assert all(len(concept["position"]) == 2 for concept in result["concepts"])
    assert all(concept["radius"] <= 250 for concept in result["concepts"])

    tables = {
        row[0]
        for row in sqlite3.connect(isolated_database).execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert tables == {"concepts"}


def test_existing_mass_increments_once_per_request(isolated_database):
    manager = TrainingManager(PhysicsConfig(steps=2))
    manager.learn("кот ест рыбу")
    result = manager.learn("кот кот ест")
    masses = {concept["token"]: concept["mass"] for concept in result["concepts"]}
    assert masses["кот"] == pytest.approx(1.1)
    assert masses["ест"] == pytest.approx(1.1)
    assert masses["рыбу"] == pytest.approx(1.0)


def test_word_order_changes_trajectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first_db = tmp_path / "first.sqlite"
    monkeypatch.setattr(database, "DB_PATH", first_db)
    init_db()
    first = TrainingManager(PhysicsConfig(steps=5)).learn("кот ест рыбу")
    first_positions = {concept["token"]: concept["position"] for concept in first["concepts"]}

    second_db = tmp_path / "second.sqlite"
    monkeypatch.setattr(database, "DB_PATH", second_db)
    init_db()
    second = TrainingManager(PhysicsConfig(steps=5)).learn("рыбу ест кот")
    second_positions = {concept["token"]: concept["position"] for concept in second["concepts"]}

    assert first_positions != second_positions


def test_simulation_is_finite_and_bounded():
    config = PhysicsConfig(steps=10)
    concepts = [
        ConceptState(id=1, token="а", position=[0, 0], mass=1),
        ConceptState(id=2, token="б", position=[1600, 1000], mass=100000),
    ]
    run_simulation(concepts, [["а", "б"]], config)
    for concept in concepts:
        assert all(abs(value) < 1e9 for value in concept.position + concept.velocity)
        assert config.boundary_margin <= concept.position[0] <= config.width - config.boundary_margin
        assert config.boundary_margin <= concept.position[1] <= config.height - config.boundary_margin


def test_reset_removes_all_concepts(isolated_database):
    manager = TrainingManager(PhysicsConfig(steps=2))
    manager.learn("кот ест рыбу")
    result = manager.reset_space()
    assert result["concepts"] == []
    assert get_concepts() == []


def test_api_has_no_relation_fields(isolated_database):
    with TestClient(app) as client:
        response = client.post("/api/v1/training/learn", json={"text": "кот ест рыбу"})
    assert response.status_code == 200
    payload = response.json()
    forbidden = {"connections", "edges", "neighbors", "phrases", "frequency", "gravity", "halo"}
    assert not forbidden.intersection(payload)
    assert not forbidden.intersection(payload["concepts"][0])
    assert set(payload["stats"]) == {"concepts", "total_mass", "tokens"}
