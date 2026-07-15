from fastapi.testclient import TestClient

from server.server import app
from server.v2.physics import PlacementPhysicsV2
from server.v2.repository import V2Repository
from server.v2.schema import SCHEMA_VERSION
from server.v2.training import TrainingPipelineV2
from server.v2.validation import ModelInvariantValidator


def test_word_structures_are_unique_and_scene_occurrences_are_placements():
    pipeline = TrainingPipelineV2()
    pipeline.train("Кот ест рыбу. Рыбак ловит рыбу. Кот ест мясо.")
    repository = V2Repository()
    with repository.transaction() as conn:
        expectations = {"кот": 3, "ест": 3, "рыбу": 4}
        for word, length in expectations.items():
            cloud = conn.execute(
                "SELECT id FROM clouds WHERE cloud_type = 'word_form' AND canonical_name = ?", (word,)
            ).fetchone()
            assert cloud
            assert conn.execute(
                "SELECT COUNT(*) FROM structural_components WHERE parent_cloud_id = ?", (cloud["id"],)
            ).fetchone()[0] == length
            assert conn.execute(
                "SELECT COUNT(*) FROM spaces WHERE space_type = 'word_structure_space' AND owner_cloud_id = ?",
                (cloud["id"],),
            ).fetchone()[0] == 1
        fish = conn.execute(
            "SELECT id FROM clouds WHERE cloud_type = 'word_form' AND canonical_name = 'рыбу'"
        ).fetchone()
        placements = conn.execute(
            """SELECT sc.placement_id, p.space_id FROM scene_components sc
            JOIN cloud_placements p ON p.id = sc.placement_id WHERE sc.word_form_cloud_id = ?""",
            (fish["id"],),
        ).fetchall()
        assert len(placements) == 2
        assert len({row["placement_id"] for row in placements}) == 2
        assert len({row["space_id"] for row in placements}) == 2
    assert ModelInvariantValidator().validate()["valid"]


def test_retraining_strengthens_without_growing_structure_or_scene():
    pipeline = TrainingPipelineV2()
    first = pipeline.train("Кот ест рыбу.")
    repository = V2Repository()
    with repository.transaction() as conn:
        before = repository.stats(conn)
        fish_cloud_id = conn.execute(
            "SELECT id FROM clouds WHERE cloud_type='lexeme' AND canonical_name='рыба'"
        ).fetchone()[0]
        first_mass = conn.execute("SELECT mass FROM clouds WHERE id=?", (fish_cloud_id,)).fetchone()[0]
    second = pipeline.train("Кот ест рыбу.")
    with repository.transaction() as conn:
        after = repository.stats(conn)
        scene = conn.execute("SELECT * FROM scenes").fetchone()
        assert scene["observation_count"] == 2
        second_mass = conn.execute("SELECT mass FROM clouds WHERE id=?", (fish_cloud_id,)).fetchone()[0]
    pipeline.train("Кот ест рыбу.")
    with repository.transaction() as conn:
        third_mass = conn.execute("SELECT mass FROM clouds WHERE id=?", (fish_cloud_id,)).fetchone()[0]
    for key in (
        "clouds_total", "spaces_total", "placements_total", "scene_components_total",
        "structural_components_total",
    ):
        assert after[key] == before[key]
    assert first["created_clouds"]
    assert second["created_clouds"] == []
    assert second["strengthened_clouds"]
    assert second["reused_scenes"]
    assert 0 < third_mass - second_mass < second_mass - first_mass


def test_scene_dto_is_normalized_and_structure_is_lazy():
    pipeline = TrainingPipelineV2()
    result = pipeline.train("Кот ест рыбу.")
    scene_id = result["scenes"][0]["scene_cloud_id"]
    with TestClient(app) as client:
        scene = client.get(f"/api/v2/scenes/{scene_id}").json()["scene"]
        assert "components" in scene
        assert "words" not in scene
        assert "word_forms" not in scene
        assert all("cloud_id" in component and "placement_id" in component for component in scene["components"])
        assert all("characters" not in component for component in scene["components"])
        word_id = scene["components"][0]["cloud_id"]
        structure = client.get(f"/api/v2/clouds/{word_id}/structure").json()
        assert len(structure["components"]) == 3
        assert [item["component_index"] for item in structure["components"]] == [0, 1, 2]


def test_roles_and_physics_are_local_to_scene_space():
    result = TrainingPipelineV2().train("Кот ест рыбу. Рыбак ловит рыбу.")
    repository = V2Repository()
    with repository.transaction() as conn:
        first_scene = result["scenes"][0]["scene_cloud_id"]
        second_scene = result["scenes"][1]["scene_cloud_id"]
        first_space = conn.execute("SELECT scene_space_id FROM scenes WHERE cloud_id = ?", (first_scene,)).fetchone()[0]
        second_space = conn.execute("SELECT scene_space_id FROM scenes WHERE cloud_id = ?", (second_scene,)).fetchone()[0]
        roles = [row[0] for row in conn.execute(
            "SELECT grammatical_role FROM scene_components WHERE scene_cloud_id = ? ORDER BY token_index", (first_scene,)
        )]
        assert roles == ["subject", "predicate", "object"]
        second_before = [tuple(row) for row in conn.execute(
            "SELECT id, x, y FROM cloud_placements WHERE space_id = ? ORDER BY id", (second_space,)
        )]
    PlacementPhysicsV2(first_space).tick()
    with repository.transaction() as conn:
        second_after = [tuple(row) for row in conn.execute(
            "SELECT id, x, y FROM cloud_placements WHERE space_id = ? ORDER BY id", (second_space,)
        )]
    assert second_after == second_before


def test_stats_are_typed_and_add_up():
    TrainingPipelineV2().train("Кот ест рыбу.")
    repository = V2Repository()
    with repository.transaction() as conn:
        stats = repository.stats(conn)
    assert stats["clouds_total"] == sum(stats["clouds_by_type"].values())
    assert stats["concepts_total"] == stats["clouds_by_type"].get("concept", 0)
    assert stats["scene_components_total"] == 3


def test_trained_model_snapshot_contains_canonical_data_without_hive_runtime():
    TrainingPipelineV2().train("Кот ест рыбу.")
    with TestClient(app) as client:
        snapshot = client.get("/api/v2/model")
    assert snapshot.status_code == 200
    data = snapshot.json()
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["stats"]["clouds_total"] == len(data["model"]["clouds"])
    assert data["model"]["scene_components"]
    assert "metadata" in data["model"]["clouds"][0]
    assert "metadata_json" not in data["model"]["clouds"][0]
    assert "hives" not in data["model"]
    assert data["model"]["word_form_features"]
    assert "semantic_evidence" in data["model"]
    assert "concept_fog_registry" in data["model"]
    assert "concept_candidate_registry" in data["model"]
    assert "semantic_backfill_state" in data["model"]


def test_definition_sentence_roles():
    result = TrainingPipelineV2().train("Кружка — это стакан с ручкой.")
    repository = V2Repository()
    with repository.transaction() as conn:
        roles = [tuple(row) for row in conn.execute(
            """SELECT c.canonical_name, sc.grammatical_role FROM scene_components sc
            JOIN clouds c ON c.id = sc.word_form_cloud_id
            WHERE sc.scene_cloud_id = ? ORDER BY sc.token_index""",
            (result["scenes"][0]["scene_cloud_id"],),
        )]
    assert roles == [
        ("кружка", "subject"),
        ("это", "service"),
        ("стакан", "definition"),
        ("с", "preposition"),
        ("ручкой", "complement"),
    ]


def test_concept_cloud_exposes_its_owned_fog_space():
    TrainingPipelineV2().train("Кошка это кошечка.")
    repository = V2Repository()
    with repository.transaction() as conn:
        fog = conn.execute(
            "SELECT concept_cloud_id,concept_space_id FROM concept_fog_registry"
        ).fetchone()
        global_placement = conn.execute(
            """SELECT 1 FROM cloud_placements p JOIN spaces s ON s.id=p.space_id
            WHERE p.cloud_id=? AND s.space_type='global_field'""",
            (fog["concept_cloud_id"],),
        ).fetchone()
    with TestClient(app) as client:
        payload = client.get(f"/api/v2/clouds/{fog['concept_cloud_id']}").json()
    assert global_placement
    assert payload["owned_spaces"][0]["id"] == fog["concept_space_id"]
    assert payload["owned_spaces"][0]["space_type"] == "concept_space"
