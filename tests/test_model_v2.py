from server.v2.hive import V2HiveService
from server.v2.repository import V2Repository
from server.v2.training import TrainingPipelineV2
from server.v2.validation import ModelInvariantValidator


def test_v2_word_structure_and_scene_placements():
    pipeline = TrainingPipelineV2()
    pipeline.train("Кот ест рыбу. Рыбу ловит рыбак.")
    repository = V2Repository()
    with repository.transaction() as conn:
        word = conn.execute("SELECT id FROM v2_clouds WHERE cloud_type = 'word_form' AND canonical_name = 'рыбу'").fetchone()
        assert word
        components = repository.components(conn, word["id"])
        assert [component["component_index"] for component in components] == [0, 1, 2, 3]
        assert "".join(conn.execute("SELECT canonical_name FROM v2_clouds WHERE id = ?", (component["child_cloud_id"],)).fetchone()[0] for component in components) == "рыбу"
        placements = conn.execute("""SELECT sc.placement_id, s.scene_space_id FROM v2_scene_components sc
            JOIN v2_scenes s ON s.cloud_id = sc.scene_cloud_id WHERE sc.word_form_cloud_id = ?""", (word["id"],)).fetchall()
        assert len(placements) == 2
        assert placements[0]["placement_id"] != placements[1]["placement_id"]
        assert placements[0]["scene_space_id"] != placements[1]["scene_space_id"]
    assert ModelInvariantValidator().validate()["valid"]


def test_v2_retraining_is_idempotent_and_hive_is_local():
    pipeline = TrainingPipelineV2()
    pipeline.train("Кот ест рыбу.")
    pipeline.train("Кот ест рыбу.")
    repository = V2Repository()
    with repository.transaction() as conn:
        scene = conn.execute("SELECT * FROM v2_scenes").fetchone()
        assert scene["observation_count"] == 2
        assert conn.execute("SELECT COUNT(*) FROM v2_scene_components").fetchone()[0] == 3
    hive = V2HiveService().forage("кот ест")
    assert hive["cells"]
    for cell in hive["cells"]:
        assert abs(sum(component["composition_share"] for component in cell["components"]) - 1.0) < 0.001
    assert ModelInvariantValidator().validate()["valid"]
