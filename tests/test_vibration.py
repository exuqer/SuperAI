from server.v2.hive import V2HiveService
from server.v2.repository import V2Repository
from server.v2.training import TrainingPipelineV2


def setup_hive():
    TrainingPipelineV2().train("Кот ест рыбу. Рыбак ловит рыбу. Мяч катится.")
    service = V2HiveService()
    hive = service.create(24, "conversation-test")
    service.query(hive["hive"]["id"], "Кот ест рыбу")
    return service, hive["hive"]["id"]


def test_zero_steps_creates_only_initial_snapshot():
    service, hive_id = setup_hive()
    result = service.reason(hive_id, "Кот ест рыбу", {"reasoning_steps": 0, "random_seed": 7})
    assert result["run"]["completed_steps"] == 0
    assert len(result["snapshots"]) == 1
    assert result["snapshots"][0]["phase"] == "INITIAL"


def test_reasoning_is_sequential_and_local():
    service, hive_id = setup_hive()
    repository = V2Repository()
    with repository.transaction() as conn:
        masses_before = {row["id"]: row["mass"] for row in conn.execute("SELECT id, mass FROM clouds")}
        global_before = {row["id"]: (row["x"], row["y"]) for row in conn.execute("SELECT p.id, p.x, p.y FROM cloud_placements p JOIN spaces s ON s.id=p.space_id WHERE s.space_type='global_field'")}
    result = service.reason(hive_id, config={"reasoning_steps": 2, "random_seed": 11})
    assert result["run"]["completed_steps"] == 2
    after = [item for item in result["snapshots"] if item["phase"] == "AFTER_SETTLE"]
    assert after[0]["step"] == 1
    assert after[1]["step"] == 2
    with repository.transaction() as conn:
        assert masses_before == {row["id"]: row["mass"] for row in conn.execute("SELECT id, mass FROM clouds")}
        global_after = {row["id"]: (row["x"], row["y"]) for row in conn.execute("SELECT p.id, p.x, p.y FROM cloud_placements p JOIN spaces s ON s.id=p.space_id WHERE s.space_type='global_field'")}
        assert global_before == global_after
        nodes = conn.execute("SELECT * FROM hive_node_states WHERE hive_id=?", (hive_id,)).fetchall()
        assert nodes
        assert all(0 <= row["local_activation"] <= 1 and 0 <= row["retention"] <= 1 for row in nodes)
        for cell in conn.execute("SELECT id, dominant_cloud_id FROM hive_cells WHERE hive_id=?", (hive_id,)):
            component_clouds = {row["cloud_id"] for row in conn.execute("SELECT cloud_id FROM hive_cell_components WHERE cell_id=?", (cell["id"],))}
            assert cell["dominant_cloud_id"] not in component_clouds


def test_export_current_initial_and_trace():
    service, hive_id = setup_hive()
    result = service.reason(hive_id, config={"reasoning_steps": 1, "random_seed": 13})
    run_id = result["run"]["id"]
    assert service.export(hive_id, "current")["schema_version"] == 2
    assert service.export(hive_id, "initial", run_id)["step"] == 0
    assert len(service.export(hive_id, "trace", run_id)["snapshots"]) == 3
    diff = service.diff(run_id, 0, 1)
    assert diff["from_step"] == 0 and diff["to_step"] == 1
    assert service.runs(hive_id)[0]["id"] == run_id
    assert len(service.snapshots(hive_id, run_id)) == 3
    assert service.restore(hive_id, run_id, 0)["restored"] is True
