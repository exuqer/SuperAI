from __future__ import annotations

import math
from pathlib import Path

import pytest

import server.database as database
from server.v2.acceleration import AccelerationRuntime, ProjectionIndex, RouteGraphIndex
from server.v2.graph_repository import GraphRepository
from server.v2.graph_service import GraphDialogueService, GraphTrainingService
from server.v2.universe import SparseResidualDiscoverer


def test_numpy_projection_index_is_exact_stable_and_ignores_invalid_vectors():
    runtime = AccelerationRuntime("python")
    index = ProjectionIndex(runtime, ("left", "right")).rebuild([
        ("z", {"left": 1.0, "right": 0.0}),
        ("a", {"left": 1.0, "right": 0.0}),
        ("empty", {"left": 0.0, "right": 0.0}),
        ("nan", {"left": math.nan, "right": 0.0}),
        ("text", {"left": "not-a-number", "right": 0.0}),
        ("wrong-size", [1.0]),
    ])

    assert index.backend == "numpy"
    assert [item.id for item in index.search({"left": 1.0}, 10)] == ["a", "z"]
    assert index.search({"left": 0.0, "right": 0.0}) == []
    assert index.skipped == 4


def test_python_route_fallback_handles_cycles_disconnections_and_parallel_paths():
    index = RouteGraphIndex(AccelerationRuntime("python")).rebuild([
        {"id": "ab", "source_id": "a", "target_id": "b", "weight": 1.0},
        {"id": "bc", "source_id": "b", "target_id": "c", "weight": 1.0},
        {"id": "ca", "source_id": "c", "target_id": "a", "weight": 1.0},
        {"id": "ad", "source_id": "a", "target_id": "d", "weight": 0.25},
        {"id": "dc", "source_id": "d", "target_id": "c", "weight": 0.25},
    ])

    assert index.expand(["a"], budget=3) == ["a", "b", "d"]
    assert index.search("a", "c", budget=10).nodes == ("a", "b", "c")
    assert index.search("a", "missing", budget=10) is None


def test_route_expansion_never_exceeds_the_budget_even_with_many_seeds():
    index = RouteGraphIndex(AccelerationRuntime("python")).rebuild([])

    assert index.expand(["c", "b", "a"], budget=2) == ["a", "b"]
    assert index.expand(["a"], budget=0) == []


def test_backend_build_errors_fallback_only_in_auto_mode():
    class BrokenFaiss:
        class IndexFlatIP:
            def __init__(self, dimensions):
                raise RuntimeError(f"cannot build {dimensions}")

    auto = AccelerationRuntime("python")
    auto.mode = "auto"
    auto._modules["faiss"] = BrokenFaiss
    index = ProjectionIndex(auto, ("x",)).rebuild([("item", [1.0])])
    assert index.backend == "numpy"
    assert "faiss: cannot build 1" in auto.fallback_reasons

    native = AccelerationRuntime("python")
    native.mode = "native"
    native._modules["faiss"] = BrokenFaiss
    with pytest.raises(RuntimeError, match="faiss acceleration backend failed"):
        ProjectionIndex(native, ("x",)).rebuild([("item", [1.0])])


def test_sparse_matrix_preserves_scalar_semantic_values():
    runtime = AccelerationRuntime("auto")
    if not runtime.use("scipy"):
        pytest.skip("SciPy acceleration extra is not installed")
    samples = [
        {"context_vector_json": '{"context:left:one":1.0,"position:sentence:0":0.25}'},
        {"context_vector_json": '{"context:left:one":0.5,"context:right:two":1.0}'},
    ]
    matrix, features, _ = SparseResidualDiscoverer.build_sparse_matrix(samples, runtime)
    values = matrix.toarray()
    assert values.nbytes <= 64 * 1024 * 1024
    assert "position:sentence:0" not in features
    assert values[0, features.index("context:left:one")] == pytest.approx(1.0, abs=1e-6)
    assert values[1, features.index("context:right:two")] == pytest.approx(1.0, abs=1e-6)


def test_sklearn_shadow_candidates_are_deterministic_for_the_fixed_seed():
    samples = [
        {
            "context_vector_json": (
                '{"context:left:a":1.0,"context:right:x":0.25}'
            ),
        },
        {
            "context_vector_json": (
                '{"context:left:b":1.0,"context:right:x":0.5}'
            ),
        },
        {
            "context_vector_json": (
                '{"context:left:a":0.5,"context:right:y":1.0}'
            ),
        },
        {
            "context_vector_json": (
                '{"context:left:b":0.25,"context:right:y":1.0}'
            ),
        },
    ]
    first_runtime = AccelerationRuntime("auto")
    if not first_runtime.use("scipy") or not first_runtime.use("sklearn"):
        pytest.skip("SciPy/scikit-learn acceleration extras are not installed")
    second_runtime = AccelerationRuntime("auto")

    first = SparseResidualDiscoverer.shadow_candidates(samples, first_runtime)
    second = SparseResidualDiscoverer.shadow_candidates(samples, second_runtime)

    assert first
    assert first == second


def test_reset_revision_cannot_reuse_a_stale_process_cache_key():
    repository = GraphRepository()
    with repository.transaction() as conn:
        repository.bump_revisions(conn)
    before = repository.graph_meta()
    repository.reset()
    after = repository.graph_meta()
    assert int(after["projection_revision"]) == int(before["projection_revision"]) + 1
    assert int(after["transition_revision"]) == int(before["transition_revision"]) + 1


def test_train_and_retraction_each_invalidate_projection_and_route_revisions():
    repository = GraphRepository()
    training = GraphTrainingService(repository)
    baseline = repository.graph_meta()
    trained = training.train("Механик поднял ключ.", independent_key="accel-revision")
    after_train = repository.graph_meta()
    training.retract(trained["source_id"], "test")
    after_retraction = repository.graph_meta()
    for name in ("projection_revision", "transition_revision"):
        assert int(after_train[name]) == int(baseline[name]) + 1
        assert int(after_retraction[name]) == int(after_train[name]) + 1


def _parity_scenario(mode: str, path: Path):
    database.DB_PATH = path
    database.init_db()
    runtime = AccelerationRuntime(mode)
    repository = GraphRepository()
    training = GraphTrainingService(repository, runtime=runtime)
    dialogue = GraphDialogueService(repository, runtime=runtime)
    training.train("Механик дал роботу болт.", independent_key="accel-parity")
    hive_id = dialogue.create(conversation_id="accel-parity")["hive"]["id"]
    results = []
    for question in ("Что механик дал роботу?", "Кому?", "Кто?"):
        response = dialogue.query(hive_id, question)
        assert response["trace"]["sql_ms"] > 0
        assert response["trace"]["serialization_ms"] > 0
        assert response["trace"]["numerical_ms"] > 0
        assert response["trace"]["sqlite_execute_count"] > 0
        results.append({
            "status": response["answer"]["status"],
            "surface": response["answer"]["surface"],
            "event_ids": [
                item["event_id"]
                for item in response["trace"]["selected_bindings"]
            ],
            "resolved_nodes": [
                item["resolved_node_id"]
                for item in response["trace"]["selected_bindings"]
            ],
            "validation": response["answer"]["validation"],
        })
    return results


def test_auto_and_python_end_to_end_results_are_equivalent(tmp_path):
    original_path = database.DB_PATH
    try:
        reference = _parity_scenario("python", tmp_path / "python.sqlite")
        accelerated = _parity_scenario("auto", tmp_path / "auto.sqlite")
    finally:
        database.DB_PATH = original_path

    assert accelerated == reference
