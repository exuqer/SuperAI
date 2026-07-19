from __future__ import annotations

import math

import pytest

from server.v2.acceleration import AccelerationRuntime, ProjectionIndex, RouteGraphIndex
from server.v2.graph_repository import GraphRepository
from server.v2.graph_service import GraphTrainingService
from server.v2.universe import SparseResidualDiscoverer


def test_numpy_projection_index_is_exact_stable_and_ignores_invalid_vectors():
    runtime = AccelerationRuntime("python")
    index = ProjectionIndex(runtime, ("left", "right")).rebuild([
        ("z", {"left": 1.0, "right": 0.0}),
        ("a", {"left": 1.0, "right": 0.0}),
        ("empty", {"left": 0.0, "right": 0.0}),
        ("nan", {"left": math.nan, "right": 0.0}),
    ])

    assert index.backend == "numpy"
    assert [item.id for item in index.search({"left": 1.0}, 10)] == ["a", "z"]
    assert index.search({"left": 0.0, "right": 0.0}) == []
    assert index.skipped == 2


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
