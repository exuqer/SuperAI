from __future__ import annotations

from server.v2.graph_repository import GraphRepository
from server.v2.testing_reset import ResetMode, ResetScope, TestingResetService
from server.v2.universe import UniverseService


def test_empty_derived_reset_keeps_canonical_universe_registry() -> None:
    repository = GraphRepository()
    UniverseService(repository)
    report = TestingResetService(repository).reset(
        ResetScope.DERIVED_SEMANTIC_SPACE,
        ResetMode.CLEAR_DATA,
        requested_by="pytest",
    )
    assert report["invariants"]["field_empty"] is True
    assert report["invariants"]["universes_empty"] is True
    assert report["invariants"]["universe_registry_present"] is True
