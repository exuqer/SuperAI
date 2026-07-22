from __future__ import annotations

import sqlite3

import server.database as database

from server.v2.graph_repository import GraphRepository, utcnow
from server.v2.testing_reset import ResetMode, ResetScope, TestingResetService
from server.v2.universe import UniverseService


def _seed_state(repository: GraphRepository) -> None:
    UniverseService(repository)
    now = utcnow()
    with repository.transaction() as conn:
        conn.execute(
            """INSERT INTO knowledge_sources
               (id,raw_text,normalized_text,content_hash,source_type,status,
                confidence,independent_key,metadata_json,created_at,updated_at)
               VALUES('source-reset','Робот поднял ключ.','робот поднял ключ.',
                      'hash-reset','training','CONFIRMED',1,'reset-test','{}',?,?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO hives
               (id,conversation_id,max_cells,active_query_graph_id,state_json,created_at,updated_at)
               VALUES('hive-reset','conversation-reset',24,NULL,'{}',?,?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO universe_entities
               (id,universe_id,observable_key,display_value,prototype_vector_json,
                base_position_json,mass,gravity,stability,frequency,dispersion,
                created_at,updated_at)
               VALUES('universe-entity-reset','words','ru:ключ:','ключ','{}','[]',
                      1,0,0.5,1,0,?,?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO semantic_clouds
               (id,cloud_type,concept_id,mass,density,halo,stability,permeability,
                bootstrap_center_json,learned_center_json,active_dimensions_json,
                position_status,provenance_json,created_at,updated_at)
               VALUES('semantic-cloud-reset','concept','entity-reset',1,0.5,0.2,0.5,0.5,
                      '[0,0,0]','[0,0,0]','[]','LEARNED','[]',?,?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO field_revisions
               (revision,based_on_event_revision,status,metrics_json,created_at,applied_at)
               VALUES(1,0,'APPLIED','{}',?,?)""",
            (now, now),
        )


def test_full_test_state_clear_data_is_idempotent() -> None:
    repository = GraphRepository()
    _seed_state(repository)
    service = TestingResetService(repository)

    first = service.reset(
        ResetScope.FULL_TEST_STATE,
        ResetMode.CLEAR_DATA,
        requested_by="pytest",
    )
    second = service.reset(
        ResetScope.FULL_TEST_STATE,
        ResetMode.CLEAR_DATA,
        requested_by="pytest",
    )

    assert first["reset"] is True
    assert second["reset"] is True
    assert all(value == 0 for value in first["after"].values())
    assert all(value == 0 for value in second["after"].values())
    assert first["invariants"]["universe_registry_present"] is True
    assert second["field_revision"] == 0
    assert first["database_generation_id"] != second["database_generation_id"]


def test_derived_reset_preserves_evidence_and_requires_explicit_rebuild() -> None:
    repository = GraphRepository()
    _seed_state(repository)
    report = TestingResetService(repository).reset(
        ResetScope.DERIVED_SEMANTIC_SPACE,
        ResetMode.CLEAR_DATA,
        requested_by="pytest",
    )

    assert report["reset"] is True
    assert report["rebuild_available"] is True
    assert report["after"]["knowledge_sources"] == 1
    assert report["after"]["semantic_clouds"] == 0
    assert report["after"]["universe_entities"] == 0
    assert report["field_revision"] == 0
    with repository.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM universes").fetchone()[0] > 0


def test_fresh_schema_removes_database_state_and_allows_reuse() -> None:
    repository = GraphRepository()
    _seed_state(repository)
    service = TestingResetService(repository)

    report = service.reset(
        ResetScope.FULL_TEST_STATE,
        ResetMode.FRESH_SCHEMA,
        requested_by="pytest",
    )

    assert report["reset"] is True
    assert report["mode"] == "FRESH_SCHEMA"
    assert all(value == 0 for value in report["after"].values())
    assert GraphRepository().graph_meta()["database_generation_id"] == report["database_generation_id"]


def test_fresh_schema_reset_does_not_delete_an_open_database_file() -> None:
    repository = GraphRepository()
    _seed_state(repository)
    external_handle = sqlite3.connect(database.get_db_path())
    try:
        report = TestingResetService(repository).reset(
            ResetScope.FULL_TEST_STATE,
            ResetMode.FRESH_SCHEMA,
            requested_by="pytest",
        )
    finally:
        external_handle.close()

    assert report["reset"] is True
