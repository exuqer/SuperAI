"""Regression coverage for context inheritance and learned concept fogs."""

from server.v2.hive import V2HiveService
from server.v2.query_scene import QuerySceneService
from server.v2.repository import V2Repository, decode, encode
from server.v2.semantic_fog import SemanticFogService
from server.v2.training import TrainingPipelineV2


def test_continuation_uses_archived_session_and_accumulates_exclusions():
    TrainingPipelineV2().train("Лисичка ест ягоду. Лисичка ест грушу.")
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]

    service.query(hive_id, "Лисичка ест ягоду.")
    repository = V2Repository()
    with repository.transaction() as conn:
        row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
        metadata = decode(row["metadata_json"], {})
        working = metadata["query_working_memory"]
        archived = working.pop("query_session")
        archived["status"] = "ARCHIVED"
        working.setdefault("query_sessions", []).append(archived)
        metadata["query_working_memory"] = working
        conn.execute("UPDATE hives SET metadata_json=? WHERE id=?", (encode(metadata), hive_id))
    follow = service.query(hive_id, "А ещё что?")
    assert follow["query_frame"]["query_type"] == "continuation_role_question" or follow["query_frame"]["ellipsis_follow_up"]
    assert follow["query_frame"]["continuation_of"]
    assert set(follow["query_frame"]["inherited_roles"]) >= {"agent", "action"}
    assert [item["lemma"] for item in follow["candidates"]] == ["груша"]

    service.vibration_run(hive_id, 3)
    repeated = service.query(hive_id, "А ещё что?")
    excluded = repeated["query_frame"]["excluded_roles"]["object"]
    assert {item["lemma"] for item in excluded} >= {"ягода", "груша"}
    assert not repeated["candidates"]
    assert any(item["status"] == "ARCHIVED" for item in repeated["query_sessions"])


def test_continuation_without_session_never_broadens_search():
    TrainingPipelineV2().train("Кошечка ест рыбу.")
    service = V2HiveService()
    result = service.query(service.create()["hive"]["id"], "А ещё что?")
    assert result["query_frame"]["context_resolution"]["status"] == "UNRESOLVED_CONTEXT"
    assert result["candidates"] == []
    assert result["retrieval_scope"]["imported"] == 0


def test_definition_evidence_materializes_idempotent_local_concept_fog():
    pipeline = TrainingPipelineV2()
    pipeline.train("Кошка это кошечка. Кошка это животное. Собака это животное.")
    repository = V2Repository()
    with repository.transaction() as conn:
        global_before = [tuple(row) for row in conn.execute(
            "SELECT cloud_id,x,y FROM cloud_placements WHERE space_id=(SELECT id FROM spaces WHERE space_type='global_field') ORDER BY cloud_id"
        )]
        cat = conn.execute("SELECT cloud_id FROM lexemes WHERE lemma='кошка'").fetchone()[0]
        kitten = conn.execute("SELECT cloud_id FROM lexemes WHERE lemma='кошечка'").fetchone()[0]
        detail = QuerySceneService(repository)._semantic_membership_detail(conn, {"lexeme_cloud_id": cat}, {"lexeme_cloud_id": kitten})
        fog_count = conn.execute("SELECT COUNT(*) FROM concept_fog_registry").fetchone()[0]
        assert detail["score"] == .85
        assert detail["role_match_score"] == .85
        assert detail["concept_space_ids"]
        assert conn.execute("SELECT COUNT(*) FROM semantic_evidence WHERE evidence_type='definition'").fetchone()[0] == 3
        evidence = conn.execute("SELECT evidence_weight, independence, evidence_key FROM semantic_evidence WHERE evidence_type='definition' LIMIT 1").fetchone()
        assert evidence["evidence_weight"] == .90
        assert evidence["independence"] == 1
        assert evidence["evidence_key"]
        assert conn.execute("SELECT COUNT(*) FROM semantic_backfill_state").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM concept_candidate_registry WHERE is_search_eligible=0").fetchone()[0] >= 2

    pipeline.train("")
    with repository.transaction() as conn:
        global_after = [tuple(row) for row in conn.execute(
            "SELECT cloud_id,x,y FROM cloud_placements WHERE space_id=(SELECT id FROM spaces WHERE space_type='global_field') ORDER BY cloud_id"
        )]
        assert global_after == global_before
        assert conn.execute("SELECT COUNT(*) FROM concept_fog_registry").fetchone()[0] == fog_count


def test_verbal_definition_connects_query_predicate_to_memory_predicate():
    pipeline = TrainingPipelineV2()
    trained = pipeline.train("Кот ест рыбу. Кушать — это есть.")
    repository = V2Repository()
    definition_scene_id = trained["scenes"][1]["scene_cloud_id"]

    with repository.transaction() as conn:
        roles = [
            tuple(row)
            for row in conn.execute(
                """SELECT l.lemma, sc.grammatical_role
                FROM scene_components sc
                JOIN lexemes l ON l.cloud_id=sc.lexeme_cloud_id
                WHERE sc.scene_cloud_id=? AND sc.grammatical_role <> 'service'
                ORDER BY sc.token_index""",
                (definition_scene_id,),
            )
        ]
        evidence = conn.execute(
            """SELECT evidence_type FROM semantic_evidence
            WHERE source_scene_cloud_id=?""",
            (definition_scene_id,),
        ).fetchone()

    assert roles == [("кушать", "subject"), ("есть", "definition")]
    assert evidence["evidence_type"] == "definition"

    with repository.transaction() as conn:
        conn.execute(
            """UPDATE scene_components SET grammatical_role='predicate'
            WHERE scene_cloud_id=? AND grammatical_role <> 'service'""",
            (definition_scene_id,),
        )
        conn.execute(
            "DELETE FROM semantic_evidence WHERE source_scene_cloud_id=?",
            (definition_scene_id,),
        )
        conn.execute(
            """UPDATE semantic_backfill_state SET semantic_extractor_version=4
            WHERE source_scene_cloud_id=?""",
            (definition_scene_id,),
        )
        SemanticFogService(repository)._evidence_for_scene(conn, definition_scene_id)
        legacy_evidence = conn.execute(
            """SELECT evidence_type FROM semantic_evidence
            WHERE source_scene_cloud_id=?""",
            (definition_scene_id,),
        ).fetchone()

    assert legacy_evidence["evidence_type"] == "definition"

    service = V2HiveService(repository)
    result = service.query(service.create()["hive"]["id"], "А кто кушает рыбу?")
    assert result["candidates"][0]["lemma"] == "кот"
    action_match = next(
        scene["role_match_details"]["action"]
        for scene in result["memory_scenes"]
        if scene["source_text"].startswith("Кот ест рыбу")
    )
    assert action_match["match_type"] == "stable_concept"
    assert action_match["role_match_score"] == .85

    polar = service.query(service.create()["hive"]["id"], "Кот кушает рыбу?")
    assert polar["answer"]["status"] == "RESOLVED"
    assert polar["answer"]["resolved_value"] is True
    assert polar["answer"]["evidence_status"] == "SUPPORTED"


def test_distributional_evidence_compares_equal_roles_not_cooccurring_words():
    TrainingPipelineV2().train(
        "Кот ест рыбу. Кошечка ест рыбу. "
        "Кот любит молоко. Кошечка любит молоко."
    )
    repository = V2Repository()
    with repository.transaction() as conn:
        lexemes = {
            row["lemma"]: int(row["cloud_id"])
            for row in conn.execute("SELECT cloud_id,lemma FROM lexemes")
        }
        service = QuerySceneService(repository)
        related = service._semantic_membership_detail(
            conn,
            {"lexeme_cloud_id": lexemes["кот"]},
            {"lexeme_cloud_id": lexemes["кошечка"]},
        )
        cooccurring = service._semantic_membership_detail(
            conn,
            {"lexeme_cloud_id": lexemes["кот"]},
            {"lexeme_cloud_id": lexemes["рыба"]},
        )
        assert related["role_match_score"] == .65
        assert related["match_type"] == "related_concept"
        assert len(related["supporting_scenes"]) == 2
        assert cooccurring["role_match_score"] == 0


def test_shared_category_keeps_definition_direction_when_category_existed_first():
    TrainingPipelineV2().train(
        "Животное. Кошка это животное. Собака это животное."
    )
    repository = V2Repository()
    with repository.transaction() as conn:
        lexemes = {
            row["lemma"]: int(row["cloud_id"])
            for row in conn.execute("SELECT cloud_id,lemma FROM lexemes")
        }
        detail = QuerySceneService(repository)._semantic_membership_detail(
            conn,
            {"lexeme_cloud_id": lexemes["кошка"]},
            {"lexeme_cloud_id": lexemes["собака"]},
        )
        category_weight = conn.execute(
            "SELECT evidence_weight FROM semantic_evidence WHERE evidence_type='shared_category' LIMIT 1"
        ).fetchone()[0]
    assert detail["role_match_score"] == .45
    assert detail["match_type"] == "shared_category"
    assert len(detail["supporting_scenes"]) == 2
    assert category_weight == .70


def test_learned_continuation_keeps_rejected_broad_scenes_visible():
    TrainingPipelineV2().train(
        "Кот это кошечка. Кот ест рыбу. Кошечка ест птицу. "
        "Рыбу продают на рынке."
    )
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]

    first = service.query(hive_id, "А что ест кошечка?")
    assert [item["lemma"] for item in first["candidates"]][:2] == ["птица", "рыба"]
    assert service.vibration_run(hive_id, 3)["answer"]["resolved_value"] == "птицу"

    follow = service.query(hive_id, "А ещё что?")
    assert [item["lemma"] for item in follow["candidates"]] == ["рыба"]
    assert all("рынке" not in item["source_text"] for item in follow["memory_scenes"])
    assert follow["query_frame"]["retrieval_metrics"]["indexed"] is True
    assert (
        follow["query_frame"]["retrieval_metrics"]["scenes_considered"]
        < follow["query_frame"]["retrieval_metrics"]["scenes_total"]
    )
    stages = [item["stage"] for item in follow["reasoning_trace"]["stages"]]
    assert stages[1:6] == [
        "QUERY_FRAME", "CONTEXT_INHERITANCE", "QUERY_SCENE_COMPLETION",
        "MEMORY_SCENE_SEARCH", "CANDIDATE_RANKING",
    ]


def test_explicit_except_query_is_self_contained_and_excludes_requested_value():
    TrainingPipelineV2().train("Кот ест рыбу. Кот ест мясо.")
    service = V2HiveService()
    result = service.query(service.create()["hive"]["id"], "Что кроме рыбы ест кот?")

    assert result["query_frame"]["query_type"] == "continuation_role_question"
    assert result["query_frame"]["context_resolution"]["status"] == "RESOLVED"
    assert result["query_frame"]["continuation_of"] is None
    assert result["query_frame"]["excluded_roles"]["object"][0]["lemma"] == "рыба"
    assert [item["lemma"] for item in result["candidates"]] == ["мясо"]


def test_inflected_other_marker_inherits_previous_statement_exclusion():
    TrainingPipelineV2().train("Кот ест рыбу. Кот ест мясо.")
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]
    service.query(hive_id, "Кот ест рыбу.")

    result = service.query(hive_id, "Что другое ест кот?")

    assert "другое" in result["query_frame"]["continuation_markers"]
    assert result["query_frame"]["continuation_of"]
    assert result["query_frame"]["excluded_roles"]["object"][0]["lemma"] == "рыба"
    assert [item["lemma"] for item in result["candidates"]] == ["мясо"]


def test_model_clear_removes_semantic_evidence_and_fogs_from_stats():
    TrainingPipelineV2().train("Кошка это кошечка.")
    repository = V2Repository()
    repository.clear()
    with repository.transaction() as conn:
        stats = repository.stats(conn)
    assert stats["semantic_evidence_total"] == 0
    assert stats["concept_fogs_total"] == 0
    assert stats["concept_candidates_total"] == 0
    assert stats["semantic_backfill_scenes_total"] == 0
