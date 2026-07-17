from server.v2.concept_relations import ConceptRelationTrainer
from server.v2.hive import V2HiveService
from server.v2.repository import V2Repository
from server.v2.taxonomy_resolver import TaxonomyResolver
from server.v2.training import TrainingPipelineV2


def test_classification_definition_materializes_direct_is_a_with_deduplicated_evidence():
    TrainingPipelineV2().train("Кот — животное. Кот — животное.")
    with V2Repository().transaction() as conn:
        relation = conn.execute(
            """SELECT * FROM concept_relations cr JOIN lexemes s ON s.cloud_id=cr.subject_lexeme_cloud_id
               JOIN lexemes o ON o.cloud_id=cr.object_lexeme_cloud_id
               WHERE cr.relation_type='IS_A' AND s.lemma='кот' AND o.lemma='животное'"""
        ).fetchone()
        assert relation is not None
        assert int(relation["evidence_count"]) == 1
        resolution = TaxonomyResolver(conn).resolve_is_a(
            int(relation["subject_lexeme_cloud_id"]), int(relation["object_lexeme_cloud_id"]), 1
        )
    assert resolution["passed"] is True
    assert resolution["depth"] == 1
    assert resolution["evidence_scene_ids"]


def test_candidate_taxonomy_is_checked_after_semantic_extraction_and_rejections_are_traced():
    TrainingPipelineV2().train(
        "Кот ест рыбу. Медведь ест рыбу. Пингвин ест рыбу. Выдра питается рыбой. Цапля питается рыбой. "
        "Корова не ест рыбу. Корова ест траву. Робот употребляет рыбу в пищу. "
        "Кот это животное. Медведь это животное. Пингвин это животное. Выдра это животное. Цапля это животное. "
        "Есть и питаться — близкие действия, связанные с употреблением пищи."
    )
    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]
    result = service.query(hive_id, "Какие животные употребляют рыбу в пищу?")
    assert [candidate["lemma"] for candidate in result["accepted_candidates"]] == ["кот", "медведь", "пингвин", "выдра", "цапля"]
    rejected = {candidate["lemma"]: candidate["rejection_reason"] for candidate in result["rejected_candidates"]}
    assert rejected == {"корова": "POLARITY_MISMATCH", "робот": "TAXONOMY_RELATION_NOT_FOUND"}
    assert all(candidate["fact_evidence"] and candidate["constraint_evidence"] for candidate in result["accepted_candidates"])
    assert service.vibration_run(hive_id, 3)["answer"]["surface_answer"] == "Кот, медведь, пингвин, выдра и цапля."


def test_taxonomy_resolver_stops_on_cycles():
    TrainingPipelineV2().train("Кот это животное. Животное это кот.")
    with V2Repository().transaction() as conn:
        ConceptRelationTrainer().rebuild(conn)
        rows = conn.execute("SELECT cloud_id, lemma FROM lexemes WHERE lemma IN ('кот', 'животное')").fetchall()
        ids = {row["lemma"]: int(row["cloud_id"]) for row in rows}
        result = TaxonomyResolver(conn).resolve_is_a(ids["кот"], ids["животное"], 3)
    assert result["passed"] is True
    assert result["depth"] == 1
