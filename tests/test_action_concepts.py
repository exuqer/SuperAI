from pathlib import Path

from server.v2.hive import V2HiveService
from server.v2.repository import V2Repository
from server.v2.training import TrainingPipelineV2


def test_clean_model_has_no_automatic_action_concepts_or_projections():
    with V2Repository().transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM action_concepts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM concept_relations").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM scene_concept_projections").fetchone()[0] == 0


def test_explicit_action_definition_enables_conceptual_holdout():
    repository = V2Repository()
    TrainingPipelineV2(repository).train(
        "Мастер ремонтирует модуль. Ремонтировать — это чинить."
    )
    service = V2HiveService(repository)
    hive_id = service.create()["hive"]["id"]
    result = service.query(hive_id, "Кто чинит модуль?")

    assert result["query_frame"]["conceptual_query_frame"]["action_concept_id"]
    assert [candidate["lemma"] for candidate in result["candidates"]] == ["мастер"]
    assert service.vibration_run(hive_id, 3)["answer"]["surface_answer"] == "Мастер."

    with repository.transaction() as conn:
        assert conn.execute(
            """SELECT COUNT(*) FROM concept_evidence
               WHERE evidence_type='EXPLICIT_DEFINITION'"""
        ).fetchone()[0] == 1
        assert conn.execute(
            """SELECT COUNT(*) FROM scene_concept_projections scp
               JOIN events e ON e.source_scene_id=scp.scene_id
               WHERE e.predicate_lemma='ремонтировать'"""
        ).fetchone()[0] >= 1


def test_universal_core_contains_no_removed_domain_branches():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (Path(__file__).parents[1] / "server" / "v2").glob("*.py")
    )
    for prohibited in (
        "consume_as_food",
        "agent_or_object",
        "OBJECT_PREDICATES",
        "LOCATION_PREDICATES",
    ):
        assert prohibited not in source
