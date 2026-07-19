from __future__ import annotations

import re
from pathlib import Path

from server.core.settings import settings
from server.v2.experiment import CompactExperiment, ExperimentConfig
from server.v2.graph_repository import GraphRepository, decode
from server.v2.graph_service import GraphDialogueService, GraphTrainingService


def services():
    repository = GraphRepository()
    return (
        repository,
        GraphTrainingService(repository),
        GraphDialogueService(repository),
    )


def test_schema_contains_iteration_evidence_and_experiment_tables():
    repository, _, _ = services()
    required = {
        "binding_configurations",
        "swarm_runs",
        "bee_missions",
        "bee_steps",
        "nectar_packets",
        "candidate_event_observations",
        "dimension_history",
        "dimension_lineage",
        "dimension_evaluations",
        "shadow_retrieval_runs",
        "experiment_runs",
        "experiment_metrics",
    }
    with repository.transaction() as conn:
        tables = {
            str(row["name"])
            for row in conn.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='table'"""
            ).fetchall()
        }
        dimension_columns = {
            str(row["name"])
            for row in conn.execute(
                "PRAGMA table_info(latent_dimensions)"
            ).fetchall()
        }
    assert required <= tables
    assert {
        "canonical_dimension_id",
        "revision",
        "stability_lower_bound",
        "holdout_retrieval_gain",
        "shadow_retrieval_gain",
        "validated_answer_contribution_count",
    } <= dimension_columns


def test_multi_gap_contract_is_canonical_and_learns_one_configuration():
    repository, training, dialogue = services()
    training.train(
        "Механик дал роботу болт.",
        independent_key="plan:multi-gap",
    )
    result = dialogue.query(
        dialogue.create()["hive"]["id"],
        "Кто, кому и что дал?",
    )

    assert len(result["query_graph"]["question_operators"]) == 3
    gaps = result["query_graph"]["event_pattern"]["target_gaps"]
    assert len(gaps) == 3
    assert len({gap["coordination_group_id"] for gap in gaps}) == 1
    assert all(gap["required"] for gap in gaps)
    assert len(result["selected_bindings"]) == 3
    assert result["binding_configuration"]["status"] == "SELECTED"

    with repository.transaction() as conn:
        episode = conn.execute(
            """SELECT selected_bindings_json,binding_configuration_id
               FROM training_episodes"""
        ).fetchone()
        configuration = conn.execute(
            """SELECT status,validation_json FROM binding_configurations"""
        ).fetchone()
    assert len(decode(episode["selected_bindings_json"], [])) == 3
    assert episode["binding_configuration_id"]
    assert configuration["status"] == "SELECTED"
    assert decode(configuration["validation_json"], {})["valid"] is True


def test_incomplete_multi_gap_configuration_stays_unresolved():
    _, training, dialogue = services()
    training.train(
        "Механик передал болт.",
        independent_key="plan:incomplete",
    )
    result = dialogue.query(
        dialogue.create()["hive"]["id"],
        "Кто, кому и что передал?",
    )

    assert result["selected_bindings"] == []
    assert result["binding_configuration"] is None
    assert result["answer"]["status"] == "UNRESOLVED"
    assert (
        result["answer"]["validation"]["reason"]
        == "INCOMPLETE_BINDING_CONFIGURATION"
    )


def test_underconstrained_events_are_ambiguous_not_randomly_selected():
    _, training, dialogue = services()
    training.train(
        "Механик дал роботу болт.",
        independent_key="plan:ambiguity:first",
    )
    training.train(
        "Инженер дал машине кабель.",
        independent_key="plan:ambiguity:second",
    )

    result = dialogue.query(
        dialogue.create()["hive"]["id"],
        "Кто дал?",
    )

    assert result["answer"]["status"] == "AMBIGUOUS_BINDING"
    assert result["selected_bindings"] == []
    assert (
        result["answer"]["validation"]["reason"]
        == "MULTIPLE_UNCONSTRAINED_EVENTS"
    )


def test_complete_new_topic_does_not_inherit_an_event_anchor():
    _, training, dialogue = services()
    training.train(
        "Механик дал роботу болт.",
        independent_key="plan:topic:first",
    )
    training.train(
        "Собака подняла палку.",
        independent_key="plan:topic:second",
    )
    hive_id = dialogue.create()["hive"]["id"]
    dialogue.query(hive_id, "Что механик дал роботу?")
    second = dialogue.query(hive_id, "Кто поднял палку?")

    assert second["answer"]["surface"] == "Собака."
    assert second["query_graph"]["continuation_of"] is None
    assert second["query_graph"]["trace"]["continuation_mode"] == "NONE"
    assert second["query_graph"]["trace"]["event_anchor_id"] is None


def test_gap_swarm_uses_all_bee_types_and_obeys_configured_budget():
    repository, training, dialogue = services()
    training.train(
        "Борис настроил датчик.",
        independent_key="plan:swarm",
    )
    with repository.transaction() as conn:
        dimension = conn.execute(
            """SELECT id FROM latent_dimensions
               WHERE universe_id='words'
               ORDER BY evidence_count DESC,id LIMIT 1"""
        ).fetchone()
        conn.execute(
            """UPDATE latent_dimensions
               SET status='active',holdout_retrieval_gain=.1,
                   stability=.9,stability_lower_bound=.7
               WHERE id=?""",
            (dimension["id"],),
        )

    result = dialogue.query(
        dialogue.create()["hive"]["id"],
        "Кто настроил датчик?",
    )
    swarm = result["trace"]["swarm"]
    run = swarm["gap_swarms"][0]

    assert swarm["retrieval_mode"] == "SWARM_DIMENSIONAL"
    assert {mission["bee_type"] for mission in run["missions"]} == {
        "Scout", "Worker", "Assembly", "Observer",
    }
    assert run["bee_count"] <= settings.swarm_max_bees
    assert run["round_count"] <= settings.swarm_max_rounds
    assert run["packet_count"] <= settings.swarm_max_nectar_packets
    assert result["answer"]["surface"] == "Борис."
    with repository.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM bee_steps"
        ).fetchone()[0] > 0
        utility = conn.execute(
            """SELECT validated_answer_contribution_count
               FROM latent_dimensions WHERE id=?""",
            (dimension["id"],),
        ).fetchone()[0]
    assert utility > 0


def test_control_features_cannot_create_a_latent_dimension_alone():
    repository, training, _ = services()
    for index, text in enumerate((
        "Робот поднял ключ.",
        "Ключ поднял робот.",
        "Роботом был поднят ключ.",
        "Механик поднял болт.",
    )):
        training.train(text, independent_key=f"plan:control:{index}")
    with repository.transaction() as conn:
        bases = [
            decode(row["basis_json"], {})
            for row in conn.execute(
                "SELECT basis_json FROM latent_dimensions"
            ).fetchall()
        ]
    assert bases
    assert all(
        basis.get("feature_family") == "semantic_structural"
        for basis in bases
    )
    assert all(
        not str(basis.get("residual_feature") or "").startswith(
            ("morph:", "position:", "shape:")
        )
        for basis in bases
    )


def test_deprecated_singular_binding_is_http_only():
    root = Path(__file__).parents[1]
    singular = re.compile(r"\bselected_binding\b|selected_binding_id")
    offenders = []
    for path in (
        root / "server" / "v2"
    ).rglob("*.py"):
        if singular.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_compact_dataset_and_configuration_are_reproducible():
    assert CompactExperiment.validate_dataset() == {
        "train": 48,
        "holdout": 16,
        "continual": 16,
        "blind": 8,
    }
    first = ExperimentConfig(random_seed=41)
    second = ExperimentConfig(random_seed=41)
    different = ExperimentConfig(random_seed=42)
    assert first.hash() == second.hash()
    assert first.hash() != different.hash()
