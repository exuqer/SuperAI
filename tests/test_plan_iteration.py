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
        "query_operator_profiles",
        "query_operator_occurrences",
        "query_operator_experiences",
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


def test_query_operator_schema_extension_preserves_a_v28_database():
    repository, _, _ = services()
    with repository.transaction() as conn:
        for table in (
            "query_operator_experiences",
            "query_operator_occurrences",
            "query_operator_profiles",
        ):
            conn.execute(f"DROP TABLE {table}")
        conn.execute(
            """INSERT INTO hives
               (id,conversation_id,max_cells,active_query_graph_id,state_json,
                created_at,updated_at)
               VALUES('existing-hive','existing-conversation',24,NULL,'{}',
                      '2026-01-01','2026-01-01')"""
        )

    upgraded = GraphRepository()
    with upgraded.transaction() as conn:
        tables = {
            str(row["name"])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        preserved = conn.execute(
            "SELECT id FROM hives WHERE id='existing-hive'"
        ).fetchone()
    assert preserved is not None
    assert {
        "query_operator_profiles",
        "query_operator_occurrences",
        "query_operator_experiences",
    } <= tables


def test_future_graph_schema_is_not_silently_downgraded_as_compatible():
    repository, _, _ = services()
    with repository.transaction() as conn:
        conn.execute("CREATE TABLE future_only_marker(id TEXT PRIMARY KEY)")
        conn.execute(
            "UPDATE graph_meta SET value='34' WHERE key='schema_version'"
        )

    recreated = GraphRepository()
    with recreated.transaction() as conn:
        marker = conn.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type='table' AND name='future_only_marker'"""
        ).fetchone()
        schema_version = conn.execute(
            "SELECT value FROM graph_meta WHERE key='schema_version'"
        ).fetchone()[0]
    assert marker is None
    assert str(schema_version) == "33"


def test_query_operator_profiles_are_shadow_learned_from_validated_bindings():
    repository, training, dialogue = services()
    training.train(
        "Механик дал роботу болт.",
        independent_key="query-operator-shadow",
    )

    first = dialogue.query(
        dialogue.create()["hive"]["id"],
        "Что механик дал роботу?",
    )
    assert first["answer"]["surface"] == "болт."
    first_gap = first["query_graph"]["event_pattern"]["target_gaps"][0]
    assert first_gap["evidence"]["learned_gap_profile"]["profile_status"] == "UNSEEN"

    with repository.transaction() as conn:
        tables = {
            str(row["name"])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        profile = conn.execute(
            """SELECT compatible_slots_json,validated_count,status
               FROM query_operator_profiles WHERE profile_key='surface:что'"""
        ).fetchone()
        occurrences = conn.execute(
            """SELECT status FROM query_operator_occurrences
               WHERE operator_normalized='что'"""
        ).fetchall()
        experiences = conn.execute(
            """SELECT outcome,validated FROM query_operator_experiences
               WHERE outcome='VALIDATED_BINDING'"""
        ).fetchall()
    assert {
        "query_operator_profiles",
        "query_operator_occurrences",
        "query_operator_experiences",
    } <= tables
    assert profile["status"] == "SHADOW"
    assert int(profile["validated_count"]) == 1
    assert decode(profile["compatible_slots_json"], {})
    assert [row["status"] for row in occurrences] == ["VALIDATED"]
    assert [(row["outcome"], row["validated"]) for row in experiences] == [
        ("VALIDATED_BINDING", 1),
    ]

    second = dialogue.query(
        dialogue.create()["hive"]["id"],
        "Что механик дал роботу?",
    )
    shadow = second["query_graph"]["event_pattern"]["target_gaps"][0][
        "evidence"
    ]["learned_gap_profile"]
    assert second["answer"]["surface"] == "болт."
    assert shadow["mode"] == "SHADOW"
    assert shadow["support_count"] == 1
    assert shadow["compatible_local_slots"]
    assert "D-Q-local_slot-" in shadow["projection_dimensions"][
        "local_slot"
    ]["dimension_id"]


def test_query_operator_support_alone_never_promotes_shadow_profile():
    repository, training, dialogue = services()
    training.train(
        "Механик дал роботу болт.",
        independent_key="query-operator-no-automatic-promotion",
    )
    for _ in range(3):
        result = dialogue.query(
            dialogue.create()["hive"]["id"],
            "Что механик дал роботу?",
        )
        assert result["answer"]["surface"] == "болт."

    with repository.transaction() as conn:
        profile = conn.execute(
            """SELECT validated_count,status
               FROM query_operator_profiles WHERE profile_key='surface:что'"""
        ).fetchone()
    assert int(profile["validated_count"]) == 3
    assert profile["status"] == "SHADOW"


def test_active_instrument_construction_does_not_override_passive_structure():
    _, training, dialogue = services()
    for index, text in enumerate((
        "Повар разрезал хлеб ножом.",
        "Девочка разрезала яблоко ножницами.",
        "Мастер разрезал провод кусачками.",
        "Автомат разрезал лист резаком.",
        "Бобр разрезал ветку зубами.",
        "Хозяин разрезал корм ножом.",
        "Оператор разрезал ленту ножницами.",
        "Клиент разрезал упаковку ножом.",
        "Ткань была разрезана портным ножницами.",
    )):
        training.train(
            text,
            independent_key=f"query-operator-passive:{index}",
        )

    result = dialogue.query(
        dialogue.create()["hive"]["id"],
        "Кем была разрезана ткань?",
    )

    assert result["answer"]["status"] == "RESOLVED"
    assert result["answer"]["surface"] == "портным."


def test_multi_gap_query_creates_an_independent_operator_occurrence_per_gap():
    repository, training, dialogue = services()
    training.train(
        "Механик дал роботу болт.",
        independent_key="query-operator-multi-gap",
    )
    result = dialogue.query(
        dialogue.create()["hive"]["id"],
        "Кто, кому и что дал?",
    )
    assert result["answer"]["status"] == "RESOLVED"

    with repository.transaction() as conn:
        rows = conn.execute(
            """SELECT operator_normalized,status
               FROM query_operator_occurrences ORDER BY operator_normalized"""
        ).fetchall()
        profiles = conn.execute(
            """SELECT compatible_slots_json
               FROM query_operator_profiles ORDER BY profile_key"""
        ).fetchall()
        experiences = conn.execute(
            """SELECT occurrence_id FROM query_operator_experiences
               WHERE outcome='VALIDATED_BINDING'"""
        ).fetchall()
    assert [(row["operator_normalized"], row["status"]) for row in rows] == [
        ("кому", "VALIDATED"),
        ("кто", "VALIDATED"),
        ("что", "VALIDATED"),
    ]
    assert len({row["occurrence_id"] for row in experiences}) == 3
    assert all(decode(row["compatible_slots_json"], {}) for row in profiles)


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
