from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.v2.graph_models import ObservationSignature
from server.v2.graph_repository import GraphRepository
from server.v2.graph_service import GraphDialogueService, GraphTrainingService


def services():
    repository = GraphRepository()
    return (
        repository,
        GraphTrainingService(repository),
        GraphDialogueService(repository),
    )


def ask(dialogue: GraphDialogueService, text: str) -> dict:
    hive_id = dialogue.create()["hive"]["id"]
    return dialogue.query(hive_id, text)


def test_fresh_schema_contains_only_graph_slot_and_gap_contracts():
    repository, _, _ = services()
    with repository.transaction() as conn:
        tables = {
            row["name"]: str(row["sql"] or "").casefold()
            for row in conn.execute(
                """SELECT name,sql FROM sqlite_master
                   WHERE type='table' ORDER BY name"""
            ).fetchall()
        }
        assert {
            "graph_events",
            "graph_participants",
            "local_slots",
            "slot_prototypes",
            "construction_clusters",
            "query_graphs",
            "candidate_bindings",
            "lexemes",
            "word_forms",
            "word_usages",
        }.issubset(tables)
        full_schema = "\n".join(tables.values())
        for prohibited in (
            "requested_role",
            "fixed_roles",
            "semantic_role",
            "agent",
            "patient",
            "recipient",
        ):
            assert prohibited not in full_schema


def test_incompatible_database_is_recreated_without_legacy_tables(
    isolated_database,
):
    isolated_database.unlink()
    conn = sqlite3.connect(isolated_database)
    conn.execute(
        "CREATE TABLE old_scenes(id INTEGER PRIMARY KEY, agent TEXT)"
    )
    conn.commit()
    conn.close()
    repository = GraphRepository()
    with repository.transaction() as conn:
        tables = {
            str(row["name"])
            for row in conn.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='table'"""
            ).fetchall()
        }
    assert "old_scenes" not in tables
    assert "graph_events" in tables


def test_observation_signature_accepts_only_observable_namespaces():
    signature = ObservationSignature({
        "morph:case:nomn": 0.8,
        "position:before_predicate": 0.7,
    })
    assert signature.as_dict()["morph:case:nomn"] == 0.8
    with pytest.raises(ValueError):
        ObservationSignature({"agent": 1.0})


def test_event_persists_participants_top_k_morphology_and_distinct_slots():
    repository, training, _ = services()
    result = training.train(
        "Борис настроил датчик.",
        independent_key="diagnostic-1",
    )
    event = result["events"][0]
    assert len(event["participants"]) == 2
    selected_slots = [
        participant["slot_hypotheses"][0]["local_slot_id"]
        for participant in event["participants"]
    ]
    assert len(set(selected_slots)) == 2
    assert all(
        "semantic_role" not in participant
        for participant in event["participants"]
    )
    with repository.transaction() as conn:
        hypotheses = int(conn.execute(
            "SELECT COUNT(*) FROM graph_morph_hypotheses"
        ).fetchone()[0])
        tokens = int(conn.execute(
            "SELECT COUNT(*) FROM graph_tokens"
        ).fetchone()[0])
        assert hypotheses > tokens


def test_local_slots_reach_stability_only_with_diverse_support():
    repository, training, _ = services()
    examples = [
        ("Борис настроил датчик.", "people"),
        ("Анна настроила терминал.", "office"),
        ("Робот настроил контроллер.", "robotics"),
        ("Система настроила соединение.", "software"),
        ("Инженер настроил сервер.", "infrastructure"),
        ("Модуль настроил канал.", "electronics"),
    ]
    for index, (text, domain) in enumerate(examples):
        training.train(
            text,
            independent_key=f"stable-{index}",
            domain_key=domain,
        )
    with repository.transaction() as conn:
        rows = conn.execute(
            """SELECT support_count,contradiction_count,domain_diversity,
                      confidence,status,centroid_signature_json
               FROM local_slots ORDER BY support_count DESC"""
        ).fetchall()
        assert rows
        assert any(
            row["status"] in {"STABLE", "GENERALIZED"}
            and int(row["support_count"]) >= 5
            and int(row["domain_diversity"]) >= 2
            for row in rows
        )


def test_generalized_prototype_support_is_not_double_counted():
    repository, training, _ = services()
    examples = [
        ("Борис настроил датчик.", "configure-1"),
        ("Анна настроила терминал.", "configure-2"),
        ("Робот настроил контроллер.", "configure-3"),
        ("Система настроила соединение.", "configure-4"),
        ("Инженер настроил сервер.", "configure-5"),
        ("Модуль настроил канал.", "configure-6"),
        ("Борис открыл дверь.", "open-1"),
        ("Анна открыла окно.", "open-2"),
        ("Робот открыл шлюз.", "open-3"),
        ("Система открыла порт.", "open-4"),
        ("Инженер открыл люк.", "open-5"),
        ("Модуль открыл клапан.", "open-6"),
    ]
    for text, domain in examples:
        training.train(text, independent_key=domain, domain_key=domain)
    with repository.transaction() as conn:
        prototypes = conn.execute(
            "SELECT id,support_count FROM slot_prototypes"
        ).fetchall()
        assert prototypes
        for prototype in prototypes:
            member_support = int(conn.execute(
                """SELECT COALESCE(SUM(s.support_count),0)
                   FROM slot_prototype_members m
                   JOIN local_slots s ON s.id=m.local_slot_id
                   WHERE m.prototype_id=? AND s.status<>'DEPRECATED'""",
                (prototype["id"],),
            ).fetchone()[0])
            assert int(prototype["support_count"]) == member_support


def test_query_graph_binds_different_event_attachments_without_named_types():
    _, training, dialogue = services()
    training.train("Борис настроил датчик.", independent_key="binding")
    first = ask(dialogue, "Кто настроил датчик?")
    second = ask(dialogue, "Что настроил Борис?")
    assert first["answer"]["surface"] == "Борис."
    assert second["answer"]["surface"] == "датчик."
    for result in (first, second):
        graph = result["query_graph"]
        assert graph["event_pattern"]["gap_node"]["gap_kind"] == "EVENT_ATTACHMENT"
        serialized = str(graph)
        assert "requested_role" not in serialized
        assert "fixed_roles" not in serialized


def test_question_function_reweights_homonyms_and_distinguishes_three_slots():
    _, training, dialogue = services()
    training.train(
        "Борис передал Марине книгу.",
        independent_key="three-participants",
    )
    item = ask(dialogue, "Что передал Борис?")
    addressee = ask(dialogue, "Кому Борис передал книгу?")
    assert item["answer"]["surface"] == "книгу."
    assert addressee["answer"]["surface"] == "Марине."
    selected = next(
        hypothesis
        for hypothesis in item["trace"]["token_hypotheses"][0]["hypotheses"]
        if hypothesis["selected"]
    )
    assert selected["pos"] == "NPRO"
    assert "question_operator_function" in selected["evidence"]


def test_coordinated_question_binds_every_requested_gap_in_one_event():
    _, training, dialogue = services()
    training.train(
        "Механик дал роботу болт.",
        independent_key="coordinated-gaps",
    )
    result = ask(dialogue, "Кто, кому и что дал?")

    pattern = result["query_graph"]["event_pattern"]
    assert [gap["surface"] for gap in pattern["target_gaps"]] == [
        "Кто", "кому", "что",
    ]
    assert "target_gap" not in pattern
    assert len(pattern["implicit_gaps"]) == 0
    assert result["answer"]["surface"] == "Механик дал роботу болт."
    assert [item["resolved_surface"] for item in result["selected_bindings"]] == [
        "Механик", "роботу", "болт",
    ]
    assert result["answer"]["validation"]["all_requested_gaps_bound"]


def test_follow_up_rotates_gap_within_the_selected_event():
    _, training, dialogue = services()
    training.train("Механик дал роботу болт.", independent_key="rotate-bolt")
    training.train("Оператор дал дрону посылку.", independent_key="rotate-parcel")
    hive_id = dialogue.create()["hive"]["id"]

    first = dialogue.query(hive_id, "Что механик дал роботу?")
    second = dialogue.query(hive_id, "А кто дал?")
    third = dialogue.query(hive_id, "А что?")

    assert first["answer"]["surface"] == "болт."
    assert second["answer"]["surface"] == "Механик."
    assert third["answer"]["surface"] == "болт."
    assert second["query_graph"]["continuation_of"] == (
        first["query_graph"]["query_graph_id"]
    )
    assert second["query_graph"]["trace"]["event_anchor_id"] == (
        first["selected_binding"]["event_id"]
    )
    predicate = third["query_graph"]["event_pattern"]["predicate"]
    assert predicate["origin"] == "INHERITED"
    assert predicate["token_index"] is None
    assert predicate["source_token_index"] is not None
    assert {
        node["head"]["lemma"]
        for node in third["query_graph"]["event_pattern"]["known_nodes"]
    } == {"механик", "робот"}


def test_bare_what_keeps_case_hypotheses_and_separates_implicit_agreement_gap():
    _, training, dialogue = services()
    training.train(
        "Мальчик разрезал яблоко ножом.",
        independent_key="cut-apple",
    )
    created = dialogue.create(conversation_id="cut-conversation")
    result = dialogue.query(created["hive"]["id"], "Что разрезал?")

    assert result["answer"]["surface"] == "яблоко."
    pattern = result["query_graph"]["event_pattern"]
    target = pattern["target_gap"]
    assert target["requested"] is True
    assert target["morphology_hypotheses"]["case:accs"] > 0
    assert target["morphology_hypotheses"]["case:nomn"] > 0
    assert "agreement:predicate_gender" not in target["question_signature"]
    assert len(pattern["implicit_gaps"]) == 1
    implicit = pattern["implicit_gaps"][0]
    assert implicit["evidence"]["predicate_gender"] == "masc"
    assert implicit["evidence"]["predicate_number"] == "sing"

    scores = {
        item["resolved_lemma"]: item["scores"]["total"]
        for item in result["candidate_bindings"]
    }
    assert scores["яблоко"] - scores["мальчик"] > 0.08
    knife = next(
        item for item in result["candidate_bindings"]
        if item["resolved_lemma"] == "нож"
    )
    assert knife["slot_compatibility_state"] == "below_threshold"
    assert knife["evidence"][0]["slot_compatibility"]["reason"] == (
        "LOCAL_SLOT_NOT_IN_COMPATIBLE_HYPOTHESES"
    )

    assert result["trace"]["preliminary_query_graph"]["trace"][
        "language_analysis"
    ]["utterance"]["conversation_id"] == "cut-conversation"
    assert result["trace"]["preliminary_query_graph"]["trace"][
        "language_analysis"
    ]["utterance"]["turn_index"] == 1


def test_event_trace_is_deduplicated_and_participant_provenance_is_nested():
    _, training, dialogue = services()
    training.train(
        "Мальчик разрезал яблоко ножом.",
        independent_key="trace-cut",
    )
    result = ask(dialogue, "Что разрезал?")
    event_candidates = result["trace"]["event_candidates"]
    assert len(event_candidates) == len({item["event_id"] for item in event_candidates})
    assert len(event_candidates) == 1
    signatures = result["trace"]["participant_signatures"]
    assert len(signatures) == 1
    assert signatures[0]["event_signature"]["event_id"] == event_candidates[0]["event_id"]
    assert len(signatures[0]["participants"]) == 3


def test_permutation_uses_known_node_and_gap_signatures_together():
    _, training, dialogue = services()
    training.train("Кот съел рыбу.", independent_key="permutation-1")
    training.train("Рыба съела мышь.", independent_key="permutation-2")
    result = ask(dialogue, "Что съела рыба?")
    assert result["answer"]["surface"] == "мышь."
    assert result["candidate_bindings"][0]["scores"]["structural"] == 1.0
    assert result["candidate_bindings"][1]["scores"]["structural"] < 1.0


def test_node_component_gap_returns_observed_modifier():
    _, training, dialogue = services()
    training.train(
        "Бабочки живут короткую жизнь.",
        independent_key="modifier",
    )
    result = ask(dialogue, "Какую жизнь живут бабочки?")
    assert result["query_graph"]["event_pattern"]["gap_node"]["gap_kind"] == (
        "NODE_COMPONENT"
    )
    assert result["answer"]["surface"] == "короткую."


def test_required_mention_component_rejects_conflicting_event():
    _, training, dialogue = services()
    training.train(
        "Картина находится в синем зале.",
        independent_key="blue",
    )
    training.train(
        "Словарь находится в читальном зале.",
        independent_key="reading",
    )
    result = ask(dialogue, "Что находится в синем зале?")
    assert result["answer"]["surface"] == "Картина."
    rejection = next(
        item for item in result["rejected_events"]
        if item.get("failed_constraint") == "KNOWN_MENTION_COMPONENT"
    )
    assert rejection["reason"] == "REQUIRED_COMPONENT_MISMATCH"


def test_active_passive_and_free_order_keep_predicate_and_answers_compatible():
    repository, training, dialogue = services()
    for index, text in enumerate((
        "Борис настроил датчик.",
        "Датчик был настроен Борисом.",
        "Датчик Борис настроил.",
        "Настроил датчик Борис.",
    )):
        training.train(text, independent_key=f"voice-{index}")
    with repository.transaction() as conn:
        concepts = {
            str(row["predicate_concept_id"])
            for row in conn.execute(
                "SELECT predicate_concept_id FROM graph_events"
            ).fetchall()
        }
        assert len(concepts) == 1
    passive = ask(dialogue, "Кем был настроен датчик?")
    active = ask(dialogue, "Кто настроил датчик?")
    assert passive["answer"]["surface"] == "Борисом."
    assert active["answer"]["surface"] == "Борис."


def test_answer_form_is_generated_from_gap_when_only_passive_fact_exists():
    _, training, dialogue = services()
    training.train(
        "Датчик был настроен Борисом.",
        independent_key="passive-only",
    )
    assert ask(dialogue, "Кто настроил датчик?")["answer"]["surface"] == "Борис."
    assert (
        ask(dialogue, "Кем был настроен датчик?")["answer"]["surface"]
        == "Борисом."
    )


def test_relation_and_quantity_gaps_bind_observed_values():
    _, training, dialogue = services()
    training.train(
        "Автобус идёт до Казани.",
        independent_key="relation-value",
    )
    training.train(
        "Борис настроил три датчика.",
        independent_key="quantity-value",
    )
    relation = ask(dialogue, "До какого города идёт автобус?")
    quantity = ask(dialogue, "Сколько датчиков настроил Борис?")
    assert relation["answer"]["surface"] == "Казани."
    assert (
        relation["query_graph"]["event_pattern"]["gap_node"]["gap_kind"]
        == "RELATION_VALUE"
    )
    assert quantity["answer"]["surface"] == "три."
    assert (
        quantity["query_graph"]["event_pattern"]["gap_node"]["gap_kind"]
        == "QUANTITY_VALUE"
    )


def test_follow_up_replaces_gap_and_inherits_prior_binding():
    _, training, dialogue = services()
    training.train(
        "Администратор перезапустил сервер ночью.",
        independent_key="follow-up",
    )
    hive_id = dialogue.create()["hive"]["id"]
    first = dialogue.query(hive_id, "Кто перезапустил сервер?")
    second = dialogue.query(hive_id, "А когда?")
    assert first["answer"]["surface"] == "Администратор."
    assert second["answer"]["surface"] == "ночью."
    assert second["query_graph"]["continuation_of"] == (
        first["query_graph"]["query_graph_id"]
    )
    assert second["query_graph"]["trace"]["inherited_previous_binding"] is True


def test_event_property_keeps_preposition_and_event_local_full_answer():
    _, training, dialogue = services()
    trained = training.train(
        "На столе лежит красное яблоко. "
        "На подоконнике лежит красный помидор.",
        independent_key="event-local-answer",
    )
    assert [event["source_surface"] for event in trained["events"]] == [
        "На столе лежит красное яблоко",
        "На подоконнике лежит красный помидор",
    ]

    result = ask(dialogue, "Где лежит помидор?")
    assert result["answer"]["short_answer"] == "На подоконнике."
    assert result["answer"]["full_answer"] == (
        "На подоконнике лежит красный помидор."
    )
    assert result["selected_binding"]["resolved_features"]["preposition"] == "на"


def test_bare_noun_follow_up_replaces_known_node_and_reuses_gap():
    _, training, dialogue = services()
    training.train(
        "На столе лежит красное яблоко. "
        "На подоконнике лежит красный помидор.",
        independent_key="replace-known-node",
    )
    hive_id = dialogue.create()["hive"]["id"]
    first = dialogue.query(hive_id, "Где лежит помидор?")
    second = dialogue.query(hive_id, "Яблоко?")

    assert first["answer"]["surface"] == "На подоконнике."
    assert second["answer"]["surface"] == "На столе."
    pattern = second["query_graph"]["event_pattern"]
    assert second["query_graph"]["continuation_of"] == (
        first["query_graph"]["query_graph_id"]
    )
    assert pattern["predicate"]["lemma"] == "лежать"
    assert [node["head"]["lemma"] for node in pattern["known_nodes"]] == [
        "яблоко"
    ]
    assert pattern["gap_node"]["gap_kind"] == "EVENT_PROPERTY"
    assert pattern["gap_node"]["question_signature"] == (
        first["query_graph"]["event_pattern"]["gap_node"]["question_signature"]
    )


def test_local_slot_identity_survives_free_word_order():
    _, training, _ = services()
    first = training.train(
        "На подоконнике лежит помидор.",
        independent_key="slot-before-predicate",
    )
    second = training.train(
        "Помидор лежит на подоконнике.",
        independent_key="slot-after-predicate",
    )
    first_location = next(
        participant for participant in first["events"][0]["participants"]
        if participant["mention"]["preposition"] == "на"
    )
    second_location = next(
        participant for participant in second["events"][0]["participants"]
        if participant["mention"]["preposition"] == "на"
    )
    assert (
        first_location["slot_hypotheses"][0]["local_slot_id"]
        == second_location["slot_hypotheses"][0]["local_slot_id"]
    )


def test_more_reuses_gap_and_accumulates_canonical_exclusions_after_restart():
    repository, training, dialogue = services()
    training.train(
        "Сергей хранит контейнер на складе.",
        independent_key="more-1",
    )
    training.train(
        "Сергей хранит сканер на складе.",
        independent_key="more-2",
    )
    hive_id = dialogue.create()["hive"]["id"]
    first = dialogue.query(hive_id, "Что Сергей хранит на складе?")
    restarted = GraphDialogueService(repository)
    second = restarted.query(hive_id, "А ещё?")
    third = restarted.query(hive_id, "А ещё?")
    assert [
        first["answer"]["surface"],
        second["answer"]["surface"],
        third["answer"]["surface"],
    ] == [
        "контейнер.",
        "сканер.",
        "В доступной памяти других подходящих значений нет.",
    ]
    assert len(third["query_graph"]["exclusions"]) == 2


def test_synthetic_vocabulary_uses_structure_not_known_lexicon():
    _, training, dialogue = services()
    training.train(
        "Флукс мерцал нарвис.",
        independent_key="anti-hardcode",
    )
    result = ask(dialogue, "Кто мерцал нарвис?")
    assert result["answer"]["surface"] == "Флукс."


def test_display_labels_do_not_change_binding():
    repository, training, dialogue = services()
    training.train("Борис настроил датчик.", independent_key="labels")
    before = ask(dialogue, "Кто настроил датчик?")["answer"]["surface"]
    with repository.transaction() as conn:
        conn.execute("UPDATE local_slots SET display_label='renamed'")
        conn.execute("UPDATE semantic_clusters SET display_label='переведено'")
        conn.execute("UPDATE slot_prototypes SET display_label='anything'")
    after = ask(dialogue, "Кто настроил датчик?")["answer"]["surface"]
    assert before == after == "Борис."


def test_questions_and_hypotheses_do_not_enter_confirmed_world_memory():
    repository, training, _ = services()
    question = training.train(
        "Борис настроил датчик?",
        independent_key="unsafe-question",
    )
    hypothesis = training.train(
        "Наверное, Борис настроил датчик.",
        independent_key="unsafe-hypothesis",
    )
    assert question["status"] == "STAGED"
    assert hypothesis["status"] == "STAGED"
    assert question["events"] == []
    assert hypothesis["events"] == []
    with repository.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM graph_events"
        ).fetchone()[0] == 0


def test_manual_commit_cannot_turn_question_into_world_fact():
    repository, training, _ = services()
    staged = training.stage(
        "Борис настроил датчик?",
        independent_key="manual-question",
    )
    committed = training.commit(
        staged["source_id"],
        manual_validation=True,
    )
    assert committed["status"] == "STAGED"
    assert committed["events"] == []
    with repository.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM graph_events"
        ).fetchone()[0] == 0


def test_staging_duplicate_does_not_demote_confirmed_source():
    _, training, dialogue = services()
    trained = training.train(
        "Борис настроил датчик.",
        independent_key="monotonic-source",
    )
    duplicate = training.stage(
        "Борис настроил датчик.",
        independent_key="monotonic-source",
    )
    assert duplicate["source_id"] == trained["source_id"]
    assert duplicate["status"] == "CONFIRMED"
    assert ask(dialogue, "Кто настроил датчик?")["answer"]["surface"] == "Борис."


def test_batches_commit_and_rollback_only_their_own_sources():
    repository, training, dialogue = services()
    first = training.preview_batch([{
        "text": "Борис настроил датчик.",
        "independent_key": "batch-first",
    }])
    second = training.preview_batch([{
        "text": "Анна открыла дверь.",
        "independent_key": "batch-second",
    }])
    committed = training.commit_batch(first["batch_id"])
    rolled_back = training.rollback_batch(second["batch_id"])
    assert committed["status"] == "COMMITTED"
    assert rolled_back["status"] == "ROLLED_BACK"
    assert ask(dialogue, "Кто настроил датчик?")["answer"]["surface"] == "Борис."
    assert ask(dialogue, "Кто открыл дверь?")["answer"]["status"] == "UNRESOLVED"
    with repository.transaction() as conn:
        states = {
            str(row["id"]): str(row["status"])
            for row in conn.execute(
                "SELECT id,status FROM graph_batches"
            ).fetchall()
        }
    assert states == {
        first["batch_id"]: "COMMITTED",
        second["batch_id"]: "ROLLED_BACK",
    }


def test_query_graph_identity_is_scoped_to_hive_and_turn():
    repository, training, dialogue = services()
    training.train("Борис настроил датчик.", independent_key="query-identity")
    first_hive = dialogue.create()["hive"]["id"]
    second_hive = dialogue.create()["hive"]["id"]
    first = dialogue.query(first_hive, "Кто настроил датчик?")
    second = dialogue.query(second_hive, "Кто настроил датчик?")
    assert (
        first["query_graph"]["query_graph_id"]
        != second["query_graph"]["query_graph_id"]
    )
    with repository.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM query_graphs"
        ).fetchone()[0] == 2


def test_explicit_new_query_does_not_inherit_previous_graph():
    _, training, dialogue = services()
    training.train(
        "Администратор перезапустил сервер ночью.",
        independent_key="new-query-boundary",
    )
    hive_id = dialogue.create()["hive"]["id"]
    dialogue.query(hive_id, "Кто перезапустил сервер?")
    result = dialogue.query(
        hive_id,
        "А когда?",
        resolved_mode="NEW_QUERY",
    )
    assert result["answer"]["status"] == "UNRESOLVED"
    assert result["query_graph"]["continuation_of"] is None


def test_boolean_gap_checks_polarity():
    _, training, dialogue = services()
    training.train("Кот ест рыбу.", independent_key="polarity")
    positive = ask(dialogue, "Кот ест рыбу?")
    negative = ask(dialogue, "Кот не ест рыбу?")
    assert positive["answer"]["surface"] == "Да."
    assert negative["answer"]["surface"] == "Нет."
    assert (
        positive["query_graph"]["event_pattern"]["gap_node"]["gap_kind"]
        == "BOOLEAN_RESULT"
    )


def test_resolved_answer_creates_dependent_training_episode_not_world_event():
    repository, training, dialogue = services()
    training.train("Борис настроил датчик.", independent_key="episode")
    result = ask(dialogue, "Кто настроил датчик?")
    assert result["answer"]["validation"]["valid"] is True
    with repository.transaction() as conn:
        episode = conn.execute(
            """SELECT eligible_for_learning,event_ids_json
               FROM training_episodes"""
        ).fetchone()
        assert int(episode["eligible_for_learning"]) == 1
        assert conn.execute(
            """SELECT COUNT(*) FROM knowledge_sources
               WHERE source_type='dialogue_question' AND status='STAGED'"""
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM graph_events"
        ).fetchone()[0] == 1


def test_retraction_removes_event_from_search_and_weakens_dependencies():
    repository, training, dialogue = services()
    trained = training.train(
        "Борис настроил датчик.",
        independent_key="retraction",
    )
    assert ask(dialogue, "Кто настроил датчик?")["answer"]["surface"] == "Борис."
    result = training.retract(trained["source_id"], "invalidated")
    assert result["status"] == "RETRACTED"
    assert result["recalculated_slot_ids"]
    unknown = ask(dialogue, "Кто настроил датчик?")
    assert unknown["answer"]["status"] == "UNRESOLVED"


def test_versions_and_trace_are_saved_with_every_answer():
    _, training, dialogue = services()
    trained = training.train(
        "Борис настроил датчик.",
        independent_key="versions",
    )
    result = ask(dialogue, "Кто настроил датчик?")
    assert trained["events"][0]["versions"]["event_schema_version"] == "2.7.0"
    assert result["answer"]["versions"]["query_graph_version"] == "2.7.0"
    trace = result["trace"]
    assert trace["final_query_graph"]
    assert trace["candidate_bindings"]
    assert trace["selected_binding"]
    assert trace["validation"]["valid"] is True


def test_active_pipeline_has_no_named_participant_contracts():
    root = Path(__file__).parents[1]
    active_files = [
        "server/server.py",
        "server/database.py",
        "server/modules/training/application/services.py",
        "server/modules/hive/application/services.py",
        "server/v2/event_graph.py",
        "server/v2/graph_learning.py",
        "server/v2/graph_models.py",
        "server/v2/graph_repository.py",
        "server/v2/graph_schema.py",
        "server/v2/graph_service.py",
        "server/v2/hive.py",
        "server/v2/query_graph.py",
        "server/v2/query_scene.py",
        "server/v2/language/analyzer.py",
        "server/v2/language/clause_parser.py",
        "server/v2/language/scope_parser.py",
        "server/v2/language/question_operator_parser.py",
        "server/v2/language/relation_phrase_parser.py",
    ]
    prohibited = re.compile(
        r"\b(agent|patient|recipient|requested_role|fixed_roles|semantic_role)\b",
        re.IGNORECASE,
    )
    for relative_path in active_files:
        source = (root / relative_path).read_text(encoding="utf-8")
        assert prohibited.search(source) is None, relative_path


def test_http_api_exposes_graph_contract(isolated_database):
    from server.server import app

    with TestClient(app) as client:
        learned = client.post(
            "/api/v2/training/learn",
            json={
                "text": "Борис настроил датчик.",
                "independent_key": "api",
            },
        )
        assert learned.status_code == 200
        created = client.post("/api/v2/hives", json={})
        hive_id = created.json()["hive"]["id"]
        queried = client.post(
            f"/api/v2/hives/{hive_id}/query",
            json={"text": "Кто настроил датчик?"},
        )
        assert queried.status_code == 200
        payload = queried.json()
        assert payload["answer"]["surface"] == "Борис."
        assert "query_graph" in payload
        health = client.get("/api/health").json()
        assert health == {
            "status": "ok",
            "model": "role-free-event-graph",
            "version": "v2.7",
        }
