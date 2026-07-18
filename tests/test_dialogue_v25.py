from dataclasses import FrozenInstanceError

import pytest

from server.v2.dialogue_state import DialogueStateService
from server.v2.hive import V2HiveService
from server.v2.language import UniversalLanguageAnalyzer
from server.v2.language.evidence import EvidenceAggregator
from server.v2.language.models import (
    DialogueState,
    EvidencePacket,
    InterpretationHypothesis,
    ResponseType,
)
from server.v2.repository import V2Repository
from server.v2.response_planner import ResponsePlanner
from server.v2.training import RussianMorphology, TrainingPipelineV2


def analyzer():
    return UniversalLanguageAnalyzer(RussianMorphology())


def test_compound_utterance_has_acts_clauses_evidence_and_stable_cycles():
    analysis = analyzer().analyze(
        "Привет! Если робот готов, запусти его.",
        conversation_id="dialogue-v25",
        turn_index=1,
    )

    assert [act.act_type.value for act in analysis.dialogue_acts] == [
        "GREETING",
        "CONDITION",
        "COMMAND",
    ]
    assert [clause.mode.value for clause in analysis.clauses] == [
        "ASSERTION",
        "CONDITION",
        "COMMAND",
    ]
    assert analysis.clauses[1].actuality.value == "HYPOTHETICAL"
    assert [
        relation.relation_type.value
        for relation in analysis.clause_relations
    ] == ["CONDITION"]
    assert analysis.interpretation_status.value == "STABLE"
    assert analysis.interpretation_trace["cycles_completed"] == 2
    assert analysis.interpretation_trace["stop_reason"] == "STABLE_INTERPRETATION"
    assert analysis.evidence_packets
    assert {
        packet.independent_group for packet in analysis.evidence_packets
    } >= {"morphology", "syntax", "discourse"}
    assert len({
        packet.dedupe_key for packet in analysis.evidence_packets
    }) == len(analysis.evidence_packets)
    boundary_hypotheses = [
        hypothesis for hypothesis in analysis.hypotheses
        if hypothesis.hypothesis_type == "clause_boundary"
    ]
    assert boundary_hypotheses
    assert all(
        set(hypothesis.value) == {"token_start", "token_end"}
        for hypothesis in boundary_hypotheses
    )


def test_negation_scope_modality_reported_speech_and_multiple_events():
    language = analyzer()
    role_value = language.analyze(
        "Артём ремонтирует не робота, а двигатель."
    )
    participant = language.analyze(
        "Не Артём ремонтирует робота."
    )
    frequency = language.analyze(
        "Артём не всегда ремонтирует робота."
    )
    desire = language.analyze(
        "Артём хотел отремонтировать робота, но не успел."
    )
    report = language.analyze(
        "Анна сказала, что робот готов."
    )
    definition = language.analyze(
        "Ремонтировать — это чинить."
    )
    location = language.analyze(
        "Что продают на рынке?"
    )

    assert role_value.clauses[0].negation_scope["scope_type"] == "ROLE_VALUE"
    assert (
        role_value.clauses[0].negation_scope["asserted_alternative"]["lemma"]
        == "двигатель"
    )
    assert participant.clauses[0].negation_scope["scope_type"] == "PARTICIPANT"
    assert frequency.clauses[0].negation_scope["scope_type"] == "FREQUENCY"
    assert desire.clauses[0].modality.value == "WANT"
    assert desire.clauses[0].completion_status.value == "NOT_STARTED"
    assert desire.clauses[1].completion_status.value == "INTERRUPTED"
    assert desire.interpretation_status.value == "STABLE"
    assert len(report.clauses) == 2
    assert report.clauses[1].mode.value == "REPORTED_SPEECH"
    assert report.clauses[1].quoted_speaker == "Анна"
    assert [
        act.act_type.value for act in definition.dialogue_acts
    ] == ["DEFINITION"]
    assert definition.clauses[0].mode.value == "DEFINITION"
    assert definition.interpretation_status.value == "STABLE"
    assert [
        item["definition_role"]
        for item in definition.clauses[0].predicate_hypotheses
    ] == ["defined_term", "definition_value"]
    assert next(
        participant
        for participant in location.clauses[0].participants
        if participant["lemma"] == "рынок"
    )["role_hypotheses"][0]["role"] == "location"


def test_evidence_is_deeply_immutable_and_duplicate_safe():
    hypothesis = InterpretationHypothesis(
        id="hypothesis-test",
        scope_type="clause",
        scope_id="clause-test",
        hypothesis_type="predicate",
        value="продавать",
    )
    packet = EvidencePacket(
        id="evidence-test",
        origin="morphology",
        target_hypothesis_id=hypothesis.id,
        value={"lemma": "продавать"},
        support=0.5,
        evidence_type="verb_parse",
        independent_group="morphology",
        scope_type="clause",
        scope_id="clause-test",
        source_token_start=1,
        source_token_end=1,
    )

    with pytest.raises(FrozenInstanceError):
        packet.support = 0.9
    with pytest.raises(TypeError):
        packet.value["lemma"] = "изменить"

    aggregator = EvidenceAggregator()
    once = aggregator.aggregate(hypothesis, [packet]).support
    duplicated = aggregator.aggregate(hypothesis, [packet, packet]).support
    assert duplicated == once


def test_token_hypotheses_are_scoped_by_utterance():
    language = analyzer()
    first = language.analyze(
        "Кот спит.",
        conversation_id="scope-a",
        turn_index=1,
    )
    second = language.analyze(
        "Кот спит.",
        conversation_id="scope-b",
        turn_index=1,
    )

    first_ids = {
        item.id for item in first.hypotheses
        if item.scope_type == "token"
    }
    second_ids = {
        item.id for item in second.hypotheses
        if item.scope_type == "token"
    }
    assert first_ids.isdisjoint(second_ids)


def test_two_strong_referents_are_ambiguous_not_conflicted():
    analysis = analyzer().analyze(
        "Он осмотрел.",
        conversation_id="reference-status",
        turn_index=1,
        reference_candidates={
            0: [
                {
                    "id": "engineer",
                    "lemma": "инженер",
                    "evidence": {
                        "discourse": 0.9,
                        "temporal_context": 0.9,
                    },
                },
                {
                    "id": "robot",
                    "lemma": "робот",
                    "evidence": {
                        "discourse": 0.9,
                        "temporal_context": 0.9,
                    },
                },
            ],
        },
    )

    assert analysis.interpretation_status.value == "AMBIGUOUS"


def test_dialogue_state_persists_focus_questions_and_correction():
    TrainingPipelineV2().train(
        "Рыбу продают на рынке. В магазине продают мясо."
    )
    service = V2HiveService()
    hive_id = service.create(
        conversation_id="dialogue-v25-correction"
    )["hive"]["id"]

    first = service.query(hive_id, "Что продают на рынке?")
    corrected = service.query(
        hive_id,
        "Нет, я имел в виду магазин.",
    )

    assert first["query_frame"]["requested_role"] == "object"
    assert corrected["resolved_mode"] == "CORRECTION"
    assert corrected["query_frame"]["correction"]["target_role"] == "location"
    assert corrected["query_frame"]["roles"]["location"]["lemma"] == "магазин"
    assert corrected["query_frame"]["roles"]["object"]["status"] == "empty"
    assert "previous" not in corrected["query_frame"]["correction"]
    assert "replacement" not in corrected["query_frame"]["correction"]
    assert [candidate["lemma"] for candidate in corrected["candidates"]] == [
        "мясо"
    ]

    state = DialogueStateService(V2Repository()).load(
        "dialogue-v25-correction"
    )
    assert state.active_topic
    assert state.focus_stack
    assert state.pending_questions
    assert "source_text" not in state.last_query_frame
    assert "tokens" not in state.last_query_frame
    assert "clauses" not in state.last_query_frame
    exported = service.export(hive_id)["dialogue_v2_5"]
    assert exported["clause_relations"] is not None
    assert exported["interpretation_hypotheses"]
    assert exported["interpretation_evidence"]
    report = service.analytics(hive_id)["dialogue_v2_5"]
    assert report["evidence_groups"]
    with V2Repository().transaction() as conn:
        assert conn.execute(
            """SELECT COUNT(*) FROM utterances
               WHERE conversation_id='dialogue-v25-correction'"""
        ).fetchone()[0] == 2
        assert conn.execute(
            """SELECT COUNT(*) FROM dialogue_acts
               WHERE act_type='CORRECTION'"""
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT status FROM query_frames WHERE id=?",
            (first["query_frame"]["id"],),
        ).fetchone()["status"] == "SUPERSEDED"
        assert conn.execute(
            """SELECT COUNT(*) FROM derived_answers
               WHERE conversation_id='dialogue-v25-correction'"""
        ).fetchone()[0] == 0


def test_market_followup_correction_and_deictic_polar_flow():
    TrainingPipelineV2().train(
        "Рыбу продают на рынке. "
        "Мясо продают в магазине. Удочку продают в магазине."
    )
    service = V2HiveService()
    conversation_id = "dialogue-v25-checkpoints"
    hive_id = service.create(
        conversation_id=conversation_id
    )["hive"]["id"]

    first = service.query(hive_id, "Что продают на рынке?")
    first_answer = service.vibration_run(hive_id, 5)["answer"]
    followup = service.query(hive_id, "А ещё что?")

    assert first["query_frame"]["requested_role"] == "object"
    assert first["query_frame"]["roles"]["action"]["lemma"] == "продавать"
    assert first["query_frame"]["roles"]["location"]["lemma"] == "рынок"
    assert first["query_frame"]["roles"]["object"]["status"] == "empty"
    assert first_answer["status"] == "RESOLVED"
    assert followup["query_frame"]["requested_role"] == "object"
    assert followup["query_frame"]["roles"]["location"]["lemma"] == "рынок"
    assert "location" in followup["query_frame"]["inherited_roles"]
    assert first_answer["resolved_value"] in {
        item["surface"]
        for item in followup["query_frame"]["excluded_roles"]["object"]
    }

    corrected = service.query(
        hive_id,
        "Нет, я имел в виду магазин.",
    )

    assert corrected["resolved_mode"] == "CORRECTION"
    assert corrected["query_frame"]["roles"]["location"]["lemma"] == "магазин"
    assert corrected["query_frame"]["roles"]["object"]["status"] == "empty"
    assert corrected["query_frame"]["correction"]["target_query_frame_id"] == (
        followup["query_frame"]["id"]
    )

    polar = service.query(hive_id, "А удочку там продают?")

    assert polar["query_frame"]["query_type"] == "polar_question"
    assert polar["query_frame"]["roles"]["location"]["lemma"] == "магазин"
    assert polar["query_frame"]["roles"]["object"]["lemma"] == "удочка"
    assert polar["answer"]["status"] == "RESOLVED"
    assert polar["answer"]["resolved_value"] is True

    state = DialogueStateService(V2Repository()).load(conversation_id)
    assert state.last_query_frame["roles"]["location"]["lemma"] == "магазин"
    assert state.expected_response is None


def test_ambiguous_pronoun_creates_minimal_pending_clarification():
    state = DialogueState(
        conversation_id="dialogue-v25-reference",
        focus_stack=[
            {
                "id": "entity-engineer",
                "role": "agent",
                "lemma": "инженер",
                "surface": "инженеру",
                "grammatical_features": {
                    "gender": "masc",
                    "number": "sing",
                },
                "activation": 0.9,
                "inertia": 0.8,
            },
            {
                "id": "entity-robot",
                "role": "object",
                "lemma": "робот",
                "surface": "роботу",
                "grammatical_features": {
                    "gender": "masc",
                    "number": "sing",
                },
                "activation": 0.9,
                "inertia": 0.8,
            },
        ],
    )
    analysis = analyzer().analyze(
        "Он его осмотрел.",
        conversation_id=state.conversation_id,
        turn_index=2,
    )

    updated = DialogueStateService().update(state, analysis)

    assert updated.pending_clarification
    assert updated.pending_clarification["slot"] == "referent"
    assert len(updated.pending_clarification["candidates"]) == 2
    assert "инженеру" in updated.pending_clarification["question"]
    assert "роботу" in updated.pending_clarification["question"]


def test_correction_supersedes_but_preserves_previous_derived_answer():
    TrainingPipelineV2().train(
        "Рыбу продают на рынке. В магазине продают мясо."
    )
    service = V2HiveService()
    conversation_id = "dialogue-v25-derived-correction"
    hive_id = service.create(
        conversation_id=conversation_id
    )["hive"]["id"]
    service.query(hive_id, "Что продают на рынке?")
    answer = service.vibration_run(hive_id, 5)["answer"]

    assert answer["status"] == "RESOLVED"
    with V2Repository().transaction() as conn:
        previous = conn.execute(
            """SELECT id,status FROM derived_answers
               WHERE conversation_id=? ORDER BY created_at DESC LIMIT 1""",
            (conversation_id,),
        ).fetchone()
        assert previous["status"] == "DERIVED_ANSWER"
        previous_id = previous["id"]

    service.query(hive_id, "Нет, я имел в виду магазин.")

    with V2Repository().transaction() as conn:
        preserved = conn.execute(
            "SELECT status FROM derived_answers WHERE id=?",
            (previous_id,),
        ).fetchone()
        assert preserved is not None
        assert (
            preserved["status"]
            == "SUPERSEDED_FOR_CURRENT_INTENT"
        )


def test_stage_commit_quarantine_retraction_and_reprocess_are_separate():
    pipeline = TrainingPipelineV2()
    admitted = pipeline.stage(
        "Кот ест рыбу.",
        source_key="source-actual",
    )
    hypothetical = pipeline.stage(
        "Наверное, кот ест рыбу.",
        source_key="source-hypothesis",
    )

    committed = pipeline.commit(admitted["id"])
    quarantined = pipeline.commit(hypothetical["id"])

    assert committed["status"] == "COMMITTED"
    assert committed["decision"] == "ADMIT"
    assert committed["materialized_objects"]
    assert quarantined["decision"] == "QUARANTINE"
    assert "materialized_objects" not in quarantined or not quarantined[
        "materialized_objects"
    ]
    scene_id = next(
        int(item["id"])
        for item in committed["materialized_objects"]
        if item["type"] == "scene"
    )
    with V2Repository().transaction() as conn:
        scene = conn.execute(
            """SELECT source_interpretation_id,admission_decision_id
               FROM scenes WHERE cloud_id=?""",
            (scene_id,),
        ).fetchone()
        event = conn.execute(
            """SELECT admission_decision_id FROM events
               WHERE source_scene_id=?""",
            (scene_id,),
        ).fetchone()
        assert scene["source_interpretation_id"].startswith("hypothesis-")
        assert scene["admission_decision_id"] == committed["id"]
        assert event["admission_decision_id"] == committed["id"]

    retraction = pipeline.retract(admitted["id"], reason="source withdrawn")
    assert retraction["new_status"] == "RETRACTED"
    assert retraction["dependencies"]
    with V2Repository().transaction() as conn:
        scene = conn.execute(
            """SELECT knowledge_status,source_interpretation_id,
                      admission_decision_id
               FROM scenes WHERE cloud_id=?""",
            (scene_id,),
        ).fetchone()
        assert scene["knowledge_status"] == "RETRACTED"
        assert scene["source_interpretation_id"].startswith("hypothesis-")
        assert scene["admission_decision_id"] is None

    reprocessed = pipeline.reprocess(hypothetical["id"])
    assert reprocessed["staging"]["id"] != hypothetical["id"]
    assert reprocessed["staging"]["supersedes_staging_id"] == hypothetical["id"]
    assert reprocessed["history_preserved"] is True


def test_staging_and_commit_are_idempotent():
    pipeline = TrainingPipelineV2()
    first = pipeline.stage(
        "Кот ест рыбу.",
        source_key="idempotent-source",
    )
    second = pipeline.stage(
        "Кот ест рыбу.",
        source_key="idempotent-source",
    )

    assert first["id"] == second["id"]
    with V2Repository().transaction() as conn:
        assert conn.execute(
            """SELECT MAX(observation_count) FROM language_patterns"""
        ).fetchone()[0] == 1

    committed = pipeline.commit(first["id"])
    with V2Repository().transaction() as conn:
        runs_before = conn.execute(
            "SELECT COUNT(*) FROM training_runs"
        ).fetchone()[0]
    repeated = pipeline.commit(first["id"])
    with V2Repository().transaction() as conn:
        runs_after = conn.execute(
            "SELECT COUNT(*) FROM training_runs"
        ).fetchone()[0]

    assert committed["status"] == "COMMITTED"
    assert repeated["status"] == "COMMITTED"
    assert repeated["materialized_objects"] == committed[
        "materialized_objects"
    ]
    assert runs_after == runs_before


def test_manual_validation_cannot_admit_a_question_as_world_fact():
    pipeline = TrainingPipelineV2()
    staged = pipeline.stage(
        "Кот ест рыбу?",
        source_key="question-source",
    )
    result = pipeline.commit(
        staged["id"],
        manual_validation=True,
    )

    assert result["decision"] == "QUARANTINE"
    assert result["factuality_valid"] is False


def test_retracted_staging_cannot_be_recommitted():
    pipeline = TrainingPipelineV2()
    staged = pipeline.stage(
        "Кот ест рыбу.",
        source_key="retracted-source",
    )
    pipeline.commit(staged["id"])
    pipeline.retract(staged["id"], reason="withdrawn")

    with pytest.raises(ValueError):
        pipeline.commit(staged["id"])


def test_retraction_deprecates_source_only_semantic_derivations():
    pipeline = TrainingPipelineV2()
    staged = pipeline.stage(
        "Мастер ремонтирует модуль. Ремонтировать — это чинить.",
        source_key="semantic-source",
    )
    committed = pipeline.commit(staged["id"])
    concept_evidence = [
        item for item in committed["materialized_objects"]
        if item["type"] == "concept_evidence"
    ]

    assert concept_evidence
    concept_id = concept_evidence[0]["concept_id"]
    with V2Repository().transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM semantic_evidence"
        ).fetchone()[0] > 0

    pipeline.retract(staged["id"], reason="definition withdrawn")
    pipeline.rebuild_semantic_projections()
    pipeline.rebuild_concept_relations()

    with V2Repository().transaction() as conn:
        assert conn.execute(
            "SELECT status FROM action_concepts WHERE id=?",
            (concept_id,),
        ).fetchone()["status"] == "DEPRECATED"
        assert conn.execute(
            """SELECT COUNT(*) FROM concept_evidence
               WHERE concept_id=? AND status='ACTIVE'""",
            (concept_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            """SELECT COUNT(*) FROM semantic_evidence evidence
               JOIN scenes scene
                 ON scene.cloud_id=evidence.source_scene_cloud_id
               WHERE scene.knowledge_status='RETRACTED'"""
        ).fetchone()[0] == 0
        assert conn.execute(
            """SELECT COUNT(*) FROM semantic_backfill_state state
               JOIN scenes scene
                 ON scene.cloud_id=state.source_scene_cloud_id
               WHERE scene.knowledge_status='RETRACTED'"""
        ).fetchone()[0] == 0
        assert conn.execute(
            """SELECT COUNT(*) FROM scene_concept_projections projection
               JOIN scenes scene ON scene.cloud_id=projection.scene_id
               WHERE scene.knowledge_status='RETRACTED'"""
        ).fetchone()[0] == 0

    service = V2HiveService()
    hive_id = service.create()["hive"]["id"]
    result = service.query(hive_id, "Кто чинит модуль?")
    assert result["candidates"] == []


def test_response_plan_preserves_source_count_and_reverse_checks_axes():
    planner = ResponsePlanner()
    plan = planner.plan(
        interpretation_status="STABLE",
        query_frame={
            "requested_role": "object",
            "dialogue_acts": [{"id": "act-question", "act_type": "QUESTION"}],
        },
        answer={
            "status": "RESOLVED",
            "resolved_value": {"surface": "рыбу", "lemma": "рыба"},
            "surface_answer": "Рыбу.",
            "confidence": 0.91,
        },
        source_evidence=[{"source_scene_id": 10}],
    )

    assert plan.response_type == ResponseType.DIRECT
    assert planner.realize(plan) == "Рыбу."
    validation = planner.reverse_validate(
        plan,
        "Не должен продавать рыбу.",
        semantic_axes={
            "polarity": "NEGATIVE",
            "modality": "MUST",
            "actuality": "ACTUAL",
            "roles": {},
        },
    )
    assert validation["status"] == "PASSED"

    with V2Repository().transaction() as conn:
        persisted = planner.persist(
            conn,
            plan,
            conversation_id="dialogue-v25-response",
            source_utterance_id="utterance-question",
            surface="Рыбу.",
            independent_source_count=1,
        )
    assert persisted["status"] == "DERIVED_ANSWER"
    assert persisted["independent_source_count"] == 1


def test_batch_preview_commit_and_rollback_restore_confirmed_projection():
    pipeline = TrainingPipelineV2()
    preview = pipeline.preview_batch([
        {"text": "Кот ест рыбу.", "source_key": "batch-source-1"},
        {"text": "Собака видит кота.", "source_key": "batch-source-2"},
    ])

    assert preview["status"] == "PREVIEW"
    assert preview["metrics"]["staged"] == 2
    committed = pipeline.commit_batch(preview["batch_id"])
    assert committed["status"] == "COMMITTED"
    assert {
        item["decision"] for item in committed["decisions"]
    } == {"ADMIT"}

    rolled_back = pipeline.rollback_batch(preview["batch_id"])
    assert rolled_back["status"] == "ROLLED_BACK"
    assert len(rolled_back["retractions"]) == 2

    with V2Repository().transaction() as conn:
        assert conn.execute(
            "SELECT status FROM knowledge_batches WHERE id=?",
            (preview["batch_id"],),
        ).fetchone()["status"] == "ROLLED_BACK"
        assert conn.execute(
            """SELECT COUNT(*) FROM knowledge_staging
               WHERE id IN (?,?) AND status='RETRACTED'""",
            tuple(preview["staging_ids"]),
        ).fetchone()[0] == 2
        assert conn.execute(
            """SELECT COUNT(*) FROM scenes
               WHERE knowledge_status='RETRACTED'"""
        ).fetchone()[0] == 2
