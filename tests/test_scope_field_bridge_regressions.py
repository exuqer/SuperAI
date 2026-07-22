from __future__ import annotations

from server.v2.hybrid.bees import dispatch_bees
from server.v2.hybrid.contracts import BeeTask, QueryFrame, Gap, RetrievalHit
from server.v2.hybrid.pipeline import HybridDialoguePipeline
from server.v2.hybrid.workspace import build_workspace, spatial_support_identity
from server.v2.hybrid.contracts import ActivationResult


def _analysis_for_where_bowl() -> dict:
    language_analysis = {
        "utterance": {"raw_text": "Где стоит миска?"},
        "predicate": {
            "morphological_analyses": [
                {"lemma": "стоить", "pos": "VERB", "confidence": 0.333, "selected": True},
                {"lemma": "стоять", "pos": "VERB", "confidence": 0.333, "selected": False},
            ]
        },
        "clauses": [],
    }
    query_graph = {
        "query_graph_id": "query-where-bowl",
        "question_operators": [
            {
                "surface": "где",
                "question_lemma": "где",
                "operator_type": "EVENT_ATTACHMENT",
                "token_indices": [0],
            }
        ],
        "event_pattern": {
            "predicate": {"lemma": "стоить", "surface": "стоит", "token_index": 1},
            "known_nodes": [
                {
                    "node_id": "mention-bowl",
                    "entity_id": "entity-bowl",
                    "surface": "миска",
                    "head": {"lemma": "миска", "surface": "миска"},
                    "features": {},
                    "origin": "EXPLICIT_CURRENT",
                    "context_confidence": 1.0,
                }
            ],
            "target_gap": {
                "node_id": "gap-where",
                "gap_kind": "EVENT_PROPERTY",
                "surface": "где",
                "evidence": {
                    "learned_gap_profile": {
                        "mode": "SHADOW",
                        "profile_status": "UNSEEN",
                        "support_count": 0,
                    }
                },
            },
            "target_gaps": [
                {
                    "node_id": "gap-where",
                    "gap_kind": "EVENT_PROPERTY",
                    "surface": "где",
                    "evidence": {
                        "learned_gap_profile": {
                            "mode": "SHADOW",
                            "profile_status": "UNSEEN",
                            "support_count": 0,
                        }
                    },
                }
            ],
        },
        "trace": {"language_analysis": language_analysis},
    }
    return {"query_graph": query_graph, "language_analysis": language_analysis}


def _event_record() -> dict:
    return {
        "element_id": "event-bowl-near-fence",
        "element_type": "event",
        "event_id": "event-bowl-near-fence",
        "source_id": "source-training-1",
        "predicate_lemma": "стоять",
        "participants": [
            {
                "entity_id": "entity-fence",
                "surface": "забора",
                "head_surface": "забора",
                "head_lemma": "забор",
                "preposition": "у",
                "features": {"case": "gent"},
            },
            {
                "entity_id": "entity-bowl",
                "surface": "миска",
                "head_surface": "миска",
                "head_lemma": "миска",
                "preposition": "",
                "features": {"case": "nomn"},
            },
        ],
        "provenance": [
            {
                "source_id": "source-training-1",
                "source_type": "training",
                "independent_source_key": "training-1",
            }
        ],
        "retrieval_path": ["graph_index", "event-bowl-near-fence"],
    }


def test_structured_anchor_uses_concept_identity_and_anchor_section() -> None:
    frame = QueryFrame(
        query_id="query-anchor",
        session_id="session",
        raw_text="Где миска?",
        normalized_text="где миска",
        query_type="canonical_query",
        known_elements=(
            {
                "concept_id": "entity-bowl",
                "node_id": "mention-bowl",
                "lemma": "миска",
                "surface": "миска",
            },
        ),
        gaps=(Gap("gap", "query-anchor", expected_type="attachment"),),
    )
    workspace = build_workspace(frame, ActivationResult(activations={}))

    assert len(workspace.anchors) == 1
    assert workspace.anchors[0].element_id == "entity-bowl"
    assert not workspace.anchors[0].element_id.startswith("{")
    assert workspace.anchors[0].payload["reference"]["node_id"] == "mention-bowl"


def test_unseen_where_operator_stays_broad_and_ambiguous_predicate_resolves_by_graph() -> None:
    result = HybridDialoguePipeline(
        config={
            "resonance": {
                "min_iterations": 1,
                "stable_iterations_required": 1,
                "answer_threshold": 0.45,
                "leader_margin": 0.0,
            }
        }
    ).run(
        "Где стоит миска?",
        analysis=_analysis_for_where_bowl(),
        indexes={"records": [_event_record()]},
    )

    assert result["query_frame"]["explicit_predicate"] is None
    assert {item["predicate"] for item in result["query_frame"]["predicate_hypotheses"]} == {
        "стоить",
        "стоять",
    }
    assert result["query_frame"]["gaps"][0]["expected_type"] == "attachment"
    assert result["retrieval"]["graph_hit_count"] == 1
    assert "predicate_hypothesis:стоять" in result["retrieval"]["hits"][-1]["matched_features"]
    assert result["answer_structure"]["status"] == "STABLE"
    assert result["answer_structure"]["epistemic_mode"] == "OBSERVED"
    assert result["answer_text"].casefold() == "у забора"
    assert result["answer_structure"]["independent_source_count"] == 1


def test_field_to_graph_bridge_turns_cloud_navigation_into_graph_evidence() -> None:
    analysis = _analysis_for_where_bowl()
    field_hit = RetrievalHit(
        hit_id="field-hit-fence",
        element_id="cloud-fence",
        element_type="cloud",
        source_id="semantic_field",
        match_score=0.71,
        payload={
            "cloud_id": "cloud-fence",
            "concept_id": "entity-fence",
            "field_revision": 7,
        },
        provenance=({"source_type": "FIELD", "field_revision": 7},),
        retrieval_path=("field_projection", "cloud-fence"),
        origin="FIELD",
    )

    class FieldService:
        def project_query(self, frame):
            return {
                "anchor_clouds": [],
                "positive_gradients": [],
                "field_region": {
                    "region_id": f"region:{frame.query_id}",
                    "field_revision": 7,
                },
            }

        def neighbourhood(self, projection, limit=32):
            return [field_hit]

    result = HybridDialoguePipeline(
        field_service=FieldService(),
        config={
            "resonance": {
                "min_iterations": 1,
                "stable_iterations_required": 1,
                "answer_threshold": 0.45,
                "leader_margin": 0.0,
            }
        },
    ).run(
        "Где стоит миска?",
        analysis=analysis,
        indexes={
            "records": [],
            "field_bridge_records": [_event_record()],
        },
    )

    assert result["retrieval"]["direct_graph_hit_count"] == 0
    assert result["retrieval"]["bridge_graph_hit_count"] == 1
    assert result["retrieval"]["graph_hit_count"] == 1
    assert result["workspace"]["graph_evidence"]
    assert result["answer_structure"]["epistemic_mode"] == "OBSERVED"
    assert result["answer_text"].casefold() == "у забора"


def test_field_bee_reports_spatial_support_and_reuses_support_identity() -> None:
    support_id = spatial_support_identity("query-bee", "cloud-fence", 7, "region:query-bee")
    task = BeeTask(
        bee_task_id="bee-task-field",
        task_type="FIND_GAP_FILL",
        gap_id="gap",
        anchors=(),
        excluded_elements=(),
        max_steps=4,
        energy_budget=8,
        bee_mode="FIELD_BEE",
    )
    results = dispatch_bees(
        [task],
        {
            "field_records": [
                {
                    "element_id": "cloud-fence",
                    "cloud_id": "cloud-fence",
                    "concept_id": "entity-fence",
                    "activation": 0.71,
                    "field_revision": 7,
                    "region_id": "region:query-bee",
                    "spatial_support_id": support_id,
                }
            ]
        },
    )

    assert results[0].status == "SPATIAL_SUPPORT_FOUND"
    assert results[0].status != "GAP_FILLED"
    assert results[0].graph_evidence_ids == ()
    assert results[0].spatial_support_ids == (support_id,)
    assert results[0].utility["new_cloud"] is True


def test_field_bee_does_not_claim_success_for_zero_score_hit() -> None:
    task = BeeTask(
        bee_task_id="bee-task-empty",
        task_type="FIND_GAP_FILL",
        gap_id="gap",
        anchors=(),
        excluded_elements=(),
        max_steps=4,
        energy_budget=8,
        bee_mode="FIELD_BEE",
    )
    result = dispatch_bees(
        [task],
        {"field_records": [{"element_id": "cloud-empty", "activation": 0.0}]},
    )[0]

    assert result.status == "NO_RESULT"
    assert result.utility["useful"] is False
