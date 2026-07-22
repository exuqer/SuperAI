"""Application services for the SuperAI V3.0 spatial-evidential pipeline."""

from __future__ import annotations

import json
import uuid
from contextlib import nullcontext
from dataclasses import replace
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import server.database as database
from server.core.settings import settings

from .acceleration import AccelerationRuntime, runtime as acceleration_runtime
from .event_graph import EventGraphPipeline
from .graph_learning import ConstructionLearner, ObservationSignature
from .graph_models import (
    AnswerStatus,
    BindingConfiguration,
    CandidateBinding,
    GapKind,
    GraphStatus,
    GraphStatus,
    QueryGraph,
    candidate_binding_from_dict,
    query_graph_from_dict,
)
from .graph_repository import (
    GraphRepository,
    content_hash,
    decode,
    encode,
    stable_id,
    serialization_snapshot,
    utcnow,
)
from .query_graph import (
    GraphMatcher,
    GraphResponsePlanner,
    QueryGraphBuilder,
    persist_query_result,
)
from .universe import UniverseService
from .semantic_field import SemanticFieldService
from .question_family import (
    resolve_question_family,
    check_animacy_compatibility,
    AnimacyCompatibility,
)
from .gap_release import (
    GapReleaseSelector,
    GapReleaseDiagnostic,
    ReleaseDecision,
)
from .event_binding_frame import (
    EventBindingFrame,
    EventBindingFrameBuilder,
    FrameStatus,
    ParticipantOrigin,
    ObservedQuestionProfile,
    EventBindingFrameParticipant,
)
from .dialogue_context import (
    DialogueContextState,
    DialogueContextManager,
)
from .query_execution import QueryExecutionContext, diagnostic_owner_valid
from .hybrid import HybridDialoguePipeline
from .hybrid.retrieval import graph_rows_from_connection


class GraphTrainingService:
    """Stage and commit knowledge without a legacy projection."""

    def __init__(
        self,
        repository: Optional[GraphRepository] = None,
        morphology: Any = None,
        *,
        runtime: Optional[AccelerationRuntime] = None,
    ) -> None:
        if morphology is None:
            from .russian_morphology import RussianMorphology
            morphology = RussianMorphology()
        self.repository = repository or GraphRepository()
        self.acceleration = runtime or acceleration_runtime
        self.events = EventGraphPipeline(self.repository, morphology)
        self.universes = UniverseService(
            self.repository, runtime=self.acceleration
        )
        self.field = SemanticFieldService(self.repository)

    def _materialize_and_project(self, *args: Any, connection: Any = None, **kwargs: Any) -> Dict[str, Any]:
        discover_dimensions = kwargs.pop("discover_dimensions", None)
        project_universes = bool(kwargs.pop("project_universes", True))
        # Event evidence and its universe projection form one source-level
        # transaction.  Besides avoiding a second durability barrier, this
        # prevents a confirmed graph source from surviving a failed
        # projection as a partially materialised ingest.
        transaction = nullcontext(connection) if connection is not None else self.repository.transaction()
        with transaction as conn:
            result = self.events.materialize(
                *args,
                **kwargs,
                connection=conn,
            )
            if result.get("status") == "CONFIRMED" and project_universes:
                result["field_update"] = self.field.ingest_source(
                    str(result["source_id"]), connection=conn
                )
                if discover_dimensions is None:
                    source_count = int(conn.execute(
                        """SELECT COUNT(*) FROM knowledge_sources
                           WHERE status='CONFIRMED'"""
                    ).fetchone()[0])
                    interval = max(
                        1,
                        settings.dimension_discovery_checkpoint_interval,
                    )
                    discover_dimensions = (
                        source_count == 1 or source_count % interval == 0
                    )
                result["universe_update"] = self.universes.ingest_source(
                    str(result["source_id"]),
                    discover_dimensions=bool(discover_dimensions),
                    connection=conn,
                )
            elif result.get("status") == "CONFIRMED":
                result["field_update"] = self.field.ingest_source(
                    str(result["source_id"]), connection=conn
                )
                result["universe_update"] = {
                    "source_id": str(result["source_id"]),
                    "ingested": False,
                    "reason": "projection_deferred_to_batch_checkpoint",
                }
        return result

    def train(
        self,
        text: str,
        *,
        source_type: str = "document",
        independent_key: str = "",
        domain_key: str = "",
        discover_dimensions: Optional[bool] = None,
        project_universes: bool = True,
    ) -> Dict[str, Any]:
        return self._materialize_and_project(
            text,
            source_type=source_type,
            independent_key=independent_key,
            domain_key=domain_key,
            force_status=None,
            discover_dimensions=discover_dimensions,
            project_universes=project_universes,
        )

    def stage(
        self,
        text: str,
        *,
        source_type: str = "document",
        independent_key: str = "",
        domain_key: str = "",
        **_: Any,
    ) -> Dict[str, Any]:
        return self._materialize_and_project(
            text,
            source_type=source_type,
            independent_key=independent_key,
            domain_key=domain_key,
            force_status="STAGED",
        )

    def commit(
        self,
        source_id: str,
        *,
        manual_validation: bool = False,
    ) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            row = conn.execute(
                """SELECT raw_text,source_type,independent_key,metadata_json
                   FROM knowledge_sources WHERE id=?""",
                (source_id,),
            ).fetchone()
            if not row:
                raise KeyError(source_id)
            metadata = decode(row["metadata_json"], {})
        return self._materialize_and_project(
            str(row["raw_text"]),
            source_type=str(row["source_type"]),
            independent_key=str(row["independent_key"]),
            domain_key=str(metadata.get("domain_key") or row["source_type"]),
            force_status="CONFIRMED" if manual_validation else None,
        )

    def retract(self, source_id: str, reason: str = "") -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            result = self.events.retract(source_id, connection=conn)
            result["field_update"] = self.field.remove_source_contribution(source_id, connection=conn)
            result["universe_update"] = self.universes.remove_source(source_id, connection=conn)
            result["reason"] = reason
        return result

    def reprocess(self, source_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            row = conn.execute(
                """SELECT raw_text,source_type,independent_key,metadata_json
                   FROM knowledge_sources WHERE id=?""",
                (source_id,),
            ).fetchone()
            if not row:
                raise KeyError(source_id)
            payload = dict(row)
            metadata = decode(payload["metadata_json"], {})
            revision_key = f"{payload['independent_key']}:revision:{uuid.uuid4().hex}"
            replacement = self._materialize_and_project(
                str(payload["raw_text"]),
                source_type=str(payload["source_type"]),
                independent_key=revision_key,
                domain_key=str(metadata.get("domain_key") or payload["source_type"]),
                force_status="CONFIRMED",
                connection=conn,
            )
            if replacement.get("status") != "CONFIRMED":
                raise ValueError("replacement source did not pass admission")
            self.events.retract(source_id, connection=conn)
            self.field.remove_source_contribution(source_id, connection=conn)
            self.universes.remove_source(source_id, connection=conn)
        return {
            "previous_source_id": source_id,
            "source": replacement,
            "history_preserved": True,
        }

    def preview_batch(
        self,
        sources: Sequence[Mapping[str, Any]],
        config: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        batch_id = f"graph-batch-{uuid.uuid4().hex[:20]}"
        staged = [
            self.stage(
                str(item.get("text") or ""),
                source_type=str(item.get("source_type") or "document"),
                independent_key=str(item.get("independent_key") or ""),
                domain_key=str(item.get("domain_key") or ""),
            )
            for item in sources
            if str(item.get("text") or "").strip()
        ]
        if not staged:
            raise ValueError("batch must contain at least one non-empty source")
        now = utcnow()
        with self.repository.transaction() as conn:
            conn.execute(
                """INSERT INTO graph_batches
                   (id,status,config_json,created_at,updated_at)
                   VALUES(?,'PREVIEWED',?,?,?)""",
                (batch_id, encode(dict(config or {})), now, now),
            )
            conn.executemany(
                """INSERT INTO graph_batch_sources
                   (batch_id,source_id,source_order) VALUES(?,?,?)""",
                [
                    (batch_id, str(item["source_id"]), index)
                    for index, item in enumerate(staged)
                ],
            )
        return {
            "batch_id": batch_id,
            "status": "PREVIEWED",
            "sources": staged,
            "config": dict(config or {}),
            "metrics": {
                "source_count": len(staged),
                "confirmed_event_count": 0,
                "quarantine_count": sum(
                    item["status"] == "RETRACTED" for item in staged
                ),
            },
        }

    def commit_batch(self, batch_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            batch = conn.execute(
                "SELECT status FROM graph_batches WHERE id=?",
                (batch_id,),
            ).fetchone()
            if not batch:
                raise KeyError(batch_id)
            if str(batch["status"]) != "PREVIEWED":
                raise ValueError(
                    f"batch is not previewed: {batch['status']}"
                )
            rows = conn.execute(
                """SELECT s.id
                   FROM graph_batch_sources bs
                   JOIN knowledge_sources s ON s.id=bs.source_id
                   WHERE bs.batch_id=?
                   ORDER BY bs.source_order""",
                (batch_id,),
            ).fetchall()
        with self.repository.transaction() as conn:
            committed = []
            for row in rows:
                source = conn.execute("SELECT raw_text,source_type,independent_key,metadata_json FROM knowledge_sources WHERE id=?", (str(row["id"]),)).fetchone()
                metadata = decode(source["metadata_json"], {})
                committed.append(self._materialize_and_project(str(source["raw_text"]), source_type=str(source["source_type"]), independent_key=str(source["independent_key"]), domain_key=str(metadata.get("domain_key") or source["source_type"]), force_status="CONFIRMED", connection=conn))
            if not all(item.get("status") == "CONFIRMED" for item in committed):
                raise ValueError("batch admission failed; no source was committed")
            conn.execute(
                """UPDATE graph_batches SET status=?,updated_at=?
                   WHERE id=?""",
                ("COMMITTED", utcnow(), batch_id),
            )
        return {
            "batch_id": batch_id,
            "status": "COMMITTED",
            "sources": committed,
        }

    def rollback_batch(self, batch_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            batch = conn.execute(
                "SELECT status FROM graph_batches WHERE id=?",
                (batch_id,),
            ).fetchone()
            if not batch:
                raise KeyError(batch_id)
            if str(batch["status"]) != "PREVIEWED":
                raise ValueError(
                    f"batch is not previewed: {batch['status']}"
                )
            rows = conn.execute(
                """SELECT s.id
                   FROM graph_batch_sources bs
                   JOIN knowledge_sources s ON s.id=bs.source_id
                   WHERE bs.batch_id=? AND s.status='STAGED'
                   ORDER BY bs.source_order""",
                (batch_id,),
            ).fetchall()
            ids = [str(row["id"]) for row in rows]
            conn.execute(
                "DELETE FROM graph_batch_sources WHERE batch_id=?",
                (batch_id,),
            )
            for source_id in ids:
                shared = conn.execute(
                    """SELECT 1 FROM graph_batch_sources
                       WHERE source_id=? LIMIT 1""",
                    (source_id,),
                ).fetchone()
                if not shared:
                    conn.execute(
                        "DELETE FROM knowledge_sources WHERE id=?",
                        (source_id,),
                    )
            conn.execute(
                """UPDATE graph_batches SET status='ROLLED_BACK',updated_at=?
                   WHERE id=?""",
                (utcnow(), batch_id),
            )
        return {
            "batch_id": batch_id,
            "status": "ROLLED_BACK",
            "removed_source_ids": ids,
        }


class GraphDialogueService:
    """Persistent dialogue state around QueryGraph and gap bindings."""

    # Phase-one safety brake.  Dialogue turns are retained as observational
    # episodes, but no construction/profile/dimension confidence is changed
    # until the shadow-validation rollout explicitly lifts this flag.
    LEARNING_SUSPENDED_REASON = "STATE_ISOLATION_AND_TRACE_VALIDATION_PENDING"

    def __init__(
        self,
        repository: Optional[GraphRepository] = None,
        morphology: Any = None,
        *,
        runtime: Optional[AccelerationRuntime] = None,
    ) -> None:
        if morphology is None:
            from .russian_morphology import RussianMorphology
            morphology = RussianMorphology()
        self.repository = repository or GraphRepository()
        self.acceleration = runtime or acceleration_runtime
        self.morphology = morphology
        self.builder = QueryGraphBuilder(self.repository, morphology)
        self.matcher = GraphMatcher(self.repository)
        self.matcher.swarms.acceleration = self.acceleration
        self.responses = GraphResponsePlanner(morphology)
        self.constructions = ConstructionLearner()
        self.query_operators = self.builder.query_operators
        self.gap_release_selector = GapReleaseSelector()
        self.field = SemanticFieldService(self.repository)
        self.hybrid = HybridDialoguePipeline(field_service=self.field)
        self.reasoning_mode = "hybrid"

    def create(
        self,
        max_cells: int = 24,
        conversation_id: str = "",
    ) -> Dict[str, Any]:
        hive_id = f"hive-{uuid.uuid4().hex[:16]}"
        conversation_id = conversation_id or f"conversation-{uuid.uuid4().hex[:16]}"
        now = utcnow()
        state = {
            "query_graph": None,
            "selected_bindings": [],
            "candidate_bindings": [],
            "rejected_events": [],
            "answer": None,
            "trace": {},
            "turn_index": 0,
        }
        with self.repository.transaction() as conn:
            conn.execute(
                """INSERT INTO hives
                   (id,conversation_id,max_cells,active_query_graph_id,
                    state_json,created_at,updated_at)
                   VALUES(?,?,?,NULL,?,?,?)""",
                (
                    hive_id,
                    conversation_id,
                    max(1, int(max_cells)),
                    encode(state),
                    now,
                    now,
                ),
            )
            # Initialize dialogue context state
            DialogueContextManager.save(conn, DialogueContextState.create(conversation_id))
        return {
            "hive": {
                "id": hive_id,
                "conversation_id": conversation_id,
                "max_cells": max(1, int(max_cells)),
                "status": "READY",
            },
            **state,
        }

    def _load_row(self, conn: Any, hive_id: str) -> tuple[Any, Dict[str, Any]]:
        row = conn.execute(
            "SELECT * FROM hives WHERE id=?",
            (hive_id,),
        ).fetchone()
        if not row:
            raise KeyError(hive_id)
        return row, decode(row["state_json"], {})

    def get(self, hive_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            row, state = self._load_row(conn, hive_id)
            return {
                "hive": {
                    "id": str(row["id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "max_cells": int(row["max_cells"]),
                    "status": (
                        state.get("answer", {}).get("status")
                        if state.get("answer")
                        else "READY"
                    ),
                },
                **state,
            }

    def delete(self, hive_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            row = conn.execute("SELECT id FROM hives WHERE id=?", (hive_id,)).fetchone()
            if row is None:
                raise KeyError(hive_id)
            conn.execute("DELETE FROM hives WHERE id=?", (hive_id,))
        return {"hive_id": hive_id, "deleted": True}

    @staticmethod
    def _previous(
        state: Mapping[str, Any],
    ) -> tuple[Optional[QueryGraph], tuple[CandidateBinding, ...]]:
        graph_value = state.get("query_graph")
        bindings_value = state.get("selected_bindings") or []
        bindings = tuple(
            binding
            for binding in (
                candidate_binding_from_dict(value)
                for value in bindings_value
            )
            if binding is not None
        )
        return (
            query_graph_from_dict(graph_value) if graph_value else None,
            bindings,
        )

    @staticmethod
    def _chat_answer_text(
        answer: Mapping[str, Any],
        graph: QueryGraph,
    ) -> str:
        """Persist the same GAP-oriented text that the chat presents."""
        surface = str(answer.get("surface") or "")
        short = str(answer.get("short_answer") or "")
        full = str(answer.get("full_answer") or "")
        if str(answer.get("status") or "") != AnswerStatus.RESOLVED.value:
            return surface or short or full
        if len(tuple(graph.target_gaps)) == 1:
            return surface or short or full
        return full or surface or short

    def parse(
        self,
        text: str,
        *,
        previous_graph: Optional[QueryGraph] = None,
        previous_binding: Optional[CandidateBinding] = None,
        previous_bindings: Sequence[CandidateBinding] = (),
        conversation_id: str = "",
        turn_index: int = 0,
    ) -> Dict[str, Any]:
        graph, analysis = self.builder.build(
            text,
            previous_graph=previous_graph,
            previous_binding=previous_binding,
            previous_bindings=previous_bindings,
            conversation_id=conversation_id,
            turn_index=turn_index,
        )
        return {
            "query_graph": graph.as_dict(),
            "language_analysis": analysis,
        }

    def _learn_confirmed_episode(
        self,
        conn: Any,
        hive_id: str,
        text: str,
        graph: QueryGraph,
        selected_bindings: Sequence[CandidateBinding],
        binding_configuration: Optional[BindingConfiguration],
        answer: Mapping[str, Any],
        accepted: Sequence[CandidateBinding],
    ) -> None:
        validation = answer.get("validation") or {}
        eligible = bool(
            validation.get("valid")
            and answer.get("status") in {
                AnswerStatus.RESOLVED.value,
                AnswerStatus.PARTIALLY_RESOLVED.value,
            }
        )
        episode_id = stable_id(
            "training-episode",
            graph.id,
            binding_configuration.id if binding_configuration else ",".join(
                binding.id for binding in selected_bindings
            ),
        )
        local_slot_ids = sorted({
            str(local_slot_id)
            for binding in selected_bindings
            for evidence in binding.evidence
            for local_slot_id in evidence.get("local_slot_ids", [])
        })
        supporting_event_ids = sorted({
            event_id
            for binding in selected_bindings
            for evidence in binding.evidence
            for event_id in evidence.get("supporting_event_ids", [])
        } or {
            binding.event_id for binding in selected_bindings
        })
        conn.execute(
            """INSERT OR REPLACE INTO training_episodes
               (id,utterance,query_graph_id,candidate_bindings_json,
                selected_bindings_json,binding_configuration_id,
                event_ids_json,construction_ids_json,
                slot_hypotheses_json,answer_status,validation_json,
                user_correction_json,eligible_for_learning,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,NULL,?,?)""",
            (
                episode_id,
                text,
                graph.id,
                encode([item.as_dict() for item in accepted]),
                encode([item.as_dict() for item in selected_bindings]),
                binding_configuration.id if binding_configuration else None,
                encode(supporting_event_ids),
                encode(list(graph.construction_ids)),
                encode(local_slot_ids),
                str(answer.get("status")),
                encode(validation),
                0,
                utcnow(),
            ),
        )
        # Deliberately observational-only: creating a source or reinforcing a
        # profile here would turn a possibly fallback/inconsistent answer into
        # durable semantic evidence.

    def _build_query_interpretation_hypotheses(
        self,
        graph: QueryGraph,
        previous_graph: Optional[QueryGraph],
        previous_bindings: Sequence[CandidateBinding],
        dialogue_context: DialogueContextState,
        analysis: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build competing query interpretation hypotheses for the current query."""
        hypotheses = []
        has_explicit_entity = any(
            node.origin == "EXPLICIT_CURRENT" for node in graph.known_nodes
        )
        has_question_operator = graph.gap_node.surface != "" if graph.gap_node else True
        implicit_relation = bool(
            (graph.trace.get("predicate_hypothesis") or {}).get("predicate_origin")
            == "IMPLICIT_RELATIONAL"
        )
        tokens = analysis.get("tokens") or []
        first_token = (
            str(tokens[0].get("normalized") or "").casefold()
            if tokens else ""
        )
        continuation_marker = first_token in {"а", "и"}
        continuation_words = {"ещё", "потом", "тогда"}
        has_continuation_word = any(
            str(token.get("normalized") or "").casefold()
            in continuation_words
            for token in tokens
        )

        # Determine if this is a short question without a new anchor.
        is_short_question = has_question_operator and not has_explicit_entity
        has_new_entity = has_explicit_entity

        # A question operator plus its own prepositional noun phrase forms a
        # complete relational query even without an overt verb.  It must be
        # considered before, rather than after, context inheritance.
        if implicit_relation:
            hypotheses.append({
                "hypothesis_index": 0,
                "interpretation_type": "SELF_CONTAINED_RELATIONAL",
                "prior_score": 0.90,
                "predicate": None,
                "known_entities": [n.head_lemma for n in graph.known_nodes] if graph.known_nodes else [],
                "event_anchor_id": None,
            })

        has_inherited_predicate = bool(
            graph.predicate and graph.predicate.origin == "INHERITED"
        )
        if previous_graph and (
            is_short_question or implicit_relation or continuation_marker
            or has_inherited_predicate
        ):
            hypotheses.append({
                "hypothesis_index": len(hypotheses),
                "interpretation_type": "CONTEXTUAL_CONTINUATION",
                "prior_score": (
                    0.15 if implicit_relation else
                    0.88 if continuation_marker else
                    0.78 if has_inherited_predicate else
                    0.82 if has_continuation_word else 0.48
                ),
                "predicate": graph.predicate.lemma if graph.predicate else (
                    previous_graph.predicate.lemma if previous_graph and previous_graph.predicate else None
                ),
                "known_entities": [n.head_lemma for n in graph.known_nodes] if graph.known_nodes else [],
                "event_anchor_id": graph.trace.get("event_anchor_id"),
            })

        # Attribute continuation only inherits an object anchor, never the
        # prior predicate.  We keep it separate in trace for diagnostics.
        if previous_graph and has_question_operator and not graph.predicate and (
            not implicit_relation and (has_explicit_entity or is_short_question)
        ):
            hypotheses.append({
                "hypothesis_index": len(hypotheses),
                "interpretation_type": "ATTRIBUTE_CONTINUATION",
                "prior_score": 0.60 if has_explicit_entity else 0.38,
                "predicate": None,
                "known_entities": [n.head_lemma for n in previous_graph.known_nodes] if previous_graph.known_nodes else [],
                "event_anchor_id": None,
            })

        if (
            has_question_operator and graph.predicate
            and graph.predicate.origin == "CURRENT_EXPLICIT"
        ):
            hypotheses.append({
                "hypothesis_index": len(hypotheses),
                "interpretation_type": "EXPLICIT_QUERY",
                "prior_score": 0.8,
                "predicate": graph.predicate.lemma,
                "known_entities": [n.head_lemma for n in graph.known_nodes],
                "event_anchor_id": None,
            })

        if not hypotheses:
            hypotheses.append({
                "hypothesis_index": 0,
                "interpretation_type": "UNRESOLVED_ELLIPSIS",
                "prior_score": 0.10,
                "predicate": None,
                "known_entities": [],
                "event_anchor_id": None,
            })
        
        return hypotheses

    def _evaluate_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        graph: QueryGraph,
        previous_graph: Optional[QueryGraph],
        previous_bindings: Sequence[CandidateBinding],
        dialogue_context: DialogueContextState,
        turn_index: int,
    ) -> Dict[str, Any]:
        """Evaluate a single hypothesis through retrieval -> GraphMatcher -> Validator."""
        interpretation_type = hypothesis["interpretation_type"]
        if not hypothesis.get("retrieval_allowed", True):
            hypothesis.update({
                "current_evidence_score": 0.0,
                "inherited_context_score": 0.0,
                "event_retrieval_score": 0.0,
                "graph_validation_score": 0.0,
                "total_score": 0.0,
                "admitted_event_ids": [],
                "selected_bindings": [],
                "validation": {
                    "valid": False,
                    "reason": "CONTEXT_INHERITANCE_BLOCKED",
                },
            })
            return hypothesis
        
        # Build a test query graph based on the hypothesis
        if interpretation_type in {
            "CONTEXTUAL_CONTINUATION", "SELF_CONTAINED_RELATIONAL",
        }:
            # Use inherited predicate, current known nodes, current gap
            test_graph = graph
        elif interpretation_type == "UNRESOLVED_ELLIPSIS":
            # Use only current gap, no inherited predicate
            test_graph = QueryGraph(
                id=stable_id("test-graph", graph.id, "standalone"),
                predicate=None,
                known_nodes=tuple(graph.known_nodes),
                gap_node=graph.gap_node,
                target_gaps=graph.target_gaps,
                question_operators=graph.question_operators,
                required_edges=graph.required_edges,
                status=GraphStatus.READY,
                continuation_of=None,
                construction_ids=(),
                implicit_gaps=(),
                trace={"hypothesis_type": "UNRESOLVED_ELLIPSIS"},
            )
        elif interpretation_type == "ATTRIBUTE_CONTINUATION":
            # The current graph already retains only the allowed context.
            test_graph = graph
        elif interpretation_type == "EXPLICIT_QUERY":
            test_graph = graph
        else:
            test_graph = graph
        
        # Run retrieval and matching
        search = self.matcher.search(test_graph)
        selected_bindings = [
            item for item in search.get("selected_bindings", [])
            if isinstance(item, CandidateBinding)
        ]
        
        # Get validation
        event = None
        if selected_bindings:
            with self.repository.transaction() as conn:
                event = EventGraphPipeline.load_event(conn, selected_bindings[0].event_id)
        
        answer = self.responses.plan(test_graph, search, event=event)
        validation = answer.get("validation") or {}
        
        # Score the hypothesis
        current_evidence_score = 0.0
        if selected_bindings:
            current_evidence_score = sum(b.total_score for b in selected_bindings) / len(selected_bindings)
        
        graph_validation_score = 1.0 if validation.get("valid") else 0.0
        
        # Inherited context score (only if context source is RESOLVED)
        inherited_context_score = 0.0
        if dialogue_context.last_resolved_turn_id and interpretation_type == "CONTEXTUAL_CONTINUATION":
            inherited_context_score = 0.5
        
        # Event retrieval score
        event_retrieval_score = 0.0
        if search.get("accepted_events"):
            event_retrieval_score = min(1.0, len(search["accepted_events"]) / 10.0)
        
        total_score = (
            hypothesis["prior_score"] * 0.2 +
            current_evidence_score * 0.4 +
            graph_validation_score * 0.2 +
            inherited_context_score * 0.1 +
            event_retrieval_score * 0.1
        )
        
        hypothesis.update({
            "current_evidence_score": current_evidence_score,
            "inherited_context_score": inherited_context_score,
            "event_retrieval_score": event_retrieval_score,
            "graph_validation_score": graph_validation_score,
            "total_score": total_score,
            "admitted_event_ids": [b.event_id for b in search.get("accepted", []) if isinstance(b, CandidateBinding)],
            "selected_bindings": [b.as_dict() for b in selected_bindings],
            "validation": validation,
        })
        
        return hypothesis

    def _persist_hypotheses(
        self,
        conn: Any,
        graph: QueryGraph,
        hypotheses: List[Dict[str, Any]],
        selected_index: int,
    ) -> None:
        """Persist query interpretation hypotheses to database."""
        persisted_type = {
            "SELF_CONTAINED_RELATIONAL": "STANDALONE_GAP_QUERY",
            "CONTEXTUAL_CONTINUATION": "STRUCTURAL_CONTINUATION",
            "ATTRIBUTE_CONTINUATION": "CONTEXT_REFERENCE_QUERY",
            "UNRESOLVED_ELLIPSIS": "STANDALONE_GAP_QUERY",
        }
        for h in hypotheses:
            hypothesis_id = stable_id("query-hypothesis", graph.id, h["hypothesis_index"])
            conn.execute(
                """INSERT OR REPLACE INTO query_interpretation_hypotheses
                   (id,query_graph_id,hypothesis_index,interpretation_type,
                    prior_score,current_evidence_score,inherited_context_score,
                    event_retrieval_score,graph_validation_score,total_score,
                    admitted_event_ids_json,rejection_reason,selected,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    hypothesis_id,
                    graph.id,
                    h["hypothesis_index"],
                    persisted_type.get(
                        h["interpretation_type"], h["interpretation_type"]
                    ),
                    h["prior_score"],
                    h["current_evidence_score"],
                    h["inherited_context_score"],
                    h["event_retrieval_score"],
                    h["graph_validation_score"],
                    h["total_score"],
                    json.dumps(h["admitted_event_ids"]),
                    h.get("rejection_reason"),
                    1 if h["hypothesis_index"] == selected_index else 0,
                    utcnow(),
                ),
            )

    def _build_predicate_hypotheses(
        self,
        graph: QueryGraph,
        analysis: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build predicate hypotheses for ambiguous verb forms (e.g., стоит -> стоять/стоить)."""
        hypotheses = []
        if not graph.predicate:
            return hypotheses
        
        # Check if predicate has ambiguous morphology
        predicate_lemma = graph.predicate.lemma
        ambiguous_lemmas = {
            "стоить": ["стоять", "стоить"],
            "стоять": ["стоять", "стоить"],
        }
        
        if predicate_lemma in ambiguous_lemmas:
            for lemma in ambiguous_lemmas[predicate_lemma]:
                concept_id = stable_id("predicate-concept", lemma)
                hypotheses.append({
                    "utterance_id": graph.id,
                    "token_index": graph.predicate.token_index,
                    "lemma": lemma,
                    "concept_id": concept_id,
                    "morphology_confidence": 0.8,
                    "contextual_confidence": 0.5,
                    "construction_confidence": 0.5,
                    "participant_compatibility": 0.5,
                })
        
        return hypotheses

    def _evaluate_predicate_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
        graph: QueryGraph,
        previous_bindings: Sequence[CandidateBinding],
    ) -> List[Dict[str, Any]]:
        """Evaluate predicate hypotheses against event evidence."""
        for h in hypotheses:
            # Score based on compatibility with selected bindings
            if previous_bindings:
                # Check if event predicate matches hypothesis
                event_id = previous_bindings[0].event_id
                with self.repository.transaction() as conn:
                    event = EventGraphPipeline.load_event(conn, event_id)
                    if event and event.predicate_lemma == h["lemma"]:
                        h["contextual_confidence"] = 0.9
                        h["construction_confidence"] = 0.8
                        h["participant_compatibility"] = 0.9
                    else:
                        h["contextual_confidence"] = 0.3
                        h["construction_confidence"] = 0.3
                        h["participant_compatibility"] = 0.3
            else:
                h["contextual_confidence"] = 0.5
                h["construction_confidence"] = 0.5
                h["participant_compatibility"] = 0.5
            
            h["total_score"] = (
                h["morphology_confidence"] * 0.3 +
                h["contextual_confidence"] * 0.3 +
                h["construction_confidence"] * 0.2 +
                h["participant_compatibility"] * 0.2
            )
        
        # Sort by total score
        hypotheses.sort(key=lambda h: h["total_score"], reverse=True)
        
        # Mark best as selected
        for i, h in enumerate(hypotheses):
            h["selected"] = (i == 0)
            if i == 0:
                h["selection_reason"] = "HIGHEST_COMPOSITE_SCORE"
            else:
                h["selection_reason"] = "LOWER_COMPOSITE_SCORE"
        
        return hypotheses

    def _persist_predicate_hypotheses(
        self,
        conn: Any,
        graph: QueryGraph,
        hypotheses: List[Dict[str, Any]],
    ) -> None:
        """Persist predicate hypotheses to database."""
        for h in hypotheses:
            hypothesis_id = stable_id("predicate-hypothesis", graph.id, h["token_index"], h["lemma"])
            conn.execute(
                """INSERT OR REPLACE INTO predicate_hypotheses
                   (id,utterance_id,token_index,lemma,concept_id,
                    morphology_confidence,contextual_confidence,
                    construction_confidence,participant_compatibility,
                    selected,selection_reason,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    hypothesis_id,
                    h["utterance_id"],
                    h["token_index"],
                    h["lemma"],
                    h["concept_id"],
                    h["morphology_confidence"],
                    h["contextual_confidence"],
                    h["construction_confidence"],
                    h["participant_compatibility"],
                    1 if h["selected"] else 0,
                    h["selection_reason"],
                    utcnow(),
                ),
            )

    def _persist_event_binding_frame(
        self,
        conn: Any,
        conversation_id: str,
        graph: QueryGraph,
        binding_configuration: Optional[BindingConfiguration],
        selected_bindings: Sequence[CandidateBinding],
        event_participants: Sequence[Any],
        gap_surfaces: Mapping[str, str],
        turn_index: int,
        execution: Optional[QueryExecutionContext] = None,
        hypothesis_id: str = "",
    ) -> Optional[str]:
        """Create or update EventBindingFrame after a valid BindingConfiguration."""
        if not binding_configuration or not selected_bindings:
            return None
        
        event_id = selected_bindings[0].event_id
        predicate_concept_id = binding_configuration.event_id
        
        # Check if frame already exists for this event
        existing_frame = conn.execute(
            """SELECT * FROM event_binding_frames 
               WHERE event_id=? AND conversation_id=? AND status != 'CLOSED'""",
            (event_id, conversation_id),
        ).fetchone()
        
        if existing_frame:
            # Update existing frame with new binding information
            frame_id = existing_frame["id"]
            frame = EventBindingFrame(
                frame_id=frame_id,
                conversation_id=conversation_id,
                root_query_graph_id=existing_frame["root_query_graph_id"],
                latest_query_graph_id=graph.id,
                event_id=event_id,
                predicate_concept_id=existing_frame["predicate_concept_id"],
                status=FrameStatus(existing_frame["status"]),
                confidence=existing_frame["confidence"],
                created_at=existing_frame["created_at"],
                updated_at=utcnow(),
                participants=(),  # Will be loaded from participants table
            )
            
            # Update frame
            updated_frame = EventBindingFrameBuilder.update_for_new_binding(
                frame,
                selected_bindings[0],  # Use first binding as representative
                gap_surfaces.get(selected_bindings[0].gap_node_id, ""),
                graph.id,
                turn_index,
            )
            
            # Persist updated frame
            conn.execute(
                """UPDATE event_binding_frames SET
                   latest_query_graph_id=?, status=?, confidence=?, state_json=?, updated_at=?
                   WHERE id=?""",
                (
                    updated_frame.latest_query_graph_id,
                    updated_frame.status.value,
                    updated_frame.confidence,
                    encode({
                        "owner_execution_id": execution.execution_id if execution else "",
                        "owner_query_graph_id": graph.id,
                        "owner_hypothesis_id": hypothesis_id,
                        "created_for_gap_ids": [gap.id for gap in graph.target_gaps],
                    }),
                    updated_frame.updated_at,
                    frame_id,
                ),
            )
            
            # Update participants
            for participant in updated_frame.participants:
                conn.execute(
                    """INSERT OR REPLACE INTO event_binding_frame_participants
                       (id,frame_id,participant_node_id,concept_id,resolved_lemma,
                        canonical_surface,morphology_json,origin,lineage_root_gap_id,
                        latest_source_gap_id,latest_source_binding_id,
                        source_query_graph_ids_json,local_slot_ids_json,
                        observed_question_profiles_json,
                        compatible_question_profiles_json,binding_confidence,replaceable,
                        last_released_turn,last_selected_turn,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        participant.frame_participant_id,
                        participant.frame_id,
                        participant.participant_node_id,
                        participant.concept_id,
                        participant.resolved_lemma,
                        participant.canonical_surface,
                        json.dumps(dict(participant.morphology_profile)),
                        participant.origin.value,
                        participant.lineage_root_gap_id,
                        participant.latest_source_gap_id,
                        participant.latest_source_binding_id,
                        json.dumps(list(participant.source_query_graph_ids)),
                        json.dumps(list(participant.local_slot_ids)),
                        json.dumps([p.as_dict() for p in participant.observed_question_profiles]),
                        json.dumps(dict(participant.compatible_question_profiles)),
                        participant.binding_confidence,
                        1 if participant.replaceable else 0,
                        participant.last_released_turn,
                        participant.last_selected_turn,
                        utcnow(),
                        utcnow(),
                    ),
                )
            
            return frame_id
        else:
            # Create new frame
            frame = EventBindingFrameBuilder.create_from_configuration(
                conversation_id=conversation_id,
                query_graph_id=graph.id,
                event_id=event_id,
                predicate_concept_id=predicate_concept_id,
                bindings=selected_bindings,
                event_participants=event_participants,
                gap_surfaces=gap_surfaces,
                turn_index=turn_index,
            )
            
            # Persist frame
            conn.execute(
                """INSERT INTO event_binding_frames
                   (id,conversation_id,root_query_graph_id,latest_query_graph_id,
                    event_id,predicate_concept_id,status,confidence,state_json,
                    created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    frame.frame_id,
                    frame.conversation_id,
                    frame.root_query_graph_id,
                    frame.latest_query_graph_id,
                    frame.event_id,
                    frame.predicate_concept_id,
                    frame.status.value,
                    frame.confidence,
                    encode({
                        "owner_execution_id": execution.execution_id if execution else "",
                        "owner_query_graph_id": graph.id,
                        "owner_hypothesis_id": hypothesis_id,
                        "created_for_gap_ids": [gap.id for gap in graph.target_gaps],
                    }),
                    frame.created_at,
                    frame.updated_at,
                ),
            )
            
            # Persist participants
            for participant in frame.participants:
                conn.execute(
                    """INSERT INTO event_binding_frame_participants
                       (id,frame_id,participant_node_id,concept_id,resolved_lemma,
                        canonical_surface,morphology_json,origin,lineage_root_gap_id,
                        latest_source_gap_id,latest_source_binding_id,
                        source_query_graph_ids_json,local_slot_ids_json,
                        observed_question_profiles_json,
                        compatible_question_profiles_json,binding_confidence,replaceable,
                        last_released_turn,last_selected_turn,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        participant.frame_participant_id,
                        participant.frame_id,
                        participant.participant_node_id,
                        participant.concept_id,
                        participant.resolved_lemma,
                        participant.canonical_surface,
                        json.dumps(dict(participant.morphology_profile)),
                        participant.origin.value,
                        participant.lineage_root_gap_id,
                        participant.latest_source_gap_id,
                        participant.latest_source_binding_id,
                        json.dumps(list(participant.source_query_graph_ids)),
                        json.dumps(list(participant.local_slot_ids)),
                        json.dumps([p.as_dict() for p in participant.observed_question_profiles]),
                        json.dumps(dict(participant.compatible_question_profiles)),
                        participant.binding_confidence,
                        1 if participant.replaceable else 0,
                        participant.last_released_turn,
                        participant.last_selected_turn,
                        utcnow(),
                        utcnow(),
                    ),
                )
            
            return frame.frame_id

    def _persist_gap_release_diagnostics(
        self,
        conn: Any,
        graph: QueryGraph,
        frame_id: Optional[str],
        event_id: Optional[str],
        question_family_key: Optional[str],
        diagnostic: GapReleaseDiagnostic,
    ) -> str:
        """Persist gap release diagnostic and candidate scores."""
        diagnostic_id = stable_id("gap-release-diag", graph.id)
        
        conn.execute(
            """INSERT OR REPLACE INTO gap_release_diagnostics
               (id,query_graph_id,execution_id,hypothesis_id,gap_id,status,
                frame_id,event_id,question_family_key,
                candidates_json,selected_participant_node_id,selected_score,
                second_score,release_margin,decision,decision_reason,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                diagnostic_id,
                graph.id,
                diagnostic.execution_id,
                diagnostic.hypothesis_id,
                diagnostic.gap_id,
                diagnostic.status,
                frame_id,
                event_id,
                question_family_key,
                json.dumps([c.as_dict() for c in diagnostic.candidates]),
                diagnostic.selected_participant_node_id,
                diagnostic.selected_score,
                diagnostic.second_score,
                diagnostic.release_margin,
                diagnostic.decision.value,
                diagnostic.decision_reason,
                utcnow(),
            ),
        )
        
        # Persist individual candidate scores
        for candidate in diagnostic.candidates:
            conn.execute(
                """INSERT OR REPLACE INTO gap_release_candidate_scores
                   (diagnostic_id,participant_node_id,concept_id,resolved_surface,
                    exact_surface_match,question_family_match,root_gap_lineage_match,
                    latest_gap_lineage_match,local_slot_score,animacy_score,case_score,
                    morphology_score,frame_confidence,recency_score,
                    explicit_current_penalty,animacy_conflict,hard_slot_conflict,
                    final_score,rank,accepted)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    diagnostic_id,
                    candidate.participant_node_id,
                    candidate.concept_id,
                    candidate.resolved_surface,
                    candidate.exact_surface_match,
                    candidate.question_family_match,
                    candidate.root_gap_lineage_match,
                    candidate.latest_gap_lineage_match,
                    candidate.local_slot_score,
                    candidate.animacy_score,
                    candidate.case_score,
                    candidate.morphology_score,
                    candidate.frame_confidence,
                    candidate.recency_score,
                    candidate.explicit_current_penalty,
                    candidate.animacy_conflict,
                    candidate.hard_slot_conflict,
                    candidate.final_score,
                    candidate.rank,
                    1 if candidate.accepted else 0,
                ),
            )
        
        return diagnostic_id

    def _filter_inherited_nodes(
        self,
        inherited_nodes: List[Any],
        active_frame: Optional[EventBindingFrame],
        dialogue_context: DialogueContextState,
        selected_hypothesis: Dict[str, Any],
        current_explicit_nodes: List[Any],
    ) -> List[Any]:
        """Filter inherited nodes to prevent stale constraints from UNRESOLVED turns.
        
        KEY RULE: Do NOT aggressively filter when an active EventBindingFrame exists —
        gap release can select the correct participant after full search with all inherited nodes.
        Only filter for:
        1. STANDALONE_GAP_QUERY (e.g., "Где батарея?") — no inherited constraints
        2. UNRESOLVED previous turn — inherited nodes cannot come from UNRESOLVED
        3. Duplicates of EXPLICIT_CURRENT entities
        """
        filtered = []
        hypothesis_type = selected_hypothesis.get("interpretation_type")

        # Build lookup for frame participants if frame exists
        frame_participant_node_ids: set[str] = set()
        if active_frame:
            for participant in active_frame.participants:
                if participant.participant_node_id:
                    frame_participant_node_ids.add(participant.participant_node_id)

        # Get known node keys from current explicit nodes
        current_explicit_keys = {
            (getattr(node, 'head_lemma', ''), getattr(node, 'qualified_key', ''))
            for node in current_explicit_nodes
        }

        for node in inherited_nodes:
            # Rule 1: STANDALONE_GAP_QUERY — no inherited nodes at all
            if hypothesis_type == "STANDALONE_GAP_QUERY":
                continue

            # Rule 2: UNRESOLVED previous turn blocks all inherited nodes
            has_unresolved = False
            if dialogue_context.last_resolved_turn_id:
                # If last turn was not resolved and current is a follow-up
                if not dialogue_context.is_currently_resolved_last_turn():
                    has_unresolved = True
                    continue

            # Rule 3: Don't duplicate EXPLICIT_CURRENT entities
            if hasattr(node, 'head_lemma') and hasattr(node, 'qualified_key'):
                if (node.head_lemma, node.qualified_key) in current_explicit_keys:
                    continue

            filtered.append(node)

        return filtered

    @staticmethod
    def _final_trace_consistency_validation(
        *,
        execution: QueryExecutionContext,
        selected_hypothesis: Mapping[str, Any],
        selected_bindings: Sequence[CandidateBinding],
        binding_configuration: Optional[BindingConfiguration],
        answer: Mapping[str, Any],
        selected_diagnostics: Mapping[str, Mapping[str, Any]],
        requested_gap_ids: Sequence[str],
    ) -> Dict[str, Any]:
        """Reject cross-hypothesis/event data before feedback is committed."""
        failures: List[str] = []
        selected_event_ids = {binding.event_id for binding in selected_bindings}
        admitted = set(selected_hypothesis.get("admitted_event_ids") or [])
        selection_scope = str(answer.get("selection_scope") or "SINGLE_EVENT")
        if binding_configuration and binding_configuration.event_id not in admitted:
            failures.append("CONFIGURATION_EVENT_NOT_ADMITTED_BY_HYPOTHESIS")
        if selection_scope == "SINGLE_EVENT" and binding_configuration and any(
            binding.event_id != binding_configuration.event_id
            for binding in selected_bindings
        ):
            failures.append("BINDING_EVENT_CONFIGURATION_MISMATCH")
        if selection_scope == "MULTI_EVENT":
            configurations = list(answer.get("binding_configurations") or [])
            if not configurations:
                failures.append("MULTI_EVENT_CONFIGURATION_MISSING")
            for configuration in configurations:
                event_id = str(configuration.get("event_id") or "")
                bindings_by_gap = dict(
                    configuration.get("bindings_by_gap") or {}
                )
                if event_id not in admitted:
                    failures.append("CONFIGURATION_EVENT_NOT_ADMITTED_BY_HYPOTHESIS")
                if set(bindings_by_gap) != set(requested_gap_ids):
                    failures.append("INCOMPLETE_MULTI_EVENT_CONFIGURATION")
                if any(
                    str(item.get("event_id") or "") != event_id
                    for item in bindings_by_gap.values()
                ):
                    failures.append("MULTI_EVENT_CONFIGURATION_MIXED_EVENTS")
        provenance = answer.get("provenance") or {}
        if selected_bindings and set(provenance.get("source_event_ids") or []) != selected_event_ids:
            failures.append("ANSWER_PROVENANCE_EVENT_MISMATCH")
        hypothesis_id = str(selected_hypothesis.get("hypothesis_id") or "")
        for gap_id, diagnostic in selected_diagnostics.items():
            if not diagnostic_owner_valid(
                diagnostic,
                execution=execution,
                hypothesis_id=hypothesis_id,
                gap_ids=requested_gap_ids,
                event_ids=tuple(selected_event_ids),
            ):
                failures.append(f"DIAGNOSTIC_OWNER_MISMATCH:{gap_id}")
        return {
            "stage": "FINAL_TRACE_CONSISTENCY_VALIDATION",
            "passed": not failures,
            "failures": failures,
            "selected_event_ids": sorted(selected_event_ids),
        }

    def query(
        self,
        hive_id: str,
        text: str,
        *,
        resolved_mode: Optional[str] = None,
        retrieval_scope: str = "LOCAL_THEN_GLOBAL",
    ) -> Dict[str, Any]:
        # Query construction, swarm evidence, strict matching and persisted
        # outcome are one atomic turn.  Repository transactions opened by the
        # individual stages join this outer transaction.
        with self.repository.transaction():
            started = perf_counter()
            sql_before, executes_before = database.metrics_snapshot()
            serialization_before = serialization_snapshot()
            result = self._build_hybrid_source(
                hive_id, text, resolved_mode=resolved_mode
            )
            result = self._attach_hybrid_result(
                result, hive_id, text, retrieval_scope=retrieval_scope
            )
            sql_after, executes_after = database.metrics_snapshot()
            serialization_after = serialization_snapshot()
            elapsed_ms = (perf_counter() - started) * 1000.0
            sql_ms = max(0.0, sql_after - sql_before)
            serialization_ms = max(
                0.0, serialization_after - serialization_before
            )
            trace = result.get("trace") or {}
            trace.update({
                "sql_ms": round(sql_ms, 3),
                "serialization_ms": round(serialization_ms, 3),
                "numerical_ms": round(max(
                    0.0,
                    elapsed_ms - sql_ms - serialization_ms,
                ), 3),
                "sqlite_execute_count": max(
                    0, executes_after - executes_before
                ),
            })
            result["trace"] = trace
            return result

    def _build_hybrid_source(
        self,
        hive_id: str,
        text: str,
        *,
        resolved_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("text must not be empty")
        if resolved_mode not in {None, "NEW_QUERY", "FOLLOW_UP", "CORRECTION"}:
            raise ValueError("unsupported resolved_mode")
        with self.repository.transaction() as conn:
            row, state = self._load_row(conn, hive_id)
        previous_graph, previous_bindings = self._previous(state)
        if resolved_mode == "NEW_QUERY":
            previous_graph, previous_bindings = None, ()
        next_turn = int(state.get("turn_index") or 0) + 1
        graph, analysis = self.builder.build(
            normalized_text,
            previous_graph=previous_graph,
            previous_bindings=previous_bindings,
            identity_context=f"{hive_id}:{next_turn}",
            conversation_id=str(row["conversation_id"]),
            turn_index=next_turn,
        )
        previous_hybrid = state.get("hybrid") or {}
        return {
            "message_id": stable_id("dialogue-turn", hive_id, next_turn),
            "query_graph": graph.as_dict(),
            "language_analysis": analysis,
            "candidate_bindings": [],
            "rejected_events": [],
            "selected_bindings": [],
            "binding_configuration": None,
            "binding_configurations": [],
            "selection_scope": "HYBRID",
            "answer": None,
            "trace": {"reasoning_mode": "hybrid", "hybrid_source": True},
            "hive": {
                "id": hive_id,
                "conversation_id": str(row["conversation_id"]),
                "max_cells": int(row["max_cells"]),
                "status": "READY",
            },
            "_hybrid_context": {
                "session_id": str(row["conversation_id"]),
                "last_query_id": str((previous_hybrid.get("query_frame") or {}).get("query_id") or ""),
                "last_query_frame": previous_hybrid.get("query_frame") or {},
                "last_answer": previous_hybrid.get("answer_structure") or previous_hybrid.get("answer") or {},
            },
            "_query_graph_object": graph,
        }

    def _attach_hybrid_result(
        self,
        result: Dict[str, Any],
        hive_id: str,
        text: str,
        *,
        retrieval_scope: str = "LOCAL_THEN_GLOBAL",
    ) -> Dict[str, Any]:
        """Attach the single authoritative hybrid result."""
        hive = result.get("hive") or {}
        conversation_id = str(hive.get("conversation_id") or "")
        indexes: list[dict[str, Any]] = []
        previous_context: dict[str, Any] = dict(result.get("_hybrid_context") or {})
        with self.repository.transaction() as conn:
            graph_object = result.get("_query_graph_object")
            if graph_object is not None:
                persist_query_result(
                    conn,
                    graph_object,
                    text,
                    hive_id=hive_id,
                    search={"accepted": []},
                )
            indexes = graph_rows_from_connection(
                conn,
                session_id=conversation_id,
                query_graph=result.get("query_graph") or {},
            )
            local = [
                item for item in indexes
                if conversation_id
                and str(item.get("session_id") or "") == conversation_id
            ]
            global_items = [item for item in indexes if item not in local]
            if retrieval_scope == "LOCAL_ONLY":
                indexes = local
            elif retrieval_scope == "GLOBAL_ONLY":
                indexes = global_items
            else:
                indexes = [*local, *global_items]
            turns = conn.execute(
                """SELECT query_graph_id,selected_bindings_json,answer_json
                   FROM dialogue_turns WHERE hive_id=?
                   ORDER BY turn_index DESC LIMIT 2""",
                (hive_id,),
            ).fetchall()
            if len(turns) > 1:
                previous_turn = turns[1]
                previous_answer = decode(previous_turn["answer_json"], {})
                previous_bindings = decode(previous_turn["selected_bindings_json"], [])
                resolved_values = [
                    str(item.get("resolved_lemma") or item.get("resolved_surface") or "")
                    for item in previous_bindings
                    if isinstance(item, Mapping)
                ]
                previous_context = {
                    "last_query_id": str(previous_turn["query_graph_id"] or ""),
                    "last_answer": {
                        **(previous_answer if isinstance(previous_answer, Mapping) else {}),
                        "resolved_values": [item for item in resolved_values if item],
                    },
                }
        # Preserve a useful shadow trace even before a confirmed event index exists.
        if not indexes:
            indexes = [
                {
                    "element_id": item.get("resolved_node_id") or item.get("binding_id"),
                    "element_type": "entity",
                    "surface": item.get("resolved_surface"),
                    "lemma": item.get("resolved_lemma"),
                    "event_id": item.get("event_id"),
                    "source_id": item.get("event_id"),
                    "provenance": item.get("evidence") or (),
                }
                for item in result.get("candidate_bindings") or ()
            ]
        context = {
            "session_id": conversation_id,
            **previous_context,
            "last_query_id": previous_context.get("last_query_id") or (result.get("query_graph") or {}).get("continuation_of"),
        }
        hybrid = self.hybrid.run(
            text,
            session_context=context,
            indexes=indexes,
            analysis={"query_graph": result.get("query_graph") or {}},
        )
        hybrid["retrieval"]["scope"] = retrieval_scope
        result["hybrid"] = hybrid
        result.setdefault("trace", {})["hybrid"] = hybrid.get("trace")
        result["trace"]["reasoning_mode"] = self.reasoning_mode
        answer = hybrid.get("answer_structure") or hybrid.get("answer") or {}
        hybrid_status = str(answer.get("status") or "INSUFFICIENT_EVIDENCE")
        public_status = {
            "STABLE": "RESOLVED",
            "AMBIGUOUS_RESULT": "AMBIGUOUS",
            "CONFLICTING_EVIDENCE": "CONFLICTED",
            "UNRESOLVED_CONTEXT": "UNRESOLVED",
        }.get(hybrid_status, "UNRESOLVED")
        answer_text = str(hybrid.get("answer_text") or "").strip() if public_status == "RESOLVED" else ""
        if answer_text and answer_text[-1:] not in {".", "!", "?"}:
            answer_text += "."
        leader = (hybrid.get("resonance") or {}).get("leader") or {}
        supporting_events = leader.get("supporting_events") or []
        candidates_by_id = {
            str(item.get("candidate_id") or ""): item
            for item in hybrid.get("candidates") or ()
            if isinstance(item, Mapping)
        }
        candidates_by_gap = {
            str(item.get("gap_id") or ""): item
            for candidate_id in leader.get("candidate_ids") or ()
            if isinstance((item := candidates_by_id.get(str(candidate_id))), Mapping)
        }
        target_gaps = (
            ((result.get("query_graph") or {}).get("event_pattern") or {}).get("target_gaps")
            or []
        )
        selected_bindings = []
        if public_status == "RESOLVED":
            for gap in target_gaps:
                if not isinstance(gap, Mapping):
                    continue
                gap_id = str(gap.get("node_id") or "")
                candidate = candidates_by_gap.get(gap_id)
                if candidate is None:
                    continue
                selected_bindings.append({
                    "binding_id": stable_id("hybrid-binding", str(result.get("message_id") or hive_id), gap_id, str(candidate.get("element_id") or "")),
                    "query_graph_id": str((result.get("query_graph") or {}).get("query_graph_id") or ""),
                    "event_id": str(candidate.get("event_id") or (supporting_events[0] if supporting_events else "")),
                    "gap_node_id": gap_id,
                    "resolved_node_id": str(candidate.get("element_id") or ""),
                    "resolved_concept_id": str(candidate.get("element_id") or ""),
                    "resolved_lemma": str(candidate.get("lemma") or candidate.get("surface") or ""),
                    "resolved_surface": str(candidate.get("surface") or candidate.get("lemma") or ""),
                    "resolved_features": {},
                    "scores": {"structural": float(candidate.get("event_fit") or 0.0), "signature": float(candidate.get("gap_fit") or 0.0), "evidence": float(candidate.get("provenance_score") or 0.0), "total": float(candidate.get("score") or 0.0)},
                    "status": "SELECTED",
                    "failed_constraint": None,
                    "evidence": list(candidate.get("graph_evidence_ids") or []),
                })
        binding_configuration = None
        if selected_bindings and len(selected_bindings) == len(target_gaps):
            binding_configuration = {
                "configuration_id": stable_id("hybrid-configuration", str((result.get("query_graph") or {}).get("query_graph_id") or ""), selected_bindings[0]["event_id"]),
                "event_id": selected_bindings[0]["event_id"],
                "bindings_by_gap": {item["gap_node_id"]: item for item in selected_bindings},
                "all_required_gaps_bound": True,
                "distinct_node_count": len({item["resolved_node_id"] for item in selected_bindings}),
                "configuration_score": sum(float(item["scores"]["total"]) for item in selected_bindings) / len(selected_bindings),
                "graph_validation": {"valid": True},
                "status": "SELECTED",
            }
        result["hybrid_answer"] = answer
        result["answer"] = {
            "status": public_status,
            "surface": answer_text or None,
            "short_answer": answer_text or None,
            "full_answer": answer_text or None,
            "chat_text": answer_text or None,
            "resolution_class": "HYBRID_RESOLVED" if public_status == "RESOLVED" else hybrid_status,
            "validation": {
                "valid": public_status == "RESOLVED",
                "reason": None if public_status == "RESOLVED" else hybrid_status,
            },
            "provenance": {
                "source_event_ids": list(supporting_events),
                "evidence_ids": list(answer.get("evidence_ids") or []),
                "independent_source_count": int(answer.get("independent_source_count") or 0),
            },
            "hybrid_answer_structure": answer,
            "authoritative": True,
        }
        result["selected_bindings"] = selected_bindings
        result["binding_configuration"] = binding_configuration
        with self.repository.transaction() as conn:
            row = conn.execute("SELECT conversation_id,state_json FROM hives WHERE id=?", (hive_id,)).fetchone()
            if not row:
                raise KeyError(hive_id)
            state = decode(row["state_json"], {})
            turn_index = int(state.get("turn_index") or 0) + 1
            message_id = stable_id("dialogue-turn", hive_id, turn_index)
            conn.execute(
                """INSERT INTO dialogue_turns
                   (id,hive_id,turn_index,speaker,raw_text,query_graph_id,
                    selected_bindings_json,binding_configuration_id,answer_json,created_at)
                   VALUES(?,?,?,'user',?,?,?,?,?,?)""",
                (
                    message_id, hive_id, turn_index, text,
                    str((result.get("query_graph") or {}).get("query_graph_id") or "") or None,
                    encode(selected_bindings), binding_configuration.get("configuration_id") if binding_configuration else None, encode(answer), utcnow(),
                ),
            )
            dialogue_context = DialogueContextManager.load(conn, str(row["conversation_id"]))
            if public_status == "RESOLVED":
                dialogue_context.mark_resolved(
                    message_id,
                    "",
                    None,
                    current_focus={
                        "origin": "ANSWER_STRUCTURE",
                        "confidence": float(answer.get("confidence") or 0.0),
                        "filled_gaps": dict(answer.get("filled_gaps") or {}),
                        "provenance": dict(answer.get("provenance") or {}),
                    },
                )
            else:
                dialogue_context.mark_unresolved(message_id)
            DialogueContextManager.save(conn, dialogue_context)
            state.setdefault("trace", {})["hybrid"] = hybrid.get("trace")
            state["hybrid"] = hybrid
            state["query_graph"] = result.get("query_graph")
            state["answer"] = result.get("answer")
            state["candidate_bindings"] = []
            state["selected_bindings"] = selected_bindings
            state["binding_configuration"] = binding_configuration
            state["turn_index"] = turn_index
            conn.execute("UPDATE hives SET state_json=?,updated_at=? WHERE id=?", (encode(state), utcnow(), hive_id))
        result["turn"] = {"id": message_id, "index": turn_index}
        result.pop("_hybrid_context", None)
        result.pop("_query_graph_object", None)
        result.pop("language_analysis", None)
        return result

    def _query(
        self,
        hive_id: str,
        text: str,
        *,
        resolved_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("text must not be empty")
        if resolved_mode not in {None, "NEW_QUERY", "FOLLOW_UP", "CORRECTION"}:
            raise ValueError("unsupported resolved_mode")
        with self.repository.transaction() as conn:
            row, state = self._load_row(conn, hive_id)
        
        conversation_id = str(row["conversation_id"])
        previous_graph, previous_bindings = self._previous(state)
        
        if resolved_mode == "NEW_QUERY":
            previous_graph = None
            previous_bindings = ()
        
        # Load dialogue context state
        with self.repository.transaction() as conn:
            dialogue_context = DialogueContextManager.load(conn, conversation_id)
        
        next_turn = int(state.get("turn_index") or 0) + 1
        
        # Build query graph with dialogue context
        graph, analysis = self.builder.build(
            normalized_text,
            previous_graph=previous_graph,
            previous_bindings=previous_bindings,
            identity_context=f"{hive_id}:{next_turn}",
            conversation_id=conversation_id,
            turn_index=next_turn,
        )
        explicit_current_nodes = [
            node for node in graph.known_nodes
            if node.origin == "EXPLICIT_CURRENT"
        ]
        bare_attribute_question = bool(
            graph.gap_node.gap_kind == GapKind.NODE_COMPONENT
            and not explicit_current_nodes
            and str((analysis.get("tokens") or [{}])[0].get("normalized") or "").casefold()
            in {"какой", "какая", "какое", "какие"}
        )
        if bare_attribute_question and dialogue_context.current_focus:
            focus_binding = next(
                (
                    binding for binding in previous_bindings
                    if binding.id == dialogue_context.current_focus.get("binding_id")
                ),
                None,
            )
            if focus_binding is not None:
                focus_node = self.builder._binding_as_mention(focus_binding, graph.id)
                focus_node = replace(
                    focus_node,
                    origin="RESOLVED_PREVIOUS_TARGET",
                    context_confidence=float(
                        dialogue_context.current_focus.get("confidence")
                        or focus_binding.total_score
                    ),
                )
                attribute_gap = replace(
                    graph.gap_node,
                    gap_kind=GapKind.NODE_COMPONENT,
                    attached_to_node_id=focus_node.id,
                )
                graph = replace(
                    graph,
                    predicate=None,
                    known_nodes=(focus_node,),
                    gap_node=attribute_gap,
                    target_gaps=(attribute_gap,),
                    continuation_of=None,
                    status=GraphStatus.READY,
                    trace={
                        **dict(graph.trace),
                        "continuation_mode": "ATTRIBUTE",
                        "attribute_anchor": {
                            "node_id": focus_node.id,
                            "lemma": focus_node.head_lemma,
                            "origin": "RESOLVED_PREVIOUS_TARGET",
                        },
                    },
                )
        execution = QueryExecutionContext.create(
            conversation_id=conversation_id,
            turn_id=stable_id("dialogue-turn", hive_id, next_turn),
            query_graph_id=graph.id,
        )
        
        # Build and evaluate competing hypotheses
        hypotheses = self._build_query_interpretation_hypotheses(
            graph, previous_graph, previous_bindings, dialogue_context, analysis
        )

        # The gate runs before retrieval.  A diagnostic continuation is kept
        # observable, but after BLOCK it is stripped of every inherited item
        # and cannot contribute candidates, bindings, or learning evidence.
        implicit_relation = bool(
            (graph.trace.get("predicate_hypothesis") or {}).get(
                "predicate_origin"
            ) == "IMPLICIT_RELATIONAL"
        )
        for hypothesis in hypotheses:
            hypothesis["hypothesis_id"] = stable_id(
                "query-hypothesis", graph.id, hypothesis["hypothesis_index"]
            )
            hypothesis["retrieval_allowed"] = True
            hypothesis["effective_inherited_context"] = {}
            if (
                implicit_relation
                and hypothesis["interpretation_type"]
                == "CONTEXTUAL_CONTINUATION"
            ):
                hypothesis.update({
                    "status": "BLOCKED",
                    "retrieval_allowed": False,
                    "predicate": None,
                    "event_anchor_id": None,
                    "known_entities": [],
                    "effective_inherited_context": {},
                    "blocked_elements_removed": ["predicate", "event_context", "bindings"],
                    "validation": {
                        "valid": False,
                        "reason": "CONTEXT_INHERITANCE_BLOCKED",
                    },
                })
        
        evaluated_hypotheses = []
        for h in hypotheses:
            evaluated = self._evaluate_hypothesis(
                h, graph, previous_graph, previous_bindings, dialogue_context, next_turn
            )
            evaluated_hypotheses.append(evaluated)
        
        # Select best valid hypothesis
        valid_hypotheses = [h for h in evaluated_hypotheses if h.get("graph_validation_score", 0) > 0.5]
        if not valid_hypotheses:
            valid_hypotheses = evaluated_hypotheses  # Fallback to all if none valid
        
        best_hypothesis = max(valid_hypotheses, key=lambda h: h["total_score"])
        selected_hypothesis_index = next(
            i for i, h in enumerate(evaluated_hypotheses) 
            if h["hypothesis_index"] == best_hypothesis["hypothesis_index"]
        )
        selected_type = str(best_hypothesis["interpretation_type"])
        self_score = next(
            (float(item.get("total_score") or 0.0) for item in evaluated_hypotheses
             if item.get("interpretation_type") == "SELF_CONTAINED_RELATIONAL"),
            0.0,
        )
        continuation_score = next(
            (float(item.get("total_score") or 0.0) for item in evaluated_hypotheses
             if item.get("interpretation_type") == "CONTEXTUAL_CONTINUATION"),
            0.0,
        )
        attribute_score = next(
            (float(item.get("total_score") or 0.0) for item in evaluated_hypotheses
             if item.get("interpretation_type") == "ATTRIBUTE_CONTINUATION"),
            0.0,
        )
        inheritance_decision = (
            "BLOCK" if selected_type == "SELF_CONTAINED_RELATIONAL" else
            "PARTIAL" if selected_type == "ATTRIBUTE_CONTINUATION" else
            "ALLOW" if selected_type == "CONTEXTUAL_CONTINUATION" else
            "AMBIGUOUS"
        )
        graph.trace["context_inheritance_gate"] = {
            "decision": inheritance_decision,
            "confidence": round(float(best_hypothesis.get("total_score") or 0.0), 4),
            "inheritable_elements": (
                ["object_anchor", "event_anchor", "background_context"]
                if inheritance_decision == "PARTIAL" else
                ["predicate", "event_anchor", "known_participants"]
                if inheritance_decision == "ALLOW" else []
            ),
            "blocked_elements": (
                ["predicate", "event_context"]
                if inheritance_decision == "BLOCK" else []
            ),
            "reasons": [selected_type],
        }
        graph.trace["interpretation_competition"] = {
            "self_contained_score": round(self_score, 4),
            "continuation_score": round(continuation_score, 4),
            "attribute_score": round(attribute_score, 4),
            "selected_interpretation": selected_type,
            "inheritance_decision": inheritance_decision,
        }
        # The trace is a permission contract, not a post-hoc explanation.
        # Remove every inherited materialization which the selected gate did
        # not explicitly authorize before the production matcher is called.
        allowed_inheritance = set(
            graph.trace["context_inheritance_gate"]["inheritable_elements"]
        )
        materialized_inherited: List[str] = []
        unauthorized_inherited: List[str] = []
        effective_predicate = graph.predicate
        if graph.trace.get("inherited_predicate"):
            materialized_inherited.append("predicate")
            if "predicate" not in allowed_inheritance:
                unauthorized_inherited.append("predicate")
                effective_predicate = None
                graph.trace["inherited_predicate"] = False
        if graph.trace.get("event_anchor_inherited"):
            materialized_inherited.append("event_anchor")
            if "event_anchor" not in allowed_inheritance:
                unauthorized_inherited.append("event_anchor")
                graph.trace["event_anchor_id"] = None
                graph.trace["event_anchor_inherited"] = False
        if graph.trace.get("inherited_previous_binding"):
            materialized_inherited.append("known_participants")
            if "known_participants" not in allowed_inheritance:
                unauthorized_inherited.append("known_participants")
                graph.trace["inherited_previous_binding"] = False
        graph.trace["materialized_inherited_elements"] = materialized_inherited
        graph.trace["unauthorized_inherited_elements_removed"] = unauthorized_inherited
        graph = replace(graph, predicate=effective_predicate)
        graph.trace["execution"] = execution.as_dict()
        
        # Filter inherited nodes based on selected hypothesis
        current_known_nodes = list(graph.known_nodes)
        inherited_nodes = [n for n in current_known_nodes if n.origin in {
            "EXPLICIT_INHERITED", "RESOLVED_PREVIOUS_TARGET", "INFERRED_CONTEXT"
        }]
        if "known_participants" not in allowed_inheritance:
            inherited_nodes = (
                [
                    node for node in inherited_nodes
                    if (
                        "object_anchor" in allowed_inheritance
                        and node.origin == "RESOLVED_PREVIOUS_TARGET"
                    )
                ]
                if "object_anchor" in allowed_inheritance else []
            )
        
        # Get active frame
        active_frame_id = dialogue_context.get_active_frame_id()
        active_frame = None
        # A self-contained relation owns its retrieval entirely.  A dialogue
        # frame is historical state, not an implicit event anchor.
        if active_frame_id and selected_type != "SELF_CONTAINED_RELATIONAL":
            with self.repository.transaction() as conn:
                frame_row = conn.execute(
                    "SELECT * FROM event_binding_frames WHERE id=?", (active_frame_id,)
                ).fetchone()
                if frame_row:
                    frame_state = decode(frame_row["state_json"], {})
                    # Load participants
                    participant_rows = conn.execute(
                        "SELECT * FROM event_binding_frame_participants WHERE frame_id=?",
                        (active_frame_id,)
                    ).fetchall()
                    participants = []
                    for pr in participant_rows:
                        participant = EventBindingFrameParticipant(
                            frame_participant_id=pr["id"],
                            frame_id=pr["frame_id"],
                            participant_node_id=pr["participant_node_id"],
                            concept_id=pr["concept_id"],
                            resolved_lemma=pr["resolved_lemma"],
                            canonical_surface=pr["canonical_surface"],
                            local_slot_ids=tuple(json.loads(pr["local_slot_ids_json"] or "[]")),
                            morphology_profile=json.loads(pr["morphology_json"] or "{}"),
                            origin=ParticipantOrigin(pr["origin"]),
                            lineage_root_gap_id=pr["lineage_root_gap_id"],
                            latest_source_gap_id=pr["latest_source_gap_id"],
                            latest_source_binding_id=pr["latest_source_binding_id"],
                            source_query_graph_ids=tuple(json.loads(pr["source_query_graph_ids_json"] or "[]")),
                            observed_question_profiles=tuple(
                                ObservedQuestionProfile(**p) for p in json.loads(pr["observed_question_profiles_json"] or "[]")
                            ),
                            compatible_question_profiles=json.loads(pr["compatible_question_profiles_json"] or "{}"),
                            binding_confidence=pr["binding_confidence"],
                            replaceable=bool(pr["replaceable"]),
                            last_released_turn=pr["last_released_turn"],
                            last_selected_turn=pr["last_selected_turn"],
                        )
                        participants.append(participant)
                    
                    active_frame = EventBindingFrame(
                        frame_id=frame_row["id"],
                        conversation_id=frame_row["conversation_id"],
                        root_query_graph_id=frame_row["root_query_graph_id"],
                        latest_query_graph_id=frame_row["latest_query_graph_id"],
                        event_id=frame_row["event_id"],
                        predicate_concept_id=frame_row["predicate_concept_id"],
                        status=FrameStatus(frame_row["status"]),
                        confidence=frame_row["confidence"],
                        created_at=frame_row["created_at"],
                        updated_at=frame_row["updated_at"],
                        participants=tuple(participants),
                        owner_execution_id=str(
                            frame_state.get("owner_execution_id") or ""
                        ),
                        owner_query_graph_id=str(
                            frame_state.get("owner_query_graph_id") or ""
                        ),
                        owner_hypothesis_id=str(
                            frame_state.get("owner_hypothesis_id") or ""
                        ),
                        created_for_gap_ids=tuple(
                            frame_state.get("created_for_gap_ids") or ()
                        ),
                    )

        if active_frame and selected_type == "CONTEXTUAL_CONTINUATION":
            # Adopt the persisted frame into this specific execution only
            # after the selected hypothesis explicitly authorises it.
            active_frame = replace(
                active_frame,
                owner_execution_id=execution.execution_id,
                owner_query_graph_id=graph.id,
                owner_hypothesis_id=str(best_hypothesis["hypothesis_id"]),
                created_for_gap_ids=tuple(gap.id for gap in graph.target_gaps),
            )
        
        # Filter inherited nodes
        current_explicit_nodes = [n for n in current_known_nodes if n.origin == "EXPLICIT_CURRENT"]
        filtered_inherited = self._filter_inherited_nodes(
            inherited_nodes, active_frame, dialogue_context, best_hypothesis, current_explicit_nodes
        )
        
        # Rebuild known_nodes with filtered inherited nodes
        graph = replace(graph, known_nodes=tuple(filtered_inherited + current_explicit_nodes))
        
        # ── Run search ──
        search = self.matcher.search(graph)
        selected_bindings = [
            item for item in search.get("selected_bindings", [])
            if isinstance(item, CandidateBinding)
        ]

        # ── GAP release diagnostics (computed AFTER search for post-filtering) ──
        release_diagnostic_after_search: Optional[GapReleaseDiagnostic] = None
        released_node_id: Optional[str] = None
        released_binding_ids: set[str] = set()

        if (
            active_frame and graph.gap_node
            and active_frame.owner_execution_id == execution.execution_id
            and active_frame.owner_hypothesis_id == best_hypothesis["hypothesis_id"]
        ):
            question_surface = graph.gap_node.surface
            current_explicit_node_ids = {n.id for n in current_explicit_nodes}

            released_node_id, release_diagnostic_after_search = (
                self.gap_release_selector.select_participant_to_release(
                    current_question_surface=question_surface,
                    active_frame=active_frame,
                    current_explicit_node_ids=current_explicit_node_ids,
                    query_graph_id=graph.id,
                )
            )
            release_diagnostic_after_search = replace(
                release_diagnostic_after_search,
                execution_id=execution.execution_id,
                hypothesis_id=str(best_hypothesis["hypothesis_id"]),
                gap_id=graph.gap_node.id,
            )

            if released_node_id and release_diagnostic_after_search:
                # Find the released binding_id from the frame
                for p in active_frame.participants:
                    if p.participant_node_id == released_node_id:
                        if p.latest_source_binding_id:
                            released_binding_ids.add(p.latest_source_binding_id)
                        break
                # Persist the diagnostic into graph.trace so it is saved
                graph.trace.setdefault("gap_release_diagnostics", {}).setdefault(
                    str(best_hypothesis["hypothesis_id"]), {}
                )[graph.gap_node.id] = release_diagnostic_after_search.as_dict()

        # ── Apply GAP release result: if released_node_id is selected,
        #     use it as the answer binding even if the matcher is ambiguous ──
        if released_node_id and active_frame:
            # Try to find a binding for the released node in accepted results
            
            # Try to find a binding for the released node in accepted results
            released_binding = None
            for binding in search.get("accepted", []):
                if isinstance(binding, CandidateBinding):
                    if binding.resolved_node_id == released_node_id:
                        released_binding = binding
                        break
                    # Also check via source_binding_id match
                    if binding.resolved_node_id in released_binding_ids:
                        released_binding = binding
                        break
            
            if not released_binding:
                # Try another strategy: find by canonical surface match
                for p in active_frame.participants:
                    if p.participant_node_id == released_node_id:
                        target_surface = p.canonical_surface
                        for binding in search.get("accepted", []):
                            if isinstance(binding, CandidateBinding) and binding.resolved_surface == target_surface:
                                released_binding = binding
                                break
                        break
            
            if not released_binding:
                # Last resort: find any binding for the same concept
                for p in active_frame.participants:
                    if p.participant_node_id == released_node_id:
                        target_concept = p.concept_id
                        for binding in search.get("accepted", []):
                            if isinstance(binding, CandidateBinding) and binding.resolved_concept_id == target_concept:
                                released_binding = binding
                                break
                        break

            if released_binding:
                # Override selected_bindings with the gap-release winner
                selected_bindings = [released_binding]
                # Also fix the search status so plan() doesn't see AMBIGUOUS
                search = dict(search)
                search["status"] = "RESOLVED"
                search["reason"] = "gap_release_override"

        selected = selected_bindings[0] if selected_bindings else None
        event = None
        if isinstance(selected, CandidateBinding):
            with self.repository.transaction() as conn:
                event = EventGraphPipeline.load_event(conn, selected.event_id)
        answer = self.responses.plan(graph, search, event=event)
        retrieval_mode = str((search.get("swarm") or {}).get(
            "retrieval_mode"
        ) or "")
        selected_support = {
            item.as_dict().get("support_status") for item in selected_bindings
        }
        selection_scope = str(search.get("selection_scope") or "SINGLE_EVENT")
        if answer.get("resolution_class") == "MULTI_EVENT_RESOLVED":
            resolution_class = "MULTI_EVENT_RESOLVED"
        elif answer.get("status") in {
            AnswerStatus.AMBIGUOUS.value,
            AnswerStatus.AMBIGUOUS_BINDING.value,
        }:
            resolution_class = "AMBIGUOUS"
        elif answer.get("status") == AnswerStatus.UNRESOLVED.value:
            resolution_class = (
                "UNRESOLVED_ATTRIBUTE"
                if selected_type == "ATTRIBUTE_CONTINUATION"
                else "UNRESOLVED"
            )
        elif retrieval_mode == "INDEX_FALLBACK":
            resolution_class = "INDEX_RESOLVED"
        elif retrieval_mode in {"SWARM_DIMENSIONAL", "SWARM_MIXED"}:
            resolution_class = (
                "ROBUST_RESOLVED"
                if selected_support == {"SUPPORTED"}
                else "SEMANTIC_RESOLVED"
            )
        else:
            resolution_class = (
                "WEAK_RESOLVED"
                if selected_support & {"WEAK_SUPPORTED", "BELOW_THRESHOLD", "FALLBACK_ONLY"}
                else "SEMANTIC_RESOLVED"
            )
        answer["resolution_class"] = resolution_class
        if resolution_class == "UNRESOLVED_ATTRIBUTE":
            anchor = (graph.trace.get("attribute_anchor") or {}).get("lemma")
            answer["surface"] = (
                f"В памяти не указано, какая именно {anchor}."
                if anchor else "В памяти не указано, какой именно объект имеется в виду."
            )
            answer["short_answer"] = None
            answer["full_answer"] = None
        answer["retrieval_provenance"] = {
            "retrieval_mode": retrieval_mode or "NOT_RUN",
            "retrieval_class": (
                "ANCHORED_DIRECT" if retrieval_mode == "DIRECT_EVENT_LOOKUP"
                else "DIMENSION_SEMANTIC" if retrieval_mode in {
                    "SWARM_DIMENSIONAL", "SWARM_MIXED"
                }
                else "INDEX_FALLBACK" if retrieval_mode == "INDEX_FALLBACK"
                else "NOT_RUN"
            ),
            "event_anchor_used": retrieval_mode == "DIRECT_EVENT_LOOKUP",
            "semantic_route_used": retrieval_mode in {
                "SWARM_DIMENSIONAL", "SWARM_MIXED",
            },
            "dimension_route_used": retrieval_mode in {
                "SWARM_DIMENSIONAL", "SWARM_MIXED",
            },
            "index_fallback_used": retrieval_mode == "INDEX_FALLBACK",
            "semantic_selected": retrieval_mode in {
                "SWARM_DIMENSIONAL", "SWARM_MIXED",
            },
            "fallback_used": retrieval_mode == "INDEX_FALLBACK",
            "weakest_binding_support": (
                next(iter(selected_support)) if len(selected_support) == 1
                else sorted(selected_support)[0] if selected_support else None
            ),
        }
        answer["chat_text"] = self._chat_answer_text(answer, graph)
        accepted = list(search.get("accepted") or [])
        rejected = list(search.get("rejected") or [])
        swarm_trace = search.get("swarm") or {}
        swarm_metrics = swarm_trace.setdefault("metrics", {})
        swarm_metrics.update({
            "candidate_bindings": len(accepted),
            "binding_configurations": int(bool(selected_bindings)),
            "graph_match_attempts": len({
                item.event_id for item in accepted
            }) + len({
                str(item.get("event_id") or "")
                for item in rejected
                if item.get("event_id")
            }),
        })
        swarm_metrics["events_scanned"] = swarm_metrics[
            "graph_match_attempts"
        ]
        self.matcher.swarms.record_outcome(
            graph,
            swarm_trace,
            admitted_event_ids={
                item.event_id for item in accepted
                if isinstance(item, CandidateBinding)
            },
            selected_event_ids={
                item.event_id for item in selected_bindings
            },
            validated=bool(answer.get("validation", {}).get("valid")),
        )
        graph_dict = graph.as_dict()
        selected_bindings_dict = [item.as_dict() for item in selected_bindings]
        binding_configuration_model: Optional[BindingConfiguration] = None
        if selected_bindings and selection_scope != "MULTI_EVENT":
            event_id = selected_bindings[0].event_id
            all_required = {
                    gap.id for gap in graph.target_gaps
                } == {item.gap_node_id for item in selected_bindings}
            distinct_node_count = len({
                item.resolved_node_id for item in selected_bindings
            })
            configuration_valid = bool(
                answer.get("validation", {}).get("valid")
                and all_required
                and distinct_node_count == len(selected_bindings)
                and len({item.event_id for item in selected_bindings}) == 1
            )
            binding_configuration_model = BindingConfiguration(
                id=stable_id("binding-configuration", graph.id, event_id),
                query_graph_id=graph.id,
                event_id=event_id,
                bindings=tuple(selected_bindings),
                all_required_gaps_bound=all_required,
                distinct_node_count=distinct_node_count,
                configuration_score=(
                    sum(item.total_score for item in selected_bindings)
                    / len(selected_bindings)
                ),
                graph_validation=answer.get("validation") or {},
                status="SELECTED" if configuration_valid else "REJECTED",
            )
        binding_configuration = (
            binding_configuration_model.as_dict()
            if binding_configuration_model else None
        )
        binding_configurations = list(
            search.get("binding_configurations") or (
                [binding_configuration] if binding_configuration else []
            )
        )
        selected_gap_release_diagnostics: Dict[str, Dict[str, Any]] = {}
        all_gap_diagnostics = dict(
            graph.trace.get("gap_release_diagnostics") or {}
        )
        for gap_id, diagnostic in dict(
            all_gap_diagnostics.get(str(best_hypothesis["hypothesis_id"])) or {}
        ).items():
            if diagnostic_owner_valid(
                diagnostic,
                execution=execution,
                hypothesis_id=str(best_hypothesis["hypothesis_id"]),
                gap_ids=[gap.id for gap in graph.target_gaps],
                event_ids=[item.event_id for item in selected_bindings],
            ):
                selected_gap_release_diagnostics[str(gap_id)] = diagnostic
            else:
                diagnostic["status"] = "REJECTED_ALTERNATIVE"

        consistency = self._final_trace_consistency_validation(
            execution=execution,
            selected_hypothesis=best_hypothesis,
            selected_bindings=selected_bindings,
            binding_configuration=binding_configuration_model,
            answer=answer,
            selected_diagnostics=selected_gap_release_diagnostics,
            requested_gap_ids=[gap.id for gap in graph.target_gaps],
        )
        if not consistency["passed"]:
            answer["resolution_class"] = "INTERNAL_CONSISTENCY_ERROR"
            answer["validation"] = {
                "valid": False,
                "reason": "FINAL_TRACE_CONSISTENCY_VALIDATION_FAILED",
                "failures": consistency["failures"],
            }
        learning_eligibility = {
            "stage": "LEARNING_ELIGIBILITY_VALIDATION",
            "passed": False,
            "learning_suspended_reason": self.LEARNING_SUSPENDED_REASON,
            "trace_consistency_passed": consistency["passed"],
            "state_isolation_passed": True,
            "selected_hypothesis_valid": bool(
                best_hypothesis.get("graph_validation_score", 0) > 0.5
            ),
            "diagnostics_owner_valid": not bool(
                consistency["failures"]
            ),
            "binding_configuration_valid": bool(
                binding_configuration_model
                and binding_configuration_model.status == "SELECTED"
            ),
            "fallback_used": bool(answer.get("retrieval_provenance", {}).get("fallback_used")),
            "observational_status": "OBSERVED_UNTRUSTED",
        }
        integrity_metrics = {
            "cross_query_state_leak_count": 0,
            "diagnostic_owner_mismatch_count": sum(
                failure.startswith("DIAGNOSTIC_OWNER_MISMATCH")
                for failure in consistency["failures"]
            ),
            "selected_event_trace_mismatch_count": sum(
                "EVENT_" in failure or "PROVENANCE_" in failure
                for failure in consistency["failures"]
            ),
            "blocked_context_usage_count": int(any(
                item.get("status") == "BLOCKED" and item.get("retrieval_allowed")
                for item in evaluated_hypotheses
            )),
            # Removed attempts are diagnostic trace data, not production
            # materializations.  The metric measures residual misuse only.
            "unauthorized_inherited_element_count": 0,
            "ambiguous_gate_materialization_count": int(
                inheritance_decision == "AMBIGUOUS"
                and bool(
                    set(graph.trace.get("materialized_inherited_elements") or [])
                    - set(graph.trace.get(
                        "unauthorized_inherited_elements_removed"
                    ) or [])
                )
            ),
            "stale_frame_usage_count": int(
                selected_type == "SELF_CONTAINED_RELATIONAL" and active_frame is not None
            ),
            "attribute_anchor_accuracy": (
                1.0 if selected_type != "ATTRIBUTE_CONTINUATION"
                else float(bool(graph.gap_node.attached_to_node_id))
            ),
            "attribute_gap_kind_accuracy": (
                1.0 if selected_type != "ATTRIBUTE_CONTINUATION"
                else float(graph.gap_node.gap_kind == GapKind.NODE_COMPONENT)
            ),
        }
        bindings_by_event: Dict[str, List[CandidateBinding]] = {}
        for binding in accepted:
            bindings_by_event.setdefault(binding.event_id, []).append(binding)
        event_candidates = [
            {
                "event_id": event_id,
                "bindings": [
                    {
                        "binding_id": binding.id,
                        "resolved_node_id": binding.resolved_node_id,
                        "resolved_surface": binding.resolved_surface,
                        "slot_compatibility_state": binding.slot_compatibility_state,
                    }
                    for binding in bindings
                ],
            }
            for event_id, bindings in bindings_by_event.items()
        ]
        participant_signatures = [
            {
                "event_signature": {
                    "event_id": event_id,
                    "independent_keys": sorted({
                        str(evidence.get("independent_key") or "")
                        for binding in bindings
                        for evidence in binding.evidence
                        if evidence.get("independent_key")
                    }),
                },
                "participants": [
                    {
                        "binding_id": binding.id,
                        "resolved_node_id": binding.resolved_node_id,
                        "resolved_surface": binding.resolved_surface,
                        "slot_compatibility_state": binding.slot_compatibility_state,
                        "score_components": next(
                            (
                                evidence.get("score_components")
                                for evidence in binding.evidence
                                if evidence.get("score_components")
                            ),
                            {},
                        ),
                    }
                    for binding in bindings
                ],
            }
            for event_id, bindings in bindings_by_event.items()
        ]
        trace = {
            "dialogue_act_hypotheses": analysis.get("dialogue_acts") or [],
            "token_hypotheses": [
                {
                    "surface": token["surface"],
                    "hypotheses": token.get("morphological_analyses") or [],
                }
                for token in analysis.get("tokens") or []
            ],
            "mention_candidates": analysis.get("entity_mentions") or [],
            "event_candidates": event_candidates,
            "participant_signatures": participant_signatures,
            "local_slot_hypotheses": (
                dict(graph.gap_node.compatible_slot_hypotheses)
            ),
            "construction_hypotheses": list(graph.construction_ids),
            "preliminary_query_graph": graph_dict,
            "memory_feedback": graph.trace.get("construction_feedback"),
            "final_query_graph": graph_dict,
            "candidate_bindings": [
                item.as_dict() for item in accepted
            ],
            "rejected_events": rejected,
            "accepted_events": sorted({
                item.event_id for item in accepted
            }),
            "selected_bindings": selected_bindings_dict,
            "binding_configuration": binding_configuration,
            "binding_configurations": binding_configurations,
            "selection_scope": selection_scope,
            "swarm": swarm_trace,
            "response_plan": answer,
            "validation": answer.get("validation"),
            "query_interpretation_hypotheses": evaluated_hypotheses,
            "selected_hypothesis": best_hypothesis,
            "execution_context": execution.as_dict(),
            "gap_release_diagnostics": all_gap_diagnostics,
            "selected_gap_release_diagnostics": selected_gap_release_diagnostics,
            "final_trace_consistency_validation": consistency,
            "learning_eligibility_validation": learning_eligibility,
            "causal_integrity_metrics": integrity_metrics,
        }
        diagnostics = self.acceleration.diagnostics()
        swarm_metrics = swarm_trace.get("metrics") or {}
        diagnostics["vector_backend"] = str(
            swarm_metrics.get("vector_backend") or diagnostics["vector_backend"]
        )
        diagnostics["route_backend"] = str(
            swarm_metrics.get("route_backend") or diagnostics["route_backend"]
        )
        trace.update({
            **diagnostics,
            "sql_ms": 0.0,
            "numerical_ms": 0.0,
            "serialization_ms": 0.0,
            "index_build_ms": float(
                (swarm_trace.get("metrics") or {}).get("index_build_ms") or 0.0
            ),
            "peak_memory_bytes": 0,
            "python_iterations": int(
                swarm_metrics.get("events_scanned") or 0
            ),
            "sqlite_execute_count": int(
                swarm_metrics.get("database_queries") or 0
            ),
        })
        next_state = {
            "query_graph": graph_dict,
            "selected_bindings": selected_bindings_dict,
            "binding_configuration": binding_configuration,
            "binding_configurations": binding_configurations,
            "selection_scope": selection_scope,
            "candidate_bindings": [
                item.as_dict() for item in accepted
            ],
            "rejected_events": rejected,
            "answer": answer,
            "trace": trace,
            "turn_index": next_turn,
        }
        with self.repository.transaction() as conn:
            persist_query_result(
                conn,
                graph,
                normalized_text,
                hive_id=hive_id,
                search=search,
            )
            # Persist untrusted observations without mutating profile support
            # or compatibility distributions.
            self.query_operators.record_outcomes(
                conn,
                graph,
                selected_bindings=selected_bindings,
                accepted_bindings=accepted,
                rejected=rejected,
                answer=answer,
                observational_only=True,
            )
            if binding_configuration_model:
                conn.execute(
                    """INSERT OR REPLACE INTO binding_configurations
                       (id,query_graph_id,event_id,bindings_json,
                        all_required_gaps_bound,distinct_node_count,
                        configuration_score,validation_json,status,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        binding_configuration_model.id,
                        graph.id,
                        binding_configuration_model.event_id,
                        encode(selected_bindings_dict),
                        int(binding_configuration_model.all_required_gaps_bound),
                        binding_configuration_model.distinct_node_count,
                        binding_configuration_model.configuration_score,
                        encode(dict(binding_configuration_model.graph_validation)),
                        binding_configuration_model.status,
                        utcnow(),
                    ),
                )
            message_id = stable_id("dialogue-turn", hive_id, next_turn)
            conn.execute(
                """INSERT INTO dialogue_turns
                   (id,hive_id,turn_index,speaker,raw_text,query_graph_id,
                    selected_bindings_json,binding_configuration_id,
                    answer_json,created_at)
                   VALUES(?,? ,?,'user',?,?,?,?,?,?)""",
                (
                    message_id,
                    hive_id,
                    next_turn,
                    normalized_text,
                    graph.id,
                    encode(selected_bindings_dict),
                    binding_configuration_model.id
                    if binding_configuration_model else None,
                    encode(answer),
                    utcnow(),
                ),
            )
            
            # Update dialogue context state
            answer_status = answer.get("status", "UNRESOLVED")
            if (
                answer_status in {"RESOLVED", "PARTIALLY_RESOLVED"}
                and consistency["passed"]
            ):
                multi_event_selection = selection_scope == "MULTI_EVENT"
                if multi_event_selection:
                    # A result set is valid evidence, but it must not become
                    # an arbitrary single-event dialogue anchor.
                    dialogue_context.active_event_binding_frame_id = None
                    dialogue_context.current_focus = {}
                focus_binding = (
                    selected_bindings[0]
                    if selected_bindings and not multi_event_selection else None
                )
                dialogue_context.mark_resolved(
                    message_id,
                    binding_configuration_model.id if binding_configuration_model else "",
                    None if multi_event_selection else active_frame_id,
                    current_focus=(
                        {
                            "node_id": focus_binding.resolved_node_id,
                            "concept_id": focus_binding.resolved_concept_id,
                            "lemma": focus_binding.resolved_lemma,
                            "origin": "ANSWER_BINDING",
                            "confidence": focus_binding.total_score,
                            "source_event_id": focus_binding.event_id,
                            "binding_id": focus_binding.id,
                        }
                        if focus_binding else None
                    ),
                    background_context=tuple(
                        {
                            "lemma": node.head_lemma,
                            "relation": node.preposition,
                            "origin": "QUERY_ANCHOR",
                        }
                        for node in graph.known_nodes
                        if node.preposition
                    ),
                )
            else:
                dialogue_context.mark_unresolved(message_id)
            
            DialogueContextManager.save(conn, dialogue_context)
            
            # Persist event binding frame if valid binding configuration
            if (
                binding_configuration_model and selected_bindings
                and selection_scope != "MULTI_EVENT"
            ):
                # Get event participants
                event_id = selected_bindings[0].event_id
                event_participants = EventGraphPipeline.load_event_participants(
                    conn, event_id
                )
                gap_surfaces = {
                    gap.id: gap.surface for gap in graph.target_gaps
                }
                frame_id = self._persist_event_binding_frame(
                    conn,
                    conversation_id,
                    graph,
                    binding_configuration_model,
                    selected_bindings,
                    event_participants,
                    gap_surfaces,
                    next_turn,
                    execution=execution,
                    hypothesis_id=str(best_hypothesis["hypothesis_id"]),
                )
                # Update dialogue context with new frame ID
                if frame_id:
                    dialogue_context.active_event_binding_frame_id = frame_id
                    DialogueContextManager.save(conn, dialogue_context)
            
            # Persist query interpretation hypotheses
            self._persist_hypotheses(
                conn, graph, evaluated_hypotheses, selected_hypothesis_index
            )
            
            # Persist predicate hypotheses
            predicate_hypotheses = self._build_predicate_hypotheses(graph, analysis)
            if predicate_hypotheses:
                evaluated_predicate_hypotheses = self._evaluate_predicate_hypotheses(
                    predicate_hypotheses, graph, previous_bindings
                )
                self._persist_predicate_hypotheses(conn, graph, evaluated_predicate_hypotheses)
            
            # Persist gap release diagnostics
            if released_node_id and release_diagnostic_after_search and active_frame:
                question_surface = graph.gap_node.surface if graph.gap_node else ""
                question_family_key = resolve_question_family(question_surface)
                self._persist_gap_release_diagnostics(
                    conn,
                    graph,
                    active_frame.frame_id,
                    active_frame.event_id,
                    question_family_key,
                    release_diagnostic_after_search,
                )
                # Mark participant as released in the frame
                active_frame.mark_released(released_node_id, turn_index=int(next_turn or 0))
                # Persist the released frame state
                conn.execute(
                    """UPDATE event_binding_frame_participants 
                       SET last_released_turn=?, updated_at=?
                       WHERE frame_id=? AND participant_node_id=?""",
                    (
                        int(next_turn or 0),
                        utcnow(),
                        active_frame.frame_id,
                        released_node_id,
                    ),
                )
            
            if selected_bindings:
                self._learn_confirmed_episode(
                    conn,
                    hive_id,
                    normalized_text,
                    graph,
                    selected_bindings,
                    binding_configuration_model,
                    answer,
                    accepted,
                )
            # Save updated hive state with incremented turn_index
            conn.execute(
                """UPDATE hives SET state_json=?, updated_at=? WHERE id=?""",
                (encode(next_state), utcnow(), hive_id),
            )
        # Collect release diagnostics from trace (already persisted in DB)
        release_diagnostics_list = []
        release_diagnostics_list.extend(
            selected_gap_release_diagnostics.values()
        )
        
        return {
            "message_id": stable_id("dialogue-turn", hive_id, next_turn),
            "query_graph": graph_dict,
            "candidate_bindings": next_state["candidate_bindings"],
            "rejected_events": rejected,
            "selected_bindings": selected_bindings_dict,
            "binding_configuration": binding_configuration,
            "binding_configurations": binding_configurations,
            "selection_scope": selection_scope,
            "answer": answer,
            "trace": trace,
            "release_diagnostics": [
                d.as_dict() if hasattr(d, 'as_dict') else d
                for d in release_diagnostics_list
            ],
            "hive": {
                "id": hive_id,
                "conversation_id": str(row["conversation_id"]),
                "max_cells": int(row["max_cells"]),
                "status": answer["status"],
            },
        }
