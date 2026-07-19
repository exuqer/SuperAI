"""Application services for the fresh V2.7 graph pipeline."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence

from server.core.settings import settings

from .event_graph import EventGraphPipeline
from .graph_learning import ConstructionLearner, ObservationSignature
from .graph_models import (
    AnswerStatus,
    BindingConfiguration,
    CandidateBinding,
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
    utcnow,
)
from .query_graph import (
    GraphMatcher,
    GraphResponsePlanner,
    QueryGraphBuilder,
    persist_query_result,
)
from .universe import UniverseService


class GraphTrainingService:
    """Stage and commit knowledge without a legacy projection."""

    def __init__(
        self,
        repository: Optional[GraphRepository] = None,
        morphology: Any = None,
    ) -> None:
        if morphology is None:
            from .russian_morphology import RussianMorphology
            morphology = RussianMorphology()
        self.repository = repository or GraphRepository()
        self.events = EventGraphPipeline(self.repository, morphology)
        self.universes = UniverseService(self.repository)

    def _materialize_and_project(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        discover_dimensions = kwargs.pop("discover_dimensions", None)
        project_universes = bool(kwargs.pop("project_universes", True))
        result = self.events.materialize(*args, **kwargs)
        if result.get("status") == "CONFIRMED" and project_universes:
            if discover_dimensions is None:
                with self.repository.transaction() as conn:
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
            )
        elif result.get("status") == "CONFIRMED":
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
        result = self.events.retract(source_id)
        result["universe_update"] = self.universes.remove_source(source_id)
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
            conn.execute(
                "DELETE FROM knowledge_sources WHERE id=?",
                (source_id,),
            )
        metadata = decode(payload["metadata_json"], {})
        return self._materialize_and_project(
            str(payload["raw_text"]),
            source_type=str(payload["source_type"]),
            independent_key=str(payload["independent_key"]),
            domain_key=str(metadata.get("domain_key") or payload["source_type"]),
            force_status="CONFIRMED",
        )

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
        committed = [
            self.commit(str(row["id"]), manual_validation=True)
            for row in rows
        ]
        final_status = (
            "COMMITTED"
            if all(item["status"] == "CONFIRMED" for item in committed)
            else "PARTIALLY_COMMITTED"
        )
        with self.repository.transaction() as conn:
            conn.execute(
                """UPDATE graph_batches SET status=?,updated_at=?
                   WHERE id=?""",
                (final_status, utcnow(), batch_id),
            )
        return {
            "batch_id": batch_id,
            "status": final_status,
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

    def __init__(
        self,
        repository: Optional[GraphRepository] = None,
        morphology: Any = None,
    ) -> None:
        if morphology is None:
            from .russian_morphology import RussianMorphology
            morphology = RussianMorphology()
        self.repository = repository or GraphRepository()
        self.morphology = morphology
        self.builder = QueryGraphBuilder(self.repository, morphology)
        self.matcher = GraphMatcher(self.repository)
        self.responses = GraphResponsePlanner(morphology)
        self.constructions = ConstructionLearner()

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
                int(eligible),
                utcnow(),
            ),
        )
        if not eligible:
            return
        # Dialogue questions are linguistic evidence, not world facts.
        source_id = stable_id("dialogue-question", graph.id)
        now = utcnow()
        conn.execute(
            """INSERT OR IGNORE INTO knowledge_sources
               (id,raw_text,normalized_text,content_hash,source_type,status,
                confidence,independent_key,metadata_json,created_at,updated_at)
               VALUES(?,?,?,?, 'dialogue_question','STAGED',.75,?,'{}',?,?)""",
            (
                source_id,
                text,
                " ".join(text.casefold().split()),
                content_hash(text),
                f"dialogue:{hive_id}:{graph.id}",
                now,
                now,
            ),
        )
        structural = ObservationSignature(
            graph.trace.get("structural_signature") or {}
        )
        construction = self.constructions.observe(
            conn,
            structural,
            source_id=source_id,
            domain_key="dialogue",
            gap_kind=graph.gap_node.gap_kind,
        )
        for local_slot_id in local_slot_ids:
            self.constructions.reinforce_binding(
                conn,
                construction.id,
                local_slot_id,
                accepted=True,
            )

    def query(
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
            previous_graph = None
            previous_bindings = ()
        next_turn = int(state.get("turn_index") or 0) + 1
        graph, analysis = self.builder.build(
            normalized_text,
            previous_graph=previous_graph,
            previous_bindings=previous_bindings,
            identity_context=f"{hive_id}:{next_turn}",
            conversation_id=str(row["conversation_id"]),
            turn_index=next_turn,
        )
        search = self.matcher.search(graph)
        selected_bindings = [
            item for item in search.get("selected_bindings", [])
            if isinstance(item, CandidateBinding)
        ]
        selected = selected_bindings[0] if selected_bindings else None
        event = None
        if isinstance(selected, CandidateBinding):
            with self.repository.transaction() as conn:
                event = EventGraphPipeline.load_event(conn, selected.event_id)
        answer = self.responses.plan(graph, search, event=event)
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
        if selected_bindings:
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
            "swarm": swarm_trace,
            "response_plan": answer,
            "validation": answer.get("validation"),
        }
        next_state = {
            "query_graph": graph_dict,
            "selected_bindings": selected_bindings_dict,
            "binding_configuration": binding_configuration,
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
            conn.execute(
                """UPDATE hives SET active_query_graph_id=?,state_json=?,
                   updated_at=? WHERE id=?""",
                (graph.id, encode(next_state), utcnow(), hive_id),
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
        return {
            "message_id": stable_id("dialogue-turn", hive_id, next_turn),
            "query_graph": graph_dict,
            "candidate_bindings": next_state["candidate_bindings"],
            "rejected_events": rejected,
            "selected_bindings": selected_bindings_dict,
            "binding_configuration": binding_configuration,
            "answer": answer,
            "trace": trace,
            "hive": {
                "id": hive_id,
                "conversation_id": str(row["conversation_id"]),
                "max_cells": int(row["max_cells"]),
                "status": answer["status"],
            },
        }
