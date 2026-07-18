"""Application services for the fresh V2.7 graph pipeline."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .event_graph import EventGraphPipeline
from .graph_learning import ConstructionLearner, ObservationSignature
from .graph_models import (
    AnswerStatus,
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
        result = self.events.materialize(*args, **kwargs)
        if result.get("status") == "CONFIRMED":
            result["universe_update"] = self.universes.ingest_source(
                str(result["source_id"])
            )
        return result

    def train(
        self,
        text: str,
        *,
        source_type: str = "document",
        independent_key: str = "",
        domain_key: str = "",
    ) -> Dict[str, Any]:
        return self._materialize_and_project(
            text,
            source_type=source_type,
            independent_key=independent_key,
            domain_key=domain_key,
            force_status=None,
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
            "selected_binding": None,
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

    @staticmethod
    def _previous(
        state: Mapping[str, Any],
    ) -> tuple[Optional[QueryGraph], Optional[CandidateBinding]]:
        graph_value = state.get("query_graph")
        binding_value = state.get("selected_binding")
        return (
            query_graph_from_dict(graph_value) if graph_value else None,
            candidate_binding_from_dict(binding_value),
        )

    def parse(
        self,
        text: str,
        *,
        previous_graph: Optional[QueryGraph] = None,
        previous_binding: Optional[CandidateBinding] = None,
        conversation_id: str = "",
        turn_index: int = 0,
    ) -> Dict[str, Any]:
        graph, analysis = self.builder.build(
            text,
            previous_graph=previous_graph,
            previous_binding=previous_binding,
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
        selected: CandidateBinding,
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
            selected.id,
        )
        local_slot_ids = [
            str(local_slot_id)
            for evidence in selected.evidence
            for local_slot_id in evidence.get("local_slot_ids", [])
        ]
        supporting_event_ids = sorted({
            event_id
            for evidence in selected.evidence
            for event_id in evidence.get("supporting_event_ids", [])
        } or {selected.event_id})
        conn.execute(
            """INSERT OR REPLACE INTO training_episodes
               (id,utterance,query_graph_id,candidate_bindings_json,
                selected_binding_id,event_ids_json,construction_ids_json,
                slot_hypotheses_json,answer_status,validation_json,
                user_correction_json,eligible_for_learning,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,NULL,?,?)""",
            (
                episode_id,
                text,
                graph.id,
                encode([item.as_dict() for item in accepted]),
                selected.id,
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
        previous_graph, previous_binding = self._previous(state)
        if resolved_mode == "NEW_QUERY":
            previous_graph = None
            previous_binding = None
        next_turn = int(state.get("turn_index") or 0) + 1
        graph, analysis = self.builder.build(
            normalized_text,
            previous_graph=previous_graph,
            previous_binding=previous_binding,
            identity_context=f"{hive_id}:{next_turn}",
            conversation_id=str(row["conversation_id"]),
            turn_index=next_turn,
        )
        search = self.matcher.search(graph)
        selected = search.get("selected")
        event = None
        if isinstance(selected, CandidateBinding):
            with self.repository.transaction() as conn:
                event = EventGraphPipeline.load_event(conn, selected.event_id)
        answer = self.responses.plan(graph, search, event=event)
        accepted = list(search.get("accepted") or [])
        rejected = list(search.get("rejected") or [])
        graph_dict = graph.as_dict()
        selected_dict = selected.as_dict() if isinstance(selected, CandidateBinding) else None
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
            "selected_binding": selected_dict,
            "response_plan": answer,
            "validation": answer.get("validation"),
        }
        next_state = {
            "query_graph": graph_dict,
            "selected_binding": selected_dict,
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
            message_id = stable_id("dialogue-turn", hive_id, next_turn)
            conn.execute(
                """INSERT INTO dialogue_turns
                   (id,hive_id,turn_index,speaker,raw_text,query_graph_id,
                    selected_binding_id,answer_json,created_at)
                   VALUES(?,? ,?,'user',?,?,?,?,?)""",
                (
                    message_id,
                    hive_id,
                    next_turn,
                    normalized_text,
                    graph.id,
                    selected.id if isinstance(selected, CandidateBinding) else None,
                    encode(answer),
                    utcnow(),
                ),
            )
            conn.execute(
                """UPDATE hives SET active_query_graph_id=?,state_json=?,
                   updated_at=? WHERE id=?""",
                (graph.id, encode(next_state), utcnow(), hive_id),
            )
            if isinstance(selected, CandidateBinding):
                self._learn_confirmed_episode(
                    conn,
                    hive_id,
                    normalized_text,
                    graph,
                    selected,
                    answer,
                    accepted,
                )
        return {
            "message_id": stable_id("dialogue-turn", hive_id, next_turn),
            "query_graph": graph_dict,
            "candidate_bindings": next_state["candidate_bindings"],
            "rejected_events": rejected,
            "selected_binding": selected_dict,
            "answer": answer,
            "trace": trace,
            "hive": {
                "id": hive_id,
                "conversation_id": str(row["conversation_id"]),
                "max_cells": int(row["max_cells"]),
                "status": answer["status"],
            },
        }
