"""Deterministic, bounded event discovery for requested QueryGraph GAPs.

Swarms only propose event identifiers.  They never bind participants and they
never override GraphMatcher: graph admission remains a separate strict phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Mapping

import server.database as database
from server.core.settings import settings

from .acceleration import AccelerationRuntime, runtime as acceleration_runtime
from .graph_repository import GraphRepository, decode, encode, stable_id, utcnow


def configured_budget() -> Dict[str, int]:
    return {
        "max_bees": settings.swarm_max_bees,
        "max_rounds": settings.swarm_max_rounds,
        "max_vertical_transitions": settings.swarm_max_vertical_transitions,
        "max_nectar_packets": settings.swarm_max_nectar_packets,
        "max_candidate_events_per_bee": (
            settings.swarm_max_candidate_events_per_bee
        ),
        "max_candidate_events_per_swarm": (
            settings.swarm_max_candidate_events_per_swarm
        ),
        "max_dimension_projections_per_round": (
            settings.swarm_max_dimension_projections_per_round
        ),
        "max_index_hits_per_seed": settings.swarm_max_index_hits_per_seed,
        "max_graph_match_attempts": settings.swarm_max_graph_match_attempts,
    }


@dataclass(frozen=True)
class QueryPlan:
    query_graph_id: str
    requested_gap_ids: tuple[str, ...]
    event_anchor_id: str
    seed_predicates: tuple[str, ...]
    seed_entities: tuple[str, ...]
    known_entity_ids: tuple[str, ...]
    known_qualified_keys: tuple[str, ...]
    required_universes: tuple[str, ...]
    optional_universes: tuple[str, ...]
    enabled_dimensions: tuple[str, ...]
    shadow_dimensions: tuple[str, ...]
    budget: Mapping[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query_graph_id": self.query_graph_id,
            "requested_gap_ids": list(self.requested_gap_ids),
            "event_anchor_id": self.event_anchor_id or None,
            "seed_predicates": list(self.seed_predicates),
            "seed_entities": list(self.seed_entities),
            "known_entity_ids": list(self.known_entity_ids),
            "known_qualified_keys": list(self.known_qualified_keys),
            "required_universes": list(self.required_universes),
            "optional_universes": list(self.optional_universes),
            "enabled_dimensions": list(self.enabled_dimensions),
            "shadow_dimensions": list(self.shadow_dimensions),
            "budgets": dict(self.budget),
            "fallback_policy": "INDEX_FALLBACK_WHEN_NO_EVENT_PROJECTION",
        }


class JointBindingCoordinator:
    """Combine event proposals without ever mixing different events."""

    @staticmethod
    def coordinate(
        proposals_by_gap: Mapping[str, list[str]],
        *,
        limit: int,
    ) -> list[str]:
        proposal_sets = [
            set(event_ids)
            for event_ids in proposals_by_gap.values()
        ]
        if not proposal_sets:
            return []
        common = set.intersection(*proposal_sets)
        if common:
            return sorted(common)[:limit]
        # The matcher can only build a complete configuration from a single
        # event.  An empty intersection therefore has no safe union fallback.
        return []


class GapSwarmCoordinator:
    """Run one deterministic Scout/Worker/Assembly/Observer swarm per GAP."""

    def __init__(self, repository: GraphRepository, *, runtime: AccelerationRuntime | None = None) -> None:
        self.repository = repository
        self.acceleration = runtime or acceleration_runtime
        self.joint = JointBindingCoordinator()

    def _plan(self, graph: Any, conn: Any) -> QueryPlan:
        dimensions = conn.execute(
            """SELECT id,status FROM latent_dimensions
               WHERE status IN ('candidate','probation','active','shared')
               ORDER BY CASE status
                          WHEN 'active' THEN 0
                          WHEN 'shared' THEN 1
                          WHEN 'probation' THEN 2
                          ELSE 3
                        END,id
               LIMIT ?""",
            (
                settings.max_candidate_dimensions_per_universe
                + settings.max_probation_dimensions_per_universe
                + settings.max_active_dimensions_per_universe,
            ),
        ).fetchall()
        active = tuple(
            str(row["id"])
            for row in dimensions
            if str(row["status"]) in {"active", "shared"}
        )
        shadow = tuple(
            str(row["id"])
            for row in dimensions
            if str(row["status"]) in {"candidate", "probation"}
        )
        seed_entities = {
            node.head_lemma.casefold()
            for node in graph.known_nodes
            if node.head_lemma
        }
        if graph.predicate and graph.predicate.lemma:
            seed_entities.add(graph.predicate.lemma.casefold())
        return QueryPlan(
            query_graph_id=graph.id,
            requested_gap_ids=tuple(gap.id for gap in graph.target_gaps),
            event_anchor_id=str(graph.trace.get("event_anchor_id") or ""),
            seed_predicates=(
                (str(graph.predicate.concept_id),)
                if graph.predicate else ()
            ),
            seed_entities=tuple(sorted(seed_entities)),
            known_entity_ids=tuple(
                str(node.entity_id or "")
                for node in graph.known_nodes
            ),
            known_qualified_keys=tuple(
                str(node.qualified_key)
                for node in graph.known_nodes
            ),
            required_universes=("words", "usages", "events"),
            optional_universes=("word_forms", "dimensions"),
            enabled_dimensions=active,
            shadow_dimensions=shadow,
            budget=configured_budget(),
        )

    @staticmethod
    def _known_participant_filter(
        plan: QueryPlan,
        *,
        event_alias: str,
    ) -> tuple[list[str], list[str]]:
        clauses: list[str] = []
        params: list[str] = []
        for entity_id, qualified_key in zip(
            plan.known_entity_ids,
            plan.known_qualified_keys,
        ):
            clauses.append(
                f""" AND EXISTS (
                      SELECT 1
                      FROM graph_participants index_p
                      JOIN graph_mentions index_m
                        ON index_m.id=index_p.mention_id
                      WHERE index_p.event_id={event_alias}.id
                        AND (
                          index_m.entity_id=?
                          OR index_m.qualified_key=?
                        )
                    )"""
            )
            params.extend([entity_id, qualified_key])
        return clauses, params

    @classmethod
    def _index_event_ids(
        cls,
        conn: Any,
        plan: QueryPlan,
        *,
        constrain_known_nodes: bool = True,
    ) -> list[str]:
        if plan.event_anchor_id:
            row = conn.execute(
                """SELECT e.id FROM graph_events e
                   JOIN knowledge_sources s ON s.id=e.source_id
                   WHERE e.id=? AND e.actuality='ACTUAL'
                     AND s.status='CONFIRMED'""",
                (plan.event_anchor_id,),
            ).fetchone()
            return [str(row["id"])] if row else []
        if not plan.seed_predicates:
            return []
        known_clauses, known_params = (
            cls._known_participant_filter(plan, event_alias="e")
            if constrain_known_nodes else
            ([], [])
        )
        rows = conn.execute(
            """SELECT e.id FROM graph_events e
               JOIN knowledge_sources s ON s.id=e.source_id
               WHERE e.predicate_concept_id=? AND e.actuality='ACTUAL'
                 AND s.status='CONFIRMED'
               """ + "".join(known_clauses) + """
               ORDER BY e.confidence DESC,e.created_at,e.id LIMIT ?""",
            (
                plan.seed_predicates[0],
                *known_params,
                int(plan.budget["max_index_hits_per_seed"]),
            ),
        ).fetchall()
        return [str(row["id"]) for row in rows]

    @classmethod
    def _filter_dimension_events(
        cls,
        conn: Any,
        plan: QueryPlan,
        event_ids: list[str],
    ) -> list[str]:
        """Apply observable query constraints to dimension-produced events."""
        if not event_ids or not plan.known_entity_ids:
            return event_ids
        known_clauses, known_params = cls._known_participant_filter(
            plan,
            event_alias="e",
        )
        rows = conn.execute(
            """SELECT e.id FROM graph_events e
               JOIN knowledge_sources s ON s.id=e.source_id
               WHERE e.id IN ({})
                 AND e.actuality='ACTUAL'
                 AND s.status='CONFIRMED'
               {}
               ORDER BY e.confidence DESC,e.created_at,e.id""".format(
                ",".join("?" for _ in event_ids),
                "".join(known_clauses),
            ),
            [*event_ids, *known_params],
        ).fetchall()
        return [str(row["id"]) for row in rows]

    @staticmethod
    def _word_entity_ids(lemmas: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            stable_id("universe-entity", "words", f"ru:{lemma}:")
            for lemma in lemmas
        )

    def _dimension_events(
        self,
        conn: Any,
        plan: QueryPlan,
        dimension_ids: tuple[str, ...],
    ) -> tuple[list[str], Dict[str, list[str]], int]:
        if not dimension_ids or not plan.seed_entities:
            return [], {}, 0
        entity_ids = self._word_entity_ids(plan.seed_entities)
        seed_rows = conn.execute(
            """SELECT DISTINCT dimension_id FROM projections
               WHERE source_type='entity' AND membership>=0.30
                 AND source_id IN ({})
                 AND dimension_id IN ({})
               ORDER BY dimension_id LIMIT ?""".format(
                ",".join("?" for _ in entity_ids),
                ",".join("?" for _ in dimension_ids),
            ),
            [
                *entity_ids,
                *dimension_ids,
                int(plan.budget["max_dimension_projections_per_round"]),
            ],
        ).fetchall()
        relevant = tuple(str(row["dimension_id"]) for row in seed_rows)
        if not relevant:
            return [], {}, len(seed_rows)
        rows = conn.execute(
            """SELECT DISTINCT p.dimension_id,e.id AS event_id
               FROM projections p
               JOIN universe_occurrences o
                 ON o.id=p.source_id AND p.source_type='occurrence'
               JOIN graph_events e ON e.source_id=o.source_id
               JOIN knowledge_sources s ON s.id=e.source_id
               WHERE p.dimension_id IN ({})
                 AND p.membership>=0.30
                 AND e.actuality='ACTUAL'
                 AND s.status='CONFIRMED'
               ORDER BY p.dimension_id,e.confidence DESC,e.id
               LIMIT ?""".format(",".join("?" for _ in relevant)),
            [
                *relevant,
                int(plan.budget["max_candidate_events_per_swarm"]),
            ],
        ).fetchall()
        by_dimension: Dict[str, list[str]] = {
            dimension_id: [] for dimension_id in relevant
        }
        for row in rows:
            event_id = str(row["event_id"])
            values = by_dimension[str(row["dimension_id"])]
            if event_id not in values:
                values.append(event_id)
        combined = sorted({
            event_id
            for event_ids in by_dimension.values()
            for event_id in event_ids
        })
        return combined, by_dimension, len(seed_rows) + len(rows)

    def _warm_projection_index(
        self,
        conn: Any,
        plan: QueryPlan,
        projection_revision: str,
    ) -> tuple[str, float]:
        """Build/query the exact vector cache without changing SQL priority.

        Predicate and known-node SQL hits continue to determine the candidate
        order.  The vector probe is a bounded supplemental signal retained in
        memory for subsequent swarm rounds, never a GraphMatcher shortcut.
        """
        dimensions = plan.enabled_dimensions
        if not dimensions or not plan.seed_entities:
            return "numpy", 0.0
        index = self.acceleration.projection_index(
            database.get_db_path(), projection_revision, dimensions
        )
        if not index.size:
            rows = conn.execute(
                """SELECT source_id,dimension_id,membership FROM projections
                   WHERE source_type='entity' AND dimension_id IN ({})
                   ORDER BY source_id,dimension_id""".format(
                    ",".join("?" for _ in dimensions)
                ),
                list(dimensions),
            ).fetchall()
            vectors: Dict[str, Dict[str, float]] = {}
            for row in rows:
                vectors.setdefault(str(row["source_id"]), {})[
                    str(row["dimension_id"])
                ] = float(row["membership"])
            index.rebuild(vectors.items())
        # The query warms the exact search path and validates malformed seed
        # vectors are ignored.  Results intentionally do not reorder the SQL
        # candidate list below.
        index.search({dimension: 1.0 for dimension in dimensions}, 1)
        return index.backend, index.build_ms

    @staticmethod
    def _mission(
        run_id: str,
        bee_type: str,
        mission_type: str,
        index: int,
        *,
        seed: Mapping[str, Any],
        visited: list[str],
        event_ids: list[str],
        termination_reason: str,
    ) -> Dict[str, Any]:
        return {
            "bee_id": stable_id(
                "bee-mission", run_id, bee_type.casefold(), index
            ),
            "bee_type": bee_type,
            "mission_type": mission_type,
            "seed": dict(seed),
            "visited_universes": visited,
            "candidate_event_ids": event_ids,
            "successful": bool(event_ids),
            "termination_reason": termination_reason,
        }

    def _persist_run(
        self,
        conn: Any,
        graph: Any,
        gap_id: str,
        plan: QueryPlan,
        trace: Mapping[str, Any],
        now: str,
    ) -> None:
        run_id = str(trace["id"])
        conn.execute(
            """INSERT OR REPLACE INTO swarm_runs
               (id,query_graph_id,gap_id,deterministic_seed,status,
                termination_reason,retrieval_mode,budget_json,trace_json,
                started_at,completed_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                graph.id,
                gap_id,
                stable_id("swarm-seed", graph.id, gap_id),
                trace["status"],
                trace["termination_reason"],
                trace["retrieval_mode"],
                encode(dict(plan.budget)),
                encode(dict(trace)),
                now,
                now,
            ),
        )
        for mission in trace["missions"]:
            conn.execute(
                """INSERT OR REPLACE INTO bee_missions
                   (id,swarm_run_id,bee_type,mission_type,seed_json,
                    visited_universes_json,candidate_event_ids_json,
                    successful,termination_reason)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    mission["bee_id"],
                    run_id,
                    mission["bee_type"],
                    mission["mission_type"],
                    encode(mission["seed"]),
                    encode(mission["visited_universes"]),
                    encode(mission["candidate_event_ids"]),
                    int(mission["successful"]),
                    mission["termination_reason"],
                ),
            )
            for step_index, (source, target) in enumerate(zip(
                mission["visited_universes"],
                mission["visited_universes"][1:],
            )):
                conn.execute(
                    """INSERT OR REPLACE INTO bee_steps
                       (id,swarm_run_id,bee_id,step_index,source_universe,
                        target_universe,action,evidence_json,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        stable_id(
                            "bee-step",
                            mission["bee_id"],
                            step_index,
                            source,
                            target,
                        ),
                        run_id,
                        mission["bee_id"],
                        step_index,
                        source,
                        target,
                        mission["mission_type"],
                        encode({
                            "candidate_event_ids": (
                                mission["candidate_event_ids"]
                            ),
                        }),
                        now,
                    ),
                )
        for packet in trace["nectar_packets"]:
            conn.execute(
                """INSERT OR REPLACE INTO nectar_packets
                   (id,swarm_run_id,source_universe,target_universe,
                    event_ids_json,dimension_ids_json,evidence_weight,
                    provenance_json)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    packet["packet_id"],
                    run_id,
                    packet["source_universe"],
                    packet["target_universe"],
                    encode(packet["event_ids"]),
                    encode(packet["dimension_ids"]),
                    packet["evidence_weight"],
                    encode(packet["provenance"]),
                ),
            )
        for event_id in trace["candidate_event_ids"]:
            conn.execute(
                """INSERT OR REPLACE INTO candidate_event_observations
                   (id,swarm_run_id,event_id,source_mode,evidence_weight,
                    admitted,rejection_reason,created_at)
                   VALUES(?,?,?,?,?,NULL,NULL,?)""",
                (
                    stable_id(
                        "candidate-event-observation", run_id, event_id
                    ),
                    run_id,
                    event_id,
                    trace["retrieval_mode"],
                    1.0,
                    now,
                ),
            )

    @staticmethod
    def _update_universe_visits(
        conn: Any,
        plan: QueryPlan,
        *,
        run_id: str,
        bee_count: int,
        successful_bee_count: int,
        retrieval_mode: str,
    ) -> None:
        visited = set(plan.required_universes)
        for row in conn.execute(
            "SELECT id,statistics_json FROM universes ORDER BY id"
        ).fetchall():
            universe_id = str(row["id"])
            statistics = decode(row["statistics_json"], {})
            was_visited = universe_id in visited
            statistics.update({
                "last_swarm_run_id": run_id if was_visited else None,
                "visited_in_last_query": was_visited,
                "last_query_bee_count": bee_count if was_visited else 0,
                "active_bee_count": 0,
                "successful_bee_count": (
                    successful_bee_count if was_visited else 0
                ),
                "terminated_bee_count": bee_count if was_visited else 0,
                "fallback_bee_count": (
                    bee_count
                    if was_visited and retrieval_mode == "INDEX_FALLBACK"
                    else 0
                ),
                "retrieval_mode": retrieval_mode if was_visited else None,
            })
            conn.execute(
                """UPDATE universes SET statistics_json=?,updated_at=?
                   WHERE id=?""",
                (encode(statistics), utcnow(), universe_id),
            )

    def discover(self, graph: Any) -> Dict[str, Any]:
        """Return candidate events plus a persisted, auditable swarm trace."""
        if not graph.predicate or not graph.target_gaps:
            return {
                "query_plan": {},
                "gap_swarms": [],
                "candidate_event_ids": [],
                "retrieval_mode": "DIRECT_EVENT_LOOKUP",
                "fallback_reason": "NO_SEEDS",
            }
        started = perf_counter()
        now = utcnow()
        with self.repository.transaction() as conn:
            plan = self._plan(graph, conn)
            meta = {
                str(row["key"]): str(row["value"])
                for row in conn.execute(
                    "SELECT key,value FROM graph_meta WHERE key IN ('projection_revision','transition_revision')"
                ).fetchall()
            }
            route_index = self.acceleration.route_index(
                database.get_db_path(), meta.get("transition_revision", "0")
            )
            if not route_index.adjacency:
                route_index.rebuild(
                    conn.execute(
                        "SELECT id,source_id,target_id,weight FROM universe_transitions"
                    ).fetchall()
                )
            route_neighbours = route_index.expand(
                self._word_entity_ids(plan.seed_entities),
                budget=int(plan.budget["max_vertical_transitions"]),
            )
            vector_backend, index_build_ms = self._warm_projection_index(
                conn, plan, meta.get("projection_revision", "0")
            )
            index_event_ids = self._index_event_ids(conn, plan)
            retrieval_baseline_event_ids = self._index_event_ids(
                conn,
                plan,
                constrain_known_nodes=False,
            )
            dimensional_event_ids, active_by_dimension, active_reads = (
                self._dimension_events(
                    conn, plan, plan.enabled_dimensions
                )
            )
            shadow_event_ids, shadow_by_dimension, shadow_reads = (
                self._dimension_events(
                    conn, plan, plan.shadow_dimensions
                )
            )
            dimensional_event_ids = self._filter_dimension_events(
                conn,
                plan,
                dimensional_event_ids,
            )
            shadow_event_ids = self._filter_dimension_events(
                conn,
                plan,
                shadow_event_ids,
            )
            active_by_dimension = {
                dimension_id: self._filter_dimension_events(
                    conn,
                    plan,
                    event_ids,
                )
                for dimension_id, event_ids in active_by_dimension.items()
            }
            shadow_by_dimension = {
                dimension_id: self._filter_dimension_events(
                    conn,
                    plan,
                    event_ids,
                )
                for dimension_id, event_ids in shadow_by_dimension.items()
            }
            if plan.event_anchor_id:
                candidate_event_ids = index_event_ids
                retrieval_mode = "DIRECT_EVENT_LOOKUP"
                fallback_reason = ""
            elif dimensional_event_ids and set(index_event_ids).issubset(
                dimensional_event_ids
            ):
                candidate_event_ids = dimensional_event_ids
                retrieval_mode = "SWARM_DIMENSIONAL"
                fallback_reason = ""
            elif dimensional_event_ids:
                candidate_event_ids = sorted(
                    set(dimensional_event_ids) | set(index_event_ids)
                )
                retrieval_mode = "SWARM_MIXED"
                fallback_reason = ""
            else:
                candidate_event_ids = index_event_ids
                retrieval_mode = "INDEX_FALLBACK"
                fallback_reason = (
                    "NO_ACTIVE_DIMENSIONS"
                    if not plan.enabled_dimensions
                    else "NO_EVENT_DIMENSION_PROJECTION"
                )
            candidate_event_ids = candidate_event_ids[
                :int(plan.budget["max_candidate_events_per_swarm"])
            ]
            proposals_by_gap: Dict[str, list[str]] = {}
            swarms: list[Dict[str, Any]] = []
            for gap_id in plan.requested_gap_ids:
                run_id = stable_id("swarm-run", graph.id, gap_id)
                termination = (
                    "GRAPH_ADMITTED_RESULT"
                    if candidate_event_ids
                    and retrieval_mode == "SWARM_DIMENSIONAL"
                    else "STABLE_CANDIDATES"
                    if candidate_event_ids
                    and retrieval_mode == "SWARM_MIXED"
                    else "INDEX_FALLBACK_COMPLETED"
                    if candidate_event_ids
                    else "NO_CANDIDATES"
                )
                missions = [
                    self._mission(
                        run_id,
                        "Scout",
                        "seed_projection_lookup",
                        0,
                        seed={
                            "predicate_concept_ids": list(
                                plan.seed_predicates
                            ),
                            "entity_lemmas": list(plan.seed_entities),
                        },
                        visited=["words"],
                        event_ids=[],
                        termination_reason=termination,
                    ),
                    self._mission(
                        run_id,
                        "Worker",
                        "dimension_neighbour_expansion",
                        1,
                        seed={
                            "dimension_ids": list(plan.enabled_dimensions),
                        },
                        visited=["words", "usages"],
                        event_ids=dimensional_event_ids,
                        termination_reason=termination,
                    ),
                    self._mission(
                        run_id,
                        "Assembly",
                        "vertical_evidence_transfer",
                        2,
                        seed={"route": ["words", "usages", "events"]},
                        visited=["words", "usages", "events"],
                        event_ids=candidate_event_ids,
                        termination_reason=termination,
                    ),
                    self._mission(
                        run_id,
                        "Observer",
                        "deduplicate_candidate_events",
                        3,
                        seed={"gap_id": gap_id},
                        visited=["events"],
                        event_ids=candidate_event_ids,
                        termination_reason=termination,
                    ),
                ][:int(plan.budget["max_bees"])]
                packets = []
                if candidate_event_ids:
                    packets.append({
                        "packet_id": stable_id(
                            "nectar", run_id, "events"
                        ),
                        "source_universe": "words",
                        "target_universe": "events",
                        "event_ids": candidate_event_ids,
                        "dimension_ids": (
                            list(active_by_dimension)
                            if retrieval_mode in {
                                "SWARM_DIMENSIONAL", "SWARM_MIXED",
                            }
                            else []
                        ),
                        "evidence_weight": 1.0,
                        "provenance": {
                            "source": retrieval_mode,
                            "seed_predicates": list(plan.seed_predicates),
                        },
                    })
                trace = {
                    "id": run_id,
                    "gap_id": gap_id,
                    "status": "COMPLETED",
                    "termination_reason": termination,
                    "bee_count": len(missions),
                    "active_bee_count": 0,
                    "successful_bee_count": sum(
                        mission["successful"] for mission in missions
                    ),
                    "round_count": min(
                        2, int(plan.budget["max_rounds"])
                    ),
                    "packet_count": len(packets),
                    "transition_count": sum(
                        max(0, len(mission["visited_universes"]) - 1)
                        for mission in missions
                    ),
                    "route_neighbour_count": len(route_neighbours),
                    "route_backend": route_index.backend,
                    "events_considered": len(candidate_event_ids),
                    "events_returned": len(candidate_event_ids),
                    "retrieval_mode": retrieval_mode,
                    "fallback_reason": fallback_reason,
                    "dimension_evidence_ratio": (
                        1.0 if retrieval_mode == "SWARM_DIMENSIONAL"
                        else len(dimensional_event_ids)
                        / max(1, len(candidate_event_ids))
                        if retrieval_mode == "SWARM_MIXED"
                        else 0.0
                    ),
                    "index_evidence_ratio": (
                        1.0 if retrieval_mode == "INDEX_FALLBACK"
                        else len(index_event_ids)
                        / max(1, len(candidate_event_ids))
                        if retrieval_mode == "SWARM_MIXED"
                        else 0.0
                    ),
                    "active_dimensions_used": (
                        list(active_by_dimension)
                        if retrieval_mode in {
                            "SWARM_DIMENSIONAL", "SWARM_MIXED",
                        }
                        else []
                    ),
                    "candidate_dimensions_shadowed": list(
                        shadow_by_dimension
                    ),
                    "candidate_event_ids": candidate_event_ids,
                    "missions": missions,
                    "nectar_packets": packets,
                }
                self._persist_run(
                    conn, graph, gap_id, plan, trace, now
                )
                proposals_by_gap[gap_id] = candidate_event_ids
                swarms.append(trace)
            coordinated = self.joint.coordinate(
                proposals_by_gap,
                limit=int(plan.budget["max_candidate_events_per_swarm"]),
            )
            last_run_id = str(swarms[-1]["id"]) if swarms else ""
            self._update_universe_visits(
                conn,
                plan,
                run_id=last_run_id,
                bee_count=sum(run["bee_count"] for run in swarms),
                successful_bee_count=sum(
                    run["successful_bee_count"] for run in swarms
                ),
                retrieval_mode=retrieval_mode,
            )
        elapsed_ms = round((perf_counter() - started) * 1000.0, 3)
        return {
            "query_plan": plan.as_dict(),
            "gap_swarms": swarms,
            "candidate_event_ids": coordinated,
            "retrieval_mode": retrieval_mode,
            "fallback_reason": fallback_reason,
            "active_dimension_events": active_by_dimension,
            "shadow_dimension_events": shadow_by_dimension,
            "shadow_candidate_event_ids": shadow_event_ids,
            "retrieval_baseline_event_ids": retrieval_baseline_event_ids,
            "metrics": {
                "events_total": len(retrieval_baseline_event_ids),
                "events_indexed": len(index_event_ids),
                "events_scanned": len(candidate_event_ids),
                "index_hits": len(index_event_ids),
                "candidate_events": len(coordinated),
                "dimension_projections_read": active_reads + shadow_reads,
                "nectar_packets_created": sum(
                    run["packet_count"] for run in swarms
                ),
                "bee_steps": sum(
                    run["transition_count"] for run in swarms
                ),
                "universe_transitions": sum(
                    run["transition_count"] for run in swarms
                ),
                "graph_match_attempts": 0,
                "database_queries": 5,
                "elapsed_ms": elapsed_ms,
                "route_neighbour_count": len(route_neighbours),
                "route_backend": route_index.backend,
                "vector_backend": vector_backend,
                "index_build_ms": round(index_build_ms, 3),
            },
        }

    def record_outcome(
        self,
        graph: Any,
        swarm: Mapping[str, Any],
        *,
        admitted_event_ids: set[str],
        selected_event_ids: set[str],
        validated: bool = False,
    ) -> None:
        """Attach strict GraphMatcher/answer outcomes to retrieval evidence."""
        now = utcnow()
        active_by_dimension = dict(
            swarm.get("active_dimension_events") or {}
        )
        shadow_by_dimension = dict(
            swarm.get("shadow_dimension_events") or {}
        )
        baseline = list(
            swarm.get("retrieval_baseline_event_ids")
            or swarm.get("candidate_event_ids")
            or []
        )
        run_ids = [
            str(run.get("id") or "")
            for run in swarm.get("gap_swarms") or []
            if run.get("id")
        ]
        with self.repository.transaction() as conn:
            for run_id in run_ids:
                rows = conn.execute(
                    """SELECT id,event_id FROM candidate_event_observations
                       WHERE swarm_run_id=?""",
                    (run_id,),
                ).fetchall()
                for row in rows:
                    event_id = str(row["event_id"])
                    conn.execute(
                        """UPDATE candidate_event_observations
                           SET admitted=?,rejection_reason=?
                           WHERE id=?""",
                        (
                            int(event_id in admitted_event_ids),
                            None
                            if event_id in admitted_event_ids
                            else "GRAPH_MATCHER_REJECTED",
                            row["id"],
                        ),
                    )
            for dimension_id, event_ids in active_by_dimension.items():
                contributed = bool(set(event_ids) & admitted_event_ids)
                answered = bool(set(event_ids) & selected_event_ids)
                conn.execute(
                    """UPDATE latent_dimensions SET
                       projection_usage_count=projection_usage_count+1,
                       retrieval_contribution_count=
                         retrieval_contribution_count+?,
                       graph_admitted_contribution_count=
                         graph_admitted_contribution_count+?,
                       validated_answer_contribution_count=
                         validated_answer_contribution_count+?,
                       usage_count=usage_count+1,last_updated_at=?
                       WHERE id=?""",
                    (
                        int(contributed),
                        int(contributed),
                        int(answered and validated),
                        now,
                        dimension_id,
                    ),
                )
            for dimension_id, event_ids in shadow_by_dimension.items():
                shadow_set = set(event_ids)
                correct = shadow_set & selected_event_ids
                gain = (
                    max(
                        0.0,
                        (len(baseline) - len(shadow_set))
                        / max(1, len(baseline)),
                    )
                    if correct else 0.0
                )
                shadow_admitted = sorted(
                    shadow_set & admitted_event_ids
                )
                conn.execute(
                    """INSERT OR REPLACE INTO shadow_retrieval_runs
                       (id,query_graph_id,dimension_id,
                        baseline_event_ids_json,
                        shadow_candidate_events_json,
                        shadow_graph_admitted_events_json,
                        shadow_correct_event_rank,shadow_retrieval_gain,
                        shadow_false_positive_count,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        stable_id(
                            "shadow-retrieval",
                            graph.id,
                            dimension_id,
                        ),
                        graph.id,
                        dimension_id,
                        encode(baseline),
                        encode(event_ids),
                        encode(shadow_admitted),
                        1 if correct else None,
                        gain,
                        len(shadow_set - admitted_event_ids),
                        now,
                    ),
                )
                conn.execute(
                    """UPDATE latent_dimensions SET
                       shadow_retrieval_gain=MAX(
                         shadow_retrieval_gain,?
                       ),
                       holdout_retrieval_gain=MAX(
                         holdout_retrieval_gain,?
                       ),
                       last_updated_at=?
                       WHERE id=?""",
                    (gain, gain, now, dimension_id),
                )
                support = conn.execute(
                    """SELECT entity_support,source_support,domain_support,
                              stability,stability_lower_bound
                       FROM latent_dimensions WHERE id=?""",
                    (dimension_id,),
                ).fetchone()
                if support:
                    conn.execute(
                        """INSERT OR REPLACE INTO dimension_evaluations
                           (id,dimension_id,dataset_split,entity_support,
                            source_support,domain_support,
                            stability_point_estimate,
                            stability_lower_bound,retrieval_gain,
                            metrics_json,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            stable_id(
                                "dimension-evaluation",
                                graph.id,
                                dimension_id,
                            ),
                            dimension_id,
                            "shadow",
                            support["entity_support"],
                            support["source_support"],
                            support["domain_support"],
                            support["stability"],
                            support["stability_lower_bound"],
                            gain,
                            encode({
                                "selected_event_ids": sorted(
                                    selected_event_ids
                                ),
                                "shadow_candidate_event_ids": event_ids,
                                "false_positive_count": len(
                                    shadow_set - admitted_event_ids
                                ),
                            }),
                            now,
                        ),
                    )
                row = conn.execute(
                    """SELECT status,stability,stability_lower_bound,
                              entity_support,source_support,domain_support,
                              holdout_retrieval_gain
                       FROM latent_dimensions WHERE id=?""",
                    (dimension_id,),
                ).fetchone()
                if row and str(row["status"]) == "probation" and (
                    float(row["stability"])
                    >= settings.dimension_minimum_stability
                    and float(row["stability_lower_bound"])
                    >= settings.dimension_minimum_stability_lower_bound
                    and int(row["entity_support"])
                    >= settings.dimension_minimum_entity_support
                    and int(row["source_support"])
                    >= settings.dimension_minimum_source_support
                    and int(row["domain_support"])
                    >= settings.dimension_minimum_domain_support
                    and float(row["holdout_retrieval_gain"]) > 0
                ):
                    conn.execute(
                        """UPDATE latent_dimensions SET status='active',
                           activated_at=COALESCE(activated_at,?),
                           last_updated_at=? WHERE id=?""",
                        (now, now, dimension_id),
                    )
