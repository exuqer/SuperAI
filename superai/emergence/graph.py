"""In-memory active graph for bounded reasoning over a task.

This module implements the ActiveGraph as an in-memory projection of the
canonical Cosmos, with a priority event queue, budget enforcement, and
deterministic replay snapshots.
"""

from __future__ import annotations

import heapq
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..contracts import (
    ActiveGraphSnapshot,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    ResourceBudget,
    TaskContract,
    new_id,
    utcnow,
)
from ..cosmos import Cosmos
from ..database import SqliteDatabase, json_dumps, json_loads
from ..observability import TraceRecorder


@dataclass
class GraphEvent:
    """An event in the active graph processing queue."""
    priority: float  # Lower values = higher priority (processed first)
    sequence: int    # Tiebreaker for FIFO ordering
    kind: str        # activate, propagate, inhibit, expire
    payload: Dict[str, Any]
    node_id: Optional[str] = None

    def __lt__(self, other: "GraphEvent") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.sequence < other.sequence


@dataclass
class ActiveGraph:
    """In-memory active graph for a single task.

    This is a bounded projection of the canonical Cosmos graph.
    It is never persisted as canonical knowledge; only the final
    snapshot and trace are durable.
    """
    task_id: str
    hive_id: str
    tenant_id: str
    project_id: Optional[str]
    trace_id: str
    budget: ResourceBudget
    seed: int
    cosmos_version: str
    traces: TraceRecorder

    # Graph state
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: Dict[str, GraphEdge] = field(default_factory=dict)
    node_to_edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    # Event queue and state
    event_queue: List[GraphEvent] = field(default_factory=list)
    event_sequence: int = 0
    event_count: int = 0

    # Budget tracking
    budget_ledger: Dict[str, int] = field(default_factory=lambda: {
        "steps": 0,
        "wall_time_ms": 0,
        "active_nodes": 0,
        "active_edges": 0,
        "hypotheses_created": 0,
        "experiments_run": 0,
    })

    # Cycle detection
    visited_states: Set[str] = field(default_factory=set)
    fatigue_scores: Dict[str, float] = field(default_factory=dict)

    # Random state for deterministic behavior
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        self.rng.seed(self.seed)

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the active graph."""
        if len(self.nodes) >= self.budget.max_active_nodes:
            self._emit_event("budget_exhausted", {"limit": "max_active_nodes", "value": self.budget.max_active_nodes})
            return
        self.nodes[node.node_id] = node
        self.budget_ledger["active_nodes"] = len(self.nodes)
        self._emit_event("node_activated", {"node_id": node.node_id, "node_type": node.node_type.value})

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the active graph."""
        if len(self.edges) >= self.budget.max_active_edges:
            self._emit_event("budget_exhausted", {"limit": "max_active_edges", "value": self.budget.max_active_edges})
            return
        self.edges[edge.edge_id] = edge
        self.node_to_edges[edge.source_id].add(edge.edge_id)
        self.node_to_edges[edge.target_id].add(edge.edge_id)
        self.budget_ledger["active_edges"] = len(self.edges)
        self._emit_event("edge_traversed", {"edge_id": edge.edge_id, "edge_type": edge.edge_type.value})

    def schedule_event(self, kind: str, payload: Dict[str, Any], priority: float = 1.0, node_id: Optional[str] = None) -> None:
        """Schedule a graph processing event."""
        event = GraphEvent(
            priority=priority,
            sequence=self.event_sequence,
            kind=kind,
            payload=payload,
            node_id=node_id,
        )
        self.event_sequence += 1
        heapq.heappush(self.event_queue, event)

    def _emit_event(self, kind: str, payload: Dict[str, Any]) -> None:
        """Emit a domain event to the trace recorder."""
        self.traces.record_event({
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "kind": kind,
            "producer": "ActiveGraph",
            "payload": payload,
        })

    def check_budget(self) -> bool:
        """Check if budget limits have been exceeded."""
        if self.budget_ledger["steps"] >= self.budget.max_steps:
            self._emit_event("budget_exhausted", {"limit": "max_steps", "value": self.budget.max_steps})
            return False
        if self.budget_ledger["wall_time_ms"] >= self.budget.max_wall_time_ms:
            self._emit_event("budget_exhausted", {"limit": "max_wall_time_ms", "value": self.budget.max_wall_time_ms})
            return False
        if self.budget_ledger["hypotheses_created"] >= self.budget.max_hypotheses:
            self._emit_event("budget_exhausted", {"limit": "max_hypotheses", "value": self.budget.max_hypotheses})
            return False
        if self.budget_ledger["experiments_run"] >= self.budget.max_experiments:
            self._emit_event("budget_exhausted", {"limit": "max_experiments", "value": self.budget.max_experiments})
            return False
        return True

    def compute_state_hash(self) -> str:
        """Compute a hash of the current graph state for cycle detection."""
        node_states = tuple(sorted(
            (nid, round(node.activation, 3), node.status)
            for nid, node in self.nodes.items()
        ))
        return hashlib.sha256(json.dumps(node_states, ensure_ascii=False).encode("utf-8")).hexdigest()

    def detect_cycle(self) -> bool:
        """Detect if the graph state has been seen before (cycle)."""
        state_hash = self.compute_state_hash()
        if state_hash in self.visited_states:
            return True
        self.visited_states.add(state_hash)
        return False

    def apply_fatigue(self, node_id: str, amount: float = 0.1) -> None:
        """Apply fatigue to a node to prevent infinite oscillation."""
        if node_id in self.nodes:
            self.fatigue_scores[node_id] = self.fatigue_scores.get(node_id, 0.0) + amount
            self.nodes[node_id].activation = max(0.0, self.nodes[node_id].activation - amount)
            if self.nodes[node_id].activation < 0.1:
                self.nodes[node_id].status = "inhibited"

    def step(self) -> bool:
        """Process one event from the queue. Returns True if more events exist."""
        if not self.event_queue:
            return False
        if not self.check_budget():
            return False

        event = heapq.heappop(self.event_queue)
        self.event_count += 1
        self.budget_ledger["steps"] += 1

        # Track wall time (simplified - in production use actual time)
        self.budget_ledger["wall_time_ms"] += 1

        # Process event based on kind
        if event.kind == "activate":
            self._process_activate(event)
        elif event.kind == "propagate":
            self._process_propagate(event)
        elif event.kind == "inhibit":
            self._process_inhibit(event)
        elif event.kind == "expire":
            self._process_expire(event)

        # Check for cycles
        if self.detect_cycle():
            self._emit_event("cycle_detected", {"state_hash": self.compute_state_hash()})
            # Apply fatigue to break cycle
            for nid in list(self.nodes.keys()):
                self.apply_fatigue(nid, 0.2)

        return True

    def _process_activate(self, event: GraphEvent) -> None:
        """Process node activation event."""
        node_id = event.payload.get("node_id")
        if node_id and node_id in self.nodes:
            node = self.nodes[node_id]
            node.activation = min(1.0, node.activation + event.payload.get("amount", 0.5))
            node.status = "active"
            self.schedule_event(
                "propagate",
                {"source_id": node_id},
                priority=1.0 - node.activation,
                node_id=node_id,
            )

    def _process_propagate(self, event: GraphEvent) -> None:
        """Process activation propagation along edges."""
        source_id = event.payload.get("source_id")
        if not source_id or source_id not in self.nodes:
            return

        source_node = self.nodes[source_id]
        if source_node.activation < 0.3:
            return  # Too weak to propagate

        # Propagate to neighbors
        for edge_id in self.node_to_edges.get(source_id, set()):
            edge = self.edges.get(edge_id)
            if not edge:
                continue
            target_id = edge.target_id if edge.source_id == source_id else edge.source_id
            if target_id not in self.nodes:
                continue

            target_node = self.nodes[target_id]
            # Activation decays with edge weight and distance
            propagated = source_node.activation * edge.weight * edge.confidence * 0.8
            if propagated > 0.1:
                self.schedule_event("activate", {
                    "node_id": target_id,
                    "amount": propagated,
                    "trace_id": self.task_id,
                }, priority=1.0 - propagated)

    def _process_inhibit(self, event: GraphEvent) -> None:
        """Process node inhibition event."""
        node_id = event.payload.get("node_id")
        if node_id and node_id in self.nodes:
            self.nodes[node_id].status = "inhibited"
            self.nodes[node_id].activation = 0.0

    def _process_expire(self, event: GraphEvent) -> None:
        """Process node/edge expiration event."""
        node_id = event.payload.get("node_id")
        if node_id and node_id in self.nodes:
            self.nodes[node_id].status = "expired"
            # Remove edges
            for edge_id in list(self.node_to_edges.get(node_id, set())):
                edge = self.edges.pop(edge_id, None)
                if edge:
                    self.node_to_edges[edge.source_id].discard(edge_id)
                    self.node_to_edges[edge.target_id].discard(edge_id)

    def build_snapshot(self) -> ActiveGraphSnapshot:
        """Build a snapshot of the current active graph state."""
        return ActiveGraphSnapshot(
            task_id=self.task_id,
            hive_id=self.hive_id,
            tenant_id=self.tenant_id,
            project_id=self.project_id,
            cosmos_version=self.cosmos_version,
            node_ids=list(self.nodes.keys()),
            edge_ids=list(self.edges.keys()),
            frontier=[nid for nid, n in self.nodes.items() if n.status == "active"],
            event_count=self.event_count,
            budget_ledger=dict(self.budget_ledger),
            random_seed=self.seed,
        )

    def persist_snapshot(self, database: SqliteDatabase) -> str:
        """Persist the active graph snapshot to the database."""
        snapshot = self.build_snapshot()
        snapshot_id = new_id("ags")
        with database.transaction() as conn:
            conn.execute(
                """INSERT INTO active_graph_snapshots
                   (snapshot_id, task_id, hive_id, tenant_id, cosmos_version, node_ids_json,
                    edge_ids_json, node_types_json, edge_types_json, frontier_json,
                    event_count, budget_ledger_json, random_seed, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot_id,
                    snapshot.task_id,
                    snapshot.hive_id,
                    snapshot.tenant_id,
                    snapshot.cosmos_version,
                    json_dumps(snapshot.node_ids),
                    json_dumps(snapshot.edge_ids),
                    json_dumps([node.node_type.value for node in self.nodes.values()]),
                    json_dumps([edge.edge_type.value for edge in self.edges.values()]),
                    json_dumps(snapshot.frontier),
                    snapshot.event_count,
                    json_dumps(snapshot.budget_ledger),
                    snapshot.random_seed,
                    utcnow().isoformat(),
                ),
            )
        return snapshot_id


class ActiveGraphBuilder:
    """Builds an ActiveGraph from a TaskContract and Cosmos retrieval."""

    def __init__(self, cosmos: Cosmos, database: SqliteDatabase, traces: TraceRecorder):
        self.cosmos = cosmos
        self.database = database
        self.traces = traces

    def build_initial_graph(
        self,
        contract: TaskContract,
        hive_id: str,
        trace_id: str,
        retrieval: Optional[Any] = None,
        seed: int = 42,
        reserve_steps: int = 0,
    ) -> ActiveGraph:
        """Build the initial active graph from bounded Cosmos retrieval."""
        # Get budget from contract or use defaults
        budget = ResourceBudget(
            max_steps=max(0, contract.budget.step_limit - reserve_steps),
            max_wall_time_ms=contract.budget.time_ms,
            max_active_nodes=50,
            max_active_edges=200,
            max_hypotheses=5,
            max_experiments=10,
            exploration_share=0.3,
        )

        # Get Cosmos version
        cosmos_version = "1.0"  # In practice, read from schema_meta

        graph = ActiveGraph(
            task_id=contract.task_id,
            hive_id=hive_id,
            tenant_id=contract.tenant_id,
            project_id=contract.project_id,
            trace_id=trace_id,
            budget=budget,
            seed=seed,
            cosmos_version=cosmos_version,
            traces=self.traces,
        )

        # Retrieve relevant claims from Cosmos
        retrieval = retrieval or self.cosmos.retrieve(contract, limit=20)

        # Create nodes for retrieved claims
        for item in retrieval.claims:
            node = GraphNode(
                node_type=NodeType.CLAIM,
                content_ref=item.claim.claim_id,
                confidence=item.score,
                activation=item.score,
                provenance_refs=[item.source.artifact_id],
            )
            graph.add_node(node)

            # Also add the source concept as a node
            concept_node = GraphNode(
                node_type=NodeType.CONCEPT,
                content_ref=item.claim.subject_id,
                confidence=0.7,
                activation=0.5,
                provenance_refs=[item.source.artifact_id],
            )
            graph.add_node(concept_node)

            # Add semantic edge between concept and claim
            edge = GraphEdge(
                source_id=concept_node.node_id,
                target_id=node.node_id,
                edge_type=EdgeType.SEMANTIC,
                weight=item.score,
                confidence=item.score,
                provenance_refs=[item.source.artifact_id],
            )
            graph.add_edge(edge)

        # Schedule initial propagation events
        for node_id in graph.nodes:
            graph.schedule_event("activate", {
                "node_id": node_id,
                "amount": graph.nodes[node_id].activation,
                "trace_id": trace_id,
            }, priority=1.0 - graph.nodes[node_id].activation)

        return graph
