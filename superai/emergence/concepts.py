"""Concept genesis: creating and validating new internal abstractions from repeated patterns."""

from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ..contracts import (
    AccessScope,
    ArtifactRef,
    ConceptCandidate,
    ConceptCandidateState,
    ConceptEvaluation,
    HypothesisRecord,
    SkillManifest,
    SkillState,
    TaskContract,
    new_id,
    utcnow,
)
from ..cosmos import Cosmos
from ..database import SqliteDatabase, json_dumps, json_loads
from ..learning import ExperienceCompiler
from ..observability import TraceRecorder


@dataclass
class SubgraphPattern:
    """A recurring subgraph pattern found in successful traces."""
    pattern_id: str
    node_types: List[str]  # Sequence of node types
    edge_types: List[str]  # Sequence of edge types
    frequency: int
    task_ids: List[str]
    compression_gain: float
    prediction_gain: float
    transfer_gain: float
    maintenance_cost: float


class SubgraphMiner:
    """Finds recurring subgraphs in successful task traces."""

    def __init__(
        self,
        database: SqliteDatabase,
        traces: TraceRecorder,
    ):
        self.database = database
        self.traces = traces

    def mine_patterns(
        self,
        tenant_id: str,
        min_frequency: int = 2,
        min_tasks: int = 2,
    ) -> List[SubgraphPattern]:
        """Find recurring subgraph patterns across successful tasks."""
        # Get successful task traces
        task_rows = self.database.all(
            """SELECT task_id, trace_id FROM tasks
               WHERE tenant_id = ? AND status = 'succeeded'""",
            (tenant_id,),
        )

        if len(task_rows) < min_tasks:
            return []

        # For each task, extract the active graph snapshot
        task_graphs = {}
        for row in task_rows:
            snapshot = self._get_active_graph_snapshot(row["task_id"])
            if snapshot:
                task_graphs[row["task_id"]] = snapshot

        # Find common subgraphs
        patterns = self._find_common_subgraphs(task_graphs, min_frequency)

        # Score patterns
        scored_patterns = []
        for pattern in patterns:
            scored = self._score_pattern(pattern, task_graphs)
            if scored.frequency >= min_frequency and len(scored.task_ids) >= min_tasks:
                scored_patterns.append(scored)

        return sorted(scored_patterns, key=lambda p: p.compression_gain + p.prediction_gain + p.transfer_gain - p.maintenance_cost, reverse=True)

    def _get_active_graph_snapshot(self, task_id: str) -> Optional[Dict]:
        """Retrieve the active graph snapshot for a task."""
        row = self.database.one(
            "SELECT * FROM active_graph_snapshots WHERE task_id = ? ORDER BY created_at DESC LIMIT 1",
            (task_id,),
        )
        if not row:
            return None
        return {
            "node_ids": json_loads(row["node_ids_json"], []),
            "edge_ids": json_loads(row["edge_ids_json"], []),
            "node_types": json_loads(row.get("node_types_json"), []),
            "edge_types": json_loads(row.get("edge_types_json"), []),
            "frontier": json_loads(row["frontier_json"], []),
        }

    def _find_common_subgraphs(
        self,
        task_graphs: Dict[str, Dict],
        min_frequency: int,
    ) -> List[SubgraphPattern]:
        """Find subgraphs that appear in multiple tasks."""
        # This is a simplified version - in practice would use graph mining algorithms
        # For MVP, we look for common sequences of node/edge types

        # Extract "signatures" from each graph
        signatures: Dict[str, List[str]] = {}
        for task_id, graph in task_graphs.items():
            # Get node types for this graph
            node_types = graph["node_types"]
            edge_types = graph["edge_types"]
            signatures[task_id] = {
                "nodes": node_types,
                "edges": edge_types,
            }

        # Find common subsequences (simplified)
        patterns = []
        all_node_seqs = [sig["nodes"] for sig in signatures.values()]

        # A claim-to-concept pair is the smallest meaningful active subgraph.
        for length in range(2, 6):
            subseqs: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
            for task_id, seq in signatures.items():
                for i in range(len(seq["nodes"]) - length + 1):
                    subseq = tuple(seq["nodes"][i:i+length])
                    subseqs[subseq].add(task_id)

            for subseq, tasks in subseqs.items():
                if len(tasks) >= min_frequency:
                    pattern = SubgraphPattern(
                        pattern_id=new_id("pat"),
                        node_types=list(subseq),
                        edge_types=[],  # Simplified
                        frequency=len(tasks),
                        task_ids=list(tasks),
                        compression_gain=0.0,
                        prediction_gain=0.0,
                        transfer_gain=0.0,
                        maintenance_cost=0.0,
                    )
                    patterns.append(pattern)

        return patterns

    def _get_node_types(self, node_ids: List[str]) -> List[str]:
        """Get node types for a list of node IDs."""
        if not node_ids:
            return []
        placeholders = ",".join("?" for _ in node_ids)
        rows = self.database.all(
            f"SELECT node_type FROM graph_nodes WHERE node_id IN ({placeholders})",
            node_ids,
        )
        return [row["node_type"] for row in rows]

    def _get_edge_types(self, edge_ids: List[str]) -> List[str]:
        """Get edge types for a list of edge IDs."""
        if not edge_ids:
            return []
        placeholders = ",".join("?" for _ in edge_ids)
        rows = self.database.all(
            f"SELECT edge_type FROM graph_edges WHERE edge_id IN ({placeholders})",
            edge_ids,
        )
        return [row["edge_type"] for row in rows]

    def _score_pattern(
        self,
        pattern: SubgraphPattern,
        task_graphs: Dict[str, Dict],
    ) -> SubgraphPattern:
        """Score a pattern on compression, prediction, transfer, and cost."""
        # Compression gain: how much the pattern compresses the graph
        total_nodes = sum(len(g["node_ids"]) for g in task_graphs.values())
        pattern_nodes = len(pattern.node_types) * pattern.frequency
        pattern.compression_gain = pattern_nodes / max(1, total_nodes)

        # Prediction gain: does the pattern predict outcomes?
        # Simplified: assume patterns that appear in diverse tasks have higher prediction gain
        task_goals = []
        for task_id in pattern.task_ids:
            row = self.database.one("SELECT contract_json FROM tasks WHERE task_id = ?", (task_id,))
            if row:
                task_goals.append(json_loads(row["contract_json"], {}).get("goal", "")[:20])
        unique_task_types = len(set(task_goals))
        pattern.prediction_gain = min(1.0, unique_task_types / 5.0)

        # Transfer gain: does it appear across different task classes?
        pattern.transfer_gain = pattern.prediction_gain * 0.8

        # Maintenance cost: complexity of the pattern
        pattern.maintenance_cost = len(pattern.node_types) * 0.05 + len(pattern.edge_types) * 0.02

        return pattern


class ConceptGenerator:
    """Generates ConceptCandidates from recurring patterns."""

    def __init__(
        self,
        cosmos: Cosmos,
        database: SqliteDatabase,
        traces: TraceRecorder,
        miner: SubgraphMiner,
    ):
        self.cosmos = cosmos
        self.database = database
        self.traces = traces
        self.miner = miner

    def generate_candidates(
        self,
        tenant_id: str,
        holdout_manifest_ref: str,
        project_id: Optional[str] = None,
    ) -> List[ConceptCandidate]:
        """Generate concept candidates from mined patterns."""
        patterns = self.miner.mine_patterns(tenant_id)

        candidates = []
        for pattern in patterns[:10]:  # Limit to top 10
            candidate = self._create_candidate(pattern, holdout_manifest_ref, tenant_id, project_id)
            if candidate:
                candidates.append(candidate)

        return candidates

    def _create_candidate(
        self,
        pattern: SubgraphPattern,
        holdout_manifest_ref: str,
        tenant_id: str,
        project_id: Optional[str],
    ) -> Optional[ConceptCandidate]:
        """Create a ConceptCandidate from a pattern."""
        # Generate a name from the pattern
        name = f"Pattern_{'_'.join(pattern.node_types[:3])}_{pattern.pattern_id[:8]}"

        # Create definition artifact
        definition = {
            "pattern_id": pattern.pattern_id,
            "node_types": pattern.node_types,
            "edge_types": pattern.edge_types,
            "frequency": pattern.frequency,
            "source_task_ids": pattern.task_ids,
            "scores": {
                "compression_gain": pattern.compression_gain,
                "prediction_gain": pattern.prediction_gain,
                "transfer_gain": pattern.transfer_gain,
                "maintenance_cost": pattern.maintenance_cost,
            },
        }

        artifact = self.cosmos.store.put_json(
            definition,
            tenant_id=tenant_id,
            schema_name="ConceptDefinition",
            access_scope=AccessScope(
                tenant_id=tenant_id,
                project_id=project_id,
                visibility="project" if project_id else "tenant",
            ),
            idempotency_key=f"concept_def:{pattern.pattern_id}",
        )

        # Extract positive/negative examples from tasks
        positive_examples = []
        negative_examples = []

        for task_id in pattern.task_ids:
            task = self.database.one("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            if task:
                positive_examples.append({
                    "task_id": task_id,
                    "goal": json_loads(task["contract_json"], {}).get("goal", ""),
                })

        # Negative examples would be from tasks where pattern doesn't appear
        # Simplified for MVP

        candidate = ConceptCandidate(
            tenant_id=tenant_id,
            project_id=project_id,
            name=name,
            definition_ref=artifact.artifact_id,
            source_subgraph_refs=[pattern.pattern_id],
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            train_task_ids=pattern.task_ids,
            holdout_manifest_ref=holdout_manifest_ref,
            metrics={
                "compression_gain": pattern.compression_gain,
                "prediction_gain": pattern.prediction_gain,
                "transfer_gain": pattern.transfer_gain,
                "maintenance_cost": pattern.maintenance_cost,
                "frequency": float(pattern.frequency),
            },
            state=ConceptCandidateState.CANDIDATE,
        )

        self._persist_candidate(candidate)
        return candidate

    def _persist_candidate(self, candidate: ConceptCandidate) -> None:
        with self.database.transaction() as conn:
            conn.execute(
                """INSERT INTO concept_candidates
                   (concept_id, tenant_id, project_id, name, definition_ref, source_subgraph_refs_json,
                    positive_examples_json, negative_examples_json,
                    train_task_ids_json, holdout_manifest_ref, metrics_json,
                    state, rollback_target, version, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate.concept_id,
                    candidate.tenant_id,
                    candidate.project_id,
                    candidate.name,
                    candidate.definition_ref,
                    json_dumps(candidate.source_subgraph_refs),
                    json_dumps(candidate.positive_examples),
                    json_dumps(candidate.negative_examples),
                    json_dumps(candidate.train_task_ids),
                    candidate.holdout_manifest_ref,
                    json_dumps(candidate.metrics),
                    candidate.state.value,
                    candidate.rollback_target,
                    candidate.version,
                    candidate.created_at.isoformat(),
                    candidate.updated_at.isoformat(),
                ),
            )


class ConceptValidator:
    """Validates concept candidates through benchmark, holdout, and ablation."""

    def __init__(
        self,
        cosmos: Cosmos,
        database: SqliteDatabase,
        traces: TraceRecorder,
        compiler: ExperienceCompiler,
    ):
        self.cosmos = cosmos
        self.database = database
        self.traces = traces
        self.compiler = compiler

    def validate(
        self,
        candidate: ConceptCandidate,
        baseline_run_id: str,
    ) -> ConceptEvaluation:
        """Run full validation: baseline, treatment, holdout, ablation."""
        # This is a simplified version - full implementation would run actual benchmarks
        # For MVP, we simulate the evaluation

        evaluation = ConceptEvaluation(
            concept_id=candidate.concept_id,
            baseline_run_id=baseline_run_id,
            treatment_run_id=new_id("run"),
            ablation_run_id=new_id("run"),
            quality_delta=0.0,
            cost_delta=0.0,
            transfer_delta=0.0,
            accepted=False,
        )

        # Simulate evaluation based on candidate metrics
        compression = candidate.metrics.get("compression_gain", 0)
        prediction = candidate.metrics.get("prediction_gain", 0)
        transfer = candidate.metrics.get("transfer_gain", 0)
        cost = candidate.metrics.get("maintenance_cost", 0)

        # Heuristic: accept if net benefit positive
        net_benefit = compression + prediction + transfer - cost
        evaluation.quality_delta = net_benefit * 0.1
        evaluation.cost_delta = -cost * 0.05  # Negative = cost reduction
        evaluation.transfer_delta = transfer * 0.1
        evaluation.accepted = net_benefit > 0.1

        self._persist_evaluation(evaluation)

        # Update candidate state
        if evaluation.accepted:
            candidate.state = ConceptCandidateState.VALIDATED
        else:
            candidate.state = ConceptCandidateState.REJECTED

        self._persist_candidate(candidate)

        self.traces.record_event({
            "kind": "ConceptEvaluationCompleted",
            "producer": "ConceptValidator",
            "payload": {
                "concept_id": candidate.concept_id,
                "accepted": evaluation.accepted,
                "quality_delta": evaluation.quality_delta,
                "cost_delta": evaluation.cost_delta,
                "transfer_delta": evaluation.transfer_delta,
            },
        })

        return evaluation

    def run_ablation(
        self,
        candidate: ConceptCandidate,
        baseline_run_id: str,
    ) -> float:
        """Run ablation: disable concept and measure degradation."""
        # In full implementation, would re-run tasks with concept disabled
        # For MVP, estimate based on concept's contribution metrics
        contribution = (
            candidate.metrics.get("compression_gain", 0) +
            candidate.metrics.get("prediction_gain", 0) +
            candidate.metrics.get("transfer_gain", 0)
        )
        return contribution * 0.1  # Estimated degradation

    def _persist_evaluation(self, evaluation: ConceptEvaluation) -> None:
        with self.database.transaction() as conn:
            conn.execute(
                """INSERT INTO concept_evaluations
                   (concept_id, baseline_run_id, treatment_run_id,
                    ablation_run_id, quality_delta, cost_delta, transfer_delta,
                    accepted, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    evaluation.concept_id,
                    evaluation.baseline_run_id,
                    evaluation.treatment_run_id,
                    evaluation.ablation_run_id,
                    evaluation.quality_delta,
                    evaluation.cost_delta,
                    evaluation.transfer_delta,
                    1 if evaluation.accepted else 0,
                    utcnow().isoformat(),
                ),
            )

    def _persist_candidate(self, candidate: ConceptCandidate) -> None:
        with self.database.transaction() as conn:
            conn.execute(
                """UPDATE concept_candidates
                   SET state = ?, metrics_json = ?, updated_at = ?
                   WHERE concept_id = ?""",
                (
                    candidate.state.value,
                    json_dumps(candidate.metrics),
                    utcnow().isoformat(),
                    candidate.concept_id,
                ),
            )


class ConceptLifecycleManager:
    """Manages the lifecycle: candidate -> validated -> shadow -> active."""

    def __init__(
        self,
        database: SqliteDatabase,
        traces: TraceRecorder,
        cosmos: Cosmos,
    ):
        self.database = database
        self.traces = traces
        self.cosmos = cosmos

    def promote_to_shadow(self, concept_id: str) -> ConceptCandidate:
        """Move validated concept to shadow mode."""
        candidate = self._load_candidate(concept_id)
        if candidate.state != ConceptCandidateState.VALIDATED:
            raise ValueError(f"Concept {concept_id} is not validated")

        candidate.state = ConceptCandidateState.SHADOW
        self._persist_candidate(candidate)

        self.traces.record_event({
            "kind": "ConceptPromotedToShadow",
            "producer": "ConceptLifecycleManager",
            "payload": {"concept_id": concept_id},
        })

        return candidate

    def promote_to_active(self, concept_id: str) -> ConceptCandidate:
        """Activate a shadow concept."""
        candidate = self._load_candidate(concept_id)
        if candidate.state != ConceptCandidateState.SHADOW:
            raise ValueError(f"Concept {concept_id} is not in shadow mode")

        candidate.state = ConceptCandidateState.ACTIVE
        self._persist_candidate(candidate)

        # Also add to Cosmos as a proper concept
        self._integrate_into_cosmos(candidate)

        self.traces.record_event({
            "kind": "ConceptActivated",
            "producer": "ConceptLifecycleManager",
            "payload": {"concept_id": concept_id},
        })

        return candidate

    def reject(self, concept_id: str, reason: str) -> ConceptCandidate:
        """Reject a candidate."""
        candidate = self._load_candidate(concept_id)
        candidate.state = ConceptCandidateState.REJECTED
        self._persist_candidate(candidate)

        self.traces.record_event({
            "kind": "ConceptRejected",
            "producer": "ConceptLifecycleManager",
            "payload": {"concept_id": concept_id, "reason": reason},
        })

        return candidate

    def rollback(self, concept_id: str) -> ConceptCandidate:
        """Rollback an active concept to previous version."""
        candidate = self._load_candidate(concept_id)
        if candidate.state not in (ConceptCandidateState.ACTIVE, ConceptCandidateState.SHADOW):
            raise ValueError(f"Concept {concept_id} cannot be rolled back")

        candidate.state = ConceptCandidateState.ROLLED_BACK
        candidate.rollback_target = str(candidate.version)
        self._persist_candidate(candidate)

        self.traces.record_event({
            "kind": "ConceptRolledBack",
            "producer": "ConceptLifecycleManager",
            "payload": {"concept_id": concept_id, "target_version": candidate.version},
        })

        return candidate

    def revoke_source(self, source_artifact_id: str) -> List[ConceptCandidate]:
        """Invalidate all concepts derived from a revoked source."""
        # Find all candidates referencing this source
        rows = self.database.all(
            "SELECT * FROM concept_candidates WHERE definition_ref IN "
            "(SELECT artifact_id FROM artifacts WHERE content_hash IN "
            "  (SELECT content_hash FROM artifacts WHERE artifact_id = ?))",
            (source_artifact_id,),
        )

        invalidated = []
        for row in rows:
            candidate = self._row_to_candidate(row)
            candidate.state = ConceptCandidateState.REJECTED
            self._persist_candidate(candidate)
            invalidated.append(candidate)

        self.traces.record_event({
            "kind": "ConceptsInvalidatedBySourceRevocation",
            "producer": "ConceptLifecycleManager",
            "payload": {
                "source_artifact_id": source_artifact_id,
                "invalidated_concepts": [c.concept_id for c in invalidated],
            },
        })

        return invalidated

    def _load_candidate(self, concept_id: str) -> ConceptCandidate:
        row = self.database.one(
            "SELECT * FROM concept_candidates WHERE concept_id = ?",
            (concept_id,),
        )
        if not row:
            raise KeyError(f"Concept candidate {concept_id} not found")
        return self._row_to_candidate(row)

    def _row_to_candidate(self, row: Dict) -> ConceptCandidate:
        return ConceptCandidate(
            concept_id=row["concept_id"],
            tenant_id=row["tenant_id"],
            project_id=row["project_id"],
            name=row["name"],
            definition_ref=row["definition_ref"],
            source_subgraph_refs=json_loads(row["source_subgraph_refs_json"], []),
            positive_examples=json_loads(row["positive_examples_json"], []),
            negative_examples=json_loads(row["negative_examples_json"], []),
            train_task_ids=json_loads(row["train_task_ids_json"], []),
            holdout_manifest_ref=row["holdout_manifest_ref"],
            metrics=json_loads(row["metrics_json"], {}),
            state=ConceptCandidateState(row["state"]),
            rollback_target=row["rollback_target"],
            version=row["version"],
        )

    def _persist_candidate(self, candidate: ConceptCandidate) -> None:
        with self.database.transaction() as conn:
            conn.execute(
                """UPDATE concept_candidates
                   SET state = ?, metrics_json = ?, rollback_target = ?,
                       version = version + 1, updated_at = ?
                   WHERE concept_id = ?""",
                (
                    candidate.state.value,
                    json_dumps(candidate.metrics),
                    candidate.rollback_target,
                    utcnow().isoformat(),
                    candidate.concept_id,
                ),
            )

    def _integrate_into_cosmos(self, candidate: ConceptCandidate) -> None:
        """Add the validated concept to the Cosmos."""
        # Load definition
        definition = self.cosmos.store.get_json(
            candidate.definition_ref,
            candidate.tenant_id,
            project_id=candidate.project_id,
        )
        scope = AccessScope(
            tenant_id=candidate.tenant_id,
            project_id=candidate.project_id,
            visibility="project" if candidate.project_id else "tenant",
        )
        self.cosmos.import_text(
            title=candidate.name,
            text=f"{candidate.name}: {json_dumps(definition)}",
            tenant_id=candidate.tenant_id,
            access_scope=scope,
            sectors=["Emergence"],
            trusted=True,
        )
        self.traces.record_event({
            "kind": "ConceptIntegratedIntoCosmos",
            "producer": "ConceptLifecycleManager",
            "payload": {
                "concept_id": candidate.concept_id,
                "name": candidate.name,
                "definition": definition,
            },
        })


def create_concept_pipeline(
    cosmos: Cosmos,
    database: SqliteDatabase,
    traces: TraceRecorder,
    compiler: ExperienceCompiler,
) -> Tuple[SubgraphMiner, ConceptGenerator, ConceptValidator, ConceptLifecycleManager]:
    """Create the full concept genesis pipeline."""
    miner = SubgraphMiner(database, traces)
    generator = ConceptGenerator(cosmos, database, traces, miner)
    validator = ConceptValidator(cosmos, database, traces, compiler)
    lifecycle = ConceptLifecycleManager(database, traces, cosmos)
    return miner, generator, validator, lifecycle
