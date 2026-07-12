"""Causal layer: evidence statuses, counterfactual queries, and experiment selection."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..contracts import (
    EvidenceStatus,
    ExperimentRecord,
    ExperimentStatus,
    HypothesisRecord,
    GraphEdge,
    EdgeType,
    new_id,
    utcnow,
)
from ..cosmos import Cosmos
from ..database import SqliteDatabase, json_dumps, json_loads
from ..observability import TraceRecorder


@dataclass
class CausalEdge:
    """A causal edge with mechanism and applicability conditions."""
    edge_id: str
    source_id: str
    target_id: str
    mechanism: str  # Description of the causal mechanism
    conditions: Dict[str, Any]  # Conditions under which this causal link holds
    applicability_scope: str  # local, project, tenant, global
    tenant_id: str = "local"
    project_id: Optional[str] = None
    confidence: float = 0.5
    evidence_status: EvidenceStatus = EvidenceStatus.SIMULATED
    provenance_refs: List[str] = field(default_factory=list)
    version: int = 1
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class InterventionResult:
    """Result of a counterfactual intervention."""
    intervention_id: str
    variable: str
    value: Any
    predicted_outcome: Dict[str, Any]
    actual_outcome: Optional[Dict[str, Any]] = None
    evidence_status: EvidenceStatus = EvidenceStatus.SIMULATED
    confidence: float = 0.5


class CausalEngine:
    """Manages causal edges, counterfactual queries, and experiment selection."""

    def __init__(
        self,
        cosmos: Cosmos,
        database: SqliteDatabase,
        traces: TraceRecorder,
        tenant_id: str = "local",
        project_id: Optional[str] = None,
    ):
        self.cosmos = cosmos
        self.database = database
        self.traces = traces
        self.tenant_id = tenant_id
        self.project_id = project_id
        self.causal_edges: Dict[str, CausalEdge] = {}
        self.intervention_registry: Dict[str, Callable] = {}

    def register_intervention(self, name: str, handler: Callable) -> None:
        """Register a deterministic intervention handler for the sandbox."""
        self.intervention_registry[name] = handler

    def add_causal_edge(
        self,
        source_id: str,
        target_id: str,
        mechanism: str,
        conditions: Dict[str, Any],
        applicability_scope: str,
        confidence: float,
        evidence_status: EvidenceStatus,
        provenance_refs: List[str],
    ) -> CausalEdge:
        """Add a typed causal edge to the engine."""
        edge = CausalEdge(
            edge_id=new_id("cedge"),
            source_id=source_id,
            target_id=target_id,
            tenant_id=self.tenant_id,
            project_id=self.project_id,
            mechanism=mechanism,
            conditions=conditions,
            applicability_scope=applicability_scope,
            confidence=confidence,
            evidence_status=evidence_status,
            provenance_refs=provenance_refs,
        )
        self.causal_edges[edge.edge_id] = edge
        self._persist_causal_edge(edge)
        return edge

    def _persist_causal_edge(self, edge: CausalEdge) -> None:
        with self.database.transaction() as conn:
            conn.execute(
                """INSERT INTO graph_edges
                   (edge_id, tenant_id, project_id, source_id, target_id, edge_type,
                    weight, confidence, scope, provenance_refs_json, valid_from,
                    valid_to, version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    edge.edge_id,
                    edge.tenant_id,
                    edge.project_id,
                    edge.source_id,
                    edge.target_id,
                    EdgeType.CAUSAL.value,
                    edge.confidence,  # Use confidence as weight
                    edge.confidence,
                    edge.applicability_scope,
                    json_dumps(edge.provenance_refs),
                    edge.created_at.isoformat(),
                    None,
                    edge.version,
                    edge.created_at.isoformat(),
                ),
            )

    def intervene(self, variable: str, value: Any, context: Dict[str, Any]) -> InterventionResult:
        """Execute a counterfactual intervention in the sandbox.

        This does NOT modify the canonical Cosmos - it runs in an isolated
        simulation context.
        """
        intervention_id = new_id("intv")

        # Find relevant causal edges
        relevant_edges = [
            e for e in self.causal_edges.values()
            if e.target_id == variable or e.source_id == variable
        ]

        # Predict outcome using causal model
        predicted = self._predict_counterfactual(variable, value, context, relevant_edges)

        result = InterventionResult(
            intervention_id=intervention_id,
            variable=variable,
            value=value,
            predicted_outcome=predicted,
            evidence_status=EvidenceStatus.SIMULATED,
            confidence=0.5,  # Default for simulated
        )

        # Record in trace but NOT in Cosmos
        self.traces.record_event({
            "task_id": context.get("task_id", "unknown"),
            "trace_id": context.get("trace_id", "unknown"),
            "kind": "CounterfactualIntervention",
            "producer": "CausalEngine",
            "payload": {
                "intervention_id": intervention_id,
                "variable": variable,
                "value": value,
                "predicted_outcome": predicted,
                "evidence_status": "simulated",
            },
        })

        return result

    def _predict_counterfactual(
        self,
        variable: str,
        value: Any,
        context: Dict[str, Any],
        edges: List[CausalEdge],
    ) -> Dict[str, Any]:
        """Predict outcome of intervention using causal model."""
        # Simple prediction based on causal edges
        # In practice, this would use a proper causal inference engine
        outcome = {"intervened_variable": variable, "value": value}

        for edge in edges:
            if edge.source_id == variable:
                # Forward prediction
                outcome[f"predicted_{edge.target_id}"] = {
                    "mechanism": edge.mechanism,
                    "confidence": edge.confidence,
                }
            elif edge.target_id == variable:
                # Backward prediction (what would need to change)
                outcome[f"required_{edge.source_id}"] = {
                    "mechanism": edge.mechanism,
                    "confidence": edge.confidence,
                }

        return outcome

    def verify_intervention(self, intervention_id: str, actual_outcome: Dict[str, Any]) -> None:
        """Update intervention result with actual observed outcome."""
        # This would be called after running the experiment in sandbox
        # For now, just trace it
        self.traces.record_event({
            "kind": "InterventionVerified",
            "producer": "CausalEngine",
            "payload": {
                "intervention_id": intervention_id,
                "actual_outcome": actual_outcome,
                "evidence_status": EvidenceStatus.OBSERVED.value,
            },
        })


class ExperimentSelector:
    """Selects the best experiment to differentiate competing hypotheses."""

    def __init__(
        self,
        causal_engine: CausalEngine,
        traces: TraceRecorder,
    ):
        self.causal_engine = causal_engine
        self.traces = traces
        self.registered_operations: Dict[str, Dict[str, Any]] = {}

    def register_operation(
        self,
        name: str,
        handler: Callable,
        cost_estimate: int = 1,
        risk: float = 0.0,
        description: str = "",
    ) -> None:
        """Register a sandbox operation for experiments."""
        self.registered_operations[name] = {
            "handler": handler,
            "cost_estimate": cost_estimate,
            "risk": risk,
            "description": description,
        }

    def estimate_information_gain(
        self,
        hypotheses: List[HypothesisRecord],
        operation: str,
        input_data: Dict[str, Any],
    ) -> float:
        """Estimate expected information gain from an experiment.

        Uses the difference in predictions between hypotheses.
        """
        if len(hypotheses) < 2:
            return 0.0

        # Get predictions for each hypothesis
        predictions = []
        for hyp in hypotheses:
            for pred_id in hyp.predictions:
                # In practice, would look up the prediction
                # For now, simulate based on hypothesis confidence
                pred_value = hyp.confidence * 0.8 + (1 - hyp.confidence) * 0.2
                predictions.append(pred_value)

        if not predictions:
            return 0.1  # Default small gain

        # Information gain ≈ variance in predictions
        mean_pred = sum(predictions) / len(predictions)
        variance = sum((p - mean_pred) ** 2 for p in predictions) / len(predictions)

        # Normalize to 0-1
        return min(1.0, variance * 4)

    def select_experiment(
        self,
        hypotheses: List[HypothesisRecord],
        available_budget: int,
    ) -> Optional[Tuple[str, Dict[str, Any], float, int, float]]:
        """Select the best experiment to run.

        Returns: (operation, input, expected_info_gain, estimated_cost, risk)
        """
        if len(hypotheses) < 2:
            return None

        candidates = []

        for op_name, op_info in self.registered_operations.items():
            # Generate test inputs based on hypothesis predictions
            test_inputs = self._generate_test_inputs(hypotheses, op_name)

            for test_input in test_inputs:
                info_gain = self.estimate_information_gain(hypotheses, op_name, test_input)
                cost = op_info["cost_estimate"]
                risk = op_info["risk"]

                # Only consider if within budget
                if cost <= available_budget:
                    # Score = info_gain / cost * (1 - risk)
                    score = info_gain / max(1, cost) * (1 - risk)
                    candidates.append((score, op_name, test_input, info_gain, cost, risk))

        if not candidates:
            return None

        # Pick highest scoring
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0]

        self.traces.record_event({
            "kind": "ExperimentSelected",
            "producer": "ExperimentSelector",
            "payload": {
                "operation": best[1],
                "input": best[2],
                "expected_information_gain": best[3],
                "estimated_cost": best[4],
                "risk": best[5],
                "competing_hypotheses": [h.hypothesis_id for h in hypotheses],
            },
        })

        return (best[1], best[2], best[3], best[4], best[5])

    def _generate_test_inputs(
        self,
        hypotheses: List[HypothesisRecord],
        operation: str,
    ) -> List[Dict[str, Any]]:
        """Generate test inputs that differentiate hypotheses."""
        inputs = []

        # Simple strategy: use each hypothesis's predicted values
        for hyp in hypotheses[:3]:  # Limit to top 3
            for pred_id in hyp.predictions:
                # Would look up actual prediction
                # For now, create a generic test input
                inputs.append({
                    "hypothesis_id": hyp.hypothesis_id,
                    "operation": operation,
                    "test_value": hyp.confidence,
                })

        # Add a boundary/edge case
        inputs.append({
            "hypothesis_id": "edge_case",
            "operation": operation,
            "test_value": 0.5,
        })

        return inputs[:5]  # Limit candidates


class Sandbox:
    """Deterministic sandbox for running registered operations.

    Only pre-registered operations can execute. No arbitrary code execution.
    """

    def __init__(
        self,
        operations: Dict[str, Dict[str, Any]],
        traces: TraceRecorder,
        database: SqliteDatabase,
    ):
        self.operations = operations
        self.traces = traces
        self.database = database

    def run(
        self,
        operation: str,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Any, EvidenceStatus]:
        """Run a registered operation in the sandbox.

        Returns (result, evidence_status).
        """
        if operation not in self.operations:
            raise ValueError(f"Operation '{operation}' not registered in sandbox")

        op_info = self.operations[operation]
        handler = op_info["handler"]

        span = self.traces.start_span(
            trace_id=context.get("trace_id", "unknown"),
            component="Sandbox",
            operation=operation,
            input_summary={"input_keys": list(input_data.keys())},
        )

        try:
            result = handler(input_data, context)

            # Determine evidence status based on operation type
            evidence_status = self._determine_evidence_status(operation, op_info)

            self.traces.finish_span(span, output_summary={"result_type": type(result).__name__})

            return result, evidence_status

        except Exception as e:
            self.traces.fail_span(span, {"code": "sandbox_error", "message": str(e)})
            raise

    def _determine_evidence_status(
        self,
        operation: str,
        op_info: Dict[str, Any],
    ) -> EvidenceStatus:
        """Determine the evidence status based on operation type."""
        # In a full implementation, this would be metadata on the operation
        # For MVP, use simple rules
        if operation.startswith("simulate_"):
            return EvidenceStatus.SIMULATED
        elif operation.startswith("compute_"):
            return EvidenceStatus.COMPUTED
        elif operation.startswith("observe_"):
            return EvidenceStatus.OBSERVED
        else:
            return EvidenceStatus.SIMULATED

    def run_experiment(
        self,
        experiment: ExperimentRecord,
        context: Dict[str, Any],
    ) -> Tuple[Any, EvidenceStatus]:
        """Run a full experiment and return result with evidence status."""
        result, status = self.run(experiment.operation, experiment.input, context)

        # Record experiment execution
        self.traces.record_event({
            "task_id": experiment.task_id,
            "trace_id": context.get("trace_id", experiment.task_id),
            "kind": "ExperimentExecuted",
            "producer": "Sandbox",
            "payload": {
                "experiment_id": experiment.experiment_id,
                "operation": experiment.operation,
                "evidence_status": status.value,
                "cost": experiment.estimated_cost,
            },
        })

        return result, status


# Built-in sandbox operations for the MVP

def simulate_sequence_transform(input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate a sequence transformation rule."""
    sequence = input_data.get("sequence", [])
    rule = input_data.get("rule", "identity")

    # Simple deterministic transformations for testing
    if rule == "reverse":
        result = list(reversed(sequence))
    elif rule == "sort":
        result = sorted(sequence)
    elif rule == "increment":
        result = [x + 1 for x in sequence]
    elif rule == "double":
        result = [x * 2 for x in sequence]
    elif rule == "filter_even":
        result = [x for x in sequence if x % 2 == 0]
    else:
        result = sequence

    return {"output": result, "rule": rule, "input": sequence}


def compute_pattern_match(input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Compute pattern matching score."""
    pattern = input_data.get("pattern", [])
    sequence = input_data.get("sequence", [])

    # Simple pattern matching
    matches = 0
    for i in range(len(sequence) - len(pattern) + 1):
        if sequence[i:i+len(pattern)] == pattern:
            matches += 1

    score = matches / max(1, len(sequence) - len(pattern) + 1)
    return {"matches": matches, "score": score, "pattern_length": len(pattern)}


def observe_sequence_property(input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Observe a property of a sequence (simulated observation)."""
    sequence = input_data.get("sequence", [])
    property_name = input_data.get("property", "length")

    if property_name == "length":
        value = len(sequence)
    elif property_name == "sum":
        value = sum(sequence)
    elif property_name == "max":
        value = max(sequence) if sequence else 0
    elif property_name == "min":
        value = min(sequence) if sequence else 0
    elif property_name == "unique_count":
        value = len(set(sequence))
    else:
        value = None

    return {"property": property_name, "value": value}


def create_default_sandbox(traces: TraceRecorder, database: SqliteDatabase) -> Sandbox:
    """Create a sandbox with default registered operations."""
    operations = {
        "simulate_reverse": {
            "handler": lambda i, c: simulate_sequence_transform({**i, "rule": "reverse"}, c),
            "cost_estimate": 1,
            "risk": 0.0,
            "description": "Simulate sequence reversal",
        },
        "simulate_sort": {
            "handler": lambda i, c: simulate_sequence_transform({**i, "rule": "sort"}, c),
            "cost_estimate": 1,
            "risk": 0.0,
            "description": "Simulate sequence sorting",
        },
        "simulate_increment": {
            "handler": lambda i, c: simulate_sequence_transform({**i, "rule": "increment"}, c),
            "cost_estimate": 1,
            "risk": 0.0,
            "description": "Simulate incrementing all elements",
        },
        "simulate_double": {
            "handler": lambda i, c: simulate_sequence_transform({**i, "rule": "double"}, c),
            "cost_estimate": 1,
            "risk": 0.0,
            "description": "Simulate doubling all elements",
        },
        "simulate_filter_even": {
            "handler": lambda i, c: simulate_sequence_transform({**i, "rule": "filter_even"}, c),
            "cost_estimate": 1,
            "risk": 0.0,
            "description": "Simulate filtering even numbers",
        },
        "compute_pattern_match": {
            "handler": compute_pattern_match,
            "cost_estimate": 1,
            "risk": 0.0,
            "description": "Compute pattern matching score",
        },
        "observe_length": {
            "handler": lambda i, c: observe_sequence_property({**i, "property": "length"}, c),
            "cost_estimate": 1,
            "risk": 0.0,
            "description": "Observe sequence length",
        },
        "observe_sum": {
            "handler": lambda i, c: observe_sequence_property({**i, "property": "sum"}, c),
            "cost_estimate": 1,
            "risk": 0.0,
            "description": "Observe sequence sum",
        },
        "observe_unique_count": {
            "handler": lambda i, c: observe_sequence_property({**i, "property": "unique_count"}, c),
            "cost_estimate": 1,
            "risk": 0.0,
            "description": "Observe unique element count",
        },
    }

    return Sandbox(operations, traces, database)
