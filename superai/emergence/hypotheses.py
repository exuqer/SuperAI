"""Hypothesis ecosystem: competing structured models with evidence and predictions."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..contracts import (
    EvidenceStatus,
    ExperimentRecord,
    ExperimentStatus,
    HypothesisRecord,
    HypothesisStatus,
    PredictionRecord,
    ResourceBudget,
    TaskContract,
    new_id,
    utcnow,
)
from ..cosmos import Cosmos
from ..database import SqliteDatabase, json_dumps, json_loads
from ..observability import TraceRecorder


@dataclass
class HypothesisGenerator:
    """A strategy for generating hypotheses from the active graph."""
    name: str
    generate: Callable[["HypothesisBoard", TaskContract], List[HypothesisRecord]]


@dataclass
class HypothesisBoard:
    """Manages competing hypotheses for a single task.

    The board maintains 3-5 active hypotheses, allocates budget between
    exploitation and exploration, and tracks evidence for/against each.
    """
    task_id: str
    budget: ResourceBudget
    traces: TraceRecorder
    cosmos: Cosmos
    tenant_id: str = "local"
    project_id: Optional[str] = None
    trace_id: Optional[str] = None

    # State
    hypotheses: Dict[str, HypothesisRecord] = field(default_factory=dict)
    predictions: Dict[str, PredictionRecord] = field(default_factory=dict)
    experiments: Dict[str, ExperimentRecord] = field(default_factory=dict)
    generators: List[HypothesisGenerator] = field(default_factory=list)

    # Budget tracking
    exploitation_budget: int = 0
    exploration_budget: int = 0
    total_allocated: int = 0

    def __post_init__(self):
        self._register_default_generators()

    def _register_default_generators(self) -> None:
        """Register built-in hypothesis generation strategies."""
        self.generators = [
            HypothesisGenerator("pattern_match", self._gen_pattern_match),
            HypothesisGenerator("causal_chain", self._gen_causal_chain),
            HypothesisGenerator("counterfactual", self._gen_counterfactual),
            HypothesisGenerator("analogy", self._gen_analogy),
            HypothesisGenerator("null_hypothesis", self._gen_null),
        ]

    def initialize(self, contract: TaskContract) -> List[HypothesisRecord]:
        """Create initial 3-5 hypotheses from registered generators."""
        all_hypotheses = []
        for generator in self.generators:
            try:
                hyps = generator.generate(self, contract)
                all_hypotheses.extend(hyps)
            except Exception as e:
                self.traces.record_event({
                    "task_id": self.task_id,
                    "trace_id": self.trace_id or contract.task_id,
                    "kind": "HypothesisGenerationFailed",
                    "producer": "HypothesisBoard",
                    "payload": {"generator": generator.name, "error": str(e)},
                })

        # Deduplicate by statement similarity
        unique = self._deduplicate_hypotheses(all_hypotheses)

        # Select top 3-5 by initial score
        selected = sorted(unique, key=lambda h: h.confidence * (1 + h.novelty), reverse=True)[:5]

        if not selected:
            return []

        # Assign budgets
        self._allocate_budgets(selected)

        # Persist
        for hyp in selected:
            hyp.tenant_id = self.tenant_id
            hyp.project_id = self.project_id
            hyp.status = HypothesisStatus.ACTIVE
            self.hypotheses[hyp.hypothesis_id] = hyp
            self._persist_hypothesis(hyp)
            self.add_prediction(
                PredictionRecord(
                    hypothesis_id=hyp.hypothesis_id,
                    experiment_input={"task_id": contract.task_id},
                    expected_output={"hypothesis_id": hyp.hypothesis_id},
                )
            )

        return selected

    def _deduplicate_hypotheses(self, hypotheses: List[HypothesisRecord]) -> List[HypothesisRecord]:
        """Remove hypotheses with very similar statements."""
        unique = []
        seen = set()
        for hyp in hypotheses:
            # Simple deduplication by first 50 chars of statement
            key = hyp.statement[:50].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(hyp)
        return unique

    def _allocate_budgets(self, hypotheses: List[HypothesisRecord]) -> None:
        """Allocate exploitation and exploration budgets across hypotheses."""
        total_budget = self.budget.max_steps  # Use steps as proxy for budget
        exploration_share = self.budget.exploration_share
        exploitation_budget = int(total_budget * (1 - exploration_share))
        exploration_budget = total_budget - exploitation_budget

        # Sort by confidence (exploitation) and novelty (exploration)
        sorted_by_conf = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
        sorted_by_novelty = sorted(hypotheses, key=lambda h: h.novelty, reverse=True)

        # Allocate exploitation budget to high-confidence hypotheses
        for i, hyp in enumerate(sorted_by_conf):
            share = exploitation_budget * (0.5 ** i)  # Exponential decay
            hyp.allocated_budget = int(share)

        # Allocate exploration budget to high-novelty hypotheses
        for i, hyp in enumerate(sorted_by_novelty):
            share = exploration_budget * (0.5 ** i)
            hyp.allocated_budget += int(share)

        self.exploitation_budget = exploitation_budget
        self.exploration_budget = exploration_budget
        self.total_allocated = sum(h.allocated_budget for h in hypotheses)

    def add_evidence(
        self,
        hypothesis_id: str,
        evidence_ref: str,
        supports: bool,
        status: EvidenceStatus = EvidenceStatus.OBSERVED,
    ) -> None:
        """Add evidence for or against a hypothesis."""
        hyp = self.hypotheses.get(hypothesis_id)
        if not hyp:
            return

        if supports:
            if evidence_ref not in hyp.evidence_for:
                hyp.evidence_for.append(evidence_ref)
        else:
            if evidence_ref not in hyp.evidence_against:
                hyp.evidence_against.append(evidence_ref)

        # Update confidence based on evidence balance
        self._recalculate_confidence(hyp)
        hyp.updated_at = utcnow()
        self._persist_hypothesis(hyp)

        with self.cosmos.database.transaction() as conn:
            conn.execute(
                """INSERT INTO hypothesis_evidence
                   (evidence_id, hypothesis_id, kind, source_ref, description, weight, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_id("evidence"),
                    hypothesis_id,
                    "support" if supports else "against",
                    evidence_ref,
                    status.value,
                    1.0,
                    utcnow().isoformat(),
                ),
            )

        self.traces.record_event({
            "task_id": self.task_id,
            "trace_id": self.trace_id or self.task_id,
            "kind": "EvidenceAdded",
            "producer": "HypothesisBoard",
            "payload": {
                "hypothesis_id": hypothesis_id,
                "evidence_ref": evidence_ref,
                "supports": supports,
                "status": status.value,
                "new_confidence": hyp.confidence,
            },
        })

    def _recalculate_confidence(self, hyp: HypothesisRecord) -> None:
        """Recalculate hypothesis confidence from evidence."""
        total = len(hyp.evidence_for) + len(hyp.evidence_against)
        if total == 0:
            hyp.confidence = 0.5
        else:
            # Simple Bayesian update with prior 0.5
            hyp.confidence = (len(hyp.evidence_for) + 1) / (total + 2)

    def add_prediction(self, prediction: PredictionRecord) -> None:
        """Add a testable prediction for a hypothesis."""
        self.predictions[prediction.prediction_id] = prediction
        hyp = self.hypotheses.get(prediction.hypothesis_id)
        if hyp and prediction.prediction_id not in hyp.predictions:
            hyp.predictions.append(prediction.prediction_id)
            hyp.updated_at = utcnow()
            self._persist_hypothesis(hyp)

        # Persist prediction
        with self.cosmos.database.transaction() as conn:
            conn.execute(
                """INSERT INTO predictions
                   (prediction_id, hypothesis_id, experiment_input_json,
                    expected_output_json, tolerance_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    prediction.prediction_id,
                    prediction.hypothesis_id,
                    json_dumps(prediction.experiment_input),
                    json_dumps(prediction.expected_output),
                    json_dumps(prediction.tolerance),
                    utcnow().isoformat(),
                ),
            )

    def propose_experiment(
        self,
        competing_hypothesis_ids: List[str],
        operation: str,
        input_data: Dict[str, Any],
        expected_information_gain: float,
        estimated_cost: int = 1,
        risk: float = 0.0,
    ) -> ExperimentRecord:
        """Propose an experiment to differentiate between hypotheses."""
        exp = ExperimentRecord(
            task_id=self.task_id,
            tenant_id=self.tenant_id,
            competing_hypothesis_ids=competing_hypothesis_ids,
            operation=operation,
            input=input_data,
            expected_information_gain=expected_information_gain,
            estimated_cost=estimated_cost,
            risk=risk,
            status=ExperimentStatus.PROPOSED,
        )
        self.experiments[exp.experiment_id] = exp
        self._persist_experiment(exp)
        return exp

    def select_best_experiment(self) -> Optional[ExperimentRecord]:
        """Select the experiment with highest information gain per cost."""
        proposed = [e for e in self.experiments.values() if e.status == ExperimentStatus.PROPOSED]
        if not proposed:
            return None

        # Score by information gain / cost, penalize risk
        scored = []
        for exp in proposed:
            score = exp.expected_information_gain / max(1, exp.estimated_cost)
            score *= (1 - exp.risk)
            scored.append((score, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        best.status = ExperimentStatus.APPROVED
        self._persist_experiment(best)
        return best

    def complete_experiment(
        self,
        experiment_id: str,
        result_ref: str,
        evidence_status: EvidenceStatus,
    ) -> None:
        """Mark experiment as completed and update hypothesis evidence."""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return

        exp.result_ref = result_ref
        exp.evidence_status = evidence_status
        exp.status = ExperimentStatus.COMPLETED
        self._persist_experiment(exp)

        # For each competing hypothesis, evaluate prediction
        for hyp_id in exp.competing_hypothesis_ids:
            hyp = self.hypotheses.get(hyp_id)
            if not hyp:
                continue

            # Check if any prediction matches
            matched = False
            for pred_id in hyp.predictions:
                pred = self.predictions.get(pred_id)
                if not pred:
                    continue
                # Simple matching - in practice would use tolerance
                if pred.experiment_input == exp.input:
                    matched = True
                    # If prediction matches result, it's evidence for
                    # For simplicity, we assume match = support
                    self.add_evidence(hyp_id, result_ref, supports=True, status=evidence_status)
                    break

            if not matched:
                # No prediction matched - weak evidence against
                self.add_evidence(hyp_id, result_ref, supports=False, status=evidence_status)

        self.budget_ledger["experiments_run"] = self.budget_ledger.get("experiments_run", 0) + 1

    def get_ranked_hypotheses(self) -> List[HypothesisRecord]:
        """Return hypotheses ranked by score (confidence * novelty * predictive_value)."""
        def score(h: HypothesisRecord) -> float:
            pred_value = len(h.predictions) * 0.1
            return h.confidence * (1 + h.novelty + pred_value)

        return sorted(self.hypotheses.values(), key=score, reverse=True)

    def select_best_hypothesis(self) -> Optional[HypothesisRecord]:
        """Select the highest-ranked hypothesis as the working model."""
        ranked = self.get_ranked_hypotheses()
        if not ranked:
            return None

        best = ranked[0]
        best.status = HypothesisStatus.SELECTED
        self._persist_hypothesis(best)

        # Archive others
        for hyp in ranked[1:]:
            hyp.status = HypothesisStatus.ARCHIVED
            self._persist_hypothesis(hyp)

        return best

    def merge_hypotheses(self, hypothesis_ids: List[str], new_statement: str) -> HypothesisRecord:
        """Merge multiple hypotheses into a new one without deleting parents."""
        parents = [self.hypotheses[hid] for hid in hypothesis_ids if hid in self.hypotheses]
        if len(parents) < 2:
            raise ValueError("Need at least 2 hypotheses to merge")

        # Combine evidence
        all_evidence_for = list(set().union(*[p.evidence_for for p in parents]))
        all_evidence_against = list(set().union(*[p.evidence_against for p in parents]))
        all_predictions = list(set().union(*[p.predictions for p in parents]))

        merged = HypothesisRecord(
            task_id=self.task_id,
            tenant_id=self.tenant_id,
            project_id=self.project_id,
            family_id=parents[0].family_id,
            statement=new_statement,
            assumptions=list(set().union(*[p.assumptions for p in parents])),
            evidence_for=all_evidence_for,
            evidence_against=all_evidence_against,
            predictions=all_predictions,
            confidence=sum(p.confidence for p in parents) / len(parents),
            novelty=max(p.novelty for p in parents),
            allocated_budget=0,
            spent_budget=sum(p.spent_budget for p in parents),
            status=HypothesisStatus.PROPOSED,
            parent_ids=[p.hypothesis_id for p in parents],
        )

        self.hypotheses[merged.hypothesis_id] = merged
        self._persist_hypothesis(merged)
        return merged

    def _persist_hypothesis(self, hyp: HypothesisRecord) -> None:
        with self.cosmos.database.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO hypotheses
                   (hypothesis_id, task_id, tenant_id, project_id, family_id, statement, assumptions_json,
                    evidence_for_json, evidence_against_json, predictions_json,
                    confidence, novelty, allocated_budget, spent_budget,
                    status, parent_ids_json, version, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    hyp.hypothesis_id,
                    hyp.task_id,
                    hyp.tenant_id,
                    hyp.project_id,
                    hyp.family_id,
                    hyp.statement,
                    json_dumps(hyp.assumptions),
                    json_dumps(hyp.evidence_for),
                    json_dumps(hyp.evidence_against),
                    json_dumps(hyp.predictions),
                    hyp.confidence,
                    hyp.novelty,
                    hyp.allocated_budget,
                    hyp.spent_budget,
                    hyp.status.value,
                    json_dumps(hyp.parent_ids),
                    hyp.version,
                    hyp.created_at.isoformat(),
                    hyp.updated_at.isoformat(),
                ),
            )

    def _persist_experiment(self, exp: ExperimentRecord) -> None:
        with self.cosmos.database.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO experiments
                   (experiment_id, task_id, tenant_id, competing_hypothesis_ids_json,
                    operation, input_json, expected_information_gain,
                    estimated_cost, risk, result_ref, evidence_status,
                    status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    exp.experiment_id,
                    exp.task_id,
                    exp.tenant_id,
                    json_dumps(exp.competing_hypothesis_ids),
                    exp.operation,
                    json_dumps(exp.input),
                    exp.expected_information_gain,
                    exp.estimated_cost,
                    exp.risk,
                    exp.result_ref,
                    exp.evidence_status.value,
                    exp.status.value,
                    exp.created_at.isoformat(),
                    exp.updated_at.isoformat(),
                ),
            )

    # ===================== Built-in Generators =====================

    def _gen_pattern_match(self, board: "HypothesisBoard", contract: TaskContract) -> List[HypothesisRecord]:
        """Generate hypotheses based on pattern matching in retrieved claims."""
        retrieval = board.cosmos.retrieve(contract, limit=30)
        hypotheses = []

        # Group claims by predicate
        by_predicate: Dict[str, List] = {}
        for item in retrieval.claims:
            by_predicate.setdefault(item.claim.predicate, []).append(item)

        for predicate, items in by_predicate.items():
            if len(items) >= 2:
                # Hypothesis: this predicate follows a pattern
                statement = f"Predicate '{predicate}' follows a consistent pattern across {len(items)} instances"
                hyp = HypothesisRecord(
                    task_id=contract.task_id,
                    family_id=new_id("fam"),
                    statement=statement,
                    assumptions=[f"Pattern holds for {predicate}"],
                    evidence_for=[item.claim.claim_id for item in items],
                    predictions=[],
                    confidence=0.6,
                    novelty=0.3,
                    allocated_budget=0,
                    spent_budget=0,
                    status=HypothesisStatus.PROPOSED,
                )
                hypotheses.append(hyp)

        return hypotheses[:2]  # Limit to 2 from this generator

    def _gen_causal_chain(self, board: "HypothesisBoard", contract: TaskContract) -> List[HypothesisRecord]:
        """Generate causal chain hypotheses."""
        retrieval = board.cosmos.retrieve(contract, limit=20)
        hypotheses = []

        # Look for temporal or causal language in claims
        causal_claims = [
            item for item in retrieval.claims
            if any(kw in item.claim.object_value.lower()
                   for kw in ["because", "causes", "leads to", "results in", "implies"])
        ]

        for item in causal_claims[:2]:
            hyp = HypothesisRecord(
                task_id=contract.task_id,
                family_id=new_id("fam"),
                statement=f"Causal mechanism: {item.claim.object_value[:100]}",
                assumptions=["Causal relationship holds in this context"],
                evidence_for=[item.claim.claim_id],
                predictions=[],
                confidence=item.score,
                novelty=0.5,
                allocated_budget=0,
                spent_budget=0,
                status=HypothesisStatus.PROPOSED,
            )
            hypotheses.append(hyp)

        return hypotheses

    def _gen_counterfactual(self, board: "HypothesisBoard", contract: TaskContract) -> List[HypothesisRecord]:
        """Generate counterfactual hypotheses."""
        retrieval = board.cosmos.retrieve(contract, limit=15)
        hypotheses = []

        if retrieval.claims:
            # Take the strongest claim and negate it
            strongest = max(retrieval.claims, key=lambda x: x.score)
            hyp = HypothesisRecord(
                task_id=contract.task_id,
                family_id=new_id("fam"),
                statement=f"Counterfactual: The opposite of '{strongest.claim.object_value[:80]}' holds",
                assumptions=["Standard interpretation may be incorrect"],
                evidence_for=[],
                evidence_against=[strongest.claim.claim_id],
                predictions=[],
                confidence=0.3,  # Low initial confidence
                novelty=0.8,     # High novelty
                allocated_budget=0,
                spent_budget=0,
                status=HypothesisStatus.PROPOSED,
            )
            hypotheses.append(hyp)

        return hypotheses

    def _gen_analogy(self, board: "HypothesisBoard", contract: TaskContract) -> List[HypothesisRecord]:
        """Generate analogical hypotheses from similar tasks."""
        # In a full implementation, this would query past tasks
        # For MVP, return empty - requires historical data
        return []

    def _gen_null(self, board: "HypothesisBoard", contract: TaskContract) -> List[HypothesisRecord]:
        """Generate a null hypothesis (no pattern / random)."""
        hyp = HypothesisRecord(
            task_id=contract.task_id,
            family_id=new_id("fam"),
            statement="Null hypothesis: Observed patterns are due to chance/noise",
            assumptions=["No underlying structure"],
            evidence_for=[],
            evidence_against=[],
            predictions=[],
            confidence=0.4,
            novelty=0.0,
            allocated_budget=0,
            spent_budget=0,
            status=HypothesisStatus.PROPOSED,
        )
        return [hyp]


def create_hypothesis_board(
    task_id: str,
    budget: ResourceBudget,
    traces: TraceRecorder,
    cosmos: Cosmos,
    tenant_id: str = "local",
    project_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> HypothesisBoard:
    """Factory function to create a HypothesisBoard."""
    return HypothesisBoard(
        task_id=task_id,
        budget=budget,
        traces=traces,
        cosmos=cosmos,
        tenant_id=tenant_id,
        project_id=project_id,
        trace_id=trace_id,
    )
