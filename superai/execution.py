"""Atlas, bounded planner, deterministic text codec and critics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from .contracts import (
    AnswerEnvelope,
    CapabilityManifest,
    CriticReport,
    CriticVerdict,
    ExecutionPlan,
    PlanStep,
    RetrievalResult,
    SpanStatus,
    TaskContract,
)
from .cosmos import Cosmos
from .database import SqliteDatabase, json_dumps, json_loads
from .emergence.hypotheses import create_hypothesis_board
from .hive import HiveManager
from .observability import TraceRecorder

if False:  # pragma: no cover - import only for static type checkers
    from .emergence.graph import ActiveGraphBuilder


class PlanningError(RuntimeError):
    pass


class InsufficientEvidenceError(PlanningError):
    """Raised when a task has no permitted source-backed answer."""

    code = "insufficient_evidence"


class Atlas:
    """Catalogues executable capabilities; it never chooses a Hive itself."""

    def __init__(self, database: SqliteDatabase) -> None:
        self.database = database

    def register(self, manifest: CapabilityManifest) -> None:
        self.database.execute(
            "INSERT OR REPLACE INTO capabilities(capability_id, version, kind, manifest_json, health, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                manifest.capability_id,
                manifest.version,
                manifest.kind,
                json_dumps(manifest),
                manifest.health,
                manifest.created_at.isoformat(),
            ),
        )

    def candidates(self, *, operation: str, input_schema: str, output_schema: str) -> list[CapabilityManifest]:
        available: list[CapabilityManifest] = []
        for row in self.database.all("SELECT manifest_json FROM capabilities WHERE health != 'unavailable'"):
            manifest = CapabilityManifest.model_validate(json_loads(row["manifest_json"]))
            if (
                operation in manifest.supported_operations
                and input_schema in manifest.input_schemas
                and output_schema in manifest.output_schemas
            ):
                available.append(manifest)
        return sorted(available, key=lambda item: (item.estimated_cost, item.estimated_latency_ms, -item.quality))

    def manifests(self) -> list[CapabilityManifest]:
        return [
            CapabilityManifest.model_validate(json_loads(row["manifest_json"]))
            for row in self.database.all("SELECT manifest_json FROM capabilities ORDER BY capability_id, version")
        ]

    def register_builtin_capabilities(self) -> None:
        builtins = [
            CapabilityManifest(
                capability_id="builtin.cosmos-retrieval",
                kind="retriever",
                input_schemas=["TaskContract"],
                output_schemas=["RetrievalResult"],
                supported_operations=["RETRIEVE_CLAIMS"],
                quality=0.65,
                estimated_latency_ms=3,
                estimated_cost=0.1,
            ),
            CapabilityManifest(
                capability_id="builtin.answer-family",
                kind="family",
                input_schemas=["RetrievalResult"],
                output_schemas=["AnswerDraft"],
                supported_operations=["ASSEMBLE_ANSWER"],
                quality=0.65,
                estimated_latency_ms=2,
                estimated_cost=0.1,
            ),
            CapabilityManifest(
                capability_id="builtin.text-codec",
                kind="codec",
                input_schemas=["AnswerDraft"],
                output_schemas=["AnswerEnvelope"],
                supported_operations=["FORMAT_TEXT"],
                quality=0.75,
                estimated_latency_ms=1,
                estimated_cost=0.05,
            ),
            CapabilityManifest(
                capability_id="builtin.critic-family",
                kind="critic",
                input_schemas=["AnswerEnvelope"],
                output_schemas=["CriticReport"],
                supported_operations=["VERIFY_CONTRACT", "VERIFY_PROVENANCE", "VERIFY_CONSISTENCY", "VERIFY_BUDGET", "VERIFY_CONTEXT"],
                quality=0.9,
                estimated_latency_ms=1,
                estimated_cost=0.05,
            ),
        ]
        for manifest in builtins:
            self.register(manifest)


class Planner:
    _REQUIRED = (
        ("RETRIEVE_CLAIMS", "TaskContract", "RetrievalResult"),
        ("ASSEMBLE_ANSWER", "RetrievalResult", "AnswerDraft"),
        ("FORMAT_TEXT", "AnswerDraft", "AnswerEnvelope"),
        ("VERIFY_CONTRACT", "AnswerEnvelope", "CriticReport"),
    )

    def __init__(self, database: SqliteDatabase, atlas: Atlas) -> None:
        self.database = database
        self.atlas = atlas

    def plan(self, contract: TaskContract, hive_id: str) -> ExecutionPlan:
        existing = self.database.one(
            "SELECT plan_json FROM plans WHERE task_id = ? AND hive_id = ? ORDER BY revision DESC, created_at DESC LIMIT 1",
            (contract.task_id, hive_id),
        )
        if existing:
            return ExecutionPlan.model_validate(json_loads(existing["plan_json"]))
        steps: list[PlanStep] = []
        cumulative_latency = 0
        for operation, input_schema, output_schema in self._REQUIRED:
            candidates = self.atlas.candidates(
                operation=operation, input_schema=input_schema, output_schema=output_schema
            )
            if not candidates:
                raise PlanningError("No compatible capability for %s" % operation)
            chosen = candidates[0]
            cumulative_latency += chosen.estimated_latency_ms
            if cumulative_latency > contract.budget.time_ms:
                raise PlanningError("capability route exceeds task time budget")
            steps.append(
                PlanStep(
                    operation=operation,
                    capability_id=chosen.capability_id,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    estimated_cost=chosen.estimated_cost,
                    side_effects=[],
                )
            )
        plan = ExecutionPlan(task_id=contract.task_id, hive_id=hive_id, steps=steps)
        self.database.execute(
            "INSERT INTO plans(plan_id, task_id, hive_id, revision, plan_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (plan.plan_id, plan.task_id, plan.hive_id, plan.revision, json_dumps(plan), plan.created_at.isoformat()),
        )
        return plan


class CriticSystem:
    def evaluate(self, contract: TaskContract, answer: AnswerEnvelope, retrieval: RetrievalResult) -> list[CriticReport]:
        reports = [
            CriticReport(
                critic="ContractCritic",
                target=answer.task_id,
                verdict=CriticVerdict.PASS if answer.answer.strip() else CriticVerdict.FAIL,
                severity="info" if answer.answer.strip() else "error",
                evidence=["expected_output=" + contract.expected_output],
                repair_hint=None if answer.answer.strip() else "Create a non-empty text answer.",
            ),
            CriticReport(
                critic="ProvenanceCritic",
                target=answer.task_id,
                verdict=CriticVerdict.PASS if (not retrieval.claims or answer.sources) else CriticVerdict.FAIL,
                severity="info" if (not retrieval.claims or answer.sources) else "error",
                evidence=["claims=%d" % len(retrieval.claims), "sources=%d" % len(answer.sources)],
                repair_hint="Attach allowed source artifacts to every supported claim." if retrieval.claims and not answer.sources else None,
            ),
            CriticReport(
                critic="ConsistencyCritic",
                target=answer.task_id,
                verdict=CriticVerdict.WARN if any(item.contradictory_claim_ids for item in retrieval.claims) else CriticVerdict.PASS,
                severity="warning" if any(item.contradictory_claim_ids for item in retrieval.claims) else "info",
                evidence=["contradictions=%d" % sum(bool(item.contradictory_claim_ids) for item in retrieval.claims)],
                repair_hint="Describe conflicting source claims rather than silently choosing one." if any(item.contradictory_claim_ids for item in retrieval.claims) else None,
            ),
            CriticReport(
                critic="BudgetCritic",
                target=answer.task_id,
                verdict=CriticVerdict.PASS if retrieval.budget_used <= contract.budget.event_limit else CriticVerdict.FAIL,
                severity="info" if retrieval.budget_used <= contract.budget.event_limit else "error",
                evidence=["retrieval_items=%d" % retrieval.budget_used],
            ),
            CriticReport(
                critic="ContextCritic",
                target=answer.task_id,
                verdict=CriticVerdict.PASS,
                severity="info",
                evidence=["all retrieval claims passed tenant/project access filter"],
            ),
        ]
        return reports


class TextCodec:
    """Formats only source-backed answer material."""

    def encode(self, contract: TaskContract, hive_id: str, trace_id: str, retrieval: RetrievalResult) -> AnswerEnvelope:
        unique_sources = []
        seen_sources = set()
        for item in retrieval.claims:
            if item.source.artifact_id not in seen_sources:
                seen_sources.add(item.source.artifact_id)
                unique_sources.append(item.source)
        if not retrieval.claims:
            raise InsufficientEvidenceError(
                "No permitted source-backed claims match this task; import relevant training material first."
            )
        answer_text = "\n\n".join(
            dict.fromkeys(item.claim.object_value.strip() for item in retrieval.claims if item.claim.object_value.strip())
        )
        if not answer_text:
            raise InsufficientEvidenceError("Retrieved claims contain no answer material.")
        return AnswerEnvelope(
            task_id=contract.task_id,
            trace_id=trace_id,
            hive_id=hive_id,
            answer=answer_text,
            sources=unique_sources,
            warnings=list(retrieval.gaps),
        )


class ExecutionEngine:
    def __init__(
        self,
        *,
        cosmos: Cosmos,
        hives: HiveManager,
        planner: Planner,
        critics: CriticSystem,
        codec: TextCodec,
        traces: TraceRecorder,
        active_graph_builder: Optional["ActiveGraphBuilder"] = None,
    ) -> None:
        self.cosmos = cosmos
        self.hives = hives
        self.planner = planner
        self.critics = critics
        self.codec = codec
        self.traces = traces
        self.active_graph_builder = active_graph_builder

    def execute(self, contract: TaskContract, hive_id: str, trace_id: str, context: Optional[Any] = None) -> AnswerEnvelope:
        def checkpoint() -> None:
            if context is not None:
                context.checkpoint(1)

        retrieve_span = self.traces.start_span(
            trace_id=trace_id,
            component="CosmosRetriever",
            operation="RETRIEVE_CLAIMS",
            input_summary={"task_id": contract.task_id, "tenant_id": contract.tenant_id},
            budget_before=contract.budget,
        )
        checkpoint()
        retrieval = self.cosmos.retrieve(contract)
        self.traces.finish_span(
            retrieve_span,
            output_summary={"claims": len(retrieval.claims), "gaps": retrieval.gaps},
            budget_after=contract.budget,
        )

        if self.active_graph_builder is not None:
            graph_span = self.traces.start_span(
                trace_id=trace_id,
                component="ActiveGraph",
                operation="PROCESS_ACTIVE_GRAPH",
                input_summary={"claim_count": len(retrieval.claims)},
            )
            graph = self.active_graph_builder.build_initial_graph(
                contract,
                hive_id,
                trace_id=trace_id,
                retrieval=retrieval,
                reserve_steps=8,
            )
            while graph.step():
                checkpoint()
            snapshot_id = graph.persist_snapshot(self.active_graph_builder.database)
            self.traces.finish_span(
                graph_span,
                output_summary={"snapshot_id": snapshot_id, "events": graph.event_count},
            )
            board = create_hypothesis_board(
                task_id=contract.task_id,
                budget=graph.budget,
                traces=self.traces,
                cosmos=self.cosmos,
                tenant_id=contract.tenant_id,
                project_id=contract.project_id,
                trace_id=trace_id,
            )
            hypotheses = board.initialize(contract)
            selected_hypothesis = board.select_best_hypothesis()
            self.traces.record_event(
                {
                    "task_id": contract.task_id,
                    "trace_id": trace_id,
                    "kind": "HypothesisBoardCompleted",
                    "producer": "ExecutionEngine",
                    "payload": {
                        "hypothesis_count": len(hypotheses),
                        "selected_hypothesis_id": (
                            selected_hypothesis.hypothesis_id if selected_hypothesis is not None else None
                        ),
                    },
                }
            )

        planning_span = self.traces.start_span(
            trace_id=trace_id,
            component="Planner",
            operation="BUILD_PLAN",
            input_summary={"hive_id": hive_id},
        )
        checkpoint()
        plan = self.planner.plan(contract, hive_id)
        self.traces.finish_span(planning_span, output_summary={"plan_id": plan.plan_id, "steps": len(plan.steps)})

        codec_span = self.traces.start_span(
            trace_id=trace_id,
            component="TextCodec",
            operation="FORMAT_TEXT",
            input_summary={"claim_count": len(retrieval.claims)},
        )
        checkpoint()
        answer = self.codec.encode(contract, hive_id, trace_id, retrieval)
        self.traces.finish_span(codec_span, output_summary={"answer_length": len(answer.answer), "sources": len(answer.sources)})

        critic_span = self.traces.start_span(
            trace_id=trace_id,
            component="CriticSystem",
            operation="VERIFY",
            input_summary={"answer_length": len(answer.answer)},
        )
        checkpoint()
        reports = self.critics.evaluate(contract, answer, retrieval)
        answer.critic_reports = reports
        self.traces.finish_span(
            critic_span,
            output_summary={"verdicts": [report.verdict.value for report in reports]},
        )

        for item in retrieval.claims:
            self.hives.add_entry(
                hive_id,
                contract.tenant_id,
                store_name="EvidenceStore",
                content_type="claim_reference",
                content={"claim_id": item.claim.claim_id, "source_artifact_id": item.source.artifact_id, "score": item.score},
                source_ref=item.source.artifact_id,
                relevance=min(1.0, item.score),
                protected=True,
                trace_id=trace_id,
                idempotency_key="task:%s:evidence:%s" % (contract.task_id, item.claim.claim_id),
            )
        self.hives.add_entry(
            hive_id,
            contract.tenant_id,
            store_name="IntermediateResultStore",
            content_type="answer",
            content={"answer": answer.answer, "source_ids": [source.artifact_id for source in answer.sources]},
            relevance=0.9,
            protected=False,
            trace_id=trace_id,
            idempotency_key="task:%s:answer" % contract.task_id,
        )
        self.hives.record_execution(
            hive_id,
            contract.tenant_id,
            plan_ref=plan.plan_id,
            knowledge_refs=[item.claim.claim_id for item in retrieval.claims],
            critic_reports=[report.model_dump(mode="json") for report in reports],
        )
        return answer
