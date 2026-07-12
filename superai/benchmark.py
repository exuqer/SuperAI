"""Benchmark framework for ΩE: reproducible evaluation with holdout and ablation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .contracts import (
    AccessScope,
    ArtifactRef,
    BenchmarkRun,
    Budget,
    DomainEvent,
    TaskContract,
    TaskState,
    new_id,
    utcnow,
)
from .cosmos import Cosmos
from .database import SqliteDatabase, json_dumps, json_loads
from .execution import ExecutionEngine
from .hive import HiveManager
from .observability import TraceRecorder
from .runtime import CommandRuntime
from .storage import ObjectStore


@dataclass
class BenchmarkManifest:
    """Immutable manifest defining a benchmark dataset and evaluation protocol."""
    manifest_id: str
    name: str
    description: str
    dataset_version: str
    train_split: float
    validation_split: float
    holdout_split: float
    seed: int
    metrics: List[str]  # Metric names to compute
    task_generator: str  # Name of registered task generator
    task_generator_params: Dict[str, Any]
    created_at: datetime = field(default_factory=utcnow)
    git_revision: str = ""

    def to_hash(self) -> str:
        """Compute deterministic hash of manifest (excluding timestamps)."""
        data = {
            "manifest_id": self.manifest_id,
            "name": self.name,
            "description": self.description,
            "dataset_version": self.dataset_version,
            "train_split": self.train_split,
            "validation_split": self.validation_split,
            "holdout_split": self.holdout_split,
            "seed": self.seed,
            "metrics": self.metrics,
            "task_generator": self.task_generator,
            "task_generator_params": self.task_generator_params,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


class TaskGenerator:
    """Base class for generating benchmark tasks."""

    def generate(self, params: Dict[str, Any], seed: int) -> List[TaskContract]:
        raise NotImplementedError


class SequenceTransformGenerator(TaskGenerator):
    """Generates sequence transformation tasks with hidden rules."""

    def generate(self, params: Dict[str, Any], seed: int) -> List[TaskContract]:
        import random
        rng = random.Random(seed)

        num_tasks = params.get("num_tasks", 50)
        sequence_length = params.get("sequence_length", 10)
        value_range = params.get("value_range", 20)

        # Hidden rules (unknown to the system)
        rules = [
            ("reverse", lambda seq: list(reversed(seq))),
            ("sort", lambda seq: sorted(seq)),
            ("increment", lambda seq: [x + 1 for x in seq]),
            ("double", lambda seq: [x * 2 for x in seq]),
            ("filter_even", lambda seq: [x for x in seq if x % 2 == 0]),
            ("filter_odd", lambda seq: [x for x in seq if x % 2 == 1]),
            ("square", lambda seq: [x * x for x in seq]),
            ("mod_3", lambda seq: [x % 3 for x in seq]),
        ]

        tasks = []
        for i in range(num_tasks):
            # Pick a rule (some tasks share rules)
            rule_name, rule_fn = rng.choice(rules)

            # Generate input sequence
            input_seq = [rng.randint(1, value_range) for _ in range(sequence_length)]
            output_seq = rule_fn(input_seq)

            # Create task
            goal = f"Transform the input sequence to the output sequence. Input: {input_seq}. Output: {output_seq}"
            contract = TaskContract(
                goal=goal,
                expected_output=str(output_seq),
                budget=Budget(time_ms=30000, step_limit=50),
            )
            tasks.append(contract)

        return tasks


class BenchmarkRunner:
    """Runs benchmarks with full reproducibility guarantees."""

    def __init__(
        self,
        database: SqliteDatabase,
        store: ObjectStore,
        traces: TraceRecorder,
        cosmos: Cosmos,
        hives: HiveManager,
        execution_engine: ExecutionEngine,
        runtime: CommandRuntime,
    ):
        self.database = database
        self.store = store
        self.traces = traces
        self.cosmos = cosmos
        self.hives = hives
        self.execution_engine = execution_engine
        self.runtime = runtime
        self.task_generators: Dict[str, TaskGenerator] = {
            "sequence_transform": SequenceTransformGenerator(),
        }

    def register_generator(self, name: str, generator: TaskGenerator) -> None:
        self.task_generators[name] = generator

    def create_manifest(
        self,
        name: str,
        description: str,
        dataset_version: str,
        task_generator: str,
        task_generator_params: Dict[str, Any],
        train_split: float = 0.6,
        validation_split: float = 0.2,
        holdout_split: float = 0.2,
        seed: int = 42,
        metrics: Optional[List[str]] = None,
    ) -> BenchmarkManifest:
        """Create a new benchmark manifest."""
        manifest = BenchmarkManifest(
            manifest_id=new_id("bench"),
            name=name,
            description=description,
            dataset_version=dataset_version,
            train_split=train_split,
            validation_split=validation_split,
            holdout_split=holdout_split,
            seed=seed,
            metrics=metrics or ["accuracy", "latency_ms", "cost"],
            task_generator=task_generator,
            task_generator_params=task_generator_params,
            git_revision=self._get_git_revision(),
        )
        return manifest

    def _get_git_revision(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )
            return result.stdout.strip()[:12]
        except Exception:
            return "unknown"

    def run_benchmark(
        self,
        manifest: BenchmarkManifest,
        tenant_id: str = "local",
        project_id: Optional[str] = None,
        mode: str = "baseline",  # baseline, treatment, ablation, random
        concept_id: Optional[str] = None,
    ) -> BenchmarkRun:
        """Run a full benchmark evaluation."""
        run_id = new_id("bench_run")

        # Generate all tasks deterministically
        generator = self.task_generators.get(manifest.task_generator)
        if not generator:
            raise ValueError(f"Unknown task generator: {manifest.task_generator}")

        all_tasks = generator.generate(manifest.task_generator_params, manifest.seed)

        # Split tasks deterministically
        train_count = int(len(all_tasks) * manifest.train_split)
        val_count = int(len(all_tasks) * manifest.validation_split)

        if mode == "baseline":
            eval_tasks = all_tasks[train_count:train_count + val_count]  # Validation split
        elif mode == "treatment":
            eval_tasks = all_tasks[train_count + val_count:]  # Holdout split
        elif mode == "holdout":
            eval_tasks = all_tasks[train_count + val_count:]  # Holdout split
        elif mode == "ablation":
            eval_tasks = all_tasks[train_count:train_count + val_count]  # Validation
        else:
            eval_tasks = all_tasks[train_count:train_count + val_count]

        # Create benchmark run record
        run = BenchmarkRun(
            run_id=run_id,
            task_id="",  # Will be set per task
            tenant_id=tenant_id,
            project_id=project_id,
            git_revision=manifest.git_revision,
            config_hash=manifest.to_hash(),
            dataset_version=manifest.dataset_version,
            seed=manifest.seed,
            status="running",
            mode=mode,
            concept_id=concept_id,
        )

        # Persist initial run record
        self._persist_run(run)

        # Run tasks
        total_latency = 0
        total_cost = 0.0
        correct = 0

        for task_contract in eval_tasks:
            task_contract.tenant_id = tenant_id
            task_contract.project_id = project_id

            # Select/create hive
            trace_id = new_id("trace")
            hive, decision, _ = self.hives.select_or_create(task_contract, trace_id)

            # Execute task
            start = datetime.now()
            answer = None
            try:
                answer = self.execution_engine.execute(task_contract, hive.hive_id, trace_id)
            except Exception:
                # A benchmark records unsupported tasks as incorrect rather
                # than aborting the entire reproducible evaluation run.
                pass
            latency = int((datetime.now() - start).total_seconds() * 1000)

            # Evaluate
            expected = task_contract.expected_output.strip()
            actual = answer.answer.strip() if answer is not None else ""
            is_correct = answer is not None and (expected == actual or expected in actual)

            if is_correct:
                correct += 1

            total_latency += latency
            total_cost += 0.01  # Placeholder cost

            # Record task result
            if answer is not None:
                self._record_task_result(task_contract, answer, hive.hive_id, trace_id, latency, is_correct)

        # Compute metrics
        accuracy = correct / max(1, len(eval_tasks))
        avg_latency = total_latency / max(1, len(eval_tasks))

        # Update run record
        run.latency_ms = avg_latency
        run.cost = total_cost
        run.quality = accuracy
        run.status = "completed"
        self._persist_run(run)

        # Record benchmark event
        event = DomainEvent(
            id=new_id("evt"),
            task_id=run_id,
            trace_id=run_id,
            tenant_id=tenant_id,
            kind="BenchmarkCompleted",
            producer="BenchmarkRunner",
            payload={
                "run_id": run_id,
                "mode": mode,
                "manifest_id": manifest.manifest_id,
                "tasks_evaluated": len(eval_tasks),
                "accuracy": accuracy,
                "avg_latency_ms": avg_latency,
                "total_cost": total_cost,
                "concept_id": concept_id,
            },
        )
        self.traces.record_event(event)

        return run

    def run_comparison(
        self,
        manifest: BenchmarkManifest,
        baseline_run_id: str,
        treatment_run_id: str,
        ablation_run_id: str,
    ) -> Dict[str, float]:
        """Compare baseline, treatment, and ablation runs."""
        baseline = self._load_run(baseline_run_id)
        treatment = self._load_run(treatment_run_id)
        ablation = self._load_run(ablation_run_id)

        return {
            "quality_delta": treatment.quality - baseline.quality,
            "cost_delta": treatment.cost - baseline.cost,
            "ablation_delta": baseline.quality - ablation.quality,  # Positive = concept helped
            "baseline_quality": baseline.quality,
            "treatment_quality": treatment.quality,
            "ablation_quality": ablation.quality,
        }

    def _persist_run(self, run: BenchmarkRun) -> None:
        with self.database.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO benchmark_runs
                   (run_id, task_id, tenant_id, project_id, git_revision, config_hash,
                    dataset_version, seed, latency_ms, cost, quality, status, mode, concept_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.run_id,
                    run.task_id,
                    run.tenant_id,
                    run.project_id,
                    run.git_revision,
                    run.config_hash,
                    run.dataset_version,
                    run.seed,
                    run.latency_ms,
                    run.cost,
                    run.quality,
                    run.status,
                    run.mode,
                    run.concept_id,
                    run.created_at.isoformat(),
                ),
            )

    def _load_run(self, run_id: str) -> BenchmarkRun:
        row = self.database.one(
            "SELECT * FROM benchmark_runs WHERE run_id = ?",
            (run_id,),
        )
        if not row:
            raise KeyError(f"Benchmark run {run_id} not found")
        return BenchmarkRun(
            run_id=row["run_id"],
            task_id=row["task_id"],
            tenant_id=row["tenant_id"],
            project_id=row["project_id"],
            git_revision=row["git_revision"],
            config_hash=row["config_hash"],
            dataset_version=row["dataset_version"],
            seed=row["seed"],
            latency_ms=row["latency_ms"],
            cost=row["cost"],
            quality=row["quality"],
            status=row["status"],
            mode=row["mode"],
            concept_id=row["concept_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _record_task_result(
        self,
        contract: TaskContract,
        answer,
        hive_id: str,
        trace_id: str,
        latency: int,
        correct: bool,
    ) -> None:
        with self.database.transaction() as conn:
            conn.execute(
                """UPDATE tasks SET status = ?, answer_json = ?, updated_at = ?
                   WHERE task_id = ?""",
                (
                    TaskState.SUCCEEDED.value if correct else TaskState.FAILED.value,
                    json_dumps(answer.model_dump(mode="json")),
                    utcnow().isoformat(),
                    contract.task_id,
                ),
            )


def create_default_benchmark_runner(
    database: SqliteDatabase,
    store: ObjectStore,
    traces: TraceRecorder,
    cosmos: Cosmos,
    hives: HiveManager,
    execution_engine: ExecutionEngine,
    runtime: CommandRuntime,
) -> BenchmarkRunner:
    """Factory for creating a benchmark runner with default components."""
    return BenchmarkRunner(
        database=database,
        store=store,
        traces=traces,
        cosmos=cosmos,
        hives=hives,
        execution_engine=execution_engine,
        runtime=runtime,
    )
