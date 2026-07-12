"""Small local runner; production deployment owns process supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import Budget, new_id, utcnow
from .service import ServiceConfig, SuperAIService


def _create_service() -> SuperAIService:
    return SuperAIService(ServiceConfig.from_environment())


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    uvicorn.run("superai.api:app", host=args.host, port=args.port, reload=args.reload)


def _cmd_benchmark(args: argparse.Namespace) -> None:
    """Run a baseline benchmark."""
    service = _create_service()
    try:
        from .benchmark import create_default_benchmark_runner

        runner = create_default_benchmark_runner(
            database=service.database,
            store=service.store,
            traces=service.traces,
            cosmos=service.cosmos,
            hives=service.hives,
            execution_engine=service.execution,
            runtime=service.runtime,
        )

        manifest = runner.create_manifest(
            name=args.name,
            description=args.description,
            dataset_version=args.dataset_version,
            task_generator="sequence_transform",
            task_generator_params={"num_tasks": args.num_tasks},
            seed=args.seed,
        )

        print(f"Created manifest: {manifest.manifest_id}")
        print(f"Git revision: {manifest.git_revision}")
        print(f"Manifest hash: {manifest.to_hash()}")

        run = runner.run_benchmark(
            manifest=manifest,
            tenant_id="local",
            project_id=None,
            mode=args.mode,
        )

        print(f"\nBenchmark run completed: {run.run_id}")
        print(f"Status: {run.status}")
        print(f"Accuracy: {run.quality:.3f}")
        print(f"Avg Latency: {run.latency_ms}ms")
        print(f"Cost: {run.cost:.4f}")

    finally:
        service.close()


def _cmd_replay(args: argparse.Namespace) -> None:
    """Replay a task by trace ID."""
    service = _create_service()
    try:
        task = service.task(args.task_id, "local")
        trace = service.trace(task.trace_id, "local")
        print(json.dumps(trace, indent=2, default=str))
    finally:
        service.close()


def _cmd_health(args: argparse.Namespace) -> None:
    """Check service health."""
    service = _create_service()
    try:
        health = service.health()
        print(json.dumps(health, indent=2))
    finally:
        service.close()


def _cmd_meta(args: argparse.Namespace) -> None:
    """Show service metadata."""
    service = _create_service()
    try:
        meta = service.meta()
        print(json.dumps(meta, indent=2))
    finally:
        service.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SuperAI CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run the API server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=8000, type=int)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.set_defaults(func=_cmd_serve)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run a benchmark")
    bench_parser.add_argument("--name", default="omega-baseline", help="Benchmark name")
    bench_parser.add_argument("--description", default="ΩE baseline benchmark", help="Description")
    bench_parser.add_argument("--dataset-version", default="1.0", help="Dataset version")
    bench_parser.add_argument("--num-tasks", default=20, type=int, help="Number of tasks")
    bench_parser.add_argument("--seed", default=42, type=int, help="Random seed")
    bench_parser.add_argument("--mode", default="baseline", choices=["baseline", "treatment", "holdout", "ablation"], help="Benchmark mode")
    bench_parser.set_defaults(func=_cmd_benchmark)

    # Replay command
    replay_parser = subparsers.add_parser("replay", help="Replay a task trace")
    replay_parser.add_argument("task_id", help="Task ID to replay")
    replay_parser.set_defaults(func=_cmd_replay)

    # Health command
    health_parser = subparsers.add_parser("health", help="Check service health")
    health_parser.set_defaults(func=_cmd_health)

    # Meta command
    meta_parser = subparsers.add_parser("meta", help="Show service metadata")
    meta_parser.set_defaults(func=_cmd_meta)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
