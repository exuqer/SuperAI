from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from semantic_ants.datasets import download_koziev_dialogues_dataset, download_spc_dataset, download_tatoeba_translation_dataset
from semantic_ants.engine import EngineConfig, SemanticEngine
from semantic_ants.knowledge import bootstrap_builtin_knowledge
from semantic_ants.learning import ACOTrainer, CheckpointStore, FeedbackTrainer, Trainer, default_checkpoint_path
from semantic_ants.learning.checkpoint import migrate_checkpoint


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(_normalize_argv(argv or sys.argv[1:]))
    try:
        return args.handler(args)
    except Exception as exc:
        if getattr(args, "json", False):
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        else:
            print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


def analyze_command(args: argparse.Namespace) -> int:
    engine = _engine_from_args(args)
    result = engine.analyze(
        args.text,
        lang=args.lang,
        ant_count=args.ants,
        max_depth=args.depth,
        top_concepts=args.top_concepts,
        mode=args.mode,
        session_id=getattr(args, "session_id", None),
        reset_session=getattr(args, "reset_session", False),
        strength_vector=args.strength_vector,
    )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(result.response)
    print(result.summary)
    print(f"result_id: {result.result_id}")
    if args.trace:
        for route in result.routes[: min(8, len(result.routes))]:
            print(f"ant#{route.ant_id} score={route.total_score:.3f}: {' -> '.join(route.concepts)}")
    return 0


def train_command(args: argparse.Namespace) -> int:
    engine = _engine_from_args(args)
    report = Trainer(engine, engine.store).train_file(args.path, epochs=args.epochs)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"examples={report.examples} epochs={report.epochs} reinforced={report.reinforced_edges}")
        if report.errors:
            print("errors:")
            for error in report.errors:
                print(error)
    return 1 if report.errors else 0


def learn_command(args: argparse.Namespace) -> int:
    engine = _engine_from_args(args)
    report = ACOTrainer(engine, engine.store).learn_file(args.path, epochs=args.epochs)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(
            " ".join(
                [
                    f"examples={report.examples}",
                    f"epochs={report.epochs}",
                    f"reinforced={report.reinforced_edges}",
                    f"evaporated={report.evaporated_edges}",
                    f"bridges={report.learned_bridges}",
                    f"accepted={report.accepted_answers}",
                ]
            )
        )
        if report.errors:
            print("errors:")
            for error in report.errors:
                print(error)
    return 1 if report.errors else 0


def learn_dialogues_command(args: argparse.Namespace) -> int:
    engine = _engine_from_args(args)
    report = ACOTrainer(engine, engine.store).learn_dialogue_file(
        args.path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        torch_steps=args.torch_steps,
    )
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(
            " ".join(
                [
                    f"examples={report.examples}",
                    f"epochs={report.epochs}",
                    f"reinforced={report.reinforced_edges}",
                    f"evaporated={report.evaporated_edges}",
                    f"accepted={report.accepted_answers}",
                ]
            )
        )
        if report.errors:
            print("errors:")
            for error in report.errors:
                print(error)
    return 1 if report.errors else 0


def download_dataset_command(args: argparse.Namespace) -> int:
    if args.dataset == "spc":
        count = download_spc_dataset(args.split, args.output, limit=args.limit)
        payload = {"dataset": args.dataset, "split": args.split, "output": args.output, "examples": count}
    elif args.dataset == "koziev":
        count = download_koziev_dialogues_dataset(args.output, source=args.path, limit=args.limit, timeout=args.timeout)
        payload = {
            "dataset": args.dataset,
            "source": args.path,
            "output": args.output,
            "examples": count,
        }
    elif args.dataset == "tatoeba":
        count = download_tatoeba_translation_dataset(
            args.output,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            limit=args.limit,
            bidirectional=not args.no_bidirectional,
            timeout=args.timeout,
        )
        payload = {
            "dataset": args.dataset,
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
            "output": args.output,
            "examples": count,
        }
    else:
        raise ValueError(f"Неподдерживаемый датасет: {args.dataset}")
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else f"examples={count} output={args.output}")
    return 0


def feedback_command(args: argparse.Namespace) -> int:
    engine = _engine_from_args(args)
    corrected_concepts = _split_csv(args.corrected_concepts)
    result = FeedbackTrainer(engine, engine.store).apply(
        result_id=None if args.last else args.result_id,
        score=args.score,
        corrected_concepts=corrected_concepts,
        corrected_response=args.corrected_response,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
    return 0


def bootstrap_command(args: argparse.Namespace) -> int:
    engine = _engine_from_args(args)
    report = bootstrap_builtin_knowledge(
        engine.checkpoint,
        force=args.force,
        allow_network=not getattr(args, "no_cache_refresh", False),
    )
    engine.store.save(engine.checkpoint)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_dict())
    return 0


def chat_command(args: argparse.Namespace) -> int:
    engine = _engine_from_args(args)
    if args.reset_session:
        engine.checkpoint.reset_chat_session(args.session_id)
        engine.store.save(engine.checkpoint)
    if args.once:
        return _chat_once(engine, args.once, args)
    print("semantic_ants chat. Команды выхода: /exit, /quit, выход.")
    while True:
        try:
            text = input("Вы: ").strip()
        except EOFError:
            print()
            break
        if not text:
            continue
        if text.lower() in {"/exit", "/quit", "exit", "quit", "выход", "пока"}:
            if text.lower() == "пока":
                _chat_once(engine, text, args)
            break
        _chat_once(engine, text, args)
    return 0


def _chat_once(engine: SemanticEngine, text: str, args: argparse.Namespace) -> int:
    result = engine.analyze(
        text,
        lang=args.lang,
        ant_count=args.ants,
        max_depth=args.depth,
        top_concepts=args.top_concepts,
        mode=args.mode,
        session_id=args.session_id,
        strength_vector=args.strength_vector,
    )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"AI: {result.response}")
    if args.trace:
        print(result.summary)
        for route in result.routes[: min(5, len(result.routes))]:
            print(f"  ant#{route.ant_id} score={route.total_score:.3f}: {' -> '.join(route.concepts)}")
    return 0


def eval_command(args: argparse.Namespace) -> int:
    engine = _engine_from_args(args)
    total = 0
    hits = 0
    rows = []
    with Path(args.path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            example = json.loads(stripped)
            total += 1
            result = engine.analyze(
                str(example["text"]),
                lang=str(example.get("lang", "auto")),
                strength_vector=args.strength_vector,
            )
            expected = set(map(str, example.get("target_concepts", [])))
            actual = {item["uri"] for item in result.activated_concepts}
            ok = bool(expected & actual) if expected else True
            hits += int(ok)
            rows.append({"text": example["text"], "ok": ok, "expected": sorted(expected), "actual": sorted(actual)})
    payload = {"total": total, "hits": hits, "accuracy": hits / total if total else 0.0, "rows": rows}
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else payload)
    return 0


def dream_command(args: argparse.Namespace) -> int:
    engine = _engine_from_args(args)
    report = ACOTrainer(engine, engine.store).dream(steps=args.steps)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else report)
    return 0


def inspect_memory_command(args: argparse.Namespace) -> int:
    engine = _engine_from_args(args)
    payload = ACOTrainer(engine, engine.store).inspect_memory()
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else payload)
    return 0


def export_command(args: argparse.Namespace) -> int:
    store = CheckpointStore(default_checkpoint_path(args.state_dir))
    store.export(args.destination)
    print(args.destination)
    return 0


def migrate_memory_command(args: argparse.Namespace) -> int:
    store = CheckpointStore(default_checkpoint_path(args.state_dir))
    checkpoint = store.load()
    if args.backup:
        backup = store.path.with_name(f"{store.path.stem}.backup{store.path.suffix}")
        store.export(backup)
    before = checkpoint.to_dict()
    report = migrate_checkpoint(checkpoint)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "report": report, "version": before.get("version", 0)}, ensure_ascii=False, indent=2))
        return 0
    store.save(checkpoint)
    print(json.dumps({"dry_run": False, "report": report, "version": checkpoint.version}, ensure_ascii=False, indent=2) if args.json else report)
    return 0


def interpret_vector_command(args: argparse.Namespace) -> int:
    engine = _engine_from_args(args)
    raw = sys.stdin.read() if args.path == "-" else Path(args.path).read_text(encoding="utf-8")
    payload = json.loads(raw)
    response = engine.interpret_vector(payload)
    print(json.dumps({"response": response}, ensure_ascii=False, indent=2) if args.json else response)
    return 0


def _engine_from_args(args: argparse.Namespace) -> SemanticEngine:
    config = EngineConfig(
        state_dir=Path(args.state_dir),
        lang=getattr(args, "lang", "auto"),
        ant_count=getattr(args, "ants", 32),
        max_depth=getattr(args, "depth", 4),
        top_concepts=getattr(args, "top_concepts", 5),
        allow_network=not getattr(args, "no_cache_refresh", False),
        autoload_builtin=not getattr(args, "no_builtin", False),
        strength_vector=getattr(args, "strength_vector", ()),
    )
    return SemanticEngine(config=config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantic_ants")
    parser.add_argument("--state-dir", default=".semantic_ants")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("text")
    _add_runtime_args(analyze)
    analyze.add_argument("--mode", choices=["graph", "hybrid"], default="graph")
    analyze.add_argument("--session-id")
    analyze.add_argument("--reset-session", action="store_true")
    analyze.add_argument("--trace", action="store_true")
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(handler=analyze_command)

    train = subparsers.add_parser("train")
    train.add_argument("path")
    train.add_argument("--epochs", type=int, default=1)
    _add_runtime_args(train)
    train.add_argument("--json", action="store_true")
    train.set_defaults(handler=train_command)

    learn = subparsers.add_parser("learn")
    learn.add_argument("path")
    learn.add_argument("--epochs", type=int, default=1)
    _add_runtime_args(learn)
    learn.add_argument("--json", action="store_true")
    learn.set_defaults(handler=learn_command)

    learn_dialogues = subparsers.add_parser("learn-dialogues")
    learn_dialogues.add_argument("path")
    learn_dialogues.add_argument("--epochs", type=int, default=1)
    learn_dialogues.add_argument("--batch-size", type=int, default=32)
    learn_dialogues.add_argument("--max-examples", type=int)
    learn_dialogues.add_argument("--torch-steps", type=int, default=1)
    _add_runtime_args(learn_dialogues)
    learn_dialogues.add_argument("--json", action="store_true")
    learn_dialogues.set_defaults(handler=learn_dialogues_command)

    download_dataset = subparsers.add_parser("download-dataset")
    download_dataset.add_argument("dataset", choices=["spc", "koziev", "tatoeba"])
    download_dataset.add_argument("--split", choices=["train", "dev", "valid", "test", "synth"], default="train")
    download_dataset.add_argument("--path")
    download_dataset.add_argument("--source-lang", default="ru")
    download_dataset.add_argument("--target-lang", default="en")
    download_dataset.add_argument("--no-bidirectional", action="store_true")
    download_dataset.add_argument("--timeout", type=float, default=60.0)
    download_dataset.add_argument("--limit", type=int)
    download_dataset.add_argument("--output", required=True)
    download_dataset.add_argument("--json", action="store_true")
    download_dataset.set_defaults(handler=download_dataset_command)

    bootstrap = subparsers.add_parser("bootstrap")
    _add_runtime_args(bootstrap)
    bootstrap.add_argument("--force", action="store_true")
    bootstrap.add_argument("--json", action="store_true")
    bootstrap.set_defaults(handler=bootstrap_command)

    chat = subparsers.add_parser("chat")
    _add_runtime_args(chat)
    chat.add_argument("--once")
    chat.add_argument("--mode", choices=["graph", "hybrid"], default="graph")
    chat.add_argument("--session-id", default="default")
    chat.add_argument("--reset-session", action="store_true")
    chat.add_argument("--trace", action="store_true")
    chat.add_argument("--json", action="store_true")
    chat.set_defaults(handler=chat_command)

    feedback = subparsers.add_parser("feedback")
    feedback.add_argument("--last", action="store_true")
    feedback.add_argument("--result-id")
    feedback.add_argument("--score", type=int, required=True)
    feedback.add_argument("--corrected-concepts")
    feedback.add_argument("--corrected-response")
    feedback.add_argument("--json", action="store_true")
    feedback.set_defaults(handler=feedback_command)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("path")
    _add_runtime_args(evaluate)
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(handler=eval_command)

    dream = subparsers.add_parser("dream")
    dream.add_argument("--steps", type=int, default=100)
    _add_runtime_args(dream)
    dream.add_argument("--json", action="store_true")
    dream.set_defaults(handler=dream_command)

    inspect_memory = subparsers.add_parser("inspect-memory")
    inspect_memory.add_argument("--json", action="store_true")
    inspect_memory.add_argument("--no-cache-refresh", action="store_true")
    inspect_memory.add_argument("--no-builtin", action="store_true")
    inspect_memory.set_defaults(handler=inspect_memory_command)

    interpret_vector = subparsers.add_parser("interpret-vector")
    interpret_vector.add_argument("path", nargs="?", default="-")
    interpret_vector.add_argument("--json", action="store_true")
    interpret_vector.add_argument("--no-cache-refresh", action="store_true")
    interpret_vector.add_argument("--no-builtin", action="store_true")
    interpret_vector.set_defaults(handler=interpret_vector_command)

    export = subparsers.add_parser("export")
    export.add_argument("destination")
    export.set_defaults(handler=export_command)

    migrate_memory = subparsers.add_parser("migrate-memory")
    migrate_memory.add_argument("--dry-run", action="store_true")
    migrate_memory.add_argument("--apply", action="store_true")
    migrate_memory.add_argument("--backup", action="store_true")
    migrate_memory.add_argument("--json", action="store_true")
    migrate_memory.set_defaults(handler=migrate_memory_command)
    return parser


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lang", choices=["auto", "ru", "en"], default="auto")
    parser.add_argument("--ants", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--top-concepts", type=int, default=5)
    parser.add_argument("--strength-vector", type=_strength_vector, default=())
    parser.add_argument("--no-cache-refresh", action="store_true")
    parser.add_argument("--no-builtin", action="store_true")


def _normalize_argv(argv: list[str]) -> list[str]:
    commands = {
        "analyze",
        "bootstrap",
        "chat",
        "dream",
        "download-dataset",
        "eval",
        "export",
        "feedback",
        "inspect-memory",
        "interpret-vector",
        "learn",
        "learn-dialogues",
        "migrate-memory",
        "train",
    }
    if argv and argv[0] not in commands and not argv[0].startswith("-"):
        return ["analyze", *argv]
    return argv


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _strength_vector(value: str) -> tuple[int, ...]:
    try:
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        if not parts:
            raise ValueError
        return tuple(max(int(part), 0) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ожидается список целых чисел, например 3 или 3,8") from exc
