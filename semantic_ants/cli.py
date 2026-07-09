from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn

from .engine import EngineConfig, SemanticEngine
from .server.app import create_app
from .server.service import ServerConfig


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.command == "serve":
        config = ServerConfig(state_dir=Path(args.state_dir), host=args.host, port=args.port, static_dir=Path(args.static_dir) if args.static_dir else None)
        app = create_app(config)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return 0
    if args.command == "train":
        engine = SemanticEngine(config=EngineConfig(state_dir=Path(args.state_dir)))
        progress = _train_progress_printer()
        if args.dataset:
            report = engine.train_jsonl(
                Path(args.dataset),
                session_id=args.session_id,
                epochs=args.epochs,
                max_pairs=args.max_pairs,
                max_records=args.max_records,
                batch_size=args.batch_size,
                progress_callback=progress,
            )
        else:
            report = engine.train_text(
                args.text,
                session_id=args.session_id,
                epochs=args.epochs,
                max_records=args.max_records,
                progress_callback=progress,
            )
        _print_payload(report, args.json)
        return 0
    if args.command == "chat":
        engine = SemanticEngine(config=EngineConfig(state_dir=Path(args.state_dir)))
        payload = engine.chat(args.text, session_id=args.session_id, backpack_limit=args.backpack_limit)
        _print_payload(payload, args.json)
        return 0
    if args.command == "graph":
        engine = SemanticEngine(config=EngineConfig(state_dir=Path(args.state_dir)))
        payload = engine.graph(query=args.query, limit=args.limit, result_id=args.result_id)
        _print_payload(payload, args.json)
        return 0
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantic_ants")
    parser.add_argument("--state-dir", default=".semantic_ants")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--static-dir", default="")

    train = sub.add_parser("train")
    train_input = train.add_mutually_exclusive_group(required=True)
    train_input.add_argument("--text")
    train_input.add_argument("--dataset")
    train.add_argument("--session-id", default="default")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=5000, help="Records between save checkpoints")
    train.add_argument("--max-pairs", "--max_pairs", dest="max_pairs", type=int, default=None)
    train.add_argument("--max-records", "--max_records", dest="max_records", type=int, default=None, help="Stop after processing this many records (0 = unlimited)")
    train.add_argument("--json", action="store_true")

    chat = sub.add_parser("chat")
    chat.add_argument("--text", required=True)
    chat.add_argument("--session-id", default="default")
    chat.add_argument("--backpack-limit", type=int, default=None)
    chat.add_argument("--json", action="store_true")

    graph = sub.add_parser("graph")
    graph.add_argument("--query", default=None)
    graph.add_argument("--result-id", default=None)
    graph.add_argument("--limit", type=int, default=120)
    graph.add_argument("--json", action="store_true")

    return parser


def _print_payload(payload: object, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(payload)


def _train_progress_printer():
    def _print(event: dict[str, object]) -> None:
        phase = str(event.get("phase") or "")
        kind = str(event.get("kind") or "train")
        session_id = str(event.get("session_id") or "default")
        if phase == "start":
            if kind == "preprocess":
                print(
                    f"[preprocess] started input={event.get('input_path')} output={event.get('output_path')} max_pairs={event.get('max_pairs')}",
                    file=sys.stderr,
                    flush=True,
                )
            elif kind == "jsonl":
                print(
                    f"[train] started jsonl dataset={event.get('dataset_path')} session={session_id} epochs={event.get('epochs')}",
                    file=sys.stderr,
                    flush=True,
                )
            elif kind == "hierarchy_jsonl":
                print(
                    f"[train] started hierarchy dataset={event.get('dataset_path')} session={session_id} epochs={event.get('epochs')}",
                    file=sys.stderr,
                    flush=True,
                )
            elif kind == "corpus":
                print(
                    f"[train] started corpus dataset={event.get('dataset_path')} corpus={event.get('corpus_path')} session={session_id} epochs={event.get('epochs')}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[train] started text session={session_id} epochs={event.get('epochs')} length={event.get('text_length')}",
                    file=sys.stderr,
                    flush=True,
                )
            return
        if phase == "progress":
            if kind == "preprocess":
                print(
                    f"[preprocess] running input={event.get('input_path')} output={event.get('output_path')} scanned={event.get('scanned')} accepted={event.get('accepted')} hot={event.get('skipped_hot')}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[train] running {kind} dataset={event.get('dataset_path')} session={session_id} records={event.get('records')} sequences={event.get('sequences')}",
                    file=sys.stderr,
                    flush=True,
                )
            return
        if phase == "done":
            if kind == "preprocess":
                print(
                    f"[preprocess] done input={event.get('input_path')} output={event.get('output_path')} accepted={event.get('accepted')} skipped={event.get('skipped_spam')}",
                    file=sys.stderr,
                    flush=True,
                )
            elif kind == "jsonl":
                print(
                    f"[train] done jsonl dataset={event.get('dataset_path')} session={session_id} records={event.get('dataset_records')} tokens={event.get('tokens')} edges={event.get('edges')} sequences={event.get('sequences')}",
                    file=sys.stderr,
                    flush=True,
                )
            elif kind == "hierarchy_jsonl":
                print(
                    f"[train] done hierarchy dataset={event.get('dataset_path')} session={session_id} records={event.get('dataset_records')} tokens={event.get('tokens')} edges={event.get('edges')} sequences={event.get('sequences')}",
                    file=sys.stderr,
                    flush=True,
                )
            elif kind == "corpus":
                print(
                    f"[train] done corpus dataset={event.get('dataset_path')} session={session_id} records={event.get('dataset_records')} tokens={event.get('tokens')} edges={event.get('edges')} sequences={event.get('sequences')}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[train] done text session={session_id} tokens={event.get('tokens')} edges={event.get('edges')} sequences={event.get('sequences')}",
                    file=sys.stderr,
                    flush=True,
                )

    return _print
