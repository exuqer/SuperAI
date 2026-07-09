from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from .engine import EngineConfig, SemanticEngine
from .state import save_checkpoint

DEFAULT_BATCH_SIZE = 5000
VALID_INPUT_FORMATS = {"auto", "jsonl", "txt"}


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r", " ").split()).strip()


def detect_input_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return "jsonl"
    if suffix in {".txt", ".text"}:
        return "txt"
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                return "txt"
            return "jsonl"
    return "txt"


def iter_text_blocks(path: Path) -> Iterator[str]:
    buffer: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                buffer.append(line)
                continue
            if buffer:
                yield "\n".join(buffer)
                buffer.clear()
        if buffer:
            yield "\n".join(buffer)


def iter_jsonl_texts(path: Path, engine: SemanticEngine) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            yield engine._jsonl_record_to_text(record)


def iter_jsonl_records(path: Path) -> Iterator[dict[str, Any] | None]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                yield None
                continue
            yield record if isinstance(record, dict) else None


def iter_source_texts(path: Path, engine: SemanticEngine, input_format: str) -> Iterator[str]:
    if input_format == "jsonl":
        yield from iter_jsonl_texts(path, engine)
        return
    yield from iter_text_blocks(path)


def iter_batches(items: Iterator[str], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def train_mass_dataset(
    input_path: str | Path,
    *,
    state_dir: str | Path = ".semantic_ants",
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = 1,
    input_format: str = "auto",
    session_id: str = "mass_train",
    max_records: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    dataset_path = Path(input_path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if dataset_path.is_dir():
        raise IsADirectoryError(dataset_path)
    state_path = Path(state_dir)
    batch_size = max(int(batch_size or DEFAULT_BATCH_SIZE), 1)
    epochs = max(int(epochs or 1), 1)
    input_format = normalize_text(input_format).casefold() or "auto"
    if input_format not in VALID_INPUT_FORMATS:
        raise ValueError(f"unsupported input_format: {input_format}")
    resolved_format = detect_input_format(dataset_path) if input_format == "auto" else input_format
    engine = SemanticEngine(config=EngineConfig(state_dir=state_path))
    report = engine._empty_train_report(session_id=session_id, epochs=epochs)
    run = {
        "session_id": session_id,
        "kind": "mass_train",
        "dataset_path": str(dataset_path),
        "input_format": resolved_format,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_records": max_records,
        "started_at": time.time(),
        "records": 0,
        "sequences": 0,
        "flushes": 0,
    }
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "start",
                "kind": "mass_train",
                "session_id": session_id,
                "dataset_path": str(dataset_path),
                "state_dir": str(state_path),
                "input_format": resolved_format,
                "batch_size": batch_size,
                "epochs": epochs,
                "max_records": max_records,
            }
        )
    pending = 0
    total_processed = 0
    max_records = max(int(max_records or 0), 0) if max_records is not None else 0
    for epoch_idx in range(epochs):
        if max_records > 0 and total_processed >= max_records:
            break
        if resolved_format == "jsonl":
            batches = iter_batches(iter_jsonl_records(dataset_path), batch_size)
            for batch in batches:
                if max_records > 0 and total_processed >= max_records:
                    break
                for record in batch:
                    if max_records > 0 and total_processed >= max_records:
                        break
                    if not record:
                        continue
                    if "hierarchy" in record:
                        text = normalize_text(str(record.get("text") or record.get("content") or record.get("body") or ""))
                        hierarchy = engine._normalize_hierarchy(record.get("hierarchy"))
                        if not text and not hierarchy:
                            continue
                        report["dataset_records"] = int(report.get("dataset_records", 0)) + 1
                        run["records"] = int(run.get("records", 0)) + 1
                        total_processed += 1
                        engine._train_hierarchy_record(record, report, run, session_id=session_id)
                        pending += 1
                        continue
                    text = normalize_text(engine._jsonl_record_to_text(record))
                    if not text:
                        continue
                    report["dataset_records"] = int(report.get("dataset_records", 0)) + 1
                    run["records"] = int(run.get("records", 0)) + 1
                    total_processed += 1
                    engine._apply_training_fragments(engine._training_fragments(text), report, run, session_id=session_id)
                    pending += 1
                if pending >= batch_size:
                    engine._trim_collections()
                    save_checkpoint(engine.state_path, engine.checkpoint)
                    run["flushes"] = int(run.get("flushes", 0)) + 1
                    pending = 0
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "phase": "progress",
                                "kind": "mass_train",
                                "session_id": session_id,
                                "dataset_path": str(dataset_path),
                                "state_dir": str(state_path),
                                "input_format": resolved_format,
                                "batch_size": batch_size,
                                "max_records": max_records,
                                "records": report["dataset_records"],
                                "sequences": report["source_sequences"],
                                "flushes": run["flushes"],
                            }
                        )
        else:
            batches = iter_batches(iter_source_texts(dataset_path, engine, resolved_format), batch_size)
            for batch in batches:
                if max_records > 0 and total_processed >= max_records:
                    break
                for raw_text in batch:
                    if max_records > 0 and total_processed >= max_records:
                        break
                    text = normalize_text(raw_text)
                    if not text:
                        continue
                    report["dataset_records"] = int(report.get("dataset_records", 0)) + 1
                    run["records"] = int(run.get("records", 0)) + 1
                    total_processed += 1
                    engine._apply_training_fragments(engine._training_fragments(text), report, run, session_id=session_id)
                    pending += 1
                if pending >= batch_size:
                    engine._trim_collections()
                    save_checkpoint(engine.state_path, engine.checkpoint)
                    run["flushes"] = int(run.get("flushes", 0)) + 1
                    pending = 0
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "phase": "progress",
                                "kind": "mass_train",
                                "session_id": session_id,
                                "dataset_path": str(dataset_path),
                                "state_dir": str(state_path),
                                "input_format": resolved_format,
                                "batch_size": batch_size,
                                "max_records": max_records,
                                "records": report["dataset_records"],
                                "sequences": report["source_sequences"],
                                "flushes": run["flushes"],
                            }
                        )
    if pending > 0:
        engine._trim_collections()
        save_checkpoint(engine.state_path, engine.checkpoint)
        run["flushes"] = int(run.get("flushes", 0)) + 1
        pending = 0
    engine._finish_train_report(report, run)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "done",
                "kind": "mass_train",
                "session_id": session_id,
                "dataset_path": str(dataset_path),
                "state_dir": str(state_path),
                "input_format": resolved_format,
                "batch_size": batch_size,
                "epochs": epochs,
                "max_records": max_records,
                "dataset_records": report["dataset_records"],
                "source_sequences": report["source_sequences"],
                "source_pairs": report["source_pairs"],
                "tokens": report["tokens"],
                "edges": report["edges"],
                "flushes": run["flushes"],
            }
        )
    return {
        **report,
        "input_path": str(dataset_path),
        "state_dir": str(state_path),
        "input_format": resolved_format,
        "batch_size": batch_size,
        "flushes": run["flushes"],
        "max_records": max_records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantic_ants.mass_train")
    parser.add_argument("--input", "--dataset", dest="input_path", required=True)
    parser.add_argument("--state-dir", default=".semantic_ants")
    parser.add_argument("--batch-size", "--save-every", dest="batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--format", dest="input_format", default="auto", choices=sorted(VALID_INPUT_FORMATS))
    parser.add_argument("--session-id", default="mass_train")
    parser.add_argument("--max-records", "--max_records", dest="max_records", type=int, default=5000, help="Stop after processing this many records (0 = unlimited)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = train_mass_dataset(
        args.input_path,
        state_dir=args.state_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        input_format=args.input_format,
        session_id=args.session_id,
        max_records=args.max_records,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
