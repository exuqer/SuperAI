from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROLE_USER_TOKEN = "[__user__]"
ROLE_ASSISTANT_TOKEN = "[__assistant__]"
TRAINING_LINE_SUFFIX = " ."
HOT_STARTERS = {"привет", "давай"}
HOT_STARTER_LIMIT = 50

WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:['-][0-9A-Za-zА-Яа-яЁё]+)*")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
URL_RE = re.compile(r"(?:https?://|t\.me/)", re.IGNORECASE)
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{3,}\b")
COMMAND_RE = re.compile(r"(?<!\S)/[A-Za-z_][\w-]*")
LATIN_RUN_RE = re.compile(r"[A-Za-z]{3,}")
REPEAT_CHAR_RE = re.compile(r"(.)\1{3,}", re.IGNORECASE | re.DOTALL)
REPEAT_SYLLABLE_RE = re.compile(r"([0-9A-Za-zА-Яа-яЁё]{2,4})\1{2,}", re.IGNORECASE)

CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".kts",
    ".mjs",
    ".py",
    ".pyi",
    ".rb",
    ".rs",
    ".swift",
    ".ts",
    ".tsx",
}
DOC_EXTENSIONS = {
    ".adoc",
    ".md",
    ".markdown",
    ".org",
    ".rst",
    ".text",
    ".txt",
}
SKIPPED_HIERARCHY_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".semantic_ants",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
MAX_HIERARCHY_FILE_BYTES = 1_000_000

PAIR_KEYS: tuple[tuple[str, str], ...] = (
    ("question", "answer"),
    ("prompt", "response"),
    ("input", "output"),
    ("instruction", "response"),
    ("source", "target"),
    ("query", "answer"),
)


@dataclass
class PreprocessStats:
    input_path: str
    output_path: str
    max_pairs: int | None
    scanned: int = 0
    accepted: int = 0
    skipped_invalid: int = 0
    skipped_relevance: int = 0
    skipped_missing_pair: int = 0
    skipped_length: int = 0
    skipped_spam: int = 0
    skipped_hot: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "max_pairs": self.max_pairs,
            "scanned": self.scanned,
            "accepted": self.accepted,
            "skipped_invalid": self.skipped_invalid,
            "skipped_relevance": self.skipped_relevance,
            "skipped_missing_pair": self.skipped_missing_pair,
            "skipped_length": self.skipped_length,
            "skipped_spam": self.skipped_spam,
            "skipped_hot": self.skipped_hot,
        }


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r", " ").split()).strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(normalize_text(text)))


def first_word(text: str) -> str:
    matches = WORD_RE.findall(normalize_text(text))
    return matches[0].casefold() if matches else ""


def is_relevant_record(record: dict[str, Any]) -> bool:
    relevance = record.get("relevance")
    if isinstance(relevance, bool):
        return relevance is True
    if relevance is None:
        return False
    try:
        return int(relevance) == 1
    except (TypeError, ValueError):
        return str(relevance).strip() == "1"


def extract_pair(record: dict[str, Any]) -> tuple[str, str] | None:
    for left_key, right_key in PAIR_KEYS:
        left = normalize_text(record.get(left_key, ""))
        right = normalize_text(record.get(right_key, ""))
        if left and right:
            return left, right
    messages = record.get("messages")
    if isinstance(messages, list):
        user_text = ""
        assistant_text = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = normalize_text(message.get("role", "")).casefold()
            content = normalize_text(message.get("content", message.get("text", "")))
            if not content:
                continue
            if not user_text and role in {"user", "human"}:
                user_text = content
            elif not assistant_text and role in {"assistant", "bot", "ai"}:
                assistant_text = content
            if user_text and assistant_text:
                return user_text, assistant_text
    return None


def has_spam(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned:
        return True
    if URL_RE.search(cleaned) or HANDLE_RE.search(cleaned) or COMMAND_RE.search(cleaned):
        return True
    if not re.search(r"[0-9A-Za-zА-Яа-яЁё]", cleaned):
        return True
    if LATIN_RUN_RE.search(cleaned):
        return True
    compacted = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "", cleaned.casefold())
    if compacted and REPEAT_CHAR_RE.search(compacted):
        return True
    if compacted and REPEAT_SYLLABLE_RE.search(compacted):
        return True
    return False


def is_valid_pair(question: str, answer: str) -> bool:
    return 3 <= word_count(question) <= 15 and 3 <= word_count(answer) <= 15


def format_training_line(question: str, answer: str) -> str:
    return f"{ROLE_USER_TOKEN} {normalize_text(question)} {ROLE_ASSISTANT_TOKEN} {normalize_text(answer)} ."


def _normalize_hierarchy(parts: Any) -> list[str]:
    if isinstance(parts, str):
        items = [parts]
    elif isinstance(parts, (list, tuple)):
        items = list(parts)
    else:
        return []
    normalized = [normalize_text(str(item)) for item in items if normalize_text(str(item))]
    return normalized


def _json_record(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False)


def iter_valid_dialogue_pairs(
    input_path: str | Path,
    *,
    max_pairs: int | None = None,
) -> Iterator[tuple[str, str]]:
    source_path = Path(input_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.is_dir():
        raise IsADirectoryError(source_path)
    limit = None
    if max_pairs is not None:
        try:
            candidate = int(max_pairs)
        except (TypeError, ValueError):
            candidate = 0
        if candidate > 0:
            limit = candidate
    hot_counts: dict[str, int] = {starter: 0 for starter in HOT_STARTERS}
    accepted = 0
    with source_path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if not is_relevant_record(record):
                continue
            pair = extract_pair(record)
            if pair is None:
                continue
            question, answer = pair
            if not is_valid_pair(question, answer):
                continue
            if has_spam(question) or has_spam(answer):
                continue
            starter = first_word(question)
            if starter in HOT_STARTERS:
                if hot_counts[starter] >= HOT_STARTER_LIMIT:
                    continue
                hot_counts[starter] += 1
            yield question, answer
            accepted += 1
            if limit is not None and accepted >= limit:
                break


def _file_contains_headings(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if HEADING_RE.match(line):
                    return True
    except OSError:
        return False
    return False


def _looks_like_hierarchy_source(path: Path) -> bool:
    if path.is_dir():
        return True
    suffix = path.suffix.lower()
    if suffix in CODE_EXTENSIONS:
        return True
    if suffix in {".adoc", ".md", ".markdown", ".org", ".rst"}:
        return True
    if suffix in {".txt", ".text"}:
        return _file_contains_headings(path)
    return False


def _relative_hierarchy_parts(path: Path, root: Path | None = None) -> list[str]:
    candidate = path.with_suffix("")
    if root is not None:
        try:
            candidate = path.relative_to(root).with_suffix("")
        except ValueError:
            candidate = path.with_suffix("")
    elif candidate.is_absolute():
        return [normalize_text(path.stem)] if normalize_text(path.stem) else []
    if candidate.is_absolute():
        parts = [part for part in candidate.parts if part not in {candidate.anchor, candidate.drive, candidate.root}]
    else:
        parts = list(candidate.parts)
    return [normalize_text(part) for part in parts if normalize_text(part)]


def _should_skip_hierarchy_path(path: Path, root: Path | None = None) -> bool:
    try:
        relative = path.relative_to(root) if root is not None else path
    except ValueError:
        relative = path
    if any(part in SKIPPED_HIERARCHY_DIRS for part in relative.parts):
        return True
    try:
        if path.is_file() and path.stat().st_size > MAX_HIERARCHY_FILE_BYTES:
            return True
    except OSError:
        return True
    return False


def _read_hierarchy_file(path: Path) -> str:
    if _should_skip_hierarchy_path(path):
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _iter_hierarchy_sections(path: Path, root: Path | None = None) -> Iterator[tuple[list[str], str]]:
    if _should_skip_hierarchy_path(path, root):
        return
    if path.is_dir():
        yield from _iter_hierarchy_sections_from_directory(path, root=path)
        return
    if path.suffix.lower() in CODE_EXTENSIONS:
        text = normalize_text(_read_hierarchy_file(path))
        hierarchy = _relative_hierarchy_parts(path, root=root)
        if not hierarchy:
            hierarchy = [normalize_text(path.stem)]
        if text:
            yield hierarchy, text
        return
    yield from _iter_hierarchy_sections_from_document(path, root=root)


def _iter_hierarchy_sections_from_directory(directory: Path, *, root: Path | None = None) -> Iterator[tuple[list[str], str]]:
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if _should_skip_hierarchy_path(path, root):
            continue
        suffix = path.suffix.lower()
        if suffix not in CODE_EXTENSIONS and suffix not in DOC_EXTENSIONS:
            continue
        yield from _iter_hierarchy_sections(path, root=root)


def _iter_hierarchy_sections_from_document(path: Path, root: Path | None = None) -> Iterator[tuple[list[str], str]]:
    content = _read_hierarchy_file(path)
    if not content:
        return
    lines = content.splitlines()
    base = _relative_hierarchy_parts(path, root=root)
    if not base:
        base = [normalize_text(path.stem)]
    heading_stack: list[str] = []
    buffer: list[str] = []

    def flush() -> Iterator[tuple[list[str], str]]:
        text = normalize_text("\n".join(buffer))
        if text:
            yield base + heading_stack, text

    for raw_line in lines:
        line = raw_line.rstrip()
        match = HEADING_RE.match(line.strip())
        if match:
            yield from flush()
            buffer.clear()
            level = len(match.group(1))
            title = normalize_text(match.group(2))
            if level <= 0:
                continue
            heading_stack[:] = heading_stack[: max(level - 1, 0)]
            heading_stack.append(title)
            continue
        buffer.append(line)
    yield from flush()


def preprocess_hierarchy_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_pairs: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    source_path = Path(input_path)
    destination_path = Path(output_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.is_file() and source_path.resolve() == destination_path.resolve():
        raise ValueError("output_path must differ from input_path")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    limit = None
    if max_pairs is not None:
        try:
            candidate = int(max_pairs)
        except (TypeError, ValueError):
            candidate = 0
        if candidate > 0:
            limit = candidate
    stats = PreprocessStats(str(source_path), str(destination_path), limit)

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "start",
                "kind": "preprocess",
                "mode": "hierarchy",
                "input_path": str(source_path),
                "output_path": str(destination_path),
                "max_pairs": limit,
            }
        )

    with destination_path.open("w", encoding="utf-8", newline="\n") as dst:
        for hierarchy, text in _iter_hierarchy_sections(source_path):
            stats.scanned += 1
            normalized_text = normalize_text(text)
            if not normalized_text:
                stats.skipped_invalid += 1
                continue
            record = {"hierarchy": _normalize_hierarchy(hierarchy), "text": normalized_text}
            if not record["hierarchy"]:
                record["hierarchy"] = [source_path.stem]
            dst.write(_json_record(record))
            dst.write("\n")
            stats.accepted += 1
            if progress_callback is not None and stats.scanned % 1000 == 0:
                progress_callback(
                    {
                        "phase": "progress",
                        "kind": "preprocess",
                        "mode": "hierarchy",
                        "input_path": str(source_path),
                        "output_path": str(destination_path),
                        "scanned": stats.scanned,
                        "accepted": stats.accepted,
                    }
                )
            if limit is not None and stats.accepted >= limit:
                break

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "done",
                "kind": "preprocess",
                "mode": "hierarchy",
                "input_path": str(source_path),
                "output_path": str(destination_path),
                **stats.to_dict(),
            }
        )
    return stats.to_dict()


def _preprocess_dialogue_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_pairs: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    source_path = Path(input_path)
    destination_path = Path(output_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.is_dir():
        raise IsADirectoryError(source_path)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("output_path must differ from input_path")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    limit = None
    if max_pairs is not None:
        try:
            candidate = int(max_pairs)
        except (TypeError, ValueError):
            candidate = 0
        if candidate > 0:
            limit = candidate
    stats = PreprocessStats(str(source_path), str(destination_path), limit)
    hot_counts: dict[str, int] = {starter: 0 for starter in HOT_STARTERS}

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "start",
                "kind": "preprocess",
                "mode": "dialogue",
                "input_path": str(source_path),
                "output_path": str(destination_path),
                "max_pairs": limit,
            }
        )

    with source_path.open("r", encoding="utf-8") as src, destination_path.open("w", encoding="utf-8", newline="\n") as dst:
        for line_number, raw_line in enumerate(src, start=1):
            line = raw_line.strip()
            if not line:
                continue
            stats.scanned += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats.skipped_invalid += 1
                continue
            if not isinstance(record, dict):
                stats.skipped_invalid += 1
                continue
            if not is_relevant_record(record):
                stats.skipped_relevance += 1
                continue
            pair = extract_pair(record)
            if pair is None:
                stats.skipped_missing_pair += 1
                continue
            question, answer = pair
            if not is_valid_pair(question, answer):
                stats.skipped_length += 1
                continue
            if has_spam(question) or has_spam(answer):
                stats.skipped_spam += 1
                continue
            starter = first_word(question)
            if starter in HOT_STARTERS:
                if hot_counts[starter] >= HOT_STARTER_LIMIT:
                    stats.skipped_hot += 1
                    continue
                hot_counts[starter] += 1
            dst.write(format_training_line(question, answer))
            dst.write("\n")
            stats.accepted += 1
            if progress_callback is not None and stats.scanned % 1000 == 0:
                progress_callback(
                    {
                        "phase": "progress",
                        "kind": "preprocess",
                        "mode": "dialogue",
                        "input_path": str(source_path),
                        "output_path": str(destination_path),
                        "line_number": line_number,
                        "scanned": stats.scanned,
                        "accepted": stats.accepted,
                        "skipped_relevance": stats.skipped_relevance,
                        "skipped_length": stats.skipped_length,
                        "skipped_spam": stats.skipped_spam,
                        "skipped_hot": stats.skipped_hot,
                    }
                )
            if limit is not None and stats.accepted >= limit:
                break

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "done",
                "kind": "preprocess",
                "mode": "dialogue",
                "input_path": str(source_path),
                "output_path": str(destination_path),
                **stats.to_dict(),
            }
        )
    return stats.to_dict()


def preprocess_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_pairs: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    source_path = Path(input_path)
    if _looks_like_hierarchy_source(source_path):
        return preprocess_hierarchy_dataset(source_path, output_path, max_pairs=max_pairs, progress_callback=progress_callback)
    return _preprocess_dialogue_dataset(source_path, output_path, max_pairs=max_pairs, progress_callback=progress_callback)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantic_ants.preprocess")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max-pairs", "--max_pairs", dest="max_pairs", type=int, default=None)
    parser.add_argument("--mode", choices=("auto", "dialogue", "hierarchy"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = Path(args.input_path)
    if args.mode == "hierarchy":
        stats = preprocess_hierarchy_dataset(input_path, args.output_path, max_pairs=args.max_pairs)
    elif args.mode == "dialogue":
        stats = _preprocess_dialogue_dataset(input_path, args.output_path, max_pairs=args.max_pairs)
    else:
        stats = preprocess_dataset(input_path, args.output_path, max_pairs=args.max_pairs)
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
