from __future__ import annotations

import io
import json
import re
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterable, TextIO
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from semantic_ants.core.normalization import detect_language, tokenize, text_to_concept_uri

KOZIEV_REPO_URL = "https://github.com/Koziev/NLP_Datasets"
KOZIEV_RAW_BASE_URL = "https://raw.githubusercontent.com/Koziev/NLP_Datasets/master"
KOZIEV_CHITCHAT_FILES = (
    "Conversations/Data/chan_dialogues.txt",
    "Conversations/Data/extract_dialogues_from_anekdots.txt",
)
KOZIEV_USER_AGENT = "semantic_ants/1.0 (koziev dataset)"

_TURN_PREFIX_RE = re.compile(r"^[\-\*\u2013\u2014\u2022]+\s*")
_SPEAKER_PREFIX_RE = re.compile(
    r"^(?:user|assistant|speaker|person|bot|agent)\s*\d*\s*[:\-]\s*",
    re.IGNORECASE,
)


def download_koziev_dialogues_dataset(
    output: str | Path,
    source: str | Path | None = None,
    limit: int | None = None,
    timeout: float = 60.0,
) -> int:
    candidates = _koziev_candidates(source)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            with _open_text_source(candidate, timeout=timeout) as (handle, resolved_source):
                return write_jsonl(
                    convert_koziev_dialogues(
                        handle,
                        source_url=_resolved_source_url(resolved_source),
                        source_path=_resolved_source_path(resolved_source),
                        limit=limit,
                    ),
                    output,
                )
        except FileNotFoundError as exc:
            last_error = exc
        except HTTPError as exc:
            if exc.code == 404:
                last_error = exc
                continue
            raise
    if last_error is not None:
        raise FileNotFoundError(str(last_error)) from last_error
    raise FileNotFoundError("Не удалось найти исходный Koziev dialogue dataset")


def convert_koziev_dialogues(
    handle: TextIO,
    source_url: str = "",
    source_path: str = "",
    limit: int | None = None,
) -> Iterable[dict[str, object]]:
    emitted = 0
    raw_text = handle.read()
    for chunk_index, chunk in enumerate(re.split(r"\n\s*\n", raw_text), start=1):
        turns = _parse_dialogue_chunk(chunk)
        if len(turns) < 2:
            continue
        conversation_id = _conversation_id(source_path, chunk_index)
        lang = detect_language(" ".join(turn["text"] for turn in turns[:2]))
        for turn_index in range(1, len(turns)):
            if limit is not None and emitted >= limit:
                return
            stimulus = turns[turn_index - 1]["text"]
            accepted = turns[turn_index]["text"]
            if not stimulus or not accepted:
                continue
            source_tokens = [token for token in tokenize(stimulus) if token]
            target_tokens = [token for token in tokenize(accepted) if token]
            source_concepts = [text_to_concept_uri(token, lang=lang) for token in source_tokens]
            target_concepts = [text_to_concept_uri(token, lang=lang) for token in target_tokens]
            yield {
                "stimulus": stimulus,
                "accepted_answer": accepted,
                "history": [dict(turn) for turn in turns[: turn_index - 1]],
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "lang": lang,
                "answer_lang": lang,
                "source_lang": lang,
                "target_concepts": list(dict.fromkeys(source_concepts)),
                "positive_edges": _dialogue_edges(source_concepts, target_concepts),
                "reward": 1.1,
                "metadata": {
                    "dataset": "Koziev/NLP_Datasets",
                    "source": "Koziev/NLP_Datasets",
                    "source_url": source_url,
                    "source_path": source_path,
                    "format": "dialogue",
                    "chunk_index": chunk_index,
                },
            }
            emitted += 1


def write_jsonl(examples: Iterable[dict[str, object]], output: str | Path) -> int:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


@contextmanager
def _open_text_source(source: str | Path, timeout: float = 60.0):
    resolved = str(source)
    local = Path(resolved)
    with ExitStack() as stack:
        if local.exists():
            handle = _open_local_text(stack, local)
            yield handle, local.as_posix()
            return
        if resolved.startswith("http://") or resolved.startswith("https://"):
            url = resolved
        else:
            url = f"{KOZIEV_RAW_BASE_URL}/{resolved.lstrip('/')}"
        request = Request(
            url,
            headers={
                "User-Agent": KOZIEV_USER_AGENT,
                "Accept": "text/plain,application/octet-stream;q=0.9,*/*;q=0.8",
            },
        )
        response = stack.enter_context(urlopen(request, timeout=timeout))
        handle = _open_response_text(stack, response, url)
        yield handle, url


def _open_local_text(stack: ExitStack, path: Path) -> TextIO:
    if path.suffix == ".bz2":
        import bz2

        raw = stack.enter_context(bz2.BZ2File(path, "rb"))
        return stack.enter_context(io.TextIOWrapper(raw, encoding="utf-8"))
    return stack.enter_context(path.open("r", encoding="utf-8"))


def _open_response_text(stack: ExitStack, response: object, url: str) -> TextIO:
    if url.endswith(".bz2"):
        import bz2

        raw = stack.enter_context(bz2.BZ2File(response))  # type: ignore[arg-type]
        return stack.enter_context(io.TextIOWrapper(raw, encoding="utf-8"))
    return stack.enter_context(io.TextIOWrapper(response, encoding="utf-8"))  # type: ignore[arg-type]


def _koziev_candidates(source: str | Path | None) -> list[str | Path]:
    if source is not None:
        return [source]
    return list(KOZIEV_CHITCHAT_FILES)


def _resolved_source_url(resolved: str) -> str:
    if resolved.startswith("http://") or resolved.startswith("https://"):
        return resolved
    if Path(resolved).exists():
        return Path(resolved).resolve().as_uri()
    return f"{KOZIEV_RAW_BASE_URL}/{resolved.lstrip('/')}"


def _resolved_source_path(resolved: str) -> str:
    if Path(resolved).exists():
        return Path(resolved).as_posix()
    return resolved


def _parse_dialogue_chunk(chunk: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for raw_line in chunk.splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line:
            continue
        line = _TURN_PREFIX_RE.sub("", line)
        line = _SPEAKER_PREFIX_RE.sub("", line)
        text = " ".join(line.split())
        if not text:
            continue
        role = "user" if len(turns) % 2 == 0 else "assistant"
        turns.append({"role": role, "text": text})
    return turns


def _conversation_id(source_path: str, chunk_index: int) -> str:
    stem = Path(source_path).stem if source_path else "koziev"
    return f"koziev-{stem}-{chunk_index}"


def _dialogue_edges(source_concepts: list[str], target_concepts: list[str]) -> list[list[str]]:
    edges: list[list[str]] = []
    for start, end in zip(source_concepts, target_concepts):
        if start and end:
            edges.append([start, "DialogueTurnEquivalent", end])
    return edges
