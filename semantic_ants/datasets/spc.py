from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Iterable, TextIO
from urllib.request import urlopen

SPC_BASE_URL = "https://raw.githubusercontent.com/google-research-datasets/Synthetic-Persona-Chat/main/data"
SPC_URLS = {
    "train": f"{SPC_BASE_URL}/Synthetic-Persona-Chat_train.csv",
    "dev": f"{SPC_BASE_URL}/Synthetic-Persona-Chat_valid.csv",
    "valid": f"{SPC_BASE_URL}/Synthetic-Persona-Chat_valid.csv",
    "test": f"{SPC_BASE_URL}/Synthetic-Persona-Chat_test.csv",
    "synth": f"{SPC_BASE_URL}/New-Persona-New-Conversations.csv",
}

SPEAKER_RE = re.compile(r"^(?:user|person)\s*\d+\s*:\s*(?P<text>.+)$", re.IGNORECASE)


def download_spc_dataset(
    split: str,
    output: str | Path,
    limit: int | None = None,
    timeout: float = 60.0,
) -> int:
    url = SPC_URLS[split]
    with urlopen(url, timeout=timeout) as response:
        handle = io.TextIOWrapper(response, encoding="utf-8", newline="")
        return write_jsonl(convert_spc_csv(handle, split=split, source_url=url, limit=limit), output)


def convert_spc_csv(
    handle: TextIO,
    split: str,
    source_url: str = "",
    limit: int | None = None,
) -> Iterable[dict[str, object]]:
    emitted = 0
    for row_index, row in enumerate(csv.DictReader(handle), start=1):
        personas_1 = _lines(row.get("user 1 personas", ""))
        personas_2 = _lines(row.get("user 2 personas", ""))
        utterances = _utterances(row.get("Best Generated Conversation", ""))
        if len(utterances) < 2:
            continue
        conversation_id = f"spc-{split}-{row_index}"
        for turn_index in range(1, len(utterances)):
            if limit is not None and emitted >= limit:
                return
            stimulus = utterances[turn_index - 1]["text"]
            accepted = utterances[turn_index]["text"]
            if not stimulus or not accepted:
                continue
            yield {
                "stimulus": stimulus,
                "accepted_answer": accepted,
                "history": utterances[: turn_index - 1],
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "lang": "en",
                "metadata": {
                    "dataset": "Synthetic-Persona-Chat",
                    "source": "google-research-datasets/Synthetic-Persona-Chat",
                    "source_url": source_url,
                    "split": split,
                    "license": "CC-BY 4.0",
                    "user_1_persona": personas_1,
                    "user_2_persona": personas_2,
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


def _lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _utterances(value: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SPEAKER_RE.match(line)
        text = match.group("text").strip() if match else line
        if not text:
            continue
        role = "assistant" if len(turns) % 2 else "user"
        turns.append({"role": role, "text": text})
    return turns
