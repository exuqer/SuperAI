from __future__ import annotations

import bz2
import csv
import io
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterable, TextIO
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from semantic_ants.core.normalization import tokenize, text_to_concept_uri

TATOEBA_EXPORT_BASE_URL = "https://downloads.tatoeba.org/exports/per_language"
TATOEBA_USER_AGENT = "semantic_ants/1.0 (tatoeba dataset)"
TATOEBA_LANGUAGE_CODES = {
    "en": "eng",
    "ru": "rus",
}


def download_tatoeba_translation_dataset(
    output: str | Path,
    source_lang: str = "ru",
    target_lang: str = "en",
    limit: int | None = None,
    bidirectional: bool = True,
    timeout: float = 60.0,
    use_cc0: bool = True,
) -> int:
    source_code = _tatoeba_code(source_lang)
    target_code = _tatoeba_code(target_lang)
    if source_code == target_code:
        raise ValueError("source_lang and target_lang must differ")

    sentence_urls = _sentence_urls(source_code, use_cc0=use_cc0)
    target_sentence_urls = _sentence_urls(target_code, use_cc0=use_cc0)
    link_urls = _link_urls(source_code, target_code)
    last_error: Exception | None = None

    for source_url in sentence_urls:
        for target_url in target_sentence_urls:
            for links_url in link_urls:
                try:
                    with ExitStack() as stack:
                        source_handle = stack.enter_context(_open_text_resource(source_url, timeout=timeout))
                        target_handle = stack.enter_context(_open_text_resource(target_url, timeout=timeout))
                        links_handle = stack.enter_context(_open_text_resource(links_url, timeout=timeout))
                        return write_jsonl(
                            convert_tatoeba_translation_pairs(
                                source_handle,
                                target_handle,
                                links_handle,
                                source_lang=source_lang,
                                target_lang=target_lang,
                                source_sentence_url=source_url,
                                target_sentence_url=target_url,
                                links_url=links_url,
                                bidirectional=bidirectional,
                                limit=limit,
                            ),
                            output,
                        )
                except HTTPError as exc:
                    if exc.code == 404:
                        last_error = exc
                        continue
                    raise
                except FileNotFoundError as exc:
                    last_error = exc
                    continue
    if last_error is not None:
        raise FileNotFoundError(str(last_error)) from last_error
    raise FileNotFoundError("Не удалось загрузить Tatoeba translation dataset")


def convert_tatoeba_translation_pairs(
    source_sentences: TextIO,
    target_sentences: TextIO,
    links: TextIO,
    *,
    source_lang: str,
    target_lang: str,
    source_sentence_url: str = "",
    target_sentence_url: str = "",
    links_url: str = "",
    bidirectional: bool = True,
    limit: int | None = None,
) -> Iterable[dict[str, object]]:
    source_rows = _read_sentence_rows(source_sentences)
    target_rows = _read_sentence_rows(target_sentences)
    sentences = {**source_rows, **target_rows}
    emitted = 0
    seen: set[tuple[int, int]] = set()

    for left_id, right_id in _read_link_rows(links):
        if limit is not None and emitted >= limit:
            return
        left = sentences.get(left_id)
        right = sentences.get(right_id)
        if not left or not right:
            continue
        pair = _normalize_pair(left, right, source_lang=source_lang, target_lang=target_lang)
        if pair is None:
            pair = _normalize_pair(right, left, source_lang=source_lang, target_lang=target_lang)
        if pair is None:
            continue
        canonical = tuple(sorted((pair["source_sentence_id"], pair["target_sentence_id"])))
        if canonical in seen:
            continue
        seen.add(canonical)
        yield pair["forward"]
        emitted += 1
        if limit is not None and emitted >= limit:
            return
        if bidirectional:
            yield pair["reverse"]
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
def _open_text_resource(source: str, timeout: float = 60.0):
    resolved = str(source)
    local = Path(resolved)
    with ExitStack() as stack:
        if local.exists():
            handle = _open_local_text(stack, local)
            yield handle
            return
        request = Request(
            resolved,
            headers={
                "User-Agent": TATOEBA_USER_AGENT,
                "Accept": "application/octet-stream,text/plain,*/*;q=0.8",
            },
        )
        response = stack.enter_context(urlopen(request, timeout=timeout))
        handle = _open_remote_text(stack, response, resolved)
        yield handle


def _open_local_text(stack: ExitStack, path: Path) -> TextIO:
    if path.suffix == ".bz2":
        raw = stack.enter_context(bz2.BZ2File(path, "rb"))
        return stack.enter_context(io.TextIOWrapper(raw, encoding="utf-8"))
    return stack.enter_context(path.open("r", encoding="utf-8"))


def _open_remote_text(stack: ExitStack, response: object, source: str) -> TextIO:
    if source.endswith(".bz2"):
        raw = stack.enter_context(bz2.BZ2File(response))  # type: ignore[arg-type]
        return stack.enter_context(io.TextIOWrapper(raw, encoding="utf-8"))
    return stack.enter_context(io.TextIOWrapper(response, encoding="utf-8"))  # type: ignore[arg-type]


def _sentence_urls(lang_code: str, use_cc0: bool = True) -> list[str]:
    stem = f"{lang_code}_sentences_CC0.tsv.bz2" if use_cc0 else f"{lang_code}_sentences.tsv.bz2"
    fallback = f"{lang_code}_sentences.tsv.bz2" if use_cc0 else f"{lang_code}_sentences_CC0.tsv.bz2"
    return [
        f"{TATOEBA_EXPORT_BASE_URL}/{lang_code}/{stem}",
        f"{TATOEBA_EXPORT_BASE_URL}/{lang_code}/{fallback}",
    ]


def _link_urls(source_code: str, target_code: str) -> list[str]:
    return [
        f"{TATOEBA_EXPORT_BASE_URL}/{source_code}/{source_code}-{target_code}_links.tsv.bz2",
        f"{TATOEBA_EXPORT_BASE_URL}/{target_code}/{target_code}-{source_code}_links.tsv.bz2",
    ]


def _tatoeba_code(lang: str) -> str:
    code = TATOEBA_LANGUAGE_CODES.get(lang)
    if not code:
        raise ValueError(f"Неподдерживаемый язык Tatoeba: {lang}")
    return code


def _read_sentence_rows(handle: TextIO) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    reader = csv.reader(handle, delimiter="\t")
    for row in reader:
        if len(row) < 3:
            continue
        try:
            sentence_id = int(row[0])
        except ValueError:
            continue
        lang = row[1].strip()
        text = "\t".join(row[2:]).strip()
        if not text:
            continue
        rows[sentence_id] = {"id": str(sentence_id), "lang": lang, "text": text}
    return rows


def _read_link_rows(handle: TextIO) -> Iterable[tuple[int, int]]:
    reader = csv.reader(handle, delimiter="\t")
    for row in reader:
        if len(row) < 2:
            continue
        try:
            left_id = int(row[0])
            right_id = int(row[1])
        except ValueError:
            continue
        yield left_id, right_id


def _normalize_pair(
    source: dict[str, str],
    target: dict[str, str],
    *,
    source_lang: str,
    target_lang: str,
) -> dict[str, object] | None:
    if source.get("lang") != _tatoeba_code(source_lang) or target.get("lang") != _tatoeba_code(target_lang):
        return None
    source_text = " ".join(source["text"].split())
    target_text = " ".join(target["text"].split())
    if not source_text or not target_text:
        return None
    source_tokens = [token for token in tokenize(source_text) if token]
    target_tokens = [token for token in tokenize(target_text) if token]
    source_concepts = [text_to_concept_uri(token, source_lang) for token in source_tokens]
    target_concepts = [text_to_concept_uri(token, target_lang) for token in target_tokens]
    forward = {
        "stimulus": source_text,
        "accepted_answer": target_text,
        "history": [],
        "lang": source_lang,
        "answer_lang": target_lang,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "target_concepts": list(dict.fromkeys(source_concepts)),
        "positive_edges": _translation_edges(source_concepts, target_concepts),
        "reward": 1.2,
        "metadata": {
            "dataset": "Tatoeba",
            "source": "tatoeba.org",
            "license": "CC BY 2.0 FR",
            "source_sentence_id": int(source.get("id", 0) or 0),
            "target_sentence_id": int(target.get("id", 0) or 0),
            "source_sentence_lang": source.get("lang", source_lang),
            "target_sentence_lang": target.get("lang", target_lang),
            "source_text": source_text,
            "target_text": target_text,
        },
    }
    reverse = {
        "stimulus": target_text,
        "accepted_answer": source_text,
        "history": [],
        "lang": target_lang,
        "answer_lang": source_lang,
        "source_lang": target_lang,
        "target_lang": source_lang,
        "target_concepts": list(dict.fromkeys(target_concepts)),
        "positive_edges": _translation_edges(target_concepts, source_concepts),
        "reward": 1.2,
        "metadata": {
            "dataset": "Tatoeba",
            "source": "tatoeba.org",
            "license": "CC BY 2.0 FR",
            "source_sentence_id": int(target.get("id", 0) or 0),
            "target_sentence_id": int(source.get("id", 0) or 0),
            "source_sentence_lang": target.get("lang", target_lang),
            "target_sentence_lang": source.get("lang", source_lang),
            "source_text": target_text,
            "target_text": source_text,
        },
    }
    return {
        "source_sentence_id": int(source.get("id", 0) or 0),
        "target_sentence_id": int(target.get("id", 0) or 0),
        "forward": forward,
        "reverse": reverse,
    }


def _translation_edges(source_concepts: list[str], target_concepts: list[str]) -> list[list[str]]:
    edges: list[list[str]] = []
    for start, end in zip(source_concepts, target_concepts):
        if start and end:
            edges.append([start, "TranslationEquivalent", end])
    return edges
