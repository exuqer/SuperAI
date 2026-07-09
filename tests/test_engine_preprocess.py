from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock
import unittest

from semantic_ants.engine import EngineConfig, SemanticEngine


class DummyNamedTemporaryFile:
    def __init__(self, name: Path) -> None:
        self.name = str(name)

    def close(self) -> None:
        return None


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


class EnginePreprocessTests(unittest.TestCase):
    def test_raw_jsonl_trains_without_temp_corpus_and_collapses_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state_dir = tmp_path / "state"
            input_path = tmp_path / "dataset.jsonl"
            write_jsonl(
                input_path,
                [
                    {
                        "question": "Привет как дела сегодня",
                        "answer": "Все хорошо спасибо тебе",
                        "relevance": 1,
                    },
                    {
                        "question": "Привет как дела сегодня",
                        "answer": "Все хорошо спасибо тебе",
                        "relevance": 1,
                    }
                ],
            )

            engine = SemanticEngine(config=EngineConfig(state_dir=state_dir))
            with mock.patch("semantic_ants.engine.tempfile.NamedTemporaryFile") as temp_mock:
                report = engine.train_jsonl(input_path, session_id="unit", epochs=1)

            temp_mock.assert_not_called()
            self.assertEqual(report["dataset_records"], 2)
            self.assertEqual(report["unique_pairs"], 1)
            self.assertEqual(report["duplicates_collapsed"], 1)
            self.assertEqual(report["checkpoint_format"], "sqlite")
            self.assertIn("__user__", engine.checkpoint.tokens)
            self.assertIn("__assistant__", engine.checkpoint.tokens)

    def test_raw_jsonl_respects_max_records_without_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state_dir = tmp_path / "state"
            input_path = tmp_path / "dataset.jsonl"
            write_jsonl(
                input_path,
                [
                    {
                        "question": "Привет как дела сегодня",
                        "answer": "Все хорошо спасибо тебе",
                        "relevance": 1,
                    },
                    {
                        "question": "Как настроение сегодня у тебя",
                        "answer": "Все отлично благодарю тебя",
                        "relevance": 1,
                    },
                ],
            )

            engine = SemanticEngine(config=EngineConfig(state_dir=state_dir))
            with mock.patch("semantic_ants.engine.tempfile.NamedTemporaryFile") as temp_mock:
                report = engine.train_jsonl(input_path, session_id="unit", epochs=1, max_records=1)

            temp_mock.assert_not_called()
            self.assertEqual(report["dataset_records"], 1)
            self.assertEqual(report["unique_pairs"], 1)
            self.assertEqual(report["duplicates_collapsed"], 0)

    def test_raw_jsonl_honors_max_pairs_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state_dir = tmp_path / "state"
            input_path = tmp_path / "dataset.jsonl"
            write_jsonl(
                input_path,
                [
                    {
                        "question": "Привет как дела сегодня",
                        "answer": "Все хорошо спасибо тебе",
                        "relevance": 1,
                    },
                    {
                        "question": "Как настроение сегодня у тебя",
                        "answer": "Все отлично благодарю тебя",
                        "relevance": 1,
                    },
                ],
            )

            engine = SemanticEngine(config=EngineConfig(state_dir=state_dir))
            report = engine.train_jsonl(input_path, session_id="unit", epochs=1, max_pairs=1)

            self.assertEqual(report["dataset_records"], 1)
            self.assertEqual(report["unique_pairs"], 1)

    def test_tagged_corpus_routes_roles_into_one_dialogue_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state_dir = tmp_path / "state"
            corpus_path = tmp_path / "train_ready.txt"
            corpus_path.write_text(
                "[__user__] Привет как дела сегодня [__assistant__] Все хорошо спасибо тебе .\n",
                encoding="utf-8",
            )

            engine = SemanticEngine(config=EngineConfig(state_dir=state_dir))
            calls: list[tuple[str, str]] = []

            def fake_train_dialogue_pair(prompt: str, response: str, report: dict[str, object], run: dict[str, object]) -> None:
                calls.append((prompt, response))
                report["source_sequences"] = int(report.get("source_sequences", 0)) + 1
                report["source_pairs"] = int(report.get("source_pairs", 0)) + 1

            with mock.patch.object(SemanticEngine, "_train_dialogue_pair", side_effect=fake_train_dialogue_pair):
                engine.train_jsonl(corpus_path, session_id="unit", epochs=1)

            self.assertEqual(calls, [("Привет как дела сегодня", "Все хорошо спасибо тебе .")])

    def test_hyphen_dialogue_pair_with_punctuated_prompt_trains_assistant_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))

            report = engine.train_text(
                "Привет! Расскажи анекдот - Подскажите, а тут мимо вас не пробегало стадо баранов? - А ты отстал что ли?",
                session_id="unit",
                epochs=10,
            )

            self.assertEqual(report["source_pairs"], 10)
            self.assertEqual(report["source_sequences"], 10)
            root_edges = engine.checkpoint.meta["hypernodes"]["hyper:__root__"]["subgraph"]["edges"]
            self.assertEqual(root_edges["token:__assistant__|next|token:подскажите"]["weight"], 50.0)
            self.assertIn("token:подскажите|next|token:,", root_edges)
            response = engine.chat("Анекдот расскажи", session_id="unit")["result"]["response"]
            self.assertIn("подскажите", response)
            self.assertIn("отстал", response)


if __name__ == "__main__":
    unittest.main()
