from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from semantic_ants.preprocess import preprocess_dataset


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


class PreprocessTests(unittest.TestCase):
    def test_accepts_valid_pair_and_formats_role_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "dataset.jsonl"
            output_path = tmp_path / "train_ready.txt"
            write_jsonl(
                input_path,
                [
                    {
                        "question": "Привет как дела сегодня",
                        "answer": "Все хорошо спасибо тебе",
                        "relevance": 1,
                    }
                ],
            )

            stats = preprocess_dataset(input_path, output_path, max_pairs=5)

            self.assertEqual(stats["accepted"], 1)
            self.assertTrue(output_path.exists())
            self.assertEqual(
                output_path.read_text(encoding="utf-8").strip(),
                "[__user__] Привет как дела сегодня [__assistant__] Все хорошо спасибо тебе .",
            )

    def test_accepts_all_pairs_without_explicit_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "dataset.jsonl"
            output_path = tmp_path / "train_ready.txt"
            records = [
                {
                    "question": f"Привет как дела сегодня номер {index}",
                    "answer": f"Все хорошо спасибо тебе номер {index}",
                    "relevance": 1,
                }
                for index in range(6)
            ]
            write_jsonl(input_path, records)

            stats = preprocess_dataset(input_path, output_path)

            self.assertEqual(stats["accepted"], 6)
            self.assertIsNone(stats["max_pairs"])
            self.assertEqual(len(output_path.read_text(encoding="utf-8").splitlines()), 6)

    def test_filters_relevance_length_and_spam(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "dataset.jsonl"
            output_path = tmp_path / "train_ready.txt"
            write_jsonl(
                input_path,
                [
                    {"question": "Привет как дела", "answer": "Все хорошо спасибо", "relevance": 0},
                    {"question": "Привет", "answer": "Все хорошо спасибо", "relevance": 1},
                    {
                        "question": "Привет как дела сегодня",
                        "answer": "один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать",
                        "relevance": 1,
                    },
                    {"question": "Смотри http://example.com", "answer": "Все хорошо спасибо", "relevance": 1},
                    {"question": "Hello как дела", "answer": "Все хорошо спасибо", "relevance": 1},
                    {"question": "ахахахахаха как дела", "answer": "Все хорошо спасибо", "relevance": 1},
                    {"question": "/start как дела", "answer": "Все хорошо спасибо", "relevance": 1},
                    {"question": "😂😂😂", "answer": "🙂🙂🙂", "relevance": 1},
                ],
            )

            stats = preprocess_dataset(input_path, output_path, max_pairs=20)

            self.assertEqual(stats["accepted"], 0)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "")
            self.assertEqual(stats["skipped_relevance"], 1)
            self.assertGreaterEqual(stats["skipped_length"], 3)
            self.assertGreaterEqual(stats["skipped_spam"], 4)

    def test_hot_starters_are_capped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "dataset.jsonl"
            output_path = tmp_path / "train_ready.txt"
            records: list[dict[str, object]] = []
            for index in range(55):
                records.append(
                    {
                        "question": f"Привет как жизнь номер {index}",
                        "answer": f"Все хорошо спасибо номер {index}",
                        "relevance": 1,
                    }
                )
            for index in range(55):
                records.append(
                    {
                        "question": f"Давай обсудим планы номер {index}",
                        "answer": f"Да конечно обсудим номер {index}",
                        "relevance": 1,
                    }
                )
            write_jsonl(input_path, records)

            stats = preprocess_dataset(input_path, output_path, max_pairs=200)
            lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertEqual(stats["accepted"], 100)
            self.assertEqual(len(lines), 100)
            self.assertEqual(stats["skipped_hot"], 10)


if __name__ == "__main__":
    unittest.main()
