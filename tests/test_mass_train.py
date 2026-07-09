from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock
import unittest

from semantic_ants.engine import DEFAULT_VECTOR_DIM, EngineConfig, SemanticEngine
from semantic_ants.mass_train import train_mass_dataset


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


class MassTrainTests(unittest.TestCase):
    def test_streams_jsonl_and_flushes_in_batches_without_reencoding_existing_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "dataset.jsonl"
            state_dir = tmp_path / "state"
            records = [
                {
                    "question": "Привет как дела сегодня",
                    "answer": "Все хорошо спасибо тебе",
                    "relevance": 1,
                },
                {
                    "question": "Привет как дела сегодня",
                    "answer": "Все хорошо спасибо тебе",
                    "relevance": 1,
                },
            ]
            write_jsonl(input_path, records)

            token_batches = 0
            flush_calls = 0

            def fake_embedding_many(self: SemanticEngine, texts: list[str]) -> list[list[float]]:
                nonlocal token_batches
                token_batches += 1
                return [[0.0] * DEFAULT_VECTOR_DIM for _ in texts]

            def fake_save_checkpoint(path: Path, checkpoint: object) -> None:
                nonlocal flush_calls
                flush_calls += 1
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            with mock.patch.object(SemanticEngine, "_embedding_many", new=fake_embedding_many), mock.patch(
                "semantic_ants.mass_train.save_checkpoint",
                side_effect=fake_save_checkpoint,
            ):
                report = train_mass_dataset(
                    input_path,
                    state_dir=state_dir,
                    batch_size=2,
                    input_format="jsonl",
                    session_id="unit",
                )

            self.assertEqual(report["dataset_records"], 2)
            self.assertEqual(report["source_pairs"], 2)
            self.assertEqual(report["flushes"], 1)
            self.assertEqual(flush_calls, 1)
            self.assertEqual(token_batches, 1)
            self.assertTrue((state_dir / "checkpoint.sqlite").exists())

    def test_streams_hierarchy_jsonl_into_hypernode_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "hierarchy.jsonl"
            state_dir = tmp_path / "state"
            records = [
                {
                    "hierarchy": ["docs", "intro"],
                    "text": "Привет как дела сегодня.",
                }
            ]
            write_jsonl(input_path, records)

            report = train_mass_dataset(
                input_path,
                state_dir=state_dir,
                batch_size=1,
                input_format="jsonl",
                session_id="unit",
            )

            engine = SemanticEngine(config=EngineConfig(state_dir=state_dir))
            node_id = engine._hierarchy_node_id(["docs", "intro"])
            self.assertGreaterEqual(report["dataset_records"], 1)
            self.assertIn(node_id, engine.checkpoint.meta["hypernodes"])


if __name__ == "__main__":
    unittest.main()
