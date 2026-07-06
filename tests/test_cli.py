import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from semantic_ants.learning import CheckpointStore, default_checkpoint_path


class CliTest(unittest.TestCase):
    def test_analyze_json_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "semantic_ants",
                    "--state-dir",
                    tmp,
                    "analyze",
                    "apple",
                    "--lang",
                    "en",
                    "--json",
                    "--no-cache-refresh",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["tokens"], ["apple"])

    def test_analyze_strength_vector_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "semantic_ants",
                    "--state-dir",
                    tmp,
                    "analyze",
                    "apple",
                    "--lang",
                    "en",
                    "--strength-vector",
                    "3",
                    "--json",
                    "--no-cache-refresh",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["semantic_vector"]["strength_vector"], [3])
            self.assertTrue(payload["signal_trace"])

    def test_interpret_vector_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vector.json"
            path.write_text(
                json.dumps(
                    {
                        "items": [
                            {"uri": "/m/top/object", "label": "предмет", "layer": 0, "score": 2.0},
                            {"uri": "/c/ru/яблоко", "label": "яблоко", "layer": 1, "score": 1.0},
                        ],
                        "strength_vector": [3],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "semantic_ants",
                    "--state-dir",
                    tmp,
                    "interpret-vector",
                    str(path),
                    "--no-cache-refresh",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("object", completed.stdout.lower())
            self.assertIn("\u044f\u0431\u043b\u043e\u043a\u043e", completed.stdout.lower())

    def test_train_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "text": "apple",
                        "lang": "en",
                        "target_concepts": ["/c/en/apple"],
                        "positive_edges": [["/c/en/apple", "/c/en/fruit"]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "semantic_ants",
                    "--state-dir",
                    tmp,
                    "train",
                    str(path),
                    "--epochs",
                    "1",
                    "--no-cache-refresh",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("examples=1", completed.stdout)

    def test_learn_dialogues_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dialogues.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "stimulus": "hello",
                        "accepted_answer": "learned dialogue reply",
                        "history": [],
                        "lang": "en",
                        "reward": 1.0,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "semantic_ants",
                    "--state-dir",
                    tmp,
                    "learn-dialogues",
                    str(path),
                    "--epochs",
                    "1",
                    "--max-examples",
                    "1",
                    "--no-cache-refresh",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("examples=1", completed.stdout)
            checkpoint = CheckpointStore(default_checkpoint_path(tmp)).load().to_dict()
            self.assertTrue(checkpoint["accepted_answers"])


if __name__ == "__main__":
    unittest.main()
