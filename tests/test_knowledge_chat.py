import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from semantic_ants.engine import EngineConfig, SemanticEngine


class KnowledgeChatTest(unittest.TestCase):
    def test_builtin_greeting_works_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(
                config=EngineConfig(state_dir=Path(tmp), allow_network=False, ant_count=8, max_depth=3)
            )
            result = engine.analyze("привет", lang="ru")
            self.assertIn("Привет", result.response)
            self.assertTrue(any(item["uri"] == "/m/dialogue/greeting" for item in result.activated_concepts))

    def test_builtin_alphabet_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp), allow_network=False))
            result = engine.analyze("покажи русский алфавит", lang="ru")
            self.assertIn("Русский алфавит", result.response)
            self.assertIn("а б в", result.response)

    def test_chat_once_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "semantic_ants",
                    "--state-dir",
                    tmp,
                    "chat",
                    "--once",
                    "кто ты",
                    "--no-cache-refresh",
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONUTF8": "1"},
            )
            payload = json.loads(completed.stdout)
            self.assertIn("semantic_ants", payload["response"])


if __name__ == "__main__":
    unittest.main()
