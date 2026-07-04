import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from semantic_ants.cli import main
from semantic_ants.datasets import convert_spc_csv


class DialogueDatasetTest(unittest.TestCase):
    def test_spc_multiline_csv_converts_to_dialogue_pairs(self):
        csv_text = (
            "user 1 personas,user 2 personas,Best Generated Conversation\n"
            "\"I like books.\",\"I run daily.\",\"User 1: hello there\n"
            "User 2: hi friend\n"
            "User 1: nice to meet you\"\n"
        )
        examples = list(convert_spc_csv(io.StringIO(csv_text), split="train", source_url="memory://test"))
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["stimulus"], "hello there")
        self.assertEqual(examples[0]["accepted_answer"], "hi friend")
        self.assertEqual(examples[1]["history"], [{"role": "user", "text": "hello there"}])
        self.assertEqual(examples[1]["metadata"]["license"], "CC-BY 4.0")

    def test_download_dataset_cli_uses_spc_converter(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "spc.jsonl"
            with patch("semantic_ants.cli.download_spc_dataset", return_value=3) as mocked:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    code = main(["download-dataset", "spc", "--split", "train", "--limit", "3", "--output", str(output), "--json"])
            self.assertEqual(code, 0)
            mocked.assert_called_once_with("train", str(output), limit=3)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["examples"], 3)


if __name__ == "__main__":
    unittest.main()
