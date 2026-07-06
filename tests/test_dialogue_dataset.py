import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from semantic_ants.cli import main
from semantic_ants.datasets import convert_koziev_dialogues, convert_spc_csv, convert_tatoeba_translation_pairs


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

    def test_koziev_dialogues_convert_to_training_examples(self):
        text = (
            "User: Привет\n"
            "Assistant: Привет! Как дела?\n"
            "User: Всё хорошо\n\n"
            "User: Hello\n"
            "Assistant: Hi there\n"
        )
        examples = list(
            convert_koziev_dialogues(
                io.StringIO(text),
                source_url="memory://koziev",
                source_path="Conversations/Data/chan_dialogues.txt",
            )
        )
        self.assertEqual(len(examples), 3)
        self.assertEqual(examples[0]["stimulus"], "Привет")
        self.assertEqual(examples[0]["accepted_answer"], "Привет! Как дела?")
        self.assertEqual(examples[0]["lang"], "ru")
        self.assertEqual(examples[0]["answer_lang"], "ru")
        self.assertEqual(examples[0]["source_lang"], "ru")
        self.assertTrue(examples[0]["positive_edges"])
        self.assertEqual(examples[1]["history"], [{"role": "user", "text": "Привет"}])
        self.assertEqual(examples[2]["metadata"]["dataset"], "Koziev/NLP_Datasets")

    def test_tatoeba_translation_pairs_convert_to_bidirectional_examples(self):
        source = io.StringIO("1\trus\tПривет\n")
        target = io.StringIO("10\teng\tHello\n")
        links = io.StringIO("1\t10\n")
        examples = list(
            convert_tatoeba_translation_pairs(
                source,
                target,
                links,
                source_lang="ru",
                target_lang="en",
                source_sentence_url="memory://source",
                target_sentence_url="memory://target",
                links_url="memory://links",
                bidirectional=True,
            )
        )
        self.assertEqual(len(examples), 2)
        forward, reverse = examples
        self.assertEqual(forward["stimulus"], "Привет")
        self.assertEqual(forward["accepted_answer"], "Hello")
        self.assertEqual(forward["lang"], "ru")
        self.assertEqual(forward["answer_lang"], "en")
        self.assertEqual(forward["source_lang"], "ru")
        self.assertEqual(forward["target_lang"], "en")
        self.assertTrue(forward["positive_edges"])
        self.assertEqual(reverse["stimulus"], "Hello")
        self.assertEqual(reverse["accepted_answer"], "Привет")
        self.assertEqual(reverse["lang"], "en")
        self.assertEqual(reverse["answer_lang"], "ru")
        self.assertEqual(reverse["source_lang"], "en")
        self.assertEqual(reverse["target_lang"], "ru")

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

    def test_download_dataset_cli_uses_koziev_converter(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "koziev.jsonl"
            with patch("semantic_ants.cli.download_koziev_dialogues_dataset", return_value=2) as mocked:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    code = main(
                        [
                            "download-dataset",
                            "koziev",
                            "--path",
                            "Conversations/Data/chan_dialogues.txt",
                            "--limit",
                            "2",
                            "--output",
                            str(output),
                            "--json",
                        ]
                    )
            self.assertEqual(code, 0)
            mocked.assert_called_once_with(str(output), source="Conversations/Data/chan_dialogues.txt", limit=2, timeout=60.0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["dataset"], "koziev")
            self.assertEqual(payload["examples"], 2)

    def test_download_dataset_cli_uses_tatoeba_converter(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "tatoeba.jsonl"
            with patch("semantic_ants.cli.download_tatoeba_translation_dataset", return_value=4) as mocked:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    code = main(
                        [
                            "download-dataset",
                            "tatoeba",
                            "--source-lang",
                            "ru",
                            "--target-lang",
                            "en",
                            "--no-bidirectional",
                            "--limit",
                            "4",
                            "--output",
                            str(output),
                            "--json",
                        ]
                    )
            self.assertEqual(code, 0)
            mocked.assert_called_once_with(
                str(output),
                source_lang="ru",
                target_lang="en",
                limit=4,
                bidirectional=False,
                timeout=60.0,
            )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["dataset"], "tatoeba")
            self.assertEqual(payload["examples"], 4)


if __name__ == "__main__":
    unittest.main()
