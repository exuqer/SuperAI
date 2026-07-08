import json
from pathlib import Path
import unittest


class TrainingFixtureTest(unittest.TestCase):
    def test_training_jokes_fixture_has_ten_examples(self):
        fixture = Path(__file__).resolve().parents[1] / "web" / "src" / "pages" / "training" / "fixtures" / "jokes.json"
        data = json.loads(fixture.read_text(encoding="utf-8"))

        self.assertEqual(len(data), 10)
        self.assertTrue(all(item.get("question") for item in data))
        self.assertTrue(all(item.get("expected_answer") for item in data))


if __name__ == "__main__":
    unittest.main()
