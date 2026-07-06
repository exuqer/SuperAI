import importlib.util
import tempfile
import unittest
from pathlib import Path


PYMORPHY_AVAILABLE = bool(importlib.util.find_spec("pymorphy3"))


@unittest.skipUnless(PYMORPHY_AVAILABLE, 'install dependencies with: pip install -e ".[web]"')
class UnderstandingTest(unittest.TestCase):
    def make_checkpoint(self):
        from semantic_ants.engine import EngineConfig, SemanticEngine

        engine = SemanticEngine(config=EngineConfig(state_dir=Path(self.tmp), allow_network=False))
        return engine.checkpoint

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_russian_example_lemmatizes_to_search_tokens(self):
        from semantic_ants.understanding import understand_text

        checkpoint = self.make_checkpoint()
        result = understand_text("котики едят", lang="ru", checkpoint=checkpoint)

        self.assertEqual([token.search_token for token in result.tokens], ["кот", "есть"])
        self.assertEqual(result.tokens[0].concept_uri, "/m/concept/кот")
        self.assertEqual(result.tokens[1].concept_uri, "/m/concept/есть")
        self.assertEqual(result.tokens[0].match_status, "candidate")
        self.assertEqual(result.tokens[1].match_status, "candidate")

    def test_russian_morphology_is_preserved(self):
        from semantic_ants.understanding import understand_text

        checkpoint = self.make_checkpoint()
        result = understand_text("котиками", lang="ru", checkpoint=checkpoint)

        token = result.tokens[0]
        self.assertEqual(token.lemma, "кот")
        self.assertEqual(token.morphology["number"], "plur")
        self.assertTrue(token.morphology["case"])
        self.assertEqual(token.concept_uri, "/m/concept/кот")
        self.assertEqual(token.match_status, "candidate")

    def test_known_russian_lemma_is_not_stemmed(self):
        from semantic_ants.understanding import understand_text

        checkpoint = self.make_checkpoint()
        result = understand_text("голова", lang="ru", checkpoint=checkpoint)

        token = result.tokens[0]
        self.assertEqual(token.lemma, "голова")
        self.assertEqual(token.search_token, "голова")
        self.assertEqual(token.concept_uri, "/m/concept/голова")
        self.assertEqual(token.match_status, "found_as_lemma")

    def test_russian_name_keeps_its_normalized_form(self):
        from semantic_ants.understanding import understand_text

        checkpoint = self.make_checkpoint()
        result = understand_text("Яна", lang="ru", checkpoint=checkpoint)

        token = result.tokens[0]
        self.assertEqual(token.lemma, "яна")
        self.assertEqual(token.search_token, "яна")
        self.assertEqual(token.concept_uri, "/m/concept/яна")
        self.assertEqual(token.match_status, "found_as_raw")

    def test_kushat_uses_dictionary_entry_not_thought(self):
        from semantic_ants.understanding import understand_text

        checkpoint = self.make_checkpoint()
        result = understand_text("кушать", lang="ru", checkpoint=checkpoint)

        token = result.tokens[0]
        self.assertEqual(token.concept_uri, "/m/concept/есть")
        self.assertNotEqual(token.concept_uri, "/m/concept/мысль")
        self.assertNotEqual(token.search_token, "думать")

    def test_noise_words_are_ignored(self):
        from semantic_ants.understanding import understand_text

        checkpoint = self.make_checkpoint()
        result = understand_text("эй, ну и как там мой кот?", lang="ru", checkpoint=checkpoint)

        ignored = [token.raw_token for token in result.tokens if token.match_status == "ignored_stop_word"]
        self.assertTrue({"эй", "ну", "и", "как", "там", "мой"}.issubset(set(ignored)))
        self.assertTrue(any(token.raw_token == "кот" and not token.is_stop_word for token in result.tokens))

    def test_candidate_and_typo_match(self):
        from semantic_ants.understanding import understand_text

        checkpoint = self.make_checkpoint()
        unknown = understand_text("зигзагрон", lang="ru", checkpoint=checkpoint)
        typo = understand_text("aple", lang="en", checkpoint=checkpoint)

        self.assertEqual(unknown.tokens[0].match_status, "candidate")
        self.assertEqual(typo.tokens[0].match_status, "edit_distance_match")
        self.assertEqual(typo.tokens[0].concept_uri, "/m/concept/apple")

    def test_russian_word_is_not_replaced_by_edit_distance_neighbor(self):
        from semantic_ants.understanding import understand_text

        checkpoint = self.make_checkpoint()
        checkpoint.aliases["огонь"] = "/c/ru/огонь"

        result = understand_text("осень", lang="ru", checkpoint=checkpoint)

        self.assertEqual(result.tokens[0].search_token, "осень")
        self.assertEqual(result.tokens[0].concept_uri, "/m/concept/осень")
        self.assertEqual(result.tokens[0].match_status, "candidate")


if __name__ == "__main__":
    unittest.main()
