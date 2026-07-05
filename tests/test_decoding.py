import importlib.util
import tempfile
import unittest
from pathlib import Path


PYMORPHY_AVAILABLE = bool(importlib.util.find_spec("pymorphy3"))


@unittest.skipUnless(PYMORPHY_AVAILABLE, 'install dependencies with: pip install -e ".[web]"')
class DecodingTest(unittest.TestCase):
    def make_checkpoint(self):
        from semantic_ants.engine import EngineConfig, SemanticEngine

        engine = SemanticEngine(config=EngineConfig(state_dir=Path(self.tmp), allow_network=False))
        return engine.checkpoint

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_russian_tokens_become_sentence(self):
        from semantic_ants.decoding import decode_words

        result = decode_words("кот есть рыба мясо", lang="ru", tokens=["кот", "есть", "рыба", "мясо"])

        self.assertEqual(result.lang, "ru")
        self.assertEqual(result.pattern, "svo")
        self.assertEqual(result.sentence, "кот ест рыбу и мясо")
        self.assertEqual(result.summary.total_tokens, 4)
        self.assertEqual(result.summary.used_tokens, 4)
        self.assertEqual(result.summary.objects, 2)
        self.assertEqual(result.summary.fallbacks, 0)
        self.assertEqual([token.role for token in result.tokens], ["subject", "verb", "object", "object"])
        self.assertEqual([token.surface for token in result.tokens], ["кот", "ест", "рыбу", "мясо"])

    def test_russian_tokens_can_be_unordered(self):
        from semantic_ants.decoding import decode_words

        result = decode_words("", lang="ru", tokens=["шкаф", "собирать", "рабочий"])

        self.assertEqual(result.sentence, "рабочий собирает шкаф")
        self.assertEqual([token.role for token in result.tokens], ["subject", "verb", "object"])
        self.assertEqual([token.input_token for token in result.tokens], ["рабочий", "собирать", "шкаф"])

    def test_russian_change_state_uses_modifier_and_complement(self):
        from semantic_ants.decoding import decode_words

        result = decode_words("", lang="ru", tokens=["осень", "лист", "становиться", "желтый"])

        self.assertEqual(result.sentence, "осенью лист становится жёлтым")
        self.assertEqual([token.role for token in result.tokens], ["modifier", "subject", "verb", "complement"])
        self.assertEqual([token.surface for token in result.tokens], ["осенью", "лист", "становится", "жёлтым"])

    def test_russian_single_token_does_not_duplicate_as_verb(self):
        from semantic_ants.decoding import decode_words

        result = decode_words("", lang="ru", tokens=["кот"])

        self.assertEqual(result.pattern, "s")
        self.assertEqual(result.sentence, "кот")
        self.assertEqual([token.role for token in result.tokens], ["subject"])

    def test_checkpoint_edges_bias_decoder_roles(self):
        from semantic_ants.decoding import decode_words

        checkpoint = self.make_checkpoint()
        checkpoint.add_custom_edge("/c/ru/программист", "/c/ru/писать", relation="CanDo", weight=2.6)
        checkpoint.add_custom_edge("/c/ru/писать", "/c/ru/код", relation="TakesObject", weight=2.8)
        checkpoint.add_custom_edge("/c/ru/писать", "/c/ru/компьютер", relation="UsesInstrument", weight=2.4)
        checkpoint.reinforce_edge("/c/ru/программист", "CanDo", "/c/ru/писать", amount=0.6)
        checkpoint.reinforce_edge("/c/ru/писать", "TakesObject", "/c/ru/код", amount=0.6)
        checkpoint.reinforce_edge("/c/ru/писать", "UsesInstrument", "/c/ru/компьютер", amount=0.6)

        result = decode_words(
            "",
            lang="ru",
            tokens=["компьютер", "код", "писать", "программист"],
            checkpoint=checkpoint,
        )

        self.assertEqual(result.sentence, "программист пишет код на компьютере")
        self.assertEqual([token.role for token in result.tokens], ["subject", "verb", "object", "instrument"])
        self.assertEqual(result.tokens[3].surface, "на компьютере")

    def test_checkpoint_edges_use_ablative_with_non_device_instrument(self):
        from semantic_ants.decoding import decode_words

        checkpoint = self.make_checkpoint()
        checkpoint.add_custom_edge("/c/ru/рабочий", "/c/ru/забивать", relation="CanDo", weight=2.6)
        checkpoint.add_custom_edge("/c/ru/забивать", "/c/ru/гвоздь", relation="TakesObject", weight=2.8)
        checkpoint.add_custom_edge("/c/ru/забивать", "/c/ru/молоток", relation="UsesInstrument", weight=2.4)
        checkpoint.reinforce_edge("/c/ru/рабочий", "CanDo", "/c/ru/забивать", amount=0.6)
        checkpoint.reinforce_edge("/c/ru/забивать", "TakesObject", "/c/ru/гвоздь", amount=0.6)
        checkpoint.reinforce_edge("/c/ru/забивать", "UsesInstrument", "/c/ru/молоток", amount=0.6)

        result = decode_words(
            "",
            lang="ru",
            tokens=["молоток", "гвоздь", "забивать", "рабочий"],
            checkpoint=checkpoint,
        )

        self.assertEqual(result.sentence, "рабочий забивает гвоздь с молотком")
        self.assertEqual([token.role for token in result.tokens], ["subject", "verb", "object", "instrument"])
        self.assertEqual(result.tokens[3].surface, "с молотком")

    def test_english_tokens_become_sentence(self):
        from semantic_ants.decoding import decode_words

        result = decode_words("cat eat fish meat", lang="en", tokens=["cat", "eat", "fish", "meat"])

        self.assertEqual(result.lang, "en")
        self.assertEqual(result.sentence, "cat eats fish and meat")
        self.assertEqual(result.summary.objects, 2)
        self.assertEqual(result.tokens[1].surface, "eats")

    def test_text_input_works_without_tokens(self):
        from semantic_ants.decoding import decode_words

        result = decode_words("кот есть рыба мясо", lang="ru")

        self.assertEqual(result.input_tokens, ["кот", "есть", "рыба", "мясо"])
        self.assertEqual(result.sentence, "кот ест рыбу и мясо")

    def test_tokens_override_text(self):
        from semantic_ants.decoding import decode_words

        result = decode_words("кот есть рыба мясо", lang="ru", tokens=["пёс", "есть", "кость"])

        self.assertEqual(result.input_tokens, ["пёс", "есть", "кость"])
        self.assertEqual(result.summary.total_tokens, 3)
        self.assertEqual(result.summary.objects, 1)

    def test_empty_input_returns_empty_result(self):
        from semantic_ants.decoding import decode_words

        result = decode_words("", lang="auto", tokens=[])

        self.assertEqual(result.pattern, "empty")
        self.assertEqual(result.sentence, "")
        self.assertEqual(result.summary.total_tokens, 0)
        self.assertEqual(result.tokens, [])


if __name__ == "__main__":
    unittest.main()
