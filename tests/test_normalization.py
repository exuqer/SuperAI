import unittest

from semantic_ants.core.normalization import detect_language, detect_response_language, text_to_concept_uri, tokenize


class NormalizationTest(unittest.TestCase):
    def test_detect_language(self):
        self.assertEqual(detect_language("яблоко"), "ru")
        self.assertEqual(detect_language("apple"), "en")

    def test_tokenize(self):
        self.assertEqual(tokenize("Яблоко упало!"), ["яблоко", "упало"])

    def test_concept_uri(self):
        self.assertEqual(text_to_concept_uri("Apple", "en"), "/c/en/apple")
        self.assertEqual(text_to_concept_uri("Красное яблоко", "ru"), "/c/ru/красное_яблоко")

    def test_detect_response_language(self):
        self.assertEqual(detect_response_language("переведи на английский яблоко", default="ru"), "en")
        self.assertEqual(detect_response_language("translate to russian: hello", default="en"), "ru")
        self.assertEqual(detect_response_language("привет", default="ru"), "ru")


if __name__ == "__main__":
    unittest.main()
