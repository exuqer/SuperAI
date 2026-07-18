from __future__ import annotations

from server.tokenizer import tokenize_hierarchical
from server.v2.graph_service import GraphTrainingService
from server.v2.language import UniversalLanguageAnalyzer
from server.v2.russian_morphology import Morphology


class _Morphology:
    """Small deterministic morphology for sentence-boundary regression tests."""

    _entries = {
        "на": ("на", "PREP", {}),
        "под": ("под", "PREP", {}),
        "красный": ("красный", "ADJF", {"case": "nomn", "number": "sing", "gender": "masc"}),
        "красное": ("красный", "ADJF", {"case": "nomn", "number": "sing", "gender": "neut"}),
        "красном": ("красный", "ADJF", {"case": "loct", "number": "sing", "gender": "masc"}),
        "помидор": ("помидор", "NOUN", {"case": "nomn", "number": "sing", "gender": "masc"}),
        "яблоко": ("яблоко", "NOUN", {"case": "nomn", "number": "sing", "gender": "neut"}),
        "столе": ("стол", "NOUN", {"case": "loct", "number": "sing", "gender": "masc"}),
        "столом": ("стол", "NOUN", {"case": "ablt", "number": "sing", "gender": "masc"}),
        "подоконнике": ("подоконник", "NOUN", {"case": "loct", "number": "sing", "gender": "masc"}),
        "мальчик": ("мальчик", "NOUN", {"case": "nomn", "number": "sing", "gender": "masc"}),
        "девочка": ("девочка", "NOUN", {"case": "nomn", "number": "sing", "gender": "femn"}),
        "сидит": ("сидеть", "VERB", {"number": "sing"}),
        "лежит": ("лежать", "VERB", {"number": "sing"}),
        "мяч": ("мяч", "NOUN", {"case": "nomn", "number": "sing", "gender": "masc"}),
    }

    def parse_variants(self, word: str):
        lemma, pos, features = self._entries[word.casefold()]
        return [Morphology(lemma, pos, features, 1.0)]


def test_tokenizer_closes_sentence_on_period_and_newline_before_mentions():
    hierarchy = tokenize_hierarchical("Красное яблоко.Яблоко\nНа подоконнике. Девочка")

    assert [token.sentence_index for token in hierarchy.all_tokens] == [0, 0, 1, 2, 2, 3]


def test_mentions_phrases_and_prepositions_do_not_cross_sentence_boundaries():
    analysis = UniversalLanguageAnalyzer(_Morphology()).analyze(
        "Красный помидор. Помидор лежит. "
        "На красном столе. Мальчик сидит. "
        "Под\nстолом лежит мяч."
    )

    mention_surfaces = [mention.surface for mention in analysis.mentions]
    assert "Красный помидор Помидор" not in mention_surfaces
    assert "столе Мальчик" not in mention_surfaces
    assert mention_surfaces[:4] == [
        "Красный помидор",
        "Помидор",
        "красном столе",
        "Мальчик",
    ]
    assert next(
        mention for mention in analysis.mentions if mention.surface == "столом"
    ).preposition == ""

    assert all(len(mention.sentence_indices) == 1 for mention in analysis.mentions)
    assert all(
        len(phrase.metadata["sentence_indices"]) == 1
        for phrase in analysis.phrase_graph.phrases
        if phrase.phrase_type == "noun_phrase"
    )
    assert any(
        event["code"] == "SENTENCE_BOUNDARY_CROSSING"
        and event["construction"] == "apposition"
        for event in analysis.diagnostics
    )
    assert any(
        event["code"] == "SENTENCE_BOUNDARY_CROSSING"
        and event["construction"] == "preposition"
        for event in analysis.diagnostics
    )


def test_training_does_not_persist_a_cross_sentence_participant():
    result = GraphTrainingService(morphology=_Morphology()).train(
        "Красный помидор. Помидор лежит.",
        independent_key="sentence-boundary",
    )

    participants = result["events"][0]["participants"]
    assert [item["mention"]["surface"] for item in participants] == [
        "Помидор"
    ]
