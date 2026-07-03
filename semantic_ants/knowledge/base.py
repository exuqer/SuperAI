from __future__ import annotations

from dataclasses import dataclass

from semantic_ants.learning.checkpoint import Checkpoint

SEED_VERSION = 2

ALPHABETS = {
    "ru": list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"),
    "en": list("abcdefghijklmnopqrstuvwxyz"),
}

COMMON_WORDS = {
    "ru": [
        "и",
        "в",
        "не",
        "на",
        "я",
        "быть",
        "он",
        "с",
        "что",
        "а",
        "по",
        "это",
        "она",
        "этот",
        "к",
        "но",
        "они",
        "мы",
        "как",
        "из",
        "у",
        "который",
        "то",
        "за",
        "свой",
        "весь",
        "год",
        "от",
        "так",
        "о",
        "для",
        "ты",
        "же",
        "все",
        "тот",
        "мочь",
        "вы",
        "человек",
        "такой",
        "его",
        "сказать",
        "только",
        "или",
        "еще",
        "бы",
        "себя",
        "один",
        "как",
        "когда",
        "уже",
    ],
    "en": [
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
    ],
}

ALIASES = {
    "ru": {
        "алфавит": "/c/ru/алфавит",
        "буквы": "/c/ru/буква",
        "буква": "/c/ru/буква",
        "привет": "/c/ru/привет",
        "здравствуй": "/c/ru/привет",
        "здравствуйте": "/c/ru/привет",
        "пока": "/c/ru/пока",
        "спасибо": "/c/ru/спасибо",
        "благодарю": "/c/ru/спасибо",
        "кто": "/c/ru/кто",
        "что": "/c/ru/что",
        "как": "/c/ru/как",
        "умеешь": "/c/ru/уметь",
        "можешь": "/c/ru/мочь",
        "модель": "/c/ru/модель",
        "бот": "/c/ru/бот",
        "смысл": "/c/ru/смысл",
        "смыслы": "/c/ru/смысл",
        "слово": "/c/ru/слово",
        "слова": "/c/ru/слово",
        "частые": "/c/ru/частый",
        "популярные": "/c/ru/частый",
        "используемые": "/c/ru/частый",
        "яблоко": "/c/ru/яблоко",
        "яблока": "/c/ru/яблоко",
        "упало": "/c/ru/падать",
        "упал": "/c/ru/падать",
        "падает": "/c/ru/падать",
        "голову": "/c/ru/голова",
        "голова": "/c/ru/голова",
        "пол": "/c/ru/пол",
        "саша": "/c/ru/саша",
        "шла": "/c/ru/идти",
        "шоссе": "/c/ru/шоссе",
    },
    "en": {
        "hello": "/c/en/hello",
        "hi": "/c/en/hello",
        "bye": "/c/en/goodbye",
        "thanks": "/c/en/thanks",
        "thank": "/c/en/thanks",
        "alphabet": "/c/en/alphabet",
        "letters": "/c/en/letter",
        "letter": "/c/en/letter",
        "word": "/c/en/word",
        "words": "/c/en/word",
        "meaning": "/c/en/meaning",
        "meanings": "/c/en/meaning",
        "common": "/c/en/common",
        "frequent": "/c/en/common",
        "apple": "/c/en/apple",
        "fell": "/c/en/fall",
        "fall": "/c/en/fall",
        "head": "/c/en/head",
        "floor": "/c/en/floor",
    },
}

SEED_EDGES = [
    ("/c/ru/привет", "/m/dialogue/greeting", "Expresses"),
    ("/c/en/hello", "/m/dialogue/greeting", "Expresses"),
    ("/c/ru/пока", "/m/dialogue/farewell", "Expresses"),
    ("/c/en/goodbye", "/m/dialogue/farewell", "Expresses"),
    ("/c/ru/спасибо", "/m/dialogue/gratitude", "Expresses"),
    ("/c/en/thanks", "/m/dialogue/gratitude", "Expresses"),
    ("/c/ru/кто", "/m/dialogue/identity_question", "AsksAbout"),
    ("/c/en/who", "/m/dialogue/identity_question", "AsksAbout"),
    ("/c/ru/уметь", "/m/dialogue/capability_question", "AsksAbout"),
    ("/c/ru/мочь", "/m/dialogue/capability_question", "AsksAbout"),
    ("/c/en/can", "/m/dialogue/capability_question", "AsksAbout"),
    ("/c/ru/алфавит", "/m/language/alphabet", "Means"),
    ("/c/ru/буква", "/m/language/alphabet", "PartOf"),
    ("/c/en/alphabet", "/m/language/alphabet", "Means"),
    ("/c/en/letter", "/m/language/alphabet", "PartOf"),
    ("/c/ru/слово", "/m/language/word", "Means"),
    ("/c/en/word", "/m/language/word", "Means"),
    ("/c/ru/смысл", "/m/language/meaning", "Means"),
    ("/c/en/meaning", "/m/language/meaning", "Means"),
    ("/c/ru/частый", "/m/language/common_words", "Qualifies"),
    ("/c/en/common", "/m/language/common_words", "Qualifies"),
    ("/m/dialogue/greeting", "/m/dialogue/simple_chat", "PartOf"),
    ("/m/dialogue/farewell", "/m/dialogue/simple_chat", "PartOf"),
    ("/m/dialogue/gratitude", "/m/dialogue/simple_chat", "PartOf"),
    ("/m/dialogue/identity_question", "/m/dialogue/simple_chat", "PartOf"),
    ("/m/dialogue/capability_question", "/m/dialogue/simple_chat", "PartOf"),
    ("/c/ru/яблоко", "/m/object/apple", "Means"),
    ("/c/en/apple", "/m/object/apple", "Means"),
    ("/c/ru/падать", "/m/action/fall", "Means"),
    ("/c/en/fall", "/m/action/fall", "Means"),
    ("/c/ru/голова", "/m/body/head", "Means"),
    ("/c/en/head", "/m/body/head", "Means"),
    ("/c/ru/пол", "/m/place/floor", "Means"),
    ("/c/en/floor", "/m/place/floor", "Means"),
    ("/m/object/apple", "/m/action/fall", "CanParticipateIn"),
    ("/m/action/fall", "/m/body/head", "CanAffect"),
    ("/m/action/fall", "/m/place/floor", "CanEndAt"),
    ("/m/object/apple", "/m/science/newton_story", "SymbolicallyRelatedTo"),
    ("/m/body/head", "/m/science/newton_story", "ContextOf"),
    ("/c/ru/саша", "/m/entity/person", "IsA"),
    ("/c/ru/идти", "/m/action/move", "Means"),
    ("/c/ru/шоссе", "/m/place/road", "Means"),
]

SEED_RESPONSES = [
    (
        ["/m/dialogue/greeting"],
        "Привет. Я простой исследовательский чат на смысловом графе. Можешь задать вопрос.",
    ),
    (
        ["/m/dialogue/farewell"],
        "Пока. Я сохраню обученный слой в checkpoint.",
    ),
    (
        ["/m/dialogue/gratitude"],
        "Пожалуйста.",
    ),
    (
        ["/m/dialogue/identity_question"],
        "Я прототип semantic_ants: разбираю слова, ищу смыслы в графе и учусь на feedback.",
    ),
    (
        ["/m/dialogue/capability_question"],
        "Я могу отвечать на простые вопросы, показывать алфавиты, частые слова, трассировать смыслы и запоминать feedback.",
    ),
    (
        ["/m/object/apple", "/m/action/fall", "/m/body/head"],
        "Это похоже на сюжет про падающее яблоко и ассоциацию с Ньютоном.",
    ),
    (
        ["/m/object/apple", "/m/action/fall", "/m/place/floor"],
        "Здесь смысл простой: яблоко падает вниз и оказывается на полу.",
    ),
]


@dataclass(frozen=True)
class SeedReport:
    aliases: int = 0
    edges: int = 0
    responses: int = 0
    alphabets: int = 0
    common_words: int = 0
    changed: bool = False

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "aliases": self.aliases,
            "edges": self.edges,
            "responses": self.responses,
            "alphabets": self.alphabets,
            "common_words": self.common_words,
            "changed": self.changed,
        }


def bootstrap_builtin_knowledge(checkpoint: Checkpoint, force: bool = False) -> SeedReport:
    if not force and checkpoint.metadata.get("builtin_seed_version") == SEED_VERSION:
        return SeedReport(changed=False)

    aliases = _load_aliases(checkpoint)
    edges = _load_edges(checkpoint)
    responses = _load_responses(checkpoint)
    alphabets = _load_alphabets(checkpoint)
    common_words = _load_common_words(checkpoint)
    checkpoint.metadata["builtin_seed_version"] = SEED_VERSION
    checkpoint.metadata["builtin_seed_loaded"] = True
    return SeedReport(
        aliases=aliases,
        edges=edges,
        responses=responses,
        alphabets=alphabets,
        common_words=common_words,
        changed=True,
    )


def _load_aliases(checkpoint: Checkpoint) -> int:
    changed = 0
    for aliases in ALIASES.values():
        for word, uri in aliases.items():
            if checkpoint.aliases.get(word) != uri:
                checkpoint.aliases[word] = uri
                changed += 1
    return changed


def _load_edges(checkpoint: Checkpoint) -> int:
    changed = 0
    for start, end, relation in SEED_EDGES:
        before = len(checkpoint.custom_edges)
        checkpoint.add_custom_edge(start, end, relation=relation, weight=1.4)
        checkpoint.reinforce_edge(start, relation, end, amount=0.3)
        if len(checkpoint.custom_edges) != before:
            changed += 1
    return changed


def _load_responses(checkpoint: Checkpoint) -> int:
    changed = 0
    for concepts, response in SEED_RESPONSES:
        checkpoint.remember_response(concepts, response, amount=2.0)
        changed += 1
    return changed


def _load_alphabets(checkpoint: Checkpoint) -> int:
    total = 0
    checkpoint.metadata["alphabets"] = ALPHABETS
    for lang, letters in ALPHABETS.items():
        alphabet_uri = f"/m/alphabet/{lang}"
        lang_uri = f"/c/{lang}/alphabet" if lang == "en" else f"/c/{lang}/алфавит"
        checkpoint.add_custom_edge(lang_uri, alphabet_uri, relation="HasAlphabet", weight=1.2)
        for index, letter in enumerate(letters, start=1):
            letter_uri = f"/m/alphabet/{lang}/{letter}"
            checkpoint.add_custom_edge(alphabet_uri, letter_uri, relation="ContainsLetter", weight=0.5)
            checkpoint.add_custom_edge(letter_uri, alphabet_uri, relation="LetterOf", weight=0.5)
            checkpoint.metadata.setdefault("letter_order", {})[letter_uri] = index
            total += 1
    return total


def _load_common_words(checkpoint: Checkpoint) -> int:
    total = 0
    checkpoint.metadata["common_words"] = COMMON_WORDS
    for lang, words in COMMON_WORDS.items():
        common_uri = f"/m/common_words/{lang}"
        checkpoint.add_custom_edge("/m/language/common_words", common_uri, relation="HasLanguage", weight=1.0)
        for rank, word in enumerate(words, start=1):
            word_uri = ALIASES.get(lang, {}).get(word, f"/c/{lang}/{word}")
            checkpoint.add_custom_edge(word_uri, common_uri, relation="FrequentWord", weight=max(1.0, 2.0 - rank / 50))
            checkpoint.metadata.setdefault("word_frequency_rank", {})[word_uri] = rank
            total += 1
    return total
