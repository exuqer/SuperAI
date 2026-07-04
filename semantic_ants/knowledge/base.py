from __future__ import annotations

from dataclasses import dataclass

from semantic_ants.learning.checkpoint import Checkpoint

SEED_VERSION = 3

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

BASIC_CONCEPTS = [
    {
        "uri": "/m/basic/learning",
        "category": "mind",
        "aliases": {"ru": ["учиться", "обучение", "обучи", "учи"], "en": ["learning", "learn", "teach"]},
        "label": "обучение",
        "definition": "способ становиться умнее через примеры, ошибки и повторение",
        "action": "связывает новый пример с уже знакомым смыслом",
        "image": "новая мысль делает шаг от слова к смыслу",
    },
    {
        "uri": "/m/basic/child",
        "category": "person",
        "aliases": {"ru": ["ребенок", "ребёнок", "малыш"], "en": ["child", "kid"]},
        "label": "ребенок",
        "definition": "маленький человек, который учится понимать мир через простые примеры",
        "action": "задает вопросы и запоминает понятные ответы",
        "image": "ребенок смотрит на мир как на большую книгу с картинками",
    },
    {
        "uri": "/m/basic/word",
        "category": "language",
        "aliases": {"ru": ["слово", "слова"], "en": ["word", "words"]},
        "label": "слово",
        "definition": "знак речи, который называет предмет, действие или чувство",
        "action": "помогает дать имени смысл",
        "image": "слово становится маленькой табличкой для мысли",
    },
    {
        "uri": "/m/basic/sentence",
        "category": "language",
        "aliases": {"ru": ["предложение", "предложения", "фраза", "фразу"], "en": ["sentence", "phrase"]},
        "label": "предложение",
        "definition": "несколько слов, соединенных в законченную мысль",
        "action": "соединяет слова так, чтобы появился понятный ответ",
        "image": "слова выстраиваются в дорожку, по которой идет мысль",
    },
    {
        "uri": "/m/basic/meaning",
        "category": "language",
        "aliases": {"ru": ["смысл", "смыслы", "значение"], "en": ["meaning", "meanings"]},
        "label": "смысл",
        "definition": "то, что слово или фраза хотят сказать на самом деле",
        "action": "помогает понять, о чем речь",
        "image": "смысл зажигает внутри слова маленький свет",
    },
    {
        "uri": "/m/basic/question",
        "category": "dialogue",
        "aliases": {"ru": ["вопрос", "почему", "зачем"], "en": ["question", "why"]},
        "label": "вопрос",
        "definition": "фраза, которая просит найти недостающий смысл",
        "action": "ищет ответ среди известных связей",
        "image": "вопрос открывает дверь к новому знанию",
    },
    {
        "uri": "/m/basic/answer",
        "category": "dialogue",
        "aliases": {"ru": ["ответ", "ответь"], "en": ["answer", "reply"]},
        "label": "ответ",
        "definition": "фраза, которая закрывает вопрос понятным смыслом",
        "action": "возвращает найденный смысл человеку",
        "image": "ответ складывает кусочки мысли в одну картинку",
    },
    {
        "uri": "/m/basic/imagination",
        "category": "mind",
        "aliases": {
            "ru": ["фантазия", "пофантазируй", "придумай", "вообрази", "выдумай"],
            "en": ["fantasy", "imagination", "imagine", "invent"],
        },
        "label": "фантазия",
        "definition": "умение соединять знакомые смыслы в новый образ",
        "action": "смешивает реальные знания и воображаемые связи",
        "image": "фантазия строит мост между тем, что есть, и тем, что можно представить",
    },
    {
        "uri": "/m/basic/story",
        "category": "language",
        "aliases": {"ru": ["история", "сказка", "сюжет"], "en": ["story", "tale"]},
        "label": "история",
        "definition": "цепочка событий, где есть начало, действие и итог",
        "action": "превращает отдельные смыслы в маленькое приключение",
        "image": "история ведет смысл от первого шага к завершению",
    },
    {
        "uri": "/m/basic/thought",
        "category": "mind",
        "aliases": {"ru": ["мысль", "думать", "понимать"], "en": ["thought", "think", "understand"]},
        "label": "мысль",
        "definition": "внутренний образ, который помогает выбрать ответ",
        "action": "сравнивает понятия и ищет связь",
        "image": "мысль похожа на тихую карту внутри головы",
    },
    {
        "uri": "/m/basic/sun",
        "category": "nature",
        "aliases": {"ru": ["солнце", "солнышко"], "en": ["sun"]},
        "label": "солнце",
        "definition": "звезда, которая дает Земле свет и тепло",
        "action": "светит и греет",
        "image": "солнце поднимается и делает день ярким",
    },
    {
        "uri": "/m/basic/light",
        "category": "nature",
        "aliases": {"ru": ["свет", "светло"], "en": ["light"]},
        "label": "свет",
        "definition": "то, благодаря чему можно видеть предметы",
        "action": "показывает форму и цвет",
        "image": "свет открывает предметы для глаз",
    },
    {
        "uri": "/m/basic/sky",
        "category": "nature",
        "aliases": {"ru": ["небо"], "en": ["sky"]},
        "label": "небо",
        "definition": "пространство над землей, где видны облака, солнце и звезды",
        "action": "держит над нами большую открытую высоту",
        "image": "небо становится широкой синей крышей мира",
    },
    {
        "uri": "/m/basic/water",
        "category": "nature",
        "aliases": {"ru": ["вода", "воды"], "en": ["water"]},
        "label": "вода",
        "definition": "жидкость, нужная живому для жизни",
        "action": "утоляет жажду и помогает расти",
        "image": "вода течет и оживляет сухую землю",
    },
    {
        "uri": "/m/basic/fire",
        "category": "nature",
        "aliases": {"ru": ["огонь", "пламя"], "en": ["fire", "flame"]},
        "label": "огонь",
        "definition": "горячее свечение, которое дает тепло и может обжечь",
        "action": "греет, светит и требует осторожности",
        "image": "огонь танцует теплым светом",
    },
    {
        "uri": "/m/basic/earth",
        "category": "nature",
        "aliases": {"ru": ["земля", "почва"], "en": ["earth", "soil"]},
        "label": "земля",
        "definition": "поверхность под ногами и почва для растений",
        "action": "держит предметы и дает место для роста",
        "image": "земля становится прочной основой под шагами",
    },
    {
        "uri": "/m/basic/home",
        "category": "place",
        "aliases": {"ru": ["дом", "дома"], "en": ["home", "house"]},
        "label": "дом",
        "definition": "место, где человек живет и чувствует безопасность",
        "action": "защищает и собирает людей вместе",
        "image": "дом светится окном и зовет вернуться",
    },
    {
        "uri": "/m/basic/tree",
        "category": "nature",
        "aliases": {"ru": ["дерево", "деревья"], "en": ["tree", "trees"]},
        "label": "дерево",
        "definition": "растение со стволом, ветками и листьями",
        "action": "растет из земли к свету",
        "image": "дерево тянет ветки вверх, как зеленую лестницу",
    },
    {
        "uri": "/m/basic/food",
        "category": "body",
        "aliases": {"ru": ["еда", "пища"], "en": ["food"]},
        "label": "еда",
        "definition": "то, что дает телу силы",
        "action": "питает и помогает двигаться",
        "image": "еда превращается в энергию для нового дня",
    },
    {
        "uri": "/m/basic/apple",
        "category": "object",
        "aliases": {"ru": ["яблоко", "яблока"], "en": ["apple"]},
        "label": "яблоко",
        "definition": "круглый плод, который можно есть",
        "action": "может быть едой и примером предмета",
        "image": "яблоко лежит в ладони как маленький круглый плод",
    },
    {
        "uri": "/m/basic/person",
        "category": "person",
        "aliases": {"ru": ["человек", "люди"], "en": ["person", "people"]},
        "label": "человек",
        "definition": "живое существо, которое думает, чувствует и общается словами",
        "action": "задает вопросы, учится и выбирает действия",
        "image": "человек смотрит на мир и ищет смысл",
    },
    {
        "uri": "/m/basic/color",
        "category": "perception",
        "aliases": {"ru": ["цвет", "цвета"], "en": ["color", "colour"]},
        "label": "цвет",
        "definition": "свойство предмета, которое видно благодаря свету",
        "action": "помогает отличать предметы друг от друга",
        "image": "цвет раскрашивает мир в заметные различия",
    },
    {
        "uri": "/m/basic/red",
        "category": "perception",
        "aliases": {"ru": ["красный", "красная"], "en": ["red"]},
        "label": "красный",
        "definition": "яркий цвет, похожий на спелое яблоко или огонь",
        "action": "делает предмет заметным",
        "image": "красный вспыхивает как теплый знак внимания",
    },
    {
        "uri": "/m/basic/blue",
        "category": "perception",
        "aliases": {"ru": ["синий", "синяя", "голубой"], "en": ["blue"]},
        "label": "синий",
        "definition": "цвет, который часто связывают с небом и водой",
        "action": "дает ощущение пространства и прохлады",
        "image": "синий раскрывается как спокойное небо",
    },
    {
        "uri": "/m/basic/green",
        "category": "perception",
        "aliases": {"ru": ["зеленый", "зелёный", "зеленая", "зелёная"], "en": ["green"]},
        "label": "зеленый",
        "definition": "цвет листьев, травы и роста",
        "action": "напоминает о жизни и растениях",
        "image": "зеленый появляется там, где что-то растет",
    },
    {
        "uri": "/m/basic/number",
        "category": "math",
        "aliases": {"ru": ["число", "цифра", "счет", "счёт"], "en": ["number", "count"]},
        "label": "число",
        "definition": "знак количества или порядка",
        "action": "помогает считать предметы",
        "image": "число ставит предметы в понятный порядок",
    },
    {
        "uri": "/m/basic/one",
        "category": "math",
        "aliases": {"ru": ["один", "одна", "первый"], "en": ["one", "first"]},
        "label": "один",
        "definition": "самое простое количество: один предмет",
        "action": "начинает счет",
        "image": "один становится первой точкой счета",
    },
    {
        "uri": "/m/basic/two",
        "category": "math",
        "aliases": {"ru": ["два", "две", "второй"], "en": ["two", "second"]},
        "label": "два",
        "definition": "количество из двух предметов",
        "action": "показывает пару",
        "image": "два ставит рядом две точки",
    },
    {
        "uri": "/m/basic/joy",
        "category": "emotion",
        "aliases": {"ru": ["радость", "радостно"], "en": ["joy", "happy"]},
        "label": "радость",
        "definition": "приятное чувство, когда что-то хорошо или интересно",
        "action": "делает ответ теплым и живым",
        "image": "радость улыбается внутри мысли",
    },
    {
        "uri": "/m/basic/sadness",
        "category": "emotion",
        "aliases": {"ru": ["грусть", "грустно", "печаль"], "en": ["sadness", "sad"]},
        "label": "грусть",
        "definition": "тихое чувство, когда чего-то не хватает или что-то огорчает",
        "action": "просит внимания и мягкого ответа",
        "image": "грусть делает мысль тише",
    },
]

BASIC_RELATIONS = [
    ("/m/basic/word", "Builds", "/m/basic/sentence"),
    ("/m/basic/sentence", "Carries", "/m/basic/meaning"),
    ("/m/basic/question", "Seeks", "/m/basic/answer"),
    ("/m/basic/answer", "Uses", "/m/basic/meaning"),
    ("/m/basic/learning", "Builds", "/m/basic/meaning"),
    ("/m/basic/child", "LearnsBy", "/m/basic/question"),
    ("/m/basic/imagination", "Creates", "/m/basic/story"),
    ("/m/basic/imagination", "Combines", "/m/basic/meaning"),
    ("/m/basic/thought", "Searches", "/m/basic/meaning"),
    ("/m/basic/sun", "Gives", "/m/basic/light"),
    ("/m/basic/sun", "SeenIn", "/m/basic/sky"),
    ("/m/basic/light", "Shows", "/m/basic/color"),
    ("/m/basic/water", "Helps", "/m/basic/tree"),
    ("/m/basic/earth", "Holds", "/m/basic/tree"),
    ("/m/basic/fire", "Gives", "/m/basic/light"),
    ("/m/basic/home", "Protects", "/m/basic/person"),
    ("/m/basic/apple", "IsA", "/m/basic/food"),
    ("/m/basic/apple", "HasColorExample", "/m/basic/red"),
    ("/m/basic/blue", "RelatedTo", "/m/basic/sky"),
    ("/m/basic/green", "RelatedTo", "/m/basic/tree"),
    ("/m/basic/number", "StartsWith", "/m/basic/one"),
    ("/m/basic/one", "NextCanBe", "/m/basic/two"),
    ("/m/basic/person", "CanFeel", "/m/basic/joy"),
    ("/m/basic/person", "CanFeel", "/m/basic/sadness"),
]

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

@dataclass(frozen=True)
class SeedReport:
    aliases: int = 0
    edges: int = 0
    responses: int = 0
    alphabets: int = 0
    common_words: int = 0
    basic_concepts: int = 0
    changed: bool = False

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "aliases": self.aliases,
            "edges": self.edges,
            "responses": self.responses,
            "alphabets": self.alphabets,
            "common_words": self.common_words,
            "basic_concepts": self.basic_concepts,
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
    basic_concepts = _load_basic_concepts(checkpoint)
    checkpoint.metadata["builtin_seed_version"] = SEED_VERSION
    checkpoint.metadata["builtin_seed_loaded"] = True
    return SeedReport(
        aliases=aliases,
        edges=edges,
        responses=responses,
        alphabets=alphabets,
        common_words=common_words,
        basic_concepts=basic_concepts,
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
    return 0


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


def _load_basic_concepts(checkpoint: Checkpoint) -> int:
    total = 0
    raw_definitions = checkpoint.metadata.get("concept_definitions", {})
    definitions = dict(raw_definitions) if isinstance(raw_definitions, dict) else {}
    basic_concepts: dict[str, dict[str, str]] = {}
    for item in BASIC_CONCEPTS:
        uri = str(item["uri"])
        info = {
            "label": str(item["label"]),
            "definition": str(item["definition"]),
            "action": str(item["action"]),
            "image": str(item["image"]),
            "category": str(item["category"]),
        }
        definitions[uri] = info
        basic_concepts[uri] = info
        category_uri = f"/m/basic/category/{item['category']}"
        checkpoint.add_custom_edge(uri, category_uri, relation="IsA", weight=1.0)
        total += 1
        aliases = item.get("aliases", {})
        if not isinstance(aliases, dict):
            continue
        for lang, words in aliases.items():
            if not isinstance(words, list) or not words:
                continue
            word_uri = f"/c/{lang}/{_uri_word(str(words[0]))}"
            checkpoint.add_custom_edge(word_uri, uri, relation="Means", weight=1.7)
            checkpoint.add_custom_edge(uri, word_uri, relation="HasWord", weight=0.6)
            definitions[word_uri] = info
            for word in words:
                clean = str(word).lower()
                if checkpoint.aliases.get(clean) != word_uri:
                    checkpoint.aliases[clean] = word_uri
                    total += 1
    for start, relation, end in BASIC_RELATIONS:
        checkpoint.add_custom_edge(start, end, relation=relation, weight=1.2)
        checkpoint.reinforce_edge(start, relation, end, amount=0.2)
        total += 1
    checkpoint.metadata["basic_concepts"] = basic_concepts
    checkpoint.metadata["concept_definitions"] = definitions
    return total


def _uri_word(value: str) -> str:
    return value.strip().lower().replace(" ", "_")
