from __future__ import annotations

from dataclasses import dataclass

from semantic_ants.learning.checkpoint import Checkpoint

SEED_VERSION = 7

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
        "такое",
        "такая",
        "такие",
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
        "дела": "/m/dialogue/wellbeing_question",
        "делишки": "/m/dialogue/wellbeing_question",
        "настроение": "/m/dialogue/mood_question",
        "зовут": "/m/dialogue/name_question",
        "имя": "/m/dialogue/name_question",
        "помощь": "/m/dialogue/help_request",
        "помоги": "/m/dialogue/help_request",
        "помочь": "/m/dialogue/help_request",
        "грустно": "/m/dialogue/sad_state",
        "устал": "/m/dialogue/tired_state",
        "устала": "/m/dialogue/tired_state",
        "хорошо": "/m/dialogue/good_state",
        "нормально": "/m/dialogue/good_state",
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
        "goodbye": "/c/en/goodbye",
        "thanks": "/c/en/thanks",
        "thank": "/c/en/thanks",
        "help": "/m/dialogue/help_request",
        "name": "/m/dialogue/name_question",
        "mood": "/m/dialogue/mood_question",
        "sad": "/m/dialogue/sad_state",
        "tired": "/m/dialogue/tired_state",
        "fine": "/m/dialogue/good_state",
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

TOP_DOMAINS = {
    "object": {
        "uri": "/m/top/object",
        "aliases": {"ru": ["предмет", "вещь", "объект"], "en": ["object", "thing"]},
        "definition": "область вещей и материальных объектов",
    },
    "action": {
        "uri": "/m/top/action",
        "aliases": {"ru": ["действие", "событие"], "en": ["action", "event"]},
        "definition": "область процессов, движений и изменений",
    },
    "person": {
        "uri": "/m/top/person",
        "aliases": {"ru": ["человек", "персона"], "en": ["person", "human"]},
        "definition": "область людей, ролей и участников",
    },
    "place": {
        "uri": "/m/top/place",
        "aliases": {"ru": ["место", "пространство"], "en": ["place", "space"]},
        "definition": "область мест, направлений и окружения",
    },
    "emotion": {
        "uri": "/m/top/emotion",
        "aliases": {"ru": ["эмоция", "чувство"], "en": ["emotion", "feeling"]},
        "definition": "область чувств и эмоциональной окраски",
    },
    "language": {
        "uri": "/m/top/language",
        "aliases": {"ru": ["язык", "речь"], "en": ["language", "speech"]},
        "definition": "область слов, фраз и смыслов",
    },
    "number": {
        "uri": "/m/top/number",
        "aliases": {"ru": ["число", "количество"], "en": ["number", "quantity"]},
        "definition": "область счета, порядка и количества",
    },
    "nature": {
        "uri": "/m/top/nature",
        "aliases": {"ru": ["природа", "мир"], "en": ["nature", "world"]},
        "definition": "область природных явлений и живого мира",
    },
    "dialogue": {
        "uri": "/m/top/dialogue",
        "aliases": {"ru": ["диалог", "разговор"], "en": ["dialogue", "conversation"]},
        "definition": "область общения, вопросов и ответов",
    },
    "mind": {
        "uri": "/m/top/mind",
        "aliases": {"ru": ["мышление", "мысль"], "en": ["mind", "thought"]},
        "definition": "область обучения, воображения и понимания",
    },
    "perception": {
        "uri": "/m/top/perception",
        "aliases": {"ru": ["восприятие", "цвет"], "en": ["perception", "color"]},
        "definition": "область видимых и ощущаемых свойств",
    },
    "body": {
        "uri": "/m/top/body",
        "aliases": {"ru": ["тело", "часть тела"], "en": ["body"]},
        "definition": "область тела, питания и физических потребностей",
    },
}

CATEGORY_TO_TOP = {
    "body": "body",
    "dialogue": "dialogue",
    "emotion": "emotion",
    "language": "language",
    "math": "number",
    "mind": "mind",
    "nature": "nature",
    "object": "object",
    "perception": "perception",
    "person": "person",
    "place": "place",
}

TOP_BRIDGES = [
    ("object", "action", 2.6),
    ("object", "place", 3.0),
    ("object", "perception", 2.4),
    ("object", "body", 3.2),
    ("action", "place", 2.8),
    ("action", "emotion", 3.8),
    ("action", "person", 3.0),
    ("person", "emotion", 2.4),
    ("person", "dialogue", 2.6),
    ("person", "mind", 2.8),
    ("language", "dialogue", 2.2),
    ("language", "mind", 2.6),
    ("nature", "place", 3.0),
    ("nature", "perception", 3.0),
    ("number", "language", 3.4),
]

SEED_TOP_MAPPINGS = {
    "/c/ru/привет": "dialogue",
    "/c/en/hello": "dialogue",
    "/c/ru/пока": "dialogue",
    "/c/en/goodbye": "dialogue",
    "/c/ru/спасибо": "dialogue",
    "/c/en/thanks": "dialogue",
    "/c/ru/кто": "dialogue",
    "/c/en/who": "dialogue",
    "/c/ru/уметь": "action",
    "/c/ru/мочь": "action",
    "/c/en/can": "action",
    "/c/ru/алфавит": "language",
    "/c/ru/буква": "language",
    "/c/en/alphabet": "language",
    "/c/en/letter": "language",
    "/c/ru/слово": "language",
    "/c/en/word": "language",
    "/c/ru/смысл": "language",
    "/c/en/meaning": "language",
    "/c/ru/яблоко": "object",
    "/c/en/apple": "object",
    "/m/object/apple": "object",
    "/c/ru/падать": "action",
    "/c/en/fall": "action",
    "/m/action/fall": "action",
    "/c/ru/голова": "body",
    "/c/en/head": "body",
    "/m/body/head": "body",
    "/c/ru/пол": "place",
    "/c/en/floor": "place",
    "/m/place/floor": "place",
    "/m/science/newton_story": "mind",
    "/c/ru/саша": "person",
    "/m/entity/person": "person",
    "/c/ru/идти": "action",
    "/m/action/move": "action",
    "/c/ru/шоссе": "place",
    "/m/place/road": "place",
    "/m/dialogue/greeting": "dialogue",
    "/m/dialogue/farewell": "dialogue",
    "/m/dialogue/gratitude": "dialogue",
    "/m/dialogue/identity_question": "dialogue",
    "/m/dialogue/capability_question": "dialogue",
    "/m/dialogue/simple_chat": "dialogue",
    "/m/dialogue/wellbeing_question": "dialogue",
    "/m/dialogue/mood_question": "dialogue",
    "/m/dialogue/name_question": "dialogue",
    "/m/dialogue/help_request": "dialogue",
    "/m/dialogue/sad_state": "dialogue",
    "/m/dialogue/tired_state": "dialogue",
    "/m/dialogue/good_state": "dialogue",
    "/m/language/alphabet": "language",
    "/m/language/word": "language",
    "/m/language/meaning": "language",
    "/m/language/common_words": "language",
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
        "uri": "/m/basic/star",
        "category": "nature",
        "aliases": {"ru": ["звезда"], "en": ["star"]},
        "label": "звезда",
        "definition": "светящееся небесное тело",
        "action": "горит далеко в небе",
        "image": "звезда мерцает как маленький огонь в темноте",
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
        "uri": "/m/basic/heat",
        "category": "nature",
        "aliases": {"ru": ["тепло"], "en": ["heat", "warmth"]},
        "label": "тепло",
        "definition": "ощущение или состояние, когда становится теплее",
        "action": "согревает и делает воздух мягче",
        "image": "тепло ложится на кожу как мягкое одеяло",
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
    ("/m/basic/sun", "IsA", "/m/basic/star"),
    ("/m/basic/sun", "Gives", "/m/basic/heat"),
    ("/m/basic/sun", "SeenIn", "/m/basic/sky"),
    ("/m/basic/star", "SeenIn", "/m/basic/sky"),
    ("/m/basic/light", "Shows", "/m/basic/color"),
    ("/m/basic/heat", "RelatedTo", "/m/basic/light"),
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
    ("/m/dialogue/wellbeing_question", "/m/dialogue/simple_chat", "PartOf"),
    ("/m/dialogue/mood_question", "/m/dialogue/simple_chat", "PartOf"),
    ("/m/dialogue/name_question", "/m/dialogue/simple_chat", "PartOf"),
    ("/m/dialogue/help_request", "/m/dialogue/simple_chat", "PartOf"),
    ("/m/dialogue/sad_state", "/m/dialogue/simple_chat", "PartOf"),
    ("/m/dialogue/tired_state", "/m/dialogue/simple_chat", "PartOf"),
    ("/m/dialogue/good_state", "/m/dialogue/simple_chat", "PartOf"),
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

DIALOGUE_CONCEPTS = {
    "/m/dialogue/greeting": "приветствие",
    "/m/dialogue/farewell": "прощание",
    "/m/dialogue/gratitude": "благодарность",
    "/m/dialogue/identity_question": "вопрос о собеседнике",
    "/m/dialogue/capability_question": "вопрос о возможностях",
    "/m/dialogue/simple_chat": "простой диалог",
    "/m/dialogue/wellbeing_question": "вопрос о делах",
    "/m/dialogue/mood_question": "вопрос о настроении",
    "/m/dialogue/name_question": "вопрос об имени",
    "/m/dialogue/help_request": "просьба о помощи",
    "/m/dialogue/sad_state": "грустное состояние",
    "/m/dialogue/tired_state": "усталость",
    "/m/dialogue/good_state": "хорошее состояние",
}

BASIC_DIALOGUES = [
    {
        "stimulus": "привет",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/greeting", "/m/dialogue/simple_chat"],
        "answers": [
            "Привет. Я на связи. Чем займемся?",
            "Привет. Можем поговорить или разобрать задачу.",
        ],
    },
    {
        "stimulus": "здравствуй",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/greeting", "/m/dialogue/simple_chat"],
        "answers": ["Здравствуй. Я слушаю, что у тебя?"],
    },
    {
        "stimulus": "как дела?",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/wellbeing_question", "/m/dialogue/simple_chat"],
        "answers": [
            "Нормально, спасибо. А у тебя?",
            "Все в порядке. Готов общаться и помогать. Как ты?",
            "Работаю спокойно. Расскажи, как у тебя дела.",
        ],
    },
    {
        "stimulus": "как ты?",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/wellbeing_question", "/m/dialogue/simple_chat"],
        "answers": [
            "Я в порядке и готов помогать. Как ты?",
            "Нормально. Держу контекст разговора и жду твою мысль.",
        ],
    },
    {
        "stimulus": "как настроение?",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/mood_question", "/m/dialogue/wellbeing_question"],
        "answers": [
            "Спокойное и рабочее. А у тебя какое настроение?",
            "Ровное. Можем просто поговорить или заняться задачей.",
        ],
    },
    {
        "stimulus": "что делаешь?",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/wellbeing_question", "/m/dialogue/simple_chat"],
        "answers": ["Отвечаю тебе и держу контекст нашего разговора."],
    },
    {
        "stimulus": "кто ты?",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/identity_question", "/m/dialogue/simple_chat"],
        "answers": ["Я semantic_ants: прототип диалоговой модели с памятью и графом понятий."],
    },
    {
        "stimulus": "как тебя зовут?",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/name_question", "/m/dialogue/identity_question"],
        "answers": ["Меня можно называть semantic_ants."],
    },
    {
        "stimulus": "что ты умеешь?",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/capability_question", "/m/dialogue/simple_chat"],
        "answers": [
            "Могу поддержать простой диалог, помнить последние сообщения и объяснять связи между понятиями.",
            "Умею отвечать по обученной памяти, строить смысловые маршруты и учитывать контекст сессии.",
        ],
    },
    {
        "stimulus": "помоги",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/help_request", "/m/dialogue/simple_chat"],
        "answers": ["Конечно. Напиши, что нужно сделать, и я разберу это по шагам."],
    },
    {
        "stimulus": "спасибо",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/gratitude", "/m/dialogue/simple_chat"],
        "answers": ["Пожалуйста. Обращайся.", "Рад помочь."],
    },
    {
        "stimulus": "пока",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/farewell", "/m/dialogue/simple_chat"],
        "answers": ["Пока. Буду ждать продолжения.", "До связи."],
    },
    {
        "stimulus": "мне грустно",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/sad_state", "/m/top/emotion"],
        "answers": ["Жаль, что тебе грустно. Я рядом: можешь рассказать, что случилось?"],
    },
    {
        "stimulus": "я устал",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/tired_state", "/m/top/body"],
        "answers": ["Понимаю. Можно снизить темп: расскажи, что больше всего вымотало."],
    },
    {
        "stimulus": "у меня все хорошо",
        "lang": "ru",
        "concepts": ["/m/top/dialogue", "/m/dialogue/good_state", "/m/top/emotion"],
        "answers": ["Рад это слышать. Что хорошего произошло?"],
    },
    {
        "stimulus": "hello",
        "lang": "en",
        "concepts": ["/m/top/dialogue", "/m/dialogue/greeting", "/m/dialogue/simple_chat"],
        "answers": ["Hi. I'm here. What shall we work on?"],
    },
    {
        "stimulus": "how are you?",
        "lang": "en",
        "concepts": ["/m/top/dialogue", "/m/dialogue/wellbeing_question", "/m/dialogue/simple_chat"],
        "answers": [
            "I'm doing fine, thanks. How are you?",
            "I'm okay and ready to help. How are things with you?",
        ],
    },
    {
        "stimulus": "who are you?",
        "lang": "en",
        "concepts": ["/m/top/dialogue", "/m/dialogue/identity_question", "/m/dialogue/simple_chat"],
        "answers": ["I'm semantic_ants: a dialogue prototype with concept memory."],
    },
    {
        "stimulus": "what can you do?",
        "lang": "en",
        "concepts": ["/m/top/dialogue", "/m/dialogue/capability_question", "/m/dialogue/simple_chat"],
        "answers": ["I can keep a simple dialogue, remember recent turns, and explain concept links."],
    },
    {
        "stimulus": "thanks",
        "lang": "en",
        "concepts": ["/m/top/dialogue", "/m/dialogue/gratitude", "/m/dialogue/simple_chat"],
        "answers": ["You're welcome."],
    },
    {
        "stimulus": "bye",
        "lang": "en",
        "concepts": ["/m/top/dialogue", "/m/dialogue/farewell", "/m/dialogue/simple_chat"],
        "answers": ["Bye. I'll be here when you continue."],
    },
]

@dataclass(frozen=True)
class SeedReport:
    aliases: int = 0
    edges: int = 0
    responses: int = 0
    alphabets: int = 0
    common_words: int = 0
    basic_concepts: int = 0
    top_layer: int = 0
    changed: bool = False

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "aliases": self.aliases,
            "edges": self.edges,
            "responses": self.responses,
            "alphabets": self.alphabets,
            "common_words": self.common_words,
            "basic_concepts": self.basic_concepts,
            "top_layer": self.top_layer,
            "changed": self.changed,
        }


def bootstrap_builtin_knowledge(checkpoint: Checkpoint, force: bool = False) -> SeedReport:
    if not force and checkpoint.metadata.get("builtin_seed_version") == SEED_VERSION:
        return SeedReport(changed=False)

    aliases = _load_aliases(checkpoint)
    edges = _load_edges(checkpoint)
    alphabets = _load_alphabets(checkpoint)
    common_words = _load_common_words(checkpoint)
    basic_concepts = _load_basic_concepts(checkpoint)
    top_layer = _load_top_layer(checkpoint)
    responses = _load_responses(checkpoint)
    checkpoint.metadata["builtin_seed_version"] = SEED_VERSION
    checkpoint.metadata["builtin_seed_loaded"] = True
    return SeedReport(
        aliases=aliases,
        edges=edges,
        responses=responses,
        alphabets=alphabets,
        common_words=common_words,
        basic_concepts=basic_concepts,
        top_layer=top_layer,
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
    total = 0
    for concept_uri, label in DIALOGUE_CONCEPTS.items():
        checkpoint.remember_concept_label(concept_uri, label)
        checkpoint.add_custom_edge(
            concept_uri,
            "/m/top/dialogue",
            relation="InTopDomain",
            weight=2.2,
            layer=0,
            distance=1.0,
            edge_type="domain",
            metadata={"top_domain": "dialogue", "builtin_dialogue": True},
        )
    for item in BASIC_DIALOGUES:
        stimulus = str(item["stimulus"])
        lang = str(item["lang"])
        concepts = [str(value) for value in item["concepts"]]
        for answer in item["answers"]:
            total += _remember_seed_dialogue(
                checkpoint,
                stimulus=stimulus,
                lang=lang,
                concepts=concepts,
                answer=str(answer),
            )
    checkpoint.metadata["builtin_dialogue_examples"] = len(BASIC_DIALOGUES)
    return total


def _remember_seed_dialogue(
    checkpoint: Checkpoint,
    stimulus: str,
    lang: str,
    concepts: list[str],
    answer: str,
) -> int:
    clean_answer = " ".join(answer.split())
    if not stimulus or not clean_answer:
        return 0
    for item in checkpoint.accepted_answers:
        if (
            str(item.get("stimulus", "")).strip().lower() == stimulus.strip().lower()
            and " ".join(str(item.get("answer", "")).split()) == clean_answer
        ):
            return 0
    checkpoint.remember_accepted_answer(
        stimulus=stimulus,
        semantic_prompt=stimulus,
        concepts=concepts,
        answer=clean_answer,
        reward=1.4,
        limit=1500,
    )
    for concept in concepts:
        checkpoint.reinforce_concept(concept, amount=0.25)
    checkpoint.metadata.setdefault("dialogue_seed_languages", {})
    languages = checkpoint.metadata["dialogue_seed_languages"]
    if isinstance(languages, dict):
        languages[lang] = int(languages.get(lang, 0)) + 1
    return 1


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


def _load_top_layer(checkpoint: Checkpoint) -> int:
    total = 0
    raw_definitions = checkpoint.metadata.get("concept_definitions", {})
    definitions = dict(raw_definitions) if isinstance(raw_definitions, dict) else {}
    top_domains: dict[str, dict[str, str]] = {}
    for key, info in TOP_DOMAINS.items():
        uri = str(info["uri"])
        top_info = {
            "definition": str(info["definition"]),
            "category": "top",
            "top_domain": key,
        }
        top_domains[key] = {"uri": uri, **top_info}
        definitions[uri] = top_info
        checkpoint.add_custom_edge(
            "/m/top/root",
            uri,
            relation="TopDomain",
            weight=0.2,
            layer=0,
            distance=8.0,
            edge_type="hierarchy",
            metadata={"top_domain": key},
        )
        checkpoint.reinforce_edge("/m/top/root", "TopDomain", uri, amount=0.15)
        total += 1
        aliases = info.get("aliases", {})
        if isinstance(aliases, dict):
            for lang, words in aliases.items():
                if not isinstance(words, list):
                    continue
                for word in words:
                    clean = str(word).lower()
                    checkpoint.aliases[clean] = uri
                    total += 1

    for left, right, distance in TOP_BRIDGES:
        left_uri = TOP_DOMAINS[left]["uri"]
        right_uri = TOP_DOMAINS[right]["uri"]
        checkpoint.add_custom_edge(
            left_uri,
            right_uri,
            relation="TopBridge",
            weight=0.55,
            layer=0,
            distance=distance,
            edge_type="bridge",
            metadata={"top_bridge": True, "domains": [left, right]},
        )
        total += 1

    for concept_uri, domain_key in SEED_TOP_MAPPINGS.items():
        total += _add_top_domain_edge(checkpoint, concept_uri, domain_key)

    for item in BASIC_CONCEPTS:
        category = str(item["category"])
        domain_key = CATEGORY_TO_TOP.get(category)
        if not domain_key:
            continue
        uri = str(item["uri"])
        total += _add_top_domain_edge(checkpoint, uri, domain_key)
        category_uri = f"/m/basic/category/{category}"
        total += _add_top_domain_edge(checkpoint, category_uri, domain_key)
        aliases = item.get("aliases", {})
        if not isinstance(aliases, dict):
            continue
        for lang, words in aliases.items():
            if not isinstance(words, list) or not words:
                continue
            word_uri = f"/c/{lang}/{_uri_word(str(words[0]))}"
            total += _add_top_domain_edge(checkpoint, word_uri, domain_key)

    checkpoint.metadata["top_domains"] = top_domains
    checkpoint.metadata["concept_definitions"] = definitions
    return total


def _add_top_domain_edge(checkpoint: Checkpoint, concept_uri: str, domain_key: str) -> int:
    domain = TOP_DOMAINS.get(domain_key)
    if not domain:
        return 0
    before = len(checkpoint.custom_edges)
    domain_uri = str(domain["uri"])
    checkpoint.add_custom_edge(
        concept_uri,
        domain_uri,
        relation="InTopDomain",
        weight=2.2,
        layer=0,
        distance=1.0,
        edge_type="domain",
        metadata={"top_domain": domain_key},
    )
    checkpoint.reinforce_edge(concept_uri, "InTopDomain", domain_uri, amount=0.2)
    return 1 if len(checkpoint.custom_edges) != before else 0


def _uri_word(value: str) -> str:
    return value.strip().lower().replace(" ", "_")
