from __future__ import annotations

import re
import unicodedata
from urllib.parse import quote

CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-'][0-9A-Za-zА-Яа-яЁё]+)?")
_TRANSLATION_TARGET_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "en",
        (
            "на английск",
            "по-английск",
            "to english",
            "in english",
            "english translation",
            "translate to english",
            "translate into english",
        ),
    ),
    (
        "ru",
        (
            "на русском",
            "по-русски",
            "to russian",
            "in russian",
            "russian translation",
            "translate to russian",
            "translate into russian",
        ),
    ),
)
_TRANSLATION_CUES = (
    "переведи",
    "перевод",
    "translate",
    "translation",
    "как будет",
    "как сказать",
    "how do you say",
    "how to say",
)


def normalize_text(text: str) -> str:
    """Приводит текст к стабильной форме для токенизации."""

    normalized = unicodedata.normalize("NFKC", text)
    return normalized.replace("’", "'").strip().lower()


def detect_language(text: str) -> str:
    """Возвращает `ru`, если есть кириллица, иначе `en`."""

    return "ru" if CYRILLIC_RE.search(text) else "en"


def tokenize(text: str) -> list[str]:
    """Простая токенизация слов без внешних зависимостей."""

    normalized = normalize_text(text)
    return [match.group(0).replace("-", "_") for match in TOKEN_RE.finditer(normalized)]


def detect_response_language(text: str, default: str | None = None) -> str | None:
    normalized = " ".join(normalize_text(text).split())
    if not normalized:
        return default if default in {"ru", "en"} else None
    explicit = _translation_target_language(normalized)
    if explicit:
        return explicit
    if any(cue in normalized for cue in _TRANSLATION_CUES):
        if default == "ru":
            return "en"
        if default == "en":
            return "ru"
    if default in {"ru", "en"}:
        return default
    return None


def text_to_concept_uri(text: str, lang: str | None = None) -> str:
    """Строит URI ConceptNet вида `/c/ru/яблоко` или `/c/en/apple`."""

    selected_lang = detect_language(text) if lang in (None, "auto") else lang
    phrase = "_".join(tokenize(text))
    if not phrase:
        raise ValueError("Нельзя построить ConceptNet URI из пустого текста")
    return f"/c/{selected_lang}/{phrase}"


def quote_concept_path(path: str) -> str:
    """Кодирует путь URI для HTTP-запроса, сохраняя разделители ConceptNet."""

    return quote(path, safe="/:")


def _translation_target_language(text: str) -> str | None:
    for lang, hints in _TRANSLATION_TARGET_HINTS:
        if any(hint in text for hint in hints):
            return lang
    return None
