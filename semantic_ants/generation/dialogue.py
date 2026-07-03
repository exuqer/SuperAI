from __future__ import annotations

from semantic_ants.knowledge import ALPHABETS, COMMON_WORDS
from semantic_ants.learning.checkpoint import Checkpoint


class DialogueResponder:
    """Правила простого диалога поверх смыслового графа."""

    def response_for(
        self,
        input_text: str,
        tokens: list[str],
        activated_concepts: list[dict[str, object]],
        checkpoint: Checkpoint,
    ) -> str | None:
        token_set = set(tokens)
        text = input_text.lower()
        if self._asks_alphabet(token_set, text):
            return self._alphabet_response(token_set, text, checkpoint)
        if self._asks_common_words(token_set, text):
            return self._common_words_response(token_set, text, checkpoint)
        if self._asks_meaning(token_set, text):
            return self._meaning_response(tokens, activated_concepts)
        if token_set & {"привет", "здравствуй", "здравствуйте", "hello", "hi"}:
            return "Привет. Я простой исследовательский чат на смысловом графе. Можешь задать вопрос."
        if token_set & {"спасибо", "благодарю", "thanks"}:
            return "Пожалуйста."
        if token_set & {"пока", "bye", "goodbye"}:
            return "Пока. Я сохраню обученный слой в checkpoint."
        if self._asks_identity(token_set, text):
            return "Я прототип semantic_ants: связываю слова со смыслами, строю маршруты и учусь на примерах."
        if self._asks_capability(token_set, text):
            return "Я могу отвечать на простые вопросы, показывать алфавиты, частые слова и объяснять найденные смысловые маршруты."
        if "?" in input_text or token_set & {"почему", "зачем", "где", "когда", "what", "why", "where", "when", "how"}:
            return self._question_response(activated_concepts)
        return None

    def _asks_alphabet(self, token_set: set[str], text: str) -> bool:
        return bool(token_set & {"алфавит", "буква", "буквы", "alphabet", "letter", "letters"}) or "alphabet" in text

    def _asks_common_words(self, token_set: set[str], text: str) -> bool:
        has_word = bool(token_set & {"слово", "слова", "word", "words"})
        has_common = bool(token_set & {"частый", "частые", "популярные", "используемые", "common", "frequent"})
        return has_word and has_common or "частые слова" in text or "common words" in text

    def _asks_meaning(self, token_set: set[str], text: str) -> bool:
        return "что такое" in text or "что значит" in text or "what is" in text or bool(
            token_set & {"смысл", "meaning"}
        )

    def _asks_identity(self, token_set: set[str], text: str) -> bool:
        return ("кто" in token_set and ("ты" in token_set or "вы" in token_set)) or "who are you" in text

    def _asks_capability(self, token_set: set[str], text: str) -> bool:
        return bool(token_set & {"умеешь", "можешь"}) or "what can you do" in text or "can you" in text

    def _alphabet_response(self, token_set: set[str], text: str, checkpoint: Checkpoint) -> str:
        lang = "en" if "english" in text or "англий" in text or "en" in token_set else "ru"
        if "рус" in text or "ru" in token_set:
            lang = "ru"
        alphabets = checkpoint.metadata.get("alphabets", ALPHABETS)
        letters = alphabets.get(lang, ALPHABETS[lang])
        label = "Русский" if lang == "ru" else "Английский"
        return f"{label} алфавит: {' '.join(letters)}."

    def _common_words_response(self, token_set: set[str], text: str, checkpoint: Checkpoint) -> str:
        lang = "en" if "english" in text or "англий" in text or "en" in token_set else "ru"
        if "рус" in text or "ru" in token_set:
            lang = "ru"
        words_map = checkpoint.metadata.get("common_words", COMMON_WORDS)
        words = words_map.get(lang, COMMON_WORDS[lang])[:15]
        label = "русских" if lang == "ru" else "английских"
        return f"Пример частых {label} слов: {', '.join(words)}."

    def _meaning_response(self, tokens: list[str], activated_concepts: list[dict[str, object]]) -> str:
        if not activated_concepts:
            return "Пока не вижу достаточно связей, чтобы объяснить смысл."
        target = _last_content_token(tokens)
        labels = ", ".join(str(item["label"]) for item in activated_concepts[:3])
        if target:
            return f"Для «{target}» ближайшие найденные смыслы: {labels}."
        return f"Ближайшие найденные смыслы: {labels}."

    def _question_response(self, activated_concepts: list[dict[str, object]]) -> str:
        if not activated_concepts:
            return "Я понял, что это вопрос, но пока не нашел надежных смысловых связей."
        labels = ", ".join(str(item["label"]) for item in activated_concepts[:3])
        return f"Я связываю вопрос с такими смыслами: {labels}. Мой ответ пока строится по простому графу, не как полноценная LLM."


def _last_content_token(tokens: list[str]) -> str | None:
    stop = {"что", "такое", "значит", "смысл", "what", "is", "meaning", "of"}
    for token in reversed(tokens):
        if token not in stop:
            return token
    return None
