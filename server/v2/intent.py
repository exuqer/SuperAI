"""Communicative-intent classification performed before scene parsing."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence

from .language.models import DialogueAct, DialogueActType
from .language.utterance_parser import DialogueActParser


GREETING_WORDS = {"привет", "здравствуй", "здравствуйте", "добрый день", "доброе утро", "добрый вечер"}
SMALL_TALK_PATTERNS = {
    "как дела": "status_query",
    "как ты": "status_query",
    "как поживаешь": "status_query",
    "что нового": "news_query",
}
QUESTION_WORDS = {"кто", "кого", "кому", "что", "где", "куда", "откуда", "когда", "как", "почему", "зачем", "чем", "сколько"}


class IntentClassifier:
    """Small deterministic classifier for the supported Russian dialogue intents."""

    def __init__(self) -> None:
        self.dialogue_acts = DialogueActParser()

    def classify(
        self,
        text: str,
        *,
        dialogue_acts: Optional[Sequence[DialogueAct]] = None,
    ) -> Dict[str, Any]:
        source = str(text or "").strip()
        normalized = re.sub(r"\s+", " ", source.casefold()).strip(" .!?,;:")
        resolved_acts = list(dialogue_acts) if dialogue_acts is not None else (
            self.dialogue_acts.parse(source)
        )
        act_types = {act.act_type for act in resolved_acts}
        greeting = next((item for item in GREETING_WORDS if re.search(rf"(?:^|\s){re.escape(item)}(?:$|\s|[!?,.])", source.casefold())), None)
        small_talk_surface: Optional[str] = None
        small_talk_type: Optional[str] = None
        for pattern, kind in SMALL_TALK_PATTERNS.items():
            match = re.search(rf"(?:^|\s)({re.escape(pattern)})(?:$|\s|[!?.,])", normalized)
            if match:
                small_talk_surface = source[match.start(1):match.end(1)]
                small_talk_type = kind
                break
        location = self._location(source)
        words = re.findall(r"[\w-]+", normalized, flags=re.UNICODE)
        has_question_operator = any(word in QUESTION_WORDS for word in words)
        has_scene_question = (
            has_question_operator
            or DialogueActType.QUESTION in act_types
        )
        has_question_mark = source.rstrip().endswith("?")
        if greeting and small_talk_type:
            intent = "GREETING_WITH_SMALL_TALK"
        elif small_talk_type:
            intent = "SMALL_TALK"
        elif has_scene_question or has_question_mark:
            intent = "SCENE_QUESTION"
        elif greeting:
            intent = "GREETING"
        else:
            if len(words) == 1:
                intent = "STRUCTURAL_PROBE" if len(words[0]) <= 3 else "LEXICAL_PROBE"
            elif (
                DialogueActType.COMMAND in act_types
                or any(word in {"сделай", "покажи", "найди", "добавь", "удали", "запусти"} for word in words)
            ):
                intent = "COMMAND"
            elif words:
                intent = "SCENE_STATEMENT"
            else:
                intent = "UNKNOWN"
        result: Dict[str, Any] = {
            "intent": intent,
            "source_text": source,
            "dialogue_acts": [act.as_dict() for act in resolved_acts],
        }
        if intent == "SCENE_QUESTION":
            result["question_kind"] = (
                "role" if has_question_operator else "polar"
            )
        if greeting:
            result["greeting"] = {"surface": greeting.capitalize() if greeting == "привет" else greeting}
        if small_talk_type:
            result["small_talk"] = {"type": small_talk_type, "surface": small_talk_surface or source}
        if location:
            result["context"] = {"location": location}
        return result

    @staticmethod
    def _location(text: str) -> Optional[Dict[str, str]]:
        match = re.search(r"\b(на|в|во)\s+([А-Яа-яЁё\w-]+)", text, flags=re.IGNORECASE)
        if not match:
            return None
        surface = match.group(2)
        lemma = surface.casefold()
        endings = (("ке", "ок"), ("ке", "ка"), ("е", ""), ("у", ""), ("е", ""))
        for ending, replacement in endings:
            if lemma.endswith(ending) and len(lemma) > len(ending) + 2:
                lemma = lemma[:-len(ending)] + replacement
                break
        return {"surface": surface, "lemma": lemma, "preposition": match.group(1).casefold()}
