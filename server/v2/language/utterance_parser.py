"""Deterministic communicative-act parsing for Russian dialogue."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence

from server.tokenizer import normalize_text, tokenize_hierarchical

from .models import (
    DialogueAct,
    DialogueActType,
    InterpretationStatus,
    ParsedToken,
    UtteranceEnvelope,
)


GREETING_LEMMAS = {
    "привет",
    "здравствуй",
    "здравствуйте",
    "доброе",
    "добрый",
}
QUESTION_LEMMAS = {
    "кто",
    "что",
    "какой",
    "\u043a\u043e\u0442\u043e\u0440\u044b\u0439",
    "чей",
    "где",
    "куда",
    "откуда",
    "когда",
    "как",
    "почему",
    "зачем",
    "сколько",
}
REQUEST_LEMMAS = {
    "подсказать",
    "подскажи",
    "пожалуйста",
    "просить",
    "попросить",
}
COMMAND_SURFACES = {
    "сделай",
    "покажи",
    "найди",
    "добавь",
    "удали",
    "запусти",
    "включи",
    "выключи",
    "открой",
    "закрой",
    "передай",
    "скажи",
    "проверь",
}
HYPOTHESIS_MARKERS = {
    "наверное",
    "вероятно",
    "возможно",
    "похоже",
    "кажется",
}
ASSUMPTION_MARKERS = {"допустим", "предположим", "предполагаю"}
PLAN_MARKERS = {"планирую", "собираюсь", "намерен", "намерена"}
DESIRE_MARKERS = {"хочу", "хотел", "хотела", "желаю"}
DENIAL_MARKERS = {"нет", "неверно", "неправильно"}
CONFIRMATION_MARKERS = {"да", "верно", "точно", "согласен", "согласна"}
CORRECTION_PHRASES = (
    ("я", "иметь", "в", "вид"),
    ("я", "имел", "в", "виду"),
    ("я", "имела", "в", "виду"),
    ("я", "спрашивать"),
    ("я", "спрашивал"),
    ("я", "спрашивала"),
    ("точнее",),
)
SPEECH_LEMMAS = {
    "сказать",
    "говорить",
    "сообщить",
    "ответить",
    "написать",
    "утверждать",
}
SMALL_TALK_PATTERNS = (
    ("как", "дело"),
    ("как", "ты"),
    ("что", "новый"),
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str, *parts: object) -> str:
    key = "|".join(str(part) for part in parts)
    return f"{prefix}-{uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:20]}"


class DialogueActParser:
    """Recognize multiple, independently spanned acts in one utterance."""

    @staticmethod
    def _token_data(
        text: str,
        tokens: Optional[Sequence[ParsedToken]] = None,
    ) -> tuple[List[str], List[str], List[int]]:
        if tokens is not None:
            surfaces = [token.normalized.casefold() for token in tokens]
            lemmas = [token.lemma.casefold() for token in tokens]
            sentence_indices = [
                token.features.get("sentence_index", 0)
                if isinstance(token.features, dict)
                else 0
                for token in tokens
            ]
            if len(set(sentence_indices)) == 1:
                hierarchy = tokenize_hierarchical(text)
                if len(hierarchy.all_tokens) == len(tokens):
                    sentence_indices = [
                        token.sentence_index for token in hierarchy.all_tokens
                    ]
            return surfaces, lemmas, sentence_indices
        hierarchy = tokenize_hierarchical(text)
        surfaces = [token.normalized.casefold() for token in hierarchy.all_tokens]
        return (
            surfaces,
            list(surfaces),
            [token.sentence_index for token in hierarchy.all_tokens],
        )

    @staticmethod
    def _contains_sequence(values: Sequence[str], sequence: Sequence[str]) -> Optional[int]:
        if not sequence or len(sequence) > len(values):
            return None
        for index in range(len(values) - len(sequence) + 1):
            if tuple(values[index:index + len(sequence)]) == tuple(sequence):
                return index
        return None

    def parse(
        self,
        text: str,
        *,
        utterance_id: str = "",
        tokens: Optional[Sequence[ParsedToken]] = None,
    ) -> List[DialogueAct]:
        source = str(text or "")
        surfaces, lemmas, sentence_indices = self._token_data(source, tokens)
        if not surfaces:
            return []
        utterance_id = utterance_id or _stable_id("utterance", normalize_text(source))
        candidates: List[DialogueAct] = []

        def add(
            act_type: DialogueActType,
            start: int,
            end: int,
            confidence: float,
            evidence_type: str,
            *,
            target_act_id: Optional[str] = None,
            alternatives: Optional[List[dict]] = None,
        ) -> DialogueAct:
            start = max(0, min(start, len(surfaces) - 1))
            end = max(start, min(end, len(surfaces) - 1))
            existing = next(
                (
                    item for item in candidates
                    if item.act_type == act_type
                    and item.token_start == start
                    and item.token_end == end
                ),
                None,
            )
            if existing:
                return existing
            act = DialogueAct(
                id=_stable_id("act", utterance_id, act_type.value, start, end),
                utterance_id=utterance_id,
                act_type=act_type,
                token_start=start,
                token_end=end,
                target_act_id=target_act_id,
                confidence=confidence,
                evidence=[{
                    "origin": "dialogue_act_parser",
                    "type": evidence_type,
                    "token_start": start,
                    "token_end": end,
                }],
                alternatives=list(alternatives or []),
            )
            candidates.append(act)
            return act

        sentence_spans: List[tuple[int, int, int]] = []
        for sentence_index in sorted(set(sentence_indices)):
            positions = [
                index for index, value in enumerate(sentence_indices)
                if value == sentence_index
            ]
            if positions:
                sentence_spans.append((sentence_index, positions[0], positions[-1]))

        greeting_indices = [
            index for index, value in enumerate(surfaces)
            if value in GREETING_LEMMAS or lemmas[index] in GREETING_LEMMAS
        ]
        if greeting_indices:
            start = greeting_indices[0]
            end = start
            if surfaces[start] in {"добрый", "доброе"} and start + 1 < len(surfaces):
                end = start + 1
            add(DialogueActType.GREETING, start, end, 0.99, "greeting_lexeme")

        for pattern in SMALL_TALK_PATTERNS:
            index = self._contains_sequence(lemmas, pattern)
            if index is not None:
                add(
                    DialogueActType.SMALL_TALK,
                    index,
                    index + len(pattern) - 1,
                    0.94,
                    "small_talk_construction",
                )

        if surfaces[0] in DENIAL_MARKERS:
            add(DialogueActType.DENIAL, 0, 0, 0.99, "denial_marker")
        elif surfaces[0] in CONFIRMATION_MARKERS:
            add(
                DialogueActType.CONFIRMATION,
                0,
                0,
                0.96,
                "confirmation_marker",
            )

        correction_start: Optional[int] = None
        for pattern in CORRECTION_PHRASES:
            correction_start = self._contains_sequence(lemmas, pattern)
            if correction_start is None:
                correction_start = self._contains_sequence(surfaces, pattern)
            if correction_start is not None:
                break
        contrastive_replacement = (
            "не" in surfaces and "а" in surfaces
            and surfaces.index("не") < surfaces.index("а")
        )
        if correction_start is not None or (
            surfaces[0] in DENIAL_MARKERS and len(surfaces) > 1
        ) or contrastive_replacement:
            start = correction_start if correction_start is not None else 0
            add(
                DialogueActType.CORRECTION,
                start,
                len(surfaces) - 1,
                0.93 if correction_start is not None else 0.82,
                "correction_construction",
            )

        condition_index = next(
            (
                index for index, value in enumerate(surfaces)
                if value in {"если", "когда"}
            ),
            None,
        )
        command_indices = [
            index for index, value in enumerate(surfaces)
            if value in COMMAND_SURFACES
        ]
        if condition_index is not None:
            command_after = next(
                (index for index in command_indices if index > condition_index),
                None,
            )
            condition_end = (
                command_after - 1 if command_after is not None
                else len(surfaces) - 1
            )
            counterfactual = (
                surfaces[condition_index] == "если"
                and condition_index + 1 < len(surfaces)
                and surfaces[condition_index + 1] == "бы"
            )
            add(
                DialogueActType.COUNTERFACTUAL
                if counterfactual else DialogueActType.CONDITION,
                condition_index,
                condition_end,
                0.98,
                "conditional_subordinator",
            )

        request_indices = [
            index for index, value in enumerate(surfaces)
            if value in REQUEST_LEMMAS or lemmas[index] in REQUEST_LEMMAS
        ]
        polite_request = (
            any(value in {"можешь", "могли"} for value in surfaces)
            and "ли" in surfaces
        )
        if request_indices or polite_request:
            start = request_indices[0] if request_indices else 0
            question_index = next(
                (
                    index for index, value in enumerate(lemmas)
                    if value in QUESTION_LEMMAS and index > start
                ),
                None,
            )
            add(
                DialogueActType.REQUEST,
                start,
                question_index - 1 if question_index is not None else len(surfaces) - 1,
                0.93,
                "request_construction",
            )

        for index in command_indices:
            if request_indices and index == request_indices[0]:
                continue
            start = index
            if condition_index is not None and index <= condition_index:
                continue
            add(
                DialogueActType.COMMAND,
                start,
                len(surfaces) - 1,
                0.95,
                "imperative_surface",
            )

        hypothesis_index = next(
            (
                index for index, value in enumerate(surfaces)
                if value in HYPOTHESIS_MARKERS
            ),
            None,
        )
        if hypothesis_index is not None:
            add(
                DialogueActType.HYPOTHESIS,
                hypothesis_index,
                len(surfaces) - 1,
                0.97,
                "epistemic_marker",
            )
        assumption_index = next(
            (
                index for index, value in enumerate(surfaces)
                if value in ASSUMPTION_MARKERS
            ),
            None,
        )
        if assumption_index is not None:
            add(
                DialogueActType.ASSUMPTION,
                assumption_index,
                len(surfaces) - 1,
                0.95,
                "assumption_marker",
            )
        plan_index = next(
            (
                index for index, value in enumerate(surfaces)
                if value in PLAN_MARKERS
            ),
            None,
        )
        if plan_index is not None:
            add(
                DialogueActType.PLAN,
                plan_index,
                len(surfaces) - 1,
                0.93,
                "intention_marker",
            )
        desire_index = next(
            (
                index for index, value in enumerate(surfaces)
                if value in DESIRE_MARKERS or lemmas[index] == "хотеть"
            ),
            None,
        )
        if desire_index is not None:
            add(
                DialogueActType.DESIRE,
                desire_index,
                len(surfaces) - 1,
                0.92,
                "desire_predicate",
            )

        speech_index = next(
            (
                index for index, lemma in enumerate(lemmas)
                if lemma in SPEECH_LEMMAS
            ),
            None,
        )
        has_direct_quote = bool(re.search(r"[«“\"]", source))
        reported_marker = next(
            (
                index for index, value in enumerate(surfaces)
                if value == "что" and speech_index is not None and index > speech_index
            ),
            None,
        )
        if speech_index is not None and (has_direct_quote or reported_marker is not None):
            content_start = (
                reported_marker + 1
                if reported_marker is not None
                else min(len(surfaces) - 1, speech_index + 1)
            )
            add(
                DialogueActType.ASSERTION,
                0,
                max(speech_index, content_start - 1),
                0.91,
                "speech_attribution",
            )
            add(
                DialogueActType.QUOTE
                if has_direct_quote else DialogueActType.REPORTED_SPEECH,
                content_start,
                len(surfaces) - 1,
                0.96,
                "quoted_content" if has_direct_quote else "reported_content",
            )

        definition_index = next(
            (
                index for index, value in enumerate(surfaces)
                if value == "это" and 0 < index < len(surfaces) - 1
            ),
            None,
        )
        if definition_index is not None and re.search(
            r"(?:—|–|-)\s*это\b",
            source.casefold(),
        ):
            add(
                DialogueActType.DEFINITION,
                0,
                len(surfaces) - 1,
                0.98,
                "explicit_definition_construction",
            )

        question_indices = [
            index for index, lemma in enumerate(lemmas)
            if lemma in QUESTION_LEMMAS or surfaces[index] in QUESTION_LEMMAS
        ]
        has_question_mark = "?" in source
        if question_indices or has_question_mark:
            if question_indices:
                start = question_indices[0]
            else:
                question_sentence = sentence_spans[-1]
                start = question_sentence[1]
            add(
                DialogueActType.QUESTION,
                start,
                len(surfaces) - 1,
                0.98 if question_indices else 0.88,
                "question_operator" if question_indices else "question_punctuation",
                alternatives=(
                    [{"act_type": "ASSERTION", "confidence": 0.18}]
                    if not question_indices else []
                ),
            )

        substantive = [
            act for act in candidates
            if act.act_type not in {
                DialogueActType.GREETING,
                DialogueActType.SMALL_TALK,
                DialogueActType.DENIAL,
                DialogueActType.CONFIRMATION,
                DialogueActType.CORRECTION,
            }
        ]
        if not substantive:
            greeting_only = all(
                act.act_type in {
                    DialogueActType.GREETING,
                    DialogueActType.SMALL_TALK,
                    DialogueActType.DENIAL,
                    DialogueActType.CONFIRMATION,
                }
                for act in candidates
            )
            if not candidates or not greeting_only:
                add(
                    DialogueActType.ASSERTION,
                    0,
                    len(surfaces) - 1,
                    0.82,
                    "declarative_default",
                    alternatives=[{"act_type": "HYPOTHESIS", "confidence": 0.12}],
                )
        candidates.sort(key=lambda item: (
            item.token_start,
            item.token_end,
            item.act_type.value,
        ))
        return candidates


class UtteranceParser:
    def __init__(self, dialogue_acts: Optional[DialogueActParser] = None) -> None:
        self.dialogue_acts = dialogue_acts or DialogueActParser()

    def envelope(
        self,
        text: str,
        *,
        conversation_id: str = "",
        turn_index: int = 0,
        speaker_role: str = "user",
        source_type: str = "dialogue",
        utterance_id: str = "",
        received_at: str = "",
    ) -> UtteranceEnvelope:
        normalized = normalize_text(text)
        return UtteranceEnvelope(
            id=utterance_id or _stable_id(
                "utterance",
                conversation_id,
                turn_index,
                speaker_role,
                normalized,
            ),
            conversation_id=conversation_id,
            turn_index=turn_index,
            speaker_role=speaker_role,
            raw_text=str(text or ""),
            normalized_text=normalized.casefold(),
            received_at=received_at or _utcnow(),
            source_type=source_type,
            interpretation_status=InterpretationStatus.INCOMPLETE,
        )

    def parse(
        self,
        text: str,
        *,
        conversation_id: str = "",
        turn_index: int = 0,
        speaker_role: str = "user",
        source_type: str = "dialogue",
        utterance_id: str = "",
        received_at: str = "",
        tokens: Optional[Sequence[ParsedToken]] = None,
    ) -> tuple[UtteranceEnvelope, List[DialogueAct]]:
        envelope = self.envelope(
            text,
            conversation_id=conversation_id,
            turn_index=turn_index,
            speaker_role=speaker_role,
            source_type=source_type,
            utterance_id=utterance_id,
            received_at=received_at,
        )
        acts = self.dialogue_acts.parse(
            text,
            utterance_id=envelope.id,
            tokens=tokens,
        )
        return envelope, acts


def primary_act(acts: Iterable[DialogueAct]) -> Optional[DialogueAct]:
    priority = {
        DialogueActType.CORRECTION: 100,
        DialogueActType.CLARIFICATION_REQUEST: 95,
        DialogueActType.QUESTION: 90,
        DialogueActType.REQUEST: 80,
        DialogueActType.COMMAND: 75,
        DialogueActType.CONDITION: 70,
        DialogueActType.COUNTERFACTUAL: 70,
        DialogueActType.HYPOTHESIS: 65,
        DialogueActType.ASSUMPTION: 60,
        DialogueActType.REPORTED_SPEECH: 55,
        DialogueActType.QUOTE: 55,
        DialogueActType.DEFINITION: 55,
        DialogueActType.ASSERTION: 50,
        DialogueActType.SMALL_TALK: 20,
        DialogueActType.GREETING: 10,
    }
    return max(
        acts,
        key=lambda act: (
            priority.get(act.act_type, 30),
            act.confidence,
            -act.token_start,
        ),
        default=None,
    )
