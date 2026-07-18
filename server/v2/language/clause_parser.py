"""Clause boundaries, clause relations and semantic axes."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Sequence

from server.tokenizer import TOKEN_OR_PUNCT_RE, PUNCT_TOKENS, tokenize_hierarchical

from .models import (
    Actuality,
    Clause,
    ClauseMode,
    ClauseRelation,
    ClauseRelationType,
    CompletionStatus,
    DialogueAct,
    DialogueActType,
    EvidenceStatus,
    Modality,
    ParsedToken,
)
from .scope_parser import ScopeParser


PREDICATE_POS = {"VERB", "INFN", "PRTS", "GRND"}
COPULAR_PREDICATES = {
    "готов",
    "готова",
    "готово",
    "готовы",
    "нужен",
    "нужна",
    "нужно",
    "нужны",
    "должен",
    "должна",
    "должно",
    "должны",
}
SUBORDINATORS = {
    "если": ClauseRelationType.CONDITION,
    "когда": ClauseRelationType.SEQUENCE,
    "пока": ClauseRelationType.SIMULTANEOUS,
    "потому": ClauseRelationType.CAUSE,
    "поскольку": ClauseRelationType.CAUSE,
    "чтобы": ClauseRelationType.PURPOSE,
    "хотя": ClauseRelationType.CONCESSION,
}
COORDINATORS = {
    "но": ClauseRelationType.CONTRAST,
    "а": ClauseRelationType.CONTRAST,
    "или": ClauseRelationType.ALTERNATIVE,
    "либо": ClauseRelationType.ALTERNATIVE,
    "и": ClauseRelationType.ENUMERATION,
}
SPEECH_LEMMAS = {
    "сказать",
    "говорить",
    "сообщить",
    "ответить",
    "написать",
    "утверждать",
}
MODE_BY_ACT = {
    DialogueActType.ASSERTION: ClauseMode.ASSERTION,
    DialogueActType.QUESTION: ClauseMode.QUESTION,
    DialogueActType.REQUEST: ClauseMode.REQUEST,
    DialogueActType.COMMAND: ClauseMode.COMMAND,
    DialogueActType.DEFINITION: ClauseMode.DEFINITION,
    DialogueActType.ASSUMPTION: ClauseMode.ASSUMPTION,
    DialogueActType.HYPOTHESIS: ClauseMode.HYPOTHESIS,
    DialogueActType.CONDITION: ClauseMode.CONDITION,
    DialogueActType.COUNTERFACTUAL: ClauseMode.COUNTERFACTUAL,
    DialogueActType.DESIRE: ClauseMode.DESIRE,
    DialogueActType.PLAN: ClauseMode.PLAN,
    DialogueActType.QUOTE: ClauseMode.QUOTE,
    DialogueActType.REPORTED_SPEECH: ClauseMode.REPORTED_SPEECH,
    DialogueActType.EXAMPLE: ClauseMode.EXAMPLE,
}
MODE_PRIORITY = {
    ClauseMode.COUNTERFACTUAL: 100,
    ClauseMode.CONDITION: 95,
    ClauseMode.QUOTE: 92,
    ClauseMode.REPORTED_SPEECH: 92,
    ClauseMode.QUESTION: 90,
    ClauseMode.REQUEST: 85,
    ClauseMode.COMMAND: 80,
    ClauseMode.HYPOTHESIS: 75,
    ClauseMode.ASSUMPTION: 70,
    ClauseMode.DESIRE: 65,
    ClauseMode.PLAN: 60,
    ClauseMode.DEFINITION: 55,
    ClauseMode.ASSERTION: 50,
}


def _stable_id(prefix: str, *parts: object) -> str:
    key = "|".join(str(part) for part in parts)
    return f"{prefix}-{uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:20]}"


class ClauseParser:
    def __init__(self, scope_parser: Optional[ScopeParser] = None) -> None:
        self.scope_parser = scope_parser or ScopeParser()

    @staticmethod
    def _is_predicate(token: ParsedToken) -> bool:
        return (
            token.pos in PREDICATE_POS
            or token.normalized.casefold() in COPULAR_PREDICATES
        )

    def _has_predicate(
        self,
        tokens: Sequence[ParsedToken],
        start: int,
        end: int,
    ) -> bool:
        return any(self._is_predicate(token) for token in tokens[start:end + 1])

    @staticmethod
    def _punctuation_boundaries(text: str) -> Dict[int, str]:
        boundaries: Dict[int, str] = {}
        word_index = -1
        for match in TOKEN_OR_PUNCT_RE.finditer(str(text or "")):
            value = match.group(0)
            if value in PUNCT_TOKENS:
                if word_index >= 0:
                    boundaries[word_index] = value
            else:
                word_index += 1
        return boundaries

    def _sentence_spans(
        self,
        text: str,
        tokens: Sequence[ParsedToken],
    ) -> List[tuple[int, int, int]]:
        hierarchy = tokenize_hierarchical(text)
        if len(hierarchy.all_tokens) != len(tokens):
            return [(0, 0, len(tokens) - 1)] if tokens else []
        result: List[tuple[int, int, int]] = []
        for sentence in hierarchy.sentences:
            indices = [
                token.position for token in sentence.tokens
            ]
            if indices:
                result.append((sentence.index, min(indices), max(indices)))
        return result

    def _split_sentence(
        self,
        tokens: Sequence[ParsedToken],
        start: int,
        end: int,
        punctuation: Dict[int, str],
    ) -> List[tuple[int, int, Optional[str]]]:
        if start > end:
            return []
        split_points: Dict[int, Optional[str]] = {}
        for index in range(start, end):
            normalized = tokens[index].normalized.casefold()
            punctuation_after = punctuation.get(index)
            if punctuation_after in {",", ":", ";"}:
                right = index + 1
                punctuation_connector = (
                    ":"
                    if punctuation_after == ":"
                    else tokens[right].normalized.casefold()
                )
                if (
                    right <= end
                    and tokens[right].normalized.casefold() == "что"
                    and any(
                        token.lemma.casefold() in SPEECH_LEMMAS
                        for token in tokens[start:right]
                    )
                    and right + 1 <= end
                ):
                    right += 1
                    punctuation_connector = "что"
                if (
                    self._has_predicate(tokens, start, index)
                    and self._has_predicate(tokens, right, end)
                ):
                    split_points[right] = punctuation_connector
            if normalized in COORDINATORS and index > start:
                left_end = index - 1
                right = index + 1
                if (
                    right <= end
                    and self._has_predicate(tokens, start, left_end)
                    and self._has_predicate(tokens, right, end)
                ):
                    split_points[right] = normalized
            if (
                normalized == "что"
                and index > start
                and any(
                    token.lemma.casefold() in SPEECH_LEMMAS
                    for token in tokens[start:index]
                )
                and self._has_predicate(tokens, index + 1, end)
            ):
                split_points[index + 1] = "что"
        points = sorted(point for point in split_points if start < point <= end)
        segments: List[tuple[int, int, Optional[str]]] = []
        cursor = start
        connector: Optional[str] = None
        for point in points:
            segment_end = point - 1
            while (
                segment_end >= cursor
                and tokens[segment_end].normalized.casefold() in COORDINATORS
            ):
                segment_end -= 1
            if segment_end >= cursor:
                segments.append((cursor, segment_end, connector))
            cursor = point
            connector = split_points[point]
        if cursor <= end:
            segments.append((cursor, end, connector))
        return segments or [(start, end, None)]

    @staticmethod
    def _overlapping_mode(
        acts: Sequence[DialogueAct],
        start: int,
        end: int,
    ) -> ClauseMode:
        modes: List[tuple[int, float, ClauseMode]] = []
        for act in acts:
            if act.token_end < start or act.token_start > end:
                continue
            if (
                act.act_type in {
                    DialogueActType.HYPOTHESIS,
                    DialogueActType.ASSUMPTION,
                    DialogueActType.DESIRE,
                    DialogueActType.PLAN,
                }
                and not start <= act.token_start <= end
            ):
                continue
            mode = MODE_BY_ACT.get(act.act_type)
            if mode:
                modes.append((MODE_PRIORITY[mode], act.confidence, mode))
        return max(modes, default=(0, 0.0, ClauseMode.ASSERTION))[2]

    @staticmethod
    def _actuality(mode: ClauseMode) -> Actuality:
        if mode == ClauseMode.COUNTERFACTUAL:
            return Actuality.COUNTERFACTUAL
        if mode == ClauseMode.CONDITION:
            return Actuality.HYPOTHETICAL
        if mode in {
            ClauseMode.HYPOTHESIS,
            ClauseMode.ASSUMPTION,
            ClauseMode.DESIRE,
            ClauseMode.PLAN,
        }:
            return Actuality.POSSIBLE
        if mode == ClauseMode.EXAMPLE:
            return Actuality.FICTIONAL
        if mode in {ClauseMode.QUESTION, ClauseMode.REQUEST, ClauseMode.COMMAND}:
            return Actuality.UNKNOWN
        return Actuality.ACTUAL

    @staticmethod
    def _completion(
        tokens: Sequence[ParsedToken],
        modality: Optional[Modality],
        mode: ClauseMode,
    ) -> CompletionStatus:
        if modality == Modality.WANT:
            return CompletionStatus.NOT_STARTED
        if mode in {ClauseMode.PLAN, ClauseMode.COMMAND, ClauseMode.REQUEST}:
            return CompletionStatus.PLANNED
        if any(token.lemma.casefold() in {"успеть", "прервать"} for token in tokens):
            if any(token.normalized.casefold() == "не" for token in tokens):
                return CompletionStatus.INTERRUPTED
        tenses = {
            token.features.get("tense")
            for token in tokens
            if isinstance(token.features, dict)
        }
        if "past" in tenses:
            return CompletionStatus.COMPLETED
        if "pres" in tenses:
            return CompletionStatus.ONGOING
        if "futr" in tenses:
            return CompletionStatus.PLANNED
        return CompletionStatus.UNKNOWN

    @staticmethod
    def _relation_for(
        connector: Optional[str],
        mode: ClauseMode,
    ) -> ClauseRelationType:
        if connector == ":":
            return (
                ClauseRelationType.QUOTE_CONTENT
                if mode == ClauseMode.QUOTE
                else ClauseRelationType.EXPLANATION
            )
        if connector == "что":
            return ClauseRelationType.REPORTED_CONTENT
        if connector in COORDINATORS:
            return COORDINATORS[connector]
        if connector in SUBORDINATORS:
            return SUBORDINATORS[connector]
        return ClauseRelationType.SEQUENCE

    @staticmethod
    def _quoted_speaker(
        prior_tokens: Sequence[ParsedToken],
    ) -> Optional[str]:
        speech_index = next(
            (
                index for index, token in enumerate(prior_tokens)
                if token.lemma.casefold() in SPEECH_LEMMAS
            ),
            None,
        )
        if speech_index is None:
            return None
        candidate = next(
            (
                token for token in reversed(prior_tokens[:speech_index])
                if token.pos in {"NOUN", "NPRO"}
            ),
            None,
        )
        return candidate.surface if candidate else None

    @staticmethod
    def _participants(
        mentions: Sequence[Any],
        start: int,
        end: int,
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for mention in mentions:
            mention_start = int(getattr(mention, "start", -1))
            mention_end = int(getattr(mention, "end", -1))
            if mention_start < start or mention_end > end:
                continue
            features = dict(getattr(mention, "features", {}) or {})
            grammatical_case = features.get("case")
            preposition = str(
                getattr(mention, "preposition", "") or ""
            ).casefold()
            relation_function = getattr(
                mention,
                "relation_function",
                None,
            )
            if relation_function:
                role = str(relation_function)
                confidence = 0.94
                evidence = ["compound_relation_operator"]
            elif preposition in {"в", "во", "на", "под"}:
                role = (
                    "destination"
                    if grammatical_case == "accs"
                    else "location"
                )
                confidence = 0.9
                evidence = ["preposition_case_government"]
            elif preposition in {"из", "от"} or (
                preposition in {"с", "со"}
                and grammatical_case == "gent"
            ):
                role = "source"
                confidence = 0.9
                evidence = ["preposition_case_government"]
            elif preposition in {"к", "ко"}:
                role = "destination"
                confidence = 0.9
                evidence = ["preposition_case_government"]
            elif preposition in {"с", "со"} and grammatical_case == "ablt":
                role = "instrument"
                confidence = 0.76
                evidence = ["preposition_case_government"]
            elif grammatical_case == "nomn":
                role = "agent"
                confidence = 0.82
                evidence = ["nominative_case"]
            elif grammatical_case == "datv":
                role = "recipient"
                confidence = 0.78
                evidence = ["dative_case"]
            elif grammatical_case == "ablt":
                role = "instrument"
                confidence = 0.72
                evidence = ["instrumental_case"]
            elif grammatical_case in {"loct", "loc2"}:
                role = "location"
                confidence = 0.82
                evidence = ["locative_case"]
            else:
                role = "object"
                confidence = 0.72
                evidence = ["grammatical_case"]
            result.append({
                "mention_id": getattr(mention, "id", None),
                "token_start": mention_start,
                "token_end": mention_end,
                "surface": getattr(mention, "surface", ""),
                "lemma": getattr(mention, "lemma", ""),
                "preposition": preposition,
                "role_hypotheses": [{
                    "role": role,
                    "confidence": confidence,
                    "selected": False,
                    "evidence": evidence,
                }],
            })
        return result

    def parse(
        self,
        text: str,
        tokens: Sequence[ParsedToken],
        acts: Sequence[DialogueAct],
        *,
        utterance_id: str,
        speaker: str = "user",
        mentions: Sequence[Any] = (),
    ) -> tuple[List[Clause], List[ClauseRelation]]:
        if not tokens:
            return [], []
        punctuation = self._punctuation_boundaries(text)
        raw_segments: List[tuple[int, int, int, Optional[str]]] = []
        for sentence_index, start, end in self._sentence_spans(text, tokens):
            raw_segments.extend(
                (sentence_index, left, right, connector)
                for left, right, connector in self._split_sentence(
                    tokens,
                    start,
                    end,
                    punctuation,
                )
            )
        sentence_bounds: Dict[int, tuple[int, int]] = {}
        for sentence_index, start, end, _ in raw_segments:
            current = sentence_bounds.get(sentence_index)
            sentence_bounds[sentence_index] = (
                min(current[0], start) if current else start,
                max(current[1], end) if current else end,
            )
        clauses: List[Clause] = []
        for ordinal, (sentence_index, start, end, connector) in enumerate(raw_segments):
            clause_tokens = list(tokens[start:end + 1])
            mode = self._overlapping_mode(acts, start, end)
            first = clause_tokens[0].normalized.casefold()
            if first in SUBORDINATORS:
                mode = (
                    ClauseMode.COUNTERFACTUAL
                    if first == "если"
                    and len(clause_tokens) > 1
                    and clause_tokens[1].normalized.casefold() == "бы"
                    else ClauseMode.CONDITION
                )
            scope = self.scope_parser.parse(clause_tokens, token_offset=start)
            predicate_candidates = [
                token for token in clause_tokens if self._is_predicate(token)
            ]
            definition_predicates = (
                mode == ClauseMode.DEFINITION
                and len(predicate_candidates) >= 2
            )
            predicate_hypotheses = [
                {
                    "token_index": token.index,
                    "surface": token.surface,
                    "lemma": token.lemma,
                    "part_of_speech": token.pos,
                    "confidence": round(
                        max(
                            (
                                analysis.confidence
                                for analysis in token.analyses
                                if analysis.lemma == token.lemma
                                and analysis.pos == token.pos
                            ),
                            default=0.78,
                        ),
                        4,
                    ),
                    "selected": False,
                    "embedded": (
                        (
                            definition_predicates
                            and index < len(predicate_candidates) - 1
                        )
                        or (
                            index > 0
                            and token.pos == "INFN"
                            and scope["modality"] is not None
                        )
                    ),
                    "definition_role": (
                        "defined_term"
                        if definition_predicates and index == 0
                        else "definition_value"
                        if definition_predicates
                        and index == len(predicate_candidates) - 1
                        else None
                    ),
                    "evidence": ["predicative_part_of_speech"],
                }
                for index, token in enumerate(predicate_candidates)
            ]
            relation = self._relation_for(connector, mode)
            quoted_speaker = None
            if relation in {
                ClauseRelationType.QUOTE_CONTENT,
                ClauseRelationType.REPORTED_CONTENT,
            } or mode in {ClauseMode.QUOTE, ClauseMode.REPORTED_SPEECH}:
                quoted_speaker = self._quoted_speaker(tokens[:start])
            clause = Clause(
                id=_stable_id("clause", utterance_id, ordinal, start, end),
                utterance_id=utterance_id,
                sentence_index=sentence_index,
                token_start=start,
                token_end=end,
                clause_type=(
                    "CONDITIONAL"
                    if mode in {ClauseMode.CONDITION, ClauseMode.COUNTERFACTUAL}
                    else "QUOTED"
                    if mode in {ClauseMode.QUOTE, ClauseMode.REPORTED_SPEECH}
                    else "MAIN"
                ),
                relation_to_parent=relation if ordinal else None,
                predicate_hypotheses=predicate_hypotheses,
                mode=mode,
                actuality=self._actuality(mode),
                evidence_status=EvidenceStatus.STATED,
                polarity=scope["polarity"],
                negation_scope=scope["negation_scope"],
                modality=scope["modality"],
                completion_status=self._completion(
                    clause_tokens,
                    scope["modality"],
                    mode,
                ),
                speaker=speaker,
                quoted_speaker=quoted_speaker,
                surface=" ".join(token.surface for token in clause_tokens),
                evidence=[
                    {
                        "origin": "clause_parser",
                        "type": "predicate_centers_and_connectors",
                        "token_start": start,
                        "token_end": end,
                    }
                ],
                alternative_boundaries=[
                    {
                        "token_start": sentence_bounds[sentence_index][0],
                        "token_end": sentence_bounds[sentence_index][1],
                        "confidence": 0.36,
                        "reason": "unsplit_sentence_alternative",
                    }
                ] if sentence_bounds[sentence_index] != (start, end) else [],
                participants=self._participants(mentions, start, end),
            )
            clauses.append(clause)

        relations: List[ClauseRelation] = []

        def add_relation(
            source: Clause,
            target: Clause,
            relation_type: ClauseRelationType,
            confidence: float,
            evidence_type: str,
        ) -> None:
            relations.append(ClauseRelation(
                id=_stable_id(
                    "clause-relation",
                    utterance_id,
                    source.id,
                    target.id,
                    relation_type.value,
                ),
                source_clause_id=source.id,
                target_clause_id=target.id,
                relation_type=relation_type,
                confidence=confidence,
                evidence=[{
                    "origin": "clause_parser",
                    "type": evidence_type,
                }],
            ))

        conditional_targets: set[str] = set()
        for index, clause in enumerate(clauses):
            if clause.mode not in {
                ClauseMode.CONDITION,
                ClauseMode.COUNTERFACTUAL,
            }:
                continue
            target = (
                clauses[index + 1]
                if index + 1 < len(clauses)
                else clauses[index - 1]
                if index > 0
                else None
            )
            if target is None:
                continue
            clause.parent_clause_id = target.id
            clause.relation_to_parent = ClauseRelationType.CONDITION
            conditional_targets.add(target.id)
            add_relation(
                clause,
                target,
                ClauseRelationType.CONDITION,
                0.98,
                "conditional_subordinator",
            )
        for index, clause in enumerate(clauses):
            if index == 0 or clause.mode in {
                ClauseMode.CONDITION,
                ClauseMode.COUNTERFACTUAL,
            }:
                continue
            parent = clauses[index - 1]
            if (
                clause.id in conditional_targets
                and parent.mode in {
                    ClauseMode.CONDITION,
                    ClauseMode.COUNTERFACTUAL,
                }
            ):
                clause.parent_clause_id = None
                clause.relation_to_parent = None
                continue
            relation_type = (
                clause.relation_to_parent
                or ClauseRelationType.SEQUENCE
            )
            clause.parent_clause_id = parent.id
            add_relation(
                parent,
                clause,
                relation_type,
                0.93,
                "connector_or_predicate_boundary",
            )
        unique: Dict[tuple[str, str, str], ClauseRelation] = {}
        for relation in relations:
            unique[(
                relation.source_clause_id,
                relation.target_clause_id,
                relation.relation_type.value,
            )] = relation
        return clauses, list(unique.values())
