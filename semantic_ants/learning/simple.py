from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from semantic_ants.core.normalization import text_to_concept_uri
from semantic_ants.learning.checkpoint import CheckpointStore
from semantic_ants.understanding import UnderstandingResult, UnderstandingToken, understand_text


@dataclass(frozen=True)
class SimpleConceptMeaning:
    concept: str | None = None
    label: str | None = None
    meaning: str = ""


@dataclass
class SimpleTrainingReport:
    examples: int = 0
    epochs: int = 0
    question_tokens: list[str] = field(default_factory=list)
    answer_tokens: list[str] = field(default_factory=list)
    meaning_tokens: list[str] = field(default_factory=list)
    reinforced_edges: int = 0
    role_edges: int = 0
    accepted_answers: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SimpleQATrainer:
    """Supervised обучение из пары вопрос -> ожидаемый ответ."""

    def __init__(self, engine: Any, store: CheckpointStore) -> None:
        self.engine = engine
        self.store = store

    def train_payload(self, payload: dict[str, Any]) -> SimpleTrainingReport:
        question = " ".join(str(payload.get("question") or payload.get("stimulus") or "").split())
        expected_answer = " ".join(
            str(payload.get("expected_answer") or payload.get("accepted_answer") or "").split()
        )
        if not question:
            raise ValueError("Simple training requires question")
        if not expected_answer:
            raise ValueError("Simple training requires expected_answer")

        epochs = max(int(payload.get("epochs") or 1), 1)
        lang = str(payload.get("lang", "auto"))
        reward = float(payload.get("reward", 1.0))
        meanings = _parse_meanings(payload.get("concept_meanings"))
        report = SimpleTrainingReport(examples=1, epochs=epochs)

        for _ in range(epochs):
            self._train_once(
                question=question,
                expected_answer=expected_answer,
                lang=lang,
                reward=reward,
                meanings=meanings,
                report=report,
            )
        self.store.save(self.engine.checkpoint)
        return report

    def _train_once(
        self,
        *,
        question: str,
        expected_answer: str,
        lang: str,
        reward: float,
        meanings: list[SimpleConceptMeaning],
        report: SimpleTrainingReport,
    ) -> None:
        checkpoint = self.engine.checkpoint
        question_result = understand_text(question, lang=lang, checkpoint=checkpoint)
        selected_lang = question_result.lang
        answer_result = understand_text(expected_answer, lang=selected_lang, checkpoint=checkpoint)
        question_tokens = _working_tokens(question_result)
        answer_tokens = _working_tokens(answer_result)
        question_concepts = _concepts(question_tokens)
        answer_concepts = _concepts(answer_tokens)

        report.question_tokens = [token.search_token for token in question_tokens]
        report.answer_tokens = [token.search_token for token in answer_tokens]

        for token in [*question_tokens, *answer_tokens]:
            if token.concept_uri:
                checkpoint.reinforce_concept(token.concept_uri, amount=0.2 * reward)

        for question_token in question_tokens:
            if not question_token.concept_uri:
                continue
            for answer_token in answer_tokens:
                if not answer_token.concept_uri:
                    continue
                self._add_edge(
                    question_token.concept_uri,
                    "ExpectedAnswerToken",
                    answer_token.concept_uri,
                    amount=0.45 * reward,
                    metadata={"source": "simple_training", "question": question},
                    report=report,
                )

        for left, right in zip(answer_tokens, answer_tokens[1:]):
            if left.concept_uri and right.concept_uri:
                self._add_edge(
                    left.concept_uri,
                    "AnswerNextToken",
                    right.concept_uri,
                    amount=0.55 * reward,
                    metadata={"source": "simple_training", "answer": expected_answer},
                    report=report,
                )

        meaning_token_values: list[str] = []
        for meaning in meanings:
            meaning_result = understand_text(meaning.meaning, lang=selected_lang, checkpoint=checkpoint)
            meaning_tokens = _working_tokens(meaning_result)
            meaning_token_values.extend(token.search_token for token in meaning_tokens)
            concept_uri = _meaning_concept_uri(meaning, selected_lang, checkpoint)
            if concept_uri and meaning.label:
                checkpoint.remember_concept_label(concept_uri, meaning.label)
            if concept_uri:
                _remember_concept_meaning(checkpoint, concept_uri, meaning)
            for meaning_token in meaning_tokens:
                if not meaning_token.concept_uri:
                    continue
                checkpoint.reinforce_concept(meaning_token.concept_uri, amount=0.18 * reward)
                if concept_uri:
                    self._add_edge(
                        concept_uri,
                        "DescribedByToken",
                        meaning_token.concept_uri,
                        amount=0.4 * reward,
                        metadata={"source": "simple_training", "meaning": meaning.meaning},
                        report=report,
                    )
                for question_token in question_tokens:
                    if question_token.concept_uri:
                        self._add_edge(
                            question_token.concept_uri,
                            "MeaningHint",
                            meaning_token.concept_uri,
                            amount=0.3 * reward,
                            metadata={"source": "simple_training", "meaning": meaning.meaning},
                            report=report,
                        )
        report.meaning_tokens = list(dict.fromkeys(meaning_token_values))

        report.role_edges += self._add_role_edges(answer_tokens, reward, report)

        concepts = list(dict.fromkeys([*question_concepts, *answer_concepts]))
        checkpoint.remember_response(concepts or answer_concepts, expected_answer, amount=max(reward, 0.1))
        remembered = checkpoint.remember_accepted_answer(
            stimulus=question,
            semantic_prompt="simple supervised question-answer",
            concepts=concepts,
            answer=expected_answer,
            reward=reward,
        )
        if remembered is not None:
            report.accepted_answers += 1
        checkpoint.examples_seen += 1

    def _add_role_edges(
        self,
        answer_tokens: list[UnderstandingToken],
        reward: float,
        report: SimpleTrainingReport,
    ) -> int:
        if len(answer_tokens) < 2:
            return 0
        verb_index = _first_index(answer_tokens, lambda token: token.morphology.get("POS") in {"VERB", "INFN"})
        if verb_index is None:
            return 0
        subject = _select_subject(answer_tokens, verb_index)
        verb = answer_tokens[verb_index]
        if not subject or not verb:
            return 0

        added = 0
        self._add_edge(
            subject.concept_uri,
            "CanDo",
            verb.concept_uri,
            amount=0.8 * reward,
            metadata={"source": "simple_training", "role": "subject_verb"},
            report=report,
        )
        added += 1
        for index, token in enumerate(answer_tokens):
            if index == verb_index or token is subject or not token.concept_uri:
                continue
            relation = _role_relation(token, verb_index, index)
            if relation is None:
                continue
            self._add_edge(
                verb.concept_uri,
                relation,
                token.concept_uri,
                amount=0.8 * reward,
                metadata={"source": "simple_training", "role": relation},
                report=report,
            )
            added += 1
        return added

    def _add_edge(
        self,
        start: str,
        relation: str,
        end: str,
        *,
        amount: float,
        metadata: dict[str, Any],
        report: SimpleTrainingReport,
    ) -> None:
        checkpoint = self.engine.checkpoint
        checkpoint.add_custom_edge(
            start,
            end,
            relation=relation,
            weight=max(amount, 0.1),
            layer=1,
            distance=1.0,
            edge_type="semantic",
            metadata=metadata,
        )
        checkpoint.reinforce_edge(start, relation, end, amount=max(amount, 0.1))
        report.reinforced_edges += 1


def _working_tokens(result: UnderstandingResult) -> list[UnderstandingToken]:
    tokens: list[UnderstandingToken] = []
    for token in result.tokens:
        if token.is_stop_word or not token.search_token or not token.concept_uri:
            continue
        tokens.append(_canonical_training_token(token, result.lang))
    return tokens


def _canonical_training_token(token: UnderstandingToken, lang: str) -> UnderstandingToken:
    if token.match_status != "partial_root_match" or not token.lemma or token.lemma == token.search_token:
        return token
    try:
        concept_uri = text_to_concept_uri(token.lemma, lang)
    except ValueError:
        return token
    return replace(
        token,
        search_token=token.lemma,
        concept_uri=concept_uri,
        match_status="candidate",
    )


def _concepts(tokens: list[UnderstandingToken]) -> list[str]:
    return list(dict.fromkeys(str(token.concept_uri) for token in tokens if token.concept_uri))


def _first_index(tokens: list[UnderstandingToken], predicate: Any) -> int | None:
    for index, token in enumerate(tokens):
        if predicate(token):
            return index
    return None


def _select_subject(tokens: list[UnderstandingToken], verb_index: int) -> UnderstandingToken | None:
    before = list(enumerate(tokens[:verb_index]))
    for _, token in reversed(before):
        if token.concept_uri and token.morphology.get("POS") in {"NOUN", "NPRO"} and token.morphology.get("case") == "nomn":
            return token
    for _, token in reversed(before):
        if token.concept_uri and token.morphology.get("POS") in {"NOUN", "NPRO"}:
            return token
    for index, token in enumerate(tokens):
        if index != verb_index and token.concept_uri and token.morphology.get("POS") in {"NOUN", "NPRO"}:
            return token
    return None


def _role_relation(token: UnderstandingToken, verb_index: int, token_index: int) -> str | None:
    pos = token.morphology.get("POS")
    case = token.morphology.get("case")
    if pos in {"ADJF", "ADJS", "PRTF", "PRTS"}:
        return "HasProperty"
    if pos not in {"NOUN", "NPRO"}:
        return None
    if token.search_token in _DEVICE_HINTS:
        return "UsesInstrument"
    if case == "loct":
        return "AtLocation"
    if case == "ablt":
        return "UsesInstrument"
    if token_index > verb_index:
        return "TakesObject"
    return None


_DEVICE_HINTS = {
    "компьютер",
    "ноутбук",
    "монитор",
    "экран",
    "телефон",
    "планшет",
}


def _parse_meanings(value: Any) -> list[SimpleConceptMeaning]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("concept_meanings must be a list")
    result: list[SimpleConceptMeaning] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("concept_meanings items must be objects")
        meaning = " ".join(str(item.get("meaning") or "").split())
        if not meaning:
            continue
        result.append(
            SimpleConceptMeaning(
                concept=str(item["concept"]) if item.get("concept") else None,
                label=str(item["label"]) if item.get("label") else None,
                meaning=meaning,
            )
        )
    return result


def _meaning_concept_uri(meaning: SimpleConceptMeaning, lang: str, checkpoint: Any) -> str | None:
    if meaning.concept:
        return meaning.concept
    if not meaning.label:
        return None
    label_result = understand_text(meaning.label, lang=lang, checkpoint=checkpoint)
    tokens = _working_tokens(label_result)
    return tokens[0].concept_uri if tokens else None


def _remember_concept_meaning(checkpoint: Any, concept_uri: str, meaning: SimpleConceptMeaning) -> None:
    definitions = checkpoint.metadata.setdefault("concept_definitions", {})
    if not isinstance(definitions, dict):
        return
    raw = definitions.get(concept_uri, {})
    info = dict(raw) if isinstance(raw, dict) else {}
    if meaning.label:
        info["label"] = meaning.label
    info["meaning"] = meaning.meaning
    definitions[concept_uri] = info
