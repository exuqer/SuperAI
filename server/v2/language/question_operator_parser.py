"""Question operators reserve structural gaps without semantic labels."""

from __future__ import annotations

from typing import List, Optional, Sequence, Set

from .models import ParsedToken, QuestionOperator


class QuestionOperatorParser:
    QUESTION_LEMMAS = {
        "кто", "что", "где", "куда", "откуда", "когда", "как", "почему",
        "зачем", "чем", "сколько",
    }
    TYPED_QUESTION_LEMMA = "какой"

    def gap_operator_indices(
        self,
        tokens: Sequence[ParsedToken],
    ) -> Set[int]:
        """Return tokens that reserve an argument slot instead of a noun slot.

        Interrogative pronouns can have noun-like morphology (``кому`` is an
        ``NPRO``), but they do not name an entity in the utterance.  Keeping
        their indices separate lets the phrase parser avoid treating a
        following proper noun as an apposition to the question word.

        ``какой`` is deliberately excluded: it is a determiner of a typed
        question and must remain available to construct ``какой <noun>``.
        """
        return {
            token.index
            for token in tokens
            if (
                token.lemma in self.QUESTION_LEMMAS
                or token.normalized in self.QUESTION_LEMMAS
            )
            and token.lemma != self.TYPED_QUESTION_LEMMA
        }

    def parse_all(
        self,
        tokens: Sequence[ParsedToken],
        mentions: Sequence[object],
    ) -> List[QuestionOperator]:
        """Reserve every interrogative operator, including coordinated ones."""
        result: List[QuestionOperator] = []
        for question in (
            token for token in tokens
            if token.index in self.gap_operator_indices(tokens)
            or token.lemma == self.TYPED_QUESTION_LEMMA
        ):
            if question.lemma != self.TYPED_QUESTION_LEMMA:
                result.append(QuestionOperator(
                    operator_type="EVENT_ATTACHMENT",
                    surface=question.surface,
                    token_indices=[question.index],
                    question_lemma=question.lemma,
                    grammatical_features=dict(question.features),
                ))
                continue
            typed_mention = next(
                (
                    mention for mention in mentions
                    if mention.start <= question.index <= mention.end
                    and mention.head != question.index
                ),
                None,
            )
            type_index = typed_mention.head if typed_mention else next(
                (
                    token.index for token in tokens[question.index + 1:]
                    if token.pos in {"NOUN", "NPRO"}
                ),
                None,
            )
            indices = (
                list(typed_mention.token_indices)
                if typed_mention
                else [question.index] + ([type_index] if type_index is not None else [])
            )
            surface = " ".join(tokens[index].surface for index in indices)
            result.append(QuestionOperator(
                operator_type="NODE_COMPONENT",
                surface=surface,
                token_indices=indices,
                question_lemma=question.lemma,
                grammatical_features=dict(question.features),
                type_constraint_token_index=type_index,
            ))
        return result

    def parse(
        self,
        tokens: Sequence[ParsedToken],
        mentions: Sequence[object],
    ) -> Optional[QuestionOperator]:
        """Compatibility accessor for single-gap consumers."""
        operators = self.parse_all(tokens, mentions)
        return operators[0] if operators else None
