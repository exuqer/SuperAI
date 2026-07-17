"""Question operators are parsed separately from semantic role selection."""

from __future__ import annotations

from typing import Optional, Sequence

from .models import ParsedToken, QuestionOperator


class QuestionOperatorParser:
    QUESTION_LEMMAS = {
        "кто", "что", "где", "куда", "откуда", "когда", "как", "почему",
        "зачем", "чем", "сколько",
    }
    TYPED_QUESTION_LEMMA = "какой"

    def parse(
        self,
        tokens: Sequence[ParsedToken],
        mentions: Sequence[object],
    ) -> Optional[QuestionOperator]:
        question = next(
            (
                token for token in tokens
                if token.lemma == self.TYPED_QUESTION_LEMMA
                or token.normalized in self.QUESTION_LEMMAS
                or token.lemma in self.QUESTION_LEMMAS
            ),
            None,
        )
        if not question:
            return None
        if question.lemma != self.TYPED_QUESTION_LEMMA:
            return QuestionOperator(
                operator_type="ROLE_QUERY",
                surface=question.surface,
                token_indices=[question.index],
                question_lemma=question.lemma,
                grammatical_features=dict(question.features),
            )
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
        return QuestionOperator(
            operator_type="TYPED_ROLE_QUERY",
            surface=surface,
            token_indices=indices,
            question_lemma=question.lemma,
            grammatical_features=dict(question.features),
            type_constraint_token_index=type_index,
        )
