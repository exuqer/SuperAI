"""One morphology/phrase pipeline shared by scenes and questions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from server.tokenizer import tokenize_hierarchical

from .diagnostics import MORPH_ANALYSIS_AMBIGUITY, SENTENCE_BOUNDARY_CROSSING
from .clause_parser import ClauseParser
from .models import (
    InterpretationStatus,
    LanguageAnalysis,
    MorphAnalysis,
    ParsedToken,
    Phrase,
    PhraseGraph,
)
from .noun_phrase_parser import ADJECTIVE_POS, NOUN_POS, EntityMentionParser
from .question_operator_parser import QuestionOperatorParser
from .relation_phrase_parser import RelationPhraseParser
from .utterance_parser import UtteranceParser


PREDICATE_POS = {"VERB", "INFN", "PRTS", "GRND"}


class UniversalLanguageAnalyzer:
    """Contextual morphology followed by maximal phrase construction."""

    PREPOSITION_CASES = {
        "в": {"accs", "loct", "loc2"},
        "во": {"accs", "loct", "loc2"},
        "на": {"accs", "loct", "loc2"},
        "к": {"datv"},
        "ко": {"datv"},
        "из": {"gent"},
        "от": {"gent"},
        "у": {"gent"},
        "для": {"gent"},
        "о": {"loct", "loc2"},
        "об": {"loct", "loc2"},
        "с": {"gent", "ablt"},
        "со": {"gent", "ablt"},
        "под": {"accs", "ablt"},
        "над": {"ablt"},
        "между": {"ablt"},
        "через": {"accs"},
    }

    def __init__(self, morphology: Any) -> None:
        self.morphology = morphology
        self.noun_phrases = EntityMentionParser()
        self.relation_phrases = RelationPhraseParser()
        self.questions = QuestionOperatorParser()
        self.utterances = UtteranceParser()
        self.clauses = ClauseParser()

    @staticmethod
    def _agreement(left: MorphAnalysis, right: MorphAnalysis) -> bool:
        for key in ("case", "number", "gender"):
            left_value = left.features.get(key)
            right_value = right.features.get(key)
            if left_value and right_value and left_value != right_value:
                return False
        return True

    def _variants(self, surface: str) -> List[MorphAnalysis]:
        parser = getattr(self.morphology, "parse_variants", None)
        if parser:
            variants = parser(surface)
            return [
                MorphAnalysis(
                    lemma=item.lemma,
                    pos=item.pos_tag,
                    features=dict(item.features),
                    confidence=float(getattr(item, "confidence", 1.0)),
                )
                for item in variants
            ]
        parsed = self.morphology.parse(surface)
        return [
            MorphAnalysis(
                lemma=parsed.lemma,
                pos=parsed.pos_tag,
                features=dict(parsed.features),
                confidence=float(getattr(parsed, "confidence", 1.0)),
            )
        ]

    def _select(
        self,
        variants: Sequence[List[MorphAnalysis]],
        surfaces: Sequence[str],
    ) -> List[MorphAnalysis]:
        selected: List[MorphAnalysis] = []
        for index, choices in enumerate(variants):
            scored: List[tuple[float, MorphAnalysis, List[str]]] = []
            for analysis in choices:
                score = analysis.confidence
                evidence = ["morphology_confidence"]
                if (
                    analysis.pos in ADJECTIVE_POS
                    and index + 1 < len(variants)
                ):
                    agreeing_heads: List[MorphAnalysis] = []
                    for cursor in range(
                        index + 1, min(len(variants), index + 4)
                    ):
                        noun_candidates = [
                            candidate for candidate in variants[cursor]
                            if candidate.pos in NOUN_POS
                        ]
                        if noun_candidates:
                            agreeing_heads = [
                                candidate for candidate in noun_candidates
                                if self._agreement(analysis, candidate)
                            ]
                            break
                        if not any(
                            candidate.pos in ADJECTIVE_POS
                            for candidate in variants[cursor]
                        ):
                            break
                    if agreeing_heads:
                        support = max(item.confidence for item in agreeing_heads)
                        score += .62 * support
                        evidence.extend([
                            "agreement_with_right_noun",
                            "maximal_noun_phrase_pattern",
                        ])
                if (
                    analysis.pos in NOUN_POS
                    and index > 0
                ):
                    agreeing_modifiers = [
                        candidate for candidate in variants[index - 1]
                        if candidate.pos in ADJECTIVE_POS
                        and self._agreement(candidate, analysis)
                    ]
                    if agreeing_modifiers:
                        score += .24 * max(
                            item.confidence for item in agreeing_modifiers
                        )
                        evidence.append("agreement_with_left_modifier")
                    governing_preposition = ""
                    for cursor in range(index - 1, max(-1, index - 4), -1):
                        normalized = surfaces[cursor].casefold()
                        if normalized in self.PREPOSITION_CASES:
                            governing_preposition = normalized
                            break
                        if not any(
                            candidate.pos in ADJECTIVE_POS
                            for candidate in variants[cursor]
                        ):
                            break
                    if governing_preposition:
                        grammatical_case = analysis.features.get("case")
                        if grammatical_case in self.PREPOSITION_CASES[
                            governing_preposition
                        ]:
                            # This is observable morphosyntactic compatibility,
                            # not a semantic role assignment.  Surface forms
                            # such as ``механика`` otherwise strongly prefer an
                            # unrelated nominative lemma over ``механик`` in
                            # the genitive after ``от``.
                            score += 1.00
                            evidence.extend([
                                "preposition_case_compatibility",
                                "preposition_case_government",
                            ])
                        elif grammatical_case:
                            score -= .85
                            evidence.append("preposition_case_conflict")
                if analysis.pos == "PREP":
                    score += .35
                    evidence.append("function_word_pattern")
                if (
                    analysis.pos in NOUN_POS
                    and surfaces[index][:1].isupper()
                    and analysis.features.get("proper_name")
                ):
                    score += .22
                    evidence.append("proper_name_marker")
                analysis.evidence = list(evidence)
                scored.append((score, analysis, evidence))
            _, winner, winner_evidence = max(
                scored,
                key=lambda item: (item[0], item[1].confidence),
            )
            winner.selected = True
            winner.evidence = list(winner_evidence)
            selected.append(winner)
        return selected

    def _select_question_operator_morphology(
        self,
        tokens: Sequence[ParsedToken],
    ) -> None:
        """Apply question-function evidence without assigning a semantic role.

        Russian homonyms such as ``что`` and ``когда`` are usually ranked as a
        conjunction by an isolated morphology analyzer.  In an interrogative
        position that choice erases the case or adverbial evidence required by
        the gap matcher.  Re-rank only among observable interrogative forms and
        keep every original hypothesis for late fixation.
        """
        for token in tokens:
            is_question_form = (
                token.lemma in self.questions.QUESTION_LEMMAS
                or token.normalized in self.questions.QUESTION_LEMMAS
                or token.lemma == self.questions.TYPED_QUESTION_LEMMA
                or any(
                    hypothesis.lemma == self.questions.TYPED_QUESTION_LEMMA
                    for hypothesis in token.analyses
                )
            )
            if not is_question_form:
                continue
            candidates = [
                hypothesis
                for hypothesis in token.analyses
                if (
                    hypothesis.pos in {"NPRO", "ADVB", "PRED", "NUMR", "ADJF"}
                    and (
                        hypothesis.lemma in self.questions.QUESTION_LEMMAS
                        or hypothesis.lemma
                        == self.questions.TYPED_QUESTION_LEMMA
                        or token.normalized in self.questions.QUESTION_LEMMAS
                    )
                )
            ]
            if not candidates:
                continue
            following = next(
                (
                    item for item in tokens[token.index + 1:]
                    if item.pos != "PNCT"
                ),
                None,
            )
            numeric_candidates = [
                hypothesis for hypothesis in candidates
                if hypothesis.pos == "NUMR"
            ]
            pool = (
                numeric_candidates
                if numeric_candidates
                and following is not None
                and following.pos in NOUN_POS
                else candidates
            )
            winner = max(pool, key=lambda item: item.confidence)
            for hypothesis in token.analyses:
                hypothesis.selected = hypothesis is winner
            if "question_operator_function" not in winner.evidence:
                winner.evidence.append("question_operator_function")
            token.lemma = winner.lemma
            token.pos = winner.pos
            token.features = {
                **dict(winner.features),
                "sentence_index": token.features.get("sentence_index", 0),
                "token_index_in_sentence": token.features.get(
                    "token_index_in_sentence",
                    token.index,
                ),
            }

    @staticmethod
    def _refine_question_case(
        tokens: Sequence[ParsedToken],
        mentions: Sequence[Any],
        predicate: Optional[ParsedToken],
    ) -> None:
        """Use only explicit agreement to narrow a question-word case.

        A past-tense verb can agree with an omitted participant.  Therefore a
        bare ``Что разрезал?`` must retain both nominative and accusative
        analyses: masculine agreement is not evidence that ``что`` is the
        nominative participant.  An explicit agreeing nominative mention is
        the one local observation that can safely prefer accusative here.
        """
        if predicate is None:
            return
        question = next(
            (
                token for token in tokens
                if any(
                    hypothesis.selected
                    and "question_operator_function" in hypothesis.evidence
                    for hypothesis in token.analyses
                )
            ),
            None,
        )
        if question is None or question.pos != "NPRO":
            return
        case_candidates = {
            str(hypothesis.features.get("case")): hypothesis
            for hypothesis in question.analyses
            if (
                hypothesis.pos == "NPRO"
                and hypothesis.features.get("case") in {"nomn", "accs"}
            )
        }
        if set(case_candidates) != {"nomn", "accs"}:
            return
        other_agrees = False
        for mention in mentions:
            head = tokens[mention.head]
            if head.features.get("case") != "nomn":
                continue
            comparable = [
                feature for feature in ("number", "gender")
                if head.features.get(feature)
                and predicate.features.get(feature)
            ]
            if comparable and all(
                head.features[feature] == predicate.features[feature]
                for feature in comparable
            ):
                other_agrees = True
                break
        if not other_agrees:
            # Keep the morphology parser's original selection.  Both variants
            # remain available downstream as competing gap hypotheses.
            return
        target_case = "accs"
        winner = case_candidates[target_case]
        for hypothesis in question.analyses:
            hypothesis.selected = hypothesis is winner
        winner.evidence = list(dict.fromkeys([
            *winner.evidence,
            "question_operator_function",
            "predicate_agreement_competition",
            "explicit_nominative_competitor",
        ]))
        question.lemma = winner.lemma
        question.pos = winner.pos
        question.features = {
            **dict(winner.features),
            "sentence_index": question.features.get("sentence_index", 0),
            "token_index_in_sentence": question.features.get(
                "token_index_in_sentence",
                question.index,
            ),
        }

    def analyze(
        self,
        text: str,
        *,
        token_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
        detect_question: bool = True,
        conversation_id: str = "",
        turn_index: int = 0,
        speaker_role: str = "user",
        source_type: str = "dialogue",
        utterance_id: str = "",
        received_at: str = "",
        reference_candidates: Optional[
            Dict[int, Sequence[Dict[str, Any]]]
        ] = None,
    ) -> LanguageAnalysis:
        surfaces = tokenize_hierarchical(text).all_tokens
        variants = [self._variants(token.text) for token in surfaces]
        winners = self._select(variants, [token.text for token in surfaces])
        metadata = token_metadata or {}
        tokens = [
            ParsedToken(
                index=index,
                surface=surface.text,
                normalized=surface.normalized.casefold(),
                lemma=winner.lemma,
                pos=winner.pos,
                features={
                    **dict(winner.features),
                    "sentence_index": surface.sentence_index,
                    "token_index_in_sentence": surface.token_index_in_sentence,
                },
                lexeme_cloud_id=metadata.get(index, {}).get("lexeme_cloud_id"),
                word_form_cloud_id=metadata.get(index, {}).get("word_form_cloud_id"),
                parser_annotation=str(
                    metadata.get(index, {}).get("parser_annotation") or "unknown"
                ),
                analyses=list(variants[index]),
            )
            for index, (surface, winner) in enumerate(zip(surfaces, winners))
        ]
        if detect_question:
            self._select_question_operator_morphology(tokens)
        relations = self.relation_phrases.parse(tokens)
        question_operator_indices = (
            self.questions.gap_operator_indices(tokens)
            if detect_question
            else set()
        )
        mentions = self.noun_phrases.parse(
            tokens,
            relations,
            excluded_indices=question_operator_indices,
        )
        diagnostics: List[Dict[str, Any]] = [
            *self.relation_phrases.last_diagnostics,
            *self.noun_phrases.last_diagnostics,
        ]
        valid_mentions = []
        for mention in mentions:
            sentence_indices = sorted({
                int(tokens[index].features.get("sentence_index", 0))
                for index in mention.token_indices
            })
            if len(sentence_indices) != 1:
                diagnostics.append({
                    "code": SENTENCE_BOUNDARY_CROSSING,
                    "construction": "mention",
                    "token_start": mention.start,
                    "token_end": mention.end,
                    "sentence_indices": sentence_indices,
                    "resolution": "discard_corrupted_mention",
                })
                continue
            mention.sentence_indices = sentence_indices
            valid_mentions.append(mention)
        mentions = valid_mentions
        predicate_candidates = [
            token for token in tokens if token.pos in PREDICATE_POS
        ]
        predicate = next(
            (
                token for token in predicate_candidates
                if token.pos in {"PRTS", "PRTF"}
                and any(
                    candidate.lemma == "быть"
                    for candidate in predicate_candidates
                    if candidate.index < token.index
                )
            ),
            predicate_candidates[0] if predicate_candidates else None,
        )
        self._refine_question_case(tokens, mentions, predicate)
        question_operators = (
            self.questions.parse_all(tokens, mentions) if detect_question else []
        )
        question = question_operators[0] if question_operators else None
        phrases: List[Phrase] = [
            Phrase(
                id=f"phrase-np-{index}",
                phrase_type="noun_phrase",
                token_start=mention.start,
                token_end=mention.end,
                head_token_index=mention.head,
                token_indices=list(mention.token_indices),
                surface=mention.surface,
                metadata={
                    "mention_type": mention.mention_type,
                    "preposition": mention.preposition,
                    "attributes": list(mention.attributes),
                    "relation_type": mention.relation_type,
                    "sentence_indices": list(mention.sentence_indices),
                },
            )
            for index, mention in enumerate(mentions)
        ]
        if predicate:
            phrases.append(Phrase(
                id="phrase-predicate",
                phrase_type="verb_phrase",
                token_start=predicate.index,
                token_end=predicate.index,
                head_token_index=predicate.index,
                token_indices=[predicate.index],
                surface=predicate.surface,
                metadata={"lemma": predicate.lemma},
            ))
        dependencies: List[Dict[str, Any]] = []
        if predicate:
            for index, mention in enumerate(mentions):
                dependencies.append({
                    "source": "phrase-predicate",
                    "relation": "argument",
                    "target": f"phrase-np-{index}",
                })
        for token in tokens:
            if len(token.analyses) > 1:
                ordered = sorted(
                    token.analyses, key=lambda item: item.confidence, reverse=True
                )
                if (
                    ordered[0] is not next(
                        item for item in token.analyses if item.selected
                    )
                ):
                    diagnostics.append({
                        "code": MORPH_ANALYSIS_AMBIGUITY,
                        "token_index": token.index,
                        "surface": token.surface,
                        "resolution": "contextual_agreement",
                    })
        analysis = LanguageAnalysis(
            tokens=tokens,
            mentions=mentions,
            phrase_graph=PhraseGraph(phrases, dependencies),
            predicate=predicate,
            question_operator=question,
            relation_phrases=[relation.as_dict() for relation in relations],
            diagnostics=diagnostics,
            question_operators=question_operators,
        )
        envelope, acts = self.utterances.parse(
            text,
            conversation_id=conversation_id,
            turn_index=turn_index,
            speaker_role=speaker_role,
            source_type=source_type,
            utterance_id=utterance_id,
            received_at=received_at,
            tokens=tokens,
        )
        clauses, clause_relations = self.clauses.parse(
            text,
            tokens,
            acts,
            utterance_id=envelope.id,
            speaker=speaker_role,
            mentions=mentions,
        )
        analysis.utterance = envelope
        analysis.dialogue_acts = acts
        analysis.clauses = clauses
        analysis.clause_relations = clause_relations
        analysis.interpretation_status = (
            InterpretationStatus.STABLE
            if clauses and (predicate is not None or question is not None)
            else InterpretationStatus.INCOMPLETE
        )
        analysis.interpretation_trace = {
            "pipeline": "role_free_observations",
            "morphological_hypotheses_preserved": True,
            "reference_candidate_count": sum(
                len(candidates)
                for candidates in (reference_candidates or {}).values()
            ),
        }
        return analysis
