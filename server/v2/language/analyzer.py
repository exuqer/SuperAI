"""One morphology/phrase pipeline shared by scenes and questions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from server.tokenizer import tokenize_hierarchical

from .diagnostics import MORPH_ANALYSIS_AMBIGUITY
from .models import (
    LanguageAnalysis,
    MorphAnalysis,
    ParsedToken,
    Phrase,
    PhraseGraph,
)
from .noun_phrase_parser import ADJECTIVE_POS, NOUN_POS, EntityMentionParser
from .question_operator_parser import QuestionOperatorParser
from .relation_phrase_parser import RelationPhraseParser


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
                            score += .35
                            evidence.append("preposition_case_government")
                        elif grammatical_case:
                            score -= .15
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
                scored.append((score, analysis, evidence))
            _, winner, winner_evidence = max(
                scored,
                key=lambda item: (item[0], item[1].confidence),
            )
            winner.selected = True
            winner.evidence = winner_evidence
            selected.append(winner)
        return selected

    def analyze(
        self,
        text: str,
        *,
        token_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
        detect_question: bool = True,
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
                features=dict(winner.features),
                lexeme_cloud_id=metadata.get(index, {}).get("lexeme_cloud_id"),
                word_form_cloud_id=metadata.get(index, {}).get("word_form_cloud_id"),
                grammatical_role=str(
                    metadata.get(index, {}).get("grammatical_role") or "unknown"
                ),
                analyses=list(variants[index]),
            )
            for index, (surface, winner) in enumerate(zip(surfaces, winners))
        ]
        relations = self.relation_phrases.parse(tokens)
        mentions = self.noun_phrases.parse(tokens, relations)
        predicate = next(
            (
                token for token in tokens
                if token.grammatical_role == "predicate"
                or token.pos in PREDICATE_POS
            ),
            None,
        )
        question = self.questions.parse(tokens, mentions) if detect_question else None
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
        diagnostics: List[Dict[str, Any]] = []
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
        return LanguageAnalysis(
            tokens=tokens,
            mentions=mentions,
            phrase_graph=PhraseGraph(phrases, dependencies),
            predicate=predicate,
            question_operator=question,
            relation_phrases=[relation.as_dict() for relation in relations],
            diagnostics=diagnostics,
        )
