"""Recognition of grammatical multiword relation operators.

The entries below are language resources: they describe Russian function-word
constructions and contain no domain entities or predicate-specific knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .models import ParsedToken
from .diagnostics import SENTENCE_BOUNDARY_CROSSING


@dataclass(frozen=True)
class RelationPhrase:
    token_start: int
    token_end: int
    surface: str
    normalized: str
    relation_type: str
    structural_signature: str
    confidence: float
    sentence_index: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "token_start": self.token_start,
            "token_end": self.token_end,
            "surface": self.surface,
            "normalized": self.normalized,
            "relation_type": self.relation_type,
            "structural_signature": self.structural_signature,
            "confidence": self.confidence,
            "sentence_index": self.sentence_index,
        }


class RelationPhraseParser:
    # Longest patterns must win. Variants with "со/ко" are normalized through
    # token lemmas, so one grammatical operator has one canonical form.
    OPERATORS: Dict[Tuple[str, ...], str] = {
        ("по", "направление", "к"): "ORIENTATION_TO",
        ("с", "помощь"): "USES",
        ("в", "результат"): "RESULTS_IN",
        ("в", "связь", "с"): "REFERENCE",
        ("в", "отличие", "от"): "OPPOSITE_TO",
        ("рядом", "с"): "LOCATED_NEAR",
        ("вместе", "с"): "ACCOMPANIMENT",
        ("слева", "от"): "ORIENTATION_LEFT",
        ("справа", "от"): "ORIENTATION_RIGHT",
        ("из-за",): "CAUSES",
        ("напротив",): "LOCATED_NEAR",
        ("внутри",): "LOCATED_IN",
        ("снаружи",): "REFERENCE",
        ("около",): "LOCATED_NEAR",
        ("возле",): "LOCATED_NEAR",
        ("кроме",): "EXCLUSION",
    }

    def __init__(self) -> None:
        self.last_diagnostics: List[Dict[str, object]] = []

    @staticmethod
    def _sentence_index(token: ParsedToken) -> int:
        return int(token.features.get("sentence_index", 0))

    def parse(self, tokens: Sequence[ParsedToken]) -> List[RelationPhrase]:
        self.last_diagnostics = []
        result: List[RelationPhrase] = []
        lemmas = [token.lemma.casefold() for token in tokens]
        occupied: set[int] = set()
        patterns = sorted(self.OPERATORS, key=len, reverse=True)
        for start in range(len(tokens)):
            if start in occupied:
                continue
            for pattern in patterns:
                end = start + len(pattern)
                if tuple(lemmas[start:end]) != pattern:
                    continue
                sentence_indices = {
                    self._sentence_index(token) for token in tokens[start:end]
                }
                if len(sentence_indices) != 1:
                    self.last_diagnostics.append({
                        "code": SENTENCE_BOUNDARY_CROSSING,
                        "construction": "relation_phrase",
                        "token_start": start,
                        "token_end": end - 1,
                        "sentence_indices": sorted(sentence_indices),
                        "resolution": "split_and_rebuild_mentions",
                    })
                    continue
                relation_type = self.OPERATORS[pattern]
                result.append(RelationPhrase(
                    token_start=start,
                    token_end=end - 1,
                    surface=" ".join(token.surface for token in tokens[start:end]),
                    normalized=" ".join(pattern),
                    relation_type=relation_type,
                    structural_signature=(
                        "operator:" + "_".join(pattern)
                    ),
                    confidence=.94 if len(pattern) > 1 else .86,
                    sentence_index=next(iter(sentence_indices)),
                ))
                occupied.update(range(start, end))
                break
        return result

    @staticmethod
    def governing(
        relations: Sequence[RelationPhrase],
        mention_start: int,
        *,
        sentence_index: Optional[int] = None,
    ) -> Optional[RelationPhrase]:
        candidates = [
            relation
            for relation in relations
            if relation.token_end < mention_start
            and mention_start - relation.token_end <= 2
            and (
                sentence_index is None
                or relation.sentence_index == sentence_index
            )
        ]
        return max(candidates, key=lambda item: item.token_end, default=None)
