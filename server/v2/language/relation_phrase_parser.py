"""Recognition of grammatical multiword relation operators.

The entries below are language resources: they describe Russian function-word
constructions and contain no domain entities or predicate-specific knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .models import ParsedToken


@dataclass(frozen=True)
class RelationPhrase:
    token_start: int
    token_end: int
    surface: str
    normalized: str
    relation_type: str
    grammatical_function: str
    confidence: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "token_start": self.token_start,
            "token_end": self.token_end,
            "surface": self.surface,
            "normalized": self.normalized,
            "relation_type": self.relation_type,
            "grammatical_function": self.grammatical_function,
            "confidence": self.confidence,
        }


class RelationPhraseParser:
    # Longest patterns must win. Variants with "со/ко" are normalized through
    # token lemmas, so one grammatical operator has one canonical form.
    OPERATORS: Dict[Tuple[str, ...], Tuple[str, str]] = {
        ("по", "направление", "к"): ("ORIENTATION_TO", "destination"),
        ("с", "помощь"): ("USES", "instrument"),
        ("в", "результат"): ("RESULTS_IN", "cause"),
        ("в", "связь", "с"): ("REFERENCE", "reference"),
        ("в", "отличие", "от"): ("OPPOSITE_TO", "reference"),
        ("рядом", "с"): ("LOCATED_NEAR", "location"),
        ("вместе", "с"): ("ACCOMPANIMENT", "reference"),
        ("слева", "от"): ("ORIENTATION_LEFT", "reference"),
        ("справа", "от"): ("ORIENTATION_RIGHT", "reference"),
        ("из-за",): ("CAUSES", "cause"),
        ("напротив",): ("LOCATED_NEAR", "location"),
        ("внутри",): ("LOCATED_IN", "location"),
        ("снаружи",): ("REFERENCE", "reference"),
        ("около",): ("LOCATED_NEAR", "location"),
        ("возле",): ("LOCATED_NEAR", "location"),
        ("кроме",): ("EXCLUSION", "exclusion"),
    }

    def parse(self, tokens: Sequence[ParsedToken]) -> List[RelationPhrase]:
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
                relation_type, function = self.OPERATORS[pattern]
                result.append(RelationPhrase(
                    token_start=start,
                    token_end=end - 1,
                    surface=" ".join(token.surface for token in tokens[start:end]),
                    normalized=" ".join(pattern),
                    relation_type=relation_type,
                    grammatical_function=function,
                    confidence=.94 if len(pattern) > 1 else .86,
                ))
                occupied.update(range(start, end))
                break
        return result

    @staticmethod
    def governing(
        relations: Sequence[RelationPhrase],
        mention_start: int,
    ) -> Optional[RelationPhrase]:
        candidates = [
            relation
            for relation in relations
            if relation.token_end < mention_start
            and mention_start - relation.token_end <= 2
        ]
        return max(candidates, key=lambda item: item.token_end, default=None)
