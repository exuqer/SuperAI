from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from server.factories import LexicalFactory, MorphologyFactory, SymbolFactory
from server.spaces import CloudObject, MorphemeSpace, SymbolSpace, WordSpace

from .answer_modes import ResultStatus


@dataclass(frozen=True)
class LexicalizationResult:
    surface: str | None
    status: ResultStatus
    confidence: float
    descent_path: tuple[str, ...]
    trace: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


class Lexicalizer:
    def __init__(self) -> None:
        self.lexical = LexicalFactory()
        self.morphology = MorphologyFactory()
        self.symbols = SymbolFactory()

    def realize(
        self,
        concept: str,
        features: Mapping[str, Any],
        word_space: WordSpace,
        morpheme_space: MorphemeSpace,
        symbol_space: SymbolSpace,
        *,
        root: str | None = None,
    ) -> LexicalizationResult:
        candidates = self.lexical.find(word_space, concept, features, min_relevance=0.72)
        if candidates:
            candidate = candidates[0]
            surface = str(candidate["dimensions"].get("surface") or candidate["label"])
            return LexicalizationResult(
                surface,
                ResultStatus.RETRIEVED,
                float(candidate["confidence"]),
                ("concept_space", "word_space"),
                ({"stage": "LEXICAL_RETRIEVAL", "candidate": candidate},),
            )
        morphology = self.morphology.compose(root or concept, features=features)
        word_id = f"word:{concept}:{morphology.surface}"
        morphemes = self.morphology.register(morphology, morpheme_space, word_id)
        symbols = self.symbols.register(
            morphology.surface, symbol_space, morphemes[-1].object_id if morphemes else ""
        )
        word_space.register(
            CloudObject(
                object_id=word_id,
                label=morphology.surface,
                dimensions={
                    "lemma": concept.casefold(),
                    "surface": morphology.surface,
                    "part_of_speech": features.get("part_of_speech", "unknown"),
                    "grammatical_features": dict(features),
                    "style": features.get("style", "neutral"),
                    "collocations": [],
                    "concept": concept.casefold(),
                    "role": features.get("role", ""),
                    "frequency": 0.0,
                    "morphological_similarity": morphology.confidence,
                },
                density=morphology.confidence,
                halo=0.16,
                links={
                    "up:concept_space": [f"concept:{concept.casefold()}"],
                    "down:morpheme_space": [item.object_id for item in morphemes],
                },
                provenance={"source": "morphology_factory", "status": morphology.status},
            )
        )
        return LexicalizationResult(
            morphology.surface,
            ResultStatus.COMPOSED if morphology.status == "COMPOSED" else ResultStatus.UNVERIFIED,
            morphology.confidence,
            (
                "concept_space",
                "word_space",
                "morpheme_space",
                "symbol_space",
                "morpheme_space",
                "word_space",
            ),
            (
                {"stage": "LEXICAL_MISS", "concept": concept, "features": dict(features)},
                {"stage": "MORPHOLOGY_COMPOSITION", "result": morphology.to_dict()},
                {
                    "stage": "SYMBOL_VALIDATION",
                    "result": morphology.symbol_validation,
                    "symbols": [item.object_id for item in symbols],
                },
            ),
        )
