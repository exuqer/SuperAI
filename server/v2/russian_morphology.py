"""Russian morphology with top-K hypotheses and surface realization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class Morphology:
    lemma: str
    pos_tag: str
    features: Dict[str, Any]
    confidence: float = 1.0


class RussianMorphology:
    def __init__(self) -> None:
        try:
            import pymorphy3
            self._analyzer = pymorphy3.MorphAnalyzer()
        except Exception:
            self._analyzer = None

    @staticmethod
    def _features(parsed: Any) -> Dict[str, Any]:
        tag = parsed.tag
        grammemes = set(tag.grammemes)
        features = {
            key: value
            for key, value in {
                "case": tag.case,
                "number": tag.number,
                "gender": tag.gender,
                "tense": tag.tense,
                "person": tag.person,
                "aspect": tag.aspect,
                "animacy": tag.animacy,
                "transitivity": tag.transitivity,
            }.items()
            if value is not None
        }
        features["proper_name"] = bool(
            grammemes & {"Name", "Surn", "Patr", "Geox", "Orgn", "Trad"}
        )
        return features

    def parse_variants(self, word: str, limit: int = 12) -> List[Morphology]:
        if not self._analyzer:
            return [Morphology(word.casefold(), "UNK", {}, 1.0)]
        result: List[Morphology] = []
        seen: set[tuple[Any, ...]] = set()
        for parsed in self._analyzer.parse(word):
            features = self._features(parsed)
            signature = (
                parsed.normal_form,
                str(parsed.tag.POS or "UNK"),
                tuple(sorted(features.items())),
            )
            if signature in seen:
                continue
            seen.add(signature)
            result.append(Morphology(
                parsed.normal_form,
                str(parsed.tag.POS or "UNK"),
                features,
                float(parsed.score),
            ))
            if len(result) >= max(1, int(limit)):
                break
        return result or [Morphology(word.casefold(), "UNK", {}, 1.0)]

    def parse(self, word: str) -> Morphology:
        return self.parse_variants(word, limit=1)[0]

    def inflect(self, word: str, features: Dict[str, str]) -> str:
        if not self._analyzer or not word:
            return word
        grammemes = {
            value
            for key, value in features.items()
            if key in {
                "case",
                "number",
                "gender",
                "tense",
                "person",
                "mood",
                "aspect",
            }
            and value
        }
        if not grammemes:
            return word
        parsed = self._analyzer.parse(word)
        generated = parsed[0].inflect(grammemes)
        if not generated:
            return word
        surface = generated.word
        if word[:1].isupper():
            surface = surface[:1].upper() + surface[1:]
        return surface
