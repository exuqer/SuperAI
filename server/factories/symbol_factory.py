from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from server.spaces import CloudObject, SymbolSpace


@dataclass(frozen=True)
class SymbolValidation:
    surface: str
    valid: bool
    score: float
    issues: tuple[str, ...]
    characters: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SymbolFactory:
    alphabet = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя-")
    vowels = set("аеёиоуыэюя")
    invalid_sequences = ("ъъ", "ьь", "ыы", "йй", "ъь", "ьъ")

    def validate(self, surface: str) -> SymbolValidation:
        normalized = surface.casefold().strip()
        issues: list[str] = []
        if not normalized:
            issues.append("empty")
        unknown = sorted({char for char in normalized if char not in self.alphabet})
        if unknown:
            issues.append("unknown_symbols:" + "".join(unknown))
        issues.extend(
            f"invalid_sequence:{sequence}"
            for sequence in self.invalid_sequences
            if sequence in normalized
        )
        if normalized.startswith(("ъ", "ь", "ы")):
            issues.append("invalid_initial_symbol")
        if normalized.endswith(("ъ", "ы")):
            issues.append("unlikely_final_symbol")
        if len(normalized) > 2 and not any(char in self.vowels for char in normalized):
            issues.append("missing_vowel")
        score = max(0.0, 1.0 - len(issues) * 0.24)
        return SymbolValidation(
            normalized, not issues, round(score, 6), tuple(issues), tuple(normalized)
        )

    def repair(self, surface: str) -> SymbolValidation:
        repaired = re.sub(r"[^а-яё-]", "", surface.casefold())
        for sequence in self.invalid_sequences:
            repaired = repaired.replace(sequence, sequence[0])
        repaired = repaired.lstrip("ъь ы")
        return self.validate(repaired)

    def register(
        self, surface: str, space: SymbolSpace, morpheme_id: str = ""
    ) -> list[CloudObject]:
        validation = self.validate(surface)
        clouds: list[CloudObject] = []
        for index, symbol in enumerate(validation.characters):
            object_id = f"symbol:{morpheme_id or surface}:{index}:{symbol}"
            cloud = CloudObject(
                object_id=object_id,
                label=symbol,
                dimensions={
                    "symbol": symbol,
                    "kind": "vowel" if symbol in self.vowels else "consonant",
                    "hardness": "soft_marker"
                    if symbol == "ь"
                    else "hard_marker"
                    if symbol == "ъ"
                    else "neutral",
                    "position": index,
                    "transition_frequency": 1.0,
                    "morpheme_role": morpheme_id,
                    "allowed_next": [validation.characters[index + 1]]
                    if index + 1 < len(validation.characters)
                    else [],
                },
                density=validation.score,
                halo=0.1,
                links={"up:morpheme_space": [morpheme_id] if morpheme_id else []},
                provenance={"source": "symbol_factory", "validation_score": validation.score},
            )
            space.register(cloud)
            clouds.append(cloud)
        return clouds
