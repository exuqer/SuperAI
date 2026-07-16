from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from server.spaces import CloudObject, MorphemeSpace

from .symbol_factory import SymbolFactory


@dataclass(frozen=True)
class MorphologyResult:
    surface: str
    root: str
    prefix: str
    suffixes: tuple[str, ...]
    ending: str
    model: str
    confidence: float
    status: str
    symbol_validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MorphologyFactory:
    def __init__(self, symbol_factory: SymbolFactory | None = None) -> None:
        self.symbol_factory = symbol_factory or SymbolFactory()

    def compose(
        self,
        root: str,
        *,
        prefix: str = "",
        suffixes: Iterable[str] = (),
        ending: str = "",
        features: Mapping[str, Any] | None = None,
        model: str = "explicit",
    ) -> MorphologyResult:
        normalized_root = root.casefold().strip()
        resolved_suffixes = tuple(str(item).casefold() for item in suffixes)
        resolved_ending = ending.casefold()
        resolved_model = model
        features = dict(features or {})
        if not resolved_suffixes and not resolved_ending:
            resolved_suffixes, resolved_ending, resolved_model = self._infer(
                normalized_root, features
            )
        surface = (
            f"{prefix.casefold()}{normalized_root}{''.join(resolved_suffixes)}{resolved_ending}"
        )
        validation = self.symbol_factory.validate(surface)
        confidence = min(0.94, validation.score * (0.88 if resolved_model != "explicit" else 0.94))
        return MorphologyResult(
            surface=surface,
            root=normalized_root,
            prefix=prefix.casefold(),
            suffixes=resolved_suffixes,
            ending=resolved_ending,
            model=resolved_model,
            confidence=round(confidence, 6),
            status="COMPOSED" if validation.valid else "UNVERIFIED",
            symbol_validation=validation.to_dict(),
        )

    def compose_diminutive_plural(self, root: str) -> MorphologyResult:
        return self.compose(root, suffixes=("ик",), ending="и", model="diminutive_plural")

    def register(
        self, result: MorphologyResult, space: MorphemeSpace, word_id: str
    ) -> list[CloudObject]:
        parts = []
        if result.prefix:
            parts.append(("prefix", result.prefix))
        parts.append(("root", result.root))
        parts.extend(("suffix", suffix) for suffix in result.suffixes)
        if result.ending:
            parts.append(("ending", result.ending))
        clouds: list[CloudObject] = []
        for index, (morpheme_type, surface) in enumerate(parts):
            object_id = f"morpheme:{word_id}:{index}:{surface}"
            cloud = CloudObject(
                object_id=object_id,
                label=surface,
                dimensions={
                    "surface": surface,
                    "morpheme_type": morpheme_type,
                    "grammatical_function": result.model
                    if morpheme_type != "root"
                    else "lexical_identity",
                    "semantic_effect": "diminutive" if surface == "ик" else "identity",
                    "position": index,
                    "compatibility": [parts[index - 1][1]] if index else [],
                    "formation_model": result.model,
                },
                density=result.confidence,
                halo=0.16,
                links={"up:word_space": [word_id]},
                provenance={"source": "morphology_factory", "status": result.status},
            )
            space.register(cloud)
            clouds.append(cloud)
        return clouds

    @staticmethod
    def _infer(root: str, features: Mapping[str, Any]) -> tuple[tuple[str, ...], str, str]:
        number = str(features.get("number") or "").casefold()
        diminutive = bool(
            features.get("diminutive") or features.get("semantic_effect") == "diminutive"
        )
        if diminutive and number in {"plur", "plural", "мн", "multiple"}:
            return ("ик",), "и", "diminutive_plural"
        if diminutive:
            return ("ик",), "", "diminutive"
        if number in {"plur", "plural", "мн", "multiple"}:
            if root.endswith(("к", "г", "х")):
                return (), "и", "noun_plural_i"
            return (), "ы", "noun_plural_y"
        return (), "", "identity"
