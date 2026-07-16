from .base import MultidimensionalSpace, SpaceLevel


class SymbolSpace(MultidimensionalSpace):
    level = SpaceLevel.SYMBOL
    dimensions = (
        "symbol",
        "kind",
        "hardness",
        "position",
        "transition_frequency",
        "morpheme_role",
        "allowed_next",
    )
    weights = {
        "symbol": 1.4,
        "kind": 1.15,
        "position": 0.9,
        "allowed_next": 1.2,
        "morpheme_role": 0.8,
    }
