from .base import MultidimensionalSpace, SpaceLevel


class MorphemeSpace(MultidimensionalSpace):
    level = SpaceLevel.MORPHEME
    dimensions = (
        "surface",
        "morpheme_type",
        "grammatical_function",
        "semantic_effect",
        "position",
        "compatibility",
        "formation_model",
    )
    weights = {
        "morpheme_type": 1.3,
        "grammatical_function": 1.2,
        "compatibility": 1.15,
        "position": 1.0,
        "formation_model": 1.0,
    }
