from .base import MultidimensionalSpace, SpaceLevel


class WordSpace(MultidimensionalSpace):
    level = SpaceLevel.WORD
    dimensions = (
        "lemma",
        "surface",
        "part_of_speech",
        "grammatical_features",
        "style",
        "collocations",
        "concept",
        "role",
        "frequency",
        "morphological_similarity",
    )
    weights = {
        "concept": 1.35,
        "role": 1.25,
        "grammatical_features": 1.2,
        "part_of_speech": 1.0,
        "style": 0.65,
        "frequency": 0.55,
    }
