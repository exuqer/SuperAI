from .base import MultidimensionalSpace, SpaceLevel


class ConceptSpace(MultidimensionalSpace):
    level = SpaceLevel.CONCEPT
    dimensions = (
        "themes",
        "abstractness",
        "actions",
        "properties",
        "similarity",
        "cross_domain",
        "stability",
        "scene_roles",
    )
    weights = {
        "themes": 1.2,
        "actions": 1.1,
        "properties": 1.0,
        "scene_roles": 1.1,
        "stability": 0.7,
    }
