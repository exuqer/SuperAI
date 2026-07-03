from __future__ import annotations

from semantic_ants.core.models import ConceptNode, SemanticEdge


class FakeConceptNetClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def edges_for(self, uri: str, limit: int = 30):
        self.calls.append(uri)
        lang = uri.split("/")[2]
        token = uri.split("/")[-1]
        related = f"/c/{lang}/{token}_meaning"
        nodes = [
            ConceptNode(uri=uri, label=token, language=lang, source="fixture"),
            ConceptNode(uri=related, label=f"{token} meaning", language=lang, source="fixture"),
        ]
        edges = [
            SemanticEdge(
                start=uri,
                end=related,
                relation="RelatedTo",
                weight=2.0,
                source="fixture",
            )
        ]
        return nodes, edges
