from __future__ import annotations

from typing import Any

from semantic_ants.learning.checkpoint import Checkpoint


class SenseSentenceBuilder:
    def concept_explanation(
        self,
        target_token: str | None,
        activated_concepts: list[dict[str, object]],
        checkpoint: Checkpoint,
    ) -> str | None:
        return None

    def imagination_response(
        self,
        tokens: list[str],
        activated_concepts: list[dict[str, object]],
        checkpoint: Checkpoint,
    ) -> str | None:
        return None

    def sentence_response(
        self,
        tokens: list[str],
        activated_concepts: list[dict[str, object]],
        checkpoint: Checkpoint,
    ) -> str | None:
        return None

    def question_response(
        self,
        tokens: list[str],
        activated_concepts: list[dict[str, object]],
        checkpoint: Checkpoint,
    ) -> str | None:
        return None

    def meaningful_response(
        self,
        tokens: list[str],
        activated_concepts: list[dict[str, object]],
        checkpoint: Checkpoint,
    ) -> str | None:
        return None

    def _content_infos(
        self,
        tokens: list[str],
        activated_concepts: list[dict[str, object]],
        checkpoint: Checkpoint,
        limit: int,
    ) -> list[dict[str, Any]]:
        return []
