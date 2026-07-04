from __future__ import annotations

from semantic_ants.generation.torch_dialogue import TorchDialogueNavigator
from semantic_ants.core.normalization import detect_language
from semantic_ants.learning.checkpoint import Checkpoint


class DialogueResponder:
    def __init__(self) -> None:
        self.navigator = TorchDialogueNavigator()

    def response_for(
        self,
        input_text: str,
        tokens: list[str],
        activated_concepts: list[dict[str, object]],
        checkpoint: Checkpoint,
        chat_history: list[dict[str, object]] | None = None,
    ) -> str | None:
        lang = detect_language(input_text)
        prompt = self.navigator.build_prompt(
            input_text,
            tokens,
            activated_concepts,
            [],
            checkpoint,
            chat_history=chat_history,
            lang=lang,
        )
        candidates = self.navigator.generate(prompt, checkpoint, count=1, lang=lang)
        return candidates[0] if candidates else None
