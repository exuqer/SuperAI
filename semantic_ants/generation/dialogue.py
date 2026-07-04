from __future__ import annotations

from semantic_ants.generation.torch_dialogue import TorchDialogueNavigator
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
    ) -> str | None:
        prompt = self.navigator.build_prompt(input_text, tokens, activated_concepts, [], checkpoint)
        candidates = self.navigator.generate(prompt, checkpoint, count=1)
        return candidates[0] if candidates else None
