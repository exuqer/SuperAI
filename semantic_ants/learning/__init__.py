from semantic_ants.learning.checkpoint import Checkpoint, CheckpointStore
from semantic_ants.learning.trainer import FeedbackTrainer, Trainer, TrainingReport

__all__ = [
    "ACOTrainer",
    "ACOTrainingReport",
    "Checkpoint",
    "CheckpointStore",
    "Experience",
    "FeedbackTrainer",
    "Judge",
    "RewardSignal",
    "SemanticThought",
    "SimpleQATrainer",
    "SimpleTrainingReport",
    "Trainer",
    "TrainingReport",
]


def __getattr__(name: str):
    if name in {"ACOTrainer", "ACOTrainingReport", "Experience", "Judge", "RewardSignal", "SemanticThought"}:
        from semantic_ants.learning import aco

        return getattr(aco, name)
    if name in {"SimpleQATrainer", "SimpleTrainingReport"}:
        from semantic_ants.learning import simple

        return getattr(simple, name)
    raise AttributeError(name)
