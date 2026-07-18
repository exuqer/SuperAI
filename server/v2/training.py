"""Public V2.7 training API."""

from .graph_service import GraphTrainingService
from .russian_morphology import Morphology, RussianMorphology


TrainingPipelineV2 = GraphTrainingService
TrainingPipeline = GraphTrainingService

__all__ = [
    "GraphTrainingService",
    "Morphology",
    "RussianMorphology",
    "TrainingPipeline",
    "TrainingPipelineV2",
]
