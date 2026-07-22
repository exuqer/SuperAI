"""Public V3.0 training API."""

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
