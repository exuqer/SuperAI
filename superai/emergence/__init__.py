"""Emergence module: active graph, hypotheses, causal reasoning, and concept genesis."""

from .graph import ActiveGraph, ActiveGraphBuilder, GraphEvent
from .hypotheses import HypothesisBoard, HypothesisGenerator, create_hypothesis_board
from .causal import (
    CausalEngine,
    ExperimentSelector,
    Sandbox,
    CausalEdge,
    InterventionResult,
    create_default_sandbox,
)
from .concepts import (
    SubgraphMiner,
    ConceptGenerator,
    ConceptValidator,
    ConceptLifecycleManager,
    SubgraphPattern,
    create_concept_pipeline,
)

__all__ = [
    "ActiveGraph",
    "ActiveGraphBuilder",
    "GraphEvent",
    "HypothesisBoard",
    "HypothesisGenerator",
    "create_hypothesis_board",
    "CausalEngine",
    "ExperimentSelector",
    "Sandbox",
    "CausalEdge",
    "InterventionResult",
    "create_default_sandbox",
    "SubgraphMiner",
    "ConceptGenerator",
    "ConceptValidator",
    "ConceptLifecycleManager",
    "SubgraphPattern",
    "create_concept_pipeline",
]