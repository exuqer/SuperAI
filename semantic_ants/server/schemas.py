from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str
    lang: str = "auto"
    ants: Optional[int] = None
    depth: Optional[int] = None
    top_concepts: Optional[int] = None
    mode: str = "graph"
    candidates: int = 3
    session_id: Optional[str] = None
    reset_session: bool = False
    strength_vector: Optional[Union[list[int], str, int]] = None


class UnderstandRequest(BaseModel):
    text: str
    lang: str = "auto"
    session_id: Optional[str] = None
    turn_id: Optional[str] = None


class DecodeRequest(BaseModel):
    text: str
    tokens: list[str] = Field(default_factory=list)
    lang: str = "auto"
    session_id: Optional[str] = None
    turn_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    result_id: Optional[str] = None
    score: int = Field(ge=0, le=5)
    corrected_concepts: Optional[list[str]] = None
    corrected_response: Optional[str] = None


class VectorInterpretRequest(BaseModel):
    semantic_vector: Union[dict[str, Any], list[dict[str, Any]]]


class JsonlJobRequest(BaseModel):
    path: Optional[str] = None
    jsonl: Optional[str] = None
    epochs: int = 1
    batch_size: int = 32
    max_examples: Optional[int] = None
    torch_steps: int = 1


class ConceptMeaningRequest(BaseModel):
    concept: Optional[str] = None
    label: Optional[str] = None
    meaning: str


class SimpleTrainingRequest(BaseModel):
    question: str
    expected_answer: str
    lang: str = "auto"
    concept_meanings: list[ConceptMeaningRequest] = Field(default_factory=list)
    reward: float = 1.0
    epochs: int = 1


class EvalRequest(BaseModel):
    path: Optional[str] = None
    jsonl: Optional[str] = None


class DreamRequest(BaseModel):
    steps: int = 100


class BootstrapRequest(BaseModel):
    force: bool = False


class ResetNetworkRequest(BaseModel):
    keep_builtin: bool = True


class SpcDownloadRequest(BaseModel):
    split: str = "train"
    limit: Optional[int] = None
    output: Optional[str] = None


class KozievDownloadRequest(BaseModel):
    path: Optional[str] = None
    limit: Optional[int] = None
    timeout: float = 60.0
    output: Optional[str] = None


class TatoebaDownloadRequest(BaseModel):
    source_lang: str = "ru"
    target_lang: str = "en"
    bidirectional: bool = True
    limit: Optional[int] = None
    timeout: float = 60.0
    output: Optional[str] = None


class ExportRequest(BaseModel):
    destination: str


class ResonanceResetRequest(BaseModel):
    seed: bool = False


class ResonanceSeedRequest(BaseModel):
    force: bool = False
    session_id: str = "default"


class ResonanceTrainFormRequest(BaseModel):
    lang: str = "ru"
    lemma: str
    surface: str
    pos: str = "NOUN"
    gram: dict[str, Any] = Field(default_factory=dict)
    role: Optional[str] = None
    plane_id: Optional[str] = None
    reward: float = 1.0


class ResonanceSlotRequest(BaseModel):
    id: Optional[str] = None
    lemma: str
    role: str = "subject"
    pos: str = "NOUN"
    gram: dict[str, Any] = Field(default_factory=dict)
    preposition: Optional[str] = None


class ResonanceTrainSentenceRequest(BaseModel):
    lang: str = "ru"
    sentence: str
    slots: list[ResonanceSlotRequest] = Field(default_factory=list)
    plane_id: Optional[str] = None
    reward: float = 1.0


class ResonanceQaAnnotationRequest(BaseModel):
    index: Optional[int] = None
    token: Optional[str] = None
    surface: Optional[str] = None
    lemma: Optional[str] = None
    role: Optional[str] = None
    pos: Optional[str] = None
    concept: Optional[str] = None
    concept_uri: Optional[str] = None
    planes: list[str] = Field(default_factory=list)
    plane_id: Optional[str] = None
    gram: dict[str, Any] = Field(default_factory=dict)
    preposition: Optional[str] = None


class ResonanceTrainQaRequest(BaseModel):
    question: str
    expected_answer: str
    lang: str = "ru"
    session_id: str = "default"
    plane_id: Optional[str] = None
    annotations: list[ResonanceQaAnnotationRequest] = Field(default_factory=list)
    reward: float = 1.0
    epochs: int = 1


class ResonanceGenerateRequest(BaseModel):
    text: str = ""
    lang: str = "ru"
    session_id: str = "default"
    plane_id: Optional[str] = None
    slots: list[ResonanceSlotRequest] = Field(default_factory=list)
    ants: int = 16
    creativity: float = 0.35


class ResonanceFeedbackRequest(BaseModel):
    result_id: Optional[str] = None
    session_id: str = "default"
    score: int = Field(ge=0, le=5)
    corrected_sentence: Optional[str] = None
    corrected_tokens: list[ResonanceTrainFormRequest] = Field(default_factory=list)


def model_payload(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
