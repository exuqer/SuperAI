"""Versioned transport and domain contracts shared by every subsystem.

The models in this module deliberately carry scope and correlation metadata.
They are the stable boundary between the API, runtime and diagnostic client.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_MAJOR = "1"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return "%s_%s" % (prefix, uuid4().hex)


class SchemaModel(BaseModel):
    """Base model which accepts forward-compatible fields at a boundary."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, from_attributes=True)
    schema_version: str = SCHEMA_VERSION

    @field_validator("schema_version")
    @classmethod
    def schema_major_is_supported(cls, value: str) -> str:
        if value.split(".", 1)[0] != SUPPORTED_SCHEMA_MAJOR:
            raise ValueError("Unsupported schema major version: %s" % value)
        return value


class Budget(SchemaModel):
    time_ms: int = Field(default=30_000, ge=1, le=3_600_000)
    step_limit: int = Field(default=64, ge=1, le=100_000)
    memory_bytes: int = Field(default=262_144, ge=1_024)
    event_limit: int = Field(default=256, ge=1, le=100_000)


class AccessScope(SchemaModel):
    """A deliberately small access model for the local MVP.

    ``tenant`` data is visible only to the matching tenant. ``project`` is an
    additional boundary when supplied; ``global`` records are explicitly
    publishable and never inferred from frequency.
    """

    tenant_id: str = "local"
    project_id: Optional[str] = None
    visibility: str = Field(default="tenant", pattern="^(tenant|project|global)$")
    retention: str = "standard"

    @model_validator(mode="after")
    def project_visibility_requires_project(self) -> "AccessScope":
        if self.visibility == "project" and not self.project_id:
            raise ValueError("project visibility requires project_id")
        return self


class ArtifactRef(SchemaModel):
    artifact_id: str
    content_hash: str
    media_type: str
    schema_name: str
    schema_version: str = SCHEMA_VERSION
    size: int = Field(ge=0)
    tenant_id: str
    created_at: datetime
    access_scope: AccessScope


class ErrorEnvelope(SchemaModel):
    code: str
    message: str
    retryable: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=utcnow)


class Envelope(SchemaModel):
    id: str
    occurred_at: datetime = Field(default_factory=utcnow)
    tenant_id: str = "local"
    task_id: str
    trace_id: str
    kind: str
    producer: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    artifact_ref: Optional[ArtifactRef] = None
    causation_id: Optional[str] = None
    correlation_id: Optional[str] = None


class CommandEnvelope(Envelope):
    idempotency_key: str


class DomainEvent(Envelope):
    event_id: str = Field(default_factory=lambda: new_id("evt"))


class SpanStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class TraceSpan(SchemaModel):
    span_id: str = Field(default_factory=lambda: new_id("span"))
    trace_id: str
    parent_span_id: Optional[str] = None
    causation_id: Optional[str] = None
    component: str
    operation: str
    status: SpanStatus = SpanStatus.RUNNING
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    input_ref: Optional[ArtifactRef] = None
    output_ref: Optional[ArtifactRef] = None
    input_summary: Dict[str, Any] = Field(default_factory=dict)
    output_summary: Dict[str, Any] = Field(default_factory=dict)
    budget_before: Optional[Budget] = None
    budget_after: Optional[Budget] = None
    error: Optional[ErrorEnvelope] = None


class CriticVerdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class CriticReport(SchemaModel):
    report_id: str = Field(default_factory=lambda: new_id("critic"))
    critic: str
    target: str
    verdict: CriticVerdict
    severity: str = Field(default="info", pattern="^(info|warning|error)$")
    evidence: List[str] = Field(default_factory=list)
    repair_hint: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)


class ModuleSnapshot(SchemaModel):
    snapshot_id: str = Field(default_factory=lambda: new_id("snap"))
    aggregate_type: str
    aggregate_id: str
    sequence: int
    artifact_ref: ArtifactRef
    state_hash: str
    created_at: datetime = Field(default_factory=utcnow)


class TaskContract(SchemaModel):
    task_id: str = Field(default_factory=lambda: new_id("task"))
    revision: int = Field(default=1, ge=1)
    tenant_id: str = "local"
    user_id: Optional[str] = None
    conversation_id: str = Field(default_factory=lambda: new_id("conv"))
    project_id: Optional[str] = None
    goal: str = Field(min_length=1, max_length=20_000)
    inputs: List[ArtifactRef] = Field(default_factory=list)
    expected_output: str = "text"
    constraints: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    risk_level: str = Field(default="low", pattern="^(low|medium|high)$")
    source_policy: str = Field(default="allowed_sources_only")
    budget: Budget = Field(default_factory=Budget)
    protected_context_refs: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class AnswerEnvelope(SchemaModel):
    task_id: str
    trace_id: str
    hive_id: str
    status: str = "completed"
    answer: str
    sources: List[ArtifactRef] = Field(default_factory=list)
    critic_reports: List[CriticReport] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class TaskState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEAD_LETTER = "dead_letter"


class HiveState(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    FROZEN = "frozen"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    FAILED = "failed"


class ContextEntry(SchemaModel):
    entry_id: str = Field(default_factory=lambda: new_id("ctx"))
    hive_id: str
    store_name: str
    layer: str = Field(default="hot", pattern="^(hot|warm|cold)$")
    content: Dict[str, Any]
    content_type: str
    size: int = Field(ge=0)
    source_ref: Optional[str] = None
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    protected: bool = False
    reconstruction_cost: float = Field(default=0.5, ge=0.0)
    expiry_policy: str = "until_hive_complete"
    created_at: datetime = Field(default_factory=utcnow)


class EvictionDecision(SchemaModel):
    eviction_id: str = Field(default_factory=lambda: new_id("evict"))
    hive_id: str
    entry_id: str
    reason_code: str
    score_before: float
    score_after: float
    created_at: datetime = Field(default_factory=utcnow)


class HiveView(SchemaModel):
    hive_id: str
    tenant_id: str
    conversation_id: str
    project_id: Optional[str] = None
    state: HiveState
    contract: TaskContract
    topics: List[str] = Field(default_factory=list)
    state_data: Dict[str, Any] = Field(default_factory=dict)
    entries: List[ContextEntry] = Field(default_factory=list)
    snapshot_id: Optional[str] = None
    version: int = 1
    created_at: datetime
    updated_at: datetime


class Concept(SchemaModel):
    concept_id: str = Field(default_factory=lambda: new_id("concept"))
    label: str
    concept_type: str = "term"
    aliases: List[str] = Field(default_factory=list)
    tenant_id: str = "local"
    created_at: datetime = Field(default_factory=utcnow)


class ClaimScores(SchemaModel):
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    utility: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness: float = Field(default=1.0, ge=0.0, le=1.0)
    contradiction: float = Field(default=0.0, ge=0.0, le=1.0)
    use_cost: float = Field(default=0.0, ge=0.0, le=1.0)


class Claim(SchemaModel):
    claim_id: str = Field(default_factory=lambda: new_id("claim"))
    tenant_id: str = "local"
    subject_id: str
    predicate: str
    object_value: str
    source_id: str
    source_artifact_id: str
    source_fragment: str
    sectors: List[str] = Field(default_factory=list)
    access_scope: AccessScope
    verification_status: str = Field(default="unverified", pattern="^(unverified|reviewed|verified|rejected|hypothesis)$")
    scores: ClaimScores = Field(default_factory=ClaimScores)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)


class RetrievedClaim(SchemaModel):
    claim: Claim
    subject_label: str
    source: ArtifactRef
    score: float
    reasons: List[str] = Field(default_factory=list)
    contradictory_claim_ids: List[str] = Field(default_factory=list)


class RetrievalResult(SchemaModel):
    claims: List[RetrievedClaim] = Field(default_factory=list)
    budget_used: int = 0
    gaps: List[str] = Field(default_factory=list)
    query_terms: List[str] = Field(default_factory=list)


class ImportReport(SchemaModel):
    source_id: str
    artifact: ArtifactRef
    status: str
    imported_claims: int = 0
    imported_concepts: int = 0
    duplicate: bool = False


class CapabilityManifest(SchemaModel):
    capability_id: str
    version: str = "1.0.0"
    kind: str = Field(pattern="^(factory|family|skill|critic|codec|bridge|retriever)$")
    input_schemas: List[str] = Field(default_factory=list)
    output_schemas: List[str] = Field(default_factory=list)
    supported_operations: List[str] = Field(default_factory=list)
    preconditions: List[str] = Field(default_factory=list)
    access_needs: List[str] = Field(default_factory=list)
    quality: float = Field(default=0.5, ge=0.0, le=1.0)
    estimated_latency_ms: int = Field(default=1, ge=0)
    estimated_cost: float = Field(default=0.0, ge=0.0)
    dependencies: List[str] = Field(default_factory=list)
    health: str = Field(default="healthy", pattern="^(healthy|degraded|unavailable)$")
    artifact_ref: Optional[ArtifactRef] = None
    created_at: datetime = Field(default_factory=utcnow)


class PlanStep(SchemaModel):
    step_id: str = Field(default_factory=lambda: new_id("step"))
    operation: str
    capability_id: str
    input_schema: str
    output_schema: str
    estimated_cost: float = 0.0
    side_effects: List[str] = Field(default_factory=list)
    on_error: str = "fail"


class ExecutionPlan(SchemaModel):
    plan_id: str = Field(default_factory=lambda: new_id("plan"))
    task_id: str
    hive_id: str
    revision: int = 1
    steps: List[PlanStep] = Field(default_factory=list)
    status: str = Field(default="validated", pattern="^(draft|validated|running|completed|failed)$")
    created_at: datetime = Field(default_factory=utcnow)


class CompostRecord(SchemaModel):
    compost_id: str = Field(default_factory=lambda: new_id("compost"))
    tenant_id: str = "local"
    artifact_ref: ArtifactRef
    trace_id: str
    access_scope: AccessScope
    status: str = Field(default="candidate", pattern="^(candidate|validated|integrated|rejected|deleted)$")
    created_at: datetime = Field(default_factory=utcnow)


class SkillState(str, Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    SHADOW = "shadow"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class SkillManifest(SchemaModel):
    skill_id: str = Field(default_factory=lambda: new_id("skill"))
    version: str = "1.0.0"
    tenant_id: str = "local"
    access_scope: AccessScope = Field(default_factory=AccessScope)
    state: SkillState = SkillState.CANDIDATE
    task_class: str
    procedure: List[PlanStep]
    preconditions: List[str] = Field(default_factory=list)
    train_task_ids: List[str] = Field(default_factory=list)
    holdout_task_ids: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    compiler_version: str = "1.0"
    provenance_trace_ids: List[str] = Field(default_factory=list)
    rollback_version: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def skill_scope_matches_owner(self) -> "SkillManifest":
        if self.access_scope.visibility != "global" and self.access_scope.tenant_id != self.tenant_id:
            raise ValueError("skill access scope tenant must match skill owner")
        return self


class GenomeManifest(SchemaModel):
    genome_id: str = Field(default_factory=lambda: new_id("genome"))
    version: str = "1.0.0"
    component_type: str
    module_refs: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    interface_schema: str
    parent_refs: List[str] = Field(default_factory=list)
    derivation_recipe: str = "initial"
    operator_versions: Dict[str, str] = Field(default_factory=dict)
    evaluation_refs: List[str] = Field(default_factory=list)
    content_hash: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)


class WorkItem(SchemaModel):
    command_id: str = Field(default_factory=lambda: new_id("cmd"))
    task_id: str
    trace_id: str
    handler: str
    payload: Dict[str, Any]
    status: TaskState = TaskState.QUEUED
    attempt: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    priority: int = Field(default=100, ge=0)
    scheduled_at: datetime = Field(default_factory=utcnow)
    idempotency_key: str
    budget: Budget = Field(default_factory=Budget)
    deadline_at: Optional[datetime] = None
    last_error: Optional[ErrorEnvelope] = None
    tenant_id: str = "local"


class TaskSubmission(SchemaModel):
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: Optional[str] = None
    project_id: Optional[str] = None
    tenant_id: str = "local"
    user_id: Optional[str] = None
    budget: Budget = Field(default_factory=Budget)
    source_policy: str = "allowed_sources_only"


class TaskView(SchemaModel):
    task_id: str
    trace_id: str
    hive_id: Optional[str] = None
    status: TaskState
    contract: Optional[TaskContract] = None
    answer: Optional[AnswerEnvelope] = None
    error: Optional[ErrorEnvelope] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ============================================================
# NEW CONTRACTS FOR ΩE STAGES
# ============================================================

class NodeType(str, Enum):
    CONCEPT = "concept"
    CLAIM = "claim"
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    PREDICTION = "prediction"
    PROCEDURE = "procedure"


class GraphNode(SchemaModel):
    node_id: str = Field(default_factory=lambda: new_id("node"))
    node_type: NodeType
    content_ref: str  # reference to claim, hypothesis, concept, etc.
    status: str = "active"  # active, inhibited, expired
    activation: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    utility: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance_refs: List[str] = Field(default_factory=list)
    version: int = 1


class EdgeType(str, Enum):
    SEMANTIC = "semantic"
    EVIDENCE_FOR = "evidence_for"
    EVIDENCE_AGAINST = "evidence_against"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    PROCEDURE = "procedure"


class GraphEdge(SchemaModel):
    edge_id: str = Field(default_factory=lambda: new_id("edge"))
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    scope: str = "local"  # local, project, tenant, global
    provenance_refs: List[str] = Field(default_factory=list)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    version: int = 1


class ActiveGraphSnapshot(SchemaModel):
    task_id: str
    hive_id: str
    tenant_id: str = "local"
    project_id: Optional[str] = None
    cosmos_version: str
    node_ids: List[str] = Field(default_factory=list)
    edge_ids: List[str] = Field(default_factory=list)
    frontier: List[str] = Field(default_factory=list)  # node_ids at frontier
    event_count: int = 0
    budget_ledger: Dict[str, int] = Field(default_factory=dict)
    random_seed: int


class ResourceBudget(SchemaModel):
    max_steps: int = 100
    max_wall_time_ms: int = 30_000
    max_active_nodes: int = 50
    max_active_edges: int = 200
    max_hypotheses: int = 5
    max_experiments: int = 10
    exploration_share: float = Field(default=0.3, ge=0.0, le=1.0)


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    MERGED = "merged"
    FALSIFIED = "falsified"
    SELECTED = "selected"
    ARCHIVED = "archived"


class HypothesisRecord(SchemaModel):
    hypothesis_id: str = Field(default_factory=lambda: new_id("hyp"))
    task_id: str
    tenant_id: str = "local"
    project_id: Optional[str] = None
    family_id: str  # group of related hypotheses
    statement: str
    assumptions: List[str] = Field(default_factory=list)
    evidence_for: List[str] = Field(default_factory=list)  # refs to evidence
    evidence_against: List[str] = Field(default_factory=list)
    predictions: List[str] = Field(default_factory=list)  # refs to PredictionRecord
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    allocated_budget: int = 0
    spent_budget: int = 0
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    parent_ids: List[str] = Field(default_factory=list)
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class PredictionRecord(SchemaModel):
    prediction_id: str = Field(default_factory=lambda: new_id("pred"))
    hypothesis_id: str
    experiment_input: Dict[str, Any]
    expected_output: Dict[str, Any]
    tolerance: Dict[str, Any] = Field(default_factory=dict)


class ExperimentStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvidenceStatus(str, Enum):
    SIMULATED = "simulated"
    COMPUTED = "computed"
    OBSERVED = "observed"
    SOURCE_BACKED = "source_backed"
    VERIFIED = "verified"


class ExperimentRecord(SchemaModel):
    experiment_id: str = Field(default_factory=lambda: new_id("exp"))
    task_id: str
    tenant_id: str = "local"
    competing_hypothesis_ids: List[str]
    operation: str
    input: Dict[str, Any]
    expected_information_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    estimated_cost: int = 1
    risk: float = Field(default=0.0, ge=0.0, le=1.0)
    result_ref: Optional[str] = None
    evidence_status: EvidenceStatus = EvidenceStatus.SIMULATED
    status: ExperimentStatus = ExperimentStatus.PROPOSED
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class ConceptCandidateState(str, Enum):
    CANDIDATE = "candidate"
    VALIDATING = "validating"
    VALIDATED = "validated"
    SHADOW = "shadow"
    ACTIVE = "active"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class ConceptCandidate(SchemaModel):
    concept_id: str = Field(default_factory=lambda: new_id("concept_cand"))
    tenant_id: str = "local"
    project_id: Optional[str] = None
    name: str
    definition_ref: str  # reference to artifact with definition
    source_subgraph_refs: List[str] = Field(default_factory=list)
    positive_examples: List[Dict[str, Any]] = Field(default_factory=list)
    negative_examples: List[Dict[str, Any]] = Field(default_factory=list)
    train_task_ids: List[str] = Field(default_factory=list)
    holdout_manifest_ref: str  # reference to holdout dataset manifest
    metrics: Dict[str, float] = Field(default_factory=dict)
    state: ConceptCandidateState = ConceptCandidateState.CANDIDATE
    rollback_target: Optional[str] = None
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class ConceptEvaluation(SchemaModel):
    concept_id: str
    baseline_run_id: str
    treatment_run_id: str
    ablation_run_id: str
    quality_delta: float = 0.0
    cost_delta: float = 0.0
    transfer_delta: float = 0.0
    accepted: bool = False


class BenchmarkRun(SchemaModel):
    run_id: str = Field(default_factory=lambda: new_id("bench"))
    task_id: str
    tenant_id: str = "local"
    project_id: Optional[str] = None
    git_revision: str
    config_hash: str
    dataset_version: str
    seed: int
    latency_ms: float = 0.0
    cost: float = 0.0
    quality: float = 0.0
    status: str = "running"  # running, completed, failed
    mode: str = "baseline"  # baseline, treatment, holdout, ablation
    concept_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)




