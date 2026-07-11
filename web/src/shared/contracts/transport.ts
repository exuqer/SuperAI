/**
 * DTOs at the real /api/v1 boundary. Their field names intentionally match
 * Pydantic's JSON output; conversion to the client-facing model happens in
 * ui-models.ts.
 */

export type SchemaVersion = string

export type TaskStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'dead_letter'

export type SpanStatus = 'running' | 'succeeded' | 'failed' | 'cancelled' | 'skipped'

export interface BudgetDto {
  schema_version: SchemaVersion
  time_ms: number
  step_limit: number
  memory_bytes: number
  event_limit: number
}

export interface AccessScopeDto {
  schema_version: SchemaVersion
  tenant_id: string
  project_id?: string | null
  visibility: 'tenant' | 'project' | 'global'
  retention: string
}

export interface ArtifactRefDto {
  artifact_id: string
  content_hash: string
  media_type: string
  schema_name: string
  schema_version: SchemaVersion
  size: number
  tenant_id: string
  created_at: string
  access_scope: AccessScopeDto
}

export interface ErrorDto {
  schema_version: SchemaVersion
  code: string
  message: string
  retryable: boolean
  details: Record<string, unknown>
  occurred_at: string
}

export interface CriticReportDto {
  schema_version: SchemaVersion
  report_id: string
  critic: string
  target: string
  verdict: 'pass' | 'warn' | 'fail'
  severity: 'info' | 'warning' | 'error'
  evidence: string[]
  repair_hint?: string | null
  created_at: string
}

export interface TaskContractDto {
  schema_version: SchemaVersion
  task_id: string
  revision: number
  tenant_id: string
  user_id?: string | null
  conversation_id: string
  project_id?: string | null
  goal: string
  inputs: ArtifactRefDto[]
  expected_output: string
  constraints: string[]
  success_criteria: string[]
  risk_level: 'low' | 'medium' | 'high'
  source_policy: string
  budget: BudgetDto
  protected_context_refs: string[]
  created_at: string
}

export interface TaskSubmissionDto {
  schema_version: SchemaVersion
  message: string
  conversation_id?: string
  project_id?: string
  tenant_id: string
  user_id?: string
  budget: BudgetDto
  source_policy?: string
}

export interface AnswerDto {
  schema_version: SchemaVersion
  task_id: string
  trace_id: string
  hive_id: string
  status: string
  answer: string
  sources: ArtifactRefDto[]
  critic_reports: CriticReportDto[]
  warnings: string[]
}

export interface TaskViewDto {
  schema_version: SchemaVersion
  task_id: string
  trace_id: string
  hive_id?: string | null
  status: TaskStatus
  contract?: TaskContractDto | null
  answer?: AnswerDto | null
  error?: ErrorDto | null
  created_at: string
  updated_at: string
}

export interface DomainEventDto {
  schema_version: SchemaVersion
  id: string
  event_id?: string
  occurred_at: string
  tenant_id: string
  task_id: string
  trace_id: string
  kind: string
  producer: string
  payload: Record<string, unknown>
  artifact_ref?: ArtifactRefDto | null
  causation_id?: string | null
  correlation_id?: string | null
  sequence?: number
}

export interface TraceSpanDto {
  schema_version: SchemaVersion
  span_id: string
  trace_id: string
  parent_span_id?: string | null
  causation_id?: string | null
  component: string
  operation: string
  status: SpanStatus
  started_at: string
  ended_at?: string | null
  duration_ms?: number | null
  input_ref?: ArtifactRefDto | null
  output_ref?: ArtifactRefDto | null
  input_summary: Record<string, unknown>
  output_summary: Record<string, unknown>
  budget_before?: BudgetDto | null
  budget_after?: BudgetDto | null
  error?: ErrorDto | null
}

export interface TraceDto {
  trace_id: string
  spans: TraceSpanDto[]
  events: DomainEventDto[]
}

export interface ContextEntryDto {
  schema_version: SchemaVersion
  entry_id: string
  hive_id: string
  store_name: string
  layer: 'hot' | 'warm' | 'cold'
  content: Record<string, unknown>
  content_type: string
  size: number
  source_ref?: string | null
  relevance: number
  protected: boolean
  reconstruction_cost: number
  expiry_policy: string
  created_at: string
}

export interface HiveViewDto {
  schema_version: SchemaVersion
  hive_id: string
  tenant_id: string
  conversation_id: string
  project_id?: string | null
  state: 'active' | 'idle' | 'frozen' | 'completed' | 'archived' | 'failed'
  contract: TaskContractDto
  topics: string[]
  state_data: Record<string, unknown>
  entries: ContextEntryDto[]
  snapshot_id?: string | null
  version: number
  created_at: string
  updated_at: string
}

export interface ConceptDto {
  schema_version: SchemaVersion
  concept_id: string
  label: string
  concept_type: string
  aliases: string[]
  tenant_id: string
  created_at: string
}

export interface ClaimScoresDto {
  schema_version: SchemaVersion
  confidence: number
  relevance: number
  utility: number
  freshness: number
  contradiction: number
  use_cost: number
}

export interface ClaimDto {
  schema_version: SchemaVersion
  claim_id: string
  tenant_id: string
  subject_id: string
  predicate: string
  object_value: string
  source_id: string
  source_artifact_id: string
  source_fragment: string
  sectors: string[]
  access_scope: AccessScopeDto
  verification_status: 'unverified' | 'reviewed' | 'verified' | 'rejected' | 'hypothesis'
  scores: ClaimScoresDto
  valid_from?: string | null
  valid_to?: string | null
  created_at: string
}

export interface HealthDto {
  status: 'ok' | 'degraded' | 'offline'
  runtime?: string
  data_dir?: string
  work_items?: Record<string, number>
}

export interface MetaDto {
  service: string
  api_version: string
  schema_version: SchemaVersion
  runtime: string
  capabilities: string[]
}

/**
 * A client-side composition of real health/meta/dead-letter responses. It is
 * not presented as an HTTP endpoint, so it remains separate from transport
 * DTOs above.
 */
export interface SystemSnapshotDto {
  active_tasks: number
  queued_work_items: number
  dead_letters: number
  last_errors: ErrorDto[]
  health: HealthDto
  meta: MetaDto
}
