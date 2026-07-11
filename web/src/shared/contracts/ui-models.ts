import type {
  AnswerDto,
  ArtifactRefDto,
  BudgetDto,
  ClaimDto,
  ConceptDto,
  ErrorDto,
  HiveViewDto,
  SystemSnapshotDto,
  TaskViewDto,
  TraceDto,
  TraceSpanDto,
} from './transport'

export interface UiError {
  code: string
  message: string
  retryable: boolean
  details?: Record<string, unknown>
}

export interface UiSource {
  artifactId: string
  label: string
  accessScope: string
  contentHash: string
}

export interface UiBudget {
  timeLimitMs: number
  eventLimit: number
  stepLimit: number
  memoryBytes: number
}

export interface UiAnswer {
  text: string
  sources: UiSource[]
  format: string
  verified: boolean
  warnings: string[]
}

export interface UiTask {
  id: string
  tenantId?: string
  conversationId?: string
  projectId?: string
  hiveId?: string
  traceId: string
  status: TaskViewDto['status']
  createdAt: string
  updatedAt: string
  budget?: UiBudget
  answer?: UiAnswer
  error?: UiError
}

export interface UiTraceSpan {
  id: string
  traceId: string
  parentId?: string
  causationId?: string
  component: string
  operation: string
  kind: 'span'
  status: TraceSpanDto['status']
  startedAt: string
  endedAt?: string
  durationMs?: number
  input?: Record<string, unknown>
  output?: Record<string, unknown>
  budgetBefore?: UiBudget
  budgetAfter?: UiBudget
  artifacts: Array<{
    id: string
    label: string
    mediaType: string
  }>
  error?: UiError
}

export interface UiTraceEvent {
  id: string
  sequence?: number
  kind: string
  producer: string
  occurredAt: string
  causationId?: string
  correlationId?: string
  payload: Record<string, unknown>
}

export interface UiTrace {
  id: string
  taskId?: string
  hiveId?: string
  status: UiTraceSpan['status'] | 'unknown'
  startedAt?: string
  endedAt?: string
  spans: UiTraceSpan[]
  events: UiTraceEvent[]
}

export interface UiHiveStore {
  storeId: string
  label: string
  itemCount: number
  sizeBytes: number
  protectedCount: number
}

export interface UiEvictedItem {
  entryId: string
  summary: string
  reasonCode: string
  destination: 'warm'
  occurredAt: string
}

export interface UiHiveSnapshot {
  snapshotId: string
  restorable: boolean
}

export interface UiHive {
  id: string
  taskId: string
  state: HiveViewDto['state']
  contract: {
    goal: string
    constraints: string[]
    protectedContextRefs: string[]
    revision: number
  }
  goals: string[]
  hotMemory: {
    usedBytes: number
    limitBytes: number
    activeItems: number
    utilization: number
  }
  stores: UiHiveStore[]
  evictedItems: UiEvictedItem[]
  snapshots: UiHiveSnapshot[]
  topics: string[]
}

export interface UiArtifact {
  id: string
  label: string
  mediaType: string
  schema: string
  version: string
  contentHash: string
  sizeBytes: number
  accessScope: string
  tenantId: string
  createdAt: string
}

export interface UiCosmosClaim {
  id: string
  subject: string
  predicate: string
  object: string
  verificationStatus: ClaimDto['verification_status']
  sourceArtifactId: string
  sourceFragment: string
  accessScope: string
  scores: ClaimDto['scores']
  validFrom?: string
}

export interface UiCosmosConcept {
  id: string
  label: string
  type: string
  sectors: string[]
  aliases: string[]
  claims: UiCosmosClaim[]
  neighbours: Array<{
    conceptId: string
    label: string
    relation: string
  }>
}

export interface UiSystem {
  activeTasks: number
  queuedWorkItems: number
  deadLetters: number
  lastErrors: UiError[]
  health: {
    status: string
    dependencies: Array<{
      name: string
      status: string
      detail?: string
    }>
  }
  meta: {
    apiVersion: string
    backendVersion: string
    schemaVersion: string
    build: string
    capabilities: string[]
  }
}

export function toUiError(error: ErrorDto): UiError {
  return {
    code: error.code,
    message: error.message,
    retryable: error.retryable,
    details: error.details,
  }
}

export function toUiBudget(budget: BudgetDto): UiBudget {
  return {
    timeLimitMs: budget.time_ms,
    eventLimit: budget.event_limit,
    stepLimit: budget.step_limit,
    memoryBytes: budget.memory_bytes,
  }
}

function toUiSource(source: ArtifactRefDto): UiSource {
  return {
    artifactId: source.artifact_id,
    label: source.schema_name + ' · ' + source.artifact_id,
    accessScope: source.access_scope.visibility + ':' + source.access_scope.tenant_id,
    contentHash: source.content_hash,
  }
}

function toUiAnswer(answer: AnswerDto, format: string): UiAnswer {
  return {
    text: answer.answer,
    sources: answer.sources.map(toUiSource),
    format,
    verified: answer.critic_reports.every((report) => report.verdict !== 'fail'),
    warnings: answer.warnings,
  }
}

export function toUiTask(task: TaskViewDto): UiTask {
  return {
    id: task.task_id,
    tenantId: task.contract?.tenant_id,
    conversationId: task.contract?.conversation_id,
    projectId: task.contract?.project_id ?? undefined,
    hiveId: task.hive_id ?? task.answer?.hive_id,
    traceId: task.trace_id,
    status: task.status,
    createdAt: task.created_at,
    updatedAt: task.updated_at,
    budget: task.contract ? toUiBudget(task.contract.budget) : undefined,
    answer: task.answer
      ? toUiAnswer(task.answer, task.contract?.expected_output ?? 'text')
      : undefined,
    error: task.error ? toUiError(task.error) : undefined,
  }
}

function toUiTraceSpan(span: TraceSpanDto): UiTraceSpan {
  const artifacts = [span.input_ref, span.output_ref]
    .filter((artifact): artifact is ArtifactRefDto => Boolean(artifact))
    .map((artifact) => ({
      id: artifact.artifact_id,
      label: artifact.schema_name + ' · ' + artifact.artifact_id,
      mediaType: artifact.media_type,
    }))
  return {
    id: span.span_id,
    traceId: span.trace_id,
    parentId: span.parent_span_id ?? undefined,
    causationId: span.causation_id ?? undefined,
    component: span.component,
    operation: span.operation,
    kind: 'span',
    status: span.status,
    startedAt: span.started_at,
    endedAt: span.ended_at ?? undefined,
    durationMs: span.duration_ms ?? undefined,
    input: span.input_summary,
    output: span.output_summary,
    budgetBefore: span.budget_before ? toUiBudget(span.budget_before) : undefined,
    budgetAfter: span.budget_after ? toUiBudget(span.budget_after) : undefined,
    artifacts,
    error: span.error ? toUiError(span.error) : undefined,
  }
}

function traceStatus(trace: TraceDto): UiTrace['status'] {
  if (trace.spans.some((span) => span.status === 'failed')) {
    return 'failed'
  }
  if (trace.spans.some((span) => span.status === 'cancelled')) {
    return 'cancelled'
  }
  if (trace.spans.some((span) => span.status === 'running')) {
    return 'running'
  }
  return trace.spans.length ? 'succeeded' : 'unknown'
}

export function toUiTrace(trace: TraceDto, task?: UiTask): UiTrace {
  return {
    id: trace.trace_id,
    taskId: task?.id ?? trace.events[0]?.task_id,
    hiveId: task?.hiveId,
    status: traceStatus(trace),
    startedAt: trace.spans[0]?.started_at,
    endedAt: trace.spans[trace.spans.length - 1]?.ended_at ?? undefined,
    spans: trace.spans.map(toUiTraceSpan),
    events: trace.events.map((event) => ({
      id: event.event_id ?? event.id,
      sequence: event.sequence,
      kind: event.kind,
      producer: event.producer,
      occurredAt: event.occurred_at,
      causationId: event.causation_id ?? undefined,
      correlationId: event.correlation_id ?? undefined,
      payload: event.payload,
    })),
  }
}

export function toUiHive(hive: HiveViewDto): UiHive {
  const storeMap = new Map<string, UiHiveStore>()
  for (const entry of hive.entries) {
    const current = storeMap.get(entry.store_name) ?? {
      storeId: entry.store_name,
      label: entry.store_name,
      itemCount: 0,
      sizeBytes: 0,
      protectedCount: 0,
    }
    current.itemCount += 1
    current.sizeBytes += entry.size
    current.protectedCount += entry.protected ? 1 : 0
    storeMap.set(entry.store_name, current)
  }
  const hotEntries = hive.entries.filter((entry) => entry.layer === 'hot')
  const hotBytes = hotEntries.reduce((total, entry) => total + entry.size, 0)
  const memoryBytes = hive.contract.budget.memory_bytes
  const stateGoals = hive.state_data.goals
  const goals = Array.isArray(stateGoals) && stateGoals.every((goal) => typeof goal === 'string')
    ? stateGoals
    : [hive.contract.goal]

  return {
    id: hive.hive_id,
    taskId: hive.contract.task_id,
    state: hive.state,
    contract: {
      goal: hive.contract.goal,
      constraints: hive.contract.constraints,
      protectedContextRefs: hive.contract.protected_context_refs,
      revision: hive.contract.revision,
    },
    goals,
    hotMemory: {
      usedBytes: hotBytes,
      limitBytes: memoryBytes,
      activeItems: hotEntries.length,
      utilization: memoryBytes ? Math.round((hotBytes / memoryBytes) * 100) : 0,
    },
    stores: [...storeMap.values()],
    evictedItems: hive.entries
      .filter((entry) => entry.layer === 'warm')
      .map((entry) => ({
        entryId: entry.entry_id,
        summary: entry.content_type,
        reasonCode: 'warm_context',
        destination: 'warm' as const,
        occurredAt: entry.created_at,
      })),
    snapshots: hive.snapshot_id
      ? [{ snapshotId: hive.snapshot_id, restorable: true }]
      : [],
    topics: hive.topics,
  }
}

export function toUiArtifact(artifact: ArtifactRefDto): UiArtifact {
  return {
    id: artifact.artifact_id,
    label: artifact.schema_name,
    mediaType: artifact.media_type,
    schema: artifact.schema_name,
    version: artifact.schema_version,
    contentHash: artifact.content_hash,
    sizeBytes: artifact.size,
    accessScope: artifact.access_scope.visibility + ':' + artifact.access_scope.tenant_id,
    tenantId: artifact.tenant_id,
    createdAt: artifact.created_at,
  }
}

export function toUiCosmosConcepts(
  concepts: ConceptDto[],
  claims: ClaimDto[],
): UiCosmosConcept[] {
  return concepts.map((concept) => {
    const conceptClaims = claims
      .filter((claim) => claim.subject_id === concept.concept_id)
      .map((claim) => ({
        id: claim.claim_id,
        subject: concept.label,
        predicate: claim.predicate,
        object: claim.object_value,
        verificationStatus: claim.verification_status,
        sourceArtifactId: claim.source_artifact_id,
        sourceFragment: claim.source_fragment,
        accessScope: claim.access_scope.visibility + ':' + claim.access_scope.tenant_id,
        scores: claim.scores,
        validFrom: claim.valid_from ?? undefined,
      }))
    return {
      id: concept.concept_id,
      label: concept.label,
      type: concept.concept_type,
      sectors: [...new Set(conceptClaims.flatMap((claim) => {
        const matching = claims.find((candidate) => candidate.claim_id === claim.id)
        return matching?.sectors ?? []
      }))],
      aliases: concept.aliases,
      claims: conceptClaims,
      neighbours: [],
    }
  })
}

export function toUiSystem(snapshot: SystemSnapshotDto): UiSystem {
  const workItems = snapshot.health.work_items ?? {}
  return {
    activeTasks: snapshot.active_tasks,
    queuedWorkItems: snapshot.queued_work_items,
    deadLetters: snapshot.dead_letters,
    lastErrors: snapshot.last_errors.map(toUiError),
    health: {
      status: snapshot.health.status,
      dependencies: [
        {
          name: 'runtime',
          status: snapshot.health.status,
          detail: snapshot.health.runtime,
        },
        ...Object.entries(workItems).map(([name, count]) => ({
          name: 'work item: ' + name,
          status: 'ok',
          detail: String(count),
        })),
      ],
    },
    meta: {
      apiVersion: snapshot.meta.api_version,
      backendVersion: snapshot.meta.service,
      schemaVersion: snapshot.meta.schema_version,
      build: snapshot.meta.runtime,
      capabilities: snapshot.meta.capabilities,
    },
  }
}
