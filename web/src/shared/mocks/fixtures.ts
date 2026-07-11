import type {
  ArtifactRefDto,
  ClaimDto,
  ConceptDto,
  ErrorDto,
  HiveViewDto,
  SystemSnapshotDto,
  TaskSubmissionDto,
  TaskViewDto,
  TraceDto,
  TraceSpanDto,
} from '@/shared/contracts/transport'

export type FixtureScenarioId =
  | 'success'
  | 'validation'
  | 'timeout'
  | 'retry'
  | 'cancel'
  | 'incompatible'
  | 'offline'

export interface ScenarioFixture {
  id: FixtureScenarioId
  title: string
  description: string
  request: TaskSubmissionDto
  task?: TaskViewDto
  trace?: TraceDto
  hive?: HiveViewDto
  artifacts?: ArtifactRefDto[]
  concepts?: ConceptDto[]
  claims?: ClaimDto[]
  system: SystemSnapshotDto
  unavailable?: boolean
}

const schemaVersion = '1.0'
const tenantId = 'local'
const projectId = 'project-platformer'
const timestamp = '2026-07-11T09:15:00.000Z'

const scope = {
  schema_version: schemaVersion,
  tenant_id: tenantId,
  project_id: projectId,
  visibility: 'project' as const,
  retention: 'standard',
}

const budget = {
  schema_version: schemaVersion,
  time_ms: 4_000,
  step_limit: 32,
  memory_bytes: 16_384,
  event_limit: 20,
}

const request: TaskSubmissionDto = {
  schema_version: schemaVersion,
  message: 'Почему персонаж проходит сквозь стену в Unity 2D?',
  conversation_id: 'conversation-unity-001',
  project_id: projectId,
  tenant_id: tenantId,
  user_id: 'user-local',
  budget,
  source_policy: 'allowed_sources_only',
}

const inputArtifact: ArtifactRefDto = {
  artifact_id: 'artifact-user-message-001',
  content_hash: '540cdf0c0e9f2cf472ea0b54323fc258',
  media_type: 'text/plain',
  schema_name: 'InputMessage',
  schema_version: schemaVersion,
  size: 98,
  tenant_id: tenantId,
  created_at: timestamp,
  access_scope: scope,
}

const sourceArtifact: ArtifactRefDto = {
  artifact_id: 'artifact-unity-docs-001',
  content_hash: '4a7b91e77d9b0c83c7492f7bb3d7e45a',
  media_type: 'text/markdown',
  schema_name: 'SourceDocument',
  schema_version: schemaVersion,
  size: 4_728,
  tenant_id: tenantId,
  created_at: '2026-07-11T08:58:00.000Z',
  access_scope: scope,
}

const contract = {
  schema_version: schemaVersion,
  task_id: 'task-success-001',
  revision: 1,
  tenant_id: tenantId,
  user_id: 'user-local',
  conversation_id: request.conversation_id as string,
  project_id: projectId,
  goal: request.message,
  inputs: [inputArtifact],
  expected_output: 'text',
  constraints: ['Ответить по-русски', 'Не добавлять непроверенные причины'],
  success_criteria: ['Ответ соответствует контракту и содержит допустимое происхождение.'],
  risk_level: 'low' as const,
  source_policy: 'allowed_sources_only',
  budget,
  protected_context_refs: [inputArtifact.artifact_id],
  created_at: timestamp,
}

const successTask: TaskViewDto = {
  schema_version: schemaVersion,
  task_id: contract.task_id,
  trace_id: 'trace-success-001',
  hive_id: 'hive-unity-001',
  status: 'succeeded',
  contract,
  answer: {
    schema_version: schemaVersion,
    task_id: contract.task_id,
    trace_id: 'trace-success-001',
    hive_id: 'hive-unity-001',
    status: 'completed',
    answer:
      'Сначала проверьте «Is Trigger» у Collider2D: включённый Trigger сообщает о пересечении, но не создаёт физическое препятствие. Затем проверьте Rigidbody2D и collision layers.',
    sources: [sourceArtifact],
    critic_reports: [
      {
        schema_version: schemaVersion,
        report_id: 'critic-provenance-001',
        critic: 'ProvenanceCritic',
        target: 'answer',
        verdict: 'pass',
        severity: 'info',
        evidence: ['source:artifact-unity-docs-001'],
        repair_hint: null,
        created_at: '2026-07-11T09:15:00.600Z',
      },
    ],
    warnings: [],
  },
  error: null,
  created_at: timestamp,
  updated_at: '2026-07-11T09:15:00.620Z',
}

function span(
  id: string,
  component: string,
  operation: string,
  startedAt: string,
  durationMs: number,
  extra: Partial<TraceSpanDto> = {},
): TraceSpanDto {
  return {
    schema_version: schemaVersion,
    span_id: id,
    trace_id: 'trace-success-001',
    parent_span_id: null,
    causation_id: 'cmd-success-001',
    component,
    operation,
    status: 'succeeded',
    started_at: startedAt,
    ended_at: startedAt,
    duration_ms: durationMs,
    input_summary: {},
    output_summary: {},
    budget_before: budget,
    budget_after: budget,
    ...extra,
  }
}

const successTrace: TraceDto = {
  trace_id: 'trace-success-001',
  spans: [
    span('span-execute-001', 'CommandRuntime', 'execute_task', timestamp, 620, {
      input_summary: { task_id: contract.task_id },
      output_summary: { status: 'completed' },
    }),
    span('span-contract-001', 'RequestAnalyzer', 'BUILD_TASK_CONTRACT', '2026-07-11T09:15:00.016Z', 68, {
      parent_span_id: 'span-execute-001',
      input_summary: { goal_length: request.message.length },
      output_summary: { hive_id: 'hive-unity-001', hive_decision: 'create' },
    }),
    span('span-retrieve-001', 'CosmosRetriever', 'RETRIEVE_CLAIMS', '2026-07-11T09:15:00.132Z', 130, {
      parent_span_id: 'span-execute-001',
      input_ref: inputArtifact,
      output_ref: sourceArtifact,
      input_summary: { query: 'Collider2D' },
      output_summary: { claim_count: 1, gaps: [] },
    }),
    span('span-plan-001', 'Planner', 'BUILD_PLAN', '2026-07-11T09:15:00.262Z', 120, {
      parent_span_id: 'span-execute-001',
      output_summary: { steps: 3, status: 'validated' },
    }),
    span('span-verify-001', 'CriticSystem', 'VERIFY', '2026-07-11T09:15:00.382Z', 100, {
      parent_span_id: 'span-execute-001',
      output_summary: { verdict: 'pass' },
    }),
    span('span-format-001', 'TextCodec', 'FORMAT_TEXT', '2026-07-11T09:15:00.482Z', 138, {
      parent_span_id: 'span-execute-001',
      output_summary: { source_count: 1, language: 'ru' },
    }),
  ],
  events: [
    {
      schema_version: schemaVersion,
      id: 'evt-envelope-success-001',
      event_id: 'event-queued-001',
      occurred_at: timestamp,
      tenant_id: tenantId,
      task_id: contract.task_id,
      trace_id: 'trace-success-001',
      kind: 'CommandQueued',
      producer: 'runtime',
      payload: { command_id: 'cmd-success-001', handler: 'execute_task' },
      causation_id: 'cmd-success-001',
      correlation_id: 'cmd-success-001',
      sequence: 1,
    },
    {
      schema_version: schemaVersion,
      id: 'evt-envelope-success-002',
      event_id: 'event-succeeded-001',
      occurred_at: '2026-07-11T09:15:00.620Z',
      tenant_id: tenantId,
      task_id: contract.task_id,
      trace_id: 'trace-success-001',
      kind: 'CommandSucceeded',
      producer: 'runtime',
      payload: { command_id: 'cmd-success-001' },
      causation_id: 'cmd-success-001',
      correlation_id: 'cmd-success-001',
      sequence: 2,
    },
  ],
}

const hive: HiveViewDto = {
  schema_version: schemaVersion,
  hive_id: 'hive-unity-001',
  tenant_id: tenantId,
  conversation_id: request.conversation_id as string,
  project_id: projectId,
  state: 'active',
  contract,
  topics: ['unity', 'collider', 'стену'],
  state_data: {
    goals: ['Проверить Is Trigger', 'Проверить Rigidbody2D', 'Проверить collision layers'],
    budget_ledger: { hot_bytes: 712, warm_bytes: 88, evicted_bytes: 88 },
    selected_knowledge_refs: ['claim-collider-trigger'],
    plan_refs: ['plan-collider-001'],
  },
  entries: [
    {
      schema_version: schemaVersion,
      entry_id: 'entry-goal-001',
      hive_id: 'hive-unity-001',
      store_name: 'GoalStore',
      layer: 'hot',
      content: { goal: contract.goal },
      content_type: 'task_goal',
      size: 294,
      source_ref: inputArtifact.artifact_id,
      relevance: 1,
      protected: true,
      reconstruction_cost: 1,
      expiry_policy: 'until_hive_complete',
      created_at: timestamp,
    },
    {
      schema_version: schemaVersion,
      entry_id: 'entry-work-001',
      hive_id: 'hive-unity-001',
      store_name: 'WorkingContextStore',
      layer: 'hot',
      content: { message: request.message },
      content_type: 'user_message',
      size: 418,
      source_ref: inputArtifact.artifact_id,
      relevance: 0.9,
      protected: false,
      reconstruction_cost: 0.4,
      expiry_policy: 'until_hive_complete',
      created_at: timestamp,
    },
    {
      schema_version: schemaVersion,
      entry_id: 'entry-warm-001',
      hive_id: 'hive-unity-001',
      store_name: 'WorkingContextStore',
      layer: 'warm',
      content: { summary: 'Служебное приветствие' },
      content_type: 'conversation_summary',
      size: 88,
      source_ref: null,
      relevance: 0.1,
      protected: false,
      reconstruction_cost: 0.1,
      expiry_policy: 'until_hive_complete',
      created_at: '2026-07-11T09:14:59.000Z',
    },
  ],
  snapshot_id: 'snapshot-hive-unity-001-r1',
  version: 3,
  created_at: timestamp,
  updated_at: '2026-07-11T09:15:00.620Z',
}

const concepts: ConceptDto[] = [
  {
    schema_version: schemaVersion,
    concept_id: 'concept-collider2d',
    label: 'Collider2D',
    concept_type: 'term',
    aliases: ['2D Collider'],
    tenant_id: tenantId,
    created_at: sourceArtifact.created_at,
  },
  {
    schema_version: schemaVersion,
    concept_id: 'concept-rigidbody2d',
    label: 'Rigidbody2D',
    concept_type: 'term',
    aliases: [],
    tenant_id: tenantId,
    created_at: sourceArtifact.created_at,
  },
]

const claims: ClaimDto[] = [
  {
    schema_version: schemaVersion,
    claim_id: 'claim-collider-trigger',
    tenant_id: tenantId,
    subject_id: 'concept-collider2d',
    predicate: 'states',
    object_value: 'Trigger сообщает о пересечении, но не создаёт физическое препятствие.',
    source_id: 'source-unity-001',
    source_artifact_id: sourceArtifact.artifact_id,
    source_fragment: 'Trigger сообщает о пересечении, но не создаёт физическое препятствие.',
    sectors: ['Programming', 'Game development'],
    access_scope: scope,
    verification_status: 'reviewed',
    scores: {
      schema_version: schemaVersion,
      confidence: 0.92,
      relevance: 0.98,
      utility: 0.89,
      freshness: 0.94,
      contradiction: 0.02,
      use_cost: 0.05,
    },
    valid_from: null,
    valid_to: null,
    created_at: sourceArtifact.created_at,
  },
]

const baseSystem: SystemSnapshotDto = {
  active_tasks: 0,
  queued_work_items: 0,
  dead_letters: 0,
  last_errors: [],
  health: {
    status: 'ok',
    runtime: 'sqlite-in-process',
    data_dir: 'mock://superai-data',
    work_items: { succeeded: 1 },
  },
  meta: {
    service: 'superai',
    api_version: 'v1',
    schema_version: schemaVersion,
    runtime: 'modular-monolith',
    capabilities: ['tasks', 'traces', 'hives', 'artifacts', 'cosmos'],
  },
}

function error(code: string, message: string, retryable = false): ErrorDto {
  return {
    schema_version: schemaVersion,
    code,
    message,
    retryable,
    details: {},
    occurred_at: timestamp,
  }
}

function failedTask(
  taskId: string,
  traceId: string,
  taskError: ErrorDto,
  status: TaskViewDto['status'] = 'failed',
): TaskViewDto {
  return {
    schema_version: schemaVersion,
    task_id: taskId,
    trace_id: traceId,
    hive_id: null,
    status,
    contract: {
      ...contract,
      task_id: taskId,
    },
    answer: null,
    error: taskError,
    created_at: timestamp,
    updated_at: '2026-07-11T09:15:01.000Z',
  }
}

function failureTrace(task: TaskViewDto, component: string, operation: string): TraceDto {
  const taskError = task.error ?? error('UNKNOWN', 'Unknown error')
  return {
    trace_id: task.trace_id,
    spans: [
      {
        schema_version: schemaVersion,
        span_id: 'span-' + task.task_id + '-execute',
        trace_id: task.trace_id,
        parent_span_id: null,
        causation_id: 'cmd-' + task.task_id,
        component,
        operation,
        status: task.status === 'cancelled' ? 'cancelled' : 'failed',
        started_at: task.created_at,
        ended_at: task.updated_at,
        duration_ms: 100,
        input_summary: { task_id: task.task_id },
        output_summary: {},
        budget_before: budget,
        budget_after: budget,
        error: taskError,
      },
    ],
    events: [
      {
        schema_version: schemaVersion,
        id: 'evt-envelope-' + task.task_id,
        event_id: 'evt-' + task.task_id,
        occurred_at: task.updated_at,
        tenant_id: tenantId,
        task_id: task.task_id,
        trace_id: task.trace_id,
        kind: task.status === 'cancelled' ? 'CommandCancelled' : 'CommandFailed',
        producer: 'runtime',
        payload: { error: taskError.code },
        causation_id: 'cmd-' + task.task_id,
        correlation_id: 'cmd-' + task.task_id,
        sequence: 1,
      },
    ],
  }
}

const validationError = error('validation_error', 'Request does not match the versioned API contract.')
const validationTask = failedTask('task-validation-001', 'trace-validation-001', validationError)
const timeoutError = error('budget_exceeded', 'task deadline exceeded')
const timeoutTask = failedTask('task-timeout-001', 'trace-timeout-001', timeoutError)
const cancelError = error('cancelled', 'command cancelled by user')
const cancelTask = failedTask('task-cancel-001', 'trace-cancel-001', cancelError, 'cancelled')

const retryTask: TaskViewDto = {
  ...successTask,
  task_id: 'task-retry-001',
  trace_id: 'trace-retry-001',
  contract: {
    ...contract,
    task_id: 'task-retry-001',
  },
  answer: {
    ...successTask.answer!,
    task_id: 'task-retry-001',
    trace_id: 'trace-retry-001',
  },
}

const retryTrace: TraceDto = {
  trace_id: 'trace-retry-001',
  spans: [
    {
      ...span('span-retry-failure', 'ObjectStore', 'READ_SOURCE', timestamp, 100),
      trace_id: 'trace-retry-001',
      status: 'failed',
      error: error('transient_failure', 'source temporarily unavailable', true),
    },
    {
      ...span('span-retry-success', 'CommandRuntime', 'execute_task', '2026-07-11T09:15:00.200Z', 500),
      trace_id: 'trace-retry-001',
      output_summary: { retry_attempt: 2, status: 'completed' },
    },
  ],
  events: [
    {
      schema_version: schemaVersion,
      id: 'evt-envelope-retry-001',
      event_id: 'evt-retry-001',
      occurred_at: timestamp,
      tenant_id: tenantId,
      task_id: 'task-retry-001',
      trace_id: 'trace-retry-001',
      kind: 'CommandRetryScheduled',
      producer: 'runtime',
      payload: { attempt: 1, max_attempts: 3 },
      causation_id: 'cmd-retry-001',
      correlation_id: 'cmd-retry-001',
      sequence: 1,
    },
    {
      schema_version: schemaVersion,
      id: 'evt-envelope-retry-002',
      event_id: 'evt-retry-002',
      occurred_at: '2026-07-11T09:15:00.700Z',
      tenant_id: tenantId,
      task_id: 'task-retry-001',
      trace_id: 'trace-retry-001',
      kind: 'CommandSucceeded',
      producer: 'runtime',
      payload: { attempt: 2 },
      causation_id: 'cmd-retry-001',
      correlation_id: 'cmd-retry-001',
      sequence: 2,
    },
  ],
}

export const fixtureScenarios: ScenarioFixture[] = [
  {
    id: 'success',
    title: 'Успешный ответ',
    description: 'TaskView, TraceSpan и HiveView имеют ту же форму, что и live API.',
    request,
    task: successTask,
    trace: successTrace,
    hive,
    artifacts: [sourceArtifact, inputArtifact],
    concepts,
    claims,
    system: baseSystem,
  },
  {
    id: 'validation',
    title: 'Ошибка валидации',
    description: 'Контракт отклонён, ошибка приходит как ErrorEnvelope.',
    request: { ...request, message: '' },
    task: validationTask,
    trace: failureTrace(validationTask, 'TaskSubmission', 'validate'),
    artifacts: [inputArtifact],
    concepts,
    claims,
    system: { ...baseSystem, last_errors: [validationError] },
  },
  {
    id: 'timeout',
    title: 'Превышение бюджета',
    description: 'Runtime остановил task по budget/time limit.',
    request,
    task: timeoutTask,
    trace: failureTrace(timeoutTask, 'CommandRuntime', 'execute_task'),
    hive,
    artifacts: [sourceArtifact, inputArtifact],
    concepts,
    claims,
    system: { ...baseSystem, last_errors: [timeoutError] },
  },
  {
    id: 'retry',
    title: 'Восстановимый сбой',
    description: 'Retry виден в событиях трассы, а не как выдуманное поле span.',
    request,
    task: retryTask,
    trace: retryTrace,
    hive,
    artifacts: [sourceArtifact, inputArtifact],
    concepts,
    claims,
    system: baseSystem,
  },
  {
    id: 'cancel',
    title: 'Отмена пользователем',
    description: 'Задача отменена с terminal TaskState и причинным событием.',
    request,
    task: cancelTask,
    trace: failureTrace(cancelTask, 'CommandRuntime', 'cancel_task'),
    hive,
    artifacts: [sourceArtifact, inputArtifact],
    concepts,
    claims,
    system: { ...baseSystem, last_errors: [cancelError] },
  },
  {
    id: 'incompatible',
    title: 'Несовместимая схема',
    description: 'Неизвестная major-версия должна быть отклонена validator до рендера.',
    request,
    task: {
      ...successTask,
      task_id: 'task-incompatible-001',
      trace_id: 'trace-incompatible-001',
      schema_version: '2.0',
    },
    trace: {
      ...successTrace,
      trace_id: 'trace-incompatible-001',
      spans: successTrace.spans.map((item) => ({
        ...item,
        trace_id: 'trace-incompatible-001',
        schema_version: '2.0',
      })),
      events: successTrace.events.map((item) => ({
        ...item,
        trace_id: 'trace-incompatible-001',
        schema_version: '2.0',
      })),
    },
    artifacts: [sourceArtifact],
    concepts,
    claims,
    system: baseSystem,
  },
  {
    id: 'offline',
    title: 'Backend недоступен',
    description: 'Сеть недоступна; ранее открытая трасса остаётся в Pinia store.',
    request,
    unavailable: true,
    artifacts: [sourceArtifact],
    concepts,
    claims,
    system: {
      ...baseSystem,
      health: {
        ...baseSystem.health,
        status: 'offline',
      },
    },
  },
]

export function getFixture(id: FixtureScenarioId): ScenarioFixture {
  const fixture = fixtureScenarios.find((candidate) => candidate.id === id)
  if (!fixture) {
    throw new Error('Неизвестный mock fixture: ' + id)
  }
  return fixture
}

export const defaultFixtureId: FixtureScenarioId = 'success'
