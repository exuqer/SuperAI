import type {
  ArtifactRefDto,
  BudgetDto,
  ClaimDto,
  ConceptDto,
  HealthDto,
  HiveViewDto,
  MetaDto,
  SystemSnapshotDto,
  TaskSubmissionDto,
  TaskViewDto,
  TraceDto,
} from './transport'

export const SUPPORTED_SCHEMA_MAJOR = 1

export class ContractValidationError extends Error {
  constructor(
    message: string,
    public readonly code:
      | 'invalid_payload'
      | 'unsupported_schema'
      | 'network'
      | 'backend_error' = 'invalid_payload',
  ) {
    super(message)
    this.name = 'ContractValidationError'
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function requiredRecord(value: unknown, label: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new ContractValidationError(label + ' должен быть объектом.')
  }
  return value
}

function requiredString(record: Record<string, unknown>, field: string): string {
  const value = record[field]
  if (typeof value !== 'string' || value.trim() === '') {
    throw new ContractValidationError('Поле «' + field + '» отсутствует или имеет неверный тип.')
  }
  return value
}

function requiredNumber(record: Record<string, unknown>, field: string): number {
  const value = record[field]
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ContractValidationError('Поле «' + field + '» отсутствует или имеет неверный тип.')
  }
  return value
}

function requiredArray(record: Record<string, unknown>, field: string): unknown[] {
  const value = record[field]
  if (!Array.isArray(value)) {
    throw new ContractValidationError('Поле «' + field + '» отсутствует или имеет неверный тип.')
  }
  return value
}

function optionalObject(record: Record<string, unknown>, field: string): Record<string, unknown> | undefined {
  const value = record[field]
  if (value === undefined || value === null) {
    return undefined
  }
  return requiredRecord(value, field)
}

export function assertSupportedSchema(value: unknown): asserts value is string {
  if (typeof value !== 'string') {
    throw new ContractValidationError('Ответ не содержит schema_version.')
  }
  const [major] = value.split('.')
  if (!/^\d+$/.test(major) || Number(major) !== SUPPORTED_SCHEMA_MAJOR) {
    throw new ContractValidationError(
      'Схема ' +
        value +
        ' несовместима с клиентом (поддерживается major ' +
        SUPPORTED_SCHEMA_MAJOR +
        ').',
      'unsupported_schema',
    )
  }
}

export function parseBudgetDto(value: unknown): BudgetDto {
  const record = requiredRecord(value, 'budget')
  assertSupportedSchema(requiredString(record, 'schema_version'))
  for (const field of ['time_ms', 'step_limit', 'memory_bytes', 'event_limit']) {
    const number = requiredNumber(record, field)
    if (number < 1) {
      throw new ContractValidationError('Поле «' + field + '» должно быть положительным.')
    }
  }
  return record as unknown as BudgetDto
}

function parseAccessScope(value: unknown): void {
  const record = requiredRecord(value, 'access_scope')
  assertSupportedSchema(requiredString(record, 'schema_version'))
  requiredString(record, 'tenant_id')
  const visibility = requiredString(record, 'visibility')
  if (!['tenant', 'project', 'global'].includes(visibility)) {
    throw new ContractValidationError('Получена неизвестная visibility.')
  }
  requiredString(record, 'retention')
}

export function parseArtifactRefDto(value: unknown): ArtifactRefDto {
  const record = requiredRecord(value, 'artifact_ref')
  assertSupportedSchema(requiredString(record, 'schema_version'))
  for (const field of [
    'artifact_id',
    'content_hash',
    'media_type',
    'schema_name',
    'tenant_id',
    'created_at',
  ]) {
    requiredString(record, field)
  }
  requiredNumber(record, 'size')
  parseAccessScope(record.access_scope)
  return record as unknown as ArtifactRefDto
}

function parseError(value: unknown): void {
  const record = requiredRecord(value, 'error')
  assertSupportedSchema(requiredString(record, 'schema_version'))
  requiredString(record, 'code')
  requiredString(record, 'message')
  if (typeof record.retryable !== 'boolean') {
    throw new ContractValidationError('Поле «retryable» отсутствует или имеет неверный тип.')
  }
}

function parseTaskContract(value: unknown): void {
  const record = requiredRecord(value, 'contract')
  assertSupportedSchema(requiredString(record, 'schema_version'))
  for (const field of ['task_id', 'tenant_id', 'conversation_id', 'goal', 'created_at']) {
    requiredString(record, field)
  }
  requiredNumber(record, 'revision')
  parseBudgetDto(record.budget)
  for (const field of ['inputs', 'constraints', 'success_criteria', 'protected_context_refs']) {
    requiredArray(record, field)
  }
}

export function parseTaskSubmissionDto(value: unknown): TaskSubmissionDto {
  const record = requiredRecord(value, 'TaskSubmission')
  assertSupportedSchema(requiredString(record, 'schema_version'))
  requiredString(record, 'message')
  requiredString(record, 'tenant_id')
  parseBudgetDto(record.budget)
  return record as unknown as TaskSubmissionDto
}

export function parseTaskViewDto(value: unknown): TaskViewDto {
  const record = requiredRecord(value, 'TaskView')
  assertSupportedSchema(requiredString(record, 'schema_version'))
  for (const field of ['task_id', 'trace_id', 'status', 'created_at', 'updated_at']) {
    requiredString(record, field)
  }
  if (!['queued', 'running', 'succeeded', 'failed', 'cancelled', 'dead_letter'].includes(String(record.status))) {
    throw new ContractValidationError('Получен неизвестный статус задачи.')
  }
  const contract = optionalObject(record, 'contract')
  if (contract) {
    parseTaskContract(contract)
  }
  const answer = optionalObject(record, 'answer')
  if (answer) {
    assertSupportedSchema(requiredString(answer, 'schema_version'))
    for (const field of ['task_id', 'trace_id', 'hive_id', 'status', 'answer']) {
      requiredString(answer, field)
    }
    requiredArray(answer, 'sources').forEach(parseArtifactRefDto)
    requiredArray(answer, 'critic_reports')
    requiredArray(answer, 'warnings')
  }
  const error = optionalObject(record, 'error')
  if (error) {
    parseError(error)
  }
  return record as unknown as TaskViewDto
}

function parseTraceSpan(value: unknown): void {
  const record = requiredRecord(value, 'TraceSpan')
  assertSupportedSchema(requiredString(record, 'schema_version'))
  for (const field of ['span_id', 'trace_id', 'component', 'operation', 'status', 'started_at']) {
    requiredString(record, field)
  }
  if (!['running', 'succeeded', 'failed', 'cancelled', 'skipped'].includes(String(record.status))) {
    throw new ContractValidationError('Получен неизвестный статус span.')
  }
  requiredRecord(record.input_summary, 'input_summary')
  requiredRecord(record.output_summary, 'output_summary')
  if (record.input_ref !== undefined && record.input_ref !== null) {
    parseArtifactRefDto(record.input_ref)
  }
  if (record.output_ref !== undefined && record.output_ref !== null) {
    parseArtifactRefDto(record.output_ref)
  }
  if (record.budget_before !== undefined && record.budget_before !== null) {
    parseBudgetDto(record.budget_before)
  }
  if (record.budget_after !== undefined && record.budget_after !== null) {
    parseBudgetDto(record.budget_after)
  }
  if (record.error !== undefined && record.error !== null) {
    parseError(record.error)
  }
}

function parseDomainEvent(value: unknown): void {
  const record = requiredRecord(value, 'DomainEvent')
  assertSupportedSchema(requiredString(record, 'schema_version'))
  for (const field of [
    'id',
    'occurred_at',
    'tenant_id',
    'task_id',
    'trace_id',
    'kind',
    'producer',
  ]) {
    requiredString(record, field)
  }
  requiredRecord(record.payload, 'payload')
}

export function parseTraceDto(value: unknown): TraceDto {
  const record = requiredRecord(value, 'Trace')
  requiredString(record, 'trace_id')
  requiredArray(record, 'spans').forEach(parseTraceSpan)
  requiredArray(record, 'events').forEach(parseDomainEvent)
  return record as unknown as TraceDto
}

export function parseHiveViewDto(value: unknown): HiveViewDto {
  const record = requiredRecord(value, 'HiveView')
  assertSupportedSchema(requiredString(record, 'schema_version'))
  for (const field of [
    'hive_id',
    'tenant_id',
    'conversation_id',
    'state',
    'created_at',
    'updated_at',
  ]) {
    requiredString(record, field)
  }
  parseTaskContract(record.contract)
  requiredArray(record, 'topics')
  requiredRecord(record.state_data, 'state_data')
  requiredArray(record, 'entries').forEach((entry) => {
    const entryRecord = requiredRecord(entry, 'ContextEntry')
    assertSupportedSchema(requiredString(entryRecord, 'schema_version'))
    for (const field of ['entry_id', 'hive_id', 'store_name', 'layer', 'content_type', 'created_at']) {
      requiredString(entryRecord, field)
    }
    requiredNumber(entryRecord, 'size')
    requiredRecord(entryRecord.content, 'entry.content')
  })
  return record as unknown as HiveViewDto
}

export function parseConceptDtos(value: unknown): ConceptDto[] {
  if (!Array.isArray(value)) {
    throw new ContractValidationError('Ожидался массив Concept.')
  }
  value.forEach((item) => {
    const record = requiredRecord(item, 'Concept')
    assertSupportedSchema(requiredString(record, 'schema_version'))
    for (const field of ['concept_id', 'label', 'concept_type', 'tenant_id', 'created_at']) {
      requiredString(record, field)
    }
    requiredArray(record, 'aliases')
  })
  return value as ConceptDto[]
}

export function parseClaimDtos(value: unknown): ClaimDto[] {
  if (!Array.isArray(value)) {
    throw new ContractValidationError('Ожидался массив Claim.')
  }
  value.forEach((item) => {
    const record = requiredRecord(item, 'Claim')
    assertSupportedSchema(requiredString(record, 'schema_version'))
    for (const field of [
      'claim_id',
      'tenant_id',
      'subject_id',
      'predicate',
      'object_value',
      'source_id',
      'source_artifact_id',
      'source_fragment',
      'verification_status',
      'created_at',
    ]) {
      requiredString(record, field)
    }
    requiredArray(record, 'sectors')
    parseAccessScope(record.access_scope)
    const scores = requiredRecord(record.scores, 'scores')
    assertSupportedSchema(requiredString(scores, 'schema_version'))
    for (const field of [
      'confidence',
      'relevance',
      'utility',
      'freshness',
      'contradiction',
      'use_cost',
    ]) {
      requiredNumber(scores, field)
    }
  })
  return value as ClaimDto[]
}

export function parseHealthDto(value: unknown): HealthDto {
  const record = requiredRecord(value, 'health')
  const status = requiredString(record, 'status')
  if (!['ok', 'degraded', 'offline'].includes(status)) {
    throw new ContractValidationError('Получен неизвестный health status.')
  }
  return record as unknown as HealthDto
}

export function parseMetaDto(value: unknown): MetaDto {
  const record = requiredRecord(value, 'meta')
  assertSupportedSchema(requiredString(record, 'schema_version'))
  for (const field of ['service', 'api_version', 'runtime']) {
    requiredString(record, field)
  }
  requiredArray(record, 'capabilities')
  return record as unknown as MetaDto
}

export function parseSystemSnapshotDto(value: unknown): SystemSnapshotDto {
  const record = requiredRecord(value, 'system snapshot')
  for (const field of ['active_tasks', 'queued_work_items', 'dead_letters']) {
    requiredNumber(record, field)
  }
  requiredArray(record, 'last_errors')
  parseHealthDto(record.health)
  parseMetaDto(record.meta)
  return record as unknown as SystemSnapshotDto
}
