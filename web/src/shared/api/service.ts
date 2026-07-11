import type {
  ArtifactRefDto,
  ClaimDto,
  ConceptDto,
  HealthDto,
  HiveViewDto,
  MetaDto,
  SystemSnapshotDto,
  TaskSubmissionDto,
  TaskViewDto,
  TraceDto,
} from '@/shared/contracts/transport'
import {
  ContractValidationError,
  parseArtifactRefDto,
  parseClaimDtos,
  parseConceptDtos,
  parseHealthDto,
  parseHiveViewDto,
  parseMetaDto,
  parseSystemSnapshotDto,
  parseTaskSubmissionDto,
  parseTaskViewDto,
  parseTraceDto,
} from '@/shared/contracts/validator'
import { getFixture, type FixtureScenarioId, fixtureScenarios } from '@/shared/mocks/fixtures'

export type ApiMode = 'mock' | 'live'

export interface CosmosDataDto {
  concepts: ConceptDto[]
  claims: ClaimDto[]
}

export interface TaskService {
  health(): Promise<HealthDto>
  meta(): Promise<MetaDto>
  createTask(request: TaskSubmissionDto, scenario?: FixtureScenarioId): Promise<TaskViewDto>
  cancelTask(taskId: string, projectId?: string): Promise<TaskViewDto>
  getTask(taskId: string, projectId?: string): Promise<TaskViewDto>
  getTrace(traceId: string, projectId?: string): Promise<TraceDto>
  getHive(hiveId: string, projectId?: string): Promise<HiveViewDto>
  getArtifactMetadata(artifactId: string, projectId?: string): Promise<ArtifactRefDto>
  getCosmosData(projectId?: string): Promise<CosmosDataDto>
  getSystemSnapshot(): Promise<SystemSnapshotDto>
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function networkError(message: string): ContractValidationError {
  return new ContractValidationError(message, 'network')
}

export class MockTaskService implements TaskService {
  private readonly taskOverrides = new Map<string, TaskViewDto>()

  private fixtureForTask(taskId: string) {
    return fixtureScenarios.find((fixture) => fixture.task?.task_id === taskId)
  }

  async health(): Promise<HealthDto> {
    return parseHealthDto(clone(getFixture('success').system.health))
  }

  async meta(): Promise<MetaDto> {
    return parseMetaDto(clone(getFixture('success').system.meta))
  }

  async createTask(
    request: TaskSubmissionDto,
    scenario: FixtureScenarioId = 'success',
  ): Promise<TaskViewDto> {
    parseTaskSubmissionDto(request)
    const fixture = getFixture(scenario)
    if (fixture.unavailable) {
      throw networkError('Mock backend недоступен: соединение отклонено.')
    }
    if (!fixture.task) {
      throw new ContractValidationError('Fixture не содержит TaskView.')
    }
    const task = parseTaskViewDto(clone(fixture.task))
    this.taskOverrides.set(task.task_id, task)
    return task
  }

  async cancelTask(taskId: string, _projectId?: string): Promise<TaskViewDto> {
    const existing = this.taskOverrides.get(taskId)
    const fixture = this.fixtureForTask(taskId)
    if (!existing && !fixture?.task) {
      throw new ContractValidationError('Задача не найдена.', 'backend_error')
    }
    const cancelledFixture = getFixture('cancel').task
    if (!cancelledFixture) {
      throw new ContractValidationError('Fixture отмены не содержит TaskView.')
    }
    const cancelled: TaskViewDto = {
      ...clone(cancelledFixture),
      task_id: taskId,
      trace_id: existing?.trace_id ?? fixture?.task?.trace_id ?? cancelledFixture.trace_id,
      contract: existing?.contract ?? fixture?.task?.contract ?? cancelledFixture.contract,
    }
    this.taskOverrides.set(taskId, cancelled)
    return parseTaskViewDto(cancelled)
  }

  async getTask(taskId: string, _projectId?: string): Promise<TaskViewDto> {
    const overridden = this.taskOverrides.get(taskId)
    if (overridden) {
      return parseTaskViewDto(clone(overridden))
    }
    const fixture = this.fixtureForTask(taskId)
    if (!fixture?.task) {
      throw new ContractValidationError('Задача не найдена.', 'backend_error')
    }
    return parseTaskViewDto(clone(fixture.task))
  }

  async getTrace(traceId: string, _projectId?: string): Promise<TraceDto> {
    const fixture = fixtureScenarios.find((candidate) => candidate.trace?.trace_id === traceId)
    if (!fixture?.trace) {
      throw new ContractValidationError('Трасса не найдена.', 'backend_error')
    }
    return parseTraceDto(clone(fixture.trace))
  }

  async getHive(hiveId: string, _projectId?: string): Promise<HiveViewDto> {
    const fixture = fixtureScenarios.find((candidate) => candidate.hive?.hive_id === hiveId)
    if (!fixture?.hive) {
      throw new ContractValidationError('Улей не найден.', 'backend_error')
    }
    return parseHiveViewDto(clone(fixture.hive))
  }

  async getArtifactMetadata(artifactId: string, _projectId?: string): Promise<ArtifactRefDto> {
    const artifact = fixtureScenarios
      .flatMap((fixture) => fixture.artifacts ?? [])
      .find((candidate) => candidate.artifact_id === artifactId)
    if (!artifact) {
      throw new ContractValidationError('Артефакт не найден.', 'backend_error')
    }
    return parseArtifactRefDto(clone(artifact))
  }

  async getCosmosData(_projectId?: string): Promise<CosmosDataDto> {
    const fixture = getFixture('success')
    return {
      concepts: parseConceptDtos(clone(fixture.concepts ?? [])),
      claims: parseClaimDtos(clone(fixture.claims ?? [])),
    }
  }

  async getSystemSnapshot(): Promise<SystemSnapshotDto> {
    return parseSystemSnapshotDto(clone(getFixture('success').system))
  }
}

interface LiveTaskServiceOptions {
  baseUrl?: string
  fetchImpl?: typeof fetch
  tenantId?: string
}

export class LiveTaskService implements TaskService {
  private readonly baseUrl: string
  private readonly fetchImpl: typeof fetch
  private readonly tenantId: string

  constructor(options: LiveTaskServiceOptions = {}) {
    this.baseUrl = (options.baseUrl ?? import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')
    this.fetchImpl = (options.fetchImpl ?? globalThis.fetch).bind(globalThis)
    this.tenantId = options.tenantId ?? 'local'
  }

  private withProject(path: string, projectId?: string): string {
    if (!projectId) {
      return path
    }
    return path + (path.includes('?') ? '&' : '?') + 'project_id=' + encodeURIComponent(projectId)
  }

  private async request(path: string, init?: RequestInit): Promise<unknown> {
    let response: Response
    try {
      const { headers: requestHeaders, ...requestInit } = init ?? {}
      response = await this.fetchImpl(this.baseUrl + path, {
        ...requestInit,
        headers: {
          Accept: 'application/json',
          'X-Tenant-Id': this.tenantId,
          ...(requestInit.body ? { 'Content-Type': 'application/json' } : {}),
          ...requestHeaders,
        },
      })
    } catch {
      throw networkError(
        'Backend недоступен для ' + path + '. Сохранённые трассы остаются доступны локально.',
      )
    }

    const contentType = response.headers.get('content-type') ?? ''
    const body = contentType.includes('application/json')
      ? await response.json().catch(() => undefined)
      : undefined
    if (!response.ok) {
      const message =
        body &&
        typeof body === 'object' &&
        'message' in body &&
        typeof body.message === 'string'
          ? body.message
          : 'Backend вернул HTTP ' + response.status + '.'
      throw new ContractValidationError(message, 'backend_error')
    }
    return body
  }

  async health(): Promise<HealthDto> {
    return parseHealthDto(await this.request('/api/v1/health'))
  }

  async meta(): Promise<MetaDto> {
    return parseMetaDto(await this.request('/api/v1/meta'))
  }

  async createTask(request: TaskSubmissionDto): Promise<TaskViewDto> {
    parseTaskSubmissionDto(request)
    return parseTaskViewDto(
      await this.request('/api/v1/tasks', {
        method: 'POST',
        headers: { 'X-Tenant-Id': request.tenant_id },
        body: JSON.stringify(request),
      }),
    )
  }

  async cancelTask(taskId: string, projectId?: string): Promise<TaskViewDto> {
    return parseTaskViewDto(
      await this.request(this.withProject('/api/v1/tasks/' + encodeURIComponent(taskId) + '/cancel', projectId), {
        method: 'POST',
      }),
    )
  }

  async getTask(taskId: string, projectId?: string): Promise<TaskViewDto> {
    return parseTaskViewDto(
      await this.request(this.withProject('/api/v1/tasks/' + encodeURIComponent(taskId), projectId)),
    )
  }

  async getTrace(traceId: string, projectId?: string): Promise<TraceDto> {
    return parseTraceDto(
      await this.request(this.withProject('/api/v1/traces/' + encodeURIComponent(traceId), projectId)),
    )
  }

  async getHive(hiveId: string, projectId?: string): Promise<HiveViewDto> {
    return parseHiveViewDto(
      await this.request(this.withProject('/api/v1/hives/' + encodeURIComponent(hiveId), projectId)),
    )
  }

  async getArtifactMetadata(artifactId: string, projectId?: string): Promise<ArtifactRefDto> {
    return parseArtifactRefDto(
      await this.request(
        this.withProject('/api/v1/artifacts/' + encodeURIComponent(artifactId) + '/metadata', projectId),
      ),
    )
  }

  async getCosmosData(projectId?: string): Promise<CosmosDataDto> {
    const [concepts, claims] = await Promise.all([
      this.request(this.withProject('/api/v1/cosmos/concepts', projectId)),
      this.request(this.withProject('/api/v1/cosmos/claims', projectId)),
    ])
    return {
      concepts: parseConceptDtos(concepts),
      claims: parseClaimDtos(claims),
    }
  }

  async getSystemSnapshot(): Promise<SystemSnapshotDto> {
    const [health, meta, deadLetters] = await Promise.all([
      this.health(),
      this.meta(),
      this.request('/api/v1/system/dead-letters'),
    ])
    const workItems = health.work_items ?? {}
    const queued = workItems.queued ?? 0
    const running = workItems.running ?? 0
    return parseSystemSnapshotDto({
      active_tasks: queued + running,
      queued_work_items: queued,
      dead_letters: Array.isArray(deadLetters) ? deadLetters.length : 0,
      last_errors: [],
      health,
      meta,
    })
  }
}

const mockService = new MockTaskService()
const liveService = new LiveTaskService()

export function serviceFor(mode: ApiMode): TaskService {
  return mode === 'mock' ? mockService : liveService
}
