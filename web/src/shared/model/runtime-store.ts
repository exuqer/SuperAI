import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import type { TaskSubmissionDto } from '@/shared/contracts/transport'
import {
  toUiArtifact,
  toUiCosmosConcepts,
  toUiError,
  toUiHive,
  toUiSystem,
  toUiTask,
  toUiTrace,
  type UiArtifact,
  type UiCosmosConcept,
  type UiError,
  type UiHive,
  type UiSystem,
  type UiTask,
  type UiTrace,
} from '@/shared/contracts/ui-models'
import { ContractValidationError } from '@/shared/contracts/validator'
import { serviceFor, type ApiMode } from '@/shared/api/service'
import {
  defaultFixtureId,
  fixtureScenarios,
  type FixtureScenarioId,
} from '@/shared/mocks/fixtures'

function asUiError(error: unknown): UiError {
  if (error instanceof ContractValidationError) {
    return {
      code: error.code,
      message: error.message,
      retryable: false,
    }
  }
  return {
    code: 'UNEXPECTED_CLIENT_ERROR',
    message: error instanceof Error ? error.message : 'Неизвестная ошибка клиента.',
    retryable: false,
  }
}

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
}

export const useRuntimeStore = defineStore('runtime', () => {
  const mode = ref<ApiMode>(import.meta.env.VITE_API_BASE_URL ? 'live' : 'mock')
  const selectedFixtureId = ref<FixtureScenarioId>(defaultFixtureId)
  const task = ref<UiTask>()
  const activeTraceId = ref<string>()
  const traces = ref<Record<string, UiTrace>>({})
  const hive = ref<UiHive>()
  const artifacts = ref<Record<string, UiArtifact>>({})
  const cosmos = ref<UiCosmosConcept[]>([])
  const system = ref<UiSystem>()
  const runError = ref<UiError>()
  const systemError = ref<UiError>()
  const isRunning = ref(false)
  const isBootstrapping = ref(false)

  const activeTrace = computed(() =>
    activeTraceId.value ? traces.value[activeTraceId.value] : undefined,
  )
  const traceList = computed(() =>
    Object.values(traces.value).sort((left, right) =>
      (right.startedAt ?? '').localeCompare(left.startedAt ?? ''),
    ),
  )
  const isModeToggleAvailable = import.meta.env.DEV

  function service() {
    return serviceFor(mode.value)
  }

  function setMode(nextMode: ApiMode) {
    mode.value = nextMode
    runError.value = undefined
    systemError.value = undefined
  }

  async function loadSystem() {
    try {
      system.value = toUiSystem(await service().getSystemSnapshot())
      systemError.value = undefined
    } catch (error) {
      systemError.value = asUiError(error)
    }
  }

  async function bootstrap() {
    isBootstrapping.value = true
    await Promise.all([loadSystem(), loadCosmos()])
    isBootstrapping.value = false
  }

  async function loadTrace(traceId: string, projectId = task.value?.projectId) {
    const trace = toUiTrace(
      await service().getTrace(traceId, projectId),
      task.value?.traceId === traceId ? task.value : undefined,
    )
    traces.value = { ...traces.value, [trace.id]: trace }
    activeTraceId.value = trace.id
    return trace
  }

  async function loadHive(hiveId: string, projectId = task.value?.projectId) {
    hive.value = toUiHive(await service().getHive(hiveId, projectId))
    return hive.value
  }

  async function loadArtifact(artifactId: string, projectId = task.value?.projectId) {
    const artifact = toUiArtifact(await service().getArtifactMetadata(artifactId, projectId))
    artifacts.value = { ...artifacts.value, [artifact.id]: artifact }
    return artifact
  }

  async function loadCosmos(projectId = task.value?.projectId) {
    try {
      const data = await service().getCosmosData(projectId)
      cosmos.value = toUiCosmosConcepts(data.concepts, data.claims)
    } catch (error) {
      if (mode.value === 'live') {
        systemError.value = asUiError(error)
      }
    }
  }

  async function pollTask(taskId: string, projectId?: string): Promise<UiTask> {
    let delay = 250
    let current = toUiTask(await service().getTask(taskId, projectId))
    for (let attempt = 0; attempt < 6 && (current.status === 'queued' || current.status === 'running'); attempt += 1) {
      await wait(delay)
      delay = Math.min(delay * 2, 2_000)
      current = toUiTask(await service().getTask(taskId, projectId))
    }
    return current
  }

  async function runTask(request: TaskSubmissionDto) {
    isRunning.value = true
    runError.value = undefined
    try {
      const created = await service().createTask(request, selectedFixtureId.value)
      task.value = toUiTask(created)
      if (task.value.status === 'queued' || task.value.status === 'running') {
        task.value = await pollTask(task.value.id, task.value.projectId)
      }
      if (task.value.traceId) {
        await loadTrace(task.value.traceId, task.value.projectId)
      }
      if (task.value.hiveId) {
        await loadHive(task.value.hiveId, task.value.projectId)
      }
      await loadSystem()
      return task.value
    } catch (error) {
      runError.value = asUiError(error)
      return undefined
    } finally {
      isRunning.value = false
    }
  }

  async function cancelTask() {
    if (!task.value || !['queued', 'running'].includes(task.value.status)) {
      return
    }
    isRunning.value = true
    runError.value = undefined
    try {
      task.value = toUiTask(await service().cancelTask(task.value.id, task.value.projectId))
      if (task.value.traceId) {
        await loadTrace(task.value.traceId, task.value.projectId)
      }
      await loadSystem()
    } catch (error) {
      runError.value = asUiError(error)
    } finally {
      isRunning.value = false
    }
  }

  return {
    mode,
    selectedFixtureId,
    fixtureScenarios,
    task,
    traces,
    activeTraceId,
    activeTrace,
    traceList,
    hive,
    artifacts,
    cosmos,
    system,
    runError,
    systemError,
    isRunning,
    isBootstrapping,
    isModeToggleAvailable,
    setMode,
    bootstrap,
    loadSystem,
    loadTrace,
    loadHive,
    loadArtifact,
    loadCosmos,
    runTask,
    cancelTask,
  }
})
