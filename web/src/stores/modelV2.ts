import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

export interface CloudV2 {
  id: number
  cloud_type: 'character' | 'word_form' | 'lexeme' | 'scene' | 'concept_candidate' | 'concept'
  canonical_name: string
  mass: number
  density: number
  stability: number
  base_activation: number
  observation_count: number
  metadata_json: string
}

export interface PlacementV2 {
  id: number
  cloud_id: number
  space_id: number
  x: number
  y: number
  z: number | null
  radius: number
  local_activation: number
  local_density: number
  local_gravity: number
  local_stability_modifier: number
  metadata_json: string
}

export interface SpaceV2 {
  id: number
  space_type: 'global_field' | 'scene_space' | 'word_structure_space' | 'concept_space' | 'hive_space'
  owner_cloud_id: number | null
  parent_space_id: number | null
  random_seed: number
}

export interface StructuralComponentV2 {
  id: number
  parent_cloud_id: number
  child_cloud_id: number
  component_index: number
  component_role: string
  weight: number
  local_x: number
  local_y: number
  local_z: number | null
}

export interface SceneComponentV2 {
  id: number
  placement_id: number
  cloud_id: number
  lexeme_cloud_id: number | null
  token_index: number
  grammatical_role: string
  dependency_role: string | null
  head_component_id: number | null
  confidence: number
  morphology_json: string
}

export interface StatsV2 {
  clouds_total: number
  clouds_by_type: Record<string, number>
  spaces_total: number
  spaces_by_type: Record<string, number>
  placements_total: number
  unique_word_forms: number
  scene_components_total: number
  structural_components_total: number
  concepts_total: number
}

export interface NormalizedSpaceV2 {
  space: SpaceV2
  clouds: Record<string, CloudV2>
  placements: PlacementV2[]
  stats: StatsV2
}

export interface StructureV2 {
  cloud: CloudV2
  structure_space: SpaceV2 | null
  components: StructuralComponentV2[]
  clouds: Record<string, CloudV2>
}

export interface SceneV2 {
  cloud_id: number
  scene_space_id: number
  sentence_text: string
  canonical_text: string
  observation_count: number
  parser_version: string
  components: SceneComponentV2[]
}

export interface TrainedModelSnapshotV2 {
  schema_version: number
  stats: StatsV2
  model: Record<string, unknown[]>
}

const emptyStats = (): StatsV2 => ({
  clouds_total: 0,
  clouds_by_type: {},
  spaces_total: 0,
  spaces_by_type: {},
  placements_total: 0,
  unique_word_forms: 0,
  scene_components_total: 0,
  structural_components_total: 0,
  concepts_total: 0,
})

const spaceLabels: Record<SpaceV2['space_type'], string> = {
  global_field: 'Глобальное поле',
  scene_space: 'Пространство сцены',
  word_structure_space: 'Структура словоформы',
  concept_space: 'Пространство понятия',
  hive_space: 'Пространство улья',
}

async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as { detail?: string }
    throw new Error(body.detail || `Request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const useModelV2Store = defineStore('model-v2', () => {
  const cloudsById = ref<Record<number, CloudV2>>({})
  const placementsById = ref<Record<number, PlacementV2>>({})
  const currentSpace = ref<SpaceV2 | null>(null)
  const stats = ref<StatsV2>(emptyStats())
  const breadcrumb = ref<Array<{ space: SpaceV2; label: string }>>([])
  const selectedPlacementId = ref<number | null>(null)
  const selectedStructure = ref<StructureV2 | null>(null)
  const currentStructure = ref<StructureV2 | null>(null)
  const selectedScene = ref<SceneV2 | null>(null)
  const lastTraining = ref<Record<string, any> | null>(null)
  const trainedModel = ref<TrainedModelSnapshotV2 | null>(null)
  const loading = ref(false)
  const error = ref('')

  const placements = computed(() => Object.values(placementsById.value))
  const selectedPlacement = computed(() => selectedPlacementId.value
    ? placementsById.value[selectedPlacementId.value] ?? null
    : null)
  const selectedCloud = computed(() => selectedPlacement.value
    ? cloudsById.value[selectedPlacement.value.cloud_id] ?? null
    : null)
  const selectedSceneComponent = computed(() => selectedPlacement.value && selectedScene.value
    ? selectedScene.value.components.find(item => item.placement_id === selectedPlacement.value?.id) ?? null
    : null)

  function ingest(payload: NormalizedSpaceV2, pushBreadcrumb = true) {
    currentSpace.value = payload.space
    cloudsById.value = Object.fromEntries(Object.values(payload.clouds).map(cloud => [cloud.id, cloud]))
    placementsById.value = Object.fromEntries(payload.placements.map(placement => [placement.id, placement]))
    stats.value = payload.stats
    selectedPlacementId.value = null
    currentStructure.value = null
    if (pushBreadcrumb) {
      const label = `${spaceLabels[payload.space.space_type]} #${payload.space.id}`
      breadcrumb.value.push({ space: payload.space, label })
    }
  }

  async function run<T>(operation: () => Promise<T>): Promise<T> {
    loading.value = true
    error.value = ''
    try {
      return await operation()
    } catch (cause) {
      error.value = cause instanceof Error ? cause.message : String(cause)
      throw cause
    } finally {
      loading.value = false
    }
  }

  async function loadField() {
    return run(async () => {
      const payload = await api<NormalizedSpaceV2>('/api/v2/field')
      breadcrumb.value = []
      selectedScene.value = null
      selectedStructure.value = null
      ingest(payload)
    })
  }

  async function loadSpace(spaceId: number, pushBreadcrumb = true) {
    return run(async () => ingest(await api<NormalizedSpaceV2>(`/api/v2/spaces/${spaceId}`), pushBreadcrumb))
  }

  async function loadScene(cloudId: number) {
    const payload = await api<{ scene: SceneV2 }>(`/api/v2/scenes/${cloudId}`)
    selectedScene.value = payload.scene
    return payload.scene
  }

  async function loadStructure(cloudId: number) {
    selectedStructure.value = await api<StructureV2>(`/api/v2/clouds/${cloudId}/structure`)
    return selectedStructure.value
  }

  async function selectPlacement(placementId: number | null) {
    selectedPlacementId.value = placementId
    selectedStructure.value = null
    const placement = placementId ? placementsById.value[placementId] : null
    const cloud = placement ? cloudsById.value[placement.cloud_id] : null
    if (cloud?.cloud_type === 'scene') await loadScene(cloud.id)
    if (cloud?.cloud_type === 'word_form') await loadStructure(cloud.id)
  }

  async function zoomIntoPlacement(placementId: number) {
    const placement = placementsById.value[placementId]
    const cloud = placement ? cloudsById.value[placement.cloud_id] : null
    if (!cloud) return
    if (cloud.cloud_type === 'scene') {
      const scene = await loadScene(cloud.id)
      await loadSpace(scene.scene_space_id)
      return
    }
    if (cloud.cloud_type === 'word_form') {
      const structure = await loadStructure(cloud.id)
      if (!structure.structure_space) return
      currentSpace.value = structure.structure_space
      currentStructure.value = structure
      cloudsById.value = Object.fromEntries(Object.values(structure.clouds).map(item => [item.id, item]))
      placementsById.value = {}
      breadcrumb.value.push({
        space: structure.structure_space,
        label: `${cloud.canonical_name} · структура`,
      })
    }
  }

  async function navigateTo(index: number) {
    const target = breadcrumb.value[index]
    if (!target) return
    breadcrumb.value = breadcrumb.value.slice(0, index)
    selectedScene.value = target.space.space_type === 'scene_space' ? selectedScene.value : null
    await loadSpace(target.space.id)
  }

  async function train(text: string) {
    return run(async () => {
      lastTraining.value = await api('/api/v2/training/learn', {
        method: 'POST',
        body: JSON.stringify({ text }),
      })
      trainedModel.value = null
      const payload = await api<NormalizedSpaceV2>('/api/v2/field')
      breadcrumb.value = []
      ingest(payload)
      return lastTraining.value
    })
  }

  async function tickPhysics() {
    if (!currentSpace.value) return
    await run(async () => {
      await api(`/api/v2/spaces/${currentSpace.value?.id}/physics/tick`, { method: 'POST' })
      trainedModel.value = null
      await loadSpace(currentSpace.value!.id, false)
    })
  }

  async function loadTrainedModel() {
    return run(async () => {
      trainedModel.value = await api<TrainedModelSnapshotV2>('/api/v2/model')
      return trainedModel.value
    })
  }

  async function clearModel() {
    return run(async () => {
      await api('/api/v2/model', { method: 'DELETE' })
      localStorage.removeItem('superai-v2-active-hive')
      localStorage.removeItem('superai-v2-chat-cache')
      lastTraining.value = null
      trainedModel.value = null
      selectedScene.value = null
      selectedStructure.value = null
      const payload = await api<NormalizedSpaceV2>('/api/v2/field')
      breadcrumb.value = []
      ingest(payload)
    })
  }

  return {
    cloudsById,
    placementsById,
    currentSpace,
    currentStructure,
    selectedStructure,
    selectedScene,
    selectedSceneComponent,
    selectedPlacementId,
    selectedPlacement,
    selectedCloud,
    placements,
    stats,
    breadcrumb,
    lastTraining,
    trainedModel,
    loading,
    error,
    loadField,
    loadSpace,
    selectPlacement,
    zoomIntoPlacement,
    navigateTo,
    train,
    tickPhysics,
    loadTrainedModel,
    clearModel,
  }
})
