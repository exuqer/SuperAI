import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

export interface CloudV2 {
  id: number
  cloud_type: string
  canonical_name: string
  mass: number
  density: number
  stability: number
  base_activation: number
  observation_count: number
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
}

export interface SpaceV2 {
  id: number
  space_type: 'global_field' | 'scene_space' | 'word_structure_space' | 'concept_space' | 'hive_space'
  owner_cloud_id: number | null
  parent_space_id: number | null
}

type NormalizedSpace = { space: SpaceV2; clouds: Record<string, CloudV2>; placements: PlacementV2[] }

export const useModelV2Store = defineStore('model-v2', () => {
  const cloudsById = ref<Record<number, CloudV2>>({})
  const placementsById = ref<Record<number, PlacementV2>>({})
  const currentSpace = ref<SpaceV2 | null>(null)
  const selectedPlacementId = ref<number | null>(null)
  const breadcrumb = ref<number[]>([])

  const placements = computed(() => Object.values(placementsById.value))
  const selectedPlacement = computed(() => selectedPlacementId.value ? placementsById.value[selectedPlacementId.value] ?? null : null)
  const selectedCloud = computed(() => selectedPlacement.value ? cloudsById.value[selectedPlacement.value.cloud_id] ?? null : null)

  function ingest(payload: NormalizedSpace) {
    currentSpace.value = payload.space
    cloudsById.value = Object.fromEntries(Object.values(payload.clouds).map(cloud => [cloud.id, cloud]))
    placementsById.value = Object.fromEntries(payload.placements.map(placement => [placement.id, placement]))
    breadcrumb.value = [...breadcrumb.value.filter(id => id !== payload.space.id), payload.space.id]
  }

  async function loadSpace(spaceId: number) {
    const response = await fetch(`/api/v2/spaces/${spaceId}`)
    if (!response.ok) throw new Error(`Unable to load V2 space ${spaceId}`)
    ingest(await response.json())
  }

  async function loadField() {
    const response = await fetch('/api/v2/field')
    if (!response.ok) throw new Error('Unable to load V2 field')
    ingest(await response.json())
  }

  async function zoomToCloud(cloudId: number) {
    const response = await fetch(`/api/v2/clouds/${cloudId}`)
    if (!response.ok) throw new Error(`Unable to load cloud ${cloudId}`)
    const { cloud } = await response.json() as { cloud: CloudV2 }
    const target = Object.values(placementsById.value).find(placement => placement.cloud_id === cloud.id)
    if (target) selectedPlacementId.value = target.id
  }

  return { cloudsById, placementsById, currentSpace, selectedPlacementId, breadcrumb, placements, selectedPlacement, selectedCloud, ingest, loadField, loadSpace, zoomToCloud }
})

