import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface ConceptData {
  id: number
  token: string
  position: number[]
  mass: number
  radius: number
  activation: number
}

export interface Stats {
  concepts: number
  total_mass: number
  tokens: number
}

export interface TrainResult {
  success: boolean
  concepts: ConceptData[]
  stats: Stats
  time_ms: number
  error?: string
}

export interface SpaceResult {
  concepts: ConceptData[]
  stats: Stats
}

export const useTrainingStore = defineStore('training', () => {
  const baseUrl = '/api'
  const concepts = ref<ConceptData[]>([])
  const stats = ref<Stats>({ concepts: 0, total_mass: 0, tokens: 0 })

  async function learn(text: string): Promise<TrainResult> {
    const response = await fetch(`${baseUrl}/v1/training/learn`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Training failed')
    }

    const result = await response.json() as TrainResult
    concepts.value = result.concepts
    stats.value = result.stats
    return result
  }

  async function getSpace(): Promise<SpaceResult> {
    const response = await fetch(`${baseUrl}/v1/training/space`)
    if (!response.ok) throw new Error('Failed to load space')
    const result = await response.json() as SpaceResult
    concepts.value = result.concepts
    stats.value = result.stats
    return result
  }

  async function resetSpace(): Promise<ResetResult> {
    const response = await fetch(`${baseUrl}/v1/training/space`, { method: 'DELETE' })
    if (!response.ok) throw new Error('Failed to reset space')
    const result = await response.json() as ResetResult
    concepts.value = result.concepts
    stats.value = result.stats
    return result
  }

  return { concepts, stats, learn, getSpace, resetSpace }
})

interface ResetResult {
  success: boolean
  concepts: ConceptData[]
  stats: Stats
}
