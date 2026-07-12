import { defineStore } from 'pinia'
import { ref } from 'vue'

interface WordData {
  word: string
  mass: number
  x: number
  y: number
  frequency: number
  halo: number
  permeability: number
  gravity: number
}

interface ConnectionData { word_a: string; word_b: string; strength: number; contexts: number }

interface Stats {
  tokens: number
  total_tokens: number
  phrases: number
  edges: number
}

interface TrainResult {
  success: boolean
  session_id?: string
  words: WordData[]
  connections: ConnectionData[]
  stats: Stats
  time_ms: number
  error?: string
}

interface SpaceResult {
  words: WordData[]
  connections: ConnectionData[]
  stats: Stats
}

export const useTrainingStore = defineStore('training', () => {
  const currentSessionId = ref<string | null>(null)
  const baseUrl = '/api'

  async function learn(text: string): Promise<TrainResult> {
    const body = {
      text,
      ...(currentSessionId.value ? { session_id: currentSessionId.value } : {}),
    }

    const response = await fetch(`${baseUrl}/v1/training/learn`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const err = await response.json()
      throw new Error(err.detail || 'Training failed')
    }

    const result = await response.json()
    if (result.session_id) {
      currentSessionId.value = result.session_id
    }
    return result
  }

  async function getSpace(): Promise<SpaceResult> {
    const params = new URLSearchParams()
    if (currentSessionId.value) {
      params.append('session_id', currentSessionId.value)
    }

    const response = await fetch(`${baseUrl}/v1/training/space?${params}`)
    if (!response.ok) {
      throw new Error('Failed to load space')
    }
    return response.json()
  }

  async function resetSpace(): Promise<void> {
    const params = new URLSearchParams()
    if (currentSessionId.value) params.append('session_id', currentSessionId.value)

    const response = await fetch(`${baseUrl}/v1/training/space?${params}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      throw new Error('Failed to reset space')
    }
  }

  function setSession(sessionId: string) {
    currentSessionId.value = sessionId
  }

  return {
    currentSessionId,
    learn,
    getSpace,
    resetSpace,
    setSession,
  }
})
