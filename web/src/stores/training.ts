import { defineStore } from 'pinia'
import { ref } from 'vue'

interface WordData {
  word: string
  mass: number
  x: number
  y: number
}

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
  stats: Stats
  time_ms: number
  error?: string
}

interface SpaceResult {
  words: WordData[]
  stats: Stats
}

export const useTrainingStore = defineStore('training', () => {
  const currentSessionId = ref<string | null>(null)
  const baseUrl = '/api'

  async function learn(text: string): Promise<TrainResult> {
    const body = new URLSearchParams()
    body.append('text', text)
    if (currentSessionId.value) {
      body.append('session_id', currentSessionId.value)
    }

    const response = await fetch(`${baseUrl}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body,
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

    const response = await fetch(`${baseUrl}/space?${params}`)
    if (!response.ok) {
      throw new Error('Failed to load space')
    }
    return response.json()
  }

  async function resetSpace(): Promise<void> {
    const params = new URLSearchParams()
    if (currentSessionId.value) {
      params.append('session_id', currentSessionId.value)
    }

    const response = await fetch(`${baseUrl}/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: params,
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