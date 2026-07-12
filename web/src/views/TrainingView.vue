<template>
  <div class="training-view">
    <header class="header">
      <h1>SuperAI — Обучение модели</h1>
      <div class="stats-bar" v-if="stats">
        <span class="stat">🪙 Токены: {{ stats.tokens }}</span>
        <span class="stat">🔗 Связи: {{ stats.edges }}</span>
        <span class="stat">📝 Фразы: {{ stats.phrases }}</span>
      </div>
    </header>

    <main class="main">
      <section class="panel input-panel">
        <label for="input-text">Текст для обучения</label>
        <textarea
          id="input-text"
          v-model="inputText"
          placeholder="Введите текст для обучения модели..."
          rows="6"
        />
        <div class="input-actions">
          <button
            class="btn btn-primary"
            @click="handleLearn"
            :disabled="loading || !inputText.trim()"
          >
            <span v-if="loading" class="spinner"></span>
            {{ loading ? 'Обучение...' : 'Обучить' }}
          </button>
          <button
            class="btn btn-danger"
            @click="handleReset"
            :disabled="loading"
          >
            Сбросить
          </button>
        </div>
      </section>

      <section class="panel visualization-panel">
        <SpaceVisualization :words="words" :width="width" :height="height" />
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useTrainingStore } from '@/stores/training'
import SpaceVisualization from '@/components/SpaceVisualization.vue'

const store = useTrainingStore()

const inputText = ref('')
const loading = ref(false)
const words = ref<Array<{ word: string; mass: number; x: number; y: number }>>([])
const stats = ref<{ tokens: number; total_tokens: number; phrases: number; edges: number } | null>(null)

const width = 800
const height = 500

async function loadSpace() {
  try {
    const data = await store.getSpace()
    words.value = data.words
    stats.value = data.stats
  } catch (e) {
    console.error('Failed to load space:', e)
  }
}

async function handleLearn() {
  if (!inputText.value.trim()) return
  loading.value = true
  try {
    const result = await store.learn(inputText.value)
    words.value = result.words
    stats.value = result.stats
    inputText.value = ''
  } catch (e: any) {
    alert(e.message || 'Ошибка обучения')
  } finally {
    loading.value = false
  }
}

async function handleReset() {
  if (!confirm('Очистить всё пространство слов?')) return
  loading.value = true
  try {
    await store.resetSpace()
    words.value = []
    stats.value = { tokens: 0, total_tokens: 0, phrases: 0, edges: 0 }
  } catch (e: any) {
    alert(e.message || 'Ошибка сброса')
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadSpace()
})
</script>

<style scoped lang="scss">
.training-view {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: #f5f6fa;
}

.header {
  background: #fff;
  padding: 16px 24px;
  border-bottom: 1px solid #e0e4eb;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.header h1 {
  margin: 0 0 12px;
  font-size: 20px;
  font-weight: 600;
  color: #1a1d23;
}

.stats-bar {
  display: flex;
  gap: 24px;
  font-size: 13px;
  color: #5a6174;
}

.stat {
  display: flex;
  align-items: center;
  gap: 6px;
}

.main {
  flex: 1;
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: 24px;
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}

.panel {
  background: #fff;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.input-panel {
  display: flex;
  flex-direction: column;
  height: fit-content;
}

.input-panel label {
  font-size: 13px;
  font-weight: 500;
  color: #3a3f4b;
  margin-bottom: 8px;
}

.input-panel textarea {
  flex: 1;
  padding: 12px;
  border: 1px solid #d0d5dd;
  border-radius: 8px;
  font-size: 14px;
  font-family: inherit;
  resize: vertical;
  min-height: 120px;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.input-panel textarea:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
}

.input-actions {
  display: flex;
  gap: 12px;
  margin-top: 16px;
}

.btn {
  flex: 1;
  padding: 10px 16px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-primary {
  background: #3b82f6;
  color: #fff;
}

.btn-primary:hover:not(:disabled) {
  background: #2563eb;
}

.btn-danger {
  background: #ef4444;
  color: #fff;
}

.btn-danger:hover:not(:disabled) {
  background: #dc2626;
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.visualization-panel {
  min-height: 500px;
}

@media (max-width: 1024px) {
  .main {
    grid-template-columns: 1fr;
  }
  
  .input-panel {
    order: 2;
  }
  
  .visualization-panel {
    order: 1;
  }
}
</style>