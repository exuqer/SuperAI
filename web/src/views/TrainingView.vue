<template>
  <div class="training-view">
    <header class="header">
      <div class="brand-line">
        <div class="brand-mark">S</div>
        <div>
          <p class="eyebrow">Concept field lab</p>
          <h1>SuperAI <span>/ Пространство понятий</span></h1>
        </div>
      </div>
      <p class="subtitle">Понятия существуют как области плотности в непрерывном гравитационном поле.</p>
      <div class="stats-bar">
        <span class="stat"><strong>{{ stats.concepts }}</strong> понятий</span>
        <span class="stat"><strong>{{ stats.total_mass.toFixed(1) }}</strong> масса</span>
        <span class="stat"><strong>{{ stats.tokens }}</strong> токенов</span>
      </div>
    </header>

    <main class="main">
      <section class="panel input-panel">
        <div class="panel-heading">
          <div><p class="eyebrow">01 / Input</p><h2>Текст для обучения</h2></div>
          <span class="status-dot" :class="{ active: loading }"></span>
        </div>
        <label for="input-text">Введите фразу или несколько предложений</label>
        <textarea id="input-text" v-model="inputText" placeholder="Например: Кот ест рыбу. Рыба ест кота." rows="8" @keydown.meta.enter="handleLearn" @keydown.ctrl.enter="handleLearn" />
        <p class="hint">⌘ / Ctrl + Enter — запустить обучение</p>
        <div class="input-actions">
          <button class="btn btn-primary" @click="handleLearn" :disabled="loading || !inputText.trim()">
            <span v-if="loading" class="spinner"></span>{{ loading ? 'Считаю поле…' : 'Обучить модель' }}
          </button>
          <button class="btn btn-danger" @click="handleReset" :disabled="loading">Очистить</button>
        </div>
        <p v-if="errorMessage" class="error-message">{{ errorMessage }}</p>
      </section>

      <section class="panel visualization-panel">
        <div class="panel-heading visualization-heading">
          <div><p class="eyebrow">02 / Field</p><h2>Градиентное поле</h2></div>
          <span class="live-badge"><i></i> live</span>
        </div>
        <SpaceVisualization :concepts="concepts" :width="width" :height="height" />
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useTrainingStore } from '@/stores/training'
import SpaceVisualization from '@/components/SpaceVisualization.vue'

const store = useTrainingStore()
const inputText = ref('')
const loading = ref(false)
const errorMessage = ref('')
const concepts = ref(store.concepts)
const stats = ref(store.stats)
const width = 1600
const height = 1000

async function loadSpace() {
  try {
    const result = await store.getSpace()
    concepts.value = result.concepts
    stats.value = result.stats
  } catch {
    errorMessage.value = 'Не удалось загрузить пространство'
  }
}

async function handleLearn() {
  if (!inputText.value.trim()) return
  loading.value = true
  errorMessage.value = ''
  try {
    const result = await store.learn(inputText.value)
    concepts.value = result.concepts
    stats.value = result.stats
    inputText.value = ''
  } catch (error: any) {
    errorMessage.value = error.message || 'Ошибка обучения'
  } finally {
    loading.value = false
  }
}

async function handleReset() {
  if (!confirm('Очистить всё пространство понятий?')) return
  loading.value = true
  errorMessage.value = ''
  try {
    const result = await store.resetSpace()
    concepts.value = result.concepts
    stats.value = result.stats
  } catch (error: any) {
    errorMessage.value = error.message || 'Ошибка сброса'
  } finally {
    loading.value = false
  }
}

onMounted(loadSpace)
</script>

<style scoped lang="scss">
.training-view { min-height: 100vh; color: #eef4ff; }
.header { display: grid; gap: .7rem; padding: 2.1rem clamp(1.1rem, 4vw, 3rem) 1.35rem; border-bottom: 1px solid rgba(168,190,228,.14); background: rgba(9,17,31,.55); }
.brand-line { display: flex; align-items: center; gap: .85rem; }
.brand-mark { display: grid; place-items: center; width: 2.55rem; height: 2.55rem; border-radius: .75rem; color: #061325; background: linear-gradient(135deg,#76e8cc,#6fa4ff); font-weight: 900; font-size: 1.3rem; }
.eyebrow { margin: 0 0 .25rem; color: #83b9ff; font-size: .68rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
h1, h2, p { margin-top: 0; } h1 { margin-bottom: 0; font-size: clamp(1.5rem,3vw,2rem); letter-spacing: -.04em; } h1 span { color: #8fa2c1; font-weight: 500; } h2 { margin: 0; font-size: 1rem; }
.subtitle { margin: 0; color: #9aaac5; font-size: .88rem; }
.stats-bar { display: flex; flex-wrap: wrap; gap: 1.25rem; color: #8fa2c1; font-size: .78rem; } .stat strong { color: #f2f7ff; margin-right: .25rem; }
.main { display: grid; grid-template-columns: minmax(18rem, 23rem) minmax(0, 1fr); gap: 1.25rem; padding: clamp(1.1rem, 3vw, 2.25rem); max-width: 1500px; margin: 0 auto; }
.panel { min-width: 0; border: 1px solid rgba(168,190,228,.17); border-radius: 1rem; background: rgba(16,28,49,.78); box-shadow: 0 12px 32px rgba(0,0,0,.15); }
.input-panel { align-self: start; padding: 1.1rem; } .visualization-panel { min-height: 620px; overflow: hidden; }
.panel-heading { display: flex; align-items: center; justify-content: space-between; gap: .75rem; margin-bottom: 1rem; } .visualization-heading { padding: 1.1rem 1.1rem 0; margin-bottom: 0; }
label { display: block; margin-bottom: .45rem; color: #d5e0f3; font-size: .8rem; font-weight: 650; }
textarea { width: 100%; min-height: 11rem; resize: vertical; border: 1px solid rgba(168,190,228,.25); border-radius: .65rem; color: #eff5ff; background: rgba(5,14,29,.72); padding: .8rem; line-height: 1.5; } textarea::placeholder { color: #71839e; }
.hint { margin: .45rem 0 1rem; color: #71839e; font-size: .72rem; } .input-actions { display: flex; gap: .6rem; } .btn { flex: 1; min-height: 2.65rem; border: 0; border-radius: .65rem; color: #f6f9ff; font-weight: 700; cursor: pointer; transition: .15s; } .btn:disabled { opacity: .5; cursor: not-allowed; } .btn-primary { background: #2366dd; } .btn-primary:hover:not(:disabled) { background: #3478ef; } .btn-danger { background: #963c50; } .btn-danger:hover:not(:disabled) { background: #b94c61; }
.spinner { display: inline-block; width: .9rem; height: .9rem; margin-right: .4rem; border: 2px solid rgba(255,255,255,.3); border-top-color: #fff; border-radius: 50%; animation: spin .8s linear infinite; } @keyframes spin { to { transform: rotate(360deg); } }
.status-dot, .live-badge i { display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; background: #53627a; } .status-dot.active, .live-badge i { background: #76e8cc; box-shadow: 0 0 10px #76e8cc; } .live-badge { color: #76e8cc; font-size: .72rem; } .live-badge i { margin-right: .35rem; }
.error-message { margin: .8rem 0 0; padding: .65rem .75rem; border: 1px solid rgba(255,126,139,.4); border-radius: .55rem; color: #ffd2d9; background: rgba(126,33,51,.3); font-size: .78rem; }
@media (max-width: 900px) { .main { grid-template-columns: 1fr; } .input-panel { order: 2; } .visualization-panel { order: 1; min-height: 500px; } }
</style>
