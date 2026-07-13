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

    <div class="top-bar">
      <div class="breadcrumb-container">
        <div v-if="breadcrumb.length" class="breadcrumb-bar">
          <span 
            v-for="(crumb, i) in breadcrumb" 
            :key="i" 
            class="breadcrumb-item"
            :class="{ active: i === breadcrumb.length - 1 }"
            @click="navigateToBreadcrumb(i)"
          >
            {{ crumb.label }}
            <span v-if="i < breadcrumb.length - 1" class="breadcrumb-sep">›</span>
          </span>
        </div>
      </div>
      <div class="top-controls">
        <div class="mode-toggle">
          <button 
            v-for="mode in modeButtons" 
            :key="mode"
            :class="{ active: currentMode === mode }"
            @click="switchMode(mode)"
            class="mode-btn"
          >
            {{ mode === 'structural' ? 'Structural' : 'Semantic' }}
          </button>
        </div>
        <div class="live-toggle">
          <label class="toggle-label">
            <input type="checkbox" :checked="liveMode" @change="toggleLive" />
            <span class="toggle-slider"></span>
            <span>Live</span>
          </label>
        </div>
        <div class="layer-display">
          <span class="layer-label">Layer:</span>
          <span class="layer-value">{{ currentLayer }}</span>
        </div>
      </div>
    </div>

    <main class="main" :class="{ 'has-inspector': inspectorOpen, 'panel-collapsed': !trainingPanelOpen }">
      <aside class="panel left-panel" :class="{ collapsed: !trainingPanelOpen }">
        <div class="panel-heading">
          <div><p class="eyebrow">01 / Input</p><h2>Текст для обучения</h2></div>
          <button class="collapse-btn" @click="toggleTrainingPanel" :aria-expanded="trainingPanelOpen">
            {{ trainingPanelOpen ? '−' : '+' }}
          </button>
        </div>
        <div v-if="trainingPanelOpen" class="panel-content">
          <label for="input-text">Введите фразу или несколько предложений</label>
          <textarea id="input-text" v-model="inputText" placeholder="Например: Кот ест рыбу. Рыба ест кота." rows="6" @keydown.meta.enter="handleLearn" @keydown.ctrl.enter="handleLearn" />
          <p class="hint">⌘ / Ctrl + Enter — запустить обучение</p>
          <div class="input-actions">
            <button class="btn btn-primary" @click="handleLearn" :disabled="loading || !inputText.trim()">
              <span v-if="loading" class="spinner"></span>{{ loading ? 'Считаю поле…' : 'Обучить модель' }}
            </button>
            <button class="btn btn-danger" @click="handleReset" :disabled="loading">Очистить</button>
          </div>
          <p v-if="errorMessage" class="error-message">{{ errorMessage }}</p>
        </div>
        <div v-else class="collapsed-label">Ввод</div>
      </aside>

      <section class="panel visualization-panel">
        <div class="panel-heading visualization-heading">
          <div><p class="eyebrow">02 / Field</p><h2>Градиентное поле</h2></div>
          <span class="live-badge"><i></i> live</span>
        </div>
        <div class="renderer-container">
          <NebulaRenderer 
            ref="rendererRef"
            :clouds="nebulaClouds"
            :space-id="currentSpaceId"
            :mode="currentMode"
            :debug-mode="debugMode"
            @cloud-select="onCloudSelect"
            @cloud-hover="onCloudHover"
            @double-click="onDoubleClick"
            @viewport-change="onViewportChange"
            @camera-change="onCameraChange"
          />
        </div>
      </section>

      <aside class="panel right-panel" v-show="inspectorOpen">
        <div class="panel-heading">
          <div><p class="eyebrow">03 / Inspector</p><h2>Инспектор</h2></div>
          <button class="close-btn" @click="closeInspector">×</button>
        </div>
        <div class="panel-content">
          <div v-if="selectedCloud" class="inspector-content">
            <h3>{{ selectedCloud.token }}</h3>
            <div class="inspector-grid">
              <div><span>Mass</span><strong>{{ selectedCloud.mass.toFixed(2) }}</strong></div>
              <div><span>Density</span><strong>{{ selectedCloud.density.toFixed(2) }}</strong></div>
              <div><span>Radius</span><strong>{{ selectedCloud.radius.toFixed(1) }}</strong></div>
              <div><span>Stability</span><strong>{{ selectedCloud.stability.toFixed(2) }}</strong></div>
              <div><span>Activation</span><strong>{{ selectedCloud.activation.toFixed(2) }}</strong></div>
              <div><span>Layer</span><strong>{{ selectedCloud.layerId }}</strong></div>
              <div><span>Type</span><strong>{{ selectedCloud.cloudType }}</strong></div>
            </div>
          </div>
          <div v-else class="inspector-empty">
            <p>Click a cloud to inspect</p>
          </div>
        </div>
      </aside>
    </main>

    <footer class="bottom-bar">
      <div class="zoom-controls">
        <button @click="zoomOut" aria-label="Zoom out">−</button>
        <span>{{ Math.round(cameraZoom * 100) }}%</span>
        <button @click="zoomIn" aria-label="Zoom in">+</button>
        <button class="fit" @click="resetView" aria-label="Reset view">Reset</button>
      </div>
      <div class="physics-controls">
        <label>
          <input type="checkbox" v-model="physicsPaused" @change="togglePause" />
          <span>Pause</span>
        </label>
        <label>
          Speed: <input type="range" min="0.1" max="3" step="0.1" v-model="physicsSpeed" @input="updatePhysicsSpeed" />
          <span>{{ physicsSpeed.toFixed(1) }}x</span>
        </label>
        <button class="btn" @click="stepPhysics">Single Tick</button>
      </div>
      <div class="debug-controls">
        <label>
          <input type="checkbox" v-model="debugMode" @change="toggleDebug" />
          <span>Debug</span>
        </label>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref, computed } from 'vue'
import { useTrainingStore } from '@/stores/training'
import { useNebulaStore } from '@/stores/nebula'
import NebulaRenderer from '@/components/NebulaRenderer.vue'

const trainingStore = useTrainingStore()
const nebulaStore = useNebulaStore()

const rendererRef = ref<InstanceType<typeof NebulaRenderer> | null>(null)

const inputText = ref('')
const loading = ref(false)
const errorMessage = ref('')

// Computed from nebula store
const nebulaClouds = computed(() => nebulaStore.clouds)
const currentSpaceId = computed(() => nebulaStore.currentSpaceId)
const currentMode = computed(() => nebulaStore.currentMode)
const breadcrumb = computed(() => nebulaStore.breadcrumb)
const debugMode = computed(() => nebulaStore.debugMode)
const liveMode = computed(() => nebulaStore.liveMode)
const trainingPanelOpen = computed(() => nebulaStore.trainingPanelOpen)
const inspectorOpen = computed(() => nebulaStore.inspectorOpen)
const selectedCloud = computed(() => nebulaStore.selectedCloud)
const physicsConfig = computed(() => nebulaStore.physicsConfig)
const cameraZoom = computed(() => nebulaStore.camera.zoom)
const currentLayer = computed(() => nebulaStore.currentSpace?.layerId ?? 0)

const stats = computed(() => nebulaStore.stats)

async function loadSpace() {
  nebulaStore.setLoading(true)
  nebulaStore.clearError()
  try {
    const result = await trainingStore.getSpace()
    // Convert legacy concepts to nebula clouds
    const clouds = result.concepts.map((c: any) => ({
      id: c.id,
      token: c.token,
      x: c.position?.[0] ?? 800,
      y: c.position?.[1] ?? 500,
      z: 0,
      mass: c.mass,
      density: Math.min(1, c.mass / 10),
      radius: c.radius || (22 + 12 * Math.sqrt(Math.max(0.001, c.mass))),
      stability: 0.5,
      activation: c.activation || 0,
      layerId: 0,
      cloudType: 'concept',
      color: '',
      seed: c.id,
      velocityX: 0,
      velocityY: 0,
      velocityZ: 0,
      fixed: false
    }))
    nebulaStore.setClouds(clouds)
    nebulaStore.setCurrentSpace(null)
  } catch (error: any) {
    nebulaStore.setError(error.message || 'Не удалось загрузить пространство')
  } finally {
    nebulaStore.setLoading(false)
  }
}

async function handleLearn() {
  if (!inputText.value.trim()) return
  nebulaStore.setLoading(true)
  nebulaStore.clearError()
  try {
    const result = await trainingStore.learn(inputText.value)
    const clouds = result.concepts.map((c: any) => ({
      id: c.id,
      token: c.token,
      x: c.position?.[0] ?? 800,
      y: c.position?.[1] ?? 500,
      z: 0,
      mass: c.mass,
      density: Math.min(1, c.mass / 10),
      radius: c.radius || (22 + 12 * Math.sqrt(Math.max(0.001, c.mass))),
      stability: 0.5,
      activation: c.activation || 0,
      layerId: 0,
      cloudType: 'concept',
      color: '',
      seed: c.id,
      velocityX: 0,
      velocityY: 0,
      velocityZ: 0,
      fixed: false
    }))
    nebulaStore.setClouds(clouds)
    inputText.value = ''
  } catch (error: any) {
    nebulaStore.setError(error.message || 'Ошибка обучения')
  } finally {
    nebulaStore.setLoading(false)
  }
}

async function handleReset() {
  if (!confirm('Очистить всё пространство понятий?')) return
  nebulaStore.setLoading(true)
  nebulaStore.clearError()
  try {
    await trainingStore.resetSpace()
    nebulaStore.reset()
  } catch (error: any) {
    nebulaStore.setError(error.message || 'Ошибка сброса')
  } finally {
    nebulaStore.setLoading(false)
  }
}

function onCloudSelect(cloud: any) {
  nebulaStore.selectCloud(cloud)
}

function onCloudHover(clouds: any[]) {
  nebulaStore.setHoveredClouds(clouds.map(c => c.id))
}

function onDoubleClick(_cloud: any) {
  // Handled by renderer with animation
}

function onViewportChange(viewport: any) {
  nebulaStore.updateViewport(viewport)
}

function onCameraChange(nextCamera: { x: number; y: number; zoom: number }) {
  nebulaStore.updateCamera(nextCamera.x, nextCamera.y, nextCamera.zoom)
}

function switchMode(mode: 'structural' | 'semantic') {
  nebulaStore.setCurrentSpace(nebulaStore.currentSpaceId, mode)
}

const modeButtons = ['structural', 'semantic'] as const

function toggleLive() {
  nebulaStore.setLiveMode(!liveMode.value)
}

function toggleTrainingPanel() {
  nebulaStore.setTrainingPanelOpen(!trainingPanelOpen.value)
}

function closeInspector() {
  nebulaStore.setInspectorOpen(false)
  nebulaStore.selectCloud(null)
}

function zoomIn() {
  rendererRef.value?.zoomBy(1.2)
}

function zoomOut() {
  rendererRef.value?.zoomBy(1 / 1.2)
}

function resetView() {
  rendererRef.value?.resetView()
}

function togglePause() {
  nebulaStore.togglePause()
}

function updatePhysicsSpeed() {
  nebulaStore.setPhysicsConfig({ speed: physicsSpeed.value })
}

function toggleDebug() {
  nebulaStore.setDebugMode(debugMode.value)
}

function navigateToBreadcrumb(index: number) {
  nebulaStore.navigateToBreadcrumb(index)
}

function stepPhysics() {
  nebulaStore.stepPhysics()
}

const physicsPaused = computed({
  get: () => physicsConfig.value.paused,
  set: (val) => nebulaStore.setPhysicsConfig({ paused: val })
})

const physicsSpeed = computed({
  get: () => physicsConfig.value.speed,
  set: (val) => nebulaStore.setPhysicsConfig({ speed: val })
})

onMounted(loadSpace)
</script>

<style scoped lang="scss">
.training-view { display: flex; flex-direction: column; min-height: 100vh; color: #eef4ff; }
.header { display: grid; gap: .7rem; padding: 2.1rem clamp(1.1rem, 4vw, 3rem) 1.35rem; border-bottom: 1px solid rgba(168,190,228,.14); background: rgba(9,17,31,.55); }
.brand-line { display: flex; align-items: center; gap: .85rem; }
.brand-mark { display: grid; place-items: center; width: 2.55rem; height: 2.55rem; border-radius: .75rem; color: #061325; background: linear-gradient(135deg,#76e8cc,#6fa4ff); font-weight: 900; font-size: 1.3rem; }
.eyebrow { margin: 0 0 .25rem; color: #83b9ff; font-size: .68rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
h1, h2, h3, p { margin-top: 0; } h1 { margin-bottom: 0; font-size: clamp(1.5rem,3vw,2rem); letter-spacing: -.04em; } h1 span { color: #8fa2c1; font-weight: 500; } h2 { margin: 0; font-size: 1rem; }
.subtitle { margin: 0; color: #9aaac5; font-size: .88rem; }
.stats-bar { display: flex; flex-wrap: wrap; gap: 1.25rem; color: #8fa2c1; font-size: .78rem; } .stat strong { color: #f2f7ff; margin-right: .25rem; }
.top-bar { display: flex; align-items: center; justify-content: space-between; gap: 1rem; min-height: 3.25rem; padding: .45rem clamp(1.1rem, 4vw, 3rem); border-bottom: 1px solid rgba(168,190,228,.14); background: rgba(7,16,30,.84); }
.breadcrumb-container { min-width: 0; overflow: auto hidden; }
.breadcrumb-bar { display: flex; align-items: center; gap: .35rem; white-space: nowrap; color: #8196b8; font-size: .74rem; }
.breadcrumb-item { display: inline-flex; align-items: center; gap: .35rem; padding: .3rem .45rem; border-radius: .35rem; cursor: pointer; transition: background .15s, color .15s; }
.breadcrumb-item:hover { color: #eef4ff; background: rgba(125,163,216,.12); }
.breadcrumb-item.active { color: #f2f7ff; font-weight: 700; }
.breadcrumb-sep { color: #53627a; }
.top-controls { display: flex; align-items: center; justify-content: flex-end; gap: .65rem; flex: 0 0 auto; }
.mode-toggle { display: flex; padding: .18rem; border: 1px solid rgba(168,190,228,.2); border-radius: .55rem; background: rgba(16,28,49,.78); }
.mode-btn, .collapse-btn, .close-btn { border: 0; color: #9aaac5; background: transparent; cursor: pointer; }
.mode-btn { padding: .32rem .7rem; border-radius: .38rem; font-size: .72rem; }
.mode-btn.active { color: #061325; background: linear-gradient(135deg,#76e8cc,#6fa4ff); font-weight: 700; }
.live-toggle { color: #76e8cc; font-size: .72rem; }
.toggle-label { display: flex; align-items: center; gap: .35rem; margin: 0; color: inherit; cursor: pointer; font-size: inherit; }
.toggle-label input { accent-color: #76e8cc; }
.layer-display { padding-left: .65rem; border-left: 1px solid rgba(168,190,228,.16); color: #8196b8; font-size: .72rem; }
.layer-value { margin-left: .25rem; color: #eef4ff; font-weight: 700; }
.main { display: grid; flex: 1 1 auto; grid-template-columns: minmax(18rem, 23rem) minmax(0, 1fr); gap: 1.25rem; align-items: stretch; width: 100%; min-height: 0; padding: clamp(1.1rem, 3vw, 2.25rem); max-width: 1700px; margin: 0 auto; }
.main.has-inspector { grid-template-columns: minmax(18rem, 22rem) minmax(0, 1fr) minmax(16rem, 20rem); }
.panel { min-width: 0; border: 1px solid rgba(168,190,228,.17); border-radius: 1rem; background: rgba(16,28,49,.78); box-shadow: 0 12px 32px rgba(0,0,0,.15); }
.left-panel { align-self: stretch; padding: 1.1rem; }
.left-panel.collapsed { display: flex; align-items: flex-start; justify-content: center; width: 3.2rem; padding: .7rem .45rem; }
.left-panel.collapsed .panel-heading { flex-direction: column; margin: 0; }
.left-panel.collapsed .panel-heading > div { display: none; }
.left-panel.collapsed .collapse-btn { display: grid; place-items: center; width: 2rem; height: 2rem; border-radius: .45rem; background: rgba(125,163,216,.12); }
.collapsed-label { writing-mode: vertical-rl; margin-top: 1.1rem; color: #8196b8; font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; }
.visualization-panel { display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
.panel-heading { display: flex; align-items: center; justify-content: space-between; gap: .75rem; margin-bottom: 1rem; } .visualization-heading { flex: 0 0 auto; padding: 1.1rem 1.1rem .85rem; margin-bottom: 0; background: rgba(16,28,49,.42); }
.renderer-container { flex: 1 1 auto; min-height: 0; }
.right-panel { align-self: start; padding: 1.1rem; }
.collapse-btn, .close-btn { width: 1.8rem; height: 1.8rem; border-radius: .4rem; font-size: 1.15rem; line-height: 1; }
.collapse-btn:hover, .close-btn:hover { color: #fff; background: rgba(125,163,216,.15); }
label { display: block; margin-bottom: .45rem; color: #d5e0f3; font-size: .8rem; font-weight: 650; }
textarea { width: 100%; min-height: 11rem; resize: vertical; border: 1px solid rgba(168,190,228,.25); border-radius: .65rem; color: #eff5ff; background: rgba(5,14,29,.72); padding: .8rem; line-height: 1.5; } textarea::placeholder { color: #71839e; }
.hint { margin: .45rem 0 1rem; color: #71839e; font-size: .72rem; } .input-actions { display: flex; gap: .6rem; } .btn { flex: 1; min-height: 2.65rem; border: 0; border-radius: .65rem; color: #f6f9ff; font-weight: 700; cursor: pointer; transition: .15s; } .btn:disabled { opacity: .5; cursor: not-allowed; } .btn-primary { background: #2366dd; } .btn-primary:hover:not(:disabled) { background: #3478ef; } .btn-danger { background: #963c50; } .btn-danger:hover:not(:disabled) { background: #b94c61; }
.spinner { display: inline-block; width: .9rem; height: .9rem; margin-right: .4rem; border: 2px solid rgba(255,255,255,.3); border-top-color: #fff; border-radius: 50%; animation: spin .8s linear infinite; } @keyframes spin { to { transform: rotate(360deg); } }
.status-dot, .live-badge i { display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; background: #53627a; } .status-dot.active, .live-badge i { background: #76e8cc; box-shadow: 0 0 10px #76e8cc; } .live-badge { color: #76e8cc; font-size: .72rem; } .live-badge i { margin-right: .35rem; }
.error-message { margin: .8rem 0 0; padding: .65rem .75rem; border: 1px solid rgba(255,126,139,.4); border-radius: .55rem; color: #ffd2d9; background: rgba(126,33,51,.3); font-size: .78rem; }
.inspector-content h3 { margin: 0 0 1rem; font-size: 1.2rem; }
.inspector-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .7rem; }
.inspector-grid span { display: block; color: #8196b8; font-size: .7rem; }
.inspector-grid strong { display: block; margin-top: .15rem; color: #f0f6ff; font-size: .88rem; }
.inspector-empty { color: #8196b8; font-size: .78rem; }
.bottom-bar { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: .65rem clamp(1.1rem, 4vw, 3rem); border-top: 1px solid rgba(168,190,228,.14); color: #9aaac5; background: rgba(9,17,31,.72); }
.zoom-controls, .physics-controls, .debug-controls { display: flex; align-items: center; gap: .5rem; }
.zoom-controls button, .physics-controls button { min-width: 2rem; min-height: 2rem; border: 1px solid rgba(168,190,228,.2); border-radius: .45rem; color: #dbeaff; background: rgba(16,28,49,.85); cursor: pointer; }
.zoom-controls button:hover, .physics-controls button:hover { border-color: rgba(118,232,204,.55); background: rgba(35,102,221,.45); }
.zoom-controls span { min-width: 3.7rem; color: #eef4ff; text-align: center; font-size: .75rem; font-variant-numeric: tabular-nums; }
.zoom-controls .fit { width: auto; padding: 0 .65rem; color: #9aaac5; font-size: .72rem; }
.physics-controls label, .debug-controls label { display: flex; align-items: center; gap: .35rem; margin: 0; color: #9aaac5; font-size: .72rem; font-weight: 400; }
.physics-controls input[type="range"] { width: 6rem; accent-color: #76e8cc; }
.physics-controls .btn { width: auto; padding: 0 .65rem; font-size: .72rem; }
@media (max-width: 1050px) { .main.has-inspector { grid-template-columns: minmax(15rem, 20rem) minmax(0, 1fr); } .right-panel { grid-column: 1 / -1; } }
@media (max-width: 900px) { .top-bar, .bottom-bar { align-items: flex-start; flex-direction: column; } .top-controls { width: 100%; justify-content: flex-start; flex-wrap: wrap; } .main, .main.has-inspector { flex: 0 0 auto; grid-template-columns: 1fr; } .left-panel, .left-panel.collapsed, .right-panel { width: auto; } .left-panel.collapsed { min-height: 3rem; justify-content: flex-start; } .left-panel.collapsed .panel-heading { flex-direction: row; } .left-panel.collapsed .collapsed-label { display: none; } .visualization-panel { min-height: 540px; } .bottom-bar { flex-wrap: wrap; } }
</style>
