import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface Cloud {
  id: number
  token: string
  x: number
  y: number
  z: number
  mass: number
  density: number
  radius: number
  stability: number
  activation: number
  layerId: number
  layerName: string  // signal, character, word_form, lexeme, concept, scene, context
  cloudType: string
  color: string
  seed: number
  velocityX: number
  velocityY: number
  velocityZ: number
  fixed: boolean
  // Semantic overlay fields
  center_x?: number
  center_y?: number
  members?: Array<{lexeme_id: number; canonical_form: string; weight: number}>
  concept_name?: string
}

export interface SpaceInfo {
  id: number
  hostCloudId: number | null
  layerId: number
  mode: 'structural' | 'semantic'
  scale: number
}

export interface BreadcrumbItem {
  spaceId: number | null
  label: string
  mode: 'structural' | 'semantic' | 'root'
  layer: number
}

export interface PhysicsConfig {
  ticksPerSecond: number
  maxTicksPerStep: number
  paused: boolean
  speed: number
  singleStep: boolean
  triggerStep: boolean
}

export const useNebulaStore = defineStore('nebula', () => {
  // State
  const clouds = ref<Cloud[]>([])
  const spaces = ref<Map<number, SpaceInfo>>(new Map())
  const currentSpaceId = ref<number | null>(null)
  const currentMode = ref<'structural' | 'semantic'>('structural')
  const breadcrumb = ref<BreadcrumbItem[]>([])
  const selectedCloudId = ref<number | null>(null)
  const hoveredCloudIds = ref<number[]>([])
  const physicsConfig = ref<PhysicsConfig>({
    ticksPerSecond: 20,
    maxTicksPerStep: 5,
    paused: false,
    speed: 1.0,
    singleStep: false,
    triggerStep: false
  })
  const debugMode = ref(false)
  const liveMode = ref(true)
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  
  // Camera state (synced with renderer)
  const camera = ref({
    x: 800,
    y: 500,
    zoom: 1,
    targetX: 800,
    targetY: 500,
    targetZoom: 1
  })
  
  // Viewport
  const viewport = ref({
    minX: 0,
    minY: 0,
    maxX: 1600,
    maxY: 1000,
    width: 1600,
    height: 1000
  })
  
  // Training panel state
  const trainingPanelOpen = ref(true)
  const inspectorOpen = ref(false)
  
  // Computed
  const currentSpace = computed(() => 
    currentSpaceId.value ? spaces.value.get(currentSpaceId.value) || null : null
  )
  
  const selectedCloud = computed(() => 
    selectedCloudId.value ? clouds.value.find(c => c.id === selectedCloudId.value) || null : null
  )
  
  const hoveredClouds = computed(() => 
    clouds.value.filter(c => hoveredCloudIds.value.includes(c.id))
  )
  
  const visibleClouds = computed(() => {
    const vp = viewport.value
    return clouds.value.filter(c => {
      const r = c.radius * 1.5
      return c.x + r >= vp.minX && c.x - r <= vp.maxX &&
             c.y + r >= vp.minY && c.y - r <= vp.maxY
    })
  })
  
  const stats = computed(() => ({
    concepts: clouds.value.length,
    total_mass: clouds.value.reduce((sum, c) => sum + c.mass, 0),
    tokens: new Set(clouds.value.map(c => c.token)).size
  }))
  
  // Actions
  function setClouds(newClouds: Cloud[]) {
    clouds.value = newClouds
  }
  
  function addCloud(cloud: Cloud) {
    clouds.value.push(cloud)
  }
  
  function updateCloud(id: number, updates: Partial<Cloud>) {
    const idx = clouds.value.findIndex(c => c.id === id)
    if (idx >= 0) {
      clouds.value[idx] = { ...clouds.value[idx], ...updates }
    }
  }
  
  function removeCloud(id: number) {
    const idx = clouds.value.findIndex(c => c.id === id)
    if (idx >= 0) clouds.value.splice(idx, 1)
  }
  
  function setSpace(space: SpaceInfo) {
    spaces.value.set(space.id, space)
  }
  
  function setCurrentSpace(spaceId: number | null, mode: 'structural' | 'semantic' = 'structural') {
    currentSpaceId.value = spaceId
    currentMode.value = mode
    
    if (spaceId === null) {
      breadcrumb.value = [{ spaceId: null, label: 'Root', mode: 'root', layer: 0 }]
    } else {
      const space = spaces.value.get(spaceId)
      if (space) {
        const hostLabel = space.hostCloudId 
          ? clouds.value.find(c => c.id === space.hostCloudId)?.token || `Cloud ${space.hostCloudId}`
          : 'Root'
        const breadcrumbMode = mode === 'structural' ? 'structural' : 'semantic'
        breadcrumb.value = [
          { spaceId: null, label: 'Root', mode: 'root', layer: 0 },
          ...breadcrumb.value.filter(b => b.spaceId !== spaceId && b.mode !== 'root'),
          { spaceId, label: hostLabel, mode: breadcrumbMode, layer: space.layerId }
        ]
      }
    }
  }
  
  function navigateToBreadcrumb(index: number) {
    if (index < breadcrumb.value.length - 1) {
      const target = breadcrumb.value[index]
      if (target.mode === 'root') {
        setCurrentSpace(null)
      } else {
        setCurrentSpace(target.spaceId, target.mode)
      }
    }
  }
  
  function selectCloud(cloud: Cloud | null) {
    selectedCloudId.value = cloud?.id || null
    inspectorOpen.value = !!cloud
  }
  
  function setHoveredClouds(cloudIds: number[]) {
    hoveredCloudIds.value = cloudIds
  }
  
  function updateCamera(x: number, y: number, zoom: number) {
    camera.value.x = x
    camera.value.y = y
    camera.value.zoom = zoom
  }
  
  function setCameraTarget(targetX: number, targetY: number, targetZoom: number) {
    camera.value.targetX = targetX
    camera.value.targetY = targetY
    camera.value.targetZoom = targetZoom
  }
  
  function updateViewport(vp: typeof viewport.value) {
    viewport.value = vp
  }
  
  function setPhysicsConfig(config: Partial<PhysicsConfig>) {
    physicsConfig.value = { ...physicsConfig.value, ...config }
  }
  
  function togglePause() {
    physicsConfig.value.paused = !physicsConfig.value.paused
  }
  
  function stepPhysics() {
    physicsConfig.value.singleStep = true
    // Trigger a physics step - the backend will handle this
    physicsConfig.value.triggerStep = true
  }
  
  function setDebugMode(enabled: boolean) {
    debugMode.value = enabled
  }
  
  function setLiveMode(enabled: boolean) {
    liveMode.value = enabled
  }
  
  function setTrainingPanelOpen(open: boolean) {
    trainingPanelOpen.value = open
  }
  
  function setInspectorOpen(open: boolean) {
    inspectorOpen.value = open
  }
  
  function setLoading(loading: boolean) {
    isLoading.value = loading
  }
  
  function setError(err: string | null) {
    error.value = err
  }
  
  function clearError() {
    error.value = null
  }
  
  function reset() {
    clouds.value = []
    spaces.value.clear()
    currentSpaceId.value = null
    currentMode.value = 'structural'
    breadcrumb.value = []
    selectedCloudId.value = null
    hoveredCloudIds.value = []
    camera.value = { x: 800, y: 500, zoom: 1, targetX: 800, targetY: 500, targetZoom: 1 }
    viewport.value = { minX: 0, minY: 0, maxX: 1600, maxY: 1000, width: 1600, height: 1000 }
  }
  
  return {
    // State
    clouds,
    spaces,
    currentSpaceId,
    currentMode,
    breadcrumb,
    selectedCloudId,
    hoveredCloudIds,
    physicsConfig,
    debugMode,
    liveMode,
    isLoading,
    error,
    camera,
    viewport,
    trainingPanelOpen,
    inspectorOpen,
    // Computed
    currentSpace,
    selectedCloud,
    hoveredClouds,
    visibleClouds,
    stats,
    // Actions
    setClouds,
    addCloud,
    updateCloud,
    removeCloud,
    setSpace,
    setCurrentSpace,
    navigateToBreadcrumb,
    selectCloud,
    setHoveredClouds,
    updateCamera,
    setCameraTarget,
    updateViewport,
    setPhysicsConfig,
    togglePause,
    stepPhysics,
    setDebugMode,
    setLiveMode,
    setTrainingPanelOpen,
    setInspectorOpen,
    setLoading,
    setError,
    clearError,
    reset
  }
})