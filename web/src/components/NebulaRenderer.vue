<template>
  <div class="nebula-renderer" ref="containerRef" @wheel.prevent="onWheel" @dblclick="onDoubleClick">
    <canvas
      ref="canvasRef"
      class="nebula-canvas"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @pointerleave="onPointerUp"
      @pointercancel="onPointerUp"
    />
    <div v-if="debugMode" class="debug-overlay">
      <div>FPS {{ fps }}</div>
      <div>Visible {{ visibleCount }}</div>
      <div>Draw calls {{ drawCalls }}</div>
      <div>LOD {{ lodCounts.join(' / ') }}</div>
      <div>Zoom {{ Math.round(camera.zoom * 100) }}%</div>
    </div>
    <div v-if="hoveredClouds.length && !selectedCloud" class="hover-contributors">
      <span>Вклад в точку</span>
      <strong v-for="cloud in hoveredClouds" :key="cloud.id">{{ cloud.token }}</strong>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'

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
  cloudType: string
  color: string
  seed: number
  velocityX?: number
  velocityY?: number
  velocityZ?: number
  fixed?: boolean
}

interface Viewport {
  minX: number
  minY: number
  maxX: number
  maxY: number
  width: number
  height: number
}

const props = withDefaults(defineProps<{
  clouds?: Cloud[]
  spaceId?: number | null
  mode?: 'structural' | 'semantic'
  debugMode?: boolean
}>(), {
  clouds: () => [],
  spaceId: null,
  mode: 'structural',
  debugMode: false
})

const emit = defineEmits<{
  'cloud-select': [cloud: Cloud | null]
  'cloud-hover': [clouds: Cloud[]]
  'double-click': [cloud: Cloud]
  'viewport-change': [viewport: Viewport]
  'camera-change': [camera: { x: number; y: number; zoom: number }]
}>()

const containerRef = ref<HTMLDivElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const ctx = ref<CanvasRenderingContext2D | null>(null)
const width = ref(0)
const height = ref(0)

const camera = ref({
  x: 0,
  y: 0,
  zoom: 1,
  targetX: 0,
  targetY: 0,
  targetZoom: 1,
  startX: 0,
  startY: 0,
  startZoom: 1,
  transitionStart: 0,
  transitionDuration: 450,
  transitioning: false
})

const selectedCloud = ref<Cloud | null>(null)
const hoveredCloud = ref<Cloud | null>(null)
const hoveredClouds = ref<Cloud[]>([])
const visibleCount = ref(0)
const drawCalls = ref(0)
const fps = ref(0)
const lodCounts = ref([0, 0, 0, 0])
const isPanning = ref(false)
const pointerMoved = ref(false)
const pointerStart = ref({ x: 0, y: 0 })
const panOrigin = ref({ x: 0, y: 0 })
const frameId = ref(0)
const lastFrame = ref(performance.now())
const frameSamples = ref<number[]>([])
const positions = new Map<number, { x: number; y: number }>()
const targetPositions = new Map<number, { x: number; y: number }>()
const shapeCache = new Map<number, ShapeData>()
let positionFrame = 0

interface ShapeData {
  clumps: { x: number; y: number; size: number; opacity: number; phase: number }[]
  particles: { x: number; y: number; size: number; opacity: number; phase: number }[]
  seed: number
}

const viewport = computed<Viewport>(() => ({
  minX: camera.value.x - width.value / (2 * camera.value.zoom),
  minY: camera.value.y - height.value / (2 * camera.value.zoom),
  maxX: camera.value.x + width.value / (2 * camera.value.zoom),
  maxY: camera.value.y + height.value / (2 * camera.value.zoom),
  width: width.value / camera.value.zoom,
  height: height.value / camera.value.zoom
}))

function seededRandom(seed: number): () => number {
  let value = (seed || 1) >>> 0
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0
    return value / 4294967296
  }
}

function hashString(value: string): number {
  let hash = 0
  for (let i = 0; i < value.length; i += 1) hash = ((hash << 5) - hash + value.charCodeAt(i)) | 0
  return Math.abs(hash)
}

function baseColor(cloud: Cloud): string {
  if (cloud.color) return cloud.color
  const layerHue: Record<number, number> = { 0: 216, 1: 184, 2: 148, 3: 72, 4: 35, 5: 8 }
  const hue = (layerHue[cloud.layerId] ?? ((cloud.layerId * 41) % 360)) + hashString(cloud.cloudType || 'concept') % 16 - 8
  return `hsl(${hue}, 76%, 62%)`
}

function withAlpha(color: string, alpha: number): string {
  if (color.startsWith('hsl')) return color.replace('hsl(', 'hsla(').replace(')', `, ${Math.max(0, Math.min(1, alpha))})`)
  if (color.startsWith('#') && color.length >= 7) {
    const r = parseInt(color.slice(1, 3), 16)
    const g = parseInt(color.slice(3, 5), 16)
    const b = parseInt(color.slice(5, 7), 16)
    return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, alpha))})`
  }
  return color
}

function shapeFor(cloud: Cloud): ShapeData {
  const cached = shapeCache.get(cloud.id)
  if (cached) return cached
  const random = seededRandom(cloud.seed || cloud.id)
  const clumps = Array.from({ length: 2 + Math.floor(random() * 4) }, () => {
    const angle = random() * Math.PI * 2
    const distance = 0.12 + random() * 0.48
    return { x: Math.cos(angle) * distance, y: Math.sin(angle) * distance, size: 0.13 + random() * 0.2, opacity: 0.35 + random() * 0.4, phase: random() * Math.PI * 2 }
  })
  const particles = Array.from({ length: Math.min(24, Math.max(6, Math.round(cloud.mass * 1.8))) }, () => {
    const angle = random() * Math.PI * 2
    const distance = 0.45 + random() * 0.62
    return { x: Math.cos(angle) * distance, y: Math.sin(angle) * distance, size: 0.012 + random() * 0.022, opacity: 0.18 + random() * 0.28, phase: random() * Math.PI * 2 }
  })
  const shape = { clumps, particles, seed: cloud.seed || cloud.id }
  shapeCache.set(cloud.id, shape)
  return shape
}

function cloudPosition(cloud: Cloud): { x: number; y: number } {
  const current = positions.get(cloud.id)
  return current || { x: Number.isFinite(cloud.x) ? cloud.x : camera.value.x, y: Number.isFinite(cloud.y) ? cloud.y : camera.value.y }
}

function setCloudTargets(nextClouds: Cloud[]) {
  const nextIds = new Set(nextClouds.map(cloud => cloud.id))
  for (const id of positions.keys()) if (!nextIds.has(id)) positions.delete(id)
  for (const cloud of nextClouds) {
    const target = { x: Number.isFinite(cloud.x) ? cloud.x : camera.value.x, y: Number.isFinite(cloud.y) ? cloud.y : camera.value.y }
    targetPositions.set(cloud.id, target)
    if (!positions.has(cloud.id)) positions.set(cloud.id, { ...target })
  }
  cancelAnimationFrame(positionFrame)
  const started = performance.now()
  const initial = new Map([...positions].map(([id, value]) => [id, { ...value }]))
  const animate = (now: number) => {
    const progress = Math.min(1, (now - started) / 650)
    const eased = 1 - Math.pow(1 - progress, 3)
    for (const [id, target] of targetPositions) {
      const from = initial.get(id) || target
      positions.set(id, { x: from.x + (target.x - from.x) * eased, y: from.y + (target.y - from.y) * eased })
    }
    if (progress < 1) positionFrame = requestAnimationFrame(animate)
  }
  positionFrame = requestAnimationFrame(animate)
}

function screenPosition(cloud: Cloud) {
  const point = cloudPosition(cloud)
  return { x: (point.x - camera.value.x) * camera.value.zoom + width.value / 2, y: (point.y - camera.value.y) * camera.value.zoom + height.value / 2 }
}

function screenToWorld(x: number, y: number) {
  return { x: (x - width.value / 2) / camera.value.zoom + camera.value.x, y: (y - height.value / 2) / camera.value.zoom + camera.value.y }
}

function screenRadius(cloud: Cloud) {
  return Math.max(6, Math.min(260, cloud.radius || 22 + 12 * Math.sqrt(Math.max(0.01, cloud.mass))) * camera.value.zoom)
}

function computeLOD(cloud: Cloud) {
  const radius = screenRadius(cloud)
  if (radius < 4) return 0
  if (radius < 16) return 1
  if (radius < 48) return 2
  return 3
}

function isVisible(cloud: Cloud) {
  const point = cloudPosition(cloud)
  const radius = (cloud.radius || 22) * 1.7
  return point.x + radius >= viewport.value.minX && point.x - radius <= viewport.value.maxX && point.y + radius >= viewport.value.minY && point.y - radius <= viewport.value.maxY
}

function drawCloud(cloud: Cloud, lod: number, time: number) {
  const context = ctx.value
  if (!context) return
  const point = screenPosition(cloud)
  const radius = screenRadius(cloud)
  const color = baseColor(cloud)
  const shape = shapeFor(cloud)
  const active = Math.max(0, Math.min(1, cloud.activation || 0))
  const density = Math.max(0.08, Math.min(1, cloud.density || cloud.mass / 10 || 0.2))
  const stability = Math.max(0, Math.min(1, cloud.stability ?? 0.5))
  const pulse = active * (0.05 + Math.sin(time * 4 + shape.seed) * 0.035)
  const radiusPulse = radius * (1 + pulse)
  const selected = selectedCloud.value?.id === cloud.id
  const hovered = hoveredCloud.value?.id === cloud.id

  context.save()
  context.translate(point.x, point.y)
  if (lod === 0) {
    context.globalCompositeOperation = 'screen'
    context.fillStyle = withAlpha(color, 0.35 + density * 0.45 + active * 0.2)
    context.shadowBlur = 10 + active * 12
    context.shadowColor = color
    context.beginPath()
    context.arc(0, 0, Math.max(1.5, radius * 0.18), 0, Math.PI * 2)
    context.fill()
    context.restore()
    drawCalls.value += 1
    return
  }

  const halo = context.createRadialGradient(0, 0, radius * 0.08, 0, 0, radiusPulse * (1.55 + (1 - stability) * 0.35))
  halo.addColorStop(0, withAlpha(color, 0.05 + density * 0.12 + active * 0.12))
  halo.addColorStop(0.35, withAlpha(color, 0.1 + density * 0.16 + active * 0.08))
  halo.addColorStop(1, withAlpha(color, 0))
  context.globalCompositeOperation = 'screen'
  context.fillStyle = halo
  context.beginPath()
  context.arc(0, 0, radiusPulse * (1.55 + (1 - stability) * 0.35), 0, Math.PI * 2)
  context.fill()

  if (lod >= 2) {
    for (const clump of shape.clumps) {
      const wobble = 1 + Math.sin(time * (1.3 + (1 - stability) * 2) + clump.phase) * 0.08 * (1 - stability)
      const x = clump.x * radius * wobble
      const y = clump.y * radius * wobble
      const size = clump.size * radius * wobble
      const gradient = context.createRadialGradient(x, y, 0, x, y, size)
      gradient.addColorStop(0, withAlpha(color, clump.opacity * (0.32 + density * 0.38 + active * 0.25)))
      gradient.addColorStop(0.65, withAlpha(color, clump.opacity * 0.12))
      gradient.addColorStop(1, withAlpha(color, 0))
      context.fillStyle = gradient
      context.beginPath()
      context.arc(x, y, size, 0, Math.PI * 2)
      context.fill()
    }
  }

  const core = context.createRadialGradient(0, 0, 0, 0, 0, radius * 0.58)
  core.addColorStop(0, withAlpha('#f7fbff', 0.72 + density * 0.2 + active * 0.08))
  core.addColorStop(0.12, withAlpha(color, 0.72 + density * 0.2))
  core.addColorStop(0.5, withAlpha(color, 0.2 + density * 0.26 + active * 0.16))
  core.addColorStop(1, withAlpha(color, 0))
  context.fillStyle = core
  context.globalAlpha = 0.72 + density * 0.24 + active * 0.12
  context.beginPath()
  context.arc(0, 0, radius * 0.58 * (1 + pulse * 1.5), 0, Math.PI * 2)
  context.fill()

  if (lod >= 3) {
    context.globalAlpha = 0.45 + active * 0.35
    for (const particle of shape.particles) {
      const drift = (1 - stability) * radius * 0.12
      const x = particle.x * radius + Math.sin(time * 0.7 + particle.phase) * drift
      const y = particle.y * radius + Math.cos(time * 0.9 + particle.phase) * drift
      context.fillStyle = withAlpha(color, particle.opacity + active * 0.12)
      context.beginPath()
      context.arc(x, y, Math.max(0.8, particle.size * radius), 0, Math.PI * 2)
      context.fill()
    }
  }

  if (selected || hovered) {
    context.globalAlpha = selected ? 0.9 : 0.62
    context.strokeStyle = selected ? '#dff8ff' : color
    context.lineWidth = selected ? 1.5 : 1
    context.setLineDash(selected ? [7, 5] : [3, 5])
    context.beginPath()
    context.arc(0, 0, radius * 1.18, 0, Math.PI * 2)
    context.stroke()
    context.setLineDash([])
  }
  context.restore()
  drawCalls.value += 1
}

function drawLabels(clouds: Cloud[]) {
  const context = ctx.value
  if (!context) return
  const occupied: { left: number; right: number; top: number; bottom: number }[] = []
  const sorted = [...clouds].sort((a, b) => {
    const score = (cloud: Cloud) => (selectedCloud.value?.id === cloud.id ? 10000 : 0) + (hoveredCloud.value?.id === cloud.id ? 5000 : 0) + (cloud.activation || 0) * 1000 + cloud.mass * 10 + (cloud.stability || 0)
    return score(b) - score(a)
  })
  for (const cloud of sorted) {
    const radius = screenRadius(cloud)
    const lod = computeLOD(cloud)
    if (lod === 0 || (radius < 11 && selectedCloud.value?.id !== cloud.id && hoveredCloud.value?.id !== cloud.id)) continue
    const point = screenPosition(cloud)
    const text = cloud.token
    context.font = `${selectedCloud.value?.id === cloud.id ? 700 : 600} ${Math.max(11, Math.min(16, 10 + radius * 0.08))}px system-ui, sans-serif`
    const textWidth = context.measureText(text).width
    const left = point.x - textWidth / 2 - 7
    const right = point.x + textWidth / 2 + 7
    const top = point.y - radius * 1.32 - 18
    const bottom = top + 21
    if (left < 8 || right > width.value - 8 || top < 8 || bottom > height.value - 8) continue
    if (occupied.some(box => left < box.right && right > box.left && top < box.bottom && bottom > box.top)) continue
    occupied.push({ left, right, top, bottom })
    context.save()
    context.textAlign = 'center'
    context.textBaseline = 'middle'
    context.fillStyle = 'rgba(5, 14, 29, .72)'
    context.beginPath()
    context.roundRect(left, top, right - left, bottom - top, 5)
    context.fill()
    context.fillStyle = selectedCloud.value?.id === cloud.id ? '#f4fbff' : '#dbeaff'
    context.fillText(text, point.x, top + 10.5)
    context.restore()
  }
}

function render() {
  const context = ctx.value
  if (!context) return
  const now = performance.now()
  const time = now / 1000
  if (camera.value.transitioning) {
    const progress = Math.min(1, (now - camera.value.transitionStart) / camera.value.transitionDuration)
    const eased = progress < 0.5 ? 4 * progress * progress * progress : 1 - Math.pow(-2 * progress + 2, 3) / 2
    camera.value.x = camera.value.startX + (camera.value.targetX - camera.value.startX) * eased
    camera.value.y = camera.value.startY + (camera.value.targetY - camera.value.startY) * eased
    camera.value.zoom = camera.value.startZoom + (camera.value.targetZoom - camera.value.startZoom) * eased
    if (progress >= 1) camera.value.transitioning = false
  }
  context.clearRect(0, 0, width.value, height.value)
  context.fillStyle = '#07111f'
  context.fillRect(0, 0, width.value, height.value)
  if (props.debugMode) drawGrid()
  const visible = props.clouds.filter(isVisible)
  const counts = [0, 0, 0, 0]
  visible.sort((a, b) => (a.activation - b.activation) || (a.mass - b.mass))
  context.globalCompositeOperation = 'lighter'
  for (const cloud of visible) {
    const lod = computeLOD(cloud)
    counts[lod] += 1
    drawCloud(cloud, lod, time)
  }
  context.globalCompositeOperation = 'source-over'
  drawLabels(visible)
  visibleCount.value = visible.length
  lodCounts.value = counts
  const delta = now - lastFrame.value
  frameSamples.value.push(delta)
  if (frameSamples.value.length > 45) frameSamples.value.shift()
  fps.value = Math.round(1000 / (frameSamples.value.reduce((sum, value) => sum + value, 0) / Math.max(1, frameSamples.value.length)))
  lastFrame.value = now
  frameId.value = requestAnimationFrame(render)
  emit('viewport-change', viewport.value)
  emit('camera-change', { x: camera.value.x, y: camera.value.y, zoom: camera.value.zoom })
}

function drawGrid() {
  const context = ctx.value
  if (!context) return
  const step = Math.max(32, 100 * camera.value.zoom)
  const originX = ((-camera.value.x * camera.value.zoom + width.value / 2) % step + step) % step
  const originY = ((-camera.value.y * camera.value.zoom + height.value / 2) % step + step) % step
  context.save()
  context.strokeStyle = 'rgba(125, 163, 216, .1)'
  context.lineWidth = 1
  for (let x = originX; x < width.value; x += step) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height.value); context.stroke() }
  for (let y = originY; y < height.value; y += step) { context.beginPath(); context.moveTo(0, y); context.lineTo(width.value, y); context.stroke() }
  context.restore()
}

function animateCamera(targetX: number, targetY: number, targetZoom: number, duration = 500) {
  camera.value.startX = camera.value.x
  camera.value.startY = camera.value.y
  camera.value.startZoom = camera.value.zoom
  camera.value.targetX = targetX
  camera.value.targetY = targetY
  camera.value.targetZoom = targetZoom
  camera.value.transitionStart = performance.now()
  camera.value.transitionDuration = duration
  camera.value.transitioning = true
}

function zoomBy(factor: number) {
  animateCamera(camera.value.x, camera.value.y, Math.max(0.22, Math.min(8, camera.value.zoom * factor)), 220)
}

function resetView() {
  animateCamera(width.value / 2, height.value / 2, 1, 450)
}

function onWheel(event: WheelEvent) {
  if (!width.value || !height.value) return
  const rect = canvasRef.value?.getBoundingClientRect()
  if (!rect) return
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top
  const before = screenToWorld(x, y)
  const factor = event.deltaY < 0 ? 1.22 : 1 / 1.22
  const nextZoom = Math.max(0.22, Math.min(8, camera.value.zoom * factor))
  const nextX = before.x - (x - width.value / 2) / nextZoom
  const nextY = before.y - (y - height.value / 2) / nextZoom
  animateCamera(nextX, nextY, nextZoom, 180)
}

function onPointerDown(event: PointerEvent) {
  if (event.button !== 0) return
  isPanning.value = true
  pointerMoved.value = false
  pointerStart.value = { x: event.clientX, y: event.clientY }
  panOrigin.value = { x: camera.value.x, y: camera.value.y }
  canvasRef.value?.setPointerCapture(event.pointerId)
}

function onPointerMove(event: PointerEvent) {
  const rect = canvasRef.value?.getBoundingClientRect()
  if (!rect) return
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top
  if (isPanning.value) {
    const dx = event.clientX - pointerStart.value.x
    const dy = event.clientY - pointerStart.value.y
    if (Math.abs(dx) + Math.abs(dy) > 5) pointerMoved.value = true
    if (pointerMoved.value) {
      camera.value.x = panOrigin.value.x - dx / camera.value.zoom
      camera.value.y = panOrigin.value.y - dy / camera.value.zoom
      camera.value.targetX = camera.value.x
      camera.value.targetY = camera.value.y
      camera.value.transitioning = false
    }
  }
  const world = screenToWorld(x, y)
  const contributors = props.clouds.map(cloud => {
    const point = cloudPosition(cloud)
    const sigma = Math.max(1, cloud.radius / 2)
    const distance = Math.hypot(point.x - world.x, point.y - world.y)
    return { cloud, density: (cloud.mass || 1) * Math.exp(-(distance * distance) / (2 * sigma * sigma)) }
  }).filter(item => item.density > 0.02).sort((a, b) => b.density - a.density).slice(0, 5)
  hoveredClouds.value = contributors.map(item => item.cloud)
  hoveredCloud.value = contributors[0]?.cloud || null
  emit('cloud-hover', hoveredClouds.value)
}

function onPointerUp(event?: PointerEvent) {
  if (event && canvasRef.value?.hasPointerCapture(event.pointerId)) canvasRef.value.releasePointerCapture(event.pointerId)
  if (isPanning.value && !pointerMoved.value) {
    const rect = canvasRef.value?.getBoundingClientRect()
    if (rect) {
      const world = screenToWorld(event ? event.clientX - rect.left : width.value / 2, event ? event.clientY - rect.top : height.value / 2)
      const closest = props.clouds.map(cloud => ({ cloud, distance: Math.hypot(cloudPosition(cloud).x - world.x, cloudPosition(cloud).y - world.y) / Math.max(1, cloud.radius) })).sort((a, b) => a.distance - b.distance)[0]
      if (closest && closest.distance <= 1.5) {
        selectedCloud.value = closest.cloud
        emit('cloud-select', closest.cloud)
      }
    }
  }
  isPanning.value = false
}

function onDoubleClick(event: MouseEvent) {
  const rect = canvasRef.value?.getBoundingClientRect()
  if (!rect) return
  const world = screenToWorld(event.clientX - rect.left, event.clientY - rect.top)
  const closest = props.clouds.map(cloud => ({ cloud, distance: Math.hypot(cloudPosition(cloud).x - world.x, cloudPosition(cloud).y - world.y) / Math.max(1, cloud.radius) })).sort((a, b) => a.distance - b.distance)[0]
  if (!closest || closest.distance > 1.8) return
  selectedCloud.value = closest.cloud
  emit('cloud-select', closest.cloud)
  emit('double-click', closest.cloud)
  const targetZoom = Math.max(1.5, Math.min(6, 230 / Math.max(24, closest.cloud.radius)))
  animateCamera(cloudPosition(closest.cloud).x, cloudPosition(closest.cloud).y, targetZoom, 560)
}

function resizeCanvas() {
  const canvas = canvasRef.value
  const container = containerRef.value
  if (!canvas || !container) return
  width.value = Math.max(1, container.clientWidth)
  height.value = Math.max(1, container.clientHeight)
  const dpr = Math.min(2, window.devicePixelRatio || 1)
  canvas.width = Math.round(width.value * dpr)
  canvas.height = Math.round(height.value * dpr)
  canvas.style.width = `${width.value}px`
  canvas.style.height = `${height.value}px`
  ctx.value?.setTransform(dpr, 0, 0, dpr, 0, 0)
  if (!camera.value.x && !camera.value.y) {
    camera.value.x = width.value / 2
    camera.value.y = height.value / 2
    camera.value.targetX = camera.value.x
    camera.value.targetY = camera.value.y
  }
}

watch(() => props.clouds, setCloudTargets, { deep: true, immediate: true })

defineExpose({ zoomBy, resetView })

onMounted(() => {
  ctx.value = canvasRef.value?.getContext('2d', { alpha: false, desynchronized: true }) || null
  resizeCanvas()
  window.addEventListener('resize', resizeCanvas)
  frameId.value = requestAnimationFrame(render)
})

onUnmounted(() => {
  cancelAnimationFrame(frameId.value)
  cancelAnimationFrame(positionFrame)
  window.removeEventListener('resize', resizeCanvas)
})
</script>

<style scoped lang="scss">
.nebula-renderer { position: relative; width: 100%; height: 100%; min-height: 520px; overflow: hidden; background: #07111f; }
.nebula-canvas { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }
.nebula-canvas:active { cursor: grabbing; }
.debug-overlay, .hover-contributors { position: absolute; z-index: 5; border: 1px solid rgba(168,190,228,.18); border-radius: .6rem; background: rgba(5,14,29,.82); backdrop-filter: blur(10px); pointer-events: none; }
.debug-overlay { left: .8rem; bottom: .8rem; padding: .55rem .7rem; color: #76e8cc; font: 11px/1.55 ui-monospace, SFMono-Regular, Consolas, monospace; }
.hover-contributors { right: .8rem; bottom: .8rem; display: flex; align-items: center; gap: .55rem; padding: .5rem .7rem; color: #8fa2c1; font-size: .7rem; }
.hover-contributors strong { color: #eef4ff; font-weight: 600; }
</style>
