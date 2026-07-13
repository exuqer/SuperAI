<template>
  <div class="space-visualization" @wheel.prevent="onWheel">
    <svg class="space-svg" :class="{ panning: isPanning }" :viewBox="`0 0 ${width} ${height}`" role="img" aria-label="Градиентное поле понятий" @pointerdown="startPan" @pointermove="movePan" @pointerup="endPan" @pointercancel="endPan" @pointerleave="endPan">
      <defs>
        <pattern id="space-grid" width="44" height="44" patternUnits="userSpaceOnUse"><path d="M 44 0 L 0 0 0 44" fill="none" stroke="#7da3d8" stroke-opacity=".08" /></pattern>
        <radialGradient id="concept-field"><stop stop-color="#b9d5ff" stop-opacity=".92"/><stop offset=".22" stop-color="#619cff" stop-opacity=".55"/><stop offset=".62" stop-color="#3977e8" stop-opacity=".2"/><stop offset="1" stop-color="#3977e8" stop-opacity="0"/></radialGradient>
        <radialGradient id="concept-core"><stop stop-color="#fff4d5"/><stop offset=".4" stop-color="#ffb861"/><stop offset="1" stop-color="#d94f62"/></radialGradient>
        <filter id="glow"><feGaussianBlur stdDeviation="5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      <rect width="100%" height="100%" fill="#081322" />
      <rect width="100%" height="100%" fill="url(#space-grid)" />
      <g :transform="`translate(${width / 2 + pan.x} ${height / 2 + pan.y}) scale(${zoom}) translate(${-width / 2} ${-height / 2})`">
        <g v-if="displayedConcepts.length" class="concepts">
          <g v-for="concept in displayedConcepts" :key="concept.id" class="concept-node" :class="{ selected: selectedConcept?.id === concept.id }" :transform="`translate(${xy(concept)[0]},${xy(concept)[1]})`">
            <circle :r="fieldRadius(concept)" fill="url(#concept-field)" class="concept-field" />
            <circle :r="coreRadius(concept)" fill="url(#concept-core)" class="concept-core" filter="url(#glow)" @pointerdown.stop @click.stop="selectConcept(concept)" />
            <circle :r="coreRadius(concept)" fill="none" stroke="#e7f1ff" stroke-opacity=".82" />
            <text class="concept-label" text-anchor="middle" dominant-baseline="middle">{{ concept.token }}</text>
            <text class="concept-mass" text-anchor="middle" :y="coreRadius(concept) + 16">× {{ concept.mass.toFixed(1) }}</text>
            <title>{{ concept.token }} · масса {{ concept.mass.toFixed(2) }} · радиус поля {{ fieldRadius(concept).toFixed(0) }}</title>
          </g>
        </g>
      </g>
      <g v-if="!displayedConcepts.length" class="empty-state"><circle :cx="width/2" :cy="height/2" r="54" fill="none" stroke="#6fa4ff" stroke-opacity=".25" stroke-dasharray="4 8"/><text :x="width/2" :y="height/2 - 5" text-anchor="middle">Пространство пусто</text><text :x="width/2" :y="height/2 + 18" text-anchor="middle" class="empty-subtitle">Введите текст слева, чтобы создать понятия</text></g>
    </svg>
    <aside v-if="selectedConcept" class="diagnostics">
      <div class="diagnostic-title">Понятие: <strong>{{ selectedConcept.token }}</strong></div>
      <div class="diagnostic-grid">
        <span>Масса<strong>{{ selectedConcept.mass.toFixed(2) }}</strong></span>
        <span>Активация<strong>{{ selectedConcept.activation.toFixed(2) }}</strong></span>
        <span>Координата X<strong>{{ xy(selectedConcept)[0].toFixed(1) }}</strong></span>
        <span>Координата Y<strong>{{ xy(selectedConcept)[1].toFixed(1) }}</strong></span>
        <span>Радиус поля<strong>{{ fieldRadius(selectedConcept).toFixed(1) }}</strong></span>
        <span>Размер ядра<strong>{{ coreRadius(selectedConcept).toFixed(1) }}</strong></span>
      </div>
      <button class="diagnostic-close" @click="clearSelection" aria-label="Закрыть диагностику">×</button>
    </aside>
    <div class="zoom-controls" aria-label="Масштаб карты"><button @click="zoomOut" aria-label="Уменьшить">−</button><span>{{ Math.round(zoom * 100) }}%</span><button @click="zoomIn" aria-label="Увеличить">+</button><button class="fit" @click="resetView" aria-label="Сбросить масштаб и положение">Сбросить</button></div>
    <div class="legend" aria-label="Легенда поля"><span><i class="field-swatch"></i>градиент понятия</span><span><i class="core-swatch"></i>абсолют понятия</span><span><i class="overlap-swatch"></i>области могут пересекаться</span></div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'

interface Concept { id: number; token: string; position: number[]; mass: number; radius: number; activation: number }
interface Props { concepts: Concept[]; width?: number; height?: number }

const props = withDefaults(defineProps<Props>(), { width: 1000, height: 700 })
const displayedConcepts = ref<Concept[]>([])
const selectedConcept = ref<Concept | null>(null)
const zoom = ref(1)
const pan = ref({ x: 0, y: 0 })
const isPanning = ref(false)
const panStart = ref({ x: 0, y: 0 })
const panOrigin = ref({ x: 0, y: 0 })
let animationFrame = 0

function xy(concept: Concept): [number, number] {
  return [Number.isFinite(concept.position?.[0]) ? concept.position[0] : props.width / 2, Number.isFinite(concept.position?.[1]) ? concept.position[1] : props.height / 2]
}

function fieldRadius(concept: Concept) { return Math.min(250, Math.max(24, concept.radius || 22 + 12 * Math.sqrt(Math.max(.001, concept.mass)))) }
function coreRadius(concept: Concept) { return Math.min(28, 10 + 3 * Math.sqrt(Math.max(.001, concept.mass))) }

function animateConcepts(next: Concept[]) {
  cancelAnimationFrame(animationFrame)
  const old = new Map(displayedConcepts.value.map(concept => [concept.id, concept]))
  if (!displayedConcepts.value.length) { displayedConcepts.value = next.map(concept => ({ ...concept, position: [...concept.position] })); return }
  const start = next.map(concept => ({ ...concept, position: [...(old.get(concept.id)?.position ?? [props.width / 2, props.height / 2])] }))
  const started = performance.now()
  const tick = (now: number) => {
    const progress = Math.min(1, (now - started) / 900)
    const eased = 1 - Math.pow(1 - progress, 3)
    displayedConcepts.value = next.map((concept, index) => ({
      ...concept,
      position: [
        start[index].position[0] + (concept.position[0] - start[index].position[0]) * eased,
        start[index].position[1] + (concept.position[1] - start[index].position[1]) * eased,
      ],
      mass: start[index].mass + (concept.mass - start[index].mass) * eased,
    }))
    if (progress < 1) animationFrame = requestAnimationFrame(tick)
  }
  animationFrame = requestAnimationFrame(tick)
}

watch(() => props.concepts, animateConcepts, { deep: true, immediate: true })
function zoomIn() { zoom.value = Math.min(3, +(zoom.value + .2).toFixed(1)) }
function zoomOut() { zoom.value = Math.max(.45, +(zoom.value - .2).toFixed(1)) }
function onWheel(event: WheelEvent) { event.deltaY < 0 ? zoomIn() : zoomOut() }
function startPan(event: PointerEvent) {
  if (event.button !== 0) return
  isPanning.value = true
  panStart.value = { x: event.clientX, y: event.clientY }
  panOrigin.value = { ...pan.value }
  ;(event.currentTarget as SVGSVGElement).setPointerCapture(event.pointerId)
}
function movePan(event: PointerEvent) {
  if (!isPanning.value) return
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  pan.value = { x: panOrigin.value.x + (event.clientX - panStart.value.x) * props.width / rect.width / zoom.value, y: panOrigin.value.y + (event.clientY - panStart.value.y) * props.height / rect.height / zoom.value }
}
function endPan(event?: PointerEvent) {
  if (event && (event.currentTarget as SVGSVGElement).hasPointerCapture(event.pointerId)) (event.currentTarget as SVGSVGElement).releasePointerCapture(event.pointerId)
  isPanning.value = false
}
function resetView() { zoom.value = 1; pan.value = { x: 0, y: 0 } }
function selectConcept(concept: Concept) { selectedConcept.value = concept }
function clearSelection() { selectedConcept.value = null }
</script>

<style scoped lang="scss">
.space-visualization { position: relative; width: 100%; height: calc(100% - 3.5rem); min-height: 500px; overflow: hidden; background: #081322; }
.space-svg { display: block; width: 100%; height: 100%; overflow: visible; cursor: grab; touch-action: none; } .space-svg.panning { cursor: grabbing; }
.concept-node { cursor: pointer; transition: filter .2s; } .concept-node:hover, .concept-node.selected { filter: brightness(1.25); } .concept-field { transition: r .25s; } .concept-core { opacity: .95; }
.concept-field, .concept-label, .concept-mass { pointer-events: none; }
.concept-label { fill: #f0f6ff; font-size: 13px; font-weight: 750; paint-order: stroke; stroke: #0a1424; stroke-width: 4px; stroke-linejoin: round; pointer-events: none; }
.concept-mass { fill: #91a9cd; font-size: 10px; pointer-events: none; } .empty-state text { fill: #c9d8ee; font-size: 15px; font-weight: 700; } .empty-state .empty-subtitle { fill: #71839e; font-size: 12px; font-weight: 400; }
.zoom-controls { position: absolute; left: 1rem; bottom: 1rem; display: flex; align-items: center; gap: .35rem; padding: .4rem; border: 1px solid rgba(168,190,228,.16); border-radius: .6rem; background: rgba(7,16,31,.8); backdrop-filter: blur(8px); } .zoom-controls button { min-width: 1.7rem; min-height: 1.7rem; border: 0; border-radius: .35rem; color: #e4edff; background: #19345d; cursor: pointer; } .zoom-controls .fit { padding: 0 .45rem; font-size: .68rem; } .zoom-controls span { min-width: 3rem; color: #9aaac5; text-align: center; font-size: .7rem; }
.legend { position: absolute; right: 1rem; bottom: 1rem; display: flex; flex-wrap: wrap; gap: .55rem .85rem; max-width: min(42rem, calc(100% - 2rem)); padding: .55rem .7rem; border: 1px solid rgba(168,190,228,.16); border-radius: .6rem; color: #9aaac5; background: rgba(7,16,31,.8); font-size: .7rem; backdrop-filter: blur(8px); } .legend i { display: inline-block; width: .8rem; height: .65rem; margin-right: .3rem; vertical-align: middle; border-radius: 50%; } .field-swatch { background: #3977e8; box-shadow: 0 0 8px #3977e8; } .core-swatch { background: #ffb861; box-shadow: 0 0 8px #ffb861; } .overlap-swatch { border: 1px solid #8ab7ff; background: rgba(57,119,232,.2); }
.diagnostics { position: absolute; top: 1rem; right: 1rem; width: min(19rem, calc(100% - 2rem)); padding: .85rem; border: 1px solid rgba(168,190,228,.24); border-radius: .75rem; color: #c8d7ef; background: rgba(7,16,31,.92); box-shadow: 0 10px 30px rgba(0,0,0,.25); backdrop-filter: blur(10px); font-size: .72rem; }
.diagnostic-title { padding-right: 1.3rem; margin-bottom: .7rem; color: #9bb8e9; } .diagnostic-title strong { color: #fff; }
.diagnostic-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .45rem .75rem; } .diagnostic-grid span { color: #8196b8; } .diagnostic-grid strong { display: block; margin-top: .12rem; color: #f0f6ff; font-size: .8rem; }
.diagnostic-close { position: absolute; top: .45rem; right: .5rem; border: 0; color: #94a9ca; background: none; cursor: pointer; font-size: 1.1rem; }
</style>
