<template>
  <div class="space-visualization" @wheel.prevent="onWheel">
    <svg class="space-svg" :class="{ panning: isPanning }" :viewBox="`0 0 ${width} ${height}`" role="img" aria-label="Карта слов" @pointerdown="startPan" @pointermove="movePan" @pointerup="endPan" @pointercancel="endPan" @pointerleave="endPan">
      <defs>
        <pattern id="space-grid" width="44" height="44" patternUnits="userSpaceOnUse"><path d="M 44 0 L 0 0 0 44" fill="none" stroke="#7da3d8" stroke-opacity=".08" /></pattern>
        <radialGradient id="gravity-low"><stop stop-color="#76a9ff" stop-opacity=".9"/><stop offset=".5" stop-color="#3977e8" stop-opacity=".25"/><stop offset="1" stop-color="#3977e8" stop-opacity="0"/></radialGradient>
        <radialGradient id="gravity-high"><stop stop-color="#ffbd6d" stop-opacity=".95"/><stop offset=".48" stop-color="#df5d54" stop-opacity=".28"/><stop offset="1" stop-color="#df5d54" stop-opacity="0"/></radialGradient>
        <filter id="glow"><feGaussianBlur stdDeviation="5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      <rect width="100%" height="100%" fill="#081322" />
      <rect width="100%" height="100%" fill="url(#space-grid)" />
      <g :transform="`translate(${width / 2 + pan.x} ${height / 2 + pan.y}) scale(${zoom}) translate(${-width / 2} ${-height / 2})`">
        <g class="connections">
          <g v-for="edge in displayedConnections" :key="`${edge.word_a}-${edge.word_b}`" class="connection-hit-area" @pointerdown.stop @click.stop="selectPair(edge)">
            <line class="connection-hit-target" :x1="edge.x1" :y1="edge.y1" :x2="edge.x2" :y2="edge.y2" :stroke-width="14" />
            <line class="connection-line" :x1="edge.x1" :y1="edge.y1" :x2="edge.x2" :y2="edge.y2" :stroke-width="edge.strokeWidth" :stroke-opacity="edge.opacity" />
          </g>
        </g>
        <g v-if="displayedWords.length" class="nodes">
          <g v-for="word in displayedWords" :key="word.word" class="word-node" :class="{ selected: selectedWord?.word === word.word }" :transform="`translate(${word.x},${word.y})`">
            <circle :r="haloRadius(word)" :fill="gradientId(word)" class="node-halo" />
            <circle :r="permeabilityRadius(word)" fill="none" stroke="#b4d0ff" :stroke-opacity="permeabilityOpacity(word)" stroke-width="2" :stroke-dasharray="(word.permeability ?? .5) > .72 ? '2 9' : '8 4'" class="permeability-boundary" />
            <circle :r="CORE_RADIUS" :fill="nodeColor(word)" class="node-core" filter="url(#glow)" @pointerdown.stop @click.stop="selectWord(word)" />
            <circle :r="CORE_RADIUS" fill="none" stroke="#dceaff" stroke-opacity=".7" />
            <text class="node-label" text-anchor="middle" dominant-baseline="middle">{{ word.word }}</text>
            <text class="node-mass" text-anchor="middle" :y="CORE_RADIUS + 16">× {{ word.mass.toFixed(1) }}</text>
            <title>{{ word.word }} · частота {{ word.frequency ?? 1 }} · гравитация {{ (word.gravity ?? 1).toFixed(2) }} · проницаемость {{ Math.round((word.permeability ?? .5) * 100) }}%</title>
          </g>
        </g>
      </g>
      <g v-if="!displayedWords.length" class="empty-state"><circle :cx="width/2" :cy="height/2" r="54" fill="none" stroke="#6fa4ff" stroke-opacity=".25" stroke-dasharray="4 8"/><text :x="width/2" :y="height/2 - 5" text-anchor="middle">Пространство пусто</text><text :x="width/2" :y="height/2 + 18" text-anchor="middle" class="empty-subtitle">Введите текст слева, чтобы начать</text></g>
    </svg>
    <aside v-if="selectedWord || selectedPair" class="diagnostics">
      <template v-if="selectedWord">
        <div class="diagnostic-title">Слово: <strong>{{ selectedWord.word }}</strong></div>
        <div class="diagnostic-grid">
          <span>Частота<strong>{{ selectedWord.frequency ?? 1 }}</strong></span>
          <span>Масса<strong>{{ selectedWord.mass.toFixed(1) }}</strong></span>
          <span>Размер ядра<strong>{{ CORE_RADIUS }}</strong></span>
          <span>Размер ореола<strong>{{ haloRadius(selectedWord).toFixed(0) }}</strong></span>
          <span>Гравитация<strong>{{ (selectedWord.gravity ?? 0).toFixed(2) }}</strong></span>
          <span>Проницаемость<strong>{{ (selectedWord.permeability ?? 0).toFixed(2) }}</strong></span>
          <span>Наблюдений<strong>{{ selectedWord.observations ?? 0 }}</strong></span>
          <span>Разных предложений<strong>{{ selectedWord.distinct_sentences ?? 0 }}</strong></span>
          <span>Разных контекстов<strong>{{ selectedWord.distinct_contexts ?? 0 }}</strong></span>
          <span>Уверенность<strong>{{ (selectedWord.confidence ?? 0).toFixed(2) }}</strong></span>
          <span>Уникальных соседей<strong>{{ selectedWord.unique_neighbors ?? neighborCount(selectedWord.word) }}</strong></span>
        </div>
      </template>
      <template v-else-if="selectedPair">
        <div class="diagnostic-title"><strong>{{ selectedPair.word_a }}</strong> ↔ <strong>{{ selectedPair.word_b }}</strong></div>
        <div class="diagnostic-grid">
          <span>Совместных появлений<strong>{{ selectedPair.contexts }}</strong></span>
          <span>Сила связи<strong>{{ selectedPair.strength.toFixed(2) }}</strong></span>
          <span>Уверенность связи<strong>{{ connectionConfidence(selectedPair) }}</strong></span>
          <span>Родственность<strong>{{ relatedness(selectedPair).toFixed(2) }}</strong></span>
          <span>Текущее расстояние<strong>{{ pairDistance(selectedPair).toFixed(0) }}</strong></span>
          <span>Притяжение<strong>{{ attraction(selectedPair).toFixed(2) }}</strong></span>
          <span>Отталкивание<strong>{{ repulsion(selectedPair).toFixed(2) }}</strong></span>
        </div>
      </template>
      <button class="diagnostic-close" @click="clearSelection" aria-label="Закрыть диагностику">×</button>
    </aside>
    <div class="zoom-controls" aria-label="Масштаб карты"><button @click="zoomOut" aria-label="Уменьшить">−</button><span>{{ Math.round(zoom * 100) }}%</span><button @click="zoomIn" aria-label="Увеличить">+</button><button class="fit" @click="resetView" aria-label="Сбросить масштаб и положение">Сбросить</button></div>
    <div class="legend" aria-label="Легенда карты">
      <span><i class="color blue"></i>низкая гравитация</span>
      <span><i class="color orange"></i>высокая гравитация</span>
      <span><i class="halo-swatch"></i>ореол — дальность влияния</span>
      <span><i class="line"></i>связь</span>
      <span><i class="boundary permeable"></i>проницаемость</span>
      <span><i class="boundary firm"></i>граница ядра</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
interface Word { word: string; mass: number; frequency?: number; halo?: number; permeability?: number; gravity?: number; unique_neighbors?: number; observations?: number; distinct_sentences?: number; distinct_contexts?: number; confidence?: number; x: number; y: number }
interface Connection { word_a: string; word_b: string; strength: number; contexts: number }
interface Props { words: Word[]; connections?: Connection[]; width?: number; height?: number }
const props = withDefaults(defineProps<Props>(), { connections: () => [], width: 1000, height: 700 })
const displayedWords = ref<Word[]>([])
const zoom = ref(1)
const CORE_RADIUS = 24
const selectedWord = ref<Word | null>(null)
const selectedPair = ref<Connection | null>(null)
const pan = ref({ x: 0, y: 0 })
const isPanning = ref(false)
const panStart = ref({ x: 0, y: 0 })
const panOrigin = ref({ x: 0, y: 0 })
let animationFrame = 0

function animateWords(next: Word[]) {
  cancelAnimationFrame(animationFrame)
  const old = new Map(displayedWords.value.map(w => [w.word, w]))
  if (!displayedWords.value.length) { displayedWords.value = next.map(w => ({ ...w })); return }
  const start = next.map(w => ({ ...w, ...(old.get(w.word) ?? { x: props.width / 2, y: props.height / 2 }) }))
  const started = performance.now()
  const tick = (now: number) => {
    const progress = Math.min(1, (now - started) / 900), eased = 1 - Math.pow(1 - progress, 3)
    displayedWords.value = next.map((w, i) => ({ ...w, x: start[i].x + (w.x - start[i].x) * eased, y: start[i].y + (w.y - start[i].y) * eased, mass: start[i].mass + (w.mass - start[i].mass) * eased }))
    if (progress < 1) animationFrame = requestAnimationFrame(tick)
  }
  animationFrame = requestAnimationFrame(tick)
}
watch(() => props.words, animateWords, { deep: true, immediate: true })
const positions = computed(() => new Map(displayedWords.value.map(word => [word.word, word])))
const displayedConnections = computed(() => props.connections.flatMap(edge => {
  const a = positions.value.get(edge.word_a), b = positions.value.get(edge.word_b)
  if (!a || !b) return []
  return [{ ...edge, x1: a.x, y1: a.y, x2: b.x, y2: b.y, strokeWidth: Math.min(6, 1 + edge.strength * 1.1), opacity: Math.min(.8, .08 + edge.strength * .1) }]
}))
function haloRadius(word: Word) { return Math.min(150, 38 + (word.halo ?? 0) * 6) }
function permeabilityRadius(word: Word) { return CORE_RADIUS + 10 + (word.halo ?? 0) * 8 }
function permeabilityOpacity(word: Word) { return .22 + (1 - (word.permeability ?? .5)) * .6 }
function nodeColor(word: Word) { return (word.gravity ?? 1) >= 4 ? '#d96a52' : (word.gravity ?? 1) >= 2 ? '#d99026' : '#3977e8' }
function gradientId(word: Word) { return (word.gravity ?? 1) >= 4 ? 'url(#gravity-high)' : 'url(#gravity-low)' }
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
  const scaleX = props.width / (event.currentTarget as SVGSVGElement).getBoundingClientRect().width
  const scaleY = props.height / (event.currentTarget as SVGSVGElement).getBoundingClientRect().height
  pan.value = { x: panOrigin.value.x + (event.clientX - panStart.value.x) * scaleX / zoom.value, y: panOrigin.value.y + (event.clientY - panStart.value.y) * scaleY / zoom.value }
}
function endPan(event?: PointerEvent) {
  if (event && (event.currentTarget as SVGSVGElement).hasPointerCapture(event.pointerId)) (event.currentTarget as SVGSVGElement).releasePointerCapture(event.pointerId)
  isPanning.value = false
}
function resetView() { zoom.value = 1; pan.value = { x: 0, y: 0 } }
function selectWord(word: Word) { selectedWord.value = word; selectedPair.value = null }
function selectPair(edge: Connection) { selectedPair.value = edge; selectedWord.value = null }
function clearSelection() { selectedWord.value = null; selectedPair.value = null }
function neighborCount(word: string) { return props.connections.filter(edge => edge.word_a === word || edge.word_b === word).length }
function pairDistance(edge: Connection) { const a = positions.value.get(edge.word_a), b = positions.value.get(edge.word_b); return a && b ? Math.hypot(a.x - b.x, a.y - b.y) : 0 }
function relatedness(edge: Connection) { return Math.min(.99, edge.strength / (edge.strength + .22)) }
function connectionConfidence(edge: Connection) { const value = Math.min(1, edge.contexts / 5); return value < .35 ? 'низкая' : value < .75 ? 'средняя' : 'высокая' }
function attraction(edge: Connection) { return Math.min(1, edge.strength / 3) }
function repulsion(edge: Connection) { return Math.max(0, 88 - pairDistance(edge)) / 88 }
</script>

<style scoped lang="scss">
.space-visualization { position: relative; width: 100%; height: calc(100% - 3.5rem); min-height: 500px; overflow: hidden; background: #081322; }
.space-svg { display: block; width: 100%; height: 100%; overflow: visible; cursor: grab; touch-action: none; } .space-svg.panning { cursor: grabbing; }
.connections line { fill: none; stroke-linecap: round; } .connection-line { stroke: #75a8ff; } .connection-hit-area { cursor: pointer; } .connection-hit-target { stroke: transparent !important; stroke-opacity: 0; } .connection-hit-area:hover .connection-line { stroke: #b8d3ff; }
.word-node { cursor: pointer; transition: filter .2s; } .word-node:hover, .word-node.selected { filter: brightness(1.25); } .node-halo { transition: r .25s; } .node-core { opacity: .95; }
.node-halo, .permeability-boundary, .node-label, .node-mass { pointer-events: none; }
.node-label { fill: #f0f6ff; font-size: 13px; font-weight: 750; paint-order: stroke; stroke: #0a1424; stroke-width: 4px; stroke-linejoin: round; pointer-events: none; }
.node-mass { fill: #91a9cd; font-size: 10px; pointer-events: none; } .empty-state text { fill: #c9d8ee; font-size: 15px; font-weight: 700; } .empty-state .empty-subtitle { fill: #71839e; font-size: 12px; font-weight: 400; }
.zoom-controls { position: absolute; left: 1rem; bottom: 1rem; display: flex; align-items: center; gap: .35rem; padding: .4rem; border: 1px solid rgba(168,190,228,.16); border-radius: .6rem; background: rgba(7,16,31,.8); backdrop-filter: blur(8px); } .zoom-controls button { min-width: 1.7rem; min-height: 1.7rem; border: 0; border-radius: .35rem; color: #e4edff; background: #19345d; cursor: pointer; } .zoom-controls .fit { padding: 0 .45rem; font-size: .68rem; } .zoom-controls span { min-width: 3rem; color: #9aaac5; text-align: center; font-size: .7rem; }
.legend { position: absolute; right: 1rem; bottom: 1rem; display: flex; flex-wrap: wrap; gap: .55rem .85rem; max-width: min(42rem, calc(100% - 2rem)); padding: .55rem .7rem; border: 1px solid rgba(168,190,228,.16); border-radius: .6rem; color: #9aaac5; background: rgba(7,16,31,.8); font-size: .7rem; backdrop-filter: blur(8px); } .legend i { display: inline-block; width: .8rem; height: .5rem; margin-right: .3rem; vertical-align: middle; } .color { border-radius: 50%; box-shadow: 0 0 7px currentColor; } .blue { color: #3977e8; background: #3977e8; } .orange { color: #d99026; background: #d99026; } .halo-swatch { width: 1rem !important; height: .75rem !important; border: 1px solid rgba(118,169,255,.6); border-radius: 50%; background: rgba(57,119,232,.18); box-shadow: 0 0 7px rgba(57,119,232,.65); } .line { background: #75a8ff; } .boundary { border: 1px dashed #b4d0ff; border-radius: 50%; } .permeable { opacity: .35; } .firm { opacity: .8; border-style: solid; }
.diagnostics { position: absolute; top: 1rem; right: 1rem; width: min(19rem, calc(100% - 2rem)); padding: .85rem; border: 1px solid rgba(168,190,228,.24); border-radius: .75rem; color: #c8d7ef; background: rgba(7,16,31,.92); box-shadow: 0 10px 30px rgba(0,0,0,.25); backdrop-filter: blur(10px); font-size: .72rem; }
.diagnostic-title { padding-right: 1.3rem; margin-bottom: .7rem; color: #9bb8e9; } .diagnostic-title strong { color: #fff; }
.diagnostic-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .45rem .75rem; } .diagnostic-grid span { color: #8196b8; } .diagnostic-grid strong { display: block; margin-top: .12rem; color: #f0f6ff; font-size: .8rem; }
.diagnostic-close { position: absolute; top: .45rem; right: .5rem; border: 0; color: #94a9ca; background: none; cursor: pointer; font-size: 1.1rem; }
</style>
