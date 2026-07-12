<template>
  <div class="space-visualization" @wheel.prevent="onWheel">
    <svg class="space-svg" :viewBox="`0 0 ${width} ${height}`" role="img" aria-label="Карта слов">
      <defs>
        <pattern id="space-grid" width="44" height="44" patternUnits="userSpaceOnUse"><path d="M 44 0 L 0 0 0 44" fill="none" stroke="#7da3d8" stroke-opacity=".08" /></pattern>
        <radialGradient id="gravity-low"><stop stop-color="#76a9ff" stop-opacity=".9"/><stop offset=".5" stop-color="#3977e8" stop-opacity=".25"/><stop offset="1" stop-color="#3977e8" stop-opacity="0"/></radialGradient>
        <radialGradient id="gravity-high"><stop stop-color="#ffbd6d" stop-opacity=".95"/><stop offset=".48" stop-color="#df5d54" stop-opacity=".28"/><stop offset="1" stop-color="#df5d54" stop-opacity="0"/></radialGradient>
        <filter id="glow"><feGaussianBlur stdDeviation="5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      <rect width="100%" height="100%" fill="#081322" />
      <rect width="100%" height="100%" fill="url(#space-grid)" />
      <g :transform="`translate(${width / 2} ${height / 2}) scale(${zoom}) translate(${-width / 2} ${-height / 2})`">
        <g class="connections">
          <line v-for="edge in displayedConnections" :key="`${edge.word_a}-${edge.word_b}`" :x1="edge.x1" :y1="edge.y1" :x2="edge.x2" :y2="edge.y2" :stroke-width="edge.strokeWidth" :stroke-opacity="edge.opacity" />
        </g>
        <g v-if="displayedWords.length" class="nodes">
          <g v-for="word in displayedWords" :key="word.word" class="word-node" :transform="`translate(${word.x},${word.y})`">
            <circle :r="haloRadius(word)" :fill="gradientId(word)" class="node-halo" />
            <circle :r="permeabilityRadius(word)" fill="none" stroke="#b4d0ff" :stroke-opacity="permeabilityOpacity(word)" stroke-width="2" :stroke-dasharray="(word.permeability ?? .5) > .72 ? '2 9' : '8 4'" class="permeability-boundary" />
            <circle :r="CORE_RADIUS" :fill="nodeColor(word)" class="node-core" filter="url(#glow)" />
            <circle :r="CORE_RADIUS" fill="none" stroke="#dceaff" stroke-opacity=".7" />
            <text class="node-label" text-anchor="middle" dominant-baseline="middle">{{ word.word }}</text>
            <text class="node-mass" text-anchor="middle" :y="CORE_RADIUS + 16">× {{ word.mass.toFixed(1) }}</text>
            <title>{{ word.word }} · частота {{ word.frequency ?? 1 }} · гравитация {{ (word.gravity ?? 1).toFixed(2) }} · проницаемость {{ Math.round((word.permeability ?? .5) * 100) }}%</title>
          </g>
        </g>
      </g>
      <g v-if="!displayedWords.length" class="empty-state"><circle :cx="width/2" :cy="height/2" r="54" fill="none" stroke="#6fa4ff" stroke-opacity=".25" stroke-dasharray="4 8"/><text :x="width/2" :y="height/2 - 5" text-anchor="middle">Пространство пусто</text><text :x="width/2" :y="height/2 + 18" text-anchor="middle" class="empty-subtitle">Введите текст слева, чтобы начать</text></g>
    </svg>
    <div class="zoom-controls" aria-label="Масштаб карты"><button @click="zoomOut" aria-label="Уменьшить">−</button><span>{{ Math.round(zoom * 100) }}%</span><button @click="zoomIn" aria-label="Увеличить">+</button><button class="fit" @click="zoom = 1" aria-label="Сбросить масштаб">Сбросить</button></div>
    <div class="legend"><span><i class="line"></i>связь</span><span><i class="boundary permeable"></i>проницаемость</span><span><i class="boundary firm"></i>граница ядра</span></div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
interface Word { word: string; mass: number; frequency?: number; halo?: number; permeability?: number; gravity?: number; x: number; y: number }
interface Connection { word_a: string; word_b: string; strength: number; contexts: number }
interface Props { words: Word[]; connections?: Connection[]; width?: number; height?: number }
const props = withDefaults(defineProps<Props>(), { connections: () => [], width: 1000, height: 700 })
const displayedWords = ref<Word[]>([])
const zoom = ref(1)
const CORE_RADIUS = 25
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
  return [{ ...edge, x1: a.x, y1: a.y, x2: b.x, y2: b.y, strokeWidth: Math.min(8, 1.2 + edge.strength * 1.35), opacity: Math.min(.8, .16 + edge.strength * .12) }]
}))
function haloRadius(word: Word) { return Math.min(250, 55 + (word.halo ?? 0) * 145 + Math.min(word.gravity ?? 1, 8) * 5) }
function permeabilityRadius(word: Word) { return CORE_RADIUS + 11 + (word.halo ?? 0) * 42 }
function permeabilityOpacity(word: Word) { return .22 + (1 - (word.permeability ?? .5)) * .6 }
function nodeColor(word: Word) { return (word.gravity ?? 1) >= 4 ? '#d96a52' : (word.gravity ?? 1) >= 2 ? '#d99026' : '#3977e8' }
function gradientId(word: Word) { return (word.gravity ?? 1) >= 4 ? 'url(#gravity-high)' : 'url(#gravity-low)' }
function zoomIn() { zoom.value = Math.min(3, +(zoom.value + .2).toFixed(1)) }
function zoomOut() { zoom.value = Math.max(.45, +(zoom.value - .2).toFixed(1)) }
function onWheel(event: WheelEvent) { event.deltaY < 0 ? zoomIn() : zoomOut() }
</script>

<style scoped lang="scss">
.space-visualization { position: relative; width: 100%; height: calc(100% - 3.5rem); min-height: 500px; overflow: hidden; background: #081322; }
.space-svg { display: block; width: 100%; height: 100%; }
.connections line { fill: none; stroke: #75a8ff; stroke-linecap: round; pointer-events: none; }
.word-node { cursor: default; transition: filter .2s; } .word-node:hover { filter: brightness(1.2); } .node-halo { transition: r .25s; } .node-core { opacity: .95; }
.node-label { fill: #f0f6ff; font-size: 13px; font-weight: 750; paint-order: stroke; stroke: #0a1424; stroke-width: 4px; stroke-linejoin: round; pointer-events: none; }
.node-mass { fill: #91a9cd; font-size: 10px; pointer-events: none; } .empty-state text { fill: #c9d8ee; font-size: 15px; font-weight: 700; } .empty-state .empty-subtitle { fill: #71839e; font-size: 12px; font-weight: 400; }
.zoom-controls { position: absolute; left: 1rem; bottom: 1rem; display: flex; align-items: center; gap: .35rem; padding: .4rem; border: 1px solid rgba(168,190,228,.16); border-radius: .6rem; background: rgba(7,16,31,.8); backdrop-filter: blur(8px); } .zoom-controls button { min-width: 1.7rem; min-height: 1.7rem; border: 0; border-radius: .35rem; color: #e4edff; background: #19345d; cursor: pointer; } .zoom-controls .fit { padding: 0 .45rem; font-size: .68rem; } .zoom-controls span { min-width: 3rem; color: #9aaac5; text-align: center; font-size: .7rem; }
.legend { position: absolute; right: 1rem; bottom: 1rem; display: flex; flex-wrap: wrap; gap: .75rem; padding: .55rem .7rem; border: 1px solid rgba(168,190,228,.16); border-radius: .6rem; color: #9aaac5; background: rgba(7,16,31,.8); font-size: .7rem; backdrop-filter: blur(8px); } .legend i { display: inline-block; width: .8rem; height: .5rem; margin-right: .3rem; vertical-align: middle; } .line { background: #75a8ff; } .boundary { border: 1px dashed #b4d0ff; border-radius: 50%; } .permeable { opacity: .35; } .firm { opacity: .8; border-style: solid; }
</style>
