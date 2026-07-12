<template>
  <div class="space-visualization">
    <svg class="space-svg" :viewBox="`0 0 ${width} ${height}`" role="img" aria-label="Карта слов">
      <defs>
        <pattern id="space-grid" width="44" height="44" patternUnits="userSpaceOnUse"><path d="M 44 0 L 0 0 0 44" fill="none" stroke="#7da3d8" stroke-opacity=".08" stroke-width="1" /></pattern>
        <radialGradient id="word-low"><stop stop-color="#76a9ff" stop-opacity=".95"/><stop offset=".55" stop-color="#3977e8" stop-opacity=".35"/><stop offset="1" stop-color="#3977e8" stop-opacity="0"/></radialGradient>
        <radialGradient id="word-mid"><stop stop-color="#ffc961" stop-opacity=".95"/><stop offset=".55" stop-color="#dc851f" stop-opacity=".35"/><stop offset="1" stop-color="#dc851f" stop-opacity="0"/></radialGradient>
        <radialGradient id="word-high"><stop stop-color="#ff8290" stop-opacity=".98"/><stop offset=".55" stop-color="#d44560" stop-opacity=".4"/><stop offset="1" stop-color="#d44560" stop-opacity="0"/></radialGradient>
        <filter id="glow"><feGaussianBlur stdDeviation="5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      <rect width="100%" height="100%" fill="#081322" />
      <rect width="100%" height="100%" fill="url(#space-grid)" />
      <g v-if="displayedWords.length" class="connections">
        <line v-for="(conn, i) in connections" :key="i" v-bind="conn" />
      </g>
      <g v-if="displayedWords.length" class="nodes">
        <g v-for="word in displayedWords" :key="word.word" class="word-node" :transform="`translate(${word.x},${word.y})`">
          <circle :r="haloRadius(word.mass)" :fill="gradientId(word.mass)" class="node-halo" :opacity="haloOpacity(word.mass)" />
          <circle :r="nodeRadius(word.mass)" :fill="nodeColor(word.mass)" class="node-core" filter="url(#glow)" />
          <circle :r="nodeRadius(word.mass)" fill="none" stroke="#dceaff" stroke-opacity=".65" />
          <text class="node-label" text-anchor="middle" dominant-baseline="middle">{{ word.word }}</text>
          <text class="node-mass" text-anchor="middle" :y="nodeRadius(word.mass) + 16">× {{ word.mass.toFixed(1) }}</text>
        </g>
      </g>
      <g v-else class="empty-state"><circle :cx="width/2" :cy="height/2" r="54" fill="none" stroke="#6fa4ff" stroke-opacity=".25" stroke-dasharray="4 8"/><text :x="width/2" :y="height/2 - 5" text-anchor="middle">Пространство пусто</text><text :x="width/2" :y="height/2 + 18" text-anchor="middle" class="empty-subtitle">Введите текст слева, чтобы начать</text></g>
    </svg>
    <div class="legend"><span><i class="blue"></i>новое</span><span><i class="yellow"></i>повторяется</span><span><i class="red"></i>ядро</span></div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
interface Word { word: string; mass: number; x: number; y: number }
interface Props { words: Word[]; width?: number; height?: number }
const props = withDefaults(defineProps<Props>(), { width: 1000, height: 700 })
const displayedWords = ref<Word[]>([])
let animationFrame = 0

function animateWords(next: Word[]) {
  cancelAnimationFrame(animationFrame)
  const old = new Map(displayedWords.value.map(w => [w.word, w]))
  const start = next.map(w => ({ ...w, ...(old.get(w.word) ?? { x: props.width / 2, y: props.height / 2 }) }))
  if (!displayedWords.value.length) { displayedWords.value = next.map(w => ({ ...w })); return }
  const started = performance.now()
  const tick = (now: number) => {
    const progress = Math.min(1, (now - started) / 900)
    const eased = 1 - Math.pow(1 - progress, 3)
    displayedWords.value = next.map((w, index) => ({ ...w, x: start[index].x + (w.x - start[index].x) * eased, y: start[index].y + (w.y - start[index].y) * eased, mass: start[index].mass + (w.mass - start[index].mass) * eased }))
    if (progress < 1) animationFrame = requestAnimationFrame(tick)
  }
  animationFrame = requestAnimationFrame(tick)
}
watch(() => props.words, animateWords, { deep: true, immediate: true })

const connections = computed(() => {
  const lines: Array<Record<string, string | number>> = []
  for (let i = 0; i < displayedWords.value.length; i++) for (let j = i + 1; j < displayedWords.value.length; j++) {
    const a = displayedWords.value[i], b = displayedWords.value[j], d = Math.hypot(b.x - a.x, b.y - a.y)
    if (d < 310) lines.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: d < 180 ? '#76a9ff' : '#7890b6', 'stroke-width': d < 180 ? 1.5 : 1, 'stroke-opacity': Math.max(.08, .38 - d / 900), 'stroke-dasharray': d < 180 ? '0' : '4 8' })
  }
  return lines
})
function nodeRadius(mass: number) { return Math.min(15 + Math.log2(Math.max(mass, 1)) * 7, 34) }
function haloRadius(mass: number) { return Math.min(62 + Math.log2(Math.max(mass, 1)) * 24, 250) }
function haloOpacity(mass: number) { return Math.min(.62, .25 + Math.log2(Math.max(mass, 1)) * .11) }
function nodeColor(mass: number) { return mass >= 5 ? '#d44560' : mass >= 2 ? '#d99026' : '#3977e8' }
function gradientId(mass: number) { return mass >= 5 ? 'url(#word-high)' : mass >= 2 ? 'url(#word-mid)' : 'url(#word-low)' }
</script>

<style scoped lang="scss">
.space-visualization { position: relative; width: 100%; height: calc(100% - 3.5rem); min-height: 500px; overflow: hidden; background: #081322; }
.space-svg { display: block; width: 100%; height: 100%; }
.connections line { pointer-events: none; transition: x1 .9s, x2 .9s, y1 .9s, y2 .9s; }
.word-node { cursor: default; transition: filter .2s; } .word-node:hover { filter: brightness(1.2); } .node-halo { transition: r .25s; } .node-core { opacity: .95; }
.node-label { fill: #f0f6ff; font-size: 13px; font-weight: 750; paint-order: stroke; stroke: #0a1424; stroke-width: 4px; stroke-linejoin: round; pointer-events: none; }
.node-mass { fill: #91a9cd; font-size: 10px; pointer-events: none; } .empty-state text { fill: #c9d8ee; font-size: 15px; font-weight: 700; } .empty-state .empty-subtitle { fill: #71839e; font-size: 12px; font-weight: 400; }
.legend { position: absolute; right: 1rem; bottom: 1rem; display: flex; flex-wrap: wrap; gap: .75rem; padding: .55rem .7rem; border: 1px solid rgba(168,190,228,.16); border-radius: .6rem; color: #9aaac5; background: rgba(7,16,31,.8); font-size: .7rem; backdrop-filter: blur(8px); } .legend i { display: inline-block; width: .5rem; height: .5rem; margin-right: .3rem; border-radius: 50%; } .blue { background: #3977e8; } .yellow { background: #d99026; } .red { background: #d44560; }
</style>
