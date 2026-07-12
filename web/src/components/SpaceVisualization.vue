<template>
  <div class="space-visualization">
    <svg
      ref="svg"
      class="space-svg"
      :width="width"
      :height="height"
      :viewBox="viewBox"
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <radialGradient id="word-gradient-low" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
          <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.9" />
          <stop offset="70%" stop-color="#3b82f6" stop-opacity="0.3" />
          <stop offset="100%" stop-color="#3b82f6" stop-opacity="0" />
        </radialGradient>
        <radialGradient id="word-gradient-mid" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
          <stop offset="0%" stop-color="#f59e0b" stop-opacity="0.9" />
          <stop offset="70%" stop-color="#f59e0b" stop-opacity="0.3" />
          <stop offset="100%" stop-color="#f59e0b" stop-opacity="0" />
        </radialGradient>
        <radialGradient id="word-gradient-high" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
          <stop offset="0%" stop-color="#ef4444" stop-opacity="0.9" />
          <stop offset="70%" stop-color="#ef4444" stop-opacity="0.3" />
          <stop offset="100%" stop-color="#ef4444" stop-opacity="0" />
        </radialGradient>
      </defs>

      <!-- Connections between words in same phrases -->
      <g class="connections">
        <line
          v-for="(conn, i) in connections"
          :key="i"
          :x1="conn.x1"
          :y1="conn.y1"
          :x2="conn.x2"
          :y2="conn.y2"
          :stroke="conn.color"
          :stroke-width="conn.width"
          :stroke-opacity="conn.opacity"
          stroke-linecap="round"
        />
      </g>

      <!-- Word nodes -->
      <g class="nodes">
        <g
          v-for="word in words"
          :key="word.word"
          class="word-node"
          :transform="`translate(${word.x}, ${word.y})`"
        >
          <circle
            class="node-gradient"
            :r="nodeRadius(word.mass)"
            :fill="gradientId(word.mass)"
            :filter="massFilter(word.mass)"
          />
          <circle
            class="node-border"
            :r="nodeRadius(word.mass)"
            :stroke="nodeColor(word.mass)"
            stroke-width="1.5"
            fill="none"
          />
          <text
            class="node-label"
            :font-size="fontSize(word.mass)"
            :fill="labelColor(word.mass)"
            text-anchor="middle"
            dominant-baseline="middle"
            :dy="fontSize(word.mass) * 0.1"
          >
            {{ word.word }}
          </text>
        </g>
      </g>
    </svg>

    <div class="legend">
      <div class="legend-item">
        <div class="legend-color" style="background: linear-gradient(135deg, #3b82f6, #1d4ed8)"></div>
        <span>Масса ~1.0</span>
      </div>
      <div class="legend-item">
        <div class="legend-color" style="background: linear-gradient(135deg, #f59e0b, #d97706)"></div>
        <span>Масса ~2-4</span>
      </div>
      <div class="legend-item">
        <div class="legend-color" style="background: linear-gradient(135deg, #ef4444, #dc2626)"></div>
        <span>Масса 5+</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'

interface Word {
  word: string
  mass: number
  x: number
  y: number
}

interface Connection {
  x1: number
  y1: number
  x2: number
  y2: number
  color: string
  width: number
  opacity: number
}

interface Props {
  words: Word[]
  width?: number
  height?: number
}

const props = withDefaults(defineProps<Props>(), {
  width: 800,
  height: 500,
})

const svg = ref<SVGSVGElement | null>(null)

const connections = computed<Connection[]>(() => {
  const conns: Connection[] = []
  
  // Connect words that are close to each other (simulating phrase connections)
  for (let i = 0; i < props.words.length; i++) {
    for (let j = i + 1; j < props.words.length; j++) {
      const w1 = props.words[i]
      const w2 = props.words[j]
      const dx = w2.x - w1.x
      const dy = w2.y - w1.y
      const dist = Math.sqrt(dx * dx + dy * dy)
      
      // Only draw connections for reasonably close words
      if (dist < 300) {
        const avgMass = (w1.mass + w2.mass) / 2
        const opacity = Math.max(0.05, Math.min(0.3, 0.5 - dist / 600))
        const width = Math.max(0.5, Math.min(2, avgMass * 0.5))
        
        conns.push({
          x1: w1.x,
          y1: w1.y,
          x2: w2.x,
          y2: w2.y,
          color: avgMass >= 3 ? '#ef4444' : avgMass >= 1.5 ? '#f59e0b' : '#3b82f6',
          width,
          opacity,
        })
      }
    }
  }
  return conns
})

function nodeRadius(mass: number): number {
  return Math.min(12 + Math.log2(Math.max(mass, 1)) * 6, 30)
}

function fontSize(mass: number): number {
  return Math.min(10 + Math.log2(Math.max(mass, 1)) * 2, 14)
}

function nodeColor(mass: number): string {
  if (mass >= 5) return '#ef4444'
  if (mass >= 2) return '#f59e0b'
  return '#3b82f6'
}

function labelColor(mass: number): string {
  if (mass >= 5) return '#fff'
  if (mass >= 2) return '#1f2937'
  return '#1e3a8a'
}

function gradientId(mass: number): string {
  if (mass >= 5) return 'url(#word-gradient-high)'
  if (mass >= 2) return 'url(#word-gradient-mid)'
  return 'url(#word-gradient-low)'
}

function massFilter(mass: number): string {
  if (mass >= 3) return 'drop-shadow(0 0 8px currentColor)'
  return 'none'
}

const viewBox = computed(() => {
  if (props.words.length === 0) return `0 0 ${props.width} ${props.height}`
  
  const padding = 60
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  
  for (const w of props.words) {
    const r = nodeRadius(w.mass)
    minX = Math.min(minX, w.x - r)
    maxX = Math.max(maxX, w.x + r)
    minY = Math.min(minY, w.y - r)
    maxY = Math.max(maxY, w.y + r)
  }
  
  if (minX === Infinity) return `0 0 ${props.width} ${props.height}`
  
  const width = maxX - minX + padding * 2
  const height = maxY - minY + padding * 2
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  
  return `${cx - width / 2} ${cy - height / 2} ${width} ${height}`
})

function adjustView() {
  // viewBox is computed automatically
}

watch(() => props.words, adjustView, { deep: true })
onMounted(() => {
  if (props.words.length > 0) adjustView()
})
</script>

<style scoped lang="scss">
.space-visualization {
  width: 100%;
  height: 100%;
  min-height: 500px;
  background: #fff;
  border-radius: 8px;
  overflow: hidden;
  position: relative;
}

.space-svg {
  width: 100%;
  height: 100%;
  display: block;
}

.word-node {
  cursor: default;
  transition: transform 0.2s, filter 0.2s;
}

.word-node:hover {
  transform: scale(1.1);
  filter: drop-shadow(0 4px 8px rgba(0,0,0,0.15));
}

.word-node:hover .node-border {
  stroke-width: 2;
}

.connections line {
  pointer-events: none;
}

.node-gradient {
  transition: filter 0.2s;
}

.node-label {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-weight: 500;
  pointer-events: none;
  user-select: none;
  text-shadow: 0 1px 2px rgba(255,255,255,0.8);
}

.node-label:hover {
  text-shadow: 0 1px 4px rgba(0,0,0,0.2);
}

.legend {
  position: absolute;
  bottom: 16px;
  right: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px 16px;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  backdrop-filter: blur(4px);
  font-size: 11px;
  color: #3a3f4b;
  z-index: 10;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.legend-color {
  width: 14px;
  height: 14px;
  border-radius: 3px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
</style>