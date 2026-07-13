<template>
  <div class="space-visualization" @wheel.prevent="onWheel">
    <svg class="space-svg" :class="{ panning: isPanning }" :viewBox="`0 0 ${width} ${height}`" role="img" aria-label="Градиентное поле понятий" @pointerdown="startPan" @pointermove="movePan" @pointerup="endPan" @pointercancel="endPan" @pointerleave="endPan">
      <defs>
        <pattern id="space-grid" width="44" height="44" patternUnits="userSpaceOnUse"><path d="M 44 0 L 0 0 0 44" fill="none" stroke="#7da3d8" stroke-opacity=".08" /></pattern>
        <radialGradient id="concept-field"><stop stop-color="#b9d5ff" stop-opacity=".92"/><stop offset=".22" stop-color="#619cff" stop-opacity=".55"/><stop offset=".62" stop-color="#3977e8" stop-opacity=".2"/><stop offset="1" stop-color="#3977e8" stop-opacity="0"/></radialGradient>
        <radialGradient id="concept-core"><stop stop-color="#fff4d5"/><stop offset=".4" stop-color="#ffb861"/><stop offset="1" stop-color="#d94f62"/></radialGradient>
        <radialGradient id="lexeme-field"><stop stop-color="#d5ffb9" stop-opacity=".7"/><stop offset=".22" stop-color="#9cff61" stop-opacity=".4"/><stop offset=".62" stop-color="#77e839" stop-opacity=".15"/><stop offset="1" stop-color="#77e839" stop-opacity="0"/></radialGradient>
        <radialGradient id="lexeme-core"><stop stop-color="#f4ffd5"/><stop offset=".4" stop-color="#b8ff61"/><stop offset="1" stop-color="#4fd962"/></radialGradient>
        <radialGradient id="scene-field"><stop stop-color="#ffd5b9" stop-opacity=".7"/><stop offset=".22" stop-color="#ff9c61" stop-opacity=".4"/><stop offset=".62" stop-color="#e87739" stop-opacity=".15"/><stop offset="1" stop-color="#e87739" stop-opacity="0"/></radialGradient>
        <radialGradient id="scene-core"><stop stop-color="#fff4d5"/><stop offset=".4" stop-color="#ffb861"/><stop offset="1" stop-color="#d97f4f"/></radialGradient>
        <filter id="glow"><feGaussianBlur stdDeviation="5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        <filter id="soft-glow"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      <rect width="100%" height="100%" fill="#081322" />
      <rect width="100%" height="100%" fill="url(#space-grid)" />
      <g :transform="`translate(${width / 2 + pan.x} ${height / 2 + pan.y}) scale(${zoom}) translate(${-width / 2} ${-height / 2})`">
        <!-- Semantic Overlays (Concept Projections) -->
        <g v-if="semanticOverlays.length" class="semantic-overlays">
          <g v-for="overlay in semanticOverlays" :key="overlay.id" class="semantic-overlay" :transform="`translate(${overlay.center_x},${overlay.center_y})`">
            <circle :r="overlay.radius" fill="url(#concept-field)" class="overlay-field" :style="{ opacity: overlayOpacity(overlay) }" />
            <circle :r="overlay.radius * 0.3" fill="url(#concept-core)" class="overlay-core" filter="url(#soft-glow)" :style="{ opacity: overlayOpacity(overlay) * 0.8 }" />
            <text class="overlay-label" text-anchor="middle" dominant-baseline="middle" :style="{ opacity: overlayLabelOpacity(overlay) }">
              {{ overlay.concept_name }}
            </text>
            <!-- Member lexemes as small dots -->
            <g v-for="member in overlay.members" :key="member.lexeme_id" class="overlay-member">
              <circle :cx="memberOffset(member, overlay).x" :cy="memberOffset(member, overlay).y" r="4" fill="#9cff61" :style="{ opacity: member.weight * overlayOpacity(overlay) }" />
            </g>
          </g>
        </g>
        
        <!-- Scenes -->
        <g v-if="displayedScenes.length" class="scenes">
          <g v-for="scene in displayedScenes" :key="scene.id" class="scene-node" :class="{ selected: selectedScene?.id === scene.id }" :transform="`translate(${xy(scene)[0]},${xy(scene)[1]})`">
            <circle :r="sceneFieldRadius(scene)" fill="url(#scene-field)" class="scene-field" :style="{ opacity: sceneOpacity(scene) }" />
            <circle :r="sceneCoreRadius(scene)" fill="url(#scene-core)" class="scene-core" filter="url(#soft-glow)" :style="{ opacity: sceneOpacity(scene) }" @pointerdown.stop @click.stop="selectScene(scene)" />
            <circle :r="sceneCoreRadius(scene)" fill="none" stroke="#ffe7d5" stroke-opacity=".6" />
            <text class="scene-label" text-anchor="middle" dominant-baseline="middle" :style="{ opacity: sceneLabelOpacity(scene) }">{{ scene.token }}</text>
            <title>{{ scene.token }} · масса {{ scene.mass.toFixed(2) }}</title>
          </g>
        </g>
        
        <!-- Concepts (main layer) -->
        <g v-if="displayedConcepts.length" class="concepts">
          <g v-for="concept in displayedConcepts" :key="concept.id" class="concept-node" :class="{ selected: selectedConcept?.id === concept.id }" :transform="`translate(${xy(concept)[0]},${xy(concept)[1]})`">
            <circle :r="fieldRadius(concept)" fill="url(#concept-field)" class="concept-field" :style="{ opacity: conceptOpacity(concept) }" />
            <circle :r="coreRadius(concept)" fill="url(#concept-core)" class="concept-core" filter="url(#glow)" :style="{ opacity: conceptOpacity(concept) }" @pointerdown.stop @click.stop="selectConcept(concept)" />
            <circle :r="coreRadius(concept)" fill="none" stroke="#e7f1ff" stroke-opacity=".82" :style="{ opacity: conceptOpacity(concept) }" />
            <text class="concept-label" text-anchor="middle" dominant-baseline="middle" :style="{ opacity: conceptLabelOpacity(concept) }">{{ concept.token }}</text>
            <text class="concept-mass" text-anchor="middle" :y="coreRadius(concept) + 16" :style="{ opacity: conceptLabelOpacity(concept) }">× {{ concept.mass.toFixed(1) }}</text>
            <title>{{ concept.token }} · масса {{ concept.mass.toFixed(2) }} · радиус поля {{ fieldRadius(concept).toFixed(0) }}</title>
          </g>
        </g>
        
        <!-- Lexemes (deep zoom) -->
        <g v-if="displayedLexemes.length && zoomLevel >= 2" class="lexemes">
          <g v-for="lexeme in displayedLexemes" :key="lexeme.id" class="lexeme-node" :transform="`translate(${xy(lexeme)[0]},${xy(lexeme)[1]})`">
            <circle :r="lexemeFieldRadius(lexeme)" fill="url(#lexeme-field)" class="lexeme-field" :style="{ opacity: lexemeOpacity(lexeme) }" />
            <circle :r="lexemeCoreRadius(lexeme)" fill="url(#lexeme-core)" class="lexeme-core" filter="url(#soft-glow)" :style="{ opacity: lexemeOpacity(lexeme) }" @pointerdown.stop @click.stop="selectLexeme(lexeme)" />
            <circle :r="lexemeCoreRadius(lexeme)" fill="none" stroke="#d5ffe7" stroke-opacity=".6" :style="{ opacity: lexemeOpacity(lexeme) }" />
            <text class="lexeme-label" text-anchor="middle" dominant-baseline="middle" :style="{ opacity: lexemeLabelOpacity(lexeme) }">{{ lexeme.token }}</text>
            <title>{{ lexeme.token }} · лексима · масса {{ lexeme.mass.toFixed(2) }}</title>
          </g>
        </g>
        
        <!-- Word forms (deepest zoom) -->
        <g v-if="displayedWordForms.length && zoomLevel >= 3" class="word-forms">
          <g v-for="wf in displayedWordForms" :key="wf.id" class="word-form-node" :transform="`translate(${xy(wf)[0]},${xy(wf)[1]})`">
            <circle r="8" fill="#ffb861" class="word-form-core" :style="{ opacity: wordFormOpacity(wf) }" @pointerdown.stop @click.stop="selectWordForm(wf)" />
            <circle r="8" fill="none" stroke="#ffe7d5" stroke-opacity=".8" :style="{ opacity: wordFormOpacity(wf) }" />
            <text class="word-form-label" text-anchor="middle" dominant-baseline="middle" :style="{ opacity: wordFormLabelOpacity(wf) }">{{ wf.token }}</text>
            <title>{{ wf.token }} · словоформа · масса {{ wf.mass.toFixed(2) }}</title>
          </g>
        </g>
      </g>
      <g v-if="!displayedConcepts.length && !displayedScenes.length && !displayedLexemes.length" class="empty-state"><circle :cx="width/2" :cy="height/2" r="54" fill="none" stroke="#6fa4ff" stroke-opacity=".25" stroke-dasharray="4 8"/><text :x="width/2" :y="height/2 - 5" text-anchor="middle">Пространство пусто</text><text :x="width/2" :y="height/2 + 18" text-anchor="middle" class="empty-subtitle">Введите текст слева, чтобы создать понятия</text></g>
    </svg>
    <aside v-if="selectedConcept || selectedScene || selectedLexeme || selectedWordForm" class="diagnostics">
      <div class="diagnostic-title">
        <span v-if="selectedConcept">Понятие: <strong>{{ selectedConcept.token }}</strong></span>
        <span v-else-if="selectedScene">Сцена: <strong>{{ selectedScene.token }}</strong></span>
        <span v-else-if="selectedLexeme">Лексима: <strong>{{ selectedLexeme.token }}</strong></span>
        <span v-else-if="selectedWordForm">Словоформа: <strong>{{ selectedWordForm.token }}</strong></span>
      </div>
      <div class="diagnostic-grid">
        <span>Масса<strong>{{ (selectedConcept || selectedScene || selectedLexeme || selectedWordForm)?.mass.toFixed(2) }}</strong></span>
        <span>Активация<strong>{{ (selectedConcept || selectedScene || selectedLexeme || selectedWordForm)?.activation.toFixed(2) }}</strong></span>
        <span>Координата X<strong>{{ xy(selectedConcept || selectedScene || selectedLexeme || selectedWordForm)[0].toFixed(1) }}</strong></span>
        <span>Координата Y<strong>{{ xy(selectedConcept || selectedScene || selectedLexeme || selectedWordForm)[1].toFixed(1) }}</strong></span>
        <span>Радиус поля<strong>{{ (selectedConcept ? fieldRadius(selectedConcept) : selectedScene ? sceneFieldRadius(selectedScene) : selectedLexeme ? lexemeFieldRadius(selectedLexeme) : 8).toFixed(1) }}</strong></span>
        <span>Размер ядра<strong>{{ (selectedConcept ? coreRadius(selectedConcept) : selectedScene ? sceneCoreRadius(selectedScene) : selectedLexeme ? lexemeCoreRadius(selectedLexeme) : 8).toFixed(1) }}</strong></span>
      </div>
      <button class="diagnostic-close" @click="clearSelection" aria-label="Закрыть диагностику">×</button>
    </aside>
    <div class="zoom-controls" aria-label="Масштаб карты"><button @click="zoomOut" aria-label="Уменьшить">−</button><span>{{ Math.round(zoom * 100) }}%</span><button @click="zoomIn" aria-label="Увеличить">+</button><button class="fit" @click="resetView" aria-label="Сбросить масштаб и положение">Сбросить</button></div>
    <div class="legend" aria-label="Легенда поля">
      <span><i class="field-swatch" style="background: #3977e8;"></i>понятие (concept)</span>
      <span><i class="field-swatch" style="background: #77e839;"></i>лексема (lexeme)</span>
      <span><i class="field-swatch" style="background: #e87739;"></i>сцена (scene)</span>
      <span><i class="core-swatch"></i>абсолют</span>
      <span><i class="overlap-swatch"></i>области пересекаются</span>
    </div>
    <div class="lod-indicator" :class="{ deep: zoomLevel >= 2, deeper: zoomLevel >= 3 }">
      Уровень: {{ lodLabel }}
    </div>
  </div>

<script setup lang="ts">
import { ref, watch, computed } from 'vue'

interface Concept { 
  id: number; 
  token: string; 
  position: number[]; 
  mass: number; 
  radius: number; 
  activation: number;
  layer?: string;
  center_x?: number;
  center_y?: number;
  members?: Array<{lexeme_id: number; canonical_form: string; weight: number}>;
  concept_name?: string;
}

interface Props { 
  concepts: Concept[]; 
  width?: number; 
  height?: number 
}

const props = withDefaults(defineProps<Props>(), { width: 1000, height: 700 })

// Displayed items by layer
const displayedConcepts = ref<Concept[]>([])
const displayedScenes = ref<Concept[]>([])
const displayedLexemes = ref<Concept[]>([])
const displayedWordForms = ref<Concept[]>([])
const semanticOverlays = ref<Concept[]>([])

// Selection state
const selectedConcept = ref<Concept | null>(null)
const selectedScene = ref<Concept | null>(null)
const selectedLexeme = ref<Concept | null>(null)
const selectedWordForm = ref<Concept | null>(null)

// Camera state
const zoom = ref(1)
const pan = ref({ x: 0, y: 0 })
const isPanning = ref(false)
const panStart = ref({ x: 0, y: 0 })
const panOrigin = ref({ x: 0, y: 0 })
let animationFrame = 0

// Computed zoom level (0.22–64× range mapped to 0-4)
const zoomLevel = computed(() => {
  if (zoom.value < 0.5) return 0
  if (zoom.value < 1.0) return 1
  if (zoom.value < 2.0) return 2
  if (zoom.value < 4.0) return 3
  return 4
})

const lodLabel = computed(() => {
  const labels = ['Обзор', 'Концепты', 'Лексимы', 'Словоформы', 'Символы']
  return labels[zoomLevel.value] || 'Глубоко'
})

// Smoothstep for LOD transitions
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)))
  return t * t * (3 - 2 * t)
}

// Opacity functions for LOD
function conceptOpacity(concept: Concept): number {
  const z = zoom.value
  if (z < 0.5) return smoothstep(0.22, 0.5, z) * 0.12
  if (z < 2.0) return 1.0
  if (z < 4.0) return smoothstep(2.0, 4.0, 4.0 - z) * 0.12 + 0.12
  return 0.12
}

function conceptLabelOpacity(concept: Concept): number {
  const z = zoom.value
  if (z < 0.5) return 0
  if (z < 1.0) return smoothstep(0.5, 1.0, z)
  if (z < 3.0) return 1.0
  return smoothstep(3.0, 4.0, 4.0 - z)
}

function sceneOpacity(scene: Concept): number {
  const z = zoom.value
  if (z < 0.3) return smoothstep(0.22, 0.3, z) * 0.12
  if (z < 1.5) return 1.0
  if (z < 3.0) return smoothstep(1.5, 3.0, 3.0 - z) * 0.12 + 0.12
  return 0.12
}

function sceneLabelOpacity(scene: Concept): number {
  const z = zoom.value
  if (z < 0.5) return 0
  if (z < 1.0) return smoothstep(0.5, 1.0, z)
  if (z < 2.5) return 1.0
  return smoothstep(2.5, 3.5, 3.5 - z)
}

function lexemeOpacity(lexeme: Concept): number {
  const z = zoom.value
  if (z < 1.5) return 0
  if (z < 2.5) return smoothstep(1.5, 2.5, z)
  if (z < 5.0) return 1.0
  return smoothstep(5.0, 6.0, 6.0 - z)
}

function lexemeLabelOpacity(lexeme: Concept): number {
  const z = zoom.value
  if (z < 2.0) return 0
  if (z < 3.0) return smoothstep(2.0, 3.0, z)
  if (z < 4.5) return 1.0
  return smoothstep(4.5, 5.5, 5.5 - z)
}

function wordFormOpacity(wf: Concept): number {
  const z = zoom.value
  if (z < 3.0) return 0
  if (z < 4.0) return smoothstep(3.0, 4.0, z)
  return 1.0
}

function wordFormLabelOpacity(wf: Concept): number {
  const z = zoom.value
  if (z < 3.5) return 0
  if (z < 4.5) return smoothstep(3.5, 4.5, z)
  return 1.0
}

function overlayOpacity(overlay: Concept): number {
  const z = zoom.value
  if (z < 0.5) return smoothstep(0.22, 0.5, z) * 0.12
  if (z < 2.0) return 0.6
  if (z < 4.0) return smoothstep(2.0, 4.0, 4.0 - z) * 0.6 + 0.12
  return 0.12
}

function overlayLabelOpacity(overlay: Concept): number {
  const z = zoom.value
  if (z < 0.8) return 0
  if (z < 1.5) return smoothstep(0.8, 1.5, z) * 0.8
  if (z < 3.0) return 0.8
  return smoothstep(3.0, 4.0, 4.0 - z) * 0.8
}

// Radius functions
function fieldRadius(concept: Concept) { return Math.min(250, Math.max(24, concept.radius || 22 + 12 * Math.sqrt(Math.max(.001, concept.mass)))) }
function coreRadius(concept: Concept) { return Math.min(28, 10 + 3 * Math.sqrt(Math.max(.001, concept.mass))) }

function sceneFieldRadius(scene: Concept) { return Math.min(300, Math.max(30, (scene.radius || 40) * 1.5)) }
function sceneCoreRadius(scene: Concept) { return Math.min(40, 15 + 5 * Math.sqrt(Math.max(.001, scene.mass))) }

function lexemeFieldRadius(lexeme: Concept) { return Math.min(100, Math.max(12, (lexeme.radius || 20) * 0.8)) }
function lexemeCoreRadius(lexeme: Concept) { return Math.min(16, 6 + 2 * Math.sqrt(Math.max(.001, lexeme.mass))) }

// Position helper
function xy(item: Concept): [number, number] {
  return [
    Number.isFinite(item.position?.[0]) ? item.position[0] : (item.center_x ?? props.width / 2),
    Number.isFinite(item.position?.[1]) ? item.position[1] : (item.center_y ?? props.height / 2)
  ]
}

// Member offset for semantic overlays
function memberOffset(member: {lexeme_id: number; canonical_form: string; weight: number}, overlay: Concept): {x: number, y: number} {
  // Simple hash-based positioning for consistent layout
  const hash = member.lexeme_id * 137 + member.canonical_form.length * 17
  const angle = (hash % 360) * Math.PI / 180
  const radius = overlay.radius * 0.4 * (0.5 + member.weight * 0.5)
  return { x: radius * Math.cos(angle), y: radius * Math.sin(angle) }
}

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

// Watch for prop changes and separate by layer
watch(() => props.concepts, (newConcepts) => {
  animateConcepts(newConcepts.filter(c => c.layer !== 'scene' && c.layer !== 'lexeme' && c.layer !== 'word_form' && !c.center_x))
  displayedScenes.value = newConcepts.filter(c => c.layer === 'scene')
  displayedLexemes.value = newConcepts.filter(c => c.layer === 'lexeme')
  displayedWordForms.value = newConcepts.filter(c => c.layer === 'word_form')
  semanticOverlays.value = newConcepts.filter(c => c.center_x !== undefined && c.concept_name)
}, { deep: true, immediate: true })

function zoomIn() { zoom.value = Math.min(64, +(zoom.value * 1.2).toFixed(2)) }
function zoomOut() { zoom.value = Math.max(0.22, +(zoom.value / 1.2).toFixed(2)) }
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
function selectConcept(concept: Concept) { selectedConcept.value = concept; selectedScene.value = null; selectedLexeme.value = null; selectedWordForm.value = null }
function selectScene(scene: Concept) { selectedScene.value = scene; selectedConcept.value = null; selectedLexeme.value = null; selectedWordForm.value = null }
function selectLexeme(lexeme: Concept) { selectedLexeme.value = lexeme; selectedConcept.value = null; selectedScene.value = null; selectedWordForm.value = null }
function selectWordForm(wf: Concept) { selectedWordForm.value = wf; selectedConcept.value = null; selectedScene.value = null; selectedLexeme.value = null }
function clearSelection() { selectedConcept.value = null; selectedScene.value = null; selectedLexeme.value = null; selectedWordForm.value = null }

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
