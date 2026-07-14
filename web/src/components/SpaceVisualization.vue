<template>
  <div ref="root" class="space-visualization" @wheel.prevent="onWheel">
    <svg
      class="space-svg"
      :class="{ panning: isPanning }"
      :viewBox="`0 0 ${width} ${height}`"
      role="img"
      aria-label="Непрерывное пространство туманностей"
      @pointerdown="startPan"
      @pointermove="movePan"
      @pointerup="endPan"
      @pointercancel="endPan"
      @pointerleave="endPan"
    >
      <defs>
        <radialGradient id="scene-nebula">
          <stop stop-color="#9ac7ff" stop-opacity=".82" />
          <stop offset=".22" stop-color="#3979d7" stop-opacity=".44" />
          <stop offset=".62" stop-color="#1c579f" stop-opacity=".15" />
          <stop offset="1" stop-color="#102d5c" stop-opacity="0" />
        </radialGradient>
        <radialGradient id="word-nebula">
          <stop stop-color="#e5f5ff" stop-opacity=".95" />
          <stop offset=".18" stop-color="#72c6ff" stop-opacity=".68" />
          <stop offset=".58" stop-color="#2377cf" stop-opacity=".22" />
          <stop offset="1" stop-color="#14509d" stop-opacity="0" />
        </radialGradient>
        <radialGradient id="concept-nebula">
          <stop stop-color="#e4c6ff" stop-opacity=".42" />
          <stop offset=".34" stop-color="#a061e9" stop-opacity=".25" />
          <stop offset="1" stop-color="#6d2ca8" stop-opacity="0" />
        </radialGradient>
        <radialGradient id="character-nebula">
          <stop stop-color="#fff8d7" stop-opacity="1" />
          <stop offset=".22" stop-color="#72edcf" stop-opacity=".78" />
          <stop offset="1" stop-color="#21a88c" stop-opacity="0" />
        </radialGradient>
        <filter id="soft-cloud" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2.6" />
        </filter>
        <filter id="selected-cloud" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      <rect width="100%" height="100%" fill="#071321" />
      <g :transform="worldTransform">
        <g class="parent-space-halos" :opacity="parentHaloOpacity">
          <circle
            v-for="scene in scenes"
            :key="`halo:${scene.id}`"
            :cx="scene.x"
            :cy="scene.y"
            :r="scene.radius * 1.38"
            fill="url(#scene-nebula)"
          />
        </g>
        <g class="scene-layer" :opacity="sceneOpacity">
          <g
            v-for="scene in scenes"
            :key="scene.id"
            class="scene-node"
            :class="{ selected: selectedKey === `scene:${scene.id}` }"
            :transform="`translate(${scene.x} ${scene.y})`"
            @pointerenter="hover(scene)"
            @pointerleave="hover(null)"
            @pointerdown.stop
            @click.stop="select(scene)"
          >
            <circle :r="scene.radius" fill="url(#scene-nebula)" />
            <circle
              v-for="lobe in cloudLobes(scene)"
              :key="lobe.key"
              :cx="lobe.x * scene.radius"
              :cy="lobe.y * scene.radius"
              :r="lobe.radius * scene.radius"
              fill="url(#scene-nebula)"
              :opacity="lobe.opacity"
            />
            <circle class="mass-center" :opacity="topCoreOpacity" :r="Math.max(5, Math.min(13, 4 + Math.sqrt(scene.mass) * 2))" />
            <text v-if="sceneLabelOpacity > 0.03" :opacity="sceneLabelOpacity" text-anchor="middle" :y="-scene.radius * .18" :style="{ fontSize: labelSize(13) }">
              {{ scene.token }}
            </text>
          </g>
        </g>

        <g class="semantic-layer" :opacity="semanticOpacity">
          <g
            v-for="concept in overlays"
            :key="concept.id"
            class="concept-node semantic-overlay"
            :class="{ selected: selectedKey === `concept:${concept.id}` }"
            :transform="`translate(${concept.center_x} ${concept.center_y})`"
            @pointerenter="hover(concept)"
            @pointerleave="hover(null)"
            @pointerdown.stop
            @click.stop="select(concept)"
          >
            <circle :r="concept.radius" fill="url(#concept-nebula)" />
            <circle
              v-for="lobe in cloudLobes(concept)"
              :key="lobe.key"
              :cx="lobe.x * concept.radius"
              :cy="lobe.y * concept.radius"
              :r="lobe.radius * concept.radius"
              fill="url(#concept-nebula)"
              :opacity="lobe.opacity"
            />
            <circle class="semantic-center" :opacity="topCoreOpacity" :r="Math.max(2.5, Math.min(6, 2 + Math.sqrt(concept.mass) * .8))" />
            <text v-if="semanticLabelOpacity > .03" :opacity="semanticLabelOpacity" text-anchor="middle" :y="-concept.radius * .38" :style="{ fontSize: labelSize(11) }">
              {{ concept.token }}
            </text>
          </g>
        </g>

        <g class="word-layer" :opacity="wordOpacity">
          <g
            v-for="word in mergedWords"
            :key="word.key"
            class="word-node"
            :class="{ selected: selectedKey === `word_form:${word.key}` }"
            :transform="nodeTransform(wordPosition(word))"
            @pointerenter="hover(word)"
            @pointerleave="hover(null)"
            @pointerdown.stop
            @click.stop="select(word)"
          >
            <circle :r="word.radius * 1.75" fill="url(#word-nebula)" />
            <circle class="word-center" :opacity="wordCoreOpacity" :r="Math.max(3.2, Math.min(8, 2.5 + Math.sqrt(word.mass)))" />
            <text :opacity="wordLabelOpacity" text-anchor="middle" :y="-word.radius * .54" :style="{ fontSize: labelSize(12) }">{{ word.token }}</text>
          </g>
        </g>

        <g class="character-layer" :opacity="characterOpacity">
          <g
            v-for="character in mergedCharacters"
            :key="character.key"
            class="character-node"
            :transform="nodeTransform(characterPosition(character))"
            @pointerenter="hover(character)"
            @pointerleave="hover(null)"
            @pointerdown.stop
            @click.stop="select(character)"
          >
            <circle :r="Math.max(5, character.radius * 1.8)" fill="url(#character-nebula)" />
            <text :opacity="characterLabelOpacity" text-anchor="middle" dominant-baseline="central" :style="{ fontSize: labelSize(8) }">{{ character.token }}</text>
          </g>
        </g>

        <g class="legacy-concepts" :opacity="semanticOpacity">
          <g
            v-for="concept in fallbackConcepts"
            :key="concept.id"
            class="concept-node"
            :transform="`translate(${concept.x} ${concept.y})`"
            @pointerdown.stop
            @click.stop="select(concept)"
          >
            <circle :r="concept.radius" fill="url(#concept-nebula)" />
            <text :opacity="semanticLabelOpacity" text-anchor="middle" :style="{ fontSize: labelSize(11) }">{{ concept.token }}</text>
          </g>
        </g>
      </g>

      <g v-if="!scenes.length && !overlays.length && !fallbackConcepts.length" class="empty-state">
        <circle :cx="width / 2" :cy="height / 2" r="54" />
        <text :x="width / 2" :y="height / 2 - 4" text-anchor="middle">Пространство пусто</text>
        <text :x="width / 2" :y="height / 2 + 18" text-anchor="middle" class="empty-subtitle">Введите несколько предложений</text>
      </g>
    </svg>

    <div class="legend" aria-label="Легенда поля">
      <span><i class="scene-swatch"></i>предложение</span>
      <span><i class="word-swatch"></i>слово / лексема</span>
      <span><i class="concept-swatch"></i>понятие</span>
      <span><i class="character-swatch"></i>буква</span>
    </div>
    <div class="lod-indicator">{{ lodLabel }} · depth {{ semanticDepth.toFixed(2) }} · mix {{ Math.round(levelMix * 100) }}% · {{ Math.round(zoom * 100) }}%</div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

interface CharacterItem {
  id: number
  key: string
  token: string
  x: number
  y: number
  radius: number
  mass: number
  density: number
  stability: number
  activation: number
  layer: 'character'
  cloud_type: 'character'
}

interface WordItem {
  id: number
  key: string
  token: string
  x: number
  y: number
  radius: number
  mass: number
  density: number
  stability: number
  activation: number
  layer: 'word_form'
  cloud_type: 'word_form'
  lexeme_id?: number
  lexeme?: string
  characters: CharacterItem[]
}

interface SceneItem {
  id: number
  token: string
  x: number
  y: number
  radius: number
  mass: number
  density: number
  stability: number
  activation: number
  layer: 'scene'
  cloud_type: 'scene'
  words: WordItem[]
}

interface ConceptItem {
  id: number
  token: string
  center_x: number
  center_y: number
  radius: number
  mass: number
  density: number
  stability: number
  activation: number
  layer: 'concept'
  cloud_type: 'concept'
  members?: Array<{ lexeme_id: number; canonical_form: string; weight: number }>
}

interface LegacyItem {
  id: number
  token: string
  position?: number[]
  x?: number
  y?: number
  radius?: number
  mass?: number
  density?: number
  stability?: number
  activation?: number
  layer?: string
  cloud_type?: string
}

interface Hierarchy {
  scenes?: SceneItem[]
  semantic_overlays?: ConceptItem[]
}

const props = withDefaults(defineProps<{
  hierarchy?: Hierarchy
  concepts?: LegacyItem[]
  width?: number
  height?: number
}>(), {
  hierarchy: () => ({}),
  concepts: () => [],
  width: 1000,
  height: 700,
})

const emit = defineEmits<{
  (event: 'cloud-select', value: unknown): void
  (event: 'cloud-hover', value: unknown[]): void
  (event: 'viewport-change', value: { width: number; height: number }): void
  (event: 'camera-change', value: { x: number; y: number; zoom: number }): void
}>()

const root = ref<HTMLElement | null>(null)
const zoom = ref(1)
const pan = ref({ x: 0, y: 0 })
const isPanning = ref(false)
const panStart = ref({ x: 0, y: 0 })
const panOrigin = ref({ x: 0, y: 0 })
const selectedKey = ref('')
const selectedItem = ref<any | null>(null)
let resizeObserver: ResizeObserver | null = null

const scenes = computed(() => props.hierarchy.scenes ?? [])
const overlays = computed(() => props.hierarchy.semantic_overlays ?? [])
const worldTransform = computed(() =>
  `translate(${props.width / 2 + pan.value.x} ${props.height / 2 + pan.value.y}) scale(${zoom.value}) translate(${-props.width / 2} ${-props.height / 2})`,
)

const fallbackConcepts = computed(() => props.concepts
  .filter(item => (!item.layer || item.layer === 'concept') && item.cloud_type !== 'character')
  .map(item => ({
    ...item,
    x: item.x ?? item.position?.[0] ?? props.width / 2,
    y: item.y ?? item.position?.[1] ?? props.height / 2,
    radius: item.radius ?? 32,
    mass: item.mass ?? 1,
    density: item.density ?? 1,
    stability: item.stability ?? 0,
    activation: item.activation ?? 0,
    layer: 'concept',
    cloud_type: 'concept',
  })))

const mergedWords = computed<WordItem[]>(() => {
  const result: WordItem[] = []
  for (const scene of scenes.value) {
    for (const word of scene.words ?? []) {
      const match = result.find(item =>
        item.lexeme_id === word.lexeme_id && Math.hypot(item.x - word.x, item.y - word.y) < 3,
      )
      if (match) {
        match.mass += word.mass
        match.radius = Math.max(match.radius, word.radius)
        match.characters.push(...word.characters)
      } else {
        result.push({ ...word, characters: [...word.characters] })
      }
    }
  }
  return result
})

const mergedCharacters = computed<CharacterItem[]>(() => {
  const result: CharacterItem[] = []
  for (const word of mergedWords.value) {
    for (const character of word.characters) {
      const match = result.find(item =>
        item.token === character.token && Math.hypot(item.x - character.x, item.y - character.y) < 1.5,
      )
      if (match) {
        match.mass += character.mass
      } else {
        result.push({ ...character })
      }
    }
  }
  return result
})

function smoothstep(edge0: number, edge1: number, value: number) {
  const t = Math.max(0, Math.min(1, (value - edge0) / (edge1 - edge0)))
  return t * t * (3 - 2 * t)
}

const layerDepth = { scene: 0, concept: 1, word_form: 2, character: 3 }
const layerVisibilityRanges = {
  scene: { enter: 0, exitStart: .72, exitEnd: 1.42 },
  concept: { enterStart: .62, enterEnd: 1.22, exitStart: 1.62, exitEnd: 2.18 },
  word_form: { enterStart: 1.35, enterEnd: 1.92, exitStart: 2.36, exitEnd: 2.82 },
  character: { enterStart: 2.3, enterEnd: 2.9, exitEnd: 3 },
}
const semanticDepth = computed(() => Math.max(0, Math.min(3, Math.log2(zoom.value) + 1)))
const levelMix = computed(() => {
  const lower = Math.floor(semanticDepth.value)
  return smoothstep(lower + .28, lower + .72, semanticDepth.value)
})
const sceneOpacity = computed(() => 1 - smoothstep(layerVisibilityRanges.scene.exitStart, layerVisibilityRanges.scene.exitEnd, semanticDepth.value) * .92)
const sceneLabelOpacity = computed(() => 1 - smoothstep(.48, 1.03, semanticDepth.value))
const semanticOpacity = computed(() => smoothstep(layerVisibilityRanges.concept.enterStart, layerVisibilityRanges.concept.enterEnd, semanticDepth.value) * (1 - smoothstep(layerVisibilityRanges.concept.exitStart, layerVisibilityRanges.concept.exitEnd, semanticDepth.value) * .88))
const semanticLabelOpacity = computed(() => smoothstep(.92, 1.38, semanticDepth.value) * (1 - smoothstep(1.52, 1.9, semanticDepth.value)))
const wordOpacity = computed(() => smoothstep(layerVisibilityRanges.word_form.enterStart, layerVisibilityRanges.word_form.enterEnd, semanticDepth.value) * (1 - smoothstep(layerVisibilityRanges.word_form.exitStart, layerVisibilityRanges.word_form.exitEnd, semanticDepth.value) * .9))
const wordLabelOpacity = computed(() => smoothstep(1.68, 2.12, semanticDepth.value) * (1 - smoothstep(2.3, 2.62, semanticDepth.value)))
const characterOpacity = computed(() => smoothstep(layerVisibilityRanges.character.enterStart, layerVisibilityRanges.character.enterEnd, semanticDepth.value))
const characterLabelOpacity = computed(() => smoothstep(2.56, 2.92, semanticDepth.value))
const parentHaloOpacity = computed(() => .18 + .48 * smoothstep(.78, 1.48, semanticDepth.value) * (1 - smoothstep(2.2, 2.9, semanticDepth.value)))
const wordCoreOpacity = computed(() => .48 - .18 * smoothstep(1.6, 2.8, semanticDepth.value))
const topCoreOpacity = computed(() => .34 - .08 * smoothstep(.5, 1.8, semanticDepth.value))
const lodLabel = computed(() => {
  if (semanticDepth.value < .9) return 'Предложения'
  if (semanticDepth.value < 1.8) return 'Понятия'
  if (semanticDepth.value < 2.5) return 'Слова'
  return 'Буквы'
})

const labelSize = (base: number) => `${Math.max(8, Math.min(base, base / Math.max(.72, zoom.value)))}px`

function nodeTransform(point: { x: number; y: number }) {
  return `translate(${point.x} ${point.y})`
}

function cloudLobes(item: { id: number; mass?: number; density?: number; members?: unknown[] }) {
  const count = 2 + ((item.members?.length ?? Math.round(item.mass ?? 1)) % 3)
  const seed = item.id * 17 + (item.members?.length ?? 0) * 31
  return Array.from({ length: count }, (_, index) => {
    const angle = ((seed + index * 137) % 360) * Math.PI / 180
    const distance = .28 + (((seed + index * 29) % 31) / 100)
    return {
      key: `${item.id}:${index}`,
      x: Math.cos(angle) * distance,
      y: Math.sin(angle) * distance,
      radius: .18 + (((seed + index * 11) % 17) / 100),
      opacity: .12 + Math.min(.2, (item.density ?? .5) * .12) + index * .018,
    }
  })
}

function parentSceneForWord(word: WordItem) {
  return scenes.value.find(scene => (scene.words ?? []).some(candidate => candidate.key === word.key))
}

function parentWordForCharacter(character: CharacterItem) {
  return mergedWords.value.find(word => word.characters.some(candidate => candidate.key === character.key))
}

function wordPosition(word: WordItem) {
  const parent = parentSceneForWord(word)
  const reveal = smoothstep(.92, 1.78, semanticDepth.value)
  const target = { x: word.x, y: word.y }
  if (!parent) return target
  return { x: parent.x + (target.x - parent.x) * reveal, y: parent.y + (target.y - parent.y) * reveal }
}

function characterPosition(character: CharacterItem) {
  const parent = parentWordForCharacter(character)
  const reveal = smoothstep(2.02, 2.78, semanticDepth.value)
  const target = { x: character.x, y: character.y }
  if (!parent) return target
  const origin = wordPosition(parent)
  return { x: origin.x + (target.x - origin.x) * reveal, y: origin.y + (target.y - origin.y) * reveal }
}

function itemKey(item: any) {
  if (item.layer === 'word_form') return `word_form:${item.key}`
  return `${item.layer}:${item.id}`
}

function select(item: any) {
  selectedItem.value = item
  selectedKey.value = itemKey(item)
  emit('cloud-select', item)
}

function clearSelection() {
  selectedItem.value = null
  selectedKey.value = ''
  emit('cloud-select', null)
}

function promoteSelection(nextDepth: number) {
  const item = selectedItem.value
  if (!item) return
  const itemLayer = item.layer ?? item.cloud_type
  const itemDepth = layerDepth[itemLayer as keyof typeof layerDepth]
  if (itemDepth === undefined || itemDepth <= nextDepth + .15) return
  const parent = itemLayer === 'character'
    ? parentWordForCharacter(item)
    : itemLayer === 'word_form'
      ? parentSceneForWord(item)
      : null
  if (parent) select(parent)
  else clearSelection()
}

watch(semanticDepth, value => promoteSelection(value))

function hover(item: any | null) {
  emit('cloud-hover', item ? [item] : [])
}

function notifyCamera() {
  emit('camera-change', { x: pan.value.x, y: pan.value.y, zoom: zoom.value })
}

function zoomBy(factor: number, anchorX = props.width / 2, anchorY = props.height / 2) {
  const previous = zoom.value
  const next = Math.max(.22, Math.min(64, previous * factor))
  const centerX = props.width / 2
  const centerY = props.height / 2
  pan.value = {
    x: anchorX - centerX - (anchorX - centerX - pan.value.x) * (next / previous),
    y: anchorY - centerY - (anchorY - centerY - pan.value.y) * (next / previous),
  }
  zoom.value = next
  notifyCamera()
}

function onWheel(event: WheelEvent) {
  const bounds = root.value?.getBoundingClientRect()
  const anchorX = bounds ? (event.clientX - bounds.left) * props.width / bounds.width : props.width / 2
  const anchorY = bounds ? (event.clientY - bounds.top) * props.height / bounds.height : props.height / 2
  zoomBy(Math.exp(-event.deltaY * .0012), anchorX, anchorY)
}

function startPan(event: PointerEvent) {
  isPanning.value = true
  panStart.value = { x: event.clientX, y: event.clientY }
  panOrigin.value = { ...pan.value }
  ;(event.currentTarget as Element).setPointerCapture?.(event.pointerId)
}

function movePan(event: PointerEvent) {
  if (!isPanning.value) return
  pan.value = {
    x: panOrigin.value.x + event.clientX - panStart.value.x,
    y: panOrigin.value.y + event.clientY - panStart.value.y,
  }
  notifyCamera()
}

function endPan() {
  isPanning.value = false
}

function resetView() {
  zoom.value = 1
  pan.value = { x: 0, y: 0 }
  notifyCamera()
}

defineExpose({ zoomBy, resetView })

onMounted(() => {
  if (root.value && typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(entries => {
      const bounds = entries[0]?.contentRect
      if (bounds) emit('viewport-change', { width: bounds.width, height: bounds.height })
    })
    resizeObserver.observe(root.value)
  }
})

onBeforeUnmount(() => resizeObserver?.disconnect())
</script>

<style scoped lang="scss">
.space-visualization { position: relative; width: 100%; height: 100%; min-height: 500px; overflow: hidden; background: #071321; user-select: none; }
.space-svg { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }
.space-svg.panning { cursor: grabbing; }
.scene-node, .concept-node, .word-node, .character-node { cursor: pointer; }
.scene-node circle, .concept-node circle, .word-node circle, .character-node circle { transition: filter .18s, opacity .18s; }
.scene-node:hover > circle, .concept-node:hover > circle, .word-node:hover > circle, .character-node:hover > circle,
.selected > circle { filter: url(#selected-cloud); }
.mass-center { fill: #d8ebff; opacity: .82; filter: url(#soft-cloud); }
.word-center { fill: #f4fbff; opacity: .9; filter: url(#soft-cloud); }
text { fill: #eaf4ff; font: 600 12px Inter, system-ui, sans-serif; paint-order: stroke; stroke: #071321; stroke-width: 3px; stroke-opacity: .72; }
.scene-node text { font-size: 13px; }
.semantic-overlay text { fill: #e7cfff; font-size: 11px; }
.character-node text { fill: #f8ffff; font-size: 7px; stroke-width: 1.5px; }
.empty-state circle { fill: none; stroke: #6fa4ff; stroke-dasharray: 4 8; stroke-opacity: .3; }
.empty-state text { fill: #a8b8d1; stroke: none; }
.empty-state .empty-subtitle { fill: #6f819c; font-size: 10px; }
.legend { position: absolute; right: 1rem; bottom: 1rem; display: flex; flex-wrap: wrap; gap: .65rem; max-width: calc(100% - 2rem); padding: .55rem .7rem; border: 1px solid rgba(168,190,228,.16); border-radius: .6rem; color: #9aaac5; background: rgba(7,16,31,.82); font-size: .7rem; backdrop-filter: blur(8px); }
.legend i { display: inline-block; width: .8rem; height: .65rem; margin-right: .3rem; border-radius: 50%; vertical-align: middle; }
.scene-swatch { background: #3979d7; box-shadow: 0 0 8px #3979d7; }
.word-swatch { background: #72c6ff; box-shadow: 0 0 8px #72c6ff; }
.concept-swatch { background: #a061e9; box-shadow: 0 0 8px #a061e9; }
.character-swatch { background: #72edcf; box-shadow: 0 0 8px #72edcf; }
.lod-indicator { position: absolute; left: 1rem; bottom: 1rem; padding: .45rem .65rem; border: 1px solid rgba(168,190,228,.14); border-radius: .55rem; color: #93a9ca; background: rgba(7,16,31,.8); font-size: .68rem; }
</style>
