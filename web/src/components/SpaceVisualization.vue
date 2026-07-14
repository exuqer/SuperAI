<template>
  <div ref="host" class="space-visualization">
    <svg
      :viewBox="`0 0 ${width} ${height}`"
      @wheel.prevent="onWheel"
      @pointerdown="startPan"
      @pointermove="movePan"
      @pointerup="stopPan"
      @pointerleave="stopPan"
      @click.self="emit('select-placement', null)"
    >
      <defs>
        <radialGradient id="background" cx="50%" cy="45%" r="70%">
          <stop offset="0" stop-color="#17365f" />
          <stop offset=".48" stop-color="#0d203c" />
          <stop offset="1" stop-color="#060d19" />
        </radialGradient>
        <radialGradient v-for="type in cloudTypes" :id="fieldId(type)" :key="type">
          <stop offset="0" :stop-color="colorFor(type)" stop-opacity=".42" />
          <stop offset=".2" :stop-color="colorFor(type)" stop-opacity=".24" />
          <stop offset=".58" :stop-color="colorFor(type)" stop-opacity=".075" />
          <stop offset="1" :stop-color="colorFor(type)" stop-opacity="0" />
        </radialGradient>
        <filter id="glow" x="-70%" y="-70%" width="240%" height="240%">
          <feGaussianBlur stdDeviation="5" result="blur" />
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <rect width="100%" height="100%" fill="url(#background)" />
      <g v-if="!structure" class="density-field" :transform="cameraTransform">
        <circle
          v-for="placement in placements"
          :key="`field-${placement.id}`"
          :cx="placement.x"
          :cy="placement.y"
          :r="fieldRadius(placement)"
          :fill="`url(#${fieldId(cloudFor(placement)?.cloud_type)})`"
          :opacity="fieldOpacity(placement)"
        />
      </g>
      <g class="grid" :transform="cameraTransform">
        <path v-for="line in gridLines" :key="line" :d="line" />
      </g>
      <g :transform="cameraTransform">
        <template v-if="structure">
          <g
            v-for="component in structure.components"
            :key="component.id"
            class="structure-component"
            :transform="`translate(${centerX + component.local_x} ${centerY + component.local_y})`"
          >
            <circle :r="24" :fill="colorFor(structure.clouds[String(component.child_cloud_id)]?.cloud_type)" />
            <text text-anchor="middle" y="5">{{ structure.clouds[String(component.child_cloud_id)]?.canonical_name }}</text>
            <text class="index" text-anchor="middle" y="42">#{{ component.component_index }}</text>
          </g>
        </template>
        <template v-else>
          <g
            v-for="placement in placements"
            :key="placement.id"
            :class="['placement-node', cloudFor(placement)?.cloud_type, { selected: placement.id === selectedPlacementId }]"
            :transform="`translate(${placement.x} ${placement.y})`"
            @click.stop="emit('select-placement', placement.id)"
            @dblclick.stop="emit('open-placement', placement.id)"
          >
            <circle class="halo" :r="placement.radius + 8" />
            <circle
              class="body"
              :r="placement.radius"
              :fill="colorFor(cloudFor(placement)?.cloud_type)"
              :style="{ opacity: Math.max(.38, placement.local_activation) }"
            />
            <text text-anchor="middle" :y="placement.radius + 18">{{ cloudFor(placement)?.canonical_name }}</text>
            <text class="type" text-anchor="middle" y="4">{{ shortType(cloudFor(placement)?.cloud_type) }}</text>
          </g>
        </template>
      </g>
    </svg>
    <div class="space-badge">{{ spaceLabel(space?.space_type) }} · {{ Math.round(zoom * 100) }}%</div>
    <div class="legend">
      <span v-for="type in cloudTypes" :key="type"><i :style="{ background: colorFor(type) }"></i>{{ typeLabel(type) }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import type { CloudV2, PlacementV2, SpaceV2, StructureV2 } from '@/entities/model/types'

const props = withDefaults(defineProps<{
  space: SpaceV2 | null
  clouds: Record<number, CloudV2>
  placements: PlacementV2[]
  structure?: StructureV2 | null
  selectedPlacementId?: number | null
  width?: number
  height?: number
}>(), { width: 1200, height: 760, structure: null, selectedPlacementId: null })

const emit = defineEmits<{
  (event: 'select-placement', placementId: number | null): void
  (event: 'open-placement', placementId: number): void
}>()

const host = ref<HTMLElement | null>(null)
const zoom = ref(1)
const pan = ref({ x: 0, y: 0 })
const dragging = ref(false)
const pointer = ref({ x: 0, y: 0 })
const size = ref({ width: props.width, height: props.height })
let observer: ResizeObserver | null = null

const width = computed(() => size.value.width || props.width)
const height = computed(() => size.value.height || props.height)
const centerX = computed(() => width.value / 2)
const centerY = computed(() => height.value / 2)
const cameraTransform = computed(() =>
  `translate(${centerX.value + pan.value.x} ${centerY.value + pan.value.y}) scale(${zoom.value}) translate(${-centerX.value} ${-centerY.value})`,
)
const gridLines = computed(() => {
  const lines: string[] = []
  for (let x = 0; x <= 1800; x += 100) lines.push(`M ${x} 0 V 1100`)
  for (let y = 0; y <= 1100; y += 100) lines.push(`M 0 ${y} H 1800`)
  return lines
})
const cloudTypes = ['scene', 'word_form', 'lexeme', 'concept_candidate', 'concept', 'character']

const typeLabels: Record<string, string> = {
  scene: 'сцена',
  word_form: 'словоформа',
  lexeme: 'лексема',
  concept_candidate: 'кандидат понятия',
  concept: 'понятие',
  character: 'символ',
}

const spaceLabels: Record<string, string> = {
  global_field: 'Глобальное поле',
  scene_space: 'Пространство сцены',
  word_structure_space: 'Структура словоформы',
  concept_space: 'Пространство понятия',
  hive_space: 'Пространство улья',
}

function cloudFor(placement: PlacementV2) {
  return props.clouds[placement.cloud_id]
}

function colorFor(type?: string) {
  return ({
    scene: '#7ee9d0',
    word_form: '#73b0ff',
    lexeme: '#aa8cff',
    concept_candidate: '#ffc968',
    concept: '#ff8fb6',
    character: '#d9e7fa',
  } as Record<string, string>)[type || ''] || '#879ab8'
}

function fieldId(type?: string) {
  return `continuum-${type || 'unknown'}`
}

function fieldRadius(placement: PlacementV2) {
  return Math.max(90, placement.radius * 6 + Math.sqrt(Math.max(0, placement.local_density)) * 24)
}

function fieldOpacity(placement: PlacementV2) {
  return Math.min(.9, .44 + placement.local_activation * .26 + placement.local_density * .12)
}

function typeLabel(type?: string) {
  return typeLabels[type || ''] || 'объект'
}

function spaceLabel(type?: string) {
  return spaceLabels[type || ''] || 'Загрузка'
}

function shortType(type?: string) {
  return ({ scene: 'С', word_form: 'Сл', lexeme: 'Л', concept_candidate: 'П?', concept: 'П', character: '↳' } as Record<string, string>)[type || ''] || ''
}

function onWheel(event: WheelEvent) {
  zoom.value = Math.max(0.22, Math.min(64, zoom.value * Math.exp(-event.deltaY * 0.0012)))
}

function startPan(event: PointerEvent) {
  if ((event.target as Element).closest('.placement-node')) return
  dragging.value = true
  pointer.value = { x: event.clientX, y: event.clientY }
}

function movePan(event: PointerEvent) {
  if (!dragging.value) return
  pan.value = {
    x: pan.value.x + event.clientX - pointer.value.x,
    y: pan.value.y + event.clientY - pointer.value.y,
  }
  pointer.value = { x: event.clientX, y: event.clientY }
}

function stopPan() {
  dragging.value = false
}

function zoomBy(factor: number) {
  zoom.value = Math.max(0.22, Math.min(64, zoom.value * factor))
}

function resetView() {
  zoom.value = 1
  pan.value = { x: 0, y: 0 }
}

onMounted(() => {
  if (!host.value || typeof ResizeObserver === 'undefined') return
  observer = new ResizeObserver(entries => {
    const bounds = entries[0]?.contentRect
    if (bounds) size.value = { width: bounds.width, height: bounds.height }
  })
  observer.observe(host.value)
})
onBeforeUnmount(() => observer?.disconnect())
defineExpose({ zoomBy, resetView })
</script>

<style scoped>
.space-visualization{position:relative;width:100%;height:100%;min-height:480px;overflow:hidden;border-radius:14px;background:#060d19}.space-visualization svg{display:block;width:100%;height:100%;touch-action:none}.density-field{pointer-events:none;mix-blend-mode:screen}.grid path{fill:none;stroke:rgba(148,186,235,.11);stroke-width:1}.placement-node{cursor:pointer}.placement-node .halo{fill:transparent;stroke:transparent;stroke-width:2}.placement-node:hover .halo,.placement-node.selected .halo{fill:rgba(126,233,208,.08);stroke:#dff8ff;filter:url(#glow)}.placement-node .body{stroke:rgba(235,246,255,.55);stroke-width:1.4}.placement-node text,.structure-component text{fill:#dce9fa;font:11px system-ui;paint-order:stroke;stroke:#07101e;stroke-width:3;stroke-linejoin:round}.placement-node .type{fill:#07101e;stroke:none;font-weight:800}.structure-component circle{stroke:#fff;stroke-opacity:.55}.structure-component .index{fill:#8497b5;font-size:9px}.space-badge,.legend{position:absolute;border:1px solid rgba(160,190,228,.14);border-radius:8px;color:#94a9c6;background:rgba(5,13,25,.78);font:10px system-ui}.space-badge{top:14px;left:14px;padding:7px 10px;text-transform:uppercase;letter-spacing:.08em}.legend{right:14px;bottom:14px;display:flex;gap:10px;flex-wrap:wrap;padding:8px 10px}.legend span{display:flex;align-items:center;gap:4px}.legend i{width:7px;height:7px;border-radius:50%}
</style>
