<script setup lang="ts">
import { ref, computed, onBeforeUnmount, onMounted, defineComponent, h, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { useRuntimeStore } from '@/shared/model/runtime-store'
import StatusBadge from '@/widgets/app-shell/StatusBadge.vue'
import { serviceFor } from '@/shared/api/service'

const runtime = useRuntimeStore()
const route = useRoute()
const router = useRouter()

const api = serviceFor('live')

const activeTab = ref<'overview' | 'learning' | 'architecture' | 'visualization'>('overview')

const tabs = [
  { id: 'overview', label: 'Обзор', hint: 'Архитектура и модули' },
  { id: 'learning', label: 'Обучение', hint: 'Тренировка текстовыми запросами' },
  { id: 'architecture', label: 'Архитектура', hint: 'Схема компонентов' },
  { id: 'visualization', label: 'Визуализация', hint: 'Граф системы в реальном времени' },
]

const learningMode = ref<'text' | 'dataset'>('text')
const learningModes = [
  { id: 'text', label: 'Текстовый запрос' },
  { id: 'dataset', label: 'Датасет' },
] as const

const trainingText = ref('')
const trainingHistory = ref<Array<{ id: string; timestamp: string; input: string; status: string; result?: string }>>([])
const isTraining = ref(false)
const trainingProgress = ref(0)
const trainingStatus = ref('')
const autoRefresh = ref(false)
const datasetFile = ref<File | null>(null)
const datasetPreview = ref<Array<Record<string, any>>>([])
const datasetError = ref('')
const fileInput = ref<HTMLInputElement | null>(null)
const fileInputRef = ref<HTMLInputElement | null>(null)
let visualizationRefreshTimer: number | undefined

const datasetColumns = computed(() => {
  if (datasetPreview.value.length === 0) return []
  return Object.keys(datasetPreview.value[0])
})

async function refreshVisualization() {
  await Promise.all([runtime.loadCosmos(), runtime.loadSystem()])
}

const cosmosStats = computed(() => ({
  concepts: runtime.cosmos.length,
  claims: runtime.cosmos.reduce((sum, c) => sum + c.claims.length, 0),
  sectors: [...new Set(runtime.cosmos.flatMap(c => c.sectors))].length,
}))

const hiveStats = computed(() => {
  if (!runtime.hive) return { stores: 0, hotMemory: 0, evicted: 0, snapshots: 0 }
  return {
    stores: runtime.hive.stores.length,
    hotMemory: runtime.hive.hotMemory.utilization,
    evicted: runtime.hive.evictedItems.length,
    snapshots: runtime.hive.snapshots.length,
  }
})

const systemHealth = computed(() => runtime.system?.health.status ?? 'unknown')

function trainingResultSummary(result: {
  processed: number
  imported_claims: number
  imported_concepts: number
  duplicates: number
}): string {
  return `Текстов: ${result.processed}; claims: ${result.imported_claims}; concepts: ${result.imported_concepts}; дубликатов: ${result.duplicates}`
}

async function startTextTraining() {
  if (!trainingText.value.trim() || isTraining.value) return
  
  isTraining.value = true
  trainingProgress.value = 0
  trainingStatus.value = 'Подготовка данных...'
  
  const sessionId = `train-${Date.now()}`
  const startTime = new Date().toISOString()
  
  trainingHistory.value.unshift({
    id: sessionId,
    timestamp: startTime,
    input: trainingText.value.slice(0, 100) + (trainingText.value.length > 100 ? '...' : ''),
    status: 'running',
  })
  
  try {
    trainingProgress.value = 25
    trainingStatus.value = 'Извлечение claims и concepts...'
    const result = await api.trainDataset([trainingText.value])
    trainingProgress.value = 100
    trainingStatus.value = 'Завершено'
    
    trainingHistory.value[0].status = 'completed'
    trainingHistory.value[0].result = trainingResultSummary(result)
    trainingText.value = ''
    
    await refreshVisualization()
  } catch (error) {
    trainingHistory.value[0].status = 'failed'
    trainingHistory.value[0].result = error instanceof Error ? error.message : 'Ошибка обучения'
  } finally {
    isTraining.value = false
  }
}

async function loadDataset() {
  // Placeholder for dataset loading
  trainingStatus.value = 'Загрузка датасета... (не реализовано)'
  await new Promise(r => setTimeout(r, 1000))
  trainingStatus.value = 'Готово'
}

function handleFileSelect(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files || input.files.length === 0) return
  
  const file = input.files[0]
  datasetFile.value = file
  datasetError.value = ''
  datasetPreview.value = []
  
  // Validate file extension
  const validExtensions = ['.jsonl', '.csv', '.parquet', '.json']
  const hasValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext))
  
  if (!hasValidExtension) {
    datasetError.value = 'Неподдерживаемый формат файла. Поддерживаются: .jsonl, .csv, .parquet, .json'
    return
  }
  
  // Parse file based on extension
  if (file.name.toLowerCase().endsWith('.jsonl')) {
    parseJSONL(file)
  } else if (file.name.toLowerCase().endsWith('.csv')) {
    parseCSV(file)
  } else if (file.name.toLowerCase().endsWith('.json')) {
    parseJSON(file)
  } else if (file.name.toLowerCase().endsWith('.parquet')) {
    datasetError.value = 'Parquet формат требует дополнительных библиотек. Используйте JSONL, CSV или JSON.'
  }
}

function parseJSONL(file: File) {
  const reader = new FileReader()
  reader.onload = (e) => {
    try {
      const text = e.target?.result as string
      const lines = text.trim().split('\n').filter(line => line.trim())
      const records = lines.slice(0, 5).map(line => JSON.parse(line))
      datasetPreview.value = records
    } catch (err) {
      datasetError.value = `Ошибка парсинга JSONL: ${err instanceof Error ? err.message : 'неизвестная ошибка'}`
    }
  }
  reader.readAsText(file)
}

function parseCSV(file: File) {
  const reader = new FileReader()
  reader.onload = (e) => {
    try {
      const text = e.target?.result as string
      const lines = text.trim().split('\n').filter(line => line.trim())
      if (lines.length === 0) {
        datasetError.value = 'Пустой CSV файл'
        return
      }
      
      // Simple CSV parsing (handles basic cases)
      const headers = parseCSVLine(lines[0])
      const records = lines.slice(1, 6).map(line => {
        const values = parseCSVLine(line)
        const record: Record<string, any> = {}
        headers.forEach((header, idx) => {
          record[header] = values[idx] ?? ''
        })
        return record
      })
      datasetPreview.value = records
    } catch (err) {
      datasetError.value = `Ошибка парсинга CSV: ${err instanceof Error ? err.message : 'неизвестная ошибка'}`
    }
  }
  reader.readAsText(file)
}

function parseCSVLine(line: string): string[] {
  const result: string[] = []
  let current = ''
  let inQuotes = false
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i]
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"'
        i++
      } else {
        inQuotes = !inQuotes
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim())
      current = ''
    } else {
      current += char
    }
  }
  result.push(current.trim())
  return result
}

function parseJSON(file: File) {
  const reader = new FileReader()
  reader.onload = (e) => {
    try {
      const text = e.target?.result as string
      const data = JSON.parse(text)
      // Handle both array and object with data property
      const records = Array.isArray(data) ? data : (data.data || [data])
      datasetPreview.value = records.slice(0, 5)
    } catch (err) {
      datasetError.value = `Ошибка парсинга JSON: ${err instanceof Error ? err.message : 'неизвестная ошибка'}`
    }
  }
  reader.readAsText(file)
}

async function startDatasetTraining() {
  if (!datasetFile.value || isTraining.value) return
  
  isTraining.value = true
  trainingProgress.value = 0
  trainingStatus.value = 'Подготовка датасета...'
  
  const sessionId = `train-dataset-${Date.now()}`
  const startTime = new Date().toISOString()
  
  trainingHistory.value.unshift({
    id: sessionId,
    timestamp: startTime,
    input: `Датасет: ${datasetFile.value?.name}`,
    status: 'running',
  })
  
  try {
    // Read file and extract texts
    const texts = await extractTextsFromFile(datasetFile.value)
    
    if (texts.length === 0) {
      throw new Error('Файл не содержит текстов для обучения')
    }
    
    trainingProgress.value = 25
    trainingStatus.value = 'Извлечение claims и concepts...'
    const result = await api.trainDataset(texts)
    trainingProgress.value = 100
    trainingStatus.value = 'Завершено'
    
    trainingHistory.value[0].status = 'completed'
    trainingHistory.value[0].result = trainingResultSummary(result)
    
    await refreshVisualization()
  } catch (error) {
    trainingHistory.value[0].status = 'failed'
    trainingHistory.value[0].result = error instanceof Error ? error.message : 'Ошибка обучения на датасете'
  } finally {
    isTraining.value = false
    datasetPreview.value = []
    datasetFile.value = null
  }
}

async function extractTextsFromFile(file: File): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string
        const texts: string[] = []
        
        if (file.name.toLowerCase().endsWith('.jsonl')) {
          const lines = text.trim().split('\n').filter(line => line.trim())
          for (const line of lines) {
            const obj = JSON.parse(line)
            // Try to find text field
            const textValue = obj.text || obj.content || obj.message || obj.body || Object.values(obj).find(v => typeof v === 'string')
            if (textValue) texts.push(String(textValue))
          }
        } else if (file.name.toLowerCase().endsWith('.csv')) {
          const lines = text.trim().split('\n').filter(line => line.trim())
          if (lines.length > 1) {
            const headers = parseCSVLine(lines[0])
            for (let i = 1; i < lines.length; i++) {
              const values = parseCSVLine(lines[i])
              const row: Record<string, any> = {}
              headers.forEach((header, idx) => { row[header] = values[idx] ?? '' })
              // Find text column
              const textValue = row.text || row.content || row.message || row.body || Object.values(row).find(v => typeof v === 'string' && v.length > 10)
              if (textValue) texts.push(String(textValue))
            }
          }
        } else if (file.name.toLowerCase().endsWith('.json')) {
          const data = JSON.parse(text)
          const records = Array.isArray(data) ? data : (data.data || [data])
          for (const record of records) {
            const textValue = record.text || record.content || record.message || record.body || Object.values(record).find(v => typeof v === 'string' && v.length > 10)
            if (textValue) texts.push(String(textValue))
          }
        }
        
        resolve(texts)
      } catch (err) {
        reject(err)
      }
    }
    reader.onerror = () => reject(new Error('Ошибка чтения файла'))
    reader.readAsText(file)
  })
}

function formatTime(iso: string) {
  return new Date(iso).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

onMounted(async () => {
  await refreshVisualization()
})

watch(autoRefresh, (enabled) => {
  if (visualizationRefreshTimer !== undefined) {
    window.clearInterval(visualizationRefreshTimer)
    visualizationRefreshTimer = undefined
  }
  if (enabled) {
    void refreshVisualization()
    visualizationRefreshTimer = window.setInterval(() => void refreshVisualization(), 5_000)
  }
})

onBeforeUnmount(() => {
  if (visualizationRefreshTimer !== undefined) {
    window.clearInterval(visualizationRefreshTimer)
  }
})

// Component definitions for Architecture and Visualization tabs
const ComponentBox = defineComponent({
  props: {
    name: String,
    desc: String,
    color: String,
  },
  setup(props) {
    return () => h('div', {
      class: 'component-box',
      style: { borderColor: props.color }
    }, [
      h('strong', { style: { color: props.color } }, props.name),
      h('small', props.desc),
    ])
  },
})

const SystemVisualization = defineComponent({
  props: {
    cosmos: { type: Array, default: () => [] },
    hive: { type: Object, default: null },
    system: { type: Object, default: null },
    task: { type: Object, default: null },
    traces: { type: Array, default: () => [] },
  },
  setup(props) {
    const layout = ref<'force' | 'hierarchical' | 'circular'>('force')
    const showLabels = ref(true)
    const filterLayer = ref<'all' | 'runtime' | 'cognition' | 'execution' | 'storage' | 'learning'>('all')
    
    const nodes = computed(() => buildNodes(props))
    const edges = computed(() => buildEdges(props))
    const positionedEdges = computed(() => {
      const positions = new Map(nodes.value.map(node => [node.id, node]))
      return edges.value
        .map(edge => {
          const source = positions.get(edge.source)
          const target = positions.get(edge.target)
          return source && target
            ? { ...edge, sourceX: source.x, sourceY: source.y, targetX: target.x, targetY: target.y }
            : null
        })
        .filter((edge): edge is NonNullable<typeof edge> => edge !== null)
    })
    
    const filteredNodes = computed(() => {
      if (filterLayer.value === 'all') return nodes.value
      return nodes.value.filter(n => n.layer === filterLayer.value)
    })
    
    const filteredEdges = computed(() => {
      if (filterLayer.value === 'all') return positionedEdges.value
      const allowedIds = new Set(filteredNodes.value.map(n => n.id))
      return positionedEdges.value.filter(e => allowedIds.has(e.source) && allowedIds.has(e.target))
    })
    
    const layers = [
      { id: 'runtime', label: 'Runtime', color: '#673ab7' },
      { id: 'cognition', label: 'Cognition', color: '#e91e63' },
      { id: 'execution', label: 'Execution', color: '#ff9800' },
      { id: 'storage', label: 'Storage', color: '#607d8b' },
      { id: 'learning', label: 'Learning', color: '#795548' },
    ]
    
    const edgeColor = '#5a7ab8'
    
    return { layout, showLabels, filterLayer, nodes, edges, filteredNodes, filteredEdges, layers, edgeColor }
  },
  template: `
    <div class="visualization-container">
      <div class="viz-controls">
        <label>
          <span>Layout</span>
          <select v-model="layout">
            <option value="force">Force-directed</option>
            <option value="hierarchical">Hierarchical</option>
            <option value="circular">Circular</option>
          </select>
        </label>
        <label>
          <span>Layer</span>
          <select v-model="filterLayer">
            <option value="all">All Layers</option>
            <option value="runtime">Runtime</option>
            <option value="cognition">Cognition</option>
            <option value="execution">Execution</option>
            <option value="storage">Storage</option>
            <option value="learning">Learning</option>
          </select>
        </label>
        <label class="checkbox-field">
          <input type="checkbox" v-model="showLabels" />
          <span>Labels</span>
        </label>
      </div>
      <div class="viz-canvas" ref="canvasRef">
        <svg v-if="filteredNodes.length > 0" class="viz-svg" viewBox="0 0 800 600">
          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon :points="[0,0, 10,3.5, 0,7]" :fill="edgeColor" />
            </marker>
          </defs>
          <g class="edges">
            <line
              v-for="edge in filteredEdges"
              :key="edge.id"
              :x1="edge.sourceX"
              :y1="edge.sourceY"
              :x2="edge.targetX"
              :y2="edge.targetY"
              :stroke="edge.color"
              :stroke-width="edge.width"
              marker-end="url(#arrowhead)"
              :stroke-dasharray="edge.dashed ? '5,5' : 'none'"
            />
          </g>
          <g class="nodes">
            <g
              v-for="node in filteredNodes"
              :key="node.id"
              :transform="\`translate(\${node.x}, \${node.y})\`"
              class="viz-node"
              :class="node.layer"
            >
              <circle :r="node.radius" :fill="node.color" :stroke="node.borderColor" stroke-width="2" />
              <text v-if="showLabels" text-anchor="middle" :y="node.radius + 14" :fill="node.textColor" font-size="10" font-family="monospace">
                {{ node.label }}
              </text>
              <text v-if="showLabels && node.detail" text-anchor="middle" :y="node.radius + 26" fill="#8fa1bd" font-size="8" font-family="monospace">
                {{ node.detail }}
              </text>
            </g>
          </g>
        </svg>
        <div v-else class="viz-empty">Нет данных для отображения. Запустите задачу или загрузите Космос.</div>
      </div>
      <div class="viz-legend">
        <div v-for="layer in layers" :key="layer.id" class="legend-item">
          <span class="legend-color" :style="{ background: layer.color }" />
          <span>{{ layer.label }}</span>
        </div>
      </div>
    </div>
  `,
})

function buildNodes(props: any) {
  const nodes = []
  const centerX = 400
  const centerY = 300
  
  // Runtime layer
  nodes.push(
    { id: 'runtime', label: 'CommandRuntime', layer: 'runtime', color: '#673ab7', borderColor: '#512da8', textColor: '#fff', radius: 30, detail: 'Queue + Worker', x: centerX, y: centerY - 180 },
    { id: 'workqueue', label: 'Work Queue', layer: 'runtime', color: '#7c4dff', borderColor: '#651fff', textColor: '#fff', radius: 22, detail: 'SQLite', x: centerX - 120, y: centerY - 240 },
    { id: 'trace', label: 'TraceRecorder', layer: 'runtime', color: '#7c4dff', borderColor: '#651fff', textColor: '#fff', radius: 22, detail: 'Spans + Events', x: centerX + 120, y: centerY - 240 },
    { id: 'workitem', label: 'WorkItem', layer: 'runtime', color: '#7c4dff', borderColor: '#651fff', textColor: '#fff', radius: 18, detail: 'Idempotent', x: centerX, y: centerY - 120 },
  )
  
  // Cognition layer
  nodes.push(
    { id: 'hive', label: 'HiveManager', layer: 'cognition', color: '#e91e63', borderColor: '#c2185b', textColor: '#fff', radius: 30, detail: 'Lifecycle + Stores', x: centerX - 180, y: centerY },
    { id: 'cosmos', label: 'Cosmos', layer: 'cognition', color: '#ec407a', borderColor: '#d81b60', textColor: '#fff', radius: 28, detail: `${props.cosmos?.length || 0} concepts`, x: centerX, y: centerY + 80 },
    { id: 'atlas', label: 'Atlas', layer: 'cognition', color: '#ec407a', borderColor: '#d81b60', textColor: '#fff', radius: 22, detail: 'Capabilities', x: centerX + 180, y: centerY },
  )
  
  // Execution layer
  nodes.push(
    { id: 'planner', label: 'Planner', layer: 'execution', color: '#ff9800', borderColor: '#f57c00', textColor: '#fff', radius: 24, detail: 'Bounded search', x: centerX - 180, y: centerY - 80 },
    { id: 'engine', label: 'ExecutionEngine', layer: 'execution', color: '#ffb74d', borderColor: '#ff9800', textColor: '#fff', radius: 24, detail: 'Factories + Families', x: centerX, y: centerY - 80 },
    { id: 'critics', label: 'CriticSystem', layer: 'execution', color: '#ffb74d', borderColor: '#ff9800', textColor: '#fff', radius: 22, detail: '5 critics', x: centerX + 180, y: centerY - 80 },
    { id: 'codec', label: 'TextCodec', layer: 'execution', color: '#ffb74d', borderColor: '#ff9800', textColor: '#fff', radius: 18, detail: 'Format + Verify', x: centerX + 180, y: centerY },
  )
  
  // Storage layer
  nodes.push(
    { id: 'objectstore', label: 'ObjectStore', layer: 'storage', color: '#607d8b', borderColor: '#455a64', textColor: '#fff', radius: 26, detail: 'Blobs + Meta', x: centerX - 180, y: centerY + 180 },
    { id: 'sqlite', label: 'SqliteDB', layer: 'storage', color: '#78909c', borderColor: '#546e7a', textColor: '#fff', radius: 22, detail: 'Tasks, Hives, Work', x: centerX, y: centerY + 180 },
    { id: 'artifact', label: 'ArtifactRef', layer: 'storage', color: '#78909c', borderColor: '#546e7a', textColor: '#fff', radius: 18, detail: 'Content-addressed', x: centerX + 180, y: centerY + 180 },
  )
  
  // Learning layer
  nodes.push(
    { id: 'compiler', label: 'ExperienceCompiler', layer: 'learning', color: '#795548', borderColor: '#5d4037', textColor: '#fff', radius: 26, detail: 'Decompose → Compile', x: centerX - 120, y: centerY + 260 },
    { id: 'genome', label: 'GenomeRegistry', layer: 'learning', color: '#8d6e63', borderColor: '#6d4c41', textColor: '#fff', radius: 22, detail: 'Versioned', x: centerX, y: centerY + 260 },
    { id: 'skill', label: 'SkillLifecycle', layer: 'learning', color: '#8d6e63', borderColor: '#6d4c41', textColor: '#fff', radius: 22, detail: 'Candidate→Active', x: centerX + 120, y: centerY + 260 },
  )
  
  // Current task if exists
  if (props.task) {
    nodes.push(
      { id: 'current-task', label: 'Current Task', layer: 'runtime', color: '#4caf50', borderColor: '#388e3c', textColor: '#fff', radius: 28, detail: props.task.id, x: centerX, y: centerY - 280 }
    )
  }
  
  // Cosmos concepts as small nodes around Cosmos
  if (props.cosmos && props.cosmos.length > 0) {
    props.cosmos.slice(0, 8).forEach((concept: any, i: number) => {
      const a = (i / Math.max(1, props.cosmos.length)) * Math.PI * 2
      nodes.push({
        id: `concept-${concept.id}`,
        label: concept.label,
        layer: 'cognition',
        color: '#f8bbd0',
        borderColor: '#ec407a',
        textColor: '#4a148c',
        radius: 14,
        detail: `${concept.claims?.length || 0} claims`,
        x: centerX + Math.cos(a) * 100,
        y: centerY + 80 + Math.sin(a) * 100,
      })
    })
  }
  
  // Hive stores
  if (props.hive?.stores) {
    props.hive.stores.slice(0, 4).forEach((store: any, i: number) => {
      nodes.push({
        id: `store-${store.storeId}`,
        label: store.label,
        layer: 'cognition',
        color: '#f8bbd0',
        borderColor: '#e91e63',
        textColor: '#4a148c',
        radius: 12,
        detail: `${store.itemCount} items`,
        x: centerX - 180 + (i % 2) * 60,
        y: centerY + 60 + Math.floor(i / 2) * 40,
      })
    })
  }
  
  return nodes
}

function buildEdges(props: any) {
  const edges = [
    { id: 'e1', source: 'runtime', target: 'workqueue', color: '#673ab7', width: 2 },
    { id: 'e2', source: 'runtime', target: 'trace', color: '#673ab7', width: 2 },
    { id: 'e3', source: 'runtime', target: 'workitem', color: '#673ab7', width: 2 },
    { id: 'e4', source: 'workitem', target: 'hive', color: '#9c27b0', width: 2 },
    { id: 'e5', source: 'workitem', target: 'cosmos', color: '#9c27b0', width: 2, dashed: true },
    { id: 'e6', source: 'workitem', target: 'atlas', color: '#9c27b0', width: 2, dashed: true },
    { id: 'e7', source: 'atlas', target: 'planner', color: '#ff9800', width: 2 },
    { id: 'e8', source: 'planner', target: 'engine', color: '#ff9800', width: 2 },
    { id: 'e9', source: 'engine', target: 'critics', color: '#ff9800', width: 2 },
    { id: 'e10', source: 'engine', target: 'codec', color: '#ff9800', width: 2 },
    { id: 'e11', source: 'engine', target: 'hive', color: '#e91e63', width: 2 },
    { id: 'e12', source: 'engine', target: 'cosmos', color: '#e91e63', width: 2, dashed: true },
    { id: 'e13', source: 'hive', target: 'objectstore', color: '#607d8b', width: 2 },
    { id: 'e14', source: 'hive', target: 'sqlite', color: '#607d8b', width: 2 },
    { id: 'e15', source: 'cosmos', target: 'objectstore', color: '#607d8b', width: 2, dashed: true },
    { id: 'e16', source: 'engine', target: 'sqlite', color: '#607d8b', width: 2 },
    { id: 'e17', source: 'workitem', target: 'compiler', color: '#795548', width: 2, dashed: true },
    { id: 'e18', source: 'compiler', target: 'genome', color: '#795548', width: 2 },
    { id: 'e19', source: 'compiler', target: 'skill', color: '#795548', width: 2 },
    { id: 'e20', source: 'skill', target: 'atlas', color: '#795548', width: 2, dashed: true },
  ]
  
  if (props.task) {
    edges.push(
      { id: 'e-task', source: 'current-task', target: 'runtime', color: '#4caf50', width: 3 },
      { id: 'e-task-hive', source: 'current-task', target: 'hive', color: '#4caf50', width: 2, dashed: true },
    )
  }
  
  // Connect concepts to cosmos
  if (props.cosmos && props.cosmos.length > 0) {
    props.cosmos.slice(0, 8).forEach((concept: any, i: number) => {
      edges.push({
        id: `e-concept-${concept.id}`,
        source: 'cosmos',
        target: `concept-${concept.id}`,
        color: '#ec407a',
        width: 1,
        dashed: true,
      })
    })
  }
  
  // Connect stores to hive
  if (props.hive?.stores) {
    props.hive.stores.slice(0, 4).forEach((store: any, i: number) => {
      edges.push({
        id: `e-store-${store.storeId}`,
        source: 'hive',
        target: `store-${store.storeId}`,
        color: '#e91e63',
        width: 1,
        dashed: true,
      })
    })
  }
  
  return edges
}
</script>

<template>
  <div class="page ai-model-page">
    <header class="page-heading">
      <div>
        <p class="eyebrow">AI Model</p>
        <h1>Модель ИИ</h1>
        <p>Управление архитектурой, обучением и визуализацией внутреннего состояния системы.</p>
      </div>
      <StatusBadge :status="systemHealth" :label="systemHealth === 'ok' ? 'система готова' : 'требует внимания'" />
    </header>

    <div class="tabs-bar">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        class="tab"
        :class="{ 'tab--active': activeTab === tab.id }"
        @click="activeTab = tab.id as typeof activeTab"
      >
        <span>{{ tab.label }}</span>
        <small>{{ tab.hint }}</small>
      </button>
    </div>

    <!-- Overview Tab -->
    <section v-if="activeTab === 'overview'" class="surface">
      <header class="surface__header">
        <div>
          <p class="eyebrow">System Overview</p>
          <h2>Текущее состояние модулей</h2>
        </div>
      </header>
      <div class="surface__body overview-grid">
        <article class="module-card">
          <header>
            <h3>Runtime</h3>
            <StatusBadge :status="runtime.system?.health.status ?? 'unknown'" />
          </header>
          <dl class="metrics">
            <div><dt>Активные задачи</dt><dd>{{ runtime.system?.activeTasks ?? 0 }}</dd></div>
            <div><dt>В очереди</dt><dd>{{ runtime.system?.queuedWorkItems ?? 0 }}</dd></div>
            <div><dt>Dead letters</dt><dd>{{ runtime.system?.deadLetters ?? 0 }}</dd></div>
            <div><dt>Runtime</dt><dd>{{ runtime.system?.meta.build ?? '—' }}</dd></div>
          </dl>
        </article>

        <article class="module-card">
          <header>
            <h3>Hive (Улей)</h3>
            <StatusBadge :status="runtime.hive?.state ?? 'idle'" />
          </header>
          <dl class="metrics">
            <div><dt>Магазинов</dt><dd>{{ hiveStats.stores }}</dd></div>
            <div><dt>Горячая память</dt><dd>{{ hiveStats.hotMemory }}%</dd></div>
            <div><dt>Вытеснено</dt><dd>{{ hiveStats.evicted }}</dd></div>
            <div><dt>Снимков</dt><dd>{{ hiveStats.snapshots }}</dd></div>
          </dl>
        </article>

        <article class="module-card">
          <header>
            <h3>Cosmos (Космос)</h3>
            <StatusBadge status="verified" label="read model" />
          </header>
          <dl class="metrics">
            <div><dt>Концептов</dt><dd>{{ cosmosStats.concepts }}</dd></div>
            <div><dt>Утверждений</dt><dd>{{ cosmosStats.claims }}</dd></div>
            <div><dt>Секторов</dt><dd>{{ cosmosStats.sectors }}</dd></div>
            <div><dt>Статус</dt><dd>Активен</dd></div>
          </dl>
        </article>

        <article class="module-card">
          <header>
            <h3>Atlas (Атлас)</h3>
            <StatusBadge status="verified" label="capabilities" />
          </header>
          <dl class="metrics">
            <div><dt>Возможностей</dt><dd>{{ runtime.system?.meta.capabilities.length ?? 0 }}</dd></div>
            <div><dt>API версия</dt><dd>{{ runtime.system?.meta.apiVersion ?? '—' }}</dd></div>
            <div><dt>Schema</dt><dd>{{ runtime.system?.meta.schemaVersion ?? '—' }}</dd></div>
            <div><dt>Backend</dt><dd>{{ runtime.system?.meta.backendVersion ?? '—' }}</dd></div>
          </dl>
        </article>

        <article class="module-card">
          <header>
            <h3>Execution (Исполнение)</h3>
            <StatusBadge status="verified" label="planner + critics" />
          </header>
          <dl class="metrics">
            <div><dt>Планировщик</dt><dd>Bounded search</dd></div>
            <div><dt>Критики</dt><dd>5 активных</dd></div>
            <div><dt>Кодек</dt><dd>TextCodec v1</dd></div>
            <div><dt>Repair</dt><dd>Local</dd></div>
          </dl>
        </article>

        <article class="module-card">
          <header>
            <h3>Learning (Обучение)</h3>
            <StatusBadge status="pending" label="experience compiler" />
          </header>
          <dl class="metrics">
            <div><dt>Компост</dt><dd>Активен</dd></div>
            <div><dt>Навыки</dt><dd>{{ (runtime.system?.meta.capabilities || []).filter(c => c.includes('skill')).length || 0 }}</dd></div>
            <div><dt>Геномы</dt><dd>0</dd></div>
            <div><dt>Shadow eval</dt><dd>Доступен</dd></div>
          </dl>
        </article>
      </div>
    </section>

    <!-- Learning Tab -->
    <section v-if="activeTab === 'learning'" class="surface">
      <header class="surface__header">
        <div>
          <p class="eyebrow">Training Interface</p>
          <h2>Обучение текстовыми запросами</h2>
        </div>
      </header>
      <div class="surface__body learning-layout">
        <div class="learning-input">
          <div class="mode-selector">
            <label v-for="mode in learningModes" :key="mode.id" class="mode-option">
              <input type="radio" :value="mode.id" v-model="learningMode" />
              <span>{{ mode.label }}</span>
            </label>
          </div>

          <div v-if="learningMode === 'text'" class="text-training">
            <label class="field field--full">
              <span>Текст для обучения</span>
              <textarea
                v-model="trainingText"
                :disabled="isTraining"
                placeholder="Введите текст, из которого система извлечет знания, концепты и причинные связи. Например: 'В Unity 2D Collider с включенным Is Trigger не создает физическое препятствие, он только сообщает о пересечении. Для столкновений нужен Rigidbody2D и правильные collision layers.'"
                rows="8"
              />
              <small>Система извлечет концепты, построит claims с provenance, проверит противоречия и интегрирует в Космос.</small>
            </label>
            <div class="inline-actions">
              <button class="button" :disabled="isTraining || !trainingText.trim()" @click="startTextTraining">
                {{ isTraining ? `Обучение... ${trainingProgress}%` : 'Начать обучение' }}
              </button>
              <button class="button button--quiet" :disabled="isTraining" @click="trainingText = ''">
                Очистить
              </button>
            </div>
            <div v-if="isTraining" class="progress-bar">
              <div class="progress-fill" :style="{ width: trainingProgress + '%' }" />
              <span class="progress-label">{{ trainingStatus }}</span>
            </div>
          </div>

          <div v-else-if="learningMode === 'dataset'" class="dataset-training">
            <p class="muted">Загрузка датасета (JSONL, CSV, Parquet) для пакетного обучения.</p>
            <div class="inline-actions">
              <label class="file-input-wrapper">
                <input
                  type="file"
                  ref="fileInputRef"
                  accept=".jsonl,.csv,.parquet,.json"
                  @change="handleFileSelect"
                  :disabled="isTraining"
                  style="display: none"
                />
                <button class="button button--secondary" @click="fileInputRef?.click()" :disabled="isTraining">
                  Выбрать файл датасета
                </button>
              </label>
            </div>
            <div v-if="datasetError" class="error-message">{{ datasetError }}</div>
            <div v-if="datasetPreview.length > 0" class="dataset-preview">
              <h4>Предпросмотр (первые 5 строк):</h4>
              <table>
                <thead>
                  <tr>
                    <th v-for="col in datasetColumns" :key="col">{{ col }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(row, idx) in datasetPreview" :key="idx">
                    <td v-for="col in datasetColumns" :key="col">{{ row[col] }}</td>
                  </tr>
                </tbody>
              </table>
              <div class="inline-actions">
                <button class="button" :disabled="isTraining" @click="startDatasetTraining">
                  Начать обучение на датасете ({{ datasetPreview.length }} строк)
                </button>
              </div>
            </div>
          </div>

        </div>

        <div class="learning-history">
          <h3>История обучения</h3>
          <div v-if="trainingHistory.length === 0" class="empty-state">
            <p>История пуста. Начните обучение текстовым запросом.</p>
          </div>
          <ul v-else class="history-list">
            <li v-for="item in trainingHistory" :key="item.id" class="history-item">
              <div class="history-header">
                <span class="history-time">{{ formatTime(item.timestamp) }}</span>
                <StatusBadge :status="item.status === 'completed' ? 'verified' : item.status === 'failed' ? 'failed' : 'running'" :label="item.status" />
              </div>
              <p class="history-input">{{ item.input }}</p>
              <p v-if="item.result" class="history-result">{{ item.result }}</p>
            </li>
          </ul>
        </div>
      </div>
    </section>

    <!-- Architecture Tab -->
    <section v-if="activeTab === 'architecture'" class="surface">
      <header class="surface__header">
        <div>
          <p class="eyebrow">Architecture</p>
          <h2>Архитектура компонентов</h2>
        </div>
      </header>
      <div class="surface__body">
        <div class="architecture-diagram">
          <div class="arch-layer">
            <h4>Transport Layer</h4>
            <div class="components-row">
              <ComponentBox name="FastAPI" desc="HTTP /api/v1" color="#009688" />
              <ComponentBox name="WebSocket" desc="Real-time traces" color="#009688" />
              <ComponentBox name="CORS" desc="Dev origins" color="#009688" />
            </div>
          </div>
          <div class="arch-layer">
            <h4>Composition Root</h4>
            <div class="components-row">
              <ComponentBox name="SuperAIService" desc="Wiring & lifecycle" color="#3f51b5" />
              <ComponentBox name="ServiceConfig" desc="Data dir, env" color="#3f51b5" />
            </div>
          </div>
          <div class="arch-layer">
            <h4>Core Runtime</h4>
            <div class="components-row">
              <ComponentBox name="CommandRuntime" desc="Queue, worker, budgets" color="#673ab7" />
              <ComponentBox name="WorkItem" desc="Idempotency, retry, DLQ" color="#673ab7" />
              <ComponentBox name="TraceRecorder" desc="Spans, events, budgets" color="#673ab7" />
            </div>
          </div>
          <div class="arch-layer">
            <h4>Cognition</h4>
            <div class="components-row">
              <ComponentBox name="HiveManager" desc="Lifecycle, stores, eviction" color="#e91e63" />
              <ComponentBox name="Cosmos" desc="Concepts, claims, provenance" color="#e91e63" />
              <ComponentBox name="Atlas" desc="Capability registry" color="#e91e63" />
            </div>
          </div>
          <div class="arch-layer">
            <h4>Execution</h4>
            <div class="components-row">
              <ComponentBox name="Planner" desc="Bounded route search" color="#ff9800" />
              <ComponentBox name="ExecutionEngine" desc="Factories, families" color="#ff9800" />
              <ComponentBox name="CriticSystem" desc="5 critics" color="#ff9800" />
              <ComponentBox name="TextCodec" desc="Format + verify" color="#ff9800" />
            </div>
          </div>
          <div class="arch-layer">
            <h4>Learning & Evolution</h4>
            <div class="components-row">
              <ComponentBox name="ExperienceCompiler" desc="Decompose → compile" color="#795548" />
              <ComponentBox name="GenomeRegistry" desc="Versioned components" color="#795548" />
              <ComponentBox name="SkillLifecycle" desc="Candidate → active" color="#795548" />
            </div>
          </div>
          <div class="arch-layer">
            <h4>Storage</h4>
            <div class="components-row">
              <ComponentBox name="ObjectStore" desc="Blobs + metadata" color="#607d8b" />
              <ComponentBox name="SqliteDatabase" desc="Tasks, hives, work items" color="#607d8b" />
              <ComponentBox name="ArtifactRef" desc="Content-addressed" color="#607d8b" />
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Visualization Tab -->
    <section v-if="activeTab === 'visualization'" class="surface">
      <header class="surface__header">
        <div>
          <p class="eyebrow">System Visualization</p>
          <h2>Визуализация системы в реальном времени</h2>
        </div>
        <div class="inline-actions">
          <button class="button button--secondary" @click="refreshVisualization">
            Обновить
          </button>
          <label class="checkbox-field">
            <input type="checkbox" v-model="autoRefresh" />
            <span>Автообновление (5с)</span>
          </label>
        </div>
      </header>
      <div class="surface__body">
        <SystemVisualization 
          :cosmos="runtime.cosmos" 
          :hive="runtime.hive" 
          :system="runtime.system" 
          :task="runtime.task"
          :traces="Object.values(runtime.traces)"
        />
      </div>
    </section>
  </div>
</template>

<style scoped lang="scss">
.ai-model-page {
  .tabs-bar {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(168, 190, 228, 0.15);
    flex-wrap: wrap;
    
    .tab {
      display: grid;
      gap: 0.15rem;
      padding: 0.6rem 1rem;
      border: 1px solid transparent;
      border-radius: 0.6rem;
      color: #9ab0d1;
      background: transparent;
      text-align: left;
      cursor: pointer;
      
      &:hover {
        background: rgba(115, 160, 232, 0.08);
        border-color: rgba(116, 172, 255, 0.2);
      }
      
      &--active {
        border-color: rgba(116, 172, 255, 0.4);
        background: rgba(69, 130, 224, 0.12);
        color: #eaf2ff;
      }
      
      span {
        font-weight: 600;
        font-size: 0.85rem;
      }
      
      small {
        font-size: 0.68rem;
        color: #7a8db0;
      }
    }
  }
  
  .overview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    
    .module-card {
      border: 1px solid rgba(168, 190, 228, 0.12);
      border-radius: 0.8rem;
      background: rgba(8, 18, 34, 0.5);
      padding: 1rem;
      
      header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
        
        h3 {
          margin: 0;
          font-size: 0.9rem;
          color: #d4e0f5;
        }
      }
      
      .metrics {
        display: grid;
        gap: 0.45rem;
        margin: 0;
        
        div {
          display: grid;
          grid-template-columns: auto 1fr;
          gap: 0.5rem;
          padding-bottom: 0.4rem;
          border-bottom: 1px solid rgba(168, 190, 228, 0.08);
          
          dt {
            color: #8fa3c2;
            font-size: 0.75rem;
            text-transform: uppercase;
          }
          
          dd {
            margin: 0;
            color: #e8f0ff;
            font-family: "SFMono-Regular", Consolas, monospace;
            font-size: 0.82rem;
            text-align: right;
          }
        }
      }
    }
  }
  
  .learning-layout {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr);
    gap: 1.5rem;
    
    @media (max-width: 900px) {
      grid-template-columns: 1fr;
    }
  }
  
  .learning-input {
    .mode-selector {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1rem;
      flex-wrap: wrap;
      
      .mode-option {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.4rem 0.7rem;
        border: 1px solid rgba(168, 190, 228, 0.15);
        border-radius: 0.5rem;
        color: #c5d3ea;
        font-size: 0.8rem;
        cursor: pointer;
        background: rgba(10, 20, 35, 0.4);
        
        input {
          accent-color: #73a0e8;
        }
        
        &:has(input:checked) {
          border-color: rgba(116, 172, 255, 0.4);
          background: rgba(69, 130, 224, 0.15);
          color: #eaf2ff;
        }
      }
    }
    
    .text-training {
      .field {
        textarea {
          min-height: 140px;
          resize: vertical;
          font-family: inherit;
        }
      }
      
      .progress-bar {
        margin-top: 0.75rem;
        height: 0.5rem;
        border-radius: 999px;
        background: #0a1527;
        overflow: hidden;
        
        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #50caa4, #79a6ff);
          transition: width 0.3s ease;
        }
        
        .progress-label {
          display: block;
          margin-top: 0.35rem;
          font-size: 0.75rem;
          color: #9ab0d1;
        }
      }
    }
  }
  
  .learning-history {
    h3 {
      margin: 0 0 0.75rem;
      font-size: 0.88rem;
      color: #d4e0f1;
    }
    
    .empty-state {
      padding: 2rem;
      text-align: center;
      color: #7a8db0;
      border: 1px dashed rgba(168, 190, 228, 0.2);
      border-radius: 0.7rem;
      background: rgba(8, 18, 34, 0.3);
    }
    
    .history-list {
      display: grid;
      gap: 0.5rem;
      max-height: 400px;
      overflow-y: auto;
      
      .history-item {
        padding: 0.75rem;
        border: 1px solid rgba(168, 190, 228, 0.12);
        border-radius: 0.65rem;
        background: rgba(8, 18, 34, 0.4);
        
        .history-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.4rem;
          
          .history-time {
            font-family: "SFMono-Regular", Consolas, monospace;
            font-size: 0.7rem;
            color: #8fa3c2;
          }
        }
        
        .history-input {
          margin: 0 0 0.4rem;
          font-size: 0.82rem;
          color: #cbd8eb;
          line-height: 1.4;
        }
        
        .history-result {
          margin: 0;
          font-size: 0.75rem;
          color: #65d3a9;
          font-family: "SFMono-Regular", Consolas, monospace;
        }
      }
    }
  }
  
  .architecture-diagram {
    display: grid;
    gap: 1.5rem;
    
    .arch-layer {
      h4 {
        margin: 0 0 0.75rem;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #7a8db0;
      }
    }
    
    .components-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
    }
  }
  
  .component-box {
    display: grid;
    gap: 0.2rem;
    padding: 0.7rem 1rem;
    border: 2px solid;
    border-radius: 0.65rem;
    background: rgba(8, 18, 34, 0.5);
    min-width: 140px;
    text-align: center;
    
    strong {
      font-size: 0.82rem;
    }
    
    small {
      font-size: 0.68rem;
      color: #9ab0d1;
    }
  }
  
  .visualization-container {
    .viz-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 1rem;
      padding: 0.75rem;
      background: rgba(8, 18, 34, 0.3);
      border-radius: 0.6rem;
      border: 1px solid rgba(168, 190, 228, 0.1);
      
      label {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        color: #9ab0d1;
        font-size: 0.75rem;
        
        select {
          border: 1px solid rgba(168, 190, 228, 0.2);
          border-radius: 0.4rem;
          background: #0a1527;
          color: #eaf1fc;
          padding: 0.25rem 0.5rem;
          font-size: 0.75rem;
        }
      }
    }
    
    .viz-canvas {
      min-height: 500px;
      border: 1px solid rgba(168, 190, 228, 0.12);
      border-radius: 0.75rem;
      background: #06101e;
      overflow: hidden;
      
      .viz-svg {
        width: 100%;
        height: 100%;
      }
      
      .viz-empty {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        min-height: 500px;
        color: #7a8db0;
        font-size: 0.9rem;
      }
    }
    
    .viz-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      margin-top: 1rem;
      padding: 0.75rem;
      background: rgba(8, 18, 34, 0.3);
      border-radius: 0.6rem;
      border: 1px solid rgba(168, 190, 228, 0.1);
      
      .legend-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.75rem;
        color: #c5d3ea;
        
        .legend-color {
          width: 1rem;
          height: 1rem;
          border-radius: 0.2rem;
        }
      }
    }
  }
  
  .inline-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.75rem;
  }
  
  .checkbox-field {
    display: flex !important;
    align-items: center;
    gap: 0.4rem !important;
    color: #d2def0 !important;
    font-size: 0.8rem;
    cursor: pointer;
    
    input {
      accent-color: #73a0e8;
    }
  }
  
  .empty-state {
    padding: 2rem;
    text-align: center;
    color: #7a8db0;
    border: 1px dashed rgba(168, 190, 228, 0.2);
    border-radius: 0.7rem;
    background: rgba(8, 18, 34, 0.3);
  }
  
  .muted {
    color: #8fa3c2;
    font-size: 0.85rem;
  }
  
  .file-input-wrapper {
    display: inline-block;
  }
  
  .error-message {
    margin-top: 0.75rem;
    padding: 0.75rem;
    border: 1px solid rgba(255, 100, 100, 0.3);
    border-radius: 0.5rem;
    background: rgba(255, 50, 50, 0.1);
    color: #ff8a8a;
    font-size: 0.8rem;
  }
  
  .dataset-preview {
    margin-top: 1rem;
    padding: 1rem;
    border: 1px solid rgba(168, 190, 228, 0.15);
    border-radius: 0.6rem;
    background: rgba(8, 18, 34, 0.4);
    
    h4 {
      margin: 0 0 0.75rem;
      font-size: 0.85rem;
      color: #d4e0f5;
    }
    
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.75rem;
      
      th, td {
        padding: 0.4rem 0.5rem;
        text-align: left;
        border-bottom: 1px solid rgba(168, 190, 228, 0.1);
      }
      
      th {
        color: #8fa3c2;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.65rem;
      }
      
      td {
        color: #cbd8eb;
        font-family: "SFMono-Regular", Consolas, monospace;
      }
      
      tr:last-child td {
        border-bottom: none;
      }
    }
  }
}
</style>
