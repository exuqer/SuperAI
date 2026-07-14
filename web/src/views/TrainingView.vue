<template>
  <div class="field-page">
    <header class="topbar">
      <div class="brand"><span>✦</span><div><strong>SuperAI</strong><small>Облако · Пространство · Размещение</small></div></div>
      <nav><router-link to="/">Чат с ульем</router-link><span class="status">ТОЛЬКО V2</span></nav>
    </header>

    <div class="breadcrumbs">
      <button v-for="(item, index) in store.breadcrumb" :key="item.space.id" @click="store.navigateTo(index)">
        {{ item.label }}<span v-if="index < store.breadcrumb.length - 1">›</span>
      </button>
    </div>

    <main class="workspace">
      <aside class="panel training-panel">
        <div class="panel-title"><span>ОБУЧЕНИЕ</span><h1>Обучение модели</h1></div>
        <textarea v-model="text" rows="8" placeholder="Кот ест рыбу. Рыбак ловит рыбу." @keydown.ctrl.enter="learn" />
        <div class="actions">
          <button class="primary" :disabled="store.loading || !text.trim()" @click="learn">Обучить</button>
          <button class="danger" :disabled="store.loading" @click="clear">Очистить данные</button>
        </div>
        <p v-if="store.error" class="error">{{ store.error }}</p>
        <section class="model-data">
          <div class="model-data-head">
            <span>ДАННЫЕ МОДЕЛИ</span>
            <button class="secondary" :disabled="store.loading" @click="toggleModelData">
              {{ showModelData ? 'Скрыть' : 'Показать' }}
            </button>
          </div>
          <template v-if="showModelData">
            <div class="model-data-actions">
              <button class="secondary" :disabled="store.loading" @click="refreshModelData">Обновить</button>
              <button class="secondary" :disabled="!modelJson" @click="copyModelData">{{ copyStatus || 'Копировать JSON' }}</button>
            </div>
            <pre aria-label="Объект обученной модели">{{ modelJson || 'Нет обученных данных.' }}</pre>
          </template>
        </section>
        <div v-if="store.lastTraining" class="telemetry">
          <span>ЗАПУСК ОБУЧЕНИЯ</span>
          <code>{{ store.lastTraining.training_run_id }}</code>
          <dl>
            <div><dt>Создано облаков</dt><dd>{{ count('created_clouds') }}</dd></div>
            <div><dt>Усилено</dt><dd>{{ count('strengthened_clouds') }}</dd></div>
            <div><dt>Пространства</dt><dd>{{ count('created_spaces') }}</dd></div>
            <div><dt>Размещения</dt><dd>{{ count('created_placements') }}</dd></div>
            <div><dt>Структура</dt><dd>{{ count('created_structures') }}</dd></div>
            <div><dt>Повторно сцены</dt><dd>{{ count('reused_scenes') }}</dd></div>
          </dl>
        </div>
      </aside>

      <section class="field-panel">
        <SpaceVisualization
          ref="renderer"
          :space="store.currentSpace"
          :clouds="store.cloudsById"
          :placements="store.placements"
          :structure="store.currentStructure"
          :selected-placement-id="store.selectedPlacementId"
          @select-placement="store.selectPlacement"
          @open-placement="store.zoomIntoPlacement"
        />
        <div v-if="store.loading" class="loading">Обновление пространства…</div>
        <div class="controls">
          <button @click="renderer?.zoomBy(1 / 1.2)">−</button>
          <button @click="renderer?.resetView()">Обзор</button>
          <button @click="renderer?.zoomBy(1.2)">+</button>
          <button :disabled="store.currentSpace?.space_type === 'word_structure_space'" @click="store.tickPhysics">Шаг физики</button>
        </div>
      </section>

      <aside class="panel inspector">
        <div class="panel-title"><span>ИНСПЕКТОР</span><h2>{{ store.selectedCloud?.canonical_name || 'Выберите размещение' }}</h2></div>
        <template v-if="store.selectedCloud && store.selectedPlacement">
          <section>
            <h3>Облако</h3>
            <dl>
              <div><dt>ID</dt><dd>{{ store.selectedCloud.id }}</dd></div>
              <div><dt>Тип</dt><dd>{{ cloudTypeLabel(store.selectedCloud.cloud_type) }}</dd></div>
              <div><dt>Масса</dt><dd>{{ fixed(store.selectedCloud.mass) }}</dd></div>
              <div><dt>Стабильность</dt><dd>{{ fixed(store.selectedCloud.stability) }}</dd></div>
              <div><dt>Наблюдения</dt><dd>{{ store.selectedCloud.observation_count }}</dd></div>
            </dl>
          </section>
          <section>
            <h3>Размещение</h3>
            <dl>
              <div><dt>ID</dt><dd>{{ store.selectedPlacement.id }}</dd></div>
              <div><dt>Пространство</dt><dd>{{ store.selectedPlacement.space_id }}</dd></div>
              <div><dt>Координаты</dt><dd>{{ fixed(store.selectedPlacement.x) }}, {{ fixed(store.selectedPlacement.y) }}</dd></div>
              <div><dt>Активация</dt><dd>{{ fixed(store.selectedPlacement.local_activation) }}</dd></div>
              <div><dt>Гравитация</dt><dd>{{ fixed(store.selectedPlacement.local_gravity) }}</dd></div>
            </dl>
          </section>
          <section v-if="store.selectedSceneComponent">
            <h3>Компонент сцены</h3>
            <dl>
              <div><dt>Токен</dt><dd>#{{ store.selectedSceneComponent.token_index }}</dd></div>
              <div><dt>Роль</dt><dd>{{ roleLabel(store.selectedSceneComponent.grammatical_role) }}</dd></div>
              <div><dt>Зависимость</dt><dd>{{ roleLabel(store.selectedSceneComponent.dependency_role) }}</dd></div>
              <div><dt>Уверенность</dt><dd>{{ fixed(store.selectedSceneComponent.confidence) }}</dd></div>
            </dl>
          </section>
          <section v-if="store.selectedStructure">
            <h3>Структура</h3>
            <p>{{ store.selectedStructure.components.length }} компонентов · пространство {{ store.selectedStructure.structure_space?.id }}</p>
            <button class="open" @click="store.zoomIntoPlacement(store.selectedPlacement.id)">Открыть структуру</button>
          </section>
          <button v-if="store.selectedCloud.cloud_type === 'scene'" class="open" @click="store.zoomIntoPlacement(store.selectedPlacement.id)">Открыть сцену</button>
        </template>
        <div v-else class="empty">Идентичность облака и локальные координаты показываются раздельно.</div>
      </aside>
    </main>

    <footer class="stats">
      <div><span>Облака</span><strong>{{ store.stats.clouds_total }}</strong></div>
      <div><span>Словоформы</span><strong>{{ store.stats.unique_word_forms }}</strong></div>
      <div><span>Пространства</span><strong>{{ store.stats.spaces_total }}</strong></div>
      <div><span>Размещения</span><strong>{{ store.stats.placements_total }}</strong></div>
      <div><span>Компоненты сцен</span><strong>{{ store.stats.scene_components_total }}</strong></div>
      <div><span>Структурные компоненты</span><strong>{{ store.stats.structural_components_total }}</strong></div>
      <div><span>Понятия</span><strong>{{ store.stats.concepts_total }}</strong></div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import SpaceVisualization from '@/components/SpaceVisualization.vue'
import { useModelV2Store } from '@/stores/modelV2'

const store = useModelV2Store()
const text = ref('')
const renderer = ref<InstanceType<typeof SpaceVisualization> | null>(null)
const showModelData = ref(false)
const copyStatus = ref('')
const modelJson = computed(() => store.trainedModel ? JSON.stringify(store.trainedModel, null, 2) : '')

function count(key: string) {
  const value = store.lastTraining?.[key]
  return Array.isArray(value) ? value.length : 0
}

function fixed(value: number) {
  return Number(value || 0).toFixed(2)
}

function cloudTypeLabel(type: string) {
  return ({
    scene: 'сцена', word_form: 'словоформа', lexeme: 'лексема',
    concept_candidate: 'кандидат понятия', concept: 'понятие', character: 'символ',
  } as Record<string, string>)[type] || type
}

function roleLabel(role: string | null) {
  return ({
    subject: 'подлежащее', predicate: 'сказуемое', object: 'дополнение',
    attribute: 'определение', location: 'обстоятельство места', definition: 'определение',
    complement: 'дополнение', preposition: 'предлог', service: 'служебное слово',
    root: 'корень', marker: 'маркер', modifies: 'определяет', defines: 'определяет',
    prepositional: 'предложная связь', unknown: 'не определена',
  } as Record<string, string>)[role || ''] || role || '—'
}

async function learn() {
  const value = text.value.trim()
  if (!value) return
  await store.train(value)
  if (showModelData.value) await refreshModelData()
  text.value = ''
}

async function clear() {
  if (!window.confirm('Удалить всю обученную V2-модель и все ульи?')) return
  await store.clearModel()
}

async function refreshModelData() {
  copyStatus.value = ''
  await store.loadTrainedModel()
}

async function toggleModelData() {
  showModelData.value = !showModelData.value
  if (showModelData.value && !store.trainedModel) await refreshModelData()
}

async function copyModelData() {
  if (!modelJson.value) return
  try {
    await navigator.clipboard.writeText(modelJson.value)
    copyStatus.value = 'Скопировано'
  } catch {
    copyStatus.value = 'Не удалось скопировать'
  }
}

onMounted(() => store.loadField())
</script>

<style scoped>
.field-page{min-height:100vh;color:#dce8f8;background:#07111f;font-family:Inter,system-ui,sans-serif}.topbar{height:68px;display:flex;align-items:center;justify-content:space-between;padding:0 26px;border-bottom:1px solid rgba(160,190,225,.14);background:#091524}.brand,.topbar nav{display:flex;align-items:center;gap:12px}.brand>span{display:grid;place-items:center;width:34px;height:34px;border-radius:10px;color:#07111f;background:#7ee9d0}.brand strong,.brand small{display:block}.brand small{margin-top:2px;color:#7388a8;font-size:10px;text-transform:uppercase;letter-spacing:.08em}.topbar a{color:#7ee9d0;text-decoration:none;font-size:12px}.status{padding:5px 8px;border:1px solid rgba(126,233,208,.25);border-radius:6px;color:#7ee9d0;font-size:9px}.breadcrumbs{height:38px;display:flex;align-items:center;padding:0 22px;border-bottom:1px solid rgba(160,190,225,.1);overflow:auto}.breadcrumbs button{border:0;color:#8fa5c5;background:transparent;white-space:nowrap;cursor:pointer;font:11px system-ui}.breadcrumbs button:last-child{color:#dce8f8}.breadcrumbs span{margin-left:9px;color:#4e6382}.workspace{display:grid;grid-template-columns:280px minmax(480px,1fr) 300px;gap:12px;height:calc(100vh - 156px);padding:12px;box-sizing:border-box}.panel,.field-panel{min-width:0;overflow:auto;border:1px solid rgba(160,190,225,.14);border-radius:14px;background:rgba(12,26,45,.82)}.panel{padding:17px}.panel-title span,.telemetry>span{color:#73b0ff;font-size:9px;letter-spacing:.13em}.panel-title h1,.panel-title h2{margin:4px 0 16px;font-size:17px}.training-panel textarea{width:100%;resize:vertical;box-sizing:border-box;border:1px solid rgba(160,190,225,.18);border-radius:9px;padding:10px;color:#e6effc;background:#081421;outline:none;font:12px/1.5 system-ui}.actions{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}.actions button,.open,.controls button{border:0;border-radius:7px;padding:9px;cursor:pointer;font:11px system-ui}.primary,.open{color:#06121e;background:#7ee9d0}.danger{color:#ffb29c;background:rgba(255,101,79,.12)}button:disabled{opacity:.35;cursor:not-allowed}.error{color:#ffad91;font-size:11px}.telemetry{margin-top:20px;padding-top:16px;border-top:1px solid rgba(160,190,225,.1)}.telemetry code{display:block;margin:7px 0 11px;overflow:hidden;color:#7890b0;font-size:9px;text-overflow:ellipsis}.telemetry dl,.inspector dl{margin:0}.telemetry dl div,.inspector dl div{display:flex;justify-content:space-between;gap:10px;padding:5px 0;border-bottom:1px solid rgba(160,190,225,.07);font-size:10px}.telemetry dt,.inspector dt{color:#7187a7}.telemetry dd,.inspector dd{margin:0;color:#e0ebf9}.field-panel{position:relative;overflow:hidden}.loading{position:absolute;inset:0;display:grid;place-items:center;color:#7ee9d0;background:rgba(5,13,25,.62);font-size:12px}.controls{position:absolute;left:14px;bottom:14px;display:flex;gap:5px}.controls button{border:1px solid rgba(160,190,225,.18);color:#c9d8ec;background:rgba(6,15,29,.86)}.inspector section{margin:0 0 17px;padding:0 0 13px;border-bottom:1px solid rgba(160,190,225,.09)}.inspector h3{margin:0 0 7px;color:#7ee9d0;font-size:10px;text-transform:uppercase;letter-spacing:.08em}.inspector p,.empty{color:#7187a7;font-size:11px;line-height:1.5}.open{width:100%;margin-top:7px}.stats{height:50px;display:flex;align-items:center;justify-content:center;gap:34px;border-top:1px solid rgba(160,190,225,.12);background:#091524}.stats div{display:grid;gap:2px}.stats span{color:#6e83a3;font-size:8px;text-transform:uppercase;letter-spacing:.08em}.stats strong{color:#e6effc;font-size:14px;font-weight:500}@media(max-width:1100px){.workspace{grid-template-columns:250px 1fr}.inspector{grid-column:1/-1;max-height:300px}.stats{overflow:auto;justify-content:flex-start;padding:0 18px}}@media(max-width:700px){.workspace{display:flex;flex-direction:column;height:auto}.field-panel{min-height:540px}.training-panel,.inspector{max-height:none}.stats{position:sticky;bottom:0}.topbar{padding:0 14px}}
.model-data{margin-top:20px;padding-top:16px;border-top:1px solid rgba(160,190,225,.1)}.model-data-head,.model-data-actions{display:flex;align-items:center;justify-content:space-between;gap:8px}.model-data-head>span{color:#73b0ff;font-size:9px;letter-spacing:.13em}.model-data-actions{margin-top:8px}.secondary{border:1px solid rgba(126,233,208,.27);border-radius:7px;padding:6px 8px;color:#bceee4;background:rgba(126,233,208,.08);cursor:pointer;font:10px system-ui}.model-data pre{max-height:250px;margin:9px 0 0;padding:9px;overflow:auto;border:1px solid rgba(160,190,225,.12);border-radius:7px;color:#b9cae2;background:#081421;font:9px/1.45 ui-monospace,Consolas,monospace;white-space:pre-wrap;word-break:break-word}
</style>
