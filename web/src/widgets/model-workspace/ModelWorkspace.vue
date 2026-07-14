<template>
  <div class="model-workspace">
    <header class="topbar">
      <div class="brand">
        <span>✦</span>
        <div>
          <strong>SuperAI</strong>
          <small>Облако · Пространство · Размещение</small>
        </div>
      </div>
      <nav>
        <router-link to="/">Чат с ульем</router-link>
        <span class="status">ТОЛЬКО V2</span>
      </nav>
    </header>

    <div class="breadcrumbs">
      <button
        v-for="(item, index) in modelStore.breadcrumb"
        :key="item.space.id"
        @click="modelStore.navigateTo(index)"
      >
        {{ item.label }}<span v-if="index < modelStore.breadcrumb.length - 1">›</span>
      </button>
    </div>

    <main class="workspace">
      <aside class="panel training-panel">
        <div class="panel-title">
          <span>ОБУЧЕНИЕ</span>
          <h1>Обучение модели</h1>
        </div>
        <textarea v-model="text" rows="8" placeholder="Кот ест рыбу. Рыбак ловит рыбу." @keydown.ctrl.enter="learn" />
        <div class="actions">
          <button class="primary" :disabled="modelStore.loading || !text.trim()" @click="learn">Обучить</button>
          <button class="danger" :disabled="modelStore.loading" @click="clear">Очистить данные</button>
        </div>
        <p v-if="modelStore.error" class="error">{{ modelStore.error }}</p>
        <section class="model-data">
          <div class="model-data-head">
            <span>ДАННЫЕ МОДЕЛИ</span>
            <button class="secondary" :disabled="modelStore.loading" @click="toggleModelData">
              {{ showModelData ? 'Скрыть' : 'Показать' }}
            </button>
          </div>
          <template v-if="showModelData">
            <div class="model-data-actions">
              <button class="secondary" :disabled="modelStore.loading" @click="refreshModelData">Обновить</button>
              <button class="secondary" :disabled="!modelJson" @click="copyModelData">{{ copyStatus || 'Копировать JSON' }}</button>
            </div>
            <pre aria-label="Объект обученной модели">{{ modelJson || 'Нет обученных данных.' }}</pre>
          </template>
        </section>
        <div v-if="modelStore.lastTraining" class="telemetry">
          <span>ЗАПУСК ОБУЧЕНИЯ</span>
          <code>{{ modelStore.lastTraining.training_run_id }}</code>
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
          :space="modelStore.currentSpace"
          :clouds="modelStore.cloudsById"
          :placements="modelStore.placements"
          :structure="modelStore.currentStructure"
          :selected-placement-id="modelStore.selectedPlacementId"
          @select-placement="modelStore.selectPlacement"
          @open-placement="modelStore.zoomIntoPlacement"
        />
        <div v-if="modelStore.loading" class="loading">Обновление пространства…</div>
        <div class="controls">
          <button @click="renderer?.zoomBy(1 / 1.2)">−</button>
          <button @click="renderer?.resetView()">Обзор</button>
          <button @click="renderer?.zoomBy(1.2)">+</button>
          <button :disabled="modelStore.currentSpace?.space_type === 'word_structure_space'" @click="modelStore.tickPhysics">Шаг физики</button>
        </div>
      </section>

      <aside class="panel inspector">
        <div class="panel-title">
          <span>ИНСПЕКТОР</span>
          <h2>{{ modelStore.selectedCloud?.canonical_name || 'Выберите размещение' }}</h2>
        </div>
        <template v-if="modelStore.selectedCloud && modelStore.selectedPlacement">
          <section>
            <h3>Облако</h3>
            <dl>
              <div><dt>ID</dt><dd>{{ modelStore.selectedCloud.id }}</dd></div>
              <div><dt>Тип</dt><dd>{{ cloudTypeLabel(modelStore.selectedCloud.cloud_type) }}</dd></div>
              <div><dt>Масса</dt><dd>{{ fixed(modelStore.selectedCloud.mass) }}</dd></div>
              <div><dt>Стабильность</dt><dd>{{ fixed(modelStore.selectedCloud.stability) }}</dd></div>
              <div><dt>Наблюдения</dt><dd>{{ modelStore.selectedCloud.observation_count }}</dd></div>
            </dl>
          </section>
          <section>
            <h3>Размещение</h3>
            <dl>
              <div><dt>ID</dt><dd>{{ modelStore.selectedPlacement.id }}</dd></div>
              <div><dt>Пространство</dt><dd>{{ modelStore.selectedPlacement.space_id }}</dd></div>
              <div><dt>Координаты</dt><dd>{{ fixed(modelStore.selectedPlacement.x) }}, {{ fixed(modelStore.selectedPlacement.y) }}</dd></div>
              <div><dt>Активация</dt><dd>{{ fixed(modelStore.selectedPlacement.local_activation) }}</dd></div>
              <div><dt>Гравитация</dt><dd>{{ fixed(modelStore.selectedPlacement.local_gravity) }}</dd></div>
            </dl>
          </section>
          <section v-if="modelStore.selectedSceneComponent">
            <h3>Компонент сцены</h3>
            <dl>
              <div><dt>Токен</dt><dd>#{{ modelStore.selectedSceneComponent.token_index }}</dd></div>
              <div><dt>Роль</dt><dd>{{ roleLabel(modelStore.selectedSceneComponent.grammatical_role) }}</dd></div>
              <div><dt>Зависимость</dt><dd>{{ roleLabel(modelStore.selectedSceneComponent.dependency_role) }}</dd></div>
              <div><dt>Уверенность</dt><dd>{{ fixed(modelStore.selectedSceneComponent.confidence) }}</dd></div>
            </dl>
          </section>
          <section v-if="modelStore.selectedStructure">
            <h3>Структура</h3>
            <p>{{ modelStore.selectedStructure.components.length }} компонентов · пространство {{ modelStore.selectedStructure.structure_space?.id }}</p>
            <button class="open" @click="modelStore.zoomIntoPlacement(modelStore.selectedPlacement.id)">Открыть структуру</button>
          </section>
          <button v-if="modelStore.selectedCloud.cloud_type === 'scene'" class="open" @click="modelStore.zoomIntoPlacement(modelStore.selectedPlacement.id)">Открыть сцену</button>
        </template>
        <div v-else class="empty">Идентичность облака и локальные координаты показываются раздельно.</div>
      </aside>
    </main>

    <footer class="stats">
      <div><span>Облака</span><strong>{{ modelStore.stats.clouds_total }}</strong></div>
      <div><span>Словоформы</span><strong>{{ modelStore.stats.unique_word_forms }}</strong></div>
      <div><span>Пространства</span><strong>{{ modelStore.stats.spaces_total }}</strong></div>
      <div><span>Размещения</span><strong>{{ modelStore.stats.placements_total }}</strong></div>
      <div><span>Компоненты сцен</span><strong>{{ modelStore.stats.scene_components_total }}</strong></div>
      <div><span>Структурные компоненты</span><strong>{{ modelStore.stats.structural_components_total }}</strong></div>
      <div><span>Понятия</span><strong>{{ modelStore.stats.concepts_total }}</strong></div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import SpaceVisualization from '@/components/SpaceVisualization.vue'
import { useModelStore } from '@/entities/model/store'

const modelStore = useModelStore()
const text = ref('')
const renderer = ref<InstanceType<typeof SpaceVisualization> | null>(null)
const showModelData = ref(false)
const copyStatus = ref('')
const modelJson = computed(() => modelStore.trainedModel ? JSON.stringify(modelStore.trainedModel, null, 2) : '')

function count(key: string) {
  const value = modelStore.lastTraining?.[key]
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
  await modelStore.train(value)
  if (showModelData.value) await refreshModelData()
  text.value = ''
}

async function clear() {
  if (!window.confirm('Удалить всю обученную V2-модель и все ульи?')) return
  await modelStore.clearModel()
}

async function refreshModelData() {
  copyStatus.value = ''
  await modelStore.loadTrainedModel()
}

async function toggleModelData() {
  showModelData.value = !showModelData.value
  if (showModelData.value && !modelStore.trainedModel) await refreshModelData()
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

onMounted(() => modelStore.loadField())
</script>

<style scoped>
.model-workspace {
  min-height: 100vh;
  color: #dce8f8;
  background: #07111f;
  font-family: Inter, system-ui, sans-serif;
}

.topbar {
  height: 68px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 26px;
  border-bottom: 1px solid rgba(160, 190, 225, 0.14);
  background: #091524;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand > span {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: 10px;
  color: #07111f;
  background: #7ee9d0;
}

.brand strong,
.brand small {
  display: block;
}

.brand small {
  margin-top: 2px;
  color: #7388a8;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.topbar a {
  color: #7ee9d0;
  text-decoration: none;
  font-size: 12px;
}

.status {
  padding: 5px 8px;
  border: 1px solid rgba(126, 233, 208, 0.25);
  border-radius: 6px;
  color: #7ee9d0;
  font-size: 9px;
}

.breadcrumbs {
  height: 38px;
  display: flex;
  align-items: center;
  padding: 0 22px;
  border-bottom: 1px solid rgba(160, 190, 225, 0.1);
  overflow: auto;
}

.breadcrumbs button {
  border: 0;
  color: #8fa5c5;
  background: transparent;
  white-space: nowrap;
  cursor: pointer;
  font: 11px system-ui;
}

.breadcrumbs button:last-child {
  color: #dce8f8;
}

.breadcrumbs span {
  margin-left: 9px;
  color: #4e6382;
}

.workspace {
  display: grid;
  grid-template-columns: 280px minmax(480px, 1fr) 300px;
  gap: 12px;
  height: calc(100vh - 156px);
  padding: 12px;
  box-sizing: border-box;
}

.panel,
.field-panel {
  min-width: 0;
  overflow: auto;
  border: 1px solid rgba(160, 190, 225, 0.14);
  border-radius: 14px;
  background: rgba(12, 26, 45, 0.82);
}

.panel {
  padding: 17px;
}

.panel-title span,
.telemetry > span {
  color: #73b0ff;
  font-size: 9px;
  letter-spacing: 0.13em;
}

.panel-title h1,
.panel-title h2 {
  margin: 4px 0 16px;
  font-size: 17px;
}

.training-panel textarea {
  width: 100%;
  resize: vertical;
  box-sizing: border-box;
  border: 1px solid rgba(160, 190, 225, 0.18);
  border-radius: 9px;
  padding: 10px;
  color: #e6effc;
  background: #081421;
  outline: none;
  font: 12px/1.5 system-ui;
}

.actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 10px;
}

.actions button,
.open,
.controls button {
  border: 0;
  border-radius: 7px;
  padding: 9px;
  cursor: pointer;
  font: 11px system-ui;
}

.primary,
.open {
  color: #06121e;
  background: #7ee9d0;
}

.danger {
  color: #ffb29c;
  background: rgba(255, 101, 79, 0.12);
}

button:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.error {
  color: #ffad91;
  font-size: 11px;
}

.telemetry {
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid rgba(160, 190, 225, 0.1);
}

.telemetry code {
  display: block;
  margin: 7px 0 11px;
  overflow: hidden;
  color: #7890b0;
  font-size: 9px;
  text-overflow: ellipsis;
}

.telemetry dl,
.inspector dl {
  margin: 0;
}

.telemetry dl div,
.inspector dl div {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 5px 0;
  border-bottom: 1px solid rgba(160, 190, 225, 0.07);
  font-size: 10px;
}

.telemetry dt,
.inspector dt {
  color: #7187a7;
}

.telemetry dd,
.inspector dd {
  margin: 0;
  color: #e0ebf9;
}

.field-panel {
  position: relative;
  overflow: hidden;
}

.loading {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: #7ee9d0;
  background: rgba(5, 13, 25, 0.62);
  font-size: 12px;
}

.controls {
  position: absolute;
  left: 14px;
  bottom: 14px;
  display: flex;
  gap: 5px;
}

.controls button {
  border: 1px solid rgba(160, 190, 225, 0.18);
  color: #c9d8ec;
  background: rgba(6, 15, 29, 0.86);
}

.inspector section {
  margin: 0 0 17px;
  padding: 0 0 13px;
  border-bottom: 1px solid rgba(160, 190, 225, 0.09)
}

.inspector h3 {
  margin: 0 0 7px;
  color: #7ee9d0;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.inspector p,
.empty {
  color: #7187a7;
  font-size: 11px;
  line-height: 1.5
}

.open {
  width: 100%;
  margin-top: 7px
}

.stats {
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 34px;
  border-top: 1px solid rgba(160, 190, 225, 0.12);
  background: #091524
}

.stats div {
  display: grid;
  gap: 2px
}

.stats span {
  color: #6e83a3;
  font-size: 8px;
  text-transform: uppercase;
  letter-spacing: 0.08em
}

.stats strong {
  color: #e6effc;
  font-size: 14px;
  font-weight: 500
}

@media (max-width: 1100px) {
  .workspace {
    grid-template-columns: 250px 1fr
  }
  .inspector {
    grid-column: 1 / -1;
    max-height: 300px
  }
  .stats {
    overflow: auto;
    justify-content: flex-start;
    padding: 0 18px
  }
}

@media (max-width: 700px) {
  .workspace {
    display: flex;
    flex-direction: column;
    height: auto
  }
  .field-panel {
    min-height: 540px
  }
  .training-panel,
  .inspector {
    max-height: none
  }
  .stats {
    position: sticky;
    bottom: 0
  }
  .topbar {
    padding: 0 14px
  }
}

.model-data {
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid rgba(160, 190, 225, 0.1)
}

.model-data-head,
.model-data-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px
}

.model-data-head > span {
  color: #73b0ff;
  font-size: 9px;
  letter-spacing: 0.13em
}

.model-data-actions {
  margin-top: 8px
}

.secondary {
  border: 1px solid rgba(126, 233, 208, 0.27);
  border-radius: 7px;
  padding: 6px 8px;
  color: #bceee4;
  background: rgba(126, 233, 208, 0.08);
  cursor: pointer;
  font: 10px system-ui
}

.model-data pre {
  max-height: 250px;
  margin: 9px 0 0;
  padding: 9px;
  overflow: auto;
  border: 1px solid rgba(160, 190, 225, 0.12);
  border-radius: 7px;
  color: #b9cae2;
  background: #081421;
  font: 9px/1.45 ui-monospace, Consolas, monospace;
  white-space: pre-wrap;
  word-break: break-word
}
</style>