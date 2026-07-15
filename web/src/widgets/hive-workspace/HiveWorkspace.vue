<template>
  <aside class="panel hive-panel">
    <div class="panel-head">
      <div>
        <div class="kicker">РАБОЧАЯ ПАМЯТЬ</div>
        <h2>Управление и инспектор</h2>
      </div>
      <span class="hive-count">{{ hiveStore.workingCellCount }} / {{ hiveStore.hive?.max_cells || 24 }} рабочих ячеек</span>
    </div>
    <div class="hive-summary">
      <div>
        <span class="tiny-label">Активация</span>
        <strong>{{ formatPercent(hiveStore.averageActivation) }}</strong>
      </div>
      <div>
        <span class="tiny-label">Удержание</span>
        <strong>{{ formatPercent(hiveStore.averageRetention) }}</strong>
      </div>
      <div>
        <span class="tiny-label">Источники памяти</span>
        <strong>{{ hiveStore.memorySourceCount }}</strong>
      </div>
      <div>
        <span class="tiny-label">Инспекция</span>
        <strong>{{ hiveStore.inspectionProjections.length }}</strong>
      </div>
    </div>
    <section v-if="hiveStore.queryScene" class="query-summary">
      <span class="tiny-label">СЦЕНА ЗАПРОСА · {{ hiveStore.queryScene.status }}</span>
      <strong>{{ hiveStore.queryAnswer?.surface_answer || 'Заполнение пустой роли…' }}</strong>
      <span>{{ hiveStore.queryCandidates.length }} кандидатов · шаг {{ hiveStore.vibrationHistory.length }}</span>
    </section>
    <section v-if="hiveStore.queryFrame" class="query-summary dialogue-context">
      <span class="tiny-label">ТЕКУЩИЙ ЗАПРОС</span>
      <strong>{{ queryText }}</strong>
      <template v-if="reconstructedQuery">
        <span class="tiny-label">ВОССТАНОВЛЕННЫЙ СМЫСЛ</span>
        <strong>{{ reconstructedQuery }}</strong>
      </template>
      <span class="tiny-label">КОНТЕКСТ ИЗ ПАМЯТИ</span>
      <span>{{ contextTerms.length ? contextTerms.join(' · ') : '—' }}</span>
    </section>
    <section v-if="hiveStore.dynamics" class="dynamics-summary">
      <div class="reasoning-head"><span class="tiny-label">ДИНАМИКА УЛЬЯ</span><span>{{ hiveStore.dynamics.status }}</span></div>
      <div class="dynamics-grid">
        <div><span>Температура</span><b>{{ Number(hiveStore.dynamics.temperature.current || 0).toFixed(2) }}</b></div>
        <div><span>Режим</span><b>{{ hiveStore.dynamics.temperature.state || hiveStore.dynamics.temperature.status }}</b></div>
        <div><span>Средняя масса</span><b>{{ averageDynamics('local_mass').toFixed(2) }}</b></div>
        <div><span>Гравитация</span><b>{{ averageDynamics('gravity').toFixed(2) }}</b></div>
        <div><span>Память</span><b>{{ Math.round(hiveStore.dynamics.capacity_pressure * 100) }}%</b></div>
        <div><span>Шаг</span><b>{{ hiveStore.dynamics.step }}</b></div>
      </div>
      <div class="temperature-scale"><i :style="{ left: `${Number(hiveStore.dynamics.temperature.current || 0) * 100}%` }"></i></div>
      <div class="temperature-labels"><span>заморозка</span><span>стабилизация</span><span>поиск</span><span>хаос</span></div>
      <div class="dynamics-counts"><span>активны {{ dynamicsCount('ACTIVE') }}</span><span>ослабевают {{ dynamicsCount('WEAKENING') }}</span><span>вытесняются {{ dynamicsCount('DRIFTING_OUT') + dynamicsCount('AT_BOUNDARY') }}</span></div>
      <table class="weights-table">
        <thead><tr><th>Элемент</th><th>Масса</th><th>Акт.</th><th>Грав.</th><th>Риск</th></tr></thead>
        <tbody><tr v-for="node in hiveStore.dynamics.nodes" :key="node.cell_id"><td>{{ node.label || node.cell_id }}</td><td>{{ node.mass.local.toFixed(2) }}</td><td>{{ Math.round(node.activation * 100) }}%</td><td>{{ node.gravity.toFixed(2) }}</td><td>{{ Math.round(node.eviction_score * 100) }}%</td></tr></tbody>
      </table>
    </section>
    <section class="reasoning-controls">
      <div class="reasoning-head">
        <span class="tiny-label">ВИБРАЦИЯ УЛЬЯ</span>
        <span>{{ reasoningStatus }}</span>
      </div>
      <label>Шаги <input v-model.number="hiveStore.reasoningSteps" type="number" min="0" max="32" /></label>
      <div class="reasoning-actions">
        <button class="secondary" :disabled="hiveStore.reasoningLoading || !hiveStore.hive" @click="hiveStore.runReasoningStep">Один шаг</button>
        <button class="primary" :disabled="hiveStore.reasoningLoading || !hiveStore.hive" @click="hiveStore.runReasoning">{{ hiveStore.reasoningLoading ? 'Вибрация…' : 'Запустить' }}</button>
        <button class="secondary" :disabled="!hiveStore.queryScene" @click="hiveStore.stopReasoning">Стоп</button>
      </div>
      <div v-if="hiveStore.runResult" class="reasoning-meta">run {{ hiveStore.runResult.run.id }} · {{ hiveStore.runResult.completed_steps }} шагов · {{ hiveStore.runResult.stop_reason }}</div>
      <div v-if="hiveStore.vibrationHistory.length" class="reasoning-meta">Шаг {{ hiveStore.vibrationHistory.length }} · {{ hiveStore.queryCandidates.find(item => item.status === 'winner')?.surface || 'вибрация' }}</div>
      <div class="json-actions">
        <button class="secondary" @click="hiveStore.openJson('current')">Показать JSON</button>
        <button class="secondary" @click="hiveStore.copyJson('current')">{{ hiveStore.copyStatus || 'Копировать JSON' }}</button>
      </div>
    </section>
    <section v-if="hiveStore.selectedCell" class="hive-inspector">
      <div class="inspector-head">
        <div>
          <span class="tiny-label">Выбранная ячейка</span>
          <strong>{{ hiveStore.selectedCell.label }}</strong>
        </div>
        <button class="ghost" @click="hiveStore.selectedCell = null">×</button>
      </div>
      <div class="node-stats">
        <span>активация <b>{{ formatPercent(hiveStore.selectedCell.local_activation) }}</b></span>
        <span>удержание <b>{{ formatPercent(hiveStore.selectedCell.retention) }}</b></span>
      </div>
      <nav class="inspector-tabs" aria-label="Вкладки инспектора">
        <button v-for="tab in inspectorTabs" :key="tab.id" :class="{ active: inspectorTab === tab.id }" @click="inspectorTab = tab.id">{{ tab.label }}</button>
      </nav>
      <div v-if="inspectorTab === 'overview'" class="inspector-overview">
        <span>тип <b>{{ hiveStore.selectedCell.component_class || 'контекст' }}</b></span>
        <span>глубина <b>{{ hiveStore.selectedCell.subspaces?.length || 0 }}</b></span>
        <span>масса <b>{{ formatPercent(hiveStore.selectedCell.stored_strength) }}</b></span>
        <span>статус <b>stable</b></span>
      </div>
      <div v-if="inspectorTab === 'composition'" class="component-list">
        <span
          v-for="component in hiveStore.selectedCell.components || []"
          :key="component.id || component.cloud_id"
          class="component"
        >
          <span>{{ component.canonical_name }}</span>
          <b>{{ formatPercent(component.composition_share) }}</b>
        </span>
      </div>
      <div v-else-if="inspectorTab === 'sources'" class="inspector-detail">
        <span>облако #{{ hiveStore.selectedCell.source_cloud_id }}</span>
        <span>пространство #{{ hiveStore.selectedCell.source_space_id || '—' }}</span>
        <span>{{ hiveStore.externalSearch?.sources?.length || 0 }} внешних источников</span>
      </div>
      <div v-else-if="inspectorTab === 'energy'" class="inspector-detail energy-detail">
        <span>энергия <b>{{ formatPercent(hiveStore.selectedCell.local_activation) }}</b></span>
        <span>температура <b>{{ hiveStore.runResult ? 'динамическая' : 'базовая' }}</b></span>
        <span>скорость <b>0.00</b></span>
      </div>
      <div v-else-if="inspectorTab === 'history'" class="inspector-detail">
        <span v-for="event in cellEvents" :key="event.id">{{ event.reason }}</span>
        <span v-if="!cellEvents.length">История появится после активации ячейки.</span>
      </div>
      <pre v-else class="inspector-json">{{ JSON.stringify(hiveStore.selectedCell, null, 2) }}</pre>
      <button v-if="inspectorTab === 'overview' || inspectorTab === 'forms'" class="secondary" @click="hiveStore.expandCell(hiveStore.selectedCell.id, 'word_form')">
        Раскрыть словоформы
      </button>
      <div v-if="(inspectorTab === 'forms' || inspectorTab === 'overview') && hiveStore.inspectionProjections.length" class="inspector-detail">
        <span>Инспекторная проекция словоформ</span>
        <span>{{ hiveStore.inspectionProjections.reduce((sum, projection) => sum + projection.forms.length, 0) }} известных форм</span>
      </div>
      <div v-if="(inspectorTab === 'forms' || inspectorTab === 'overview') && hiveStore.generationCandidates.length" class="component-list">
        <span v-for="candidate in hiveStore.generationCandidates" :key="String(candidate.id)" class="component">
          <span>{{ candidate.candidate_text }}</span>
          <b>{{ Array.isArray(candidate.character_sequence) ? candidate.character_sequence.join(' → ') : '' }}</b>
        </span>
      </div>
    </section>
  </aside>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useHiveStore } from '@/entities/hive/store';
import { formatPercent } from '@/shared/utils/formatters';

const hiveStore = useHiveStore();
const inspectorTab = ref('overview');
const inspectorTabs = [
  { id: 'overview', label: 'Обзор' }, { id: 'composition', label: 'Состав' }, { id: 'sources', label: 'Источники' },
  { id: 'forms', label: 'Формы' }, { id: 'energy', label: 'Энергия' }, { id: 'history', label: 'История' }, { id: 'json', label: 'JSON' },
];
const cellEvents = computed(() => hiveStore.resonanceEvents.filter(event => event.cell_id === hiveStore.selectedCell?.id));
function dynamicsCount(status: string) { return hiveStore.dynamics?.nodes.filter(node => node.eviction_status === status).length || 0; }
function averageDynamics(field: 'local_mass' | 'gravity') { const nodes = hiveStore.dynamics?.nodes || []; return nodes.length ? nodes.reduce((sum, node) => sum + Number(field === 'local_mass' ? node.mass.local : node.gravity), 0) / nodes.length : 0; }

const reasoningStatus = computed(() =>
  hiveStore.reasoningLoading ? 'выполняется'
    : hiveStore.runResult?.stop_reason || 'готов'
);
const queryText = computed(() => String(hiveStore.queryFrame?.source_text || hiveStore.goalText || '—'));
const reconstructedQuery = computed(() => String(hiveStore.queryFrame?.reconstructed_query || ''));
const contextTerms = computed(() => {
  const context = (hiveStore.queryFrame?.dialogue_context || {}) as Record<string, Record<string, unknown>>;
  return ['agent', 'action', 'object'].map(role => String(context[role]?.surface || context[role]?.lemma || '')).filter(Boolean);
});
</script>

<style scoped lang="scss">
.hive-panel {
  overflow: auto;
}

.hive-count {
  color: #ffc968;
  font-size: 10px;
}

.hive-summary {
  display: flex;
  justify-content: space-around;
  padding: 15px;
  border-bottom: 1px solid rgba(162, 189, 225, 0.1);
}

.hive-summary div {
  display: grid;
  gap: 3px;
}

.hive-summary strong {
  color: #f2f7ff;
  font-size: 19px;
  font-weight: 500;
}

.reasoning-controls {
  margin: 0 16px 16px;
  padding: 12px;
  border: 1px solid rgba(120, 231, 208, 0.2);
  border-radius: 10px;
  background: rgba(5, 14, 28, 0.4);
}

.reasoning-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
  color: #8ca2c1;
  font-size: 10px;
}

.reasoning-controls label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: #8ca2c1;
  font-size: 10px;
}

.reasoning-controls input,
.json-toolbar input,
.json-toolbar select {
  width: 70px;
  border: 1px solid rgba(160, 190, 225, 0.2);
  border-radius: 5px;
  padding: 4px;
  color: #e7f0ff;
  background: #081421;
  font: 11px system-ui;
}

.reasoning-actions {
  display: flex;
  gap: 7px;
  margin-top: 8px;
}

.reasoning-actions button {
  flex: 1;
}

.primary,
.secondary {
  border: 0;
  border-radius: 6px;
  padding: 7px;
  color: #07111f;
  background: #78e7d0;
  cursor: pointer;
  font: 10px system-ui;
}

.secondary {
  border: 1px solid rgba(126, 233, 208, 0.25);
  color: #bceee4;
  background: rgba(126, 233, 208, 0.08);
}

.reasoning-meta {
  margin-top: 8px;
  overflow: hidden;
  color: #7187a7;
  font: 9px ui-monospace, Consolas, monospace;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.json-actions {
  margin-top: 8px;
}

.hive-map {
  position: relative;
  min-height: 260px;
  height: 330px;
  margin: 16px;
  overflow: hidden;
  border: 1px solid rgba(255, 201, 104, 0.15);
  border-radius: 12px;
  background: radial-gradient(circle at center, rgba(133, 92, 31, 0.16), transparent 60%), rgba(7, 17, 31, 0.56);
}

.hive-map::before {
  content: '';
  position: absolute;
  inset: 18px;
  background-image: linear-gradient(30deg, rgba(255, 201, 104, 0.08) 12%, transparent 12.5%, transparent 87%, rgba(255, 201, 104, 0.08) 87.5%), linear-gradient(150deg, rgba(255, 201, 104, 0.08) 12%, transparent 12.5%, transparent 87%, rgba(255, 201, 104, 0.08) 87.5%);
  background-size: 34px 58px;
  clip-path: polygon(25% 6%, 75% 6%, 100% 50%, 75% 94%, 25% 94%, 0 50%);
}

.hive-graph {
  position: relative;
  width: 100%;
  height: 100%;
  display: block;
}

.hive-graph line {
  stroke: #6ca2ff;
  stroke-opacity: 0.42;
}

.hive-node {
  cursor: pointer;
}

.hive-node > circle:first-child {
  fill: rgba(120, 231, 208, 0.16);
  stroke: rgba(120, 231, 208, 0.82);
  stroke-width: 2;
}

.hive-node .node-core {
  fill: rgba(15, 38, 60, 0.88);
  stroke: rgba(255, 201, 104, 0.42);
  stroke-width: 1;
}

.hive-node.selected > circle:first-child,
.hive-node.active > circle:first-child,
.hive-node:hover > circle:first-child {
  fill: rgba(255, 201, 104, 0.2);
  stroke: #ffc968;
  filter: drop-shadow(0 0 9px rgba(255, 201, 104, 0.55));
}

.node-label {
  fill: #d9f6ef;
  font: 11px system-ui;
  pointer-events: none;
}

.node-gravity {
  fill: #ffc968;
  font: 10px system-ui;
  pointer-events: none;
}

.hive-empty {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: #607590;
  font-size: 11px;
}

.hive-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  padding: 0 18px 12px;
  color: #8ca2c1;
  font-size: 10px;
}

.hive-legend i,
.hive-legend b {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-right: 4px;
  border-radius: 50%;
  background: #78e7d0;
}

.hive-legend b {
  width: 15px;
  height: 2px;
  border-radius: 0;
  background: #6ca2ff;
  vertical-align: middle;
}

.hive-inspector {
  margin: 0 18px 16px;
  padding: 14px;
  border: 1px solid rgba(120, 231, 208, 0.24);
  border-radius: 11px;
  background: rgba(5, 14, 28, 0.46);
}

.inspector-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
}

.inspector-head strong {
  display: block;
  max-width: 250px;
  margin-top: 3px;
  color: #ecf7ff;
  font-size: 13px;
  line-height: 1.35;
}

.node-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 10px 0;
  color: #8fa7c7;
  font-size: 10px;
}

.node-stats span {
  padding: 5px 7px;
  border-radius: 5px;
  background: rgba(108, 162, 255, 0.1);
}

.node-stats b {
  color: #dff7f1;
}

.dynamics-summary { margin: 0 16px 12px; padding: 12px; border: 1px solid rgba(255, 201, 104, .25); border-radius: 10px; background: rgba(35, 27, 12, .35); }
.dynamics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 10px 0; }
.dynamics-grid div { display: grid; gap: 3px; color: #8298b6; font-size: 9px; }
.dynamics-grid b { color: #f5dfaa; font-size: 12px; font-weight: 500; }
.temperature-scale { position: relative; height: 5px; margin-top: 8px; border-radius: 5px; background: linear-gradient(90deg, #7699c7 0 10%, #78e7d0 10% 30%, #ffc968 30% 60%, #e56b6f 60%); }
.temperature-scale i { position: absolute; top: -4px; width: 3px; height: 13px; border-radius: 2px; background: #fff; box-shadow: 0 0 6px #fff; }
.temperature-labels,.dynamics-counts { display: flex; justify-content: space-between; margin-top: 5px; color: #7589a6; font-size: 8px; }
.dynamics-counts { justify-content: flex-start; gap: 9px; margin-top: 9px; color: #b5c5dc; }
.weights-table { width: 100%; margin-top: 10px; border-collapse: collapse; color: #9eb2cd; font-size: 9px; }
.weights-table th { color: #6f86a5; font-weight: 500; text-align: left; }
.weights-table td,.weights-table th { padding: 4px 3px; border-bottom: 1px solid rgba(160, 190, 225, .08); }
.weights-table td:first-child { max-width: 92px; overflow: hidden; color: #e5f3ff; text-overflow: ellipsis; white-space: nowrap; }

.query-summary { display: grid; gap: 5px; margin: 12px 16px; padding: 10px; border: 1px solid rgba(255, 201, 104, .28); border-radius: 9px; color: #9bb0cc; font-size: 10px; }
.query-summary strong { color: #f0f7ff; font-size: 12px; font-weight: 500; }

.inspector-tabs {
  display: flex;
  gap: 4px;
  overflow-x: auto;
  margin: 10px 0;
  padding-bottom: 3px;
}

.inspector-tabs button {
  flex: 0 0 auto;
  border: 0;
  border-bottom: 1px solid transparent;
  padding: 4px 2px;
  color: #7f96b5;
  background: none;
  font: 9px system-ui;
  cursor: pointer;
}

.inspector-tabs button.active {
  border-color: #78e7d0;
  color: #d7f8f0;
}

.inspector-overview,
.inspector-detail {
  display: grid;
  gap: 6px;
  margin: 8px 0;
  color: #9bb0cc;
  font-size: 10px;
}

.inspector-overview { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.inspector-overview span,
.inspector-detail span { display: flex; justify-content: space-between; gap: 6px; }
.inspector-overview b,
.inspector-detail b { color: #dff7f1; font-weight: 500; }
.energy-detail b { color: #ffc968; }
.inspector-json { max-height: 160px; overflow: auto; margin: 8px 0; color: #9dd3c8; font: 9px ui-monospace, Consolas, monospace; white-space: pre-wrap; }

.component-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.component {
  display: flex;
  align-items: center;
  gap: 7px;
  max-width: 100%;
  padding: 6px 8px;
  border: 1px solid rgba(108, 162, 255, 0.2);
  border-radius: 6px;
  color: #c7d5e9;
  background: rgba(21, 41, 68, 0.7);
  font: 11px system-ui;
}

.component b {
  color: #ffc968;
  font-weight: 500;
}

@media (max-width: 1180px) {
  .hive-panel {
    grid-column: 1 / -1;
    min-height: 360px;
  }
}

@media (max-width: 760px) {
  .hive-panel {
    grid-column: 1 / -1;
    min-height: 420px;
  }
}
</style>
