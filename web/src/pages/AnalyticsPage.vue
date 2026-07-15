<template>
  <main class="analytics-page">
    <header class="topbar">
      <div>
        <div class="kicker">HIVE ANALYTICS</div>
        <h1>Лаборатория улья</h1>
      </div>
      <RouterLink class="nav-link" :to="{ name: 'chat' }">Вернуться в чат</RouterLink>
    </header>

    <section v-if="!hasHive" class="empty-state">
      <h2>Нет активного улья</h2>
      <p>Создайте улей и выполните хотя бы один запуск вибрации, чтобы открыть его аналитику.</p>
      <RouterLink class="primary-link" :to="{ name: 'chat' }">Открыть чат с ульем</RouterLink>
    </section>

    <template v-else>
      <nav class="breadcrumbs" aria-label="Хлебные крошки">
        <RouterLink :to="{ name: 'chat' }">Чат с ульем</RouterLink>
        <RouterLink :to="{ name: 'chat' }">Улей</RouterLink>
        <span>Аналитика</span>
        <span v-if="projectionLabel">Проекция: {{ projectionLabel }}</span>
        <span v-if="primary">Запуск {{ compactId(primary.run.id) }}</span>
        <span v-if="selectedSnapshot">Шаг {{ selectedSnapshot.step }}</span>
      </nav>

      <div class="run-controls">
        <label>
          Основной запуск
          <select v-model="primaryRunId" :disabled="loading || !data?.runs.length" @change="load">
            <option v-for="run in data?.runs || []" :key="run.id" :value="run.id">
              {{ runLabel(run) }}
            </option>
          </select>
        </label>
        <label>
          Сравнить с
          <select v-model="comparisonRunId" :disabled="loading || !data?.runs.length" @change="load">
            <option value="">Не сравнивать</option>
            <option v-for="run in comparisonOptions" :key="run.id" :value="run.id">
              {{ runLabel(run) }}
            </option>
          </select>
        </label>
        <button class="secondary" :disabled="loading" @click="load">Обновить</button>
      </div>

      <p v-if="error" class="error">{{ error }}</p>
      <p v-else-if="loading" class="loading">Загрузка истории запусков…</p>
      <section v-else-if="!primary && liveSnapshot" class="panel current-only">
        <div class="section-head">
          <div>
            <div class="kicker">ТЕКУЩИЙ ЗАПРОС</div>
            <h2>{{ hasAnswerRole ? 'Кандидаты ответа без встряски' : 'Сцены текущего запроса' }}</h2>
          </div>
          <InfoTooltip label="Живой срез" text="Этот анализ строится сразу после обработки сообщения по текущей памяти улья. Вибрация для него не запускается." />
        </div>
        <div v-if="!liveSnapshot.candidates.length" class="empty-inline">Для текущего запроса нет сцен-кандидатов.</div>
        <div v-else class="candidate-list">
          <button v-for="(candidate, index) in liveSnapshot.candidates" :key="candidate.placement_id" class="candidate-row" :class="{ best: index === 0 && candidate.eviction_status === 'ACTIVE' }" :disabled="!candidate.cell_id" @click="openCell(candidate.cell_id)">
            <span class="rank">#{{ index + 1 }}</span>
            <span class="candidate-main"><b>{{ candidate.answer || candidate.scene_label }}</b><small>{{ candidate.scene_label }}</small><small>{{ candidate.explanation }}</small></span>
            <span class="candidate-score"><b>{{ formatPercent(candidate.candidate_score) }}</b><small>{{ statusLabel(candidate.eviction_status) }}</small></span>
          </button>
        </div>
      </section>
      <section v-else-if="!primary" class="empty-state compact">
        <h2>Запусков пока нет</h2>
        <p>Запустите встряску в панели улья — здесь появится её история.</p>
      </section>

      <template v-else>
        <section class="summary-grid" aria-label="Сводка запуска">
          <article class="summary-card">
            <span>Запрос</span>
            <strong>{{ primary.run.query.terms?.join(' ') || '—' }}</strong>
            <small>{{ primary.run.completed_steps }} из {{ primary.run.reasoning_steps }} шагов · {{ stopReason(primary.run.stop_reason) }}</small>
          </article>
          <article class="summary-card">
            <span class="metric-label">Температура <InfoTooltip label="Температура" text="Уровень исследования и шума при встряске. На каждом шаге затухает; это не уверенность в ответе." /></span>
            <strong>{{ formatNumber(initialSnapshot?.temperature || 0, 3) }} → {{ formatNumber(displaySnapshot?.temperature || 0, 3) }}</strong>
            <small>{{ isLiveView ? 'состояние после текущего запроса' : 'конец выбранного шага' }}</small>
          </article>
          <article class="summary-card">
            <span class="metric-label">Энергия <InfoTooltip label="Энергия" text="Локальный ресурс узлов. Он растёт от поддержки запросом и уменьшается во время распространения и затухания." /></span>
            <strong>{{ formatNumber(displaySnapshot?.metrics.total_energy || 0) }}</strong>
            <small>{{ displaySnapshot?.metrics.active_nodes || 0 }} активных · {{ displaySnapshot?.metrics.evicted_nodes || 0 }} вытесненных</small>
          </article>
          <article class="summary-card">
            <span class="metric-label">Удержание <InfoTooltip label="Удержание" text="Насколько прочно ячейка остаётся в рабочей памяти. Низкое удержание переводит узел к ослаблению и затем к вытеснению." /></span>
            <strong>{{ formatPercent(displaySnapshot?.metrics.average_retention || 0) }}</strong>
            <small>среднее по {{ displaySnapshot?.nodes.length || 0 }} узлам</small>
          </article>
        </section>

        <section class="panel timeline-panel">
          <div class="section-head">
            <div>
              <div class="kicker">ДИНАМИКА</div>
              <h2>Состояние по шагам</h2>
            </div>
            <label class="step-control">
              Выбранный шаг
              <select v-model.number="selectedStep">
                <option v-for="snapshot in timeline" :key="`${snapshot.phase}-${snapshot.step}`" :value="snapshot.step">
                  {{ snapshot.phase === 'INITIAL' ? 'Исходное состояние' : `Шаг ${snapshot.step}` }}
                </option>
              </select>
            </label>
          </div>
          <div class="charts">
            <article v-for="metric in metrics" :key="metric.key" class="trend">
              <div class="trend-head"><span>{{ metric.label }} <InfoTooltip :label="metric.label" :text="metric.info" /></span><b>{{ metric.value(selectedSnapshot) }}</b></div>
              <svg viewBox="0 0 300 110" role="img" :aria-label="`${metric.label} по шагам`">
                <line x1="12" y1="94" x2="288" y2="94" />
                <polyline :points="trendPoints(metric.key)" />
                <circle
                  v-for="(snapshot, index) in timeline"
                  :key="`${metric.key}-${snapshot.step}-${snapshot.phase}`"
                  :cx="trendX(index)"
                  :cy="trendY(metric.key, index)"
                  :class="{ selected: snapshot.step === selectedStep }"
                  r="4"
                  @click="selectHistoryStep(snapshot.step)"
                />
              </svg>
            </article>
          </div>
          <div class="event-strip" aria-label="События запуска">
            <span v-if="!primary.events.length">Событий не зафиксировано</span>
            <button v-for="event in primary.events" :key="String(event.id)" class="event" @click="selectHistoryStep(Number(event.step))">
              шаг {{ event.step }} · {{ eventLabel(String(event.event_type)) }}
            </button>
          </div>
        </section>

        <section class="two-columns">
          <article class="panel candidates-panel">
            <div class="section-head">
              <div>
                <div class="kicker">ОБЪЯСНИМЫЙ РАНГ</div>
                <h2>{{ hasAnswerRole ? 'Кандидаты ответа' : 'Кандидаты-сцены' }}</h2>
              </div>
              <span class="note">Это внутренний ранг, не вероятность.</span>
            </div>
            <div class="view-switch" role="group" aria-label="Источник аналитики">
              <button :class="{ active: isLiveView }" @click="viewMode = 'current'">Текущий запрос</button>
              <button :class="{ active: !isLiveView }" @click="viewMode = 'history'">Выбранный шаг</button>
              <InfoTooltip label="Текущий запрос" text="Живой срез обновляется сразу после обработки сообщения. Он не запускает и не имитирует встряску." />
            </div>
            <p class="method">70% совпадение слов и ролей · 30% динамика вибрации · жизнеспособность по статусу узла.</p>
            <div v-if="!displaySnapshot?.candidates.length" class="empty-inline">В выбранном состоянии нет сцен-кандидатов.</div>
            <div v-else class="candidate-list">
              <button
                v-for="(candidate, index) in displaySnapshot.candidates"
                :key="candidate.placement_id"
                class="candidate-row"
                :class="{ best: index === 0 && candidate.eviction_status === 'ACTIVE' }"
                :disabled="!candidate.cell_id"
                @click="openCell(candidate.cell_id)"
              >
                <span class="rank">#{{ index + 1 }}</span>
                <span class="candidate-main">
                  <b>{{ candidate.answer || candidate.scene_label }}</b>
                  <small>{{ candidate.scene_label }}</small>
                  <small>{{ candidate.explanation }}</small>
                </span>
                <span class="candidate-score">
                  <b>{{ formatPercent(candidate.candidate_score) }}</b>
                  <small>{{ statusLabel(candidate.eviction_status) }}</small>
                </span>
              </button>
            </div>
          </article>

          <article class="panel nodes-panel">
            <div class="section-head">
              <div>
                <div class="kicker">{{ isLiveView ? 'УЗЛЫ ТЕКУЩЕГО ЗАПРОСА' : 'УЗЛЫ ШАГА' }}</div>
                <h2>Текущая картина</h2>
              </div>
              <span class="note">Выберите узел для перехода к улью.</span>
            </div>
            <div class="node-list">
              <button
                v-for="node in displaySnapshot?.nodes || []"
                :key="node.placement_id"
                class="node-row"
                :disabled="!node.cell_id"
                @click="openCell(node.cell_id)"
              >
                <span><b>{{ node.label }}</b><small>{{ node.node_type }} · {{ statusLabel(node.eviction_status) }}</small></span>
                <span><small>акт. {{ formatPercent(node.local_activation) }}</small><small>удерж. {{ formatPercent(node.retention) }}</small></span>
              </button>
            </div>
          </article>
        </section>

        <section v-if="comparisonFinal" class="panel comparison-panel">
          <div class="section-head">
            <div>
              <div class="kicker">СРАВНЕНИЕ</div>
              <h2>Последний и предыдущий запуск</h2>
            </div>
            <span class="note">{{ compactId(primary.run.id) }} против {{ compactId(comparison?.run.id || '') }}</span>
          </div>
          <div class="comparison-grid">
            <div><span>Средняя активация</span><b>{{ formatPercent(selectedSnapshot?.metrics.average_activation || 0) }}</b><small>{{ signedDelta((selectedSnapshot?.metrics.average_activation || 0) - comparisonFinal.metrics.average_activation) }}</small></div>
            <div><span>Среднее удержание</span><b>{{ formatPercent(selectedSnapshot?.metrics.average_retention || 0) }}</b><small>{{ signedDelta((selectedSnapshot?.metrics.average_retention || 0) - comparisonFinal.metrics.average_retention) }}</small></div>
            <div><span>Вытеснено узлов</span><b>{{ selectedSnapshot?.metrics.evicted_nodes || 0 }}</b><small>{{ signedNumber((selectedSnapshot?.metrics.evicted_nodes || 0) - comparisonFinal.metrics.evicted_nodes) }}</small></div>
            <div><span>Лучший кандидат</span><b>{{ selectedSnapshot?.candidates[0]?.answer || selectedSnapshot?.candidates[0]?.scene_label || '—' }}</b><small>было: {{ comparisonFinal.candidates[0]?.answer || comparisonFinal.candidates[0]?.scene_label || '—' }}</small></div>
          </div>
        </section>
      </template>
    </template>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { useHiveAnalytics } from '@/features/hive-analytics';
import InfoTooltip from '@/components/InfoTooltip.vue';
import type { HiveAnalyticsRunResultV2, HiveAnalyticsSnapshotV2, HiveAnalyticsRunV2 } from '@/entities/hive/types';
import { formatNumber, formatPercent } from '@/shared/utils/formatters';

const router = useRouter();
const route = useRoute();
const projectionLabel = computed(() => typeof route.query.label === 'string' && route.query.label
  ? route.query.label
  : typeof route.query.level === 'string' ? route.query.level : '');
const { data, loading, error, primaryRunId, comparisonRunId, hasHive, load } = useHiveAnalytics();
const selectedStep = ref(0);
const viewMode = ref<'current' | 'history'>('current');
const primary = computed(() => data.value?.primary || null);
const comparison = computed(() => data.value?.comparison || null);
const comparisonOptions = computed(() => (data.value?.runs || []).filter(run => run.id !== primaryRunId.value));
const timeline = computed(() => primary.value ? snapshotsForTimeline(primary.value) : []);
const initialSnapshot = computed(() => timeline.value[0] || null);
const selectedSnapshot = computed(() => timeline.value.find(item => item.step === selectedStep.value) || lastItem(timeline.value) || null);
const liveSnapshot = computed(() => data.value?.current?.snapshot || null);
const displaySnapshot = computed(() => viewMode.value === 'current' ? liveSnapshot.value || selectedSnapshot.value : selectedSnapshot.value);
const comparisonFinal = computed(() => comparison.value ? lastItem(snapshotsForTimeline(comparison.value)) || null : null);
const isLiveView = computed(() => viewMode.value === 'current' && Boolean(liveSnapshot.value));
const hasAnswerRole = computed(() => {
  const components = isLiveView.value ? data.value?.current?.query_components : primary.value?.query_components;
  return components?.some(item => item.word_form_cloud_id === null) || false;
});

const metrics = [
  { key: 'average_activation', label: 'Активация', info: 'Насколько сильно узлы вовлечены в текущее состояние. Это активность, а не оценка правильности ответа.', value: (snapshot: HiveAnalyticsSnapshotV2 | null) => formatPercent(snapshot?.metrics.average_activation || 0) },
  { key: 'average_retention', label: 'Удержание', info: 'Насколько устойчиво ячейки сохраняются в рабочей памяти; низкое значение ведёт к ослаблению и вытеснению.', value: (snapshot: HiveAnalyticsSnapshotV2 | null) => formatPercent(snapshot?.metrics.average_retention || 0) },
  { key: 'total_energy', label: 'Энергия', info: 'Суммарный ресурс узлов. Поддержка запросом увеличивает его, а затухание физики уменьшает.', value: (snapshot: HiveAnalyticsSnapshotV2 | null) => formatNumber(snapshot?.metrics.total_energy || 0) },
  { key: 'temperature', label: 'Температура', info: 'Масштаб исследовательского шума во время встряски. Она затухает на каждом шаге и не означает уверенность.', value: (snapshot: HiveAnalyticsSnapshotV2 | null) => formatNumber(snapshot?.temperature || 0, 3) },
];

onMounted(() => { void load(); });
watch(() => primary.value?.run.id, () => {
  selectedStep.value = lastItem(timeline.value)?.step || 0;
});

function selectHistoryStep(step: number) {
  selectedStep.value = step;
  viewMode.value = 'history';
}

function snapshotsForTimeline(run: HiveAnalyticsRunResultV2) {
  return run.snapshots.filter(snapshot => snapshot.phase === 'INITIAL' || snapshot.phase === 'AFTER_SETTLE');
}

function lastItem<T>(items: T[]): T | undefined {
  return items[items.length - 1];
}

function compactId(value: string) {
  return value.replace('run-', '').slice(0, 8);
}

function runLabel(run: HiveAnalyticsRunV2) {
  return `${new Date(run.created_at).toLocaleTimeString('ru-RU')} · ${run.completed_steps} шаг.`;
}

function stopReason(value: string | null) {
  return value === 'CONVERGED' ? 'сошёлся' : value === 'COMPLETED' ? 'завершён' : value || '—';
}

function statusLabel(value: string) {
  return ({ ACTIVE: 'активен', WEAKENING: 'ослабевает', EVICTION_CANDIDATE: 'кандидат на вытеснение', EVICTED: 'вытеснен' } as Record<string, string>)[value] || value;
}

function eventLabel(value: string) {
  return ({ QUERY_ENERGY: 'энергия запроса', WEAKENING: 'ослабление', EVICTED: 'вытеснение', SETTLE: 'стабилизация' } as Record<string, string>)[value] || value;
}

function trendValues(key: string) {
  return timeline.value.map(snapshot => key === 'temperature' ? snapshot.temperature : Number(snapshot.metrics[key as keyof HiveAnalyticsSnapshotV2['metrics']]));
}

function trendX(index: number) {
  const count = Math.max(1, timeline.value.length - 1);
  return 12 + (276 * index) / count;
}

function trendY(key: string, index: number) {
  const values = trendValues(key);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const relative = max === min ? 0.5 : (values[index] - min) / (max - min);
  return 94 - relative * 76;
}

function trendPoints(key: string) {
  return timeline.value.map((_, index) => `${trendX(index)},${trendY(key, index)}`).join(' ');
}

function signedDelta(value: number) {
  return `${value >= 0 ? '+' : ''}${formatPercent(value)}`;
}

function signedNumber(value: number) {
  return `${value >= 0 ? '+' : ''}${value}`;
}

function openCell(cellId: string | null) {
  if (!cellId) return;
  void router.push({ name: 'chat', query: { cell: cellId } });
}
</script>

<style scoped lang="scss">
.analytics-page { min-height: 100vh; padding: 24px clamp(16px, 4vw, 56px) 48px; color: #e7f0ff; background: #07111f; }
.topbar, .section-head, .run-controls, .breadcrumbs, .trend-head, .node-row, .candidate-row { display: flex; align-items: center; }
.topbar, .section-head { justify-content: space-between; gap: 18px; }
.topbar { max-width: 1500px; margin: 0 auto 18px; }
h1, h2, p { margin: 0; }
h1 { margin-top: 3px; font-size: 28px; }
h2 { margin-top: 3px; font-size: 18px; }
.kicker { color: #73b0ff; font-size: 10px; letter-spacing: .14em; }
.nav-link, .breadcrumbs a { color: #78e7d0; text-decoration: none; }
.breadcrumbs { flex-wrap: wrap; gap: 8px; max-width: 1500px; margin: 0 auto 18px; color: #8496b5; font-size: 12px; }
.breadcrumbs span:not(:last-child)::after, .breadcrumbs a::after { margin-left: 8px; color: #526783; content: '›'; }
.run-controls { flex-wrap: wrap; gap: 12px; max-width: 1500px; margin: 0 auto 18px; }
.run-controls label, .step-control { display: grid; gap: 5px; color: #8ca2c1; font-size: 11px; }
select, button { font: inherit; }
select { min-width: 180px; border: 1px solid rgba(160,190,225,.22); border-radius: 7px; padding: 7px 9px; color: #e7f0ff; background: #081421; }
.secondary, .primary-link { border: 1px solid rgba(120,231,208,.35); border-radius: 7px; padding: 8px 11px; color: #bceee4; background: rgba(120,231,208,.09); cursor: pointer; text-decoration: none; }
.error { max-width: 1500px; margin: auto; color: #ff9f96; }
.loading { max-width: 1500px; margin: auto; color: #8ca2c1; }
.summary-grid, .two-columns, .comparison-grid { display: grid; gap: 14px; max-width: 1500px; margin: 0 auto 14px; }
.summary-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.summary-card, .panel, .empty-state { border: 1px solid rgba(162,189,225,.16); border-radius: 14px; background: rgba(14,28,48,.76); }
.summary-card { min-height: 100px; padding: 16px; }
.summary-card span, .summary-card small, .candidate-main small, .node-row small, .comparison-grid span, .comparison-grid small { display: block; color: #8496b5; font-size: 11px; }
.summary-card strong { display: block; margin: 8px 0 6px; font-size: 18px; font-weight: 600; }
.panel { padding: 18px; }
.timeline-panel, .comparison-panel { max-width: 1500px; margin: 0 auto 14px; }
.charts { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }
.trend { min-width: 0; border-left: 1px solid rgba(162,189,225,.12); padding-left: 12px; }
.trend:first-child { border-left: 0; padding-left: 0; }
.trend-head { justify-content: space-between; color: #8ca2c1; font-size: 11px; }
.trend-head b { color: #e7f0ff; font-size: 14px; }
.trend svg { width: 100%; margin-top: 6px; overflow: visible; }
.trend line { stroke: rgba(162,189,225,.22); }
.trend polyline { fill: none; stroke: #78e7d0; stroke-width: 2; }
.trend circle { fill: #73b0ff; cursor: pointer; }
.trend circle.selected { fill: #ffc968; stroke: #07111f; stroke-width: 2; }
.event-strip { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 12px; color: #8496b5; font-size: 11px; }
.event { border: 1px solid rgba(115,176,255,.24); border-radius: 999px; padding: 4px 8px; color: #a9c8ef; background: transparent; cursor: pointer; }
.two-columns { grid-template-columns: minmax(0, 1.35fr) minmax(320px, .85fr); }
.note, .method { color: #8496b5; font-size: 11px; }
.metric-label { display: flex !important; align-items: center; }
.view-switch { display: flex; align-items: center; gap: 6px; margin-top: 10px; }
.view-switch button { border: 1px solid rgba(162,189,225,.22); border-radius: 999px; padding: 4px 8px; color: #9db0ce; background: transparent; cursor: pointer; font-size: 11px; }
.view-switch button.active { border-color: rgba(120,231,208,.55); color: #bceee4; background: rgba(120,231,208,.1); }
.method { margin: 10px 0; }
.candidate-list, .node-list { display: grid; gap: 7px; }
.candidate-row, .node-row { width: 100%; gap: 12px; border: 1px solid rgba(162,189,225,.13); border-radius: 9px; padding: 10px; color: #e7f0ff; background: rgba(4,13,26,.42); text-align: left; cursor: pointer; }
.candidate-row:disabled, .node-row:disabled { opacity: .7; cursor: default; }
.candidate-row.best { border-color: rgba(120,231,208,.6); background: rgba(120,231,208,.08); }
.rank { width: 23px; color: #ffc968; font-size: 12px; }
.candidate-main { min-width: 0; flex: 1; }
.candidate-main b, .node-row b { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.candidate-score { min-width: 74px; text-align: right; }
.candidate-score b { display: block; color: #78e7d0; }
.node-row { justify-content: space-between; }
.node-row > span:last-child { text-align: right; }
.comparison-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); margin-top: 14px; }
.comparison-grid > div { border-left: 1px solid rgba(162,189,225,.13); padding-left: 14px; }
.comparison-grid > div:first-child { border-left: 0; padding-left: 0; }
.comparison-grid b { display: block; margin: 6px 0; font-size: 16px; }
.empty-state { max-width: 640px; margin: 13vh auto; padding: 28px; color: #b9c8dd; text-align: center; }
.empty-state h2 { color: #e7f0ff; }
.empty-state p { margin: 10px 0 18px; line-height: 1.55; }
.empty-state.compact { margin: 40px auto; }
.current-only { max-width: 900px; margin: 34px auto; }
.empty-inline { padding: 22px 0; color: #8496b5; text-align: center; }
@media (max-width: 1100px) { .summary-grid, .charts, .comparison-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .trend:nth-child(3) { border-left: 0; padding-left: 0; } }
@media (max-width: 760px) { .analytics-page { padding: 16px; } .topbar { align-items: flex-start; } h1 { font-size: 23px; } .summary-grid, .two-columns, .charts, .comparison-grid { grid-template-columns: 1fr; } .trend, .trend:nth-child(3), .comparison-grid > div { border-left: 0; padding-left: 0; } .section-head { align-items: flex-start; flex-direction: column; } .step-control select { width: 100%; } }
</style>
