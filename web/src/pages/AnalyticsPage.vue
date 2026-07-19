<template>
  <main class="analytics-page">
    <header>
      <div>
        <span class="kicker">HIVE ANALYTICS</span>
        <h1>Аналитика запроса</h1>
      </div>
      <nav>
        <RouterLink :to="{ name: 'chat' }">Диалог</RouterLink>
        <a href="/space">Пространства</a>
      </nav>
    </header>

    <p v-if="!hasHive" class="notice">Сначала откройте или создайте диалог.</p>
    <p v-else-if="loading" class="notice">Загрузка внутренней статистики…</p>
    <p v-else-if="error" class="notice error">{{ error }}</p>

    <template v-else-if="snapshot">
      <section class="summary">
        <div>
          <span>{{ primary ? 'Сохранённый прогон' : 'Подробная статистика текущего запроса' }}</span>
          <h2>{{ primary?.run.status || 'Текущая картина' }}</h2>
        </div>
        <p>Это внутренний ранг, не вероятность.</p>
      </section>

      <section class="metrics">
        <article>
          <span>Активация</span>
          <b>{{ percent(snapshot.metrics.average_activation) }}</b>
        </article>
        <article>
          <span>Удержание</span>
          <b>{{ percent(snapshot.metrics.average_retention) }}</b>
        </article>
        <article>
          <span>Энергия</span>
          <b>{{ number(snapshot.metrics.total_energy) }}</b>
        </article>
        <article>
          <span>Активные узлы</span>
          <b>{{ snapshot.metrics.active_nodes }}</b>
        </article>
      </section>

      <section class="panel">
        <div class="panel-head">
          <h2>Кандидаты ответа</h2>
          <span>{{ snapshot.candidates.length }}</span>
        </div>
        <div class="rows">
          <button
            v-for="candidate in snapshot.candidates"
            :key="`${candidate.placement_id}:${candidate.answer}`"
            class="candidate-row"
            type="button"
            @click="openCandidate(candidate.cell_id)"
          >
            <span>
              <b>{{ candidate.answer || 'Без ответа' }}</b>
              <small>{{ candidate.scene_label }}</small>
            </span>
            <strong>{{ percent(candidate.candidate_score) }}</strong>
            <em>{{ candidate.explanation }}</em>
          </button>
          <p v-if="!snapshot.candidates.length" class="empty">Кандидатов нет.</p>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <h2>Текущая картина</h2>
          <span>{{ snapshot.nodes.length }} узлов</span>
        </div>
        <div class="rows">
          <article
            v-for="node in snapshot.nodes"
            :key="node.placement_id"
            class="node-row"
          >
            <span>
              <b>{{ node.label }}</b>
              <small>{{ node.node_type }} · {{ node.eviction_status }}</small>
            </span>
            <strong>{{ percent(node.energy) }}</strong>
          </article>
          <p v-if="!snapshot.nodes.length" class="empty">Узлы отсутствуют.</p>
        </div>
      </section>
    </template>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue';
import { RouterLink, useRouter } from 'vue-router';
import { useHiveAnalytics } from '@/features/hive-analytics';
import type { HiveAnalyticsSnapshotV2 } from '@/entities/hive/types';

const router = useRouter();
const analytics = useHiveAnalytics();
const {
  data,
  loading,
  error,
  hasHive,
  load,
} = analytics;

const primary = computed(() => data.value?.primary || null);
const snapshot = computed<HiveAnalyticsSnapshotV2 | null>(() => {
  const snapshots = primary.value?.snapshots || [];
  return snapshots[snapshots.length - 1]
    || data.value?.current?.snapshot
    || null;
});

function percent(value: number): string {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function number(value: number): string {
  return Number(value || 0).toFixed(2);
}

async function openCandidate(cellId: string | null): Promise<void> {
  await router.push({
    name: 'chat',
    query: cellId ? { cell: cellId } : {},
  });
}

onMounted(() => {
  void load();
});
</script>

<style scoped lang="scss">
.analytics-page {
  min-height: 100vh;
  padding: 24px;
  color: #e8f0fa;
  background: #0d1623;
}

header,
.summary,
.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
}

header {
  max-width: 1180px;
  margin: 0 auto 22px;

  h1 { margin: 4px 0 0; }
  nav { display: flex; gap: 14px; }
  a { color: #8ecbf2; text-decoration: none; }
}

.kicker {
  color: #78e2ca;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .16em;
}

.notice,
.summary,
.metrics,
.panel {
  max-width: 1180px;
  margin-right: auto;
  margin-left: auto;
}

.notice { padding: 28px; color: #94a9c2; }
.notice.error { color: #f2a4b1; }

.summary {
  margin-bottom: 14px;
  padding: 18px;
  border: 1px solid #29425c;
  border-radius: 12px;
  background: #132238;

  span, p { color: #8fa6c1; }
  h2, p { margin: 4px 0; }
  p { font-size: 12px; }
}

.metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(120px, 1fr));
  gap: 10px;
  margin-bottom: 14px;

  article {
    display: grid;
    gap: 6px;
    padding: 15px;
    border: 1px solid #243a52;
    border-radius: 10px;
    background: #111f32;
  }

  span { color: #8da2bc; font-size: 12px; }
  b { color: #7be3ca; font-size: 22px; }
}

.panel {
  margin-bottom: 14px;
  padding: 16px;
  border: 1px solid #253b54;
  border-radius: 12px;
  background: #101d2e;
}

.panel-head {
  h2 { margin: 0; font-size: 17px; }
  span { color: #7f95ae; font-size: 12px; }
}

.rows { display: grid; gap: 8px; margin-top: 12px; }

.candidate-row,
.node-row {
  display: grid;
  grid-template-columns: minmax(180px, 1fr) 64px;
  gap: 7px 14px;
  width: 100%;
  padding: 12px;
  border: 1px solid #28415c;
  border-radius: 9px;
  color: inherit;
  text-align: left;
  background: #13253a;

  span { display: grid; gap: 3px; }
  small, em { color: #8da2bc; font-size: 11px; }
  strong { color: #7be3ca; text-align: right; }
  em { grid-column: 1 / -1; font-style: normal; }
}

.candidate-row { cursor: pointer; }
.candidate-row:hover { border-color: #5ca9d7; }
.empty { color: #8296ad; }

@media (max-width: 720px) {
  .metrics { grid-template-columns: repeat(2, 1fr); }
  header, .summary { align-items: flex-start; flex-direction: column; }
}
</style>
