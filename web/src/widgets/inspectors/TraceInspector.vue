<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'

import type { UiTrace, UiTraceSpan } from '@/shared/contracts/ui-models'
import JsonViewer from './JsonViewer.vue'
import StatusBadge from '@/widgets/app-shell/StatusBadge.vue'

const props = defineProps<{
  trace: UiTrace
  compareWith?: UiTrace
}>()

const componentFilter = ref('')
const kindFilter = ref('')
const statusFilter = ref('')
const criticalOnly = ref(false)
const selectedSpanId = ref<string>()

watch(
  () => props.trace.id,
  () => {
    selectedSpanId.value = props.trace.spans[0]?.id
  },
  { immediate: true },
)

const components = computed(() =>
  [...new Set(props.trace.spans.map((span) => span.component))].sort(),
)

const criticalPathIds = computed(() => {
  const byId = new Map(props.trace.spans.map((span) => [span.id, span]))
  let cursor = props.trace.spans.reduce<UiTraceSpan | undefined>(
    (largest, span) =>
      (span.durationMs ?? 0) > (largest?.durationMs ?? 0) ? span : largest,
    undefined,
  )
  const ids = new Set<string>()
  while (cursor) {
    ids.add(cursor.id)
    cursor = cursor.parentId ? byId.get(cursor.parentId) : undefined
  }
  return ids
})

const filteredSpans = computed(() =>
  props.trace.spans.filter((span) => {
    if (componentFilter.value && span.component !== componentFilter.value) {
      return false
    }
    if (kindFilter.value && span.kind !== kindFilter.value) {
      return false
    }
    if (statusFilter.value && span.status !== statusFilter.value) {
      return false
    }
    if (criticalOnly.value && !criticalPathIds.value.has(span.id)) {
      return false
    }
    return true
  }),
)

const selectedSpan = computed(() =>
  props.trace.spans.find((span) => span.id === selectedSpanId.value) ?? filteredSpans.value[0],
)

const totalDuration = computed(() =>
  props.trace.spans.reduce((total, span) => total + (span.durationMs ?? 0), 0),
)

const comparison = computed(() => {
  if (!props.compareWith) {
    return undefined
  }
  const baselineDuration = props.compareWith.spans.reduce(
    (total, span) => total + (span.durationMs ?? 0),
    0,
  )
  return {
    spanDelta: props.trace.spans.length - props.compareWith.spans.length,
    durationDelta: totalDuration.value - baselineDuration,
  }
})

function formatTime(iso: string) {
  return new Intl.DateTimeFormat('ru-RU', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    timeZone: 'UTC',
  }).format(new Date(iso))
}

function formatDuration(value?: number) {
  return value === undefined ? '—' : value + ' мс'
}
</script>

<template>
  <section class="trace-inspector surface" aria-label="Инспектор трассы">
    <header class="surface__header trace-inspector__header">
      <div>
        <p class="eyebrow">Trace inspector</p>
        <h2>{{ trace.id }}</h2>
      </div>
      <div class="inline-actions">
        <StatusBadge :status="trace.status" />
        <span class="muted">{{ trace.spans.length }} spans · {{ totalDuration }} мс</span>
      </div>
    </header>

    <div class="surface__body trace-inspector__body">
      <div class="filters" aria-label="Фильтры трассы">
        <label>
          <span>Компонент</span>
          <select v-model="componentFilter">
            <option value="">Все</option>
            <option v-for="component in components" :key="component" :value="component">
              {{ component }}
            </option>
          </select>
        </label>
        <label>
          <span>Тип</span>
          <select v-model="kindFilter">
            <option value="">Все</option>
            <option value="command">command</option>
            <option value="event">event</option>
            <option value="query">query</option>
            <option value="critic">critic</option>
            <option value="codec">codec</option>
          </select>
        </label>
        <label>
          <span>Статус</span>
          <select v-model="statusFilter">
            <option value="">Все</option>
            <option value="succeeded">успешно</option>
            <option value="failed">ошибка</option>
            <option value="cancelled">отменено</option>
            <option value="skipped">пропущено</option>
          </select>
        </label>
        <label class="checkbox-field">
          <input v-model="criticalOnly" type="checkbox" />
          <span>Только критический путь</span>
        </label>
      </div>

      <div v-if="comparison" class="comparison">
        <strong>Сравнение с {{ compareWith?.id }}</strong>
        <span>spans: {{ comparison.spanDelta >= 0 ? '+' : '' }}{{ comparison.spanDelta }}</span>
        <span>длительность: {{ comparison.durationDelta >= 0 ? '+' : '' }}{{ comparison.durationDelta }} мс</span>
      </div>

      <div class="timeline" aria-label="Хронология span">
        <button
          v-for="span in filteredSpans"
          :key="span.id"
          class="timeline__item"
          :class="{
            'timeline__item--selected': selectedSpan?.id === span.id,
            'timeline__item--critical': criticalPathIds.has(span.id),
          }"
          type="button"
          @click="selectedSpanId = span.id"
        >
          <span class="timeline__marker" :class="'timeline__marker--' + span.status" />
          <span class="timeline__time">{{ formatTime(span.startedAt) }}</span>
          <span class="timeline__operation">{{ span.component }} · {{ span.operation }}</span>
          <span class="timeline__duration">{{ formatDuration(span.durationMs) }}</span>
        </button>
      </div>

      <div class="data-table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Span / причина</th>
              <th>Компонент</th>
              <th>Тип</th>
              <th>Статус</th>
              <th>Время</th>
              <th>Бюджет после</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="span in filteredSpans"
              :key="span.id"
              :class="{ 'table-row--selected': selectedSpan?.id === span.id }"
              @click="selectedSpanId = span.id"
            >
              <td>
                <strong>{{ span.id }}</strong>
                <small v-if="span.parentId" class="table-subvalue">parent: {{ span.parentId }}</small>
                <small v-if="span.causationId" class="table-subvalue">cause: {{ span.causationId }}</small>
              </td>
              <td>{{ span.component }}<small class="table-subvalue">{{ span.operation }}</small></td>
              <td>{{ span.kind }}</td>
              <td><StatusBadge :status="span.status" /></td>
              <td>{{ formatDuration(span.durationMs) }}</td>
              <td>
                <template v-if="span.budgetAfter">
                  limit {{ span.budgetAfter.timeLimitMs }} мс
                  <small class="table-subvalue">
                    events ≤ {{ span.budgetAfter.eventLimit }}
                  </small>
                </template>
                <template v-else>—</template>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div v-if="selectedSpan" class="split-grid trace-details">
        <article class="detail-block">
          <h3>Вход</h3>
          <JsonViewer :value="selectedSpan.input" empty-label="Нет inline input; см. artifact references." />
        </article>
        <article class="detail-block">
          <h3>Выход</h3>
          <JsonViewer :value="selectedSpan.output" empty-label="Нет inline output." />
        </article>
        <article v-if="selectedSpan.error" class="detail-block detail-block--error">
          <h3>Нормализованная ошибка</h3>
          <p><strong>{{ selectedSpan.error.code }}</strong> — {{ selectedSpan.error.message }}</p>
        </article>
        <article v-if="selectedSpan.artifacts.length" class="detail-block">
          <h3>Артефакты</h3>
          <ul class="artifact-links">
            <li v-for="artifact in selectedSpan.artifacts" :key="artifact.id">
              <RouterLink :to="{ name: 'storage', params: { artifactId: artifact.id } }">
                {{ artifact.label }}
              </RouterLink>
              <small>{{ artifact.mediaType }}</small>
            </li>
          </ul>
        </article>
      </div>

      <section v-if="trace.events.length" class="trace-events">
        <h3>Доменные события</h3>
        <div class="trace-events__list">
          <article v-for="event in trace.events" :key="event.id">
            <strong>#{{ event.sequence ?? '—' }} {{ event.kind }}</strong>
            <span>{{ event.producer }}</span>
            <small v-if="event.causationId">cause: {{ event.causationId }}</small>
          </article>
        </div>
      </section>
    </div>
  </section>
</template>

<style scoped lang="scss">
.trace-inspector__header h2 {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.95rem;
}

.trace-inspector__body {
  display: grid;
  gap: 1rem;
}

.filters {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;

  label {
    display: grid;
    gap: 0.28rem;
    color: #9eafca;
    font-size: 0.74rem;
  }

  select {
    min-height: 2.2rem;
    border: 1px solid rgba(168, 190, 228, 0.24);
    border-radius: 0.5rem;
    color: #eaf1fc;
    background: #0a1527;
    padding: 0.35rem 0.45rem;
  }
}

.checkbox-field {
  display: flex !important;
  align-items: center;
  gap: 0.45rem !important;
  margin-top: 1.22rem;
  color: #d2def0 !important;
}

.comparison {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  padding: 0.65rem 0.75rem;
  border: 1px solid rgba(125, 174, 247, 0.25);
  border-radius: 0.65rem;
  color: #bed9ff;
  background: rgba(59, 118, 205, 0.12);
  font-size: 0.8rem;
}

.timeline {
  display: grid;
  gap: 0.3rem;
  padding-left: 0.75rem;
  border-left: 1px solid rgba(157, 183, 226, 0.24);
}

.timeline__item {
  display: grid;
  grid-template-columns: 0.75rem 5.7rem minmax(0, 1fr) auto;
  align-items: center;
  gap: 0.5rem;
  border: 0;
  border-radius: 0.55rem;
  color: #d4e0f5;
  background: transparent;
  padding: 0.45rem 0.55rem;
  text-align: left;

  &:hover,
  &--selected {
    background: rgba(94, 145, 228, 0.12);
  }

  &--critical .timeline__operation {
    color: #83bcff;
  }
}

.timeline__marker {
  width: 0.58rem;
  height: 0.58rem;
  border-radius: 50%;
  background: #8398b9;

  &--succeeded {
    background: #65d3a9;
  }

  &--failed,
  &--cancelled {
    background: #f27d8b;
  }
}

.timeline__time,
.timeline__duration {
  color: #8fa2c0;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.72rem;
}

.timeline__operation {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 0.82rem;
}

.table-row--selected {
  background: rgba(92, 143, 224, 0.12);
}

.data-table tbody tr {
  cursor: pointer;
}

.table-subvalue {
  display: block;
  margin-top: 0.18rem;
  color: #8e9fba;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.68rem;
}

.trace-details {
  margin-top: 0.3rem;
}

.detail-block {
  padding: 0.85rem;
  border: 1px solid rgba(168, 190, 228, 0.15);
  border-radius: 0.68rem;
  background: rgba(6, 16, 31, 0.35);

  h3 {
    margin: 0 0 0.65rem;
    color: #d8e4f8;
    font-size: 0.85rem;
  }

  p {
    margin: 0.35rem 0;
    color: #d4dff2;
    font-size: 0.83rem;
  }

  &--error {
    border-color: rgba(247, 126, 140, 0.35);
    background: rgba(133, 40, 54, 0.2);
  }
}

.artifact-links {
  display: grid;
  gap: 0.55rem;
  margin: 0;
  padding-left: 1.1rem;

  a {
    color: #91c0ff;
  }

  small {
    display: block;
    margin-top: 0.13rem;
    color: #8e9fbb;
  }
}

.trace-events {
  padding: 0.85rem;
  border: 1px solid rgba(168, 190, 228, 0.15);
  border-radius: 0.68rem;
  background: rgba(6, 16, 31, 0.25);

  h3 {
    margin: 0 0 0.65rem;
    color: #d8e4f8;
    font-size: 0.85rem;
  }
}

.trace-events__list {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.55rem;

  @media (max-width: 760px) {
    grid-template-columns: 1fr;
  }

  article {
    display: grid;
    gap: 0.22rem;
    padding: 0.6rem;
    border: 1px solid rgba(168, 190, 228, 0.12);
    border-radius: 0.56rem;
  }

  strong {
    color: #c9dbf6;
    font-size: 0.77rem;
  }

  span,
  small {
    color: #91a4c1;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.68rem;
  }
}

@media (max-width: 620px) {
  .timeline__item {
    grid-template-columns: 0.75rem minmax(0, 1fr) auto;
  }

  .timeline__time {
    display: none;
  }
}
</style>
