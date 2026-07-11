<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'

import { useRuntimeStore } from '@/shared/model/runtime-store'
import TraceInspector from '@/widgets/inspectors/TraceInspector.vue'
import StatusBadge from '@/widgets/app-shell/StatusBadge.vue'

const runtime = useRuntimeStore()
const route = useRoute()
const loading = ref(false)
const pageError = ref<string>()
const comparisonTraceId = ref('')

const activeTrace = computed(() => runtime.activeTrace)
const comparisonTrace = computed(() =>
  comparisonTraceId.value ? runtime.traces[comparisonTraceId.value] : undefined,
)

async function openTrace(traceId: string) {
  if (!traceId) {
    return
  }
  loading.value = true
  pageError.value = undefined
  try {
    await runtime.loadTrace(traceId)
  } catch (error) {
    pageError.value = error instanceof Error ? error.message : 'Не удалось загрузить трассу.'
  } finally {
    loading.value = false
  }
}

watch(
  () => route.params.traceId,
  (traceId) => {
    if (typeof traceId === 'string') {
      void openTrace(traceId)
    }
  },
  { immediate: true },
)

onMounted(() => {
  if (!activeTrace.value && runtime.mode === 'mock') {
    void openTrace('trace-success-001')
  }
})
</script>

<template>
  <div class="page">
    <header class="page-heading">
      <div>
        <p class="eyebrow">Observability</p>
        <h1>Трассы выполнения</h1>
        <p>
          Хронология, причинность, бюджеты, артефакты и нормализованные ошибки — без
          необходимости читать серверный лог.
        </p>
      </div>
      <StatusBadge v-if="activeTrace" :status="activeTrace.status" />
    </header>

    <section class="surface trace-controls">
      <div class="surface__body">
        <div class="trace-controls__grid">
          <label class="field">
            <span>Открытая трасса</span>
            <select
              :value="runtime.activeTraceId"
              :disabled="loading"
              @change="openTrace(($event.target as HTMLSelectElement).value)"
            >
              <option value="">Выберите трассу</option>
              <option v-for="trace in runtime.traceList" :key="trace.id" :value="trace.id">
                {{ trace.id }} · {{ trace.status }}
              </option>
              <option v-if="runtime.mode === 'mock' && !runtime.traces['trace-success-001']" value="trace-success-001">
                trace-success-001 · fixture
              </option>
            </select>
          </label>
          <label class="field">
            <span>Сравнить с</span>
            <select v-model="comparisonTraceId">
              <option value="">Не сравнивать</option>
              <option
                v-for="trace in runtime.traceList.filter((candidate) => candidate.id !== activeTrace?.id)"
                :key="trace.id"
                :value="trace.id"
              >
                {{ trace.id }}
              </option>
            </select>
          </label>
        </div>
        <div v-if="pageError" class="state-message state-message--error" role="alert">{{ pageError }}</div>
      </div>
    </section>

    <TraceInspector
      v-if="activeTrace"
      :trace="activeTrace"
      :compare-with="comparisonTrace"
    />
    <section v-else class="state-message">
      <strong>Трасса ещё не выбрана.</strong>
      Запустите fixture на странице «Запуск» или откройте сохранённый trace_id.
    </section>
  </div>
</template>

<style scoped lang="scss">
.trace-controls__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.9rem;

  @media (max-width: 620px) {
    grid-template-columns: 1fr;
  }
}
</style>
