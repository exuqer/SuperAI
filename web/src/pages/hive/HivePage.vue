<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'

import { useRuntimeStore } from '@/shared/model/runtime-store'
import StatusBadge from '@/widgets/app-shell/StatusBadge.vue'

const runtime = useRuntimeStore()
const route = useRoute()
const loading = ref(false)
const pageError = ref<string>()

const hive = computed(() => runtime.hive)

function formatBytes(value: number) {
  if (value < 1024) {
    return value + ' B'
  }
  return (value / 1024).toFixed(1) + ' KiB'
}

async function openHive(hiveId: string) {
  if (!hiveId) {
    return
  }
  loading.value = true
  pageError.value = undefined
  try {
    await runtime.loadHive(hiveId)
  } catch (error) {
    pageError.value = error instanceof Error ? error.message : 'Не удалось загрузить Улей.'
  } finally {
    loading.value = false
  }
}

watch(
  () => route.params.hiveId,
  (hiveId) => {
    if (typeof hiveId === 'string') {
      void openHive(hiveId)
    }
  },
  { immediate: true },
)

onMounted(() => {
  if (!hive.value && runtime.mode === 'mock') {
    void openHive('hive-unity-001')
  }
})
</script>

<template>
  <div class="page">
    <header class="page-heading">
      <div>
        <p class="eyebrow">Рабочее пространство</p>
        <h1>Улей</h1>
        <p>
          Улей — контекст конкретной задачи, а не предметная база. Здесь видны цель,
          защищённые ограничения, магазины и объяснимое вытеснение.
        </p>
      </div>
      <StatusBadge v-if="hive" :status="hive.state" />
    </header>

    <section class="surface">
      <div class="surface__body hive-picker">
        <label class="field">
          <span>Hive ID</span>
          <input
            :value="hive?.id ?? 'hive-unity-001'"
            :disabled="loading"
            autocomplete="off"
            @change="openHive(($event.target as HTMLInputElement).value)"
          />
        </label>
        <p class="muted">В MVP это read model; команда restore появится после публичного command endpoint.</p>
      </div>
    </section>

    <div v-if="pageError" class="state-message state-message--error" role="alert">{{ pageError }}</div>

    <template v-if="hive">
      <div class="split-grid">
        <section class="surface">
          <header class="surface__header">
            <div>
              <p class="eyebrow">TaskContract rev. {{ hive.contract.revision }}</p>
              <h2>Контракт и цели</h2>
            </div>
            <span class="hive-id">{{ hive.id }}</span>
          </header>
          <div class="surface__body contract">
            <div>
              <h3>Цель</h3>
              <p>{{ hive.contract.goal }}</p>
            </div>
            <div>
              <h3>Активные цели</h3>
              <ul>
                <li v-for="goal in hive.goals" :key="goal">{{ goal }}</li>
              </ul>
            </div>
            <div>
              <h3>Ограничения</h3>
              <ul>
                <li v-for="constraint in hive.contract.constraints" :key="constraint">{{ constraint }}</li>
              </ul>
            </div>
            <div>
              <h3>Защищённые ссылки</h3>
              <code v-for="reference in hive.contract.protectedContextRefs" :key="reference">
                {{ reference }}
              </code>
            </div>
          </div>
        </section>

        <section class="surface">
          <header class="surface__header">
            <div>
              <p class="eyebrow">CapacityController</p>
              <h2>Горячая память</h2>
            </div>
            <strong>{{ hive.hotMemory.utilization }}%</strong>
          </header>
          <div class="surface__body memory">
            <div class="memory__bar" aria-label="Заполнение горячей памяти">
              <span :style="{ width: hive.hotMemory.utilization + '%' }" />
            </div>
            <dl class="metric-grid">
              <div class="metric">
                <dt>Использовано</dt>
                <dd>{{ formatBytes(hive.hotMemory.usedBytes) }}</dd>
              </div>
              <div class="metric">
                <dt>Лимит</dt>
                <dd>{{ formatBytes(hive.hotMemory.limitBytes) }}</dd>
              </div>
              <div class="metric">
                <dt>Активных элементов</dt>
                <dd>{{ hive.hotMemory.activeItems }}</dd>
              </div>
              <div class="metric">
                <dt>Task</dt>
                <dd class="hive-id">{{ hive.taskId ?? '—' }}</dd>
              </div>
            </dl>
          </div>
        </section>
      </div>

      <section class="surface">
        <header class="surface__header">
          <div>
            <p class="eyebrow">Typed stores</p>
            <h2>Локальные магазины</h2>
          </div>
        </header>
        <div class="surface__body data-table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Магазин</th>
                <th>Элементов</th>
                <th>Размер</th>
                <th>Защищено</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="store in hive.stores" :key="store.storeId">
                <td><strong>{{ store.label }}</strong><small class="table-subvalue">{{ store.storeId }}</small></td>
                <td>{{ store.itemCount }}</td>
                <td>{{ formatBytes(store.sizeBytes) }}</td>
                <td>{{ store.protectedCount }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <div class="split-grid">
        <section class="surface">
          <header class="surface__header">
            <div>
              <p class="eyebrow">Eviction buffer</p>
              <h2>Недавно вытеснено</h2>
            </div>
          </header>
          <div class="surface__body">
            <div v-if="hive.evictedItems.length" class="evictions">
              <article v-for="item in hive.evictedItems" :key="item.entryId">
                <StatusBadge status="queued" :label="item.destination" />
                <strong>{{ item.summary }}</strong>
                <span>{{ item.reasonCode }}</span>
                <small>{{ item.occurredAt }}</small>
              </article>
            </div>
            <p v-else class="muted">Вытесненных элементов нет.</p>
          </div>
        </section>

        <section class="surface">
          <header class="surface__header">
            <div>
              <p class="eyebrow">Snapshot</p>
              <h2>Точки восстановления</h2>
            </div>
          </header>
          <div class="surface__body">
            <div v-if="hive.snapshots.length" class="snapshots">
              <article v-for="snapshot in hive.snapshots" :key="snapshot.snapshotId">
                <div>
                  <strong>{{ snapshot.snapshotId }}</strong>
                  <span>ссылка на согласованный снимок агрегата</span>
                </div>
                <StatusBadge
                  :status="snapshot.restorable ? 'verified' : 'pending'"
                  :label="snapshot.restorable ? 'восстанавливаемый' : 'недоступен'"
                />
              </article>
            </div>
            <p v-else class="muted">Снимки ещё не созданы.</p>
          </div>
        </section>
      </div>
    </template>
    <section v-else class="state-message">
      Откройте hive_id из результата задачи, чтобы увидеть его диагностическое состояние.
    </section>
  </div>
</template>

<style scoped lang="scss">
.hive-picker {
  display: flex;
  flex-wrap: wrap;
  align-items: end;
  gap: 1rem;

  .field {
    width: min(100%, 28rem);
  }

  p {
    margin: 0;
    font-size: 0.82rem;
  }
}

.hive-id {
  overflow: hidden;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.78rem;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.contract {
  display: grid;
  gap: 1rem;

  h3 {
    margin: 0 0 0.4rem;
    color: #b6c8e4;
    font-size: 0.83rem;
  }

  p {
    margin: 0;
    line-height: 1.5;
  }

  ul {
    display: grid;
    gap: 0.28rem;
    margin: 0;
    padding-left: 1.1rem;
    color: #d6e0f1;
    font-size: 0.86rem;
  }

  code {
    display: block;
    overflow: hidden;
    color: #a9c9f8;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.74rem;
    text-overflow: ellipsis;
  }
}

.memory {
  display: grid;
  gap: 1rem;
}

.memory__bar {
  overflow: hidden;
  height: 0.72rem;
  border-radius: 999px;
  background: #091527;

  span {
    display: block;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #50caa4, #79a6ff);
  }
}

.evictions,
.snapshots {
  display: grid;
  gap: 0.65rem;

  article {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    align-items: center;
    gap: 0.45rem 0.7rem;
    padding: 0.65rem;
    border: 1px solid rgba(168, 190, 228, 0.13);
    border-radius: 0.6rem;
    background: rgba(6, 16, 31, 0.3);

    strong,
    span,
    small {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    strong {
      font-size: 0.83rem;
    }

    span,
    small {
      grid-column: 2;
      color: #96a8c3;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.7rem;
    }
  }
}

.snapshots article {
  grid-template-columns: minmax(0, 1fr) auto;

  div {
    display: grid;
    gap: 0.25rem;
    min-width: 0;
  }

  span {
    grid-column: auto;
  }
}

.table-subvalue {
  display: block;
  margin-top: 0.18rem;
  color: #8e9fba;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.68rem;
}
</style>
