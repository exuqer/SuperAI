<script setup lang="ts">
import { onMounted } from 'vue'

import { useRuntimeStore } from '@/shared/model/runtime-store'
import StatusBadge from '@/widgets/app-shell/StatusBadge.vue'

const runtime = useRuntimeStore()

onMounted(() => {
  void runtime.loadSystem()
})
</script>

<template>
  <div class="page">
    <header class="page-heading">
      <div>
        <p class="eyebrow">Runtime diagnostics</p>
        <h1>Система</h1>
        <p>
          Health, readiness, очередь и ошибки представлены как read model. Это не замена
          серверной observability, а её безопасный диагностический срез.
        </p>
      </div>
      <button class="button button--secondary" :disabled="runtime.isBootstrapping" type="button" @click="runtime.loadSystem">
        Обновить
      </button>
    </header>

    <div v-if="runtime.systemError" class="state-message state-message--error" role="alert">
      <strong>{{ runtime.systemError.code }}</strong> — {{ runtime.systemError.message }}
    </div>

    <template v-if="runtime.system">
      <section class="surface">
        <header class="surface__header">
          <div>
            <p class="eyebrow">Health / readiness</p>
            <h2>Состояние компонентов</h2>
          </div>
          <StatusBadge :status="runtime.system.health.status" />
        </header>
        <div class="surface__body">
          <dl class="metric-grid">
            <div class="metric">
              <dt>Активные задачи</dt>
              <dd>{{ runtime.system.activeTasks }}</dd>
            </div>
            <div class="metric">
              <dt>Work items в очереди</dt>
              <dd>{{ runtime.system.queuedWorkItems }}</dd>
            </div>
            <div class="metric">
              <dt>Dead letters</dt>
              <dd>{{ runtime.system.deadLetters }}</dd>
            </div>
            <div class="metric">
              <dt>Runtime</dt>
              <dd class="timestamp">{{ runtime.system.meta.build }}</dd>
            </div>
          </dl>
          <div class="dependencies">
            <article v-for="dependency in runtime.system.health.dependencies" :key="dependency.name">
              <StatusBadge :status="dependency.status" />
              <div>
                <strong>{{ dependency.name }}</strong>
                <span v-if="dependency.detail">{{ dependency.detail }}</span>
              </div>
            </article>
          </div>
        </div>
      </section>

      <div class="split-grid">
        <section class="surface">
          <header class="surface__header">
            <div>
              <p class="eyebrow">API metadata</p>
              <h2>Версии и возможности</h2>
            </div>
          </header>
          <div class="surface__body">
            <dl class="metadata">
              <div><dt>API</dt><dd>{{ runtime.system.meta.apiVersion }}</dd></div>
              <div><dt>Backend</dt><dd>{{ runtime.system.meta.backendVersion }}</dd></div>
              <div><dt>Schema</dt><dd>{{ runtime.system.meta.schemaVersion }}</dd></div>
              <div><dt>Build</dt><dd>{{ runtime.system.meta.build }}</dd></div>
            </dl>
            <div class="capabilities">
              <span v-for="capability in runtime.system.meta.capabilities" :key="capability">
                {{ capability }}
              </span>
            </div>
          </div>
        </section>

        <section class="surface">
          <header class="surface__header">
            <div>
              <p class="eyebrow">Last errors</p>
              <h2>Последние ошибки</h2>
            </div>
          </header>
          <div class="surface__body">
            <div v-if="runtime.system.lastErrors.length" class="errors">
              <article v-for="error in runtime.system.lastErrors" :key="error.code + error.message">
                <strong>{{ error.code }}</strong>
                <span>{{ error.message }}</span>
              </article>
            </div>
            <p v-else class="muted">Нормализованных ошибок в этом срезе нет.</p>
          </div>
        </section>
      </div>
    </template>
    <section v-else class="state-message">
      Загрузка системной сводки…
    </section>
  </div>
</template>

<style scoped lang="scss">
.timestamp {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.76rem !important;
}

.dependencies {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.7rem;
  margin-top: 1rem;

  @media (max-width: 800px) {
    grid-template-columns: 1fr;
  }

  article {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    padding: 0.72rem;
    border: 1px solid rgba(168, 190, 228, 0.12);
    border-radius: 0.64rem;
    background: rgba(5, 14, 29, 0.35);
  }

  div {
    display: grid;
    gap: 0.18rem;
  }

  strong {
    font-size: 0.82rem;
  }

  span {
    color: #94a6c1;
    font-size: 0.75rem;
  }
}

.metadata {
  display: grid;
  gap: 0.62rem;
  margin: 0;

  div {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    border-bottom: 1px solid rgba(168, 190, 228, 0.1);
    padding-bottom: 0.52rem;
  }

  dt {
    color: #91a5c2;
    font-size: 0.76rem;
  }

  dd {
    margin: 0;
    color: #dce7f8;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.78rem;
    text-align: right;
  }
}

.capabilities {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 1rem;

  span {
    border-radius: 999px;
    color: #b8d4ff;
    background: rgba(72, 132, 224, 0.16);
    padding: 0.25rem 0.45rem;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.7rem;
  }
}

.errors {
  display: grid;
  gap: 0.6rem;

  article {
    display: grid;
    gap: 0.25rem;
    border: 1px solid rgba(248, 131, 145, 0.27);
    border-radius: 0.62rem;
    padding: 0.72rem;
    background: rgba(123, 37, 51, 0.18);
  }

  strong {
    color: #ffb0b9;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.78rem;
  }

  span {
    color: #e8c9cf;
    font-size: 0.8rem;
  }
}
</style>
