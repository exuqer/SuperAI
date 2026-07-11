<script setup lang="ts">
import { computed, reactive } from 'vue'
import { RouterLink } from 'vue-router'

import { useRuntimeStore } from '@/shared/model/runtime-store'
import StatusBadge from '@/widgets/app-shell/StatusBadge.vue'

const runtime = useRuntimeStore()

const form = reactive({
  message: '',
  conversationId: '',
  projectId: '',
  timeLimitMs: 4_000,
  eventLimit: 20,
  stepLimit: 32,
  memoryBytes: 16_384,
})

const canCancel = computed(
  () => runtime.task && (runtime.task.status === 'queued' || runtime.task.status === 'running'),
)
const taskBudget = computed(() => runtime.task?.budget)

async function submit() {
  await runtime.runTask({
    schema_version: '1.0',
    tenant_id: 'local',
    user_id: 'user-local',
    message: form.message,
    conversation_id: form.conversationId || 'conversation-local',
    project_id: form.projectId || undefined,
    budget: {
      schema_version: '1.0',
      time_ms: Number(form.timeLimitMs),
      step_limit: Number(form.stepLimit),
      memory_bytes: Number(form.memoryBytes),
      event_limit: Number(form.eventLimit),
    },
  })
}
</script>

<template>
  <div class="page">
    <header class="page-heading">
      <div>
        <p class="eyebrow">Первый вертикальный срез</p>
        <h1>Запуск задачи</h1>
        <p>
          Запрос отправляется в локальный live API. Клиент показывает результат,
          бюджет и переход к полной трассе.
        </p>
      </div>
      <StatusBadge
        status="running"
        label="live API"
      />
    </header>

    <div class="run-layout">
      <section class="surface">
        <header class="surface__header">
          <div>
            <p class="eyebrow">TaskContract request</p>
            <h2>Входные параметры</h2>
          </div>
        </header>
        <div class="surface__body">
          <form class="form-grid" @submit.prevent="submit">
            <div class="field field--full">
              <label for="message">Сообщение пользователя</label>
              <textarea
                id="message"
                v-model="form.message"
                :disabled="runtime.isRunning"
                autocomplete="off"
                placeholder="Опишите задачу"
              />
              <small>Ответ и трасса будут сохранены бэкендом в локальном хранилище.</small>
            </div>
            <div class="field">
              <label for="conversation-id">Conversation ID</label>
              <input
                id="conversation-id"
                v-model="form.conversationId"
                :disabled="runtime.isRunning"
                autocomplete="off"
              />
            </div>
            <div class="field">
              <label for="project-id">Project ID <span class="muted">(необязательно)</span></label>
              <input
                id="project-id"
                v-model="form.projectId"
                :disabled="runtime.isRunning"
                autocomplete="off"
              />
            </div>
            <div class="field">
              <label for="time-limit">Бюджет времени, мс</label>
              <input
                id="time-limit"
                v-model.number="form.timeLimitMs"
                :disabled="runtime.isRunning"
                min="1"
                type="number"
              />
            </div>
            <div class="field">
              <label for="event-limit">Бюджет событий</label>
              <input
                id="event-limit"
                v-model.number="form.eventLimit"
                :disabled="runtime.isRunning"
                min="1"
                type="number"
              />
            </div>
            <div class="field field--full">
              <div class="inline-actions">
                <button class="button" :disabled="runtime.isRunning" type="submit">
                  {{ runtime.isRunning ? 'Выполняется…' : 'Запустить задачу' }}
                </button>
                <button
                  class="button button--danger"
                  :disabled="!canCancel || runtime.isRunning"
                  type="button"
                  @click="runtime.cancelTask"
                >
                  Отменить
                </button>
              </div>
            </div>
          </form>
          <div v-if="runtime.runError" class="state-message state-message--error" role="alert">
            <strong>{{ runtime.runError.code }}</strong> — {{ runtime.runError.message }}
          </div>
        </div>
      </section>

      <aside class="scenario-catalog surface">
        <header class="surface__header">
          <div>
            <p class="eyebrow">Fixture catalog</p>
            <h2>Эталонные сценарии</h2>
          </div>
        </header>
        <div class="scenario-catalog__body">
          <button
            v-for="fixture in runtime.fixtureScenarios"
            :key="fixture.id"
            class="scenario-option"
            :class="{ 'scenario-option--selected': runtime.selectedFixtureId === fixture.id }"
            type="button"
            @click="runtime.selectedFixtureId = fixture.id"
          >
            <strong>{{ fixture.title }}</strong>
            <span>{{ fixture.description }}</span>
          </button>
        </div>
      </aside>
    </div>

    <section v-if="runtime.task" class="surface result-card" aria-live="polite">
      <header class="surface__header">
        <div>
          <p class="eyebrow">Task result</p>
          <h2>{{ runtime.task.id }}</h2>
        </div>
        <StatusBadge :status="runtime.task.status" />
      </header>
      <div class="surface__body result-card__body">
        <dl class="metric-grid">
          <div class="metric">
            <dt>Время</dt>
            <dd>{{ taskBudget?.timeLimitMs ?? '—' }} мс</dd>
          </div>
          <div class="metric">
            <dt>События</dt>
            <dd>{{ taskBudget?.eventLimit ?? '—' }}</dd>
          </div>
          <div class="metric">
            <dt>Conversation</dt>
            <dd class="metric__id">{{ runtime.task.conversationId }}</dd>
          </div>
          <div class="metric">
            <dt>Project</dt>
            <dd class="metric__id">{{ runtime.task.projectId ?? '—' }}</dd>
          </div>
        </dl>

        <article v-if="runtime.task.answer" class="answer">
          <header>
            <h3>Ответ</h3>
            <StatusBadge
              :status="runtime.task.answer.verified ? 'verified' : 'pending'"
              :label="runtime.task.answer.verified ? 'проверен' : 'не проверен'"
            />
          </header>
          <p>{{ runtime.task.answer.text }}</p>
          <h4>Использованные источники</h4>
          <ul>
            <li v-for="source in runtime.task.answer.sources" :key="source.artifactId">
              <RouterLink :to="{ name: 'storage', params: { artifactId: source.artifactId } }">
                {{ source.label }}
              </RouterLink>
              <small>{{ source.accessScope }}</small>
              <span>{{ source.contentHash }}</span>
            </li>
          </ul>
        </article>

        <article v-if="runtime.task.error" class="task-error">
          <h3>{{ runtime.task.error.code }}</h3>
          <p>{{ runtime.task.error.message }}</p>
          <small v-if="runtime.task.error.retryable">Ошибка допускает повторный запуск.</small>
        </article>

        <div class="inline-actions">
          <RouterLink
            v-if="runtime.task.traceId"
            class="button button--secondary"
            :to="{ name: 'traces', params: { traceId: runtime.task.traceId } }"
          >
            Открыть трассу
          </RouterLink>
          <RouterLink
            v-if="runtime.task.hiveId"
            class="button button--secondary"
            :to="{ name: 'hive', params: { hiveId: runtime.task.hiveId } }"
          >
            Открыть Улей
          </RouterLink>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped lang="scss">
.run-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(16rem, 0.45fr);
  gap: 1.25rem;

  @media (max-width: 900px) {
    grid-template-columns: 1fr;
  }
}

.fixture-picker {
  display: grid;
  gap: 0.3rem;
  color: #a9bad4;
  font-size: 0.75rem;

  select {
    max-width: 17rem;
    border: 1px solid rgba(168, 190, 228, 0.23);
    border-radius: 0.45rem;
    color: #eaf2ff;
    background: #101e36;
    padding: 0.35rem 0.45rem;
  }
}

.fixture-description {
  margin: 0 0 1rem;
  color: #b4c3db;
  font-size: 0.9rem;
}

.scenario-catalog__body {
  display: grid;
  gap: 0.3rem;
  padding: 0.55rem;
}

.scenario-option {
  display: grid;
  gap: 0.22rem;
  border: 1px solid transparent;
  border-radius: 0.65rem;
  color: #cbd8eb;
  background: transparent;
  padding: 0.7rem;
  text-align: left;

  &:hover {
    background: rgba(115, 160, 232, 0.09);
  }

  &--selected {
    border-color: rgba(116, 172, 255, 0.32);
    background: rgba(69, 130, 224, 0.16);
  }

  strong {
    font-size: 0.82rem;
  }

  span {
    color: #8fa1bd;
    font-size: 0.74rem;
    line-height: 1.35;
  }
}

.result-card__body {
  display: grid;
  gap: 1.1rem;
}

.metric__id {
  overflow: hidden;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.78rem !important;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.answer,
.task-error {
  border: 1px solid rgba(168, 190, 228, 0.15);
  border-radius: 0.76rem;
  padding: 1rem;
  background: rgba(5, 15, 29, 0.4);

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
  }

  h3,
  h4 {
    margin: 0;
  }

  h4 {
    margin-top: 1rem;
    color: #b9cae4;
    font-size: 0.85rem;
  }

  p {
    color: #e5edf9;
    line-height: 1.55;
  }

  ul {
    display: grid;
    gap: 0.55rem;
    margin: 0.6rem 0 0;
    padding-left: 1.1rem;

    a {
      color: #91c1ff;
    }

    small,
    span {
      display: block;
      margin-top: 0.18rem;
      color: #94a6c1;
      font-size: 0.76rem;
    }
  }
}

.task-error {
  border-color: rgba(251, 128, 145, 0.35);
  background: rgba(129, 38, 53, 0.23);

  h3 {
    color: #ffabb6;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.9rem;
  }
}
</style>
