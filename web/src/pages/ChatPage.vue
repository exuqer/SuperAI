<template>
  <div class="graph-page">
    <header class="topbar">
      <RouterLink class="brand" :to="{ name: 'chat' }">
        <span class="brand-mark">S</span>
        <span>
          <b>SuperAI</b>
          <small>Event Graph · V2.7</small>
        </span>
      </RouterLink>
      <nav aria-label="Основная навигация">
        <RouterLink :to="{ name: 'chat' }">Диалог</RouterLink>
        <RouterLink :to="{ name: 'universe' }">Пространство</RouterLink>
      </nav>
      <div class="session">
        <span class="live-dot" />
        <span>{{ hive ? compactId(hive.id) : 'создание…' }}</span>
        <button class="ghost-button" :disabled="store.loading" @click="newDialogue">
          Новый диалог
        </button>
        <button class="ghost-button danger" :disabled="!hive || store.loading" @click="clearChatSpace">
          Очистить пространство
        </button>
        <button class="ghost-button" :disabled="!hive || exportingSpace" @click="exportChatSpace">
          {{ exportingSpace ? 'Копирование…' : exportStatus || 'Копировать пространство' }}
        </button>
      </div>
    </header>

    <main class="workspace">
      <section class="panel chat-panel">
        <div class="panel-head">
          <div>
            <div class="kicker">ДИАЛОГ</div>
            <h1>Запрос к памяти</h1>
          </div>
          <span class="turn-counter">{{ state?.turn_index || 0 }} ходов</span>
        </div>

        <div ref="messageList" class="messages" aria-live="polite">
          <div v-if="!store.messages.length" class="welcome">
            <span class="welcome-icon">⌁</span>
            <h2>Задайте вопрос</h2>
            <p>
              Запрос будет преобразован в граф с неизвестным узлом GAP.
              Ответ выбирается только из структурно совместимых событий.
            </p>
          </div>
          <article
            v-for="message in store.messages"
            :key="message.id"
            class="message"
            :class="message.role"
          >
            <span class="message-author">
              {{ message.role === 'user' ? 'Вы' : 'SuperAI' }}
            </span>
            <div class="bubble">{{ message.text }}</div>
            <span v-if="message.role === 'assistant' && message.status" class="message-status">
              {{ answerStatusLabel(message.status) }}
            </span>
          </article>
          <article v-if="store.loading" class="message assistant pending">
            <span class="message-author">SuperAI</span>
            <div class="bubble typing"><i /><i /><i /></div>
          </article>
        </div>

        <form class="composer" @submit.prevent="submit">
          <label class="sr-only" for="question">Вопрос к памяти</label>
          <textarea
            id="question"
            v-model="question"
            rows="3"
            placeholder="Например: Кто настроил датчик?"
            :disabled="store.loading || store.restoring"
            @keydown.enter.exact.prevent="submit"
          />
          <div class="composer-footer">
            <label class="mode-select">
              <span>Режим</span>
              <select v-model="mode" :disabled="store.loading">
                <option value="">Авто</option>
                <option value="NEW_QUERY">Новый вопрос</option>
                <option value="FOLLOW_UP">Продолжение</option>
                <option value="CORRECTION">Исправление</option>
              </select>
            </label>
            <button class="primary-button" type="submit" :disabled="!canSubmit">
              <span>{{ store.loading ? 'Обработка…' : 'Отправить' }}</span>
              <span aria-hidden="true">↗</span>
            </button>
          </div>
          <p v-if="store.error" class="error">{{ store.error }}</p>
        </form>
      </section>

      <section class="panel graph-panel">
        <div class="panel-head">
          <div>
            <div class="kicker">QUERY GRAPH</div>
            <h2>Структура текущего вопроса</h2>
          </div>
          <span v-if="queryGraph" class="status-chip" :class="queryGraph.status.toLowerCase()">
            {{ graphStatusLabel(queryGraph.status) }}
          </span>
          <span v-else class="status-chip idle">Ожидание</span>
        </div>

        <div v-if="!queryGraph" class="graph-empty">
          <div class="empty-orbit">
            <span />
            <span />
            <b>GAP</b>
          </div>
          <h3>Граф появится после вопроса</h3>
          <p>Узлы отражают предикат, известные упоминания и неизвестное значение.</p>
        </div>

        <div v-else class="graph-content">
          <div class="graph-meta">
            <span title="Идентификатор графа">{{ compactId(queryGraph.query_graph_id) }}</span>
            <span v-if="queryGraph.continuation_of">
              продолжает {{ compactId(queryGraph.continuation_of) }}
            </span>
            <span>{{ queryGraph.construction_ids.length }} конструкций</span>
          </div>

          <div class="event-graph" aria-label="Граф текущего вопроса">
            <div class="known-row">
              <article
                v-for="node in queryGraph.event_pattern.known_nodes"
                :key="node.node_id"
                class="graph-node known-node"
              >
                <small>MENTION</small>
                <strong>{{ node.surface }}</strong>
                <span>{{ node.head.lemma }}</span>
                <div v-if="node.components.length" class="node-components">
                  <i v-for="component in node.components" :key="component.component_id">
                    + {{ component.surface }}
                  </i>
                </div>
              </article>
              <article v-if="!queryGraph.event_pattern.known_nodes.length" class="graph-node muted-node">
                <small>MENTION</small>
                <strong>Нет известных узлов</strong>
              </article>
            </div>

            <div class="connector"><span /></div>

            <article class="graph-node predicate-node">
              <small>EVENT · PREDICATE</small>
              <strong>
                {{ queryGraph.event_pattern.predicate?.surface || 'Предикат не определён' }}
              </strong>
              <span>{{ queryGraph.event_pattern.predicate?.lemma || '—' }}</span>
            </article>

            <div class="connector gap-connector"><span /></div>

            <article class="graph-node gap-node">
              <small>GAP · {{ gapKindLabel(queryGraph.event_pattern.gap_node.gap_kind) }}</small>
              <strong>{{ queryGraph.event_pattern.gap_node.surface || 'неизвестное значение' }}</strong>
              <span>
                {{ signatureCount(queryGraph.event_pattern.gap_node.question_signature) }}
                наблюдаемых признаков
              </span>
            </article>
          </div>

          <section class="signature-panel">
            <div class="subhead">
              <h3>Сигнатура GAP</h3>
              <span>Наблюдения, не именованные роли</span>
            </div>
            <div class="signature-list">
              <span
                v-for="([key, value]) in topEntries(queryGraph.event_pattern.gap_node.question_signature)"
                :key="key"
                class="signature"
              >
                {{ readableSignature(key) }}
                <b>{{ percent(value) }}</b>
              </span>
              <span
                v-if="!Object.keys(queryGraph.event_pattern.gap_node.question_signature).length"
                class="empty-inline"
              >
                Сигнатура пуста
              </span>
            </div>
          </section>

          <section v-if="targetGaps.length > 1" class="multi-gap-panel">
            <div class="subhead">
              <h3>Совместное заполнение GAP</h3>
              <span>{{ targetGaps.length }} обязательных значения</span>
            </div>
            <div class="gap-map">
              <article v-for="gap in targetGaps" :key="gap.node_id" class="gap-map-item">
                <span>{{ gap.surface }}</span>
                <i>→</i>
                <b>{{ bindingForGap(gap.node_id)?.resolved_surface || 'ожидание' }}</b>
              </article>
            </div>
            <p v-if="bindingConfiguration" class="configuration-state" :class="bindingConfiguration.status.toLowerCase()">
              {{ bindingConfiguration.all_required_gaps_bound ? '✓ Одно событие и уникальные участники подтверждены' : '× Конфигурация не прошла строгую проверку' }}
            </p>
          </section>
        </div>
      </section>

      <aside class="panel inspector-panel">
        <div class="panel-head">
          <div>
            <div class="kicker">BINDINGS</div>
            <h2>Кандидаты заполнения</h2>
          </div>
          <span class="count-badge">{{ candidateBindings.length }}</span>
        </div>

        <div class="inspector-scroll">
          <section v-if="answer" class="answer-card" :class="answer.status.toLowerCase()">
            <div class="answer-head">
              <span>Итог</span>
              <b>{{ answerStatusLabel(answer.status) }}</b>
            </div>
            <strong class="answer-surface">{{ answer.surface || answer.short_answer || '—' }}</strong>
            <p v-if="answer.full_answer && answer.full_answer !== answer.surface">
              {{ answer.full_answer }}
            </p>
            <div class="validation">
              <span :class="{ valid: answer.validation.valid }">
                {{ answer.validation.valid ? '✓ Проверка пройдена' : '× Проверка не пройдена' }}
              </span>
              <small v-if="answer.provenance">
                {{ answer.provenance.independent_source_count }} независимых источников
              </small>
            </div>
          </section>

          <section v-if="swarmRuns.length" class="swarm-card">
            <div class="answer-head">
              <span>Поиск роем</span>
              <b :class="retrievalModeClass">{{ retrievalModeLabel(retrievalMode) }}</b>
            </div>
            <p v-if="swarmFallbackReason" class="swarm-note">{{ fallbackLabel(swarmFallbackReason) }}</p>
            <div class="swarm-runs">
              <article v-for="(run, index) in swarmRuns" :key="run.id" class="swarm-run">
                <span class="swarm-index">GAP {{ index + 1 }}</span>
                <b>{{ gapLabel(run.gap_id) }}</b>
                <small>{{ run.events_returned }} / {{ run.events_considered }} событий · {{ run.termination_reason }}</small>
                <div class="mission-list">
                  <div v-for="mission in run.missions" :key="mission.bee_id" class="mission-row">
                    <strong>{{ mission.bee_type }}</strong>
                    <small>{{ mission.mission_type }}</small>
                    <div class="universe-route">
                      <span v-for="universe in mission.visited_universes" :key="universe">{{ universe }}</span>
                    </div>
                  </div>
                </div>
                <div v-if="run.nectar_packets?.length" class="packet-list">
                  <small v-for="packet in run.nectar_packets" :key="packet.packet_id">
                    Nectar {{ packet.source_universe }} → {{ packet.target_universe }} · {{ packet.event_ids.length }} событий
                  </small>
                </div>
              </article>
            </div>
          </section>

          <div v-if="!candidateBindings.length" class="empty-candidates">
            <span>◇</span>
            <p>{{ queryGraph ? 'Допущенных связываний нет' : 'Здесь появятся допущенные значения' }}</p>
          </div>

          <section v-else class="candidate-list">
            <article
              v-for="(binding, index) in candidateBindings"
              :key="binding.binding_id"
              class="candidate"
              :class="{ selected: binding.status === 'SELECTED' }"
            >
              <div class="candidate-head">
                <span class="rank">#{{ index + 1 }}</span>
                <div>
                  <strong>{{ binding.resolved_surface }}</strong>
                  <small>{{ binding.resolved_lemma }}</small>
                </div>
                <b>{{ percent(binding.scores.total) }}</b>
              </div>
              <div class="score-bar">
                <i :style="{ width: `${binding.scores.total * 100}%` }" />
              </div>
              <dl class="scores">
                <div><dt>структура</dt><dd>{{ percent(binding.scores.structural) }}</dd></div>
                <div><dt>сигнатура</dt><dd>{{ percent(binding.scores.signature) }}</dd></div>
                <div><dt>свидетельства</dt><dd>{{ percent(binding.scores.evidence) }}</dd></div>
              </dl>
              <small class="event-id">event · {{ compactId(binding.event_id) }}</small>
            </article>
          </section>

          <details v-if="rejectedEvents.length" class="diagnostic-details">
            <summary>Отклонённые события <b>{{ rejectedEvents.length }}</b></summary>
            <pre>{{ pretty(rejectedEvents) }}</pre>
          </details>

          <details v-if="queryGraph" class="diagnostic-details">
            <summary>Трасса интерпретации</summary>
            <div class="trace-summary">
              <span>
                акты
                <b>{{ arrayLength(trace.dialogue_act_hypotheses) }}</b>
              </span>
              <span>
                токены
                <b>{{ arrayLength(trace.token_hypotheses) }}</b>
              </span>
              <span>
                упоминания
                <b>{{ arrayLength(trace.mention_candidates) }}</b>
              </span>
              <span>
                события
                <b>{{ arrayLength(trace.accepted_events) }}</b>
              </span>
            </div>
            <pre>{{ pretty(trace) }}</pre>
          </details>
        </div>
      </aside>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue';
import { storeToRefs } from 'pinia';
import { useGraphStore } from '@/entities/graph/store';
import type { AnswerStatus, GapNode, GraphStatus, QueryMode } from '@/entities/graph/types';
import { api } from '@/shared/api/client';
import { copyJsonToClipboard } from '@/shared/utils/clipboard';

const store = useGraphStore();
const {
  state,
  hive,
  queryGraph,
  answer,
  candidateBindings,
  rejectedEvents,
  selectedBindings,
  bindingConfiguration,
  swarm,
  trace,
} = storeToRefs(store);

const question = ref('');
const mode = ref<'' | QueryMode>('');
const messageList = ref<HTMLElement | null>(null);
const exportingSpace = ref(false);
const exportStatus = ref('');
const canSubmit = computed(() =>
  Boolean(question.value.trim()) && !store.loading && !store.restoring,
);
const targetGaps = computed<GapNode[]>(() => {
  const pattern = queryGraph.value?.event_pattern;
  return pattern?.target_gaps?.length ? pattern.target_gaps : (pattern?.target_gap ? [pattern.target_gap] : []);
});
const swarmRuns = computed(() => swarm.value?.gap_swarms || []);
const retrievalMode = computed(() => swarm.value?.retrieval_mode || swarmRuns.value[0]?.retrieval_mode || 'DIRECT_EVENT_LOOKUP');
const swarmFallbackReason = computed(() => swarm.value?.fallback_reason || swarmRuns.value[0]?.fallback_reason || '');
const retrievalModeClass = computed(() => retrievalMode.value.toLowerCase().replace(/_/g, '-'));

function compactId(value: string): string {
  if (!value) return '—';
  return value.length <= 20 ? value : `${value.slice(0, 11)}…${value.slice(-6)}`;
}

function percent(value: number): string {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function pretty(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function arrayLength(value: unknown): number {
  return Array.isArray(value) ? value.length : 0;
}

function signatureCount(value: Record<string, number>): number {
  return Object.keys(value || {}).length;
}

function topEntries(value: Record<string, number>): Array<[string, number]> {
  return Object.entries(value || {})
    .sort((left, right) => right[1] - left[1])
    .slice(0, 10);
}

function readableSignature(value: string): string {
  return value.split('_').join(' ').split(':').join(' · ');
}

function bindingForGap(gapId: string) {
  return selectedBindings.value.find((binding) => binding.gap_node_id === gapId);
}

function gapLabel(gapId: string): string {
  return targetGaps.value.find((gap) => gap.node_id === gapId)?.surface || 'GAP';
}

function retrievalModeLabel(mode: string): string {
  return {
    SWARM_DIMENSIONAL: 'Размерный рой',
    SWARM_MIXED: 'Смешанный рой',
    INDEX_FALLBACK: 'Индексный fallback',
    DIRECT_EVENT_LOOKUP: 'Прямой поиск',
  }[mode] || mode;
}

function fallbackLabel(reason: string): string {
  return {
    NO_ACTIVE_DIMENSIONS: 'Нет активных измерений — используется индекс событий.',
    NO_EVENT_DIMENSION_PROJECTION: 'Измерения не проецируются в события — используется индекс событий.',
  }[reason] || reason;
}

async function exportChatSpace(): Promise<void> {
  if (!hive.value) return;
  exportingSpace.value = true;
  try {
    const payload = await api.get<Record<string, unknown>>(`/api/v2/hives/${hive.value.id}/space-export`);
    const dialogueExport = { ...payload, chat_messages: store.messages };
    if (!await copyJsonToClipboard(dialogueExport)) {
      throw new Error('Браузер не предоставил доступ к буферу обмена');
    }
    exportStatus.value = 'Скопировано';
    setTimeout(() => { exportStatus.value = ''; }, 1500);
  } catch (cause) {
    store.error = cause instanceof Error ? cause.message : 'Не удалось скопировать пространство';
  } finally {
    exportingSpace.value = false;
  }
}

function gapKindLabel(value: string): string {
  const labels: Record<string, string> = {
    EVENT_ATTACHMENT: 'участник события',
    NODE_COMPONENT: 'компонент узла',
    RELATION_VALUE: 'значение связи',
    EVENT_PROPERTY: 'свойство события',
    BOOLEAN_RESULT: 'да / нет',
    QUANTITY_VALUE: 'количество',
    WHOLE_EVENT: 'целое событие',
  };
  return labels[value] || value;
}

function graphStatusLabel(value: GraphStatus): string {
  return {
    READY: 'Готов',
    AMBIGUOUS: 'Неоднозначно',
    INCOMPLETE: 'Неполно',
    CONFLICTED: 'Конфликт',
  }[value];
}

function answerStatusLabel(value: AnswerStatus | 'PENDING' | 'ERROR'): string {
  const labels: Record<string, string> = {
    RESOLVED: 'Ответ найден',
    PARTIALLY_RESOLVED: 'Частичный ответ',
    UNRESOLVED: 'Нет связывания',
    AMBIGUOUS: 'Неоднозначно',
    CONFLICTED: 'Конфликт',
    BUILD_FAILED: 'Ошибка сборки',
    PENDING: 'Обработка',
    ERROR: 'Ошибка',
  };
  return labels[value] || value;
}

async function scrollToEnd(): Promise<void> {
  await nextTick();
  if (messageList.value) {
    messageList.value.scrollTop = messageList.value.scrollHeight;
  }
}

async function submit(): Promise<void> {
  if (!canSubmit.value) return;
  const text = question.value;
  question.value = '';
  try {
    await store.query(text, mode.value || undefined);
  } catch {
    // The store exposes the normalized error and keeps the failed turn visible.
  }
  await scrollToEnd();
}

async function newDialogue(): Promise<void> {
  await store.resetHive();
  question.value = '';
  mode.value = '';
}

async function clearChatSpace(): Promise<void> {
  if (!hive.value || !window.confirm('Очистить внутреннее пространство этого чата? Его графы, ходы и результаты будут удалены.')) return;
  try {
    await api.delete(`/api/v2/hives/${hive.value.id}`);
    await store.resetHive();
    question.value = '';
    mode.value = '';
  } catch (cause) {
    store.error = cause instanceof Error ? cause.message : 'Не удалось очистить пространство чата';
  }
}

watch(() => store.messages.length, () => {
  void scrollToEnd();
});

onMounted(async () => {
  try {
    await store.restoreHive();
  } catch {
    // The error banner is rendered next to the composer.
  }
  await scrollToEnd();
});
</script>

<style scoped lang="scss">
.graph-page {
  min-height: 100vh;
  color: #eaf2ff;
}

.topbar {
  min-height: 70px;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 24px;
  padding: 12px 28px;
  border-bottom: 1px solid rgba(159, 188, 229, 0.14);
  background: rgba(7, 15, 28, 0.9);
  backdrop-filter: blur(18px);
}

.brand {
  width: max-content;
  display: flex;
  align-items: center;
  gap: 11px;
  color: inherit;
  text-decoration: none;

  span:last-child { display: grid; }
  b { font-size: 16px; }
  small { color: #7f93b2; font-size: 10px; letter-spacing: .08em; text-transform: uppercase; }
}

.brand-mark {
  width: 34px;
  height: 34px;
  display: grid;
  place-items: center;
  border-radius: 10px;
  color: #07111f;
  background: linear-gradient(145deg, #76ead0, #6c9eff);
  font-weight: 900;
}

nav {
  display: flex;
  align-items: center;
  gap: 8px;

  a {
    padding: 8px 13px;
    border-radius: 9px;
    color: #91a4c1;
    text-decoration: none;
  }

  a.router-link-exact-active {
    color: #f4f8ff;
    background: rgba(111, 164, 255, .13);
  }
}

.session {
  justify-self: end;
  display: flex;
  align-items: center;
  gap: 9px;
  color: #8195b2;
  font-size: 11px;
}

.live-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #76e8cc;
  box-shadow: 0 0 12px #76e8cc;
}

.ghost-button {
  margin-left: 6px;
  padding: 8px 11px;
  border: 1px solid rgba(162, 189, 225, .17);
  border-radius: 8px;
  color: #a8b9d1;
  background: transparent;
  cursor: pointer;

  &:hover { border-color: rgba(118, 232, 204, .45); color: #dffcf5; }
  &:disabled { opacity: .5; cursor: wait; }
}

.ghost-button.danger {
  border-color: rgba(235, 122, 141, .45);
  color: #f3aab5;

  &:hover { border-color: #f18b9b; color: #ffe1e5; }
}

.multi-gap-panel,
.swarm-card {
  margin-top: 14px;
  padding: 13px;
  border: 1px solid rgba(115, 181, 255, .18);
  border-radius: 11px;
  background: rgba(19, 43, 72, .45);
}

.gap-map,
.swarm-runs { display: grid; gap: 7px; margin-top: 10px; }

.gap-map-item {
  display: grid;
  grid-template-columns: minmax(64px, .8fr) 20px minmax(90px, 1fr);
  align-items: center;
  gap: 5px;
  padding: 8px 9px;
  border-radius: 7px;
  color: #b8c8dc;
  background: rgba(5, 16, 30, .32);

  i { color: #76ead0; font-style: normal; text-align: center; }
  b { color: #edf7ff; overflow-wrap: anywhere; }
}

.configuration-state,
.swarm-note { margin: 10px 0 0; color: #91a8c4; font-size: 11px; line-height: 1.45; }
.configuration-state.selected { color: #79e7cd; }

.swarm-card { border-color: rgba(118, 232, 204, .2); }
.swarm-card .answer-head b { font-size: 10px; }
.swarm-card .answer-head b.index-fallback { color: #f2c879; }
.swarm-card .answer-head b.swarm-dimensional { color: #79e7cd; }

.swarm-run {
  display: grid;
  grid-template-columns: 44px 1fr;
  gap: 2px 8px;
  padding: 9px;
  border-left: 2px solid rgba(118, 232, 204, .55);
  border-radius: 0 7px 7px 0;
  background: rgba(5, 16, 30, .27);

  b { color: #e2f4ef; font-size: 12px; }
  > small { grid-column: 2; color: #8296b2; font-size: 10px; }
}

.swarm-index { grid-row: span 2; align-self: center; color: #76ead0; font-size: 9px; letter-spacing: .05em; }
.mission-list,
.packet-list { grid-column: 2; display: grid; gap: 5px; margin-top: 5px; }
.mission-row { display: grid; grid-template-columns: 58px 1fr; gap: 3px 7px; }
.mission-row strong { color: #76ead0; font-size: 10px; }
.mission-row small { color: #8296b2; font-size: 10px; }
.packet-list small { color: #9eb3cc; font-size: 9px; }
.universe-route { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 4px; margin-top: 3px; }
.universe-route span { padding: 2px 5px; border-radius: 4px; color: #a8bfda; background: rgba(106, 157, 220, .13); font-size: 9px; }

.workspace {
  width: min(1880px, 100%);
  height: calc(100vh - 70px);
  display: grid;
  grid-template-columns: minmax(290px, 360px) minmax(470px, 1fr) minmax(320px, 400px);
  gap: 14px;
  margin: 0 auto;
  padding: 16px;
}

.panel {
  overflow: hidden;
  background: rgba(11, 23, 41, .82);
}

.panel-head {
  min-height: 72px;
  padding: 17px 19px 14px;

  h1,
  h2 { font-size: 16px; }
}

.turn-counter,
.count-badge {
  color: #91a5c3;
  font-size: 11px;
}

.count-badge {
  min-width: 26px;
  height: 26px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  color: #8ee9d5;
  background: rgba(118, 232, 204, .1);
}

.chat-panel {
  display: grid;
  grid-template-rows: auto 1fr auto;
}

.messages {
  overflow-y: auto;
  min-height: 0;
  padding: 18px;
  scrollbar-color: rgba(115, 176, 255, .25) transparent;
}

.welcome {
  height: 100%;
  min-height: 260px;
  display: grid;
  align-content: center;
  justify-items: center;
  text-align: center;

  h2 { margin: 12px 0 5px; font-size: 17px; }
  p { max-width: 270px; margin: 0; color: #8194b1; font-size: 12px; line-height: 1.7; }
}

.welcome-icon {
  width: 54px;
  height: 54px;
  display: grid;
  place-items: center;
  border: 1px solid rgba(118, 232, 204, .3);
  border-radius: 18px;
  color: #80ead3;
  background: rgba(118, 232, 204, .07);
  font-size: 28px;
}

.message {
  display: grid;
  margin-bottom: 17px;

  &.user { justify-items: end; }
  &.assistant { justify-items: start; }
}

.message-author,
.message-status {
  margin: 0 4px 5px;
  color: #6f85a5;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: .08em;
}

.message-status {
  margin-top: 5px;
  text-transform: none;
  letter-spacing: 0;
}

.bubble {
  max-width: 92%;
  padding: 11px 13px;
  border: 1px solid rgba(161, 188, 226, .13);
  border-radius: 5px 14px 14px;
  color: #dce8f8;
  background: rgba(20, 39, 65, .86);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.user .bubble {
  border-color: rgba(111, 164, 255, .22);
  border-radius: 14px 5px 14px 14px;
  background: rgba(54, 100, 174, .24);
}

.typing {
  display: flex;
  gap: 4px;
  padding-block: 15px;

  i {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #75b1ff;
    animation: pulse 1s infinite alternate;
  }
  i:nth-child(2) { animation-delay: .2s; }
  i:nth-child(3) { animation-delay: .4s; }
}

@keyframes pulse {
  to { opacity: .25; transform: translateY(-2px); }
}

.composer {
  padding: 14px;
  border-top: 1px solid rgba(162, 189, 225, .1);

  textarea {
    width: 100%;
    resize: none;
    padding: 12px 13px;
    border: 1px solid rgba(162, 189, 225, .2);
    border-radius: 11px;
    outline: 0;
    color: #eef5ff;
    background: rgba(5, 14, 27, .68);

    &::placeholder { color: #596f8e; }
    &:focus { border-color: rgba(111, 164, 255, .62); }
  }
}

.composer-footer {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 10px;
  margin-top: 10px;
}

.mode-select {
  display: grid;
  gap: 3px;
  color: #6f83a1;
  font-size: 9px;
  letter-spacing: .08em;
  text-transform: uppercase;

  select {
    border: 0;
    outline: 0;
    color: #aabbd2;
    background: transparent;
    font-size: 11px;
    text-transform: none;
  }
}

.primary-button {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 9px 13px;
  border: 0;
  border-radius: 9px;
  color: #07111f;
  background: linear-gradient(135deg, #79e6d0, #78b3ff);
  font-weight: 700;
  cursor: pointer;

  &:disabled { opacity: .45; cursor: not-allowed; }
}

.error {
  margin: 9px 0 0;
  color: #ff8998;
  font-size: 11px;
}

.graph-panel {
  display: grid;
  grid-template-rows: auto 1fr;
}

.status-chip {
  padding: 5px 9px;
  border-radius: 999px;
  color: #89ebd4;
  background: rgba(118, 232, 204, .09);
  font-size: 10px;
  text-transform: uppercase;

  &.ambiguous,
  &.incomplete { color: #ffd37e; background: rgba(255, 201, 97, .1); }
  &.conflicted { color: #ff8a99; background: rgba(212, 81, 99, .12); }
  &.idle { color: #788ba7; background: rgba(125, 145, 176, .08); }
}

.graph-empty {
  display: grid;
  align-content: center;
  justify-items: center;
  text-align: center;

  h3 { margin: 22px 0 6px; font-size: 16px; }
  p { max-width: 360px; margin: 0; color: #7185a4; font-size: 12px; }
}

.empty-orbit {
  position: relative;
  width: 126px;
  height: 126px;
  display: grid;
  place-items: center;
  border: 1px solid rgba(115, 176, 255, .22);
  border-radius: 50%;
  box-shadow: 0 0 80px rgba(67, 132, 231, .13);

  &::before {
    content: '';
    position: absolute;
    inset: 17px;
    border: 1px dashed rgba(118, 232, 204, .2);
    border-radius: 50%;
  }
  b { color: #79e7d0; font-size: 14px; letter-spacing: .12em; }
  span {
    position: absolute;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #72aaff;
  }
  span:first-child { top: 13px; left: 26px; }
  span:nth-child(2) { right: 12px; bottom: 31px; background: #76e8cc; }
}

.graph-content {
  min-height: 0;
  overflow: auto;
  display: grid;
  grid-template-rows: auto minmax(380px, 1fr) auto;
}

.graph-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 11px 20px;

  span {
    padding: 4px 7px;
    border: 1px solid rgba(160, 188, 228, .12);
    border-radius: 6px;
    color: #7388a8;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 9px;
  }
}

.event-graph {
  min-height: 390px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 24px 22px;
  background:
    linear-gradient(rgba(113, 151, 209, .035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(113, 151, 209, .035) 1px, transparent 1px);
  background-size: 28px 28px;
}

.known-row {
  width: 100%;
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;
  justify-content: center;
  gap: 10px;
}

.graph-node {
  min-width: 150px;
  max-width: 220px;
  display: grid;
  gap: 2px;
  padding: 12px 15px;
  border: 1px solid rgba(111, 164, 255, .26);
  border-radius: 11px;
  background: rgba(17, 36, 62, .92);
  box-shadow: 0 12px 30px rgba(0, 0, 0, .18);
  text-align: center;

  small { color: #70a9ff; font-size: 8px; letter-spacing: .1em; }
  strong { overflow-wrap: anywhere; font-size: 14px; }
  > span { color: #8296b3; font-size: 10px; }
}

.node-components {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 3px;
  margin-top: 4px;

  i {
    color: #a9b9d0;
    font-size: 9px;
    font-style: normal;
  }
}

.predicate-node {
  min-width: 220px;
  border-color: rgba(118, 232, 204, .38);
  background: linear-gradient(145deg, rgba(20, 69, 71, .78), rgba(20, 47, 72, .9));

  small { color: #77ead0; }
}

.gap-node {
  min-width: 210px;
  border-style: dashed;
  border-color: rgba(255, 201, 97, .55);
  background: rgba(77, 57, 25, .33);

  small { color: #ffd075; }
}

.muted-node { opacity: .56; }

.connector {
  width: 1px;
  height: 35px;
  position: relative;
  background: linear-gradient(#659eea, #76e8cc);

  span {
    position: absolute;
    left: -3px;
    bottom: -1px;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #76e8cc;
  }
}

.gap-connector {
  background: linear-gradient(#76e8cc, #ffc961);
  span { background: #ffc961; }
}

.signature-panel {
  padding: 15px 19px 19px;
  border-top: 1px solid rgba(162, 189, 225, .08);
}

.subhead {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;

  h3 { margin: 0; font-size: 12px; }
  span { color: #667b9b; font-size: 9px; }
}

.signature-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.signature {
  display: flex;
  gap: 8px;
  padding: 5px 7px;
  border: 1px solid rgba(148, 178, 221, .13);
  border-radius: 6px;
  color: #8fa2bf;
  background: rgba(8, 18, 33, .48);
  font-size: 9px;

  b { color: #7ee7d0; }
}

.inspector-panel {
  display: grid;
  grid-template-rows: auto 1fr;
}

.inspector-scroll {
  min-height: 0;
  overflow-y: auto;
  padding: 14px;
}

.answer-card {
  margin-bottom: 14px;
  padding: 14px;
  border: 1px solid rgba(118, 232, 204, .22);
  border-radius: 11px;
  background: rgba(21, 65, 64, .3);

  &.unresolved,
  &.ambiguous { border-color: rgba(255, 201, 97, .25); background: rgba(91, 65, 21, .18); }
  &.conflicted,
  &.build_failed { border-color: rgba(212, 81, 99, .3); background: rgba(91, 25, 38, .19); }

  > p { margin: 8px 0 0; color: #9eafc7; font-size: 11px; line-height: 1.6; }
}

.answer-head,
.validation {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 9px;
}

.answer-head {
  margin-bottom: 9px;
  color: #8195b2;
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: .08em;

  b { color: #7ce5cf; }
}

.answer-surface {
  display: block;
  font-size: 17px;
}

.validation {
  margin-top: 12px;
  padding-top: 9px;
  border-top: 1px solid rgba(161, 188, 226, .09);
  color: #ff8898;
  font-size: 9px;

  .valid { color: #7fe9d2; }
  small { color: #7185a4; }
}

.empty-candidates {
  min-height: 180px;
  display: grid;
  align-content: center;
  justify-items: center;
  gap: 8px;
  color: #667b99;
  text-align: center;

  span { font-size: 32px; }
  p { max-width: 210px; margin: 0; font-size: 11px; }
}

.candidate-list { display: grid; gap: 9px; }

.candidate {
  padding: 12px;
  border: 1px solid rgba(162, 189, 225, .13);
  border-radius: 10px;
  background: rgba(9, 19, 34, .55);

  &.selected {
    border-color: rgba(118, 232, 204, .35);
    box-shadow: inset 3px 0 #76e8cc;
  }
}

.candidate-head {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 9px;

  div { display: grid; }
  strong { overflow-wrap: anywhere; font-size: 13px; }
  small { color: #7589a7; font-size: 9px; }
  > b { color: #80e7d1; font-size: 12px; }
}

.rank {
  width: 25px;
  height: 25px;
  display: grid;
  place-items: center;
  border-radius: 7px;
  color: #7fa9e8;
  background: rgba(111, 164, 255, .1);
  font-size: 9px;
}

.score-bar {
  height: 3px;
  overflow: hidden;
  margin: 10px 0;
  border-radius: 4px;
  background: rgba(164, 191, 228, .1);

  i {
    height: 100%;
    display: block;
    border-radius: inherit;
    background: linear-gradient(90deg, #6fa4ff, #76e8cc);
  }
}

.scores {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 5px;
  margin: 0;

  div { display: grid; gap: 1px; }
  dt { color: #617694; font-size: 8px; }
  dd { margin: 0; color: #a8bad1; font-size: 9px; }
}

.event-id {
  display: block;
  margin-top: 9px;
  color: #536985;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 8px;
}

.diagnostic-details {
  margin-top: 11px;
  border: 1px solid rgba(162, 189, 225, .11);
  border-radius: 9px;
  background: rgba(6, 15, 28, .4);

  summary {
    padding: 11px;
    color: #96a9c4;
    cursor: pointer;
    font-size: 10px;
  }
  summary b { float: right; color: #ffba70; }
  pre {
    max-height: 320px;
    overflow: auto;
    margin: 0;
    padding: 0 11px 11px;
    color: #8197b6;
    font-size: 8px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
}

.trace-summary {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 5px;
  padding: 0 10px 10px;

  span {
    display: flex;
    justify-content: space-between;
    padding: 6px;
    border-radius: 6px;
    color: #7287a7;
    background: rgba(111, 164, 255, .05);
    font-size: 8px;
  }
  b { color: #7edfcf; }
}

.empty-inline { color: #617695; font-size: 10px; }
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
}

@media (max-width: 1180px) {
  .topbar { grid-template-columns: 1fr auto; }
  .topbar nav { order: 3; grid-column: 1 / -1; justify-content: center; }
  .workspace {
    height: auto;
    grid-template-columns: minmax(290px, .8fr) minmax(480px, 1.2fr);
  }
  .inspector-panel { grid-column: 1 / -1; min-height: 500px; }
}

@media (max-width: 760px) {
  .topbar { display: flex; flex-wrap: wrap; padding: 11px 14px; }
  .topbar nav { width: 100%; order: 3; }
  .session { margin-left: auto; }
  .session > span:not(.live-dot) { display: none; }
  .workspace { display: flex; flex-direction: column; padding: 9px; }
  .chat-panel { min-height: 650px; }
  .graph-panel { min-height: 680px; }
  .inspector-panel { min-height: 500px; }
}
</style>
