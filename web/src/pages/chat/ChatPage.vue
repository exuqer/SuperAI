<template>
  <section class="page chat-page">
    <div class="split chat-layout">
      <section class="panel chat-panel">
        <div class="chat-header">
          <div>
            <p class="eyebrow">Диалог</p>
            <h2>session_id=default</h2>
            <p class="muted">
              Режим resonance хранит контекст сессии, активные плоскости и tentative-формы.
            </p>
          </div>
          <div class="row">
            <span class="badge">turns {{ activeSession?.turn_count ?? 0 }}</span>
            <span class="badge" v-if="activeSession">updated {{ formatTime(activeSession.updated_at) }}</span>
            <button type="button" @click="resetSession">Сбросить</button>
          </div>
        </div>

        <div ref="historyRef" class="chat-history">
          <template v-if="turns.length">
            <article
              v-for="turn in turns"
              :key="`${turn.result_id}:${turn.created_at ?? 0}:${turn.role}:${turn.text}`"
              class="bubble"
              :class="`bubble--${turn.role === 'assistant' ? 'assistant' : 'user'}`"
            >
              <div class="bubble-head">
                <strong>{{ turn.role === 'assistant' ? 'Ассистент' : 'Пользователь' }}</strong>
                <span class="muted">{{ formatTime(turn.created_at) }}</span>
              </div>
              <p>{{ turn.text }}</p>
              <div v-if="turn.concepts?.length" class="concept-tags">
                <span v-for="concept in turn.concepts" :key="concept" class="badge">{{ concept }}</span>
              </div>
              <div v-if="turn.role === 'assistant'" class="bubble-actions">
                <button
                  type="button"
                  :data-feedback-result-id="turn.result_id"
                  data-feedback-score="5"
                  @click="feedback(turn.result_id, 5)"
                >
                  Хорошо
                </button>
                <button
                  type="button"
                  :data-feedback-result-id="turn.result_id"
                  data-feedback-score="1"
                  @click="feedback(turn.result_id, 1)"
                >
                  Плохо
                </button>
              </div>
            </article>
          </template>
          <div v-else class="empty-state">
            <strong>Чат пуст.</strong>
            <p class="muted">Отправьте первое сообщение, чтобы заполнить persistent session.</p>
          </div>
        </div>

        <form class="composer" @submit.prevent="send">
          <label class="composer-field">
            Сообщение
            <textarea
              v-model="text"
              rows="3"
              placeholder="Напишите сообщение..."
              @keydown.enter.exact.prevent="send"
            ></textarea>
          </label>
          <div class="row composer-actions">
            <button class="primary" type="submit" :disabled="runtime.loading || !text.trim()">
              Отправить
            </button>
            <span class="muted">Enter отправляет, Shift+Enter добавляет новую строку.</span>
          </div>
        </form>
      </section>

      <aside class="chat-side">
        <section class="panel compact settings-panel">
          <div class="row settings-head">
            <h3>Настройки</h3>
            <span class="badge">session default</span>
          </div>
          <div class="toolbar settings-grid">
            <label>
              Lang
              <select v-model="lang">
                <option value="auto">auto</option>
                <option value="ru">ru</option>
                <option value="en">en</option>
              </select>
            </label>
            <label>
              Mode
              <select v-model="mode">
                <option value="resonance">resonance</option>
                <option value="hybrid">hybrid</option>
                <option value="graph">graph</option>
              </select>
            </label>
            <label>
              Creativity
              <input v-model.number="creativity" type="number" min="0" max="1" step="0.05" />
            </label>
            <label>
              Strength vector
              <input v-model="strengthVector" placeholder="3" />
            </label>
            <label>
              Ants
              <input v-model.number="ants" type="number" min="1" />
            </label>
            <label>
              Depth
              <input v-model.number="depth" type="number" min="1" />
            </label>
          </div>
        </section>

        <details class="panel diagnostics-panel" :open="Boolean(runtime.lastAnalysis)">
          <summary>
            <span>Диагностика последнего ответа</span>
            <span class="muted" v-if="runtime.lastAnalysis">{{ runtime.lastAnalysis.result.result_id }}</span>
          </summary>
          <div v-if="runtime.lastAnalysis" class="diagnostics-grid">
            <section class="diagnostic-card">
              <h3>Ответ</h3>
              <p class="answer">{{ runtime.lastAnalysis.result.response }}</p>
              <p class="muted">{{ runtime.lastAnalysis.result.summary }}</p>
              <div class="row">
                <span class="badge">plane {{ activePlane }}</span>
                <span class="badge signal" v-if="tentativeForms.length">tentative {{ tentativeForms.length }}</span>
              </div>
            </section>

            <section class="diagnostic-card">
              <h3>Активированные понятия</h3>
              <div class="grid-list">
                <div
                  v-for="item in runtime.lastAnalysis.result.activated_concepts"
                  :key="item.uri"
                  class="concept-row"
                >
                  <strong>{{ item.label }}</strong>
                  <span class="muted">{{ item.uri }}</span>
                  <span class="badge">score {{ item.score }}</span>
                </div>
              </div>
            </section>

            <section class="diagnostic-card">
              <h3>Semantic vector</h3>
              <pre>{{ compactJson(runtime.lastAnalysis.result.semantic_vector) }}</pre>
            </section>
          </div>
          <p v-else class="muted">Отправьте сообщение, чтобы увидеть диагностику ответа.</p>
        </details>

        <div v-if="runtime.graph" class="graph-stack">
          <GraphViewer :graph="runtime.graph" @select-node="selectNode" @select-edge="selectEdge" />
          <NodeInspector :node="selectedNode" :edge="selectedEdge" :detail="detail" @select-edge="selectedEdge = $event" />
        </div>
      </aside>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue';
import { useRuntimeStore } from '@/app/stores/runtime';
import GraphViewer from '@/features/graph-viewer/ui/GraphViewer.vue';
import NodeInspector from '@/features/node-inspector/ui/NodeInspector.vue';
import { api } from '@/shared/api/client';
import type { ChatSession, ConceptDetail, GraphEdge, GraphNode } from '@/shared/api/types';
import { compactJson, formatTime, parseStrengthVector } from '@/shared/lib/format';

const runtime = useRuntimeStore();
const text = ref('');
const sessionId = 'default';
const lang = ref('auto');
const mode = ref<'resonance' | 'graph' | 'hybrid'>('resonance');
const strengthVector = ref('3');
const ants = ref(32);
const depth = ref(4);
const creativity = ref(0.35);
const sessions = ref<ChatSession[]>([]);
const selectedNode = ref<GraphNode | null>(null);
const selectedEdge = ref<GraphEdge | null>(null);
const detail = ref<ConceptDetail | null>(null);
const historyRef = ref<HTMLElement | null>(null);

const activeSession = computed(() => sessions.value.find((item) => item.session_id === sessionId) ?? null);
const turns = computed(() => activeSession.value?.turns ?? []);
const activePlane = computed(() => String(runtime.lastAnalysis?.result.semantic_vector?.active_plane ?? ''));
const tentativeForms = computed(() => {
  const values = runtime.lastAnalysis?.result.semantic_vector?.tentative_forms;
  return Array.isArray(values) ? values : [];
});

async function refreshSessions() {
  sessions.value = await api.getSessions();
  await nextTick();
  scrollHistory();
}

function scrollHistory() {
  const node = historyRef.value;
  if (!node) return;
  node.scrollTop = node.scrollHeight;
}

async function send() {
  if (!text.value.trim()) return;
  if (mode.value === 'resonance') {
    await runtime.resonanceChat({
      text: text.value,
      session_id: sessionId,
      lang: lang.value === 'auto' ? 'ru' : lang.value,
      ants: ants.value,
      creativity: creativity.value,
    });
  } else {
    await runtime.chat({
      text: text.value,
      session_id: sessionId,
      lang: lang.value,
      mode: mode.value,
      ants: ants.value,
      depth: depth.value,
      strength_vector: parseStrengthVector(strengthVector.value),
    });
  }
  text.value = '';
  await refreshSessions();
}

async function resetSession() {
  await api.resetSession(sessionId);
  await refreshSessions();
}

async function feedback(resultId: string, score: number) {
  if (!resultId) return;
  if (
    runtime.lastAnalysis?.result.result_id === resultId &&
    runtime.lastAnalysis?.result.semantic_vector?.mode === 'resonance'
  ) {
    await api.resonanceFeedback({ result_id: resultId, score, session_id: sessionId });
    return;
  }
  await api.sendFeedback({ result_id: resultId, score });
}

async function selectNode(node: GraphNode) {
  selectedNode.value = node;
  selectedEdge.value = null;
  detail.value = await api.getConceptDetail(node.uri, runtime.lastAnalysis?.result.result_id);
}

function selectEdge(edge: GraphEdge) {
  selectedEdge.value = edge;
  selectedNode.value = null;
  detail.value = null;
}

onMounted(() => {
  refreshSessions().catch(() => undefined);
});
</script>

<style scoped lang="scss">
.chat-page {
  align-content: start;
}

.chat-layout {
  align-items: start;
}

.chat-panel {
  display: grid;
  gap: 16px;
}

.chat-header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 16px;
}

.chat-header h2,
.settings-head h3,
.diagnostic-card h3 {
  margin: 0;
}

.eyebrow {
  margin: 0 0 4px;
  color: var(--accent-2);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.chat-history {
  display: grid;
  gap: 12px;
  max-height: 64vh;
  overflow: auto;
  padding-right: 4px;
}

.bubble {
  display: grid;
  gap: 8px;
  max-width: min(88%, 760px);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 14px 16px;
  background: linear-gradient(180deg, #ffffff, #f6f8fa);
  box-shadow: 0 10px 20px rgba(20, 24, 30, 0.05);
}

.bubble p {
  margin: 0;
  white-space: pre-wrap;
  line-height: 1.5;
}

.bubble--user {
  justify-self: end;
  border-color: rgba(23, 107, 87, 0.2);
  background: linear-gradient(135deg, rgba(23, 107, 87, 0.16), rgba(15, 79, 67, 0.22));
}

.bubble--assistant {
  justify-self: start;
  background: #fff;
}

.bubble-head,
.composer-actions,
.settings-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 8px;
  align-items: center;
}

.concept-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.bubble-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.empty-state {
  display: grid;
  gap: 4px;
  min-height: 220px;
  place-content: center;
  text-align: center;
  border: 1px dashed var(--line);
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(23, 107, 87, 0.04), rgba(23, 107, 87, 0.01));
}

.composer {
  display: grid;
  gap: 10px;
  border-top: 1px solid var(--line);
  padding-top: 16px;
}

.composer-field {
  display: grid;
  gap: 6px;
}

.chat-side {
  display: grid;
  gap: 16px;
}

.settings-panel {
  display: grid;
  gap: 12px;
}

.settings-grid {
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
}

.diagnostics-panel {
  display: grid;
  gap: 12px;
}

.diagnostics-panel > summary {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  cursor: pointer;
  list-style: none;
  font-weight: 600;
}

.diagnostics-panel > summary::-webkit-details-marker {
  display: none;
}

.diagnostics-grid {
  display: grid;
  gap: 12px;
}

.diagnostic-card {
  display: grid;
  gap: 8px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--surface);
  padding: 14px;
}

.answer {
  font-size: 18px;
  line-height: 1.5;
}

.concept-row {
  display: grid;
  gap: 4px;
  padding-top: 8px;
  border-top: 1px solid var(--line);
}

.graph-stack {
  display: grid;
  gap: 16px;
}

@media (max-width: 980px) {
  .chat-history {
    max-height: 48vh;
  }
}
</style>
