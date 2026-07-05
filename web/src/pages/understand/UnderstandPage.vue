<template>
  <section class="page understand-page">
    <div class="split understand-layout">
      <section class="panel understand-panel">
        <div class="header">
          <div>
            <p class="eyebrow">Пониматель</p>
            <h2>Текст → леммы → search tokens</h2>
            <p class="muted">
              Диагностический слой только читает checkpoint, строит токены и не пишет в память.
            </p>
          </div>
          <div class="row">
            <span class="badge">read only</span>
            <span class="badge">pymorphy3</span>
          </div>
        </div>

        <form class="form" @submit.prevent="understand">
          <label class="wide">
            Текст
            <textarea
              v-model="text"
              rows="5"
              placeholder="котики едят"
              @keydown.enter.exact.prevent="understand"
            ></textarea>
          </label>
          <label>
            Lang
            <select v-model="lang">
              <option value="auto">auto</option>
              <option value="ru">ru</option>
              <option value="en">en</option>
            </select>
          </label>
          <label>
            session_id
            <input v-model="sessionId" placeholder="diagnostic-session" />
          </label>
          <label>
            turn_id
            <input v-model="turnId" placeholder="turn-1" />
          </label>
          <div class="row wide">
            <button class="primary" type="submit" :disabled="loading || !text.trim()">Разобрать</button>
            <span class="muted">Enter отправляет, Shift+Enter добавляет новую строку.</span>
          </div>
        </form>
      </section>

      <aside class="stack">
        <section class="panel compact summary-panel" v-if="result">
          <div class="row summary-head">
            <h3>Summary</h3>
            <span class="badge">{{ result.lang }}</span>
          </div>
          <div class="summary-grid">
            <div class="summary-item">
              <strong>{{ result.summary.total_tokens }}</strong>
              <span class="muted">tokens</span>
            </div>
            <div class="summary-item">
              <strong>{{ result.summary.working_tokens }}</strong>
              <span class="muted">working</span>
            </div>
            <div class="summary-item">
              <strong>{{ result.summary.stop_words }}</strong>
              <span class="muted">stop words</span>
            </div>
            <div class="summary-item">
              <strong>{{ result.summary.candidates }}</strong>
              <span class="muted">candidates</span>
            </div>
          </div>
          <div class="chip-row" v-if="result.summary.search_tokens.length">
            <span v-for="token in result.summary.search_tokens" :key="token" class="badge">{{ token }}</span>
          </div>
        </section>

        <section class="panel compact summary-panel" v-if="result">
          <div class="row summary-head">
            <h3>Контракт</h3>
            <span class="badge">{{ result.session_id ?? 'no session' }}</span>
          </div>
          <p class="muted">session_id и turn_id возвращаются без записи в checkpoint.</p>
          <div class="row">
            <span class="badge">session {{ result.session_id ?? 'null' }}</span>
            <span class="badge">turn {{ result.turn_id ?? 'null' }}</span>
          </div>
        </section>
      </aside>
    </div>

    <section v-if="result" class="panel results-panel">
      <div class="row results-head">
        <h3>Tokens</h3>
        <span class="badge">raw → lemma → search token → concept uri → status → morphology</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>raw</th>
              <th>lemma</th>
              <th>search token</th>
              <th>concept uri</th>
              <th>status</th>
              <th>morphology</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="token in result.tokens"
              :key="`${token.raw_token}:${token.search_token}:${token.match_status}`"
              :class="rowClass(token.match_status)"
            >
              <td>
                <strong>{{ token.raw_token }}</strong>
                <span v-if="token.is_stop_word" class="badge signal">stop</span>
              </td>
              <td>{{ token.lemma }}</td>
              <td>{{ token.search_token || '—' }}</td>
              <td>{{ token.concept_uri || '—' }}</td>
              <td>
                <span class="badge" :class="badgeClass(token.match_status)">{{ token.match_status }}</span>
              </td>
              <td class="morphology">
                <span v-for="entry in morphologyEntries(token.morphology)" :key="entry" class="badge">{{ entry }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { api } from '@/shared/api/client';
import type { UnderstandingResponse, UnderstandingToken } from '@/shared/api/types';

const text = ref('котики едят');
const lang = ref<'auto' | 'ru' | 'en'>('auto');
const sessionId = ref('');
const turnId = ref('');
const loading = ref(false);
const result = ref<UnderstandingResponse | null>(null);

async function understand() {
  if (!text.value.trim()) return;
  loading.value = true;
  try {
    result.value = await api.understand({
      text: text.value,
      lang: lang.value,
      session_id: sessionId.value || undefined,
      turn_id: turnId.value || undefined,
    });
  } finally {
    loading.value = false;
  }
}

function rowClass(status: UnderstandingToken['match_status']) {
  return [
    'token-row',
    status === 'ignored_stop_word' ? 'token-row--stop' : '',
    status === 'candidate' ? 'token-row--candidate' : '',
    status === 'partial_root_match' ? 'token-row--partial' : '',
    status === 'edit_distance_match' ? 'token-row--edit' : '',
  ]
    .filter(Boolean)
    .join(' ');
}

function badgeClass(status: UnderstandingToken['match_status']) {
  return {
    signal: status === 'candidate' || status === 'edit_distance_match',
    highlight: status === 'partial_root_match',
  };
}

function morphologyEntries(morphology: UnderstandingToken['morphology']) {
  return Object.entries(morphology)
    .filter(([, value]) => value)
    .map(([key, value]) => `${key}=${value}`);
}
</script>

<style scoped lang="scss">
.understand-page {
  align-content: start;
}

.understand-layout {
  align-items: start;
}

.understand-panel,
.results-panel {
  display: grid;
  gap: 16px;
}

.header,
.summary-head,
.results-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 12px;
  align-items: start;
}

.eyebrow {
  margin: 0 0 4px;
  color: var(--accent-2);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.header h2,
.summary-head h3,
.results-head h3 {
  margin: 0;
}

.form {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.wide {
  grid-column: 1 / -1;
}

.stack {
  display: grid;
  gap: 16px;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.summary-item {
  display: grid;
  gap: 4px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 10px;
  background: linear-gradient(180deg, rgba(23, 107, 87, 0.04), rgba(23, 107, 87, 0.01));
}

.summary-item strong {
  font-size: 20px;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.table-wrap {
  overflow: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 820px;
}

th,
td {
  border-bottom: 1px solid var(--line);
  padding: 10px 8px;
  text-align: left;
  vertical-align: top;
}

th {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.token-row--stop {
  background: rgba(215, 47, 47, 0.04);
}

.token-row--candidate {
  background: rgba(245, 166, 35, 0.08);
}

.token-row--partial {
  background: rgba(23, 107, 87, 0.06);
}

.token-row--edit {
  background: rgba(15, 79, 67, 0.08);
}

.morphology {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.badge.highlight {
  background: rgba(23, 107, 87, 0.18);
  color: var(--accent-2);
}

@media (max-width: 980px) {
  .summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .form {
    grid-template-columns: 1fr;
  }
}
</style>
