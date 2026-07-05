<template>
  <section class="page decode-page">
    <div class="split decode-layout">
      <section class="panel decode-panel">
        <div class="header">
          <div>
            <p class="eyebrow">Декодер</p>
            <h2>Tokens → SVO → фраза</h2>
            <p class="muted">
              Диагностический декодер собирает предложение из токенов и не пишет в checkpoint.
            </p>
          </div>
          <div class="row">
            <span class="badge">read only</span>
            <span class="badge">ru / en</span>
          </div>
        </div>

        <form class="form" @submit.prevent="decode">
          <label class="wide">
            Текст
            <textarea
              v-model="text"
              rows="4"
              placeholder="кот есть рыба мясо"
              @keydown.enter.exact.prevent="decode"
            ></textarea>
          </label>
          <label class="wide">
            Tokens
            <textarea
              v-model="tokensText"
              rows="3"
              placeholder="кот, есть, рыба, мясо"
              @keydown.enter.exact.prevent="decode"
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
            <input v-model="sessionId" placeholder="decode-session" />
          </label>
          <label>
            turn_id
            <input v-model="turnId" placeholder="turn-1" />
          </label>
          <div class="row wide">
            <button class="primary" type="submit" :disabled="loading">Декодировать</button>
            <span class="muted">Если tokens заполнен, он имеет приоритет над text.</span>
          </div>
        </form>
      </section>

      <aside class="stack">
        <section class="panel compact summary-panel" v-if="result">
          <div class="row summary-head">
            <h3>Sentence</h3>
            <span class="badge">{{ result.pattern }}</span>
          </div>
          <p class="sentence">{{ result.sentence || '—' }}</p>
          <div class="chip-row" v-if="result.input_tokens.length">
            <span v-for="token in result.input_tokens" :key="token" class="badge">{{ token }}</span>
          </div>
        </section>

        <section class="panel compact summary-panel" v-if="result">
          <div class="row summary-head">
            <h3>Summary</h3>
            <span class="badge">{{ result.lang }}</span>
          </div>
          <div class="summary-grid">
            <div class="summary-item">
              <strong>{{ result.summary.total_tokens }}</strong>
              <span class="muted">total</span>
            </div>
            <div class="summary-item">
              <strong>{{ result.summary.used_tokens }}</strong>
              <span class="muted">used</span>
            </div>
            <div class="summary-item">
              <strong>{{ result.summary.objects }}</strong>
              <span class="muted">objects</span>
            </div>
            <div class="summary-item">
              <strong>{{ result.summary.fallbacks }}</strong>
              <span class="muted">fallbacks</span>
            </div>
          </div>
        </section>

        <section class="panel compact summary-panel" v-if="result">
          <div class="row summary-head">
            <h3>Contract</h3>
            <span class="badge">{{ result.session_id ?? 'no session' }}</span>
          </div>
          <p class="muted">session_id и turn_id только проходят через request-response.</p>
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
        <span class="badge">role → surface → morphology</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>input token</th>
              <th>normalized</th>
              <th>role</th>
              <th>surface</th>
              <th>status</th>
              <th>concept uri</th>
              <th>morphology</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="token in result.tokens" :key="`${token.role}:${token.input_token}:${token.surface}`">
              <td><strong>{{ token.input_token }}</strong></td>
              <td>{{ token.normalized_token }}</td>
              <td><span class="badge">{{ token.role }}</span></td>
              <td>{{ token.surface }}</td>
              <td><span class="badge" :class="badgeClass(token.transform_status)">{{ token.transform_status }}</span></td>
              <td>{{ token.concept_uri || '—' }}</td>
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
import type { DecodeResponse, DecodeToken } from '@/shared/api/types';

const text = ref('кот есть рыба мясо');
const tokensText = ref('кот, есть, рыба, мясо');
const lang = ref<'auto' | 'ru' | 'en'>('auto');
const sessionId = ref('');
const turnId = ref('');
const loading = ref(false);
const result = ref<DecodeResponse | null>(null);

async function decode() {
  loading.value = true;
  try {
    result.value = await api.decode({
      text: text.value,
      tokens: parseTokens(tokensText.value),
      lang: lang.value,
      session_id: sessionId.value || undefined,
      turn_id: turnId.value || undefined,
    });
  } finally {
    loading.value = false;
  }
}

function parseTokens(value: string) {
  return value
    .split(/[\s,]+/u)
    .map((token) => token.trim())
    .filter(Boolean);
}

function badgeClass(status: DecodeToken['transform_status']) {
  return {
    signal: status === 'fallback',
    highlight: status === 'inflected',
  };
}

function morphologyEntries(morphology: DecodeToken['morphology']) {
  return Object.entries(morphology)
    .filter(([, value]) => value)
    .map(([key, value]) => `${key}=${value}`);
}
</script>

<style scoped lang="scss">
.decode-page {
  align-content: start;
}

.decode-layout {
  align-items: start;
}

.decode-panel,
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
  background: linear-gradient(180deg, rgba(15, 79, 67, 0.06), rgba(15, 79, 67, 0.01));
}

.summary-item strong {
  font-size: 20px;
}

.sentence {
  margin: 0;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: linear-gradient(180deg, rgba(23, 107, 87, 0.08), rgba(23, 107, 87, 0.02));
  font-size: 18px;
  line-height: 1.5;
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
  min-width: 960px;
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
