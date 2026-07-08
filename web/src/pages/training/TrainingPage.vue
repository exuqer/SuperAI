<template>
  <section class="page training-page">
    <div class="split training-layout">
      <section class="panel training-panel">
        <div class="training-header">
          <div>
            <p class="eyebrow">Resonance training</p>
            <h2>Вопрос → ответ</h2>
            <p class="muted">
              Основной сценарий теперь автоматический: вводите вопрос и ожидаемый ответ, а разнесение по ролям и плоскостям
              делает backend. Ручная разметка спрятана в advanced.
            </p>
          </div>
          <div class="row header-actions">
            <button class="danger" type="button" data-testid="full-reset-button" @click="resetNetwork">
              Полный сброс
            </button>
            <button class="primary" type="button" @click="seedResonance">Seed grammar</button>
          </div>
        </div>

        <div class="toolbar training-toolbar">
          <label>
            Sample
            <select v-model.number="selectedSample" data-testid="sample-select" @change="loadSample">
              <option v-for="(sample, index) in jokeSamples" :key="sample.id ?? index" :value="index">
                {{ sample.id ?? `sample-${index + 1}` }}
              </option>
            </select>
          </label>
          <label>
            Lang
            <select v-model="qaDraft.lang">
              <option value="ru">ru</option>
              <option value="en">en</option>
            </select>
          </label>
          <label>
            Session
            <input v-model="qaDraft.sessionId" placeholder="default" />
          </label>
          <label>
            Plane
            <input v-model="qaDraft.planeId" placeholder="auto" />
          </label>
          <label>
            Epochs
            <input v-model.number="qaDraft.epochs" type="number" min="1" />
          </label>
          <label>
            Reward
            <input v-model.number="qaDraft.reward" type="number" min="0.1" step="0.1" />
          </label>
          <label class="wide">
            Вопрос
            <textarea v-model="qaDraft.question" rows="3" placeholder="расскажи анекдот"></textarea>
          </label>
          <label class="wide">
            Ожидаемый ответ
            <textarea
              v-model="qaDraft.expectedAnswer"
              rows="4"
              placeholder="Короткий анекдот для обучения."
            ></textarea>
          </label>
        </div>

        <div class="row advanced-toggle">
          <button type="button" @click="showAdvanced = !showAdvanced">
            {{ showAdvanced ? 'Скрыть advanced' : 'Показать advanced' }}
          </button>
          <span class="muted">Нужен только если хотите вручную переопределить слоты, роли или плоскости.</span>
        </div>

        <section v-if="showAdvanced" class="annotation-block">
          <div class="row preview-head">
            <div>
              <h3>Разметка ответа</h3>
              <p class="muted">Плоскостей может быть несколько: `language:ru, dev:filesystem`.</p>
            </div>
            <div class="row">
              <button type="button" @click="syncAnnotations(true)">Пересобрать</button>
              <button type="button" @click="addAnnotation">+</button>
            </div>
          </div>

          <article
            v-for="(annotation, index) in qaDraft.annotations"
            :key="annotation.key"
            class="annotation-row"
            data-testid="qa-annotation"
          >
            <label>
              Token
              <input v-model="annotation.token" />
            </label>
            <label>
              Lemma
              <input v-model="annotation.lemma" />
            </label>
            <label>
              Role
              <select v-model="annotation.role">
                <option value="subject">subject</option>
                <option value="predicate">predicate</option>
                <option value="object">object</option>
                <option value="instrument">instrument</option>
                <option value="location">location</option>
                <option value="modifier">modifier</option>
              </select>
            </label>
            <label>
              POS
              <select v-model="annotation.pos">
                <option value="NOUN">NOUN</option>
                <option value="VERB">VERB</option>
                <option value="ADJ">ADJ</option>
              </select>
            </label>
            <label>
              Concept URI
              <input v-model="annotation.concept" placeholder="/c/ru/дерево" />
            </label>
            <label>
              Planes
              <input v-model="annotation.planes" placeholder="language:ru, dev:filesystem" />
            </label>
            <label>
              Gram JSON
              <input v-model="annotation.gramJson" placeholder='{"case":"nomn","number":"sing"}' />
            </label>
            <label>
              Prep
              <input v-model="annotation.preposition" placeholder="на" />
            </label>
            <button type="button" @click="removeAnnotation(index)">Удалить</button>
          </article>
        </section>

        <section class="preview-card">
          <div class="row preview-head">
            <h3>Preview</h3>
            <span class="badge">resonance QA</span>
          </div>
          <dl class="qa-preview">
            <dt>sample</dt>
            <dd>{{ currentSampleLabel }}</dd>
            <dt>question</dt>
            <dd>{{ qaDraft.question || '—' }}</dd>
            <dt>expected answer</dt>
            <dd>{{ qaDraft.expectedAnswer || '—' }}</dd>
            <dt>answer tokens</dt>
            <dd>{{ preview.answerTokens.join(', ') || '—' }}</dd>
            <dt>lemmas</dt>
            <dd>{{ preview.lemmas.join(', ') || '—' }}</dd>
            <dt>roles</dt>
            <dd>{{ preview.roles.join(', ') || '—' }}</dd>
            <dt>planes</dt>
            <dd>{{ preview.planes.join(', ') || 'auto' }}</dd>
          </dl>
        </section>

        <div class="row run-row">
          <button
            class="primary"
            type="button"
            data-testid="qa-train-button"
            :disabled="runtime.loading"
            @click="submit"
          >
            Train question-answer
          </button>
          <span class="muted">Отправляет пример через `/api/resonance/train-qa` без обязательной ручной разметки.</span>
        </div>
      </section>

      <JobPanel />
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue';
import { useRuntimeStore } from '@/app/stores/runtime';
import JobPanel from '@/features/job-panel/ui/JobPanel.vue';
import { api } from '@/shared/api/client';

import jokeSamples from './fixtures/jokes.json';

type JokeSample = {
  id?: string;
  question: string;
  expected_answer: string;
  lang?: string;
};

type AnnotationDraft = {
  key: string;
  index: number;
  token: string;
  surface: string;
  lemma: string;
  role: string;
  pos: string;
  concept: string;
  planes: string;
  gramJson: string;
  preposition: string;
};

const runtime = useRuntimeStore();
const jokeList = jokeSamples as JokeSample[];
const firstSample = jokeList[0] ?? {
  question: 'расскажи анекдот',
  expected_answer: 'Короткий анекдот для проверки.',
  lang: 'ru',
};

const qaDraft = reactive({
  question: firstSample.question,
  expectedAnswer: firstSample.expected_answer,
  lang: firstSample.lang ?? 'ru',
  sessionId: 'default',
  planeId: '',
  epochs: 1,
  reward: 1,
  annotations: [] as AnnotationDraft[],
});
const selectedSample = ref(0);
const showAdvanced = ref(false);
const RU_PREPOSITIONS = new Set(['в', 'во', 'на', 'с', 'со', 'к', 'ко', 'по', 'о', 'об', 'при', 'из', 'за']);
const VERB_SURFACES = new Set(['пишет', 'растет', 'растёт', 'растут', 'делает']);
const VERB_LEMMAS = new Set(['писать', 'расти', 'делать']);
const LEMMA_OVERRIDES: Record<string, string> = {
  деревья: 'дерево',
  деревом: 'дерево',
  деревьями: 'дерево',
  столы: 'стол',
  столом: 'стол',
  столами: 'стол',
  машины: 'машина',
  машиной: 'машина',
  машинами: 'машина',
  пишет: 'писать',
  растет: 'расти',
  растёт: 'расти',
  растут: 'расти',
  делает: 'делать',
  компьютере: 'компьютер',
};
const GRAM_OVERRIDES: Record<string, Record<string, string>> = {
  деревья: { case: 'nomn', number: 'plur', gender: 'neut' },
  деревом: { case: 'ablt', number: 'sing', gender: 'neut' },
  деревьями: { case: 'ablt', number: 'plur', gender: 'neut' },
  пишет: { tense: 'pres', person: '3', number: 'sing' },
  растет: { tense: 'pres', person: '3', number: 'sing' },
  растёт: { tense: 'pres', person: '3', number: 'sing' },
  растут: { tense: 'pres', person: '3', number: 'plur' },
  делает: { tense: 'pres', person: '3', number: 'sing' },
  компьютере: { case: 'loct', number: 'sing', gender: 'masc' },
};

const preview = computed(() => {
  const planes = qaDraft.annotations.flatMap((item) => splitPlanes(item.planes));
  return {
    answerTokens: tokenizeText(qaDraft.expectedAnswer),
    lemmas: qaDraft.annotations.map((item) => item.lemma).filter(Boolean),
    roles: qaDraft.annotations.map((item) => item.role).filter(Boolean),
    planes: Array.from(new Set(planes)),
  };
});
const currentSampleLabel = computed(() => {
  const sample = jokeList[selectedSample.value];
  if (!sample) {
    return 'manual';
  }
  return sample.id ?? `sample-${selectedSample.value + 1}`;
});

watch(
  () => qaDraft.expectedAnswer,
  () => {
    if (showAdvanced.value) {
      syncAnnotations(false);
    }
  },
);

watch(showAdvanced, (value) => {
  if (value) {
    syncAnnotations(false);
  }
});

syncAnnotations(false);

async function submit() {
  const job = await api.resonanceTrainQa({
    question: qaDraft.question,
    expected_answer: qaDraft.expectedAnswer,
    lang: qaDraft.lang,
    session_id: qaDraft.sessionId || 'default',
    plane_id: qaDraft.planeId.trim() || undefined,
    reward: qaDraft.reward,
    epochs: qaDraft.epochs,
    annotations: showAdvanced.value
      ? qaDraft.annotations.map((item) => ({
          index: item.index,
          token: item.token,
          surface: item.surface || item.token,
          lemma: item.lemma,
          role: item.role,
          pos: item.pos,
          concept: item.concept.trim() || undefined,
          planes: splitPlanes(item.planes),
          gram: parseJsonObject(item.gramJson),
          preposition: item.preposition.trim() || undefined,
        }))
      : [],
  });
  await runtime.trackJob(job);
}

async function resetNetwork() {
  clearRuntimeView();
  const job = await api.resetNetwork({ keep_builtin: false });
  await runtime.trackJob(job);
  clearRuntimeView();
}

async function seedResonance() {
  const job = await api.resonanceSeed({ force: true, session_id: qaDraft.sessionId || 'default' });
  await runtime.trackJob(job);
}

function loadSample() {
  const sample = jokeList[selectedSample.value];
  if (!sample) {
    return;
  }
  qaDraft.question = sample.question;
  qaDraft.expectedAnswer = sample.expected_answer;
  qaDraft.lang = sample.lang ?? qaDraft.lang;
  qaDraft.annotations = [];
  showAdvanced.value = false;
}

function clearRuntimeView() {
  runtime.graph = null;
  runtime.lastAnalysis = null;
  runtime.lastResult = null;
}

function syncAnnotations(force: boolean) {
  const existing = new Map(qaDraft.annotations.map((item) => [`${item.index}:${item.token}`, item]));
  const tokens = tokenizeText(qaDraft.expectedAnswer);
  const predicateIndex = tokens.findIndex((token) => VERB_SURFACES.has(token));
  const nextRows: AnnotationDraft[] = [];
  let pendingPreposition = '';
  tokens.forEach((token, index) => {
    if (RU_PREPOSITIONS.has(token)) {
      pendingPreposition = token;
      return;
    }
    const previous = force ? undefined : existing.get(`${index}:${token}`);
    const fallback = defaultAnnotation(token, index, predicateIndex, pendingPreposition);
    nextRows.push(previous ? { ...fallback, ...previous, index, token } : fallback);
    pendingPreposition = '';
  });
  qaDraft.annotations = nextRows;
}

function addAnnotation() {
  const index = qaDraft.annotations.length;
  qaDraft.annotations.push({
    key: `manual-${Date.now()}-${index}`,
    index,
    token: '',
    surface: '',
    lemma: '',
    role: 'modifier',
    pos: 'NOUN',
    concept: '',
    planes: '',
    gramJson: '{}',
    preposition: '',
  });
}

function removeAnnotation(index: number) {
  qaDraft.annotations.splice(index, 1);
}

function defaultAnnotation(token: string, index: number, predicateIndex: number, preposition: string): AnnotationDraft {
  const lemma = LEMMA_OVERRIDES[token] ?? token;
  const pos = VERB_SURFACES.has(token) || VERB_LEMMAS.has(lemma) ? 'VERB' : 'NOUN';
  const role = inferRole(index, predicateIndex, pos, preposition);
  return {
    key: `token-${index}-${token}`,
    index,
    token,
    surface: token,
    lemma,
    role,
    pos,
    concept: conceptUriForLemma(lemma, qaDraft.lang),
    planes: '',
    gramJson: JSON.stringify(defaultGram(token, role)),
    preposition,
  };
}

function inferRole(index: number, predicateIndex: number, pos: string, preposition: string): string {
  if (pos === 'VERB') {
    return 'predicate';
  }
  if (['на', 'в', 'о', 'об', 'при'].includes(preposition)) {
    return 'instrument';
  }
  if (predicateIndex >= 0 && index < predicateIndex) {
    return 'subject';
  }
  if (predicateIndex >= 0 && index > predicateIndex) {
    return 'object';
  }
  return 'subject';
}

function defaultGram(token: string, role: string): Record<string, string> {
  const seeded = GRAM_OVERRIDES[token] ?? {};
  if (role === 'subject') {
    return { ...seeded, case: 'nomn' };
  }
  if (role === 'object') {
    return { ...seeded, case: 'accs' };
  }
  if (role === 'instrument' || role === 'location') {
    return { ...seeded, case: 'loct' };
  }
  if (role === 'predicate') {
    return { tense: 'pres', person: '3', ...seeded };
  }
  return seeded;
}

function conceptUriForLemma(lemma: string, lang: string): string {
  const cleaned = normalizeText(lemma);
  const slug = cleaned.toLowerCase().replace(/[^\p{Letter}\p{Number}]+/gu, '_').replace(/^_+|_+$/g, '');
  return `/c/${lang}/${slug || 'concept'}`;
}

function splitPlanes(value: string): string[] {
  return value
    .replace(/;/g, ',')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseJsonObject(value: string): Record<string, string> {
  if (!value.trim()) {
    return {};
  }
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, string>) : {};
  } catch {
    return {};
  }
}

function normalizeText(value: string): string {
  return value.replace(/\s+/g, ' ').trim();
}

function tokenizeText(value: string): string[] {
  return normalizeText(value).match(/[0-9A-Za-zА-Яа-яЁё]+/gu) ?? [];
}
</script>

<style scoped lang="scss">
.training-layout {
  align-items: start;
}

.training-panel {
  display: grid;
  gap: 18px;
}

.training-header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: start;
}

.header-actions {
  flex-wrap: wrap;
  justify-content: flex-end;
}

.advanced-toggle {
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

h2,
h3 {
  margin-top: 0;
}

.annotation-block {
  display: grid;
  gap: 12px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 14px;
}

.annotation-row {
  display: grid;
  gap: 10px;
  border-top: 1px solid var(--line);
  padding-top: 10px;
}

.qa-preview {
  display: grid;
  grid-template-columns: minmax(120px, 180px) minmax(0, 1fr);
  gap: 10px 14px;
  margin: 0;
}

.qa-preview dt {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.qa-preview dd {
  margin: 0;
  min-width: 0;
  word-break: break-word;
}

.danger {
  border-color: rgba(185, 28, 28, 0.35);
  color: #8f1d1d;
}

@media (max-width: 920px) {
  .training-header,
  .advanced-toggle {
    flex-direction: column;
  }

  .header-actions {
    justify-content: flex-start;
  }

  .qa-preview {
    grid-template-columns: 1fr;
  }
}
</style>
