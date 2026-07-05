<template>
  <section class="page training-page">
    <div class="split training-layout">
      <section class="panel training-panel">
        <div class="training-header">
          <div>
            <p class="eyebrow">Конструктор примера</p>
            <h2>Вопрос → ожидаемый ответ</h2>
            <p class="muted">
              Простой режим сам строит связи по токенам; расширенный оставляет ручной JSONL-конструктор.
            </p>
          </div>
          <div class="row">
            <button
              type="button"
              :class="{ primary: mode === 'simple' }"
              data-testid="simple-mode"
              @click="mode = 'simple'"
            >
              Простой
            </button>
            <button
              type="button"
              :class="{ primary: mode === 'advanced' }"
              @click="mode = 'advanced'"
            >
              Расширенный
            </button>
          </div>
        </div>

        <div v-if="mode === 'simple'" class="toolbar training-toolbar">
          <label>
            Lang
            <select v-model="simpleDraft.lang">
              <option value="auto">auto</option>
              <option value="ru">ru</option>
              <option value="en">en</option>
            </select>
          </label>
          <label>
            Epochs
            <input v-model.number="simpleDraft.epochs" type="number" min="1" />
          </label>
          <label>
            Reward
            <input v-model.number="simpleDraft.reward" type="number" min="0.1" step="0.1" />
          </label>
          <label class="wide">
            Вопрос
            <textarea v-model="simpleDraft.question" rows="3" placeholder="что делает программист?"></textarea>
          </label>
          <label class="wide">
            Ожидаемый ответ
            <textarea
              v-model="simpleDraft.expectedAnswer"
              rows="4"
              placeholder="Программист пишет код на компьютере."
            ></textarea>
          </label>
        </div>

        <section v-if="mode === 'simple'" class="meanings-block">
          <div class="row preview-head">
            <h3>Смыслы понятий</h3>
            <button type="button" @click="addMeaning">+</button>
          </div>
          <p class="muted">
            Строки строятся из токенов вопроса. Если смысл уже сохранен в памяти, поле заполнится автоматически.
          </p>
          <article v-for="(meaning, index) in simpleDraft.conceptMeanings" :key="meaning.key" class="meaning-row">
            <label>
              Токен
              <input v-model="meaning.token" placeholder="осень" />
            </label>
            <label>
              Concept URI
              <input v-model="meaning.concept" placeholder="/c/ru/программист" />
            </label>
            <label>
              Label
              <input v-model="meaning.label" placeholder="программист" />
            </label>
            <label class="wide">
              Meaning
              <textarea v-model="meaning.meaning" rows="2" placeholder="человек, который пишет код"></textarea>
            </label>
            <button type="button" @click="removeMeaning(index)">Удалить</button>
          </article>
        </section>

        <section v-if="mode === 'simple'" class="preview-card">
          <div class="row preview-head">
            <h3>Preview</h3>
            <span class="badge">simple summary</span>
          </div>
          <dl class="simple-preview">
            <dt>question tokens</dt>
            <dd>{{ simplePreview.questionTokens.join(', ') || '—' }}</dd>
            <dt>answer tokens</dt>
            <dd>{{ simplePreview.answerTokens.join(', ') || '—' }}</dd>
            <dt>meaning tokens</dt>
            <dd>{{ simplePreview.meaningTokens.join(', ') || '—' }}</dd>
            <dt>estimated edges</dt>
            <dd>{{ simplePreview.estimatedEdges }}</dd>
          </dl>
        </section>

        <div v-if="mode === 'advanced'" class="toolbar training-toolbar">
          <label>
            Lang
            <select v-model="draft.lang">
              <option value="ru">ru</option>
              <option value="en">en</option>
            </select>
          </label>
          <label class="wide">
            Вопрос
            <textarea v-model="draft.question" rows="3" placeholder="как дела?"></textarea>
          </label>
          <label class="wide">
            Ожидаемый ответ
            <textarea
              v-model="draft.expectedAnswer"
              rows="4"
              placeholder="Нормально, спасибо. А у тебя?"
            ></textarea>
          </label>
          <label>
            strength_vector
            <input v-model="strengthVectorInput" placeholder="3, 8, 8" />
          </label>
          <label>
            Epochs
            <input v-model.number="epochs" type="number" min="1" />
          </label>
        </div>

        <details v-if="mode === 'advanced'" class="layers-block" open>
          <summary>
            <span>Слои</span>
            <span class="muted">JSON layers 0, 1, 2</span>
          </summary>
          <div class="layers-grid">
            <article v-for="(layer, index) in draft.layers" :key="layer.level" class="layer-card">
              <div class="layer-card-head">
                <div>
                  <strong>Уровень {{ layer.level }}</strong>
                  <p class="muted">JSON layer {{ layer.level - 1 }}</p>
                </div>
                <span class="badge">{{ resolvedLayers[index].uri }}</span>
              </div>
              <label v-if="layer.level === 1">
                Builtin top domain
                <select v-model="layer.builtinTopDomain">
                  <option v-for="option in topDomainOptions" :key="option.key" :value="option.key">
                    {{ option.label }}
                  </option>
                </select>
              </label>
              <label v-else>
                Label
                <input
                  v-model="layer.label"
                  :placeholder="layer.level === 2 ? 'Вопрос' : 'дела'"
                />
              </label>
              <div class="layer-footnote">
                <span class="badge">{{ resolvedLayers[index].label }}</span>
                <span class="muted">{{ layer.level === 1 ? 'builtin top domain' : resolvedLayers[index].uri }}</span>
              </div>
            </article>
          </div>
        </details>

        <section v-if="mode === 'advanced'" class="preview-card">
          <div class="row preview-head">
            <h3>Preview</h3>
            <span class="badge">JSONL</span>
          </div>
          <pre>{{ previewJsonl }}</pre>
        </section>

        <div class="row run-row">
          <button class="primary" type="button" :disabled="runtime.loading" @click="submit">
            Запустить
          </button>
          <span class="muted">{{ submitHint }}</span>
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
import type { Job, UnderstandingToken } from '@/shared/api/types';
import {
  buildTrainingExampleJsonl,
  createDefaultTrainingDraft,
  parseTrainingStrengthVector,
  resolveLayerTarget,
  TRAINING_TOP_DOMAINS,
  type TrainingExampleDraft,
} from './model/training-builder';

const runtime = useRuntimeStore();
const mode = ref<'simple' | 'advanced'>('simple');
const draft = reactive<TrainingExampleDraft>(createDefaultTrainingDraft());
const simpleDraft = reactive({
  question: 'что делает программист?',
  expectedAnswer: 'Программист пишет код на компьютере.',
  lang: 'ru',
  epochs: 1,
  reward: 1,
  conceptMeanings: [] as SimpleMeaningDraft[],
});
const strengthVectorInput = ref('3, 8, 8');
const epochs = ref(1);
const topDomainOptions = TRAINING_TOP_DOMAINS;
let meaningSyncVersion = 0;

type SimpleMeaningDraft = {
  key: string;
  token: string;
  concept: string;
  label: string;
  meaning: string;
  auto: boolean;
};

watch(
  strengthVectorInput,
  (value) => {
    draft.strengthVector = parseTrainingStrengthVector(value, draft.strengthVector);
  },
  { immediate: true },
);

watch(
  () => [simpleDraft.question, simpleDraft.lang] as const,
  () => {
    void syncMeaningsFromQuestion();
  },
  { immediate: true },
);

const resolvedLayers = computed(() => draft.layers.map((layer) => resolveLayerTarget(draft.lang, layer)));
const previewJsonl = computed(() => buildTrainingExampleJsonl(draft));
const simplePreview = computed(() => {
  const questionTokens = simpleDraft.conceptMeanings.map((item) => item.token).filter(Boolean);
  const answerTokens = simpleTokenize(simpleDraft.expectedAnswer);
  const meaningTokens = simpleDraft.conceptMeanings.flatMap((item) => simpleTokenize(item.meaning));
  const estimatedEdges =
    questionTokens.length * answerTokens.length +
    Math.max(answerTokens.length - 1, 0) +
    questionTokens.length * meaningTokens.length +
    simpleDraft.conceptMeanings.filter((item) => item.concept || item.label).length * meaningTokens.length;
  return {
    questionTokens,
    answerTokens,
    meaningTokens: Array.from(new Set(meaningTokens)),
    estimatedEdges,
  };
});
const submitHint = computed(() =>
  mode.value === 'simple'
    ? 'Отправляет форму через `/api/training/simple`.'
    : 'Отправляет preview через `/api/training/learn`.',
);

async function submit() {
  const job: Job =
    mode.value === 'simple'
      ? await api.simpleTrain({
          question: simpleDraft.question,
          expected_answer: simpleDraft.expectedAnswer,
          lang: simpleDraft.lang,
          reward: simpleDraft.reward,
          epochs: simpleDraft.epochs,
          concept_meanings: simpleDraft.conceptMeanings
            .map((item) => ({
              concept: item.concept.trim() || undefined,
              label: item.label.trim() || undefined,
              meaning: item.meaning.trim(),
            }))
            .filter((item) => item.meaning),
        })
      : await api.learn({
          jsonl: previewJsonl.value,
          epochs: epochs.value,
        });
  await runtime.trackJob(job);
}

function addMeaning() {
  simpleDraft.conceptMeanings.push({
    key: `manual-${Date.now()}-${simpleDraft.conceptMeanings.length}`,
    token: '',
    concept: '',
    label: '',
    meaning: '',
    auto: false,
  });
}

function removeMeaning(index: number) {
  simpleDraft.conceptMeanings.splice(index, 1);
}

async function syncMeaningsFromQuestion() {
  const version = ++meaningSyncVersion;
  const question = simpleDraft.question.trim();
  if (!question) {
    simpleDraft.conceptMeanings = simpleDraft.conceptMeanings.filter((item) => !item.auto);
    return;
  }
  try {
    const understood = await api.understand({ text: question, lang: simpleDraft.lang });
    if (version !== meaningSyncVersion) {
      return;
    }
    const currentByKey = new Map(simpleDraft.conceptMeanings.map((item) => [item.key, item]));
    const tokenRows = understood.tokens
      .filter((token) => !token.is_stop_word && token.search_token)
      .map((token) => meaningDraftFromToken(token, understood.lang));
    const enriched = await Promise.all(
      tokenRows.map(async (row) => {
        const existing = currentByKey.get(row.key);
        const merged = existing ? { ...row, meaning: existing.meaning, label: existing.label || row.label } : row;
        if (merged.meaning || !merged.concept) {
          return merged;
        }
        try {
          const detail = await api.getConceptDetail(merged.concept);
          const metadata = detail.node.metadata ?? {};
          return {
            ...merged,
            label: String(metadata.label || merged.label),
            meaning: String(metadata.meaning || metadata.description || ''),
          };
        } catch {
          return merged;
        }
      }),
    );
    if (version !== meaningSyncVersion) {
      return;
    }
    const manual = simpleDraft.conceptMeanings.filter((item) => !item.auto);
    simpleDraft.conceptMeanings = [...enriched, ...manual];
  } catch {
    const fallbackRows = simpleTokenize(question).map((token) => ({
      key: `fallback:${token}`,
      token,
      concept: conceptUriForToken(token, simpleDraft.lang === 'auto' ? 'ru' : simpleDraft.lang),
      label: token,
      meaning: '',
      auto: true,
    }));
    simpleDraft.conceptMeanings = [...fallbackRows, ...simpleDraft.conceptMeanings.filter((item) => !item.auto)];
  }
}

function meaningDraftFromToken(token: UnderstandingToken, lang: string): SimpleMeaningDraft {
  const canonicalToken = token.match_status === 'partial_root_match' && token.lemma ? token.lemma : token.search_token;
  const concept = conceptUriForToken(canonicalToken, lang) || token.concept_uri || '';
  return {
    key: `token:${concept || canonicalToken}`,
    token: canonicalToken,
    concept,
    label: canonicalToken,
    meaning: '',
    auto: true,
  };
}

function conceptUriForToken(token: string, lang: string): string {
  const clean = token
    .toLowerCase()
    .normalize('NFKC')
    .replace(/['"`’]/g, '')
    .replace(/[^\p{Letter}\p{Number}]+/gu, '_')
    .replace(/^_+|_+$/g, '')
    .replace(/_+/g, '_');
  if (!clean) {
    return '';
  }
  return `/c/${lang === 'en' ? 'en' : 'ru'}/${clean}`;
}

function simpleTokenize(value: string): string[] {
  return value
    .toLowerCase()
    .match(/[0-9a-zа-яё]+(?:[-'][0-9a-zа-яё]+)?/giu)
    ?.filter((token) => !SIMPLE_STOP_WORDS.has(token)) ?? [];
}

const SIMPLE_STOP_WORDS = new Set([
  'а',
  'в',
  'и',
  'на',
  'с',
  'что',
  'который',
  'the',
  'a',
  'an',
  'and',
  'to',
  'of',
]);
</script>

<style scoped lang="scss">
.training-page {
  align-content: start;
}

.training-layout {
  align-items: start;
}

.training-panel {
  display: grid;
  gap: 18px;
}

.training-header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 16px;
}

.training-header h2,
.preview-head h3 {
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

.training-toolbar {
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
}

.training-toolbar .wide {
  grid-column: 1 / -1;
}

.layers-block {
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: linear-gradient(180deg, rgba(23, 107, 87, 0.04), rgba(23, 107, 87, 0.01));
  padding: 14px;
}

.meanings-block {
  display: grid;
  gap: 12px;
}

.meaning-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  align-items: end;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 12px;
}

.meaning-row .wide {
  grid-column: span 2;
}

.layers-block > summary {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  cursor: pointer;
  list-style: none;
  font-weight: 600;
}

.layers-block > summary::-webkit-details-marker {
  display: none;
}

.layers-grid {
  display: grid;
  gap: 12px;
  margin-top: 14px;
}

.layer-card {
  display: grid;
  gap: 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--surface);
  padding: 14px;
}

.layer-card-head,
.layer-footnote {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 8px;
  align-items: center;
}

.layer-card-head p,
.layer-footnote p {
  margin: 0;
}

.layer-card strong,
.preview-card h3 {
  font-size: 16px;
}

.preview-card {
  display: grid;
  gap: 8px;
}

.preview-head {
  justify-content: space-between;
}

.preview-card pre {
  min-height: 180px;
  margin: 0;
}

.simple-preview {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 8px 14px;
  margin: 0;
}

.simple-preview dt {
  color: var(--muted);
  font-weight: 600;
}

.simple-preview dd {
  margin: 0;
  overflow-wrap: anywhere;
}

.run-row {
  justify-content: space-between;
}

@media (max-width: 980px) {
  .training-toolbar .wide {
    grid-column: auto;
  }
}
</style>
