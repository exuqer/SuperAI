<template>
  <section class="page training-page">
    <div class="split training-layout">
      <section class="panel training-panel">
        <div class="training-header">
          <div>
            <p class="eyebrow">Конструктор примера</p>
            <h2>Вопрос → ожидаемый ответ</h2>
            <p class="muted">
              Собирает канонический JSONL для `POST /api/training/learn` без ручного редактирования.
            </p>
          </div>
          <div class="row">
            <span class="badge">mode learn</span>
            <span class="badge">layers 0 / 1 / 2</span>
          </div>
        </div>

        <div class="toolbar training-toolbar">
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

        <details class="layers-block" open>
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

        <section class="preview-card">
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
          <span class="muted">Отправляет preview через `/api/training/learn`.</span>
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
import type { Job } from '@/shared/api/types';
import {
  buildTrainingExampleJsonl,
  createDefaultTrainingDraft,
  parseTrainingStrengthVector,
  resolveLayerTarget,
  TRAINING_TOP_DOMAINS,
  type TrainingExampleDraft,
} from './model/training-builder';

const runtime = useRuntimeStore();
const draft = reactive<TrainingExampleDraft>(createDefaultTrainingDraft());
const strengthVectorInput = ref('3, 8, 8');
const epochs = ref(1);
const topDomainOptions = TRAINING_TOP_DOMAINS;

watch(
  strengthVectorInput,
  (value) => {
    draft.strengthVector = parseTrainingStrengthVector(value, draft.strengthVector);
  },
  { immediate: true },
);

const resolvedLayers = computed(() => draft.layers.map((layer) => resolveLayerTarget(draft.lang, layer)));
const previewJsonl = computed(() => buildTrainingExampleJsonl(draft));

async function submit() {
  const job: Job = await api.learn({
    jsonl: previewJsonl.value,
    epochs: epochs.value,
  });
  await runtime.trackJob(job);
}
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

.run-row {
  justify-content: space-between;
}

@media (max-width: 980px) {
  .training-toolbar .wide {
    grid-column: auto;
  }
}
</style>
