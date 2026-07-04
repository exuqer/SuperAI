<template>
  <section class="page">
    <div class="split">
      <section class="panel">
        <h2>Слои</h2>
        <div class="toolbar">
          <label>
            Stimulus
            <input v-model="stimulus" placeholder="яблоко" />
          </label>
          <label>
            Lang
            <select v-model="lang">
              <option>ru</option>
              <option>en</option>
              <option>auto</option>
            </select>
          </label>
          <label>
            Strength vector
            <input v-model="strengthVector" placeholder="3,8" />
          </label>
          <label>
            Layer target
            <input v-model="target" placeholder="/m/top/object" />
          </label>
          <label>
            Label
            <input v-model="label" placeholder="предмет" />
          </label>
          <button class="primary" type="button" @click="analyze">Analyze</button>
        </div>
        <h3>JSONL для обучения слоя</h3>
        <pre>{{ example }}</pre>
      </section>
      <section class="panel">
        <div class="row">
          <h2>Top domains</h2>
          <button type="button" @click="loadTopDomains">Обновить</button>
        </div>
        <div class="grid-list">
          <article v-for="item in topDomains" :key="String(item.uri)" class="domain">
            <strong>{{ item.label }}</strong>
            <span class="muted">{{ item.uri }}</span>
            <span class="badge">pheromone {{ Number(item.concept_pheromone).toFixed(2) }}</span>
          </article>
        </div>
      </section>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { useRuntimeStore } from '@/app/stores/runtime';
import { api } from '@/shared/api/client';
import { parseStrengthVector } from '@/shared/lib/format';

const runtime = useRuntimeStore();
const stimulus = ref('яблоко');
const lang = ref('ru');
const strengthVector = ref('3');
const target = ref('/m/top/object');
const label = ref('предмет');
const topDomains = ref<Array<Record<string, unknown>>>([]);

const example = computed(() =>
  JSON.stringify(
    {
      stimulus: stimulus.value,
      lang: lang.value,
      strength_vector: parseStrengthVector(strengthVector.value),
      layer_targets: { '0': [target.value] },
      target_concepts: [target.value],
      concept_labels: { [target.value]: label.value },
      positive_edges: [[`/c/${lang.value}/${stimulus.value}`, 'InTopDomain', target.value]],
      accepted_answer: `${stimulus.value} относится к области: ${label.value}.`,
    },
    null,
    0,
  ),
);

async function analyze() {
  await runtime.analyze({
    text: stimulus.value,
    lang: lang.value,
    strength_vector: parseStrengthVector(strengthVector.value),
  });
}

async function loadTopDomains() {
  topDomains.value = (await api.getConcepts({ layer: 0, limit: 200 })) as Array<Record<string, unknown>>;
}

onMounted(loadTopDomains);
</script>

<style scoped lang="scss">
h2,
h3 {
  margin-top: 0;
}

.domain {
  display: grid;
  gap: 4px;
  border-top: 1px solid var(--line);
  padding-top: 8px;
}
</style>
