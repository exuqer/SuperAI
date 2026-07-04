<template>
  <section class="page">
    <div class="row">
      <button class="primary" type="button" @click="load">Обновить память</button>
    </div>
    <div class="split">
      <section class="panel">
        <h2>Summary</h2>
        <pre>{{ compactJson(summary) }}</pre>
      </section>
      <section class="panel">
        <h2>Collections</h2>
        <div class="toolbar">
          <label>
            Collection
            <select v-model="selected">
              <option>accepted_answers</option>
              <option>negative_memory</option>
              <option>response_memory</option>
              <option>experiences</option>
              <option>aliases</option>
              <option>suppressed_concepts</option>
            </select>
          </label>
        </div>
        <pre>{{ compactJson(collections[selected]) }}</pre>
      </section>
    </div>
    <section class="panel">
      <h2>Last results</h2>
      <div class="grid-list">
        <article v-for="item in results.slice(0, 20)" :key="String(item.result_id)" class="result">
          <strong>{{ item.input_text }}</strong>
          <span class="muted">{{ item.result_id }}</span>
          <p>{{ item.response }}</p>
        </article>
      </div>
    </section>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { api } from '@/shared/api/client';
import { compactJson } from '@/shared/lib/format';

const summary = ref<Record<string, unknown>>({});
const collections = ref<Record<string, unknown>>({});
const results = ref<Array<Record<string, unknown>>>([]);
const selected = ref('accepted_answers');

async function load() {
  summary.value = await api.getMemorySummary();
  collections.value = await api.getMemoryCollections();
  results.value = (await api.getMemoryResults()) as Array<Record<string, unknown>>;
}

onMounted(load);
</script>

<style scoped lang="scss">
h2,
p {
  margin-top: 0;
}

.result {
  display: grid;
  gap: 5px;
  border-top: 1px solid var(--line);
  padding-top: 10px;
}
</style>
