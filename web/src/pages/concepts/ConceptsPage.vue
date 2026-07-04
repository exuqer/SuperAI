<template>
  <section class="page">
    <div class="panel">
      <div class="toolbar">
        <label>
          Поиск понятия
          <input v-model="query" placeholder="яблоко, object, /m/top" @keydown.enter="load" />
        </label>
        <label>
          Layer
          <input v-model="layer" placeholder="0" />
        </label>
        <label>
          Limit
          <input v-model.number="limit" type="number" min="1" max="1000" />
        </label>
        <button class="primary" type="button" @click="load">Искать</button>
      </div>
    </div>

    <div class="split">
      <section class="panel">
        <div class="grid-list">
          <button v-for="item in concepts" :key="String(item.uri)" type="button" class="concept" @click="open(item)">
            <strong>{{ item.label }}</strong>
            <span class="muted">{{ item.uri }}</span>
            <span class="badge">layer {{ item.layer }}</span>
          </button>
        </div>
      </section>
      <NodeInspector :node="detail?.node ?? null" :edge="null" :detail="detail" @select-edge="selectedEdge = $event" />
    </div>
    <section v-if="selectedEdge" class="panel">
      <h3>{{ selectedEdge.relation }}</h3>
      <p class="muted">{{ selectedEdge.start }} → {{ selectedEdge.end }}</p>
    </section>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import NodeInspector from '@/features/node-inspector/ui/NodeInspector.vue';
import { api } from '@/shared/api/client';
import type { ConceptDetail, GraphEdge } from '@/shared/api/types';

const query = ref('');
const layer = ref('');
const limit = ref(200);
const concepts = ref<Array<Record<string, unknown>>>([]);
const detail = ref<ConceptDetail | null>(null);
const selectedEdge = ref<GraphEdge | null>(null);

async function load() {
  concepts.value = (await api.getConcepts({
    query: query.value,
    layer: layer.value || undefined,
    limit: limit.value,
  })) as Array<Record<string, unknown>>;
}

async function open(item: Record<string, unknown>) {
  detail.value = await api.getConceptDetail(String(item.uri));
  selectedEdge.value = null;
}

onMounted(load);
</script>

<style scoped lang="scss">
.concept {
  display: grid;
  gap: 4px;
  justify-items: start;
  text-align: left;
}

h3,
p {
  margin-top: 0;
}
</style>
