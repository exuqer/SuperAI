<template>
  <section class="page">
    <div class="panel">
      <div class="toolbar">
        <label>
          Поиск
          <input v-model="filters.query" placeholder="/m/top/object, яблоко, InTopDomain" />
        </label>
        <label>
          Layer
          <input v-model="filters.layer" placeholder="0" />
        </label>
        <label>
          Relation
          <input v-model="filters.relation" placeholder="InTopDomain" />
        </label>
        <label>
          Edge type
          <input v-model="filters.edge_type" placeholder="domain" />
        </label>
        <label>
          Min pheromone
          <input v-model="filters.min_pheromone" placeholder="1.0" />
        </label>
        <label>
          Limit
          <input v-model.number="filters.limit" type="number" min="1" max="5000" />
        </label>
        <label class="checkbox">
          <input v-model="filters.only_signal" type="checkbox" />
          Только сигнал
        </label>
        <button class="primary" type="button" @click="load">Загрузить граф</button>
      </div>
    </div>

    <div v-if="runtime.graph" class="row">
      <span class="badge">nodes {{ runtime.graph.stats.nodes }}</span>
      <span class="badge">edges {{ runtime.graph.stats.edges }}</span>
      <span class="badge signal">signal edges {{ runtime.graph.stats.signal_edges }}</span>
    </div>

    <div class="split">
      <GraphViewer :graph="runtime.graph" @select-node="selectNode" @select-edge="selectEdge" />
      <NodeInspector :node="selectedNode" :edge="selectedEdge" :detail="detail" @select-edge="selectedEdge = $event" />
    </div>
  </section>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref, watch } from 'vue';
import { useRuntimeStore } from '@/app/stores/runtime';
import GraphViewer from '@/features/graph-viewer/ui/GraphViewer.vue';
import NodeInspector from '@/features/node-inspector/ui/NodeInspector.vue';
import { api } from '@/shared/api/client';
import type { ConceptDetail, GraphEdge, GraphNode } from '@/shared/api/types';

const runtime = useRuntimeStore();
const filters = reactive({
  query: '',
  layer: '',
  relation: '',
  edge_type: '',
  min_pheromone: '',
  only_signal: false,
  limit: 800,
});
const selectedNode = ref<GraphNode | null>(null);
const selectedEdge = ref<GraphEdge | null>(null);
const detail = ref<ConceptDetail | null>(null);

async function load() {
  await runtime.loadGraph({
    ...filters,
    layer: filters.layer || undefined,
    min_pheromone: filters.min_pheromone || undefined,
    result_id: runtime.lastResult?.result_id,
  });
}

async function selectNode(node: GraphNode) {
  selectedNode.value = node;
  selectedEdge.value = null;
  detail.value = await api.getConceptDetail(node.uri, runtime.lastResult?.result_id);
}

function selectEdge(edge: GraphEdge) {
  selectedEdge.value = edge;
  selectedNode.value = null;
  detail.value = null;
}

onMounted(load);
watch(() => runtime.lastResult?.result_id, () => {
  if (runtime.graph) {
    load().catch(() => undefined);
  }
});
</script>

<style scoped lang="scss">
.checkbox {
  display: flex;
  gap: 8px;
  align-items: center;
}

.checkbox input {
  width: auto;
}
</style>
