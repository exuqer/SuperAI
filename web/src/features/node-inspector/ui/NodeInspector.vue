<template>
  <section class="panel inspector">
    <template v-if="node">
      <div class="row">
        <h3>{{ node.label }}</h3>
        <span class="badge">layer {{ node.layer }}</span>
        <span v-for="layer in node.layers" :key="`layer-${layer}`" class="badge">plane {{ layer }}</span>
        <span v-for="layer in node.active_layers" :key="`active-${layer}`" class="badge signal">active {{ layer }}</span>
        <span v-if="node.signal.active" class="badge signal">signal {{ node.signal.count }}</span>
      </div>
      <p class="uri">{{ node.uri }}</p>
      <div class="metrics">
        <span>degree: {{ node.degree }}</span>
        <span>pheromone: {{ node.concept_pheromone.toFixed(3) }}</span>
        <span>suppression: {{ node.suppression.toFixed(3) }}</span>
      </div>
      <div v-if="detail">
        <h4>Связи</h4>
        <div class="edge-list">
          <button
            v-for="edge in [...detail.outgoing, ...detail.incoming].slice(0, 80)"
            :key="edge.id"
            type="button"
            @click="$emit('selectEdge', edge)"
          >
            {{ edge.start === node.uri ? '→' : '←' }} {{ edge.relation }}
            <span class="muted">{{ edge.start === node.uri ? edge.end : edge.start }}</span>
          </button>
        </div>
        <h4>Aliases</h4>
        <p class="muted">{{ detail.aliases.join(', ') || 'нет' }}</p>
      </div>
      <h4>Metadata</h4>
      <pre>{{ compactJson(node.metadata) }}</pre>
    </template>
    <template v-else-if="edge">
      <div class="row">
        <h3>{{ edge.relation }}</h3>
        <span class="badge">layer {{ edge.layer }}</span>
        <span v-if="edge.from_layer !== undefined && edge.from_layer !== null" class="badge">from {{ edge.from_layer }}</span>
        <span v-if="edge.to_layer !== undefined && edge.to_layer !== null" class="badge">to {{ edge.to_layer }}</span>
        <span v-if="edge.signal.active" class="badge signal">signal {{ edge.signal.score.toFixed(3) }}</span>
      </div>
      <p class="uri">{{ edge.start }} → {{ edge.end }}</p>
      <div class="metrics">
        <span>weight: {{ edge.weight.toFixed(3) }}</span>
        <span>pheromone: {{ edge.pheromone.toFixed(3) }}</span>
        <span>layer pheromone: {{ (edge.layer_pheromone ?? 1).toFixed(3) }}</span>
        <span>distance: {{ edge.distance.toFixed(3) }}</span>
        <span>{{ edge.edge_type }}</span>
        <span v-if="edge.context_plane">context: {{ edge.context_plane }}</span>
      </div>
      <h4>Route stats</h4>
      <pre>{{ compactJson(edge.route_stats) }}</pre>
      <h4>Metadata</h4>
      <pre>{{ compactJson(edge.metadata) }}</pre>
    </template>
    <template v-else>
      <h3>Inspector</h3>
      <p class="muted">Выберите узел или связь на графе.</p>
    </template>
  </section>
</template>

<script setup lang="ts">
import type { ConceptDetail, GraphEdge, GraphNode } from '@/shared/api/types';
import { compactJson } from '@/shared/lib/format';

defineProps<{
  node: GraphNode | null;
  edge: GraphEdge | null;
  detail: ConceptDetail | null;
}>();

defineEmits<{
  selectEdge: [edge: GraphEdge];
}>();
</script>

<style scoped lang="scss">
.inspector {
  display: grid;
  gap: 10px;
  align-content: start;
}

h3,
h4,
p {
  margin: 0;
}

h4 {
  margin-top: 8px;
}

.uri {
  word-break: break-word;
  color: var(--muted);
  font-size: 13px;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 6px;
  color: var(--muted);
  font-size: 13px;
}

.edge-list {
  display: grid;
  gap: 6px;
  max-height: 260px;
  overflow: auto;
}

.edge-list button {
  justify-content: start;
  text-align: left;
}
</style>
