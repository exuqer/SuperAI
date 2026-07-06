<template>
  <div class="graph-viewer">
    <div ref="container" class="graph-canvas" />
  </div>
</template>

<script setup lang="ts">
import cytoscape from 'cytoscape';
import fcose from 'cytoscape-fcose';
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import type { GraphEdge, GraphNode, GraphPayload } from '@/shared/api/types';
import { selectGraphViewport } from '../model/select-graph';

cytoscape.use(fcose);

const props = defineProps<{
  graph: GraphPayload | null;
  viewportLimit?: number;
}>();

const emit = defineEmits<{
  selectNode: [node: GraphNode];
  selectEdge: [edge: GraphEdge];
}>();

const container = ref<HTMLDivElement | null>(null);
let cy: cytoscape.Core | null = null;

function renderGraph() {
  if (!container.value || !props.graph) {
    cy?.destroy();
    cy = null;
    return;
  }
  const viewportLimit =
    typeof props.viewportLimit === 'number' && Number.isFinite(props.viewportLimit)
      ? props.viewportLimit
      : undefined;
  const viewport =
    viewportLimit == null
      ? selectGraphViewport(props.graph)
      : selectGraphViewport(props.graph, {
          focusedNodeLimit: viewportLimit,
          focusedEdgeLimit: viewportLimit,
          fallbackNodeLimit: viewportLimit,
          fallbackEdgeLimit: viewportLimit,
        });
  if (!viewport.nodes.length && !viewport.edges.length) {
    cy?.destroy();
    cy = null;
    return;
  }
  const nodeById = new Map(viewport.nodes.map((node) => [node.id, node]));
  const edgeById = new Map(viewport.edges.map((edge) => [edge.id, edge]));
  const elements: cytoscape.ElementDefinition[] = [
    ...viewport.nodes.map((node) => ({
      group: 'nodes' as const,
      data: {
        id: node.id,
        label: node.label || node.uri,
        layer: node.layer,
        signal: node.signal.active,
        pheromone: node.concept_pheromone,
      },
      classes: [node.signal.active ? 'signal' : '', `layer-${node.layer}`].filter(Boolean).join(' '),
    })),
    ...viewport.edges.map((edge) => ({
      group: 'edges' as const,
      data: {
        id: edge.id,
        source: edge.start,
        target: edge.end,
        label: edge.relation,
        signal: edge.signal.active,
        pheromone: edge.pheromone,
      },
      classes: edge.signal.active ? 'signal' : '',
    })),
  ];

  if (cy) {
    cy.destroy();
  }

  cy = cytoscape({
    container: container.value,
    elements,
    wheelSensitivity: 0.2,
    style: [
      {
        selector: 'node',
        style: {
          width: 'mapData(layer, 0, 3, 70, 22)',
          height: 'mapData(layer, 0, 3, 70, 22)',
          'background-color': '#d8dadd',
          'border-color': '#9ea4ad',
          'border-width': '1',
          label: 'data(label)',
          color: '#17191c',
          'font-size': '11',
          'text-wrap': 'wrap',
          'text-max-width': '92',
          'text-valign': 'bottom',
          'text-halign': 'center',
        },
      },
      {
        selector: 'node.signal',
        style: {
          'background-color': '#ffd8d8',
          'border-color': '#d72f2f',
          'border-width': '3',
        },
      },
      {
        selector: 'edge',
        style: {
          width: 'mapData(pheromone, 0, 10, 1, 4)',
          'line-color': '#20242a',
          'target-arrow-color': '#20242a',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          label: 'data(label)',
          'font-size': '9',
          color: '#5f6670',
          'text-background-color': '#ffffff',
          'text-background-opacity': 0.72,
          'text-background-padding': '2',
        },
      },
      {
        selector: 'edge.signal',
        style: {
          width: '4',
          'line-color': '#d72f2f',
          'target-arrow-color': '#d72f2f',
          color: '#d72f2f',
        },
      },
    ],
    layout: {
      name: 'fcose',
      animate: false,
      fit: true,
      padding: 30,
      nodeRepulsion: 9000,
      idealEdgeLength: 90,
    } as cytoscape.LayoutOptions,
  });

  cy.on('tap', 'node', (event) => {
    const node = nodeById.get(event.target.id());
    if (node) emit('selectNode', node);
  });
  cy.on('tap', 'edge', (event) => {
    const edge = edgeById.get(event.target.id());
    if (edge) emit('selectEdge', edge);
  });
}

onMounted(renderGraph);
watch(() => [props.graph, props.viewportLimit], renderGraph);

onBeforeUnmount(() => {
  cy?.destroy();
  cy = null;
});
</script>

<style scoped lang="scss">
.graph-viewer {
  min-height: 620px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  overflow: hidden;
  background: #fff;
}

.graph-canvas {
  width: 100%;
  height: 620px;
}
</style>
