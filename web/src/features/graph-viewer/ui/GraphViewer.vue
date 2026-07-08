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
  const areaNodes = collectAreaNodes(viewport.nodes);
  const elements: cytoscape.ElementDefinition[] = [
    ...areaNodes.map((area) => ({
      group: 'nodes' as const,
      data: {
        id: area.id,
        label: area.label,
        layer: -1,
        pheromone: 1,
      },
      classes: 'area',
    })),
    ...viewport.nodes.map((node) => ({
      group: 'nodes' as const,
      data: {
        id: node.id,
        parent: firstAreaId(node) ?? undefined,
        label: node.label || node.uri,
        layer: node.active_layers[0] ?? node.layers[0] ?? node.layer,
        layers: node.layers.join(','),
        activeLayers: node.active_layers.join(','),
        signal: node.signal.active,
        pheromone: node.concept_pheromone,
      },
      classes: [node.signal.active ? 'signal' : '', ...node.layers.map((layer) => `layer-${layer}`)].filter(Boolean).join(' '),
    })),
    ...viewport.edges.map((edge) => ({
      group: 'edges' as const,
      data: {
        id: edge.id,
        source: edge.start,
        target: edge.end,
        label: `${edge.relation} d=${formatDistance(edge.distance)}`,
        signal: edge.signal.active,
        pheromone: edge.pheromone,
        distance: edge.distance,
        edgeLength: edgeLength(edge.distance),
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
          width: 'mapData(layer, 0, 5, 74, 24)',
          height: 'mapData(layer, 0, 5, 74, 24)',
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
          'z-index': 10,
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
        selector: 'node.area',
        style: {
          width: 120,
          height: 80,
          padding: '28px',
          'background-color': 'rgba(23, 107, 87, 0.06)',
          'border-color': 'rgba(23, 107, 87, 0.28)',
          'border-width': '2',
          'border-style': 'dashed',
          label: 'data(label)',
          color: '#0f4f43',
          'font-size': '12',
          'font-weight': 600,
          'text-valign': 'top',
          'text-halign': 'center',
          'text-margin-y': -8,
          'z-index': 0,
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
      nodeRepulsion: 11000,
      idealEdgeLength: (edge: cytoscape.EdgeSingular) => Number(edge.data('edgeLength')) || 120,
      edgeElasticity: (edge: cytoscape.EdgeSingular) => Math.max(0.08, 0.55 / Math.max(Number(edge.data('distance')) || 1, 0.08)),
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

type AreaView = {
  id: string;
  label: string;
};

function collectAreaNodes(nodes: GraphNode[]): AreaView[] {
  const areas = new Map<string, AreaView>();
  for (const node of nodes) {
    const ids = metadataList(node.metadata.area_ids);
    const labels = metadataList(node.metadata.area_labels);
    const id = ids[0];
    if (!id || areas.has(id)) continue;
    areas.set(id, { id: areaElementId(id), label: labels[0] || id.replace(/^area:/, '') });
  }
  return [...areas.values()];
}

function firstAreaId(node: GraphNode): string | null {
  const id = metadataList(node.metadata.area_ids)[0];
  return id ? areaElementId(id) : null;
}

function areaElementId(areaId: string): string {
  return `area::${areaId}`;
}

function metadataList(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item)).filter(Boolean) : [];
}

function edgeLength(distance: number): number {
  const clean = Number.isFinite(distance) ? Math.max(distance, 0.03) : 1;
  return Math.min(460, Math.max(38, 42 + clean * 95));
}

function formatDistance(distance: number): string {
  return Number.isFinite(distance) ? distance.toFixed(distance < 1 ? 2 : 1) : '?';
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
