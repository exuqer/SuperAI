import type { GraphEdge, GraphNode, GraphPayload } from '@/shared/api/types';

export type GraphViewportOptions = {
  focusedNodeLimit?: number;
  focusedEdgeLimit?: number;
  fallbackNodeLimit?: number;
  fallbackEdgeLimit?: number;
};

export type GraphViewport = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  focused: boolean;
};

const DEFAULT_FOCUSED_EDGE_LIMIT = 240;
const DEFAULT_FOCUSED_NODE_LIMIT = 240;
const DEFAULT_FALLBACK_NODE_LIMIT = 240;
const DEFAULT_FALLBACK_EDGE_LIMIT = 360;

export function selectGraphViewport(graph: GraphPayload | null, options: GraphViewportOptions = {}): GraphViewport {
  if (!graph) {
    return { nodes: [], edges: [], focused: false };
  }

  if (
    options.focusedNodeLimit === 0 ||
    options.focusedEdgeLimit === 0 ||
    options.fallbackNodeLimit === 0 ||
    options.fallbackEdgeLimit === 0
  ) {
    return {
      nodes: [...graph.nodes].sort(compareFallbackNodes),
      edges: [...graph.edges].sort(compareFallbackEdges),
      focused: Boolean(graph.nodes.some((node) => node.signal.active) || graph.edges.some((edge) => edge.signal.active)),
    };
  }

  const focusedEdgeLimit = options.focusedEdgeLimit ?? DEFAULT_FOCUSED_EDGE_LIMIT;
  const focusedNodeLimit = options.focusedNodeLimit ?? DEFAULT_FOCUSED_NODE_LIMIT;
  const fallbackNodeLimit = options.fallbackNodeLimit ?? DEFAULT_FALLBACK_NODE_LIMIT;
  const fallbackEdgeLimit = options.fallbackEdgeLimit ?? DEFAULT_FALLBACK_EDGE_LIMIT;

  const activeNodeIds = new Set(graph.nodes.filter((node) => node.signal.active).map((node) => node.id));
  const activeEdgeIds = new Set(graph.edges.filter((edge) => edge.signal.active).map((edge) => edge.id));
  const requiredEdgeIds = new Set(activeEdgeIds);

  if (activeNodeIds.size || activeEdgeIds.size) {
    for (const edgeId of collectCoverageEdgeIds(graph.edges, activeNodeIds, activeEdgeIds)) {
      requiredEdgeIds.add(edgeId);
    }

    const seedNodeIds = new Set(activeNodeIds);
    for (const edge of graph.edges) {
      if (activeEdgeIds.has(edge.id)) {
        seedNodeIds.add(edge.start);
        seedNodeIds.add(edge.end);
      }
    }

    const edges = [...graph.edges]
      .filter(
        (edge) =>
          requiredEdgeIds.has(edge.id) || seedNodeIds.has(edge.start) || seedNodeIds.has(edge.end),
      )
      .sort((left, right) => compareFocusedEdges(left, right, activeNodeIds, activeEdgeIds));
    const requiredEdges = edges.filter((edge) => requiredEdgeIds.has(edge.id));
    const optionalEdges = edges.filter((edge) => !requiredEdgeIds.has(edge.id));
    const selectedEdgeIds = new Set<string>();
    const nodeIds = new Set(activeNodeIds);

    for (const edge of requiredEdges) {
      includeEdge(edge, nodeIds, selectedEdgeIds, focusedNodeLimit, true);
    }

    for (const edge of optionalEdges) {
      if (selectedEdgeIds.size >= focusedEdgeLimit) {
        break;
      }
      includeEdge(edge, nodeIds, selectedEdgeIds, focusedNodeLimit, false);
    }

    return {
      nodes: graph.nodes.filter((node) => nodeIds.has(node.id)),
      edges: edges.filter((edge) => selectedEdgeIds.has(edge.id)),
      focused: true,
    };
  }

  const nodes = [...graph.nodes]
    .sort(compareFallbackNodes)
    .slice(0, fallbackNodeLimit);
  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = graph.edges
    .filter((edge) => nodeIds.has(edge.start) && nodeIds.has(edge.end))
    .sort(compareFallbackEdges)
    .slice(0, fallbackEdgeLimit);

  return { nodes, edges, focused: false };
}

function compareFocusedEdges(
  left: GraphEdge,
  right: GraphEdge,
  activeNodeIds: Set<string>,
  activeEdgeIds: Set<string>,
) {
  return compareRank(
    rankFocusedEdge(left, activeNodeIds, activeEdgeIds),
    rankFocusedEdge(right, activeNodeIds, activeEdgeIds),
  );
}

function rankFocusedEdge(edge: GraphEdge, activeNodeIds: Set<string>, activeEdgeIds: Set<string>) {
  const touchesActiveNode = activeNodeIds.has(edge.start) || activeNodeIds.has(edge.end);
  return [
    activeEdgeIds.has(edge.id) ? 0 : 1,
    touchesActiveNode ? 0 : 1,
    edge.signal.active ? 0 : 1,
    -edge.pheromone,
    edge.to_layer ?? edge.from_layer ?? edge.layer,
    edge.distance,
    edge.relation,
    edge.id,
  ];
}

function compareFallbackNodes(left: GraphNode, right: GraphNode) {
  return compareRank(rankFallbackNode(left), rankFallbackNode(right));
}

function rankFallbackNode(node: GraphNode) {
  return [
    node.signal.active ? 0 : 1,
    node.active_layers[0] ?? node.layers[0] ?? node.layer,
    -node.degree,
    -node.concept_pheromone,
    node.label,
    node.id,
  ];
}

function compareFallbackEdges(left: GraphEdge, right: GraphEdge) {
  return compareRank(rankFallbackEdge(left), rankFallbackEdge(right));
}

function includeEdge(
  edge: GraphEdge,
  nodeIds: Set<string>,
  selectedEdgeIds: Set<string>,
  nodeLimit: number,
  allowNodeOverflow: boolean,
) {
  if (selectedEdgeIds.has(edge.id)) {
    return false;
  }

  const nextNodeCount =
    nodeIds.size +
    (nodeIds.has(edge.start) ? 0 : 1) +
    (nodeIds.has(edge.end) ? 0 : 1);
  if (!allowNodeOverflow && nextNodeCount > nodeLimit) {
    return false;
  }

  selectedEdgeIds.add(edge.id);
  nodeIds.add(edge.start);
  nodeIds.add(edge.end);
  return true;
}

function rankFallbackEdge(edge: GraphEdge) {
  return [
    edge.signal.active ? 0 : 1,
    -edge.pheromone,
    edge.to_layer ?? edge.from_layer ?? edge.layer,
    edge.distance,
    edge.relation,
    edge.id,
  ];
}

function collectCoverageEdgeIds(
  edges: GraphEdge[],
  activeNodeIds: Set<string>,
  activeEdgeIds: Set<string>,
) {
  const incidentEdges = new Map<string, GraphEdge[]>();
  for (const edge of edges) {
    const startEdges = incidentEdges.get(edge.start) ?? [];
    startEdges.push(edge);
    incidentEdges.set(edge.start, startEdges);

    const endEdges = incidentEdges.get(edge.end) ?? [];
    endEdges.push(edge);
    incidentEdges.set(edge.end, endEdges);
  }

  const selected = new Set<string>();
  for (const nodeId of activeNodeIds) {
    const candidates = incidentEdges.get(nodeId);
    if (!candidates?.length) {
      continue;
    }

    let best = candidates[0];
    for (let index = 1; index < candidates.length; index += 1) {
      const candidate = candidates[index];
      if (compareRank(
        rankFocusedEdge(candidate, activeNodeIds, activeEdgeIds),
        rankFocusedEdge(best, activeNodeIds, activeEdgeIds),
      ) < 0) {
        best = candidate;
      }
    }

    selected.add(best.id);
  }

  return selected;
}

type RankValue = number | string;

function compareRank(left: readonly RankValue[], right: readonly RankValue[]) {
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    if (left[index] < right[index]) return -1;
    if (left[index] > right[index]) return 1;
  }
  return left.length - right.length;
}
