import { describe, expect, it } from 'vitest';
import type { GraphEdge, GraphNode, GraphPayload } from '@/shared/api/types';
import { selectGraphViewport } from './select-graph';

function node(id: string, active = false): GraphNode {
  return {
    id,
    uri: id,
    label: id,
    language: 'en',
    source: 'checkpoint',
    layer: 1,
    metadata: {},
    concept_pheromone: active ? 4 : 1,
    suppression: 0,
    degree: active ? 3 : 1,
    signal: { active, count: active ? 1 : 0 },
  };
}

function edge(id: string, start: string, end: string, active = false): GraphEdge {
  return {
    id,
    start,
    end,
    relation: id,
    weight: 1,
    source: 'checkpoint',
    surface_text: null,
    layer: 1,
    distance: 1,
    edge_type: 'test',
    metadata: {},
    pheromone: active ? 4 : 1,
    route_stats: {},
    signal: { active, score: active ? 1 : 0 },
  };
}

describe('selectGraphViewport', () => {
  it('keeps the active neighborhood when signal nodes are present', () => {
    const graph: GraphPayload = {
      nodes: [node('a', true), node('b'), node('c')],
      edges: [edge('a-b', 'a', 'b'), edge('b-c', 'b', 'c')],
      stats: { nodes: 3, edges: 2, signal_nodes: 1, signal_edges: 0 },
    };

    const viewport = selectGraphViewport(graph);

    expect(viewport.focused).toBe(true);
    expect(viewport.nodes.map((item) => item.id)).toEqual(['a', 'b']);
    expect(viewport.edges.map((item) => item.id)).toEqual(['a-b']);
  });

  it('keeps a coverage edge for every active node even when the focused limit is tight', () => {
    const graph: GraphPayload = {
      nodes: [node('a', true), node('b'), node('c', true), node('d')],
      edges: [edge('a-b', 'a', 'b'), edge('c-d', 'c', 'd'), edge('b-d', 'b', 'd')],
      stats: { nodes: 4, edges: 3, signal_nodes: 2, signal_edges: 0 },
    };

    const viewport = selectGraphViewport(graph, { focusedEdgeLimit: 1 });

    expect(viewport.focused).toBe(true);
    expect(viewport.edges.map((item) => item.id)).toEqual(
      expect.arrayContaining(['a-b', 'c-d']),
    );
    expect(viewport.nodes.map((item) => item.id)).toEqual(
      expect.arrayContaining(['a', 'b', 'c', 'd']),
    );
  });

  it('falls back to a bounded subset when no signal is available', () => {
    const graph: GraphPayload = {
      nodes: [node('a'), node('b'), node('c')],
      edges: [edge('a-b', 'a', 'b'), edge('b-c', 'b', 'c')],
      stats: { nodes: 3, edges: 2, signal_nodes: 0, signal_edges: 0 },
    };

    const viewport = selectGraphViewport(graph, { fallbackNodeLimit: 2, fallbackEdgeLimit: 10 });

    expect(viewport.focused).toBe(false);
    expect(viewport.nodes).toHaveLength(2);
    expect(viewport.edges).toHaveLength(1);
    expect(viewport.edges[0].start).toBe(viewport.nodes[0].id);
  });
});
