import { describe, expect, it } from 'vitest';
import { getChatAnswerText } from './answerText';
import type { GraphAnswer, QueryGraph } from './types';

const resolved = (surface: string, fullAnswer = 'Механик дал роботу болт.'): GraphAnswer => ({
  status: 'RESOLVED', surface, short_answer: surface, full_answer: fullAnswer,
  validation: { valid: true },
});

const graph = (gapCount: number): QueryGraph => ({
  query_graph_id: 'query', status: 'READY', continuation_of: null, construction_ids: [], trace: {}, versions: {}, exclusions: [],
  event_pattern: {
    predicate: null, known_nodes: [], gap_node: {} as QueryGraph['event_pattern']['gap_node'], required_edges: [],
    target_gaps: Array.from({ length: gapCount }, () => ({} as QueryGraph['event_pattern']['gap_node'])),
  },
});

describe('getChatAnswerText', () => {
  it.each([
    ['Что механик дал роботу?', 'Болт.'],
    ['Кто дал роботу болт?', 'Механик.'],
    ['Кому механик дал болт?', 'Роботу.'],
  ])('uses the resolved single GAP for %s', (_question, surface) => {
    expect(getChatAnswerText(resolved(surface), graph(1))).toBe(surface);
  });

  it('uses the restored event for multiple GAPs', () => {
    expect(getChatAnswerText(resolved('Механик; роботу; болт.'), graph(3))).toBe('Механик дал роботу болт.');
  });

  it('uses a complete event for an aggregating question without a GAP', () => {
    expect(getChatAnswerText(resolved('Робот включился.', 'Робот включился.'), graph(0))).toBe('Робот включился.');
  });

  it('keeps surface for non-resolved answers', () => {
    expect(getChatAnswerText({ ...resolved('Уточните объект.'), status: 'UNRESOLVED' }, graph(1))).toBe('Уточните объект.');
  });
});
