import { flushPromises, mount } from '@vue/test-utils';
import { createMemoryHistory, createRouter } from 'vue-router';
import { afterEach, describe, expect, it, vi } from 'vitest';
import AnalyticsPage from './AnalyticsPage.vue';
import { storage } from '@/shared/storage';

const run = (id: string, createdAt: string) => ({
  id,
  hive_id: 'hive-analytics',
  status: 'COMPLETED',
  reasoning_steps: 1,
  completed_steps: 1,
  stop_reason: 'COMPLETED',
  random_seed: 1,
  created_at: createdAt,
  completed_at: createdAt,
  query: { terms: ['кто', 'ловит', 'рыбу'], roles: ['subject', 'predicate', 'object'] },
  config: {},
});

function snapshot(step: number, answer = 'рыбак') {
  return {
    step,
    phase: step === 0 ? 'INITIAL' : 'AFTER_SETTLE',
    created_at: '2026-07-14T16:00:00Z',
    temperature: step ? .72 : 1,
    metrics: { average_activation: .62, average_retention: .47, total_energy: 4.4, active_nodes: 6, weakening_nodes: 0, evicted_nodes: 0 },
    nodes: [{ placement_id: 1, cell_id: 'cell-fisher', cloud_id: 20, node_type: 'scene', label: 'рыбак ловит рыбу', local_activation: .62, local_gravity: .64, retention: .47, energy: .73, eviction_status: 'ACTIVE' }],
    candidates: [{ placement_id: 1, cell_id: 'cell-fisher', scene_cloud_id: 20, scene_label: 'рыбак ловит рыбу', answer, matched_components: [], answer_components: [], semantic_score: 1, dynamic_score: .57, viability: 1, candidate_score: .87, eviction_status: 'ACTIVE', explanation: 'совпали ловит и рыбу; ответ извлечён из роли подлежащее' }],
    delta: {},
    events: [],
  };
}

function response() {
  const primaryRun = run('run-new', '2026-07-14T16:01:00Z');
  const comparisonRun = run('run-old', '2026-07-14T16:00:00Z');
  return {
    hive_id: 'hive-analytics',
    runs: [primaryRun, comparisonRun],
    current: { query_components: [{ term: 'кто', role: 'subject', word_form_cloud_id: null }], snapshot: snapshot(1), updated_at: '2026-07-14T16:01:00Z' },
    primary: { run: primaryRun, query_components: [{ term: 'кто', role: 'subject', word_form_cloud_id: null }], snapshots: [snapshot(0), snapshot(1)], events: [], clusters: [] },
    comparison: { run: comparisonRun, query_components: [], snapshots: [snapshot(0), snapshot(1, 'кот')], events: [], clusters: [] },
  };
}

describe('AnalyticsPage', () => {
  afterEach(() => {
    localStorage.clear();
    vi.unstubAllGlobals();
  });

  it('shows an explained answer candidate and opens its hive cell', async () => {
    storage.setActiveHive('hive-analytics');
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response()), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const router = createRouter({ history: createMemoryHistory(), routes: [
      { path: '/', name: 'chat', component: { template: '<div />' } },
      { path: '/analytics', name: 'analytics', component: AnalyticsPage },
    ] });
    await router.push('/analytics');
    await router.isReady();
    const wrapper = mount(AnalyticsPage, { global: { plugins: [router] } });
    await flushPromises();

    expect(fetchMock.mock.calls[0][0]).toContain('/api/v2/hives/hive-analytics/analytics');
    expect(wrapper.text()).toContain('рыбак');
    expect(wrapper.text()).toContain('Это внутренний ранг, не вероятность.');
    await wrapper.find('.candidate-row').trigger('click');
    await flushPromises();
    expect(router.currentRoute.value).toMatchObject({ name: 'chat', query: { cell: 'cell-fisher' } });
  });
});
