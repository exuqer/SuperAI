import { flushPromises, mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { createMemoryHistory, createRouter } from 'vue-router';
import { describe, expect, it } from 'vitest';

import { useHiveStore } from '@/entities/hive/store';
import HiveModePanel from './HiveModePanel.vue';

describe('HiveModePanel search scenes', () => {
  it('separates found, validated and candidate counts and opens a concept fog', async () => {
    const pinia = createPinia();
    setActivePinia(pinia);
    const store = useHiveStore();
    store.queryFrame = { source_text: 'А ещё что?' };
    store.queryScene = {
      id: 'query-scene',
      type: 'query_scene',
      status: 'INCOMPLETE',
      requested_role: 'object',
      slots: [{ id: 'slot-object', role: 'object', status: 'empty' }],
    };
    store.queryCandidates = [{
      id: 'candidate-fish', lemma: 'рыба', surface: 'рыбу', target_role: 'object',
      status: 'new', sources: ['scene-good'], scores: { total: .91 },
    }];
    store.memoryScenes = [
      {
        id: 'scene-good', source_text: 'Кот ест рыбу.', result_type: 'ROLE_HIT',
        scores: { total_score: .91 }, anchor_validation: { status: 'PASSED' },
        role_match_details: {
          agent: { score: .85, match_type: 'stable_concept', concept_space_ids: [91] },
          action: { score: 1, match_type: 'exact_form', concept_space_ids: [] },
        },
      },
      {
        id: 'scene-rejected', source_text: 'Рыбу продают на рынке.', result_type: 'PARTIAL_HIT',
        scores: { total_score: .24 }, anchor_validation: { status: 'FAILED' },
        role_match_details: {
          agent: { score: 0, match_type: 'none', concept_space_ids: [] },
          action: { score: 0, match_type: 'none', concept_space_ids: [] },
        },
      },
    ];

    const router = createRouter({
      history: createMemoryHistory(),
      routes: [
        { path: '/', name: 'chat', component: { template: '<div />' } },
        { path: '/field', name: 'training', component: { template: '<div />' } },
      ],
    });
    await router.push('/');
    await router.isReady();
    const wrapper = mount(HiveModePanel, { global: { plugins: [pinia, router] } });

    await wrapper.findAll('button').find(button => button.text() === 'Показать ход поиска')!.trigger('click');
    expect(wrapper.text()).toContain('НАЙДЕННЫЕ СЦЕНЫ');
    expect(wrapper.text()).toContain('найдено 2');
    expect(wrapper.text()).toContain('прошли проверку 1');
    expect(wrapper.text()).toContain('кандидатов 1');
    expect(wrapper.text()).toContain('семантическое приближение');
    expect(wrapper.findAll('.source-card.rejected')).toHaveLength(1);

    await wrapper.find('.concept-open').trigger('click');
    await flushPromises();
    expect(router.currentRoute.value).toMatchObject({
      name: 'training', query: { conceptSpace: '91' },
    });
  });
});
