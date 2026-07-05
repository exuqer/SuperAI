import { flushPromises, shallowMount } from '@vue/test-utils';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import UnderstandPage from './UnderstandPage.vue';

const { apiMock } = vi.hoisted(() => ({
  apiMock: {
    understand: vi.fn(),
  },
}));

vi.mock('@/shared/api/client', () => ({
  api: apiMock,
}));

describe('UnderstandPage', () => {
  beforeEach(() => {
    apiMock.understand.mockReset();
    apiMock.understand.mockResolvedValue({
      input_text: 'котики едят',
      lang: 'ru',
      session_id: 'diag-session',
      turn_id: 'turn-1',
      summary: {
        total_tokens: 2,
        working_tokens: 2,
        stop_words: 0,
        matched: 2,
        candidates: 0,
        partial_root_matches: 0,
        edit_distance_matches: 0,
        search_tokens: ['кот', 'есть'],
      },
      tokens: [
        {
          raw_token: 'котики',
          lemma: 'кот',
          search_token: 'кот',
          concept_uri: '/c/ru/кот',
          match_status: 'found_as_lemma',
          is_stop_word: false,
          morphology: {
            POS: 'NOUN',
            case: 'nomn',
            number: 'plur',
            gender: null,
            tense: null,
            person: null,
          },
        },
        {
          raw_token: 'едят',
          lemma: 'есть',
          search_token: 'есть',
          concept_uri: '/c/ru/есть',
          match_status: 'found_as_lemma',
          is_stop_word: false,
          morphology: {
            POS: 'VERB',
            case: null,
            number: 'plur',
            gender: null,
            tense: 'pres',
            person: '3per',
          },
        },
      ],
    });
  });

  it('calls the understand endpoint and renders token diagnostics', async () => {
    const wrapper = shallowMount(UnderstandPage);

    await wrapper.get('textarea').setValue('котики едят');
    await wrapper.get('form').trigger('submit');
    await flushPromises();

    expect(apiMock.understand).toHaveBeenCalledWith(
      expect.objectContaining({
        text: 'котики едят',
        lang: 'auto',
      }),
    );
    expect(wrapper.text()).toContain('кот');
    expect(wrapper.text()).toContain('/c/ru/кот');
    expect(wrapper.text()).toContain('found_as_lemma');
    expect(wrapper.text()).toContain('POS=NOUN');
    expect(wrapper.text()).toContain('search token');
  });
});
