import { flushPromises, shallowMount } from '@vue/test-utils';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import DecodePage from './DecodePage.vue';

const { apiMock } = vi.hoisted(() => ({
  apiMock: {
    decode: vi.fn(),
  },
}));

vi.mock('@/shared/api/client', () => ({
  api: apiMock,
}));

describe('DecodePage', () => {
  beforeEach(() => {
    apiMock.decode.mockReset();
    apiMock.decode.mockResolvedValue({
      input_text: 'кот есть рыба мясо',
      input_tokens: ['кот', 'есть', 'рыба', 'мясо'],
      lang: 'ru',
      sentence: 'кот ест рыбу и мясо',
      pattern: 'svo',
      session_id: 'decode-session',
      turn_id: 'turn-1',
      summary: {
        total_tokens: 4,
        used_tokens: 4,
        objects: 2,
        fallbacks: 0,
      },
      tokens: [
        {
          input_token: 'кот',
          normalized_token: 'кот',
          role: 'subject',
          surface: 'кот',
          concept_uri: '/c/ru/кот',
          transform_status: 'inflected',
          morphology: {
            POS: 'NOUN',
            case: 'nomn',
            number: 'sing',
            gender: 'masc',
            tense: null,
            person: null,
          },
        },
      ],
    });
  });

  it('calls the decode endpoint and renders the sentence', async () => {
    const wrapper = shallowMount(DecodePage);

    await wrapper.get('textarea').setValue('кот есть рыба мясо');
    const textareas = wrapper.findAll('textarea');
    await textareas[1].setValue('кот, есть, рыба, мясо');
    await wrapper.get('form').trigger('submit');
    await flushPromises();

    expect(apiMock.decode).toHaveBeenCalledWith(
      expect.objectContaining({
        text: 'кот есть рыба мясо',
        tokens: ['кот', 'есть', 'рыба', 'мясо'],
        lang: 'auto',
      }),
    );
    expect(wrapper.text()).toContain('кот ест рыбу и мясо');
    expect(wrapper.text()).toContain('subject');
    expect(wrapper.text()).toContain('surface');
  });
});
