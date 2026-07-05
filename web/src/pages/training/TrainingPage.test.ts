import { flushPromises, shallowMount } from '@vue/test-utils';
import { reactive } from 'vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import TrainingPage from './TrainingPage.vue';

const { runtimeState, apiMock } = vi.hoisted(() => {
  const runtimeState = {
    loading: false,
    trackJob: vi.fn(async () => undefined),
  };
  const apiMock = {
    simpleTrain: vi.fn(),
    learn: vi.fn(),
    understand: vi.fn(),
    getConceptDetail: vi.fn(),
  };
  return { runtimeState, apiMock };
});

const runtimeMock = reactive(runtimeState);

vi.mock('@/app/stores/runtime', () => ({
  useRuntimeStore: () => runtimeMock,
}));

vi.mock('@/shared/api/client', () => ({
  api: apiMock,
}));

describe('TrainingPage', () => {
  beforeEach(() => {
    runtimeState.loading = false;
    runtimeState.trackJob.mockReset();
    apiMock.simpleTrain.mockReset();
    apiMock.learn.mockReset();
    apiMock.understand.mockReset();
    apiMock.getConceptDetail.mockReset();
    apiMock.simpleTrain.mockResolvedValue({ job_id: 'job-1', name: 'simple-train', status: 'queued', created_at: 1 });
    apiMock.learn.mockResolvedValue({ job_id: 'job-2', name: 'learn', status: 'queued', created_at: 1 });
    apiMock.understand.mockResolvedValue({
      input_text: 'что делает программист?',
      lang: 'ru',
      tokens: [
        {
          raw_token: 'что',
          lemma: 'что',
          search_token: '',
          concept_uri: null,
          match_status: 'ignored_stop_word',
          is_stop_word: true,
          morphology: {},
        },
        {
          raw_token: 'программист',
          lemma: 'программист',
          search_token: 'программист',
          concept_uri: '/c/ru/программист',
          match_status: 'candidate',
          is_stop_word: false,
          morphology: {},
        },
      ],
      summary: { search_tokens: ['программист'] },
    });
    apiMock.getConceptDetail.mockResolvedValue({
      node: {
        uri: '/c/ru/программист',
        metadata: {
          label: 'программист',
          meaning: 'человек, который пишет код',
        },
      },
      incoming: [],
      outgoing: [],
      aliases: [],
    });
  });

  it('renders simple mode preview and submits to simple training endpoint', async () => {
    const wrapper = shallowMount(TrainingPage);
    await flushPromises();

    expect(wrapper.get('[data-testid="simple-mode"]').exists()).toBe(true);
    expect(wrapper.text()).toContain('question tokens');
    expect(wrapper.text()).toContain('answer tokens');
    expect(wrapper.text()).toContain('программист');
    expect((wrapper.findAll('textarea')[2].element as HTMLTextAreaElement).value).toBe(
      'человек, который пишет код',
    );

    await wrapper.findAll('textarea')[0].setValue('что делает программист?');
    await wrapper.findAll('textarea')[1].setValue('Программист пишет код на компьютере.');
    await flushPromises();
    await wrapper.get('.run-row button').trigger('click');
    await flushPromises();

    expect(apiMock.simpleTrain).toHaveBeenCalledWith(
      expect.objectContaining({
        question: 'что делает программист?',
        expected_answer: 'Программист пишет код на компьютере.',
        lang: 'ru',
        concept_meanings: [
          expect.objectContaining({
            concept: '/c/ru/программист',
            label: 'программист',
            meaning: 'человек, который пишет код',
          }),
        ],
      }),
    );
    expect(runtimeState.trackJob).toHaveBeenCalledWith(expect.objectContaining({ job_id: 'job-1' }));
  });
});
