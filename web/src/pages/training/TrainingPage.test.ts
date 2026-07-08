import { flushPromises, shallowMount } from '@vue/test-utils';
import { reactive } from 'vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import TrainingPage from './TrainingPage.vue';
import jokeSamples from './fixtures/jokes.json';

const { runtimeState, apiMock } = vi.hoisted(() => {
  const runtimeState = {
    loading: false,
    graph: null as any,
    lastAnalysis: null as any,
    lastResult: null as any,
    trackJob: vi.fn(async () => undefined),
  };
  const apiMock = {
    resetNetwork: vi.fn(),
    resonanceSeed: vi.fn(),
    resonanceTrainQa: vi.fn(),
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
    runtimeState.graph = null;
    runtimeState.lastAnalysis = null;
    runtimeState.lastResult = null;
    runtimeState.trackJob.mockReset();
    apiMock.resetNetwork.mockReset();
    apiMock.resonanceSeed.mockReset();
    apiMock.resonanceTrainQa.mockReset();
    apiMock.resetNetwork.mockResolvedValue({ job_id: 'job-r', name: 'reset-network', status: 'queued', created_at: 1 });
    apiMock.resonanceSeed.mockResolvedValue({ job_id: 'job-s', name: 'resonance-seed', status: 'queued', created_at: 1 });
    apiMock.resonanceTrainQa.mockResolvedValue({ job_id: 'job-qa', name: 'resonance-train-qa', status: 'queued', created_at: 1 });
  });

  it('renders question-answer training, submits auto qa payload, and supports full reset', async () => {
    const wrapper = shallowMount(TrainingPage);
    await flushPromises();

    expect(wrapper.text()).toContain('Вопрос → ответ');
    expect(wrapper.findAll('[data-testid="qa-annotation"]')).toHaveLength(0);
    expect(wrapper.text()).toContain('lemmas');
    expect(wrapper.text()).toContain('roles');
    expect(wrapper.get('[data-testid="sample-select"]').element).toBeTruthy();

    await wrapper.get('[data-testid="qa-train-button"]').trigger('click');
    await flushPromises();

    expect(apiMock.resonanceTrainQa).toHaveBeenCalledWith(
      expect.objectContaining({
        question: jokeSamples[0].question,
        expected_answer: jokeSamples[0].expected_answer,
        lang: jokeSamples[0].lang,
        annotations: [],
      }),
    );
    expect(runtimeState.trackJob).toHaveBeenCalledWith(expect.objectContaining({ job_id: 'job-qa' }));

    await wrapper.get('[data-testid="full-reset-button"]').trigger('click');
    await flushPromises();

    expect(apiMock.resetNetwork).toHaveBeenCalledWith({ keep_builtin: false });
    expect(runtimeState.graph).toBeNull();
    expect(runtimeState.lastAnalysis).toBeNull();
    expect(runtimeState.lastResult).toBeNull();
    expect(runtimeState.trackJob).toHaveBeenCalledWith(expect.objectContaining({ job_id: 'job-r' }));
  });
});
