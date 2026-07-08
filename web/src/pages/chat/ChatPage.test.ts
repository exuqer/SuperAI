import { flushPromises, shallowMount } from '@vue/test-utils';
import { nextTick, reactive } from 'vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import ChatPage from './ChatPage.vue';

const { runtimeState, apiMock } = vi.hoisted(() => {
  const runtimeState = {
    loading: false,
    lastAnalysis: null as any,
    graph: null as any,
    chat: vi.fn(async () => undefined),
    resonanceChat: vi.fn(async () => undefined),
  };
  const apiMock = {
    getSessions: vi.fn(),
    resetSession: vi.fn(),
    sendFeedback: vi.fn(),
    resonanceFeedback: vi.fn(),
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

describe('ChatPage', () => {
  beforeEach(() => {
    runtimeState.loading = false;
    runtimeState.lastAnalysis = null;
    runtimeState.graph = null;
    runtimeState.chat.mockReset();
    runtimeState.resonanceChat.mockReset();
    runtimeState.resonanceChat.mockImplementation(async () => {
      runtimeState.lastAnalysis = {
        result: {
          result_id: 'result-1',
          response: 'Ассистент отвечает',
          summary: 'Краткая сводка',
          activated_concepts: [
            {
              uri: '/m/top/dialogue',
              label: 'Общение',
              language: 'ru',
              layer: 0,
              layers: [0],
              active_layers: [0],
              score: 1.2,
              sources: ['input'],
            },
          ],
          semantic_vector: {
            mode: 'resonance',
            active_plane: 'language:ru',
            tentative_forms: [],
            items: [{ uri: '/m/top/dialogue', label: 'Общение', layer: 0, layers: [0], active_layers: [0], score: 1.2 }],
            strength_vector: [3],
          },
        },
      };
      runtimeState.graph = { nodes: [], edges: [], stats: { nodes: 0, edges: 0, signal_nodes: 0, signal_edges: 0 } };
    });
    apiMock.getSessions.mockReset();
    apiMock.resetSession.mockReset();
    apiMock.sendFeedback.mockReset();
    apiMock.resonanceFeedback.mockReset();
    apiMock.getConceptDetail.mockReset();
  });

  it('renders session history and latest diagnostics after sending a message', async () => {
    apiMock.getSessions.mockResolvedValueOnce([]).mockResolvedValueOnce([
      {
        session_id: 'default',
        turn_count: 2,
        updated_at: 1710000001,
        turns: [
          {
            role: 'user',
            text: 'hello',
            result_id: 'result-1',
            concepts: ['/m/top/dialogue'],
            created_at: 1710000000,
          },
          {
            role: 'assistant',
            text: 'Ассистент отвечает',
            result_id: 'result-1',
            concepts: ['/m/top/dialogue'],
            created_at: 1710000001,
          },
        ],
      },
    ]);

    const wrapper = shallowMount(ChatPage);
    await flushPromises();

    expect(wrapper.text()).toContain('Чат пуст.');

    await wrapper.get('textarea').setValue('hello');
    await wrapper.get('form').trigger('submit');
    await flushPromises();
    await nextTick();

    expect(runtimeState.resonanceChat).toHaveBeenCalledWith(
      expect.objectContaining({
        text: 'hello',
        session_id: 'default',
        lang: 'ru',
        ants: 32,
        creativity: 0.35,
      }),
    );
    expect(apiMock.getSessions).toHaveBeenCalledTimes(2);
    expect(wrapper.text()).toContain('hello');
    expect(wrapper.text()).toContain('Ассистент отвечает');
    expect(wrapper.text()).toContain('Ответ');
    expect(wrapper.text()).toContain('Semantic vector');

    await wrapper.get('[data-feedback-result-id="result-1"][data-feedback-score="1"]').trigger('click');
    await flushPromises();
    expect(apiMock.resonanceFeedback).toHaveBeenCalledWith(
      expect.objectContaining({
        result_id: 'result-1',
        score: 1,
        session_id: 'default',
      }),
    );
  });
});
