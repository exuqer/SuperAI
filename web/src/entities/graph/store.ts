/** State for the V2.7 QueryGraph dialogue UI. */

import { computed, ref } from 'vue';
import { defineStore } from 'pinia';
import { api, type ApiError } from '@/shared/api/client';
import { storage } from '@/shared/storage';
import type {
  ChatMessage,
  HiveQueryResponse,
  HiveState,
  HybridPipelineResult,
  QueryMode,
  RetrievalScope,
  SwarmTrace,
} from './types';
import { getChatAnswerText } from './answerText';

interface ChatCache {
  hiveId: string;
  messages: ChatMessage[];
}

function now(): string {
  return new Date().toISOString();
}

function messageId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export const useGraphStore = defineStore('graph-v27', () => {
  const state = ref<HiveState | null>(null);
  const messages = ref<ChatMessage[]>([]);
  const loading = ref(false);
  const restoring = ref(false);
  const error = ref('');

  const hive = computed(() => state.value?.hive || null);
  const queryGraph = computed(() => state.value?.query_graph || null);
  const answer = computed(() => state.value?.answer || null);
  const candidateBindings = computed(() => state.value?.candidate_bindings || []);
  const rejectedEvents = computed(() => state.value?.rejected_events || []);
  const selectedBindings = computed(() => state.value?.selected_bindings || []);
  const bindingConfiguration = computed(() => state.value?.binding_configuration || null);
  const swarm = computed<SwarmTrace | null>(() => (
    state.value?.swarm || state.value?.trace?.swarm as SwarmTrace | null
  ));
  const trace = computed(() => state.value?.trace || {});
  const hybrid = computed<HybridPipelineResult | null>(() => state.value?.hybrid || null);

  function applyState(next: HiveState): void {
    state.value = next;
    storage.setActiveHive(next.hive.id);
  }

  function persistMessages(): void {
    if (!hive.value) return;
    storage.setChatCache({
      hiveId: hive.value.id,
      messages: messages.value.slice(-100),
    } satisfies ChatCache);
  }

  function restoreMessages(hiveId: string): void {
    const cached = storage.getChatCache() as ChatCache | null;
    messages.value = cached?.hiveId === hiveId && Array.isArray(cached.messages)
      ? cached.messages.map((message: ChatMessage) => (
        message.role === 'assistant' && message.answer
          ? { ...message, text: getChatAnswerText(message.answer, message.queryGraph) }
          : message
      ))
      : [];
  }

  function clearLocalState(): void {
    state.value = null;
    messages.value = [];
    storage.removeActiveHive();
    storage.removeChatCache();
  }

  async function createHive(maxCells = 24): Promise<HiveState> {
    let conversationId = storage.getConversationId();
    if (!conversationId) {
      conversationId = crypto.randomUUID();
      storage.setConversationId(conversationId);
    }
    const next = await api.post<HiveState>('/api/v2/hives', {
      max_cells: maxCells,
      conversation_id: conversationId,
    });
    applyState(next);
    messages.value = [];
    persistMessages();
    return next;
  }

  async function restoreHive(): Promise<HiveState> {
    if (state.value) return state.value;
    restoring.value = true;
    error.value = '';
    const hiveId = storage.getActiveHive();
    try {
      if (!hiveId) return await createHive();
      const next = await api.get<HiveState>(`/api/v2/hives/${hiveId}`);
      applyState(next);
      restoreMessages(hiveId);
      return next;
    } catch (cause) {
      const apiError = cause as Partial<ApiError>;
      if (apiError.status === 404) {
        clearLocalState();
        return createHive();
      }
      error.value = cause instanceof Error ? cause.message : String(cause);
      throw cause;
    } finally {
      restoring.value = false;
    }
  }

  async function resetHive(): Promise<HiveState> {
    loading.value = true;
    error.value = '';
    clearLocalState();
    try {
      return await createHive();
    } catch (cause) {
      error.value = cause instanceof Error ? cause.message : String(cause);
      throw cause;
    } finally {
      loading.value = false;
    }
  }

  async function query(
    text: string,
    mode?: QueryMode,
    retrievalScope: RetrievalScope = 'LOCAL_ONLY',
  ): Promise<HiveQueryResponse> {
    const normalized = text.trim();
    if (!normalized) throw new Error('Введите вопрос');
    if (!state.value) await restoreHive();
    if (!hive.value) throw new Error('Не удалось создать диалог');

    const userMessage: ChatMessage = {
      id: messageId('user'),
      role: 'user',
      text: normalized,
      createdAt: now(),
      status: 'PENDING',
    };
    messages.value.push(userMessage);
    persistMessages();
    loading.value = true;
    error.value = '';
    const executeQuery = () => api.post<HiveQueryResponse>(
      `/api/v2/hives/${hive.value!.id}/query`,
      {
        text: normalized,
        resolved_mode: mode,
        retrieval_scope: retrievalScope,
      },
    );
    try {
      let result: HiveQueryResponse;
      try {
        result = await executeQuery();
      } catch (cause) {
        const apiError = cause as Partial<ApiError>;
        if (apiError.status !== 404) throw cause;
        clearLocalState();
        await createHive();
        messages.value.push(userMessage);
        persistMessages();
        result = await executeQuery();
      }
      applyState({
        ...result,
        turn_index: (state.value?.turn_index || 0) + 1,
      });
      userMessage.status = result.answer?.status || 'UNRESOLVED';
      userMessage.queryGraphId = result.query_graph?.query_graph_id;
      messages.value.push({
        id: result.message_id || messageId('assistant'),
        role: 'assistant',
        text: getChatAnswerText(result.answer, result.query_graph),
        createdAt: now(),
        status: result.answer?.status || 'UNRESOLVED',
        queryGraphId: result.query_graph?.query_graph_id,
        answer: result.answer,
        queryGraph: result.query_graph,
      });
      persistMessages();
      return result;
    } catch (cause) {
      userMessage.status = 'ERROR';
      error.value = cause instanceof Error ? cause.message : String(cause);
      messages.value.push({
        id: messageId('assistant-error'),
        role: 'assistant',
        text: `Ошибка обработки: ${error.value}`,
        createdAt: now(),
        status: 'ERROR',
      });
      persistMessages();
      throw cause;
    } finally {
      loading.value = false;
    }
  }

  async function refresh(): Promise<HiveState | null> {
    if (!hive.value) return restoreHive();
    const next = await api.get<HiveState>(`/api/v2/hives/${hive.value.id}`);
    applyState(next);
    return next;
  }

  return {
    state,
    messages,
    loading,
    restoring,
    error,
    hive,
    queryGraph,
    answer,
    candidateBindings,
    rejectedEvents,
    selectedBindings,
    bindingConfiguration,
    swarm,
    trace,
    hybrid,
    createHive,
    restoreHive,
    resetHive,
    query,
    refresh,
  };
});
