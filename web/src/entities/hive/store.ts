/** Hive store - local memory, routing, reasoning. */

import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import { api } from '@/shared/api/client';
import { storage } from '@/shared/storage';
import type {
  HiveV2,
  HiveCellV2,
  HiveMessageV2,
  HiveQueryDecisionV2,
  HiveResonanceEventV2,
} from '@/entities/hive/types';

interface HiveStateResponse {
  hive: HiveV2;
  cells: HiveCellV2[];
  messages: Array<Omit<HiveMessageV2, 'role'>>;
}

interface HiveQueryResponse extends HiveStateResponse {
  message_id: string;
  decision: HiveQueryDecisionV2;
  metrics?: Record<string, number>;
  external_search?: ExternalSearch;
  merge_results?: Array<Record<string, unknown>>;
  resonance_events?: HiveResonanceEventV2[];
}

interface ReasoningResponse {
  run: { id: string };
  completed_steps: number;
  stop_reason: string;
  hive: HiveStateResponse;
}

interface ExternalSearch {
  sources: Array<{ id: string; label: string; x: number; y: number; fitness: number }>;
  bees: Array<{ id: string }>;
  iterations: number;
  anchors: Array<Record<string, unknown>>;
}

export const useHiveStore = defineStore('hive', () => {
  const hive = ref<HiveV2 | null>(null);
  const cells = ref<HiveCellV2[]>([]);
  const messages = ref<HiveMessageV2[]>([]);
  const decision = ref<HiveQueryDecisionV2 | null>(null);
  const resonanceEvents = ref<HiveResonanceEventV2[]>([]);
  const externalSearch = ref<ExternalSearch | null>(null);
  const mergeResults = ref<Array<Record<string, unknown>>>([]);
  const metrics = ref<Record<string, number> | null>(null);
  const loading = ref(false);
  const error = ref('');
  const selectedCell = ref<HiveCellV2 | null>(null);
  const goalText = ref('');
  const reasoningSteps = ref(3);
  const reasoningLoading = ref(false);
  const runResult = ref<ReasoningResponse | null>(null);
  const copyStatus = ref('');
  const jsonOpen = ref(false);
  const jsonMode = ref<'current' | 'initial' | 'snapshot' | 'trace'>('current');
  const jsonFormat = ref<'formatted' | 'raw'>('formatted');
  const jsonStep = ref(0);
  const jsonValue = ref<unknown>(null);

  // Computed
  const averageActivation = computed(() =>
    cells.value.length ? cells.value.reduce((sum, c) => sum + c.local_activation, 0) / cells.value.length : 0
  );
  const averageRetention = computed(() =>
    cells.value.length ? cells.value.reduce((sum, c) => sum + c.retention, 0) / cells.value.length : 0
  );
  const activeCellIds = computed(() => {
    return new Set(decision.value?.matches?.map(match => match.cell_id) || []);
  });

  async function run<T>(operation: () => Promise<T>): Promise<T> {
    loading.value = true;
    error.value = '';
    try {
      return await operation();
    } catch (cause) {
      error.value = cause instanceof Error ? cause.message : String(cause);
      throw cause;
    } finally {
      loading.value = false;
    }
  }

  async function createHive(maxCells = 24, conversationId = '') {
    return run(async () => {
      let resolvedConversationId = conversationId || storage.getConversationId();
      if (!resolvedConversationId) {
        resolvedConversationId = crypto.randomUUID();
        storage.setConversationId(resolvedConversationId);
      }
      const state = await api.post<HiveStateResponse>('/api/v2/hives', {
        max_cells: maxCells,
        conversation_id: resolvedConversationId,
      });
      applyState(state, false);
      cacheState();
      return state;
    });
  }

  async function getHive(hiveId: string) {
    return run(async () => {
      const state = await api.get<HiveStateResponse>(`/api/v2/hives/${hiveId}`);
      applyState(state, false);
      cacheState();
      return state;
    });
  }

  async function preview(text: string) {
    if (!hive.value) return;
    return run(async () => {
      return await api.post<Record<string, unknown>>(
        `/api/v2/hives/${hive.value!.id}/query/preview`,
        { text }
      );
    });
  }

  async function query(text: string) {
    if (!hive.value || loading.value) return;
    loading.value = true;
    error.value = '';
    goalText.value = text;
    const userMessageId = `user-${Date.now()}`;
    messages.value.push({
      id: userMessageId,
      hive_id: hive.value.id,
      turn_index: messages.value.length + 1,
      text,
      parsed_json: {},
      created_at: new Date().toISOString(),
      role: 'user',
    });
    cacheState();
    try {
      const result = await api.post<HiveQueryResponse>(
        `/api/v2/hives/${hive.value.id}/query`,
        { text }
      );
      applyState(result);
      decision.value = result.decision;
      metrics.value = result.metrics || null;
      externalSearch.value = result.external_search || null;
      mergeResults.value = result.merge_results || [];
      resonanceEvents.value = result.resonance_events || [];
      messages.value.push({
        id: `assistant-${result.message_id}`,
        hive_id: hive.value.id,
        turn_index: messages.value.length + 1,
        text: result.decision.external_search_required
          ? `Улей активировал известные компоненты и выполнил ${result.decision.decision === 'PARTIAL_HIT' ? 'частичный' : 'целевой'} внешний поиск.`
          : 'Контекст найден в локальной памяти: внешний поиск не запускался.',
        parsed_json: {},
        created_at: new Date().toISOString(),
        role: 'assistant',
      });
    } catch (cause) {
      error.value = 'Не удалось обработать сообщение. Проверьте сервер.';
      messages.value = messages.value.filter(message => message.id !== userMessageId);
    } finally {
      loading.value = false;
      cacheState();
    }
  }

  async function runReasoning() {
    if (!hive.value || reasoningLoading.value) return;
    reasoningLoading.value = true;
    try {
      runResult.value = await api.post<ReasoningResponse>(`/api/v2/hives/${hive.value.id}/reasoning`, {
        text: goalText.value,
        config: { reasoning_steps: reasoningSteps.value },
      });
      applyState(runResult.value.hive);
    } finally {
      reasoningLoading.value = false;
    }
  }

  async function runReasoningStep() {
    reasoningSteps.value = 1;
    await runReasoning();
  }

  async function loadJson() {
    if (!hive.value) return;
    const runId = runResult.value?.run?.id;
    const params: Record<string, string> = { mode: jsonMode.value, detail: 'full' };
    if (runId) params.run_id = runId;
    if (jsonMode.value === 'snapshot' || jsonMode.value === 'initial') {
      params.step = String(jsonMode.value === 'initial' ? 0 : jsonStep.value);
    }
    try {
      jsonValue.value = await api.get(`/api/v2/hives/${hive.value.id}/reasoning/export`, params);
    } catch {
      jsonValue.value = { error: 'Нет доступного snapshot/run' };
    }
  }

  async function openJson(mode: typeof jsonMode.value = 'current') {
    jsonMode.value = mode;
    jsonOpen.value = true;
    await loadJson();
  }

  async function copyJson(mode: typeof jsonMode.value) {
    if (!jsonOpen.value) {
      await openJson(mode);
    }
    try {
      const text = jsonFormat.value === 'raw'
        ? JSON.stringify(jsonValue.value)
        : JSON.stringify(jsonValue.value, null, 2);
      await navigator.clipboard.writeText(text);
      copyStatus.value = 'Скопировано';
      setTimeout(() => { copyStatus.value = '' }, 1500);
    } catch {
      copyStatus.value = 'Ошибка копирования';
    }
  }

  function downloadJson() {
    const text = jsonFormat.value === 'raw'
      ? JSON.stringify(jsonValue.value)
      : JSON.stringify(jsonValue.value, null, 2);
    const blob = new Blob([text], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `hive-${hive.value?.id || 'export'}-${jsonMode.value}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  async function resetHive() {
    storage.removeActiveHive();
    storage.removeChatCache();
    const conversationId = crypto.randomUUID();
    storage.setConversationId(conversationId);
    messages.value = [];
    decision.value = null;
    metrics.value = null;
    externalSearch.value = null;
    mergeResults.value = [];
    resonanceEvents.value = [];
    selectedCell.value = null;
    goalText.value = '';
    runResult.value = null;
    await createHive();
  }

  function applyState(state: HiveStateResponse, preserveMessages = true) {
    hive.value = state.hive;
    cells.value = state.cells || [];
    if (!preserveMessages && state.messages?.length) {
      messages.value = state.messages.map(message => ({ ...message, role: 'user' }));
    }
    if (selectedCell.value) {
      selectedCell.value = cells.value.find(c => c.id === selectedCell.value?.id) || null;
    }
  }

  function cacheState() {
    if (!hive.value) return;
    storage.setActiveHive(hive.value.id);
    storage.setChatCache({
      messages: messages.value,
      goalText: goalText.value,
      decision: decision.value,
      metrics: metrics.value,
      externalSearch: externalSearch.value,
      mergeResults: mergeResults.value,
      resonanceEvents: resonanceEvents.value,
    });
  }

  async function restoreHive() {
    const id = storage.getActiveHive();
    const cache = storage.getChatCache();
    if (cache) {
      try {
        messages.value = cache.messages || [];
        goalText.value = cache.goalText || '';
        decision.value = cache.decision || null;
        metrics.value = cache.metrics || null;
        externalSearch.value = cache.externalSearch || null;
        mergeResults.value = cache.mergeResults || [];
        resonanceEvents.value = cache.resonanceEvents || [];
      } catch {
        storage.removeChatCache();
      }
    }
    if (!id) return createHive();
    try {
      const state = await api.get<HiveStateResponse>(`/api/v2/hives/${id}`);
      applyState(state, messages.value.length > 0);
      if (!messages.value.length && state.messages?.length) applyState(state, false);
    } catch {
      storage.removeActiveHive();
      return createHive();
    }
  }

  return {
    hive,
    cells,
    messages,
    decision,
    resonanceEvents,
    externalSearch,
    mergeResults,
    metrics,
    loading,
    error,
    selectedCell,
    goalText,
    reasoningSteps,
    reasoningLoading,
    runResult,
    copyStatus,
    jsonOpen,
    jsonMode,
    jsonFormat,
    jsonStep,
    jsonValue,
    averageActivation,
    averageRetention,
    activeCellIds,
    createHive,
    getHive,
    preview,
    query,
    runReasoning,
    runReasoningStep,
    loadJson,
    openJson,
    copyJson,
    downloadJson,
    resetHive,
    restoreHive,
    applyState,
    cacheState,
  };
});
