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
  HiveSubspaceV2,
  HiveProjectionV2,
  QuerySceneCandidateV2,
  QuerySceneV2,
  HiveInspectionProjectionV2,
  HiveLocalResonanceV2,
  HiveResonanceSessionV2,
  QueryMessageMode,
  DynamicsStateV2,
} from '@/entities/hive/types';

interface HiveStateResponse {
  hive: HiveV2;
  cells: HiveCellV2[];
  messages: HiveMessageV2[];
}

interface HiveQueryResponse extends HiveStateResponse {
  message_id: string;
  decision: HiveQueryDecisionV2;
  metrics?: Record<string, number>;
  external_search?: ExternalSearch;
  merge_results?: Array<Record<string, unknown>>;
  resonance_events?: HiveResonanceEventV2[];
  query_frame?: Record<string, unknown>;
  query_scene?: QuerySceneV2;
  memory_scenes?: Array<Record<string, unknown>>;
  candidates?: QuerySceneCandidateV2[];
  answer?: { answer_mode: string; surface_answer: string; full_surface_answer?: string; confidence: number; status?: string };
  memory_sources?: Array<Record<string, unknown>>;
  sentence_plan?: Record<string, unknown> | null;
  full_sentence_plan?: Record<string, unknown> | null;
  morphology_trace?: Array<Record<string, unknown>>;
  reverse_validation?: Record<string, unknown> | null;
  unknown_token_searches?: Array<Record<string, unknown>>;
  role_searches?: Array<Record<string, unknown>>;
  resolved_mode?: QueryMessageMode;
  local_resonance?: HiveLocalResonanceV2 | null;
  resonance_probes?: Array<Record<string, unknown>>;
  active_query?: Record<string, unknown>;
  dynamics?: DynamicsStateV2;
  reasoning_trace?: Record<string, any>;
  resonance_session?: HiveResonanceSessionV2;
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

interface HiveExpandResponse {
  subspace: HiveSubspaceV2;
  candidates: Array<Record<string, unknown>>;
}

interface HiveHierarchyResponse {
  schema_version: number;
  hive: Record<string, unknown>;
  projection: HiveProjectionV2;
  cells: HiveCellV2[];
  subspaces: HiveSubspaceV2[];
  generation_candidates: Array<Record<string, unknown>>;
  inspection_projections?: HiveInspectionProjectionV2[];
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
  let reasoningPromise: Promise<void> | null = null;
  const runResult = ref<ReasoningResponse | null>(null);
  const copyStatus = ref('');
  const jsonOpen = ref(false);
  const jsonMode = ref<'current' | 'initial' | 'snapshot' | 'trace'>('current');
  const jsonFormat = ref<'formatted' | 'raw'>('formatted');
  const jsonStep = ref(0);
  const jsonValue = ref<unknown>(null);
  const generationCandidates = ref<Array<Record<string, unknown>>>([]);
  const hierarchyData = ref<HiveHierarchyResponse | null>(null);
  const queryFrame = ref<Record<string, unknown> | null>(null);
  const queryScene = ref<QuerySceneV2 | null>(null);
  const memoryScenes = ref<Array<Record<string, unknown>>>([]);
  const queryCandidates = ref<QuerySceneCandidateV2[]>([]);
  const queryAnswer = ref<{ answer_mode: string; surface_answer: string | null; full_surface_answer?: string | null; confidence: number; status?: string; short?: Record<string, unknown>; full?: Record<string, unknown> } | null>(null);
  const queryPipeline = ref<Record<string, any> | null>(null);
  const vibrationHistory = ref<Array<Record<string, unknown>>>([]);
  const inspectionProjections = ref<HiveInspectionProjectionV2[]>([]);
  const memorySources = ref<Array<Record<string, unknown>>>([]);
  const sentencePlan = ref<Record<string, unknown> | null>(null);
  const fullSentencePlan = ref<Record<string, unknown> | null>(null);
  const morphologyTrace = ref<Array<Record<string, unknown>>>([]);
  const reverseValidation = ref<Record<string, unknown> | null>(null);
  const unknownTokenSearches = ref<Array<Record<string, unknown>>>([]);
  const roleSearches = ref<Array<Record<string, unknown>>>([]);
  const localResonance = ref<HiveLocalResonanceV2 | null>(null);
  const resonanceProbes = ref<Array<Record<string, unknown>>>([]);
  const resonanceSession = ref<HiveResonanceSessionV2 | null>(null);
  const activeQuery = ref<Record<string, unknown> | null>(null);
  const resolvedMode = ref<QueryMessageMode | null>(null);
  const hiveStructure = ref<Record<string, unknown> | null>(null);
  const dynamics = ref<DynamicsStateV2 | null>(null);
  const reasoningTrace = ref<Record<string, any> | null>(null);

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
  const workingCellCount = computed(() => Number((hive.value?.capacity as Record<string, number> | undefined)?.working_cells
    ?? cells.value.filter(cell => cell.component_class !== 'memory_source').length));
  const memorySourceCount = computed(() => cells.value.filter(cell => cell.component_class === 'memory_source').length);

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
      await loadQueryState();
      cacheState();
      return state;
    });
  }

  async function loadQueryState() {
    if (!hive.value) return null;
    try {
      const state = await api.get<any>(`/api/v2/hives/${hive.value.id}/query-state`);
      resonanceSession.value = state?.resonance_session || null;
      if (!state?.query_scene) return state || null;
      queryFrame.value = state.query_frame;
      queryScene.value = state.query_scene;
      memoryScenes.value = state.memory_scenes || [];
      queryCandidates.value = state.candidates || [];
      queryAnswer.value = state.answer || null;
      queryPipeline.value = state.pipeline || state.hive?.pipeline || null;
      vibrationHistory.value = state.vibration?.history || [];
      dynamics.value = state.dynamics || null;
      reasoningTrace.value = state.reasoning_trace || null;
      inspectionProjections.value = state.inspection_projections || [];
      memorySources.value = state.memory_sources || [];
      sentencePlan.value = state.sentence_plan || null;
      fullSentencePlan.value = state.full_sentence_plan || null;
      morphologyTrace.value = state.morphology_trace || [];
      reverseValidation.value = state.reverse_validation || null;
      unknownTokenSearches.value = state.unknown_token_searches || [];
      roleSearches.value = state.role_searches || [];
      localResonance.value = state.local_resonance || null;
      resonanceProbes.value = state.resonance_probes || [];
      activeQuery.value = state.active_query || null;
      hiveStructure.value = state.hive_structure || null;
      return state;
    } catch {
      return null;
    }
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

  async function hierarchy(manageLoading = true) {
    if (!hive.value) return null;
    const operation = async () => {
      const result = await api.get<HiveHierarchyResponse>(
        `/api/v2/hives/${hive.value!.id}/hierarchy`
      );
      const subspacesByCell = new Map(result.cells.map(cell => [cell.id, cell.subspaces || []]));
      cells.value = cells.value.map(cell => ({ ...cell, subspaces: subspacesByCell.get(cell.id) || [] }));
      generationCandidates.value = result.generation_candidates || [];
      inspectionProjections.value = result.inspection_projections || [];
      hierarchyData.value = result;
      if (selectedCell.value) {
        selectedCell.value = cells.value.find(cell => cell.id === selectedCell.value?.id) || null;
      }
      return result;
    };
    return manageLoading ? run(operation) : operation();
  }

  async function expandCell(cellId: string, targetLevel: string = 'word_form') {
    if (!hive.value) return null;
    return run(async () => {
      const result = await api.post<HiveExpandResponse>(`/api/v2/hives/${hive.value!.id}/cells/${cellId}/expand`, {
        target_level: targetLevel, reason: 'user', max_candidates: 5,
      });
      cells.value = cells.value.map(cell => cell.id === cellId
        ? { ...cell, subspaces: [...(cell.subspaces || []), result.subspace] }
        : cell);
      generationCandidates.value = [];
      await hierarchy(false);
      if (selectedCell.value?.id === cellId) {
        selectedCell.value = cells.value.find(cell => cell.id === cellId) || null;
      }
      cacheState();
      return result;
    });
  }

  async function query(
    text: string,
    mode?: QueryMessageMode,
    resonanceScope: 'LOCAL_ONLY' | 'LOCAL_THEN_GLOBAL' = 'LOCAL_THEN_GLOBAL',
  ) {
    if (!hive.value || loading.value || reasoningLoading.value) return;
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
        { text, ...(mode ? { resolved_mode: mode } : {}), resonance_scope: resonanceScope }
      );
      applyState(result);
      syncMessages(result.messages);
      decision.value = result.decision;
      metrics.value = result.metrics || null;
      externalSearch.value = result.external_search || null;
      mergeResults.value = result.merge_results || [];
      resonanceEvents.value = result.resonance_events || [];
      queryFrame.value = result.query_frame || null;
      queryScene.value = result.query_scene || null;
      memoryScenes.value = result.memory_scenes || [];
      queryCandidates.value = result.candidates || [];
      queryAnswer.value = result.answer || null;
      queryPipeline.value = (result as any).pipeline || (result as any).hive?.pipeline || null;
      memorySources.value = result.memory_sources || [];
      sentencePlan.value = result.sentence_plan || null;
      fullSentencePlan.value = result.full_sentence_plan || null;
      morphologyTrace.value = result.morphology_trace || [];
      reverseValidation.value = result.reverse_validation || null;
      unknownTokenSearches.value = result.unknown_token_searches || [];
      roleSearches.value = result.role_searches || [];
      localResonance.value = result.local_resonance || null;
      resonanceProbes.value = result.resonance_probes || [];
      resonanceSession.value = result.resonance_session || null;
      activeQuery.value = result.active_query || null;
      dynamics.value = (result as any).dynamics || null;
      reasoningTrace.value = result.reasoning_trace || null;
      hiveStructure.value = (result as any).hive_structure || null;
      resolvedMode.value = result.resolved_mode || null;
      vibrationHistory.value = [];
      await hierarchy(false);
      if (result.resolved_mode !== 'LOCAL_RESONANCE' && queryScene.value && queryAnswer.value?.status !== 'RESOLVED') {
        await runReasoning();
      }
    } catch (cause) {
      error.value = 'Не удалось обработать сообщение. Проверьте сервер.';
      messages.value = messages.value.filter(message => message.id !== userMessageId);
    } finally {
      loading.value = false;
      cacheState();
    }
  }

  function runReasoning(): Promise<void> {
    if (!hive.value) return Promise.resolve();
    if (reasoningPromise) return reasoningPromise;
    reasoningLoading.value = true;
    reasoningPromise = (async () => {
      try {
        if (queryScene.value) {
          const result = await api.post<{ hive: any }>(`/api/v2/hives/${hive.value!.id}/vibrate/run`, { steps: reasoningSteps.value, config: {} });
          queryFrame.value = result.hive.query_frame;
          syncMessages(result.hive.messages || []);
          queryScene.value = result.hive.query_scene;
          memoryScenes.value = result.hive.memory_scenes;
          queryCandidates.value = result.hive.candidates;
          queryAnswer.value = result.hive.answer;
          queryPipeline.value = result.hive.pipeline || result.hive.hive?.pipeline || null;
          memorySources.value = result.hive.memory_sources || [];
          sentencePlan.value = result.hive.sentence_plan || null;
          fullSentencePlan.value = result.hive.full_sentence_plan || null;
          morphologyTrace.value = result.hive.morphology_trace || [];
          reverseValidation.value = result.hive.reverse_validation || null;
          unknownTokenSearches.value = result.hive.unknown_token_searches || [];
          roleSearches.value = result.hive.role_searches || [];
          localResonance.value = result.hive.local_resonance || null;
          resonanceProbes.value = result.hive.resonance_probes || [];
          activeQuery.value = result.hive.active_query || null;
          vibrationHistory.value = result.hive.vibration.history;
          dynamics.value = result.hive.dynamics || null;
          reasoningTrace.value = result.hive.reasoning_trace || null;
          await hierarchy(false);
          return;
        }
        runResult.value = await api.post<ReasoningResponse>(`/api/v2/hives/${hive.value!.id}/reasoning`, {
          text: goalText.value,
          config: { reasoning_steps: reasoningSteps.value },
        });
        applyState(runResult.value.hive);
      } finally {
        reasoningLoading.value = false;
        reasoningPromise = null;
      }
    })();
    return reasoningPromise;
  }

  async function runReasoningStep() {
    reasoningSteps.value = 1;
    await runReasoning();
  }

  async function rerunLocalResonance(scope: 'LOCAL_ONLY' | 'LOCAL_THEN_GLOBAL') {
    if (!hive.value || !localResonance.value?.probe_text || loading.value) return;
    return run(async () => {
      const result = await api.post<HiveQueryResponse>(`/api/v2/hives/${hive.value!.id}/query`, {
        text: localResonance.value!.probe_text,
        resolved_mode: 'LOCAL_RESONANCE',
        resonance_scope: scope,
      });
      applyState(result);
      queryFrame.value = result.query_frame || queryFrame.value;
      queryScene.value = result.query_scene || queryScene.value;
      queryAnswer.value = result.answer || queryAnswer.value;
      queryPipeline.value = (result as any).pipeline || (result as any).hive?.pipeline || queryPipeline.value;
      localResonance.value = result.local_resonance || null;
      resonanceProbes.value = result.resonance_probes || [];
      activeQuery.value = result.active_query || activeQuery.value;
      resolvedMode.value = result.resolved_mode || null;
      await hierarchy(false);
      cacheState();
      return result;
    });
  }

  async function importResonanceMatch(matchId: string, includeScenes = false) {
    const probe = resonanceProbes.value.at(-1) as any;
    if (!hive.value || !probe?.id) return null;
    return run(async () => {
      const result = await api.post<any>(`/api/v2/hives/${hive.value!.id}/resonance/${probe.id}/import`, {
        match_id: matchId,
        include_scenes: includeScenes,
      });
      const state = await api.get<any>(`/api/v2/hives/${hive.value!.id}/query-state`);
      applyState(state);
      localResonance.value = state.local_resonance || localResonance.value;
      resonanceProbes.value = state.resonance_probes || resonanceProbes.value;
      hiveStructure.value = state.hive_structure || hiveStructure.value;
      cacheState();
      return result;
    });
  }

  async function importResonanceConcept(conceptId: string) {
    if (!hive.value || !resonanceSession.value) return null;
    return run(async () => {
      const result = await api.post<any>(`/api/v2/hives/${hive.value!.id}/import-concept`, {
        session_id: resonanceSession.value!.id,
        concept_id: conceptId,
      });
      resonanceSession.value = result.session || resonanceSession.value;
      await hierarchy(false);
      cacheState();
      return result;
    });
  }

  async function stopReasoning() {
    if (!hive.value || !queryScene.value) return;
    const result = await api.post<{ vibration: { history: Array<Record<string, unknown>> } }>(`/api/v2/hives/${hive.value.id}/vibrate/stop`);
    vibrationHistory.value = result.vibration.history;
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
    queryFrame.value = null;
    queryScene.value = null;
    memoryScenes.value = [];
    queryCandidates.value = [];
    queryAnswer.value = null;
    queryPipeline.value = null;
      vibrationHistory.value = [];
      dynamics.value = null;
      reasoningTrace.value = null;
    inspectionProjections.value = [];
    memorySources.value = [];
    sentencePlan.value = null;
    fullSentencePlan.value = null;
    morphologyTrace.value = [];
    reverseValidation.value = null;
    unknownTokenSearches.value = [];
    roleSearches.value = [];
    localResonance.value = null;
    resonanceProbes.value = [];
    activeQuery.value = null;
    resolvedMode.value = null;
    hiveStructure.value = null;
    selectedCell.value = null;
    goalText.value = '';
    runResult.value = null;
    await createHive();
  }

  function applyState(state: HiveStateResponse, preserveMessages = true) {
    hive.value = state.hive;
    cells.value = state.cells || [];
    if (!preserveMessages && state.messages?.length) {
      syncMessages(state.messages);
    }
    if (selectedCell.value) {
      selectedCell.value = cells.value.find(c => c.id === selectedCell.value?.id) || null;
    }
  }

  function syncMessages(items: HiveMessageV2[]) {
    messages.value = items.map(message => ({ ...message, role: message.role || 'user' }));
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
      applyState(state, false);
      await loadQueryState();
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
    queryFrame,
    queryScene,
    memoryScenes,
    queryCandidates,
    queryAnswer,
    queryPipeline,
    vibrationHistory,
    inspectionProjections,
    memorySources,
    sentencePlan,
    fullSentencePlan,
    morphologyTrace,
    reverseValidation,
    unknownTokenSearches,
    roleSearches,
    localResonance,
    resonanceProbes,
    resonanceSession,
    activeQuery,
    resolvedMode,
    hiveStructure,
    dynamics,
    reasoningTrace,
    copyStatus,
    jsonOpen,
    jsonMode,
    jsonFormat,
    jsonStep,
    jsonValue,
    generationCandidates,
    hierarchyData,
    averageActivation,
    averageRetention,
    workingCellCount,
    memorySourceCount,
    activeCellIds,
    createHive,
    getHive,
    preview,
    loadQueryState,
    hierarchy,
    expandCell,
    query,
    rerunLocalResonance,
    importResonanceMatch,
    importResonanceConcept,
    runReasoning,
    runReasoningStep,
    stopReasoning,
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
