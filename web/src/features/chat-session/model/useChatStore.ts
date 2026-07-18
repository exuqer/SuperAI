/** Chat session store - composes hive store for chat functionality. */

import { computed } from 'vue';
import { useHiveStore } from '@/entities/hive/store';
import { formatTime } from '@/shared/utils/time';

export function useChatStore() {
  const hiveStore = useHiveStore();

  const mode = computed(() =>
    !hiveStore.decision ? 'idle'
      : !hiveStore.decision.external_search_required ? 'local'
      : hiveStore.decision.decision === 'PARTIAL_HIT' ? 'partial'
      : 'external'
  );

  const modeLabel = computed(() =>
    mode.value === 'local' ? 'Локальный резонанс'
      : mode.value === 'partial' ? 'Частичный поиск'
      : mode.value === 'external' ? 'Внешний поиск'
      : 'Улей готов'
  );

  const hasSwarmMap = computed(() =>
    hiveStore.decision?.external_search_required && hiveStore.externalSearch?.sources?.length
  );

  const activeCellIds = computed(() =>
    new Set((hiveStore.decision as any)?.matches?.map((m: any) => m.cell_id) || [])
  );

  const externalSearch = computed(() => hiveStore.externalSearch);
  const sources = computed(() => hiveStore.externalSearch?.sources || []);
  const bees = computed(() => hiveStore.externalSearch?.bees || []);
  const metrics = computed(() => hiveStore.metrics);
  const goalText = computed(() => hiveStore.goalText);
  const decision = computed(() => hiveStore.decision);
  const eventLog = computed(() => {
    if (!hiveStore.decision) return [];
    const now = formatTime(new Date().toISOString());
    const events = [
      { id: 'route', kind: 'goal', text: `Маршрутизатор: ${hiveStore.decision.decision}`, time: now },
    ];
    if (hiveStore.decision.external_search_required) {
      events.push({
        id: 'search',
        kind: 'bee',
        text: `Рой получил ${hiveStore.metrics?.bees || 0} пчёл для недостающего контекста`,
        time: now,
      });
    } else {
      events.push({
        id: 'local',
        kind: 'done',
        text: `Активировано ${hiveStore.metrics?.activated_cells || 0} ячеек без внешнего поиска`,
        time: now,
      });
    }
    if ((hiveStore.metrics?.merged_cells || 0) > 0) {
      events.push({
        id: 'merge',
        kind: 'nectar',
        text: 'Новый нектар объединён с локальной памятью',
        time: now,
      });
    }
    return events;
  });

  const hiveGraphNodes = computed(() =>
    hiveStore.cells.map((cell: any, index: number) => ({
      cell,
      x: 105 + ((cell.x || index * 73) % 650),
      y: 90 + ((cell.y || index * 107) % 330),
      radius: 24 + cell.retention * 24,
      label: cell.label.slice(0, 14),
    }))
  );

  function sourcePoint(source: any) {
    return {
      x: 105 + (Math.abs(source.x || 0) % 790),
      y: 95 + (Math.abs(source.y || 0) % 510),
    };
  }

  return {
    hive: computed(() => hiveStore.hive),
    cells: computed(() => hiveStore.cells),
    messages: computed(() => hiveStore.messages),
    decision,
    resonanceEvents: computed(() => hiveStore.resonanceEvents),
    externalSearch,
    mergeResults: computed(() => hiveStore.mergeResults),
    metrics,
    loading: computed(() => hiveStore.loading),
    error: computed(() => hiveStore.error),
    selectedCell: computed(() => hiveStore.selectedCell),
    goalText,
    reasoningSteps: computed(() => hiveStore.reasoningSteps),
    reasoningLoading: computed(() => hiveStore.reasoningLoading),
    runResult: computed(() => hiveStore.runResult),
    copyStatus: computed(() => hiveStore.copyStatus),
    jsonOpen: computed(() => hiveStore.jsonOpen),
    jsonMode: computed(() => hiveStore.jsonMode),
    jsonFormat: computed(() => hiveStore.jsonFormat),
    jsonStep: computed(() => hiveStore.jsonStep),
    jsonValue: computed(() => hiveStore.jsonValue),
    averageActivation: computed(() => hiveStore.averageActivation),
    averageRetention: computed(() => hiveStore.averageRetention),
    activeCellIds,
    createHive: hiveStore.createHive,
    getHive: hiveStore.getHive,
    preview: hiveStore.preview,
    query: hiveStore.query,
    runReasoning: hiveStore.runReasoning,
    runReasoningStep: hiveStore.runReasoningStep,
    loadJson: hiveStore.loadJson,
    copyJson: hiveStore.copyJson,
    resetHive: hiveStore.resetHive,
    restoreHive: hiveStore.restoreHive,
    applyState: hiveStore.applyState,
    cacheState: hiveStore.cacheState,
    mode,
    modeLabel,
    hasSwarmMap,
    sources,
    bees,
    eventLog,
    hiveGraphNodes,
    sourcePoint,
  };
}
