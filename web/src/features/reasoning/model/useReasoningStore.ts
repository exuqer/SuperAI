/** Reasoning store - composes hive store for reasoning functionality. */

import { computed } from 'vue';
import { useHiveStore } from '@/entities/hive/store';

export function useReasoningStore() {
  const hiveStore = useHiveStore();

  const reasoningStatus = computed(() =>
    hiveStore.reasoningLoading ? 'выполняется'
      : hiveStore.runResult?.stop_reason || 'готов'
  );

  return {
    hive: computed(() => hiveStore.hive),
    reasoningLoading: computed(() => hiveStore.reasoningLoading),
    runResult: computed(() => hiveStore.runResult),
    copyStatus: computed(() => hiveStore.copyStatus),
    jsonOpen: computed(() => hiveStore.jsonOpen),
    jsonMode: computed(() => hiveStore.jsonMode),
    jsonFormat: computed(() => hiveStore.jsonFormat),
    jsonStep: computed(() => hiveStore.jsonStep),
    jsonValue: computed(() => hiveStore.jsonValue),
    reasoningSteps: computed(() => hiveStore.reasoningSteps),
    reasoningStatus,
    runReasoning: hiveStore.runReasoning,
    runReasoningStep: hiveStore.runReasoningStep,
    loadJson: hiveStore.loadJson,
    copyJson: hiveStore.copyJson,
    downloadJson: hiveStore.downloadJson,
  };
}