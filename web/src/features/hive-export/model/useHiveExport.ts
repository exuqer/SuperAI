/** Hive export composable - wraps hive store for export functionality. */

import { useHiveStore } from '@/entities/hive/store';
import { storeToRefs } from 'pinia';

export function useHiveExport() {
  const hiveStore = useHiveStore();
  const { jsonOpen, jsonMode, jsonFormat, jsonStep, jsonValue } = storeToRefs(hiveStore);

  return {
    jsonOpen,
    jsonMode,
    jsonFormat,
    jsonStep,
    jsonValue,
    loadJson: hiveStore.loadJson,
    copyJson: hiveStore.copyJson,
    downloadJson: hiveStore.downloadJson,
  };
}
