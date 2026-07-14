import { computed, ref } from 'vue';
import { api } from '@/shared/api/client';
import { storage } from '@/shared/storage';
import type { HiveAnalyticsResponse } from '@/entities/hive/types';

export function useHiveAnalytics() {
  const hiveId = ref(storage.getActiveHive());
  const data = ref<HiveAnalyticsResponse | null>(null);
  const loading = ref(false);
  const error = ref('');
  const primaryRunId = ref('');
  const comparisonRunId = ref('');

  const hasHive = computed(() => Boolean(hiveId.value));

  async function load() {
    hiveId.value = storage.getActiveHive();
    if (!hiveId.value) {
      data.value = null;
      return;
    }
    loading.value = true;
    error.value = '';
    try {
      const params: Record<string, string> = {};
      if (primaryRunId.value) params.run_id = primaryRunId.value;
      if (comparisonRunId.value) params.compare_run_id = comparisonRunId.value;
      data.value = await api.get<HiveAnalyticsResponse>(`/api/v2/hives/${hiveId.value}/analytics`, params);
      primaryRunId.value = data.value.primary?.run.id || '';
      comparisonRunId.value = data.value.comparison?.run.id || '';
    } catch (cause) {
      data.value = null;
      error.value = cause instanceof Error ? cause.message : 'Не удалось загрузить аналитику.';
    } finally {
      loading.value = false;
    }
  }

  return {
    hiveId,
    data,
    loading,
    error,
    primaryRunId,
    comparisonRunId,
    hasHive,
    load,
  };
}
