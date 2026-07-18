/** Training state for V2.7 events and learned slots. */

import { computed, ref } from 'vue';
import { defineStore } from 'pinia';
import { api } from '@/shared/api/client';
import { storage } from '@/shared/storage';
import type { TrainingResponse } from './types';

export const useGraphTrainingStore = defineStore('graph-training-v27', () => {
  const result = ref<TrainingResponse | null>(null);
  const loading = ref(false);
  const error = ref('');
  const text = ref(storage.getLastTrainingText());
  const sourceType = ref('training');
  const domainKey = ref('');
  const independentKey = ref('');

  const eventCount = computed(() => result.value?.events.length || 0);
  const participantCount = computed(() =>
    (result.value?.events || []).reduce(
      (total, event) => total + event.participants.length,
      0,
    ),
  );
  const slotCount = computed(() => result.value?.local_slots?.length || 0);
  const prototypeCount = computed(() => result.value?.slot_prototypes?.length || 0);

  async function learn(): Promise<TrainingResponse> {
    const normalized = text.value.trim();
    if (!normalized) throw new Error('Добавьте текст для обучения');
    loading.value = true;
    error.value = '';
    storage.setLastTrainingText(text.value);
    try {
      result.value = await api.post<TrainingResponse>('/api/v2/training/learn', {
        text: normalized,
        source_type: sourceType.value.trim() || 'training',
        domain_key: domainKey.value.trim(),
        independent_key: independentKey.value.trim(),
      });
      return result.value;
    } catch (cause) {
      error.value = cause instanceof Error ? cause.message : String(cause);
      throw cause;
    } finally {
      loading.value = false;
    }
  }

  return {
    result,
    loading,
    error,
    text,
    sourceType,
    domainKey,
    independentKey,
    eventCount,
    participantCount,
    slotCount,
    prototypeCount,
    learn,
  };
});
