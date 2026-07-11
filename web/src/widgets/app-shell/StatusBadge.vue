<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    status: string
    label?: string
  }>(),
  {
    label: undefined,
  },
)

const normalizedStatus = computed(() => props.status.toLowerCase().replace(/_/g, '-'))

const readableStatus = computed(() => {
  const labels: Record<string, string> = {
    succeeded: 'успешно',
    ok: 'готово',
    running: 'выполняется',
    queued: 'в очереди',
    failed: 'ошибка',
    cancelled: 'отменено',
    'dead-letter': 'dead letter',
    degraded: 'ограничено',
    offline: 'offline',
    verified: 'проверено',
    pending: 'ожидает',
    corrupt: 'повреждено',
    active: 'активен',
    idle: 'ожидает',
    frozen: 'заморожен',
    completed: 'завершён',
    archived: 'в архиве',
  }
  return props.label ?? labels[normalizedStatus.value] ?? props.status
})
</script>

<template>
  <span class="status-badge" :class="'status-badge--' + normalizedStatus">
    <span class="status-dot" aria-hidden="true" />
    {{ readableStatus }}
  </span>
</template>

<style scoped lang="scss">
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  width: fit-content;
  border: 1px solid currentColor;
  border-radius: 999px;
  padding: 0.24rem 0.52rem;
  color: #b9c9e3;
  background: rgba(153, 175, 210, 0.09);
  font-size: 0.75rem;
  font-weight: 700;
  line-height: 1;
  white-space: nowrap;

  &--succeeded,
  &--ok,
  &--verified,
  &--active,
  &--completed {
    color: #70e0bc;
    background: rgba(62, 188, 143, 0.12);
  }

  &--running,
  &--queued,
  &--pending,
  &--idle {
    color: #8fc3ff;
    background: rgba(73, 144, 233, 0.13);
  }

  &--failed,
  &--corrupt,
  &--offline,
  &--dead-letter {
    color: #ff9eab;
    background: rgba(215, 70, 89, 0.12);
  }

  &--cancelled,
  &--degraded,
  &--frozen,
  &--archived {
    color: #ffcf7d;
    background: rgba(211, 154, 52, 0.12);
  }
}
</style>
