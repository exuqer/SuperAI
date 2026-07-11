<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    value?: unknown
    emptyLabel?: string
  }>(),
  {
    value: undefined,
    emptyLabel: 'Нет данных',
  },
)

const formatted = computed(() =>
  props.value === undefined ? props.emptyLabel : JSON.stringify(props.value, null, 2),
)
</script>

<template>
  <pre class="code-block json-viewer">{{ formatted }}</pre>
</template>

<style scoped>
.json-viewer {
  min-height: 5rem;
}
</style>
