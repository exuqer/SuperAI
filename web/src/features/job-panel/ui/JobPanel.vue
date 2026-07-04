<template>
  <section class="panel compact jobs">
    <div class="row">
      <h3>Задачи</h3>
      <button type="button" @click="runtime.refreshJobs()">Обновить</button>
    </div>
    <div v-if="!runtime.jobs.length" class="muted">Нет задач.</div>
    <article v-for="job in runtime.jobs" :key="job.job_id" class="job">
      <div class="row">
        <strong>{{ job.name }}</strong>
        <span class="badge" :class="{ signal: job.status === 'failed' }">{{ job.status }}</span>
      </div>
      <div class="muted">{{ job.job_id }} {{ formatTime(job.created_at) }}</div>
      <p v-if="job.error" class="error">{{ job.error }}</p>
      <pre v-if="job.result">{{ compactJson(job.result) }}</pre>
    </article>
  </section>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import { useRuntimeStore } from '@/app/stores/runtime';
import { compactJson, formatTime } from '@/shared/lib/format';

const runtime = useRuntimeStore();

onMounted(() => {
  runtime.refreshJobs().catch(() => undefined);
});
</script>

<style scoped lang="scss">
.jobs {
  display: grid;
  gap: 10px;
}

h3,
p {
  margin: 0;
}

.job {
  display: grid;
  gap: 6px;
  border-top: 1px solid var(--line);
  padding-top: 10px;
}

.error {
  color: var(--signal);
}
</style>
