<template>
  <section class="page">
    <div class="split">
      <section class="panel">
        <h2>Система</h2>
        <div class="tools">
          <article class="tool">
            <h3>Bootstrap</h3>
            <label class="checkbox">
              <input v-model="forceBootstrap" type="checkbox" />
              force
            </label>
            <button type="button" @click="runBootstrap">Запустить</button>
          </article>
          <article class="tool">
            <h3>Dream</h3>
            <label>
              Steps
              <input v-model.number="dreamSteps" type="number" min="1" />
            </label>
            <button type="button" @click="runDream">Запустить</button>
          </article>
          <article class="tool">
            <h3>Eval</h3>
            <label>
              Path JSONL
              <input v-model="evalPath" placeholder="data/examples.jsonl" />
            </label>
            <textarea v-model="evalJsonl" placeholder="или inline JSONL" />
            <button type="button" @click="runEval">Запустить</button>
          </article>
          <article class="tool">
            <h3>SPC dataset</h3>
            <label>
              Split
              <select v-model="spcSplit">
                <option>train</option>
                <option>dev</option>
                <option>valid</option>
                <option>test</option>
                <option>synth</option>
              </select>
            </label>
            <label>
              Limit
              <input v-model="spcLimit" placeholder="optional" />
            </label>
            <label>
              Output
              <input v-model="spcOutput" placeholder="data/spc_dialogues.jsonl" />
            </label>
            <button type="button" @click="downloadSpc">Скачать</button>
          </article>
          <article class="tool">
            <h3>Export checkpoint</h3>
            <label>
              Destination
              <input v-model="exportDestination" placeholder=".semantic_ants/export/model.json" />
            </label>
            <button type="button" @click="exportCheckpoint">Экспорт</button>
          </article>
        </div>
      </section>
      <JobPanel />
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useRuntimeStore } from '@/app/stores/runtime';
import JobPanel from '@/features/job-panel/ui/JobPanel.vue';
import { api } from '@/shared/api/client';

const runtime = useRuntimeStore();
const forceBootstrap = ref(false);
const dreamSteps = ref(100);
const evalPath = ref('data/examples.jsonl');
const evalJsonl = ref('');
const spcSplit = ref('train');
const spcLimit = ref('');
const spcOutput = ref('data/spc_dialogues.jsonl');
const exportDestination = ref('.semantic_ants/export/model.json');

async function track(promise: Promise<Awaited<ReturnType<typeof api.dream>>>) {
  const job = await promise;
  await runtime.trackJob(job);
}

function runBootstrap() {
  void track(api.bootstrap({ force: forceBootstrap.value }));
}

function runDream() {
  void track(api.dream({ steps: dreamSteps.value }));
}

function runEval() {
  void track(api.eval({ path: evalPath.value || undefined, jsonl: evalJsonl.value || undefined }));
}

function downloadSpc() {
  void track(
    api.downloadSpc({
      split: spcSplit.value,
      limit: spcLimit.value ? Number(spcLimit.value) : undefined,
      output: spcOutput.value,
    }),
  );
}

function exportCheckpoint() {
  void track(api.exportCheckpoint({ destination: exportDestination.value }));
}
</script>

<style scoped lang="scss">
h2,
h3 {
  margin-top: 0;
}

.tools {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
}

.tool {
  display: grid;
  gap: 8px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 12px;
}

.checkbox {
  display: flex;
  gap: 8px;
  align-items: center;
}

.checkbox input {
  width: auto;
}
</style>
