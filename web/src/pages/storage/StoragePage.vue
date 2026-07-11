<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'

import { useRuntimeStore } from '@/shared/model/runtime-store'
import StatusBadge from '@/widgets/app-shell/StatusBadge.vue'

const runtime = useRuntimeStore()
const route = useRoute()
const selectedArtifactId = ref('')
const loading = ref(false)
const pageError = ref<string>()

const artifact = computed(() =>
  selectedArtifactId.value ? runtime.artifacts[selectedArtifactId.value] : undefined,
)

function formatBytes(value: number) {
  return value < 1024 ? value + ' B' : (value / 1024).toFixed(1) + ' KiB'
}

async function openArtifact(artifactId: string) {
  if (!artifactId) {
    return
  }
  selectedArtifactId.value = artifactId
  loading.value = true
  pageError.value = undefined
  try {
    await runtime.loadArtifact(artifactId)
  } catch (error) {
    pageError.value = error instanceof Error ? error.message : 'Не удалось загрузить артефакт.'
  } finally {
    loading.value = false
  }
}

watch(
  () => route.params.artifactId,
  (artifactId) => {
    if (typeof artifactId === 'string') {
      void openArtifact(artifactId)
    }
  },
  { immediate: true },
)

onMounted(() => {
  if (!selectedArtifactId.value && runtime.mode === 'mock') {
    void openArtifact('artifact-unity-docs-001')
  }
})
</script>

<template>
  <div class="page">
    <header class="page-heading">
      <div>
        <p class="eyebrow">Object storage</p>
        <h1>Хранилище</h1>
        <p>
          Артефакт — отдельный типизированный объект с checksum, access scope и версией.
          Первый API-срез возвращает только метаданные: содержимое не загружается в UI
          неявно и не становится вторым источником истины.
        </p>
      </div>
      <StatusBadge v-if="artifact" status="verified" label="metadata only" />
    </header>

    <section class="surface">
      <div class="surface__body storage-picker">
        <label class="field">
          <span>Artifact ID</span>
          <input
            :value="selectedArtifactId || 'artifact-unity-docs-001'"
            :disabled="loading"
            autocomplete="off"
            @change="openArtifact(($event.target as HTMLInputElement).value)"
          />
        </label>
        <div class="inline-actions">
          <button class="button button--secondary" type="button" @click="openArtifact('artifact-unity-docs-001')">
            Исходная документация
          </button>
          <button class="button button--secondary" type="button" @click="openArtifact('artifact-user-message-001')">
            Входное сообщение
          </button>
        </div>
      </div>
    </section>

    <div v-if="pageError" class="state-message state-message--error" role="alert">{{ pageError }}</div>

    <template v-if="artifact">
      <div class="split-grid">
        <section class="surface">
          <header class="surface__header">
            <div>
              <p class="eyebrow">{{ artifact.mediaType }}</p>
              <h2>{{ artifact.label }}</h2>
            </div>
            <StatusBadge status="verified" label="metadata only" />
          </header>
          <div class="surface__body">
            <dl class="metadata">
              <div><dt>Artifact ID</dt><dd>{{ artifact.id }}</dd></div>
              <div><dt>Schema</dt><dd>{{ artifact.schema }} · {{ artifact.version }}</dd></div>
              <div><dt>Checksum</dt><dd>{{ artifact.contentHash }}</dd></div>
              <div><dt>Размер</dt><dd>{{ formatBytes(artifact.sizeBytes) }}</dd></div>
              <div><dt>Tenant / access</dt><dd>{{ artifact.tenantId }} · {{ artifact.accessScope }}</dd></div>
              <div><dt>Создан</dt><dd>{{ artifact.createdAt }}</dd></div>
            </dl>
          </div>
        </section>

        <section class="surface">
          <header class="surface__header">
            <div>
              <p class="eyebrow">Access boundary</p>
              <h2>Политика доступа</h2>
            </div>
          </header>
          <div class="surface__body versions">
            <article>
              <strong>{{ artifact.accessScope }}</strong>
              <span>Проверка доступа и контроль целостности выполняются backend до выдачи метаданных.</span>
              <small>История версий и preview требуют отдельных versioned endpoints.</small>
            </article>
          </div>
        </section>
      </div>

      <section class="surface">
        <header class="surface__header">
          <div>
            <p class="eyebrow">Safe preview</p>
            <h2>Ограниченное содержимое</h2>
          </div>
          <span class="muted">нет content endpoint</span>
        </header>
        <div class="surface__body">
          <p class="muted">
            Endpoint metadata намеренно возвращает только ArtifactRef. Когда появится
            отдельный ограниченный preview endpoint, он будет подключён как новый typed
            transport contract.
          </p>
        </div>
      </section>
    </template>
  </div>
</template>

<style scoped lang="scss">
.storage-picker {
  display: flex;
  flex-wrap: wrap;
  align-items: end;
  gap: 1rem;

  .field {
    width: min(100%, 30rem);
  }
}

.metadata {
  display: grid;
  gap: 0.7rem;
  margin: 0;

  div {
    display: grid;
    gap: 0.18rem;
    padding-bottom: 0.62rem;
    border-bottom: 1px solid rgba(168, 190, 228, 0.1);
  }

  dt {
    color: #91a5c2;
    font-size: 0.73rem;
    text-transform: uppercase;
  }

  dd {
    overflow-wrap: anywhere;
    margin: 0;
    color: #dce7f8;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.79rem;
  }
}

.versions {
  display: grid;
  gap: 0.65rem;

  article {
    display: grid;
    gap: 0.25rem;
    padding: 0.72rem;
    border: 1px solid rgba(168, 190, 228, 0.13);
    border-radius: 0.62rem;
    background: rgba(6, 16, 31, 0.3);
  }

  strong {
    color: #b8d3ff;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.8rem;
  }

  span,
  small {
    color: #bdcbe0;
    font-size: 0.8rem;
  }

  small {
    color: #8698b6;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.71rem;
  }
}
</style>
