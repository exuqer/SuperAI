<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'

import { useRuntimeStore } from '@/shared/model/runtime-store'
import StatusBadge from '@/widgets/app-shell/StatusBadge.vue'

const runtime = useRuntimeStore()
const sectorFilter = ref('')
const timeFilter = ref<'all' | 'current'>('all')
const selectedConceptId = ref('')

const sectors = computed(() =>
  [...new Set(runtime.cosmos.flatMap((concept) => concept.sectors))].sort(),
)

const concepts = computed(() =>
  runtime.cosmos.filter((concept) => {
    if (sectorFilter.value && !concept.sectors.includes(sectorFilter.value)) {
      return false
    }
    if (timeFilter.value === 'current' && !concept.claims.some((claim) => claim.validFrom)) {
      return false
    }
    return true
  }),
)

const selectedConcept = computed(
  () =>
    concepts.value.find((concept) => concept.id === selectedConceptId.value) ??
    concepts.value[0],
)

onMounted(async () => {
  await runtime.loadCosmos()
  if (!selectedConceptId.value) {
    selectedConceptId.value = runtime.cosmos[0]?.id ?? ''
  }
})
</script>

<template>
  <div class="page">
    <header class="page-heading">
      <div>
        <p class="eyebrow">Cosmos view</p>
        <h1>Космос</h1>
        <p>
          Сектора — логические представления общего графа. Claims не схлопываются в
          единственную «истину» и всегда показывают provenance и независимые оценки.
        </p>
      </div>
      <StatusBadge status="verified" label="read model" />
    </header>

    <section class="surface">
      <div class="surface__body cosmos-filters">
        <label class="field">
          <span>Сектор</span>
          <select v-model="sectorFilter">
            <option value="">Все секторы</option>
            <option v-for="sector in sectors" :key="sector" :value="sector">{{ sector }}</option>
          </select>
        </label>
        <label class="field">
          <span>Время</span>
          <select v-model="timeFilter">
            <option value="all">Все утверждения</option>
            <option value="current">Только с периодом действия</option>
          </select>
        </label>
        <span class="muted">{{ concepts.length }} concepts</span>
      </div>
    </section>

    <section v-if="concepts.length" class="cosmos-layout">
      <aside class="surface concept-list">
        <header class="surface__header">
          <div>
            <p class="eyebrow">Concepts</p>
            <h2>Выбор узла</h2>
          </div>
        </header>
        <div class="concept-list__body">
          <button
            v-for="concept in concepts"
            :key="concept.id"
            class="concept-option"
            :class="{ 'concept-option--selected': selectedConcept?.id === concept.id }"
            type="button"
            @click="selectedConceptId = concept.id"
          >
            <strong>{{ concept.label }}</strong>
            <span>{{ concept.type }} · {{ concept.sectors.join(', ') }}</span>
          </button>
        </div>
      </aside>

      <template v-if="selectedConcept">
        <section class="surface concept-detail">
          <header class="surface__header">
            <div>
              <p class="eyebrow">{{ selectedConcept.type }}</p>
              <h2>{{ selectedConcept.label }}</h2>
            </div>
            <span class="concept-id">{{ selectedConcept.id }}</span>
          </header>
          <div class="surface__body concept-detail__body">
            <div class="chip-row">
              <span v-for="sector in selectedConcept.sectors" :key="sector" class="chip">{{ sector }}</span>
              <span v-for="alias in selectedConcept.aliases" :key="alias" class="chip chip--muted">{{ alias }}</span>
            </div>

            <div class="local-graph" aria-label="Локальный граф вокруг выбранного понятия">
              <div class="graph-node graph-node--center">{{ selectedConcept.label }}</div>
              <div class="graph-neighbours">
                <article v-for="neighbour in selectedConcept.neighbours" :key="neighbour.conceptId">
                  <span class="graph-edge">{{ neighbour.relation }}</span>
                  <strong>{{ neighbour.label }}</strong>
                </article>
              </div>
            </div>

            <div class="data-table-wrap">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Утверждение</th>
                    <th>Статус</th>
                    <th>Provenance</th>
                    <th>Оценки</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="claim in selectedConcept.claims" :key="claim.id">
                    <td>
                      <strong>{{ claim.subject }} {{ claim.predicate }}</strong>
                      <small class="table-subvalue">{{ claim.object }}</small>
                    </td>
                    <td><StatusBadge :status="claim.verificationStatus" /></td>
                    <td>
                      <RouterLink :to="{ name: 'storage', params: { artifactId: claim.sourceArtifactId } }">
                        {{ claim.sourceArtifactId }}
                      </RouterLink>
                      <small class="table-subvalue">{{ claim.sourceFragment }}</small>
                      <small class="table-subvalue">{{ claim.accessScope }}</small>
                    </td>
                    <td>
                      <span class="score-line">confidence {{ claim.scores.confidence }}</span>
                      <span class="score-line">relevance {{ claim.scores.relevance }}</span>
                      <span class="score-line">contradiction {{ claim.scores.contradiction }}</span>
                    </td>
                  </tr>
                  <tr v-if="!selectedConcept.claims.length">
                    <td class="muted" colspan="4">Для выбранного узла claims пока не загружены.</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </template>
    </section>
    <section v-else class="state-message">
      В выбранном источнике пока нет доступных concepts. В mock режиме откройте fixture «Успешный ответ».
    </section>
  </div>
</template>

<style scoped lang="scss">
.cosmos-filters {
  display: flex;
  flex-wrap: wrap;
  align-items: end;
  gap: 0.85rem;

  .field {
    min-width: 13rem;
  }

  > .muted {
    padding-bottom: 0.65rem;
    font-size: 0.84rem;
  }
}

.cosmos-layout {
  display: grid;
  grid-template-columns: minmax(15rem, 0.35fr) minmax(0, 1fr);
  gap: 1.25rem;

  @media (max-width: 850px) {
    grid-template-columns: 1fr;
  }
}

.concept-list__body {
  display: grid;
  gap: 0.32rem;
  padding: 0.55rem;
}

.concept-option {
  display: grid;
  gap: 0.22rem;
  border: 1px solid transparent;
  border-radius: 0.62rem;
  color: #cbd8eb;
  background: transparent;
  padding: 0.72rem;
  text-align: left;

  &:hover {
    background: rgba(115, 160, 232, 0.09);
  }

  &--selected {
    border-color: rgba(116, 172, 255, 0.32);
    background: rgba(69, 130, 224, 0.16);
  }

  strong {
    font-size: 0.84rem;
  }

  span {
    color: #8fa1bd;
    font-size: 0.73rem;
  }
}

.concept-id {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.72rem;
  color: #91a5c4;
}

.concept-detail__body {
  display: grid;
  gap: 1rem;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.chip {
  border-radius: 999px;
  color: #b6d5ff;
  background: rgba(71, 131, 220, 0.17);
  padding: 0.26rem 0.48rem;
  font-size: 0.72rem;

  &--muted {
    color: #a7b5ca;
    background: rgba(154, 173, 203, 0.11);
  }
}

.local-graph {
  display: grid;
  grid-template-columns: minmax(9rem, 0.55fr) minmax(0, 1fr);
  align-items: center;
  gap: 1rem;
  min-height: 10rem;
  padding: 1rem;
  border: 1px solid rgba(168, 190, 228, 0.15);
  border-radius: 0.75rem;
  background:
    linear-gradient(rgba(97, 141, 216, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(97, 141, 216, 0.05) 1px, transparent 1px),
    rgba(7, 16, 30, 0.38);
  background-size: 1.6rem 1.6rem;

  @media (max-width: 600px) {
    grid-template-columns: 1fr;
  }
}

.graph-node {
  display: grid;
  place-items: center;
  min-height: 4.5rem;
  border: 1px solid rgba(121, 182, 255, 0.58);
  border-radius: 0.8rem;
  color: #dfeeff;
  background: rgba(48, 109, 208, 0.22);
  font-weight: 750;
  text-align: center;
}

.graph-neighbours {
  display: grid;
  gap: 0.45rem;

  article {
    display: grid;
    grid-template-columns: 7rem minmax(0, 1fr);
    gap: 0.5rem;
    align-items: center;
  }

  strong {
    color: #d5e1f4;
    font-size: 0.83rem;
  }
}

.graph-edge {
  overflow: hidden;
  color: #90a9ca;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.69rem;
  text-align: right;
  text-overflow: ellipsis;
  white-space: nowrap;

  &::after {
    content: " ──›";
  }
}

.table-subvalue,
.score-line {
  display: block;
  margin-top: 0.18rem;
  color: #8e9fba;
  font-size: 0.68rem;
}

.score-line {
  font-family: "SFMono-Regular", Consolas, monospace;
}

.data-table a {
  color: #91c1ff;
}
</style>
