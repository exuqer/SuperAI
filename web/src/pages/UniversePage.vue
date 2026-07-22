<template>
  <main class="universe-page">
    <header class="topbar">
      <RouterLink class="brand" :to="{ name: 'chat' }">SuperAI <small>dynamic memory</small></RouterLink>
      <nav>
        <RouterLink :to="{ name: 'chat' }">Диалог</RouterLink>
        <RouterLink :to="{ name: 'universe' }">Пространство</RouterLink>
        <button class="export-button" :disabled="exporting" @click="exportMemory">
          {{ exporting ? 'Копирование…' : exportStatus || 'Копировать память' }}
        </button>
        <button class="reset-button" :disabled="resetting" @click="openResetDialog">
          {{ resetting ? 'Очистка…' : 'Очистить тестовое пространство' }}
        </button>
      </nav>
    </header>

    <section class="workspace">
      <aside class="panel navigator">
        <div class="kicker">МИКРОВСЕЛЕННЫЕ</div>
        <button
          v-for="item in universes"
          :key="item.id"
          :class="{ selected: item.id === selectedUniverse }"
          @click="selectUniverse(item.id)"
        >
          <strong>{{ item.name }}</strong>
          <span>{{ item.entity_count }} сущн. · {{ item.active_dimension_count }} D</span>
        </button>
      </aside>

      <section class="panel canvas-panel">
        <div class="panel-head">
          <div>
            <div class="kicker">{{ selectedDimensions.length ? 'DIMENSION PROJECTION' : 'SEMANTIC FIELD' }}</div>
            <h1>{{ selectedDimensions.length ? dimensionTitle : (activeUniverse?.name || 'Пространство') }}</h1>
          </div>
          <span>{{ displayPoints.length }} сущностей</span>
        </div>
        <form class="training-form" @submit.prevent="learn">
          <label for="training-text">Текст для обучения</label>
          <textarea
            id="training-text"
            v-model="trainingText"
            rows="3"
            :disabled="training"
            placeholder="Например: Яблоко растёт в саду."
          />
          <div class="training-actions">
            <span v-if="trainingStatus">{{ trainingStatus }}</span>
            <div>
              <button type="button" class="clear-draft" :disabled="training || !trainingText" @click="trainingText = ''; trainingStatus = ''">Очистить текст</button>
              <button type="submit" :disabled="training || !trainingText.trim()">{{ training ? 'Обучение…' : 'Добавить в память' }}</button>
            </div>
          </div>
        </form>
        <p class="notice">
          <template v-if="selectedDimensions.length === 2">X — {{ compact(selectedDimensions[0]) }}, Y — {{ compact(selectedDimensions[1]) }}. Близкие точки имеют похожие проекции в обоих измерениях.</template>
          <template v-else-if="selectedDimensions.length === 1">Показана сила принадлежности к {{ compact(selectedDimensions[0]) }}; выберите второе D для пересечения.</template>
          <template v-else>Semantic Field revision {{ semanticField.field_revision }} · {{ semanticField.projection_method }} · {{ semanticField.display_projection_warning }}</template>
        </p>
        <div class="map" aria-label="Базовое пространство">
          <button
            v-for="point in displayPoints"
            :key="point.id"
            class="entity"
            :class="{ active: point.id === selectedEntity?.entity.id || point.id === selectedFieldCloud?.cloud_id }"
            :style="position(point)"
            :title="point.label"
            @click="openEntity(point.id)"
          >{{ point.label }}</button>
          <p v-if="!displayPoints.length" class="empty">Обучите модель, чтобы появились наблюдения.</p>
        </div>
      </section>

      <aside class="panel inspector">
        <div class="kicker">{{ selectedFieldCloud ? 'SEMANTIC CLOUD' : selectedEntity ? 'ПРОФИЛЬ СУЩНОСТИ' : 'LATENT DIMENSIONS' }}</div>
        <template v-if="selectedFieldCloud">
          <h2>{{ selectedFieldCloud.canonical_lemma }}</h2>
          <p>cloud {{ selectedFieldCloud.cloud_id }}</p>
          <dl class="cloud-detail">
            <dt>Mass / density</dt><dd>{{ selectedFieldCloud.mass.toFixed(2) }} / {{ selectedFieldCloud.density.toFixed(2) }}</dd>
            <dt>Halo / stability</dt><dd>{{ selectedFieldCloud.halo.toFixed(2) }} / {{ selectedFieldCloud.stability.toFixed(2) }}</dd>
            <dt>Permeability</dt><dd>{{ selectedFieldCloud.permeability.toFixed(2) }}</dd>
            <dt>Position</dt><dd>{{ selectedFieldCloud.position_status }} · {{ selectedFieldCloud.projection_status }}</dd>
            <dt>Bootstrap</dt><dd>{{ selectedFieldCloud.bootstrap_center.join(', ') }}</dd>
            <dt>Learned display</dt><dd>{{ selectedFieldCloud.learned_center.join(', ') }}</dd>
            <dt>Latent dimensions</dt><dd>{{ selectedFieldCloud.active_dimensions.join(', ') || 'нет' }}</dd>
            <dt>Sources / events</dt><dd>{{ selectedFieldCloud.supporting_source_ids.length }} / {{ selectedFieldCloud.supporting_event_ids.length }}</dd>
            <dt>Contexts</dt><dd>{{ selectedFieldCloud.contextual_projections.length }}</dd>
          </dl>
          <h3>Force breakdown</h3>
          <div v-for="(force, index) in selectedFieldCloud.force_breakdown" :key="`${force.force_type}-${index}`" class="force-row">
            <strong>{{ force.force_type }}</strong>
            <span>{{ force.magnitude.toFixed(4) }} · {{ force.source_cloud_id || 'self' }}</span>
          </div>
          <p v-if="!selectedFieldCloud.force_breakdown.length" class="empty-inline">Нет активных сил в этой revision.</p>
          <button class="show-dimensions" @click="selectedFieldCloud = null">Закрыть cloud inspector</button>
        </template>
        <template v-else-if="selectedEntity">
          <h2>{{ selectedEntity.entity.label }}</h2>
          <p>масса {{ selectedEntity.entity.mass.toFixed(2) }} · устойчивость {{ selectedEntity.entity.stability.toFixed(2) }}</p>
          <h3>Параллельные проекции</h3>
          <button v-for="projection in selectedEntity.stable_dimensions.slice(0, 10)" :key="projection.id" class="projection">
            <span>{{ compact(projection.dimension_id) }}</span><i><b :style="{ width: `${projection.membership * 100}%` }" /></i><em>{{ projection.membership.toFixed(2) }}</em>
          </button>
          <template v-if="selectedUniverse === 'words'">
            <h3>Формы: {{ selectedEntity.word_forms?.length || 0 }} · употребления: {{ selectedEntity.usage_count ?? selectedEntity.occurrence_distribution.length }}</h3>
            <button v-for="form in selectedEntity.word_forms || []" :key="form.id" class="word-form" @click="openEntity(form.id)">
              <span>{{ form.surface }}</span><em>{{ form.usage_count }} употр.</em>
            </button>
            <button class="show-word-forms" @click="selectUniverse('word_forms')">Показать словоформы</button>
          </template>
          <h3 v-else>Употребления: {{ selectedEntity.occurrence_distribution.length }}</h3>
          <button class="show-dimensions" @click="selectedEntity = null">Выбрать измерения</button>
        </template>
        <template v-else>
          <p>Semantic Field: {{ semanticField.clouds.length }} clouds · revision {{ semanticField.field_revision }}.</p>
          <p class="dimension-help">Выберите до двух D — их проекции образуют карту пересечений. Повторный клик снимает выбор.</p>
          <button v-for="dimension in dimensions" :key="dimension.id" class="dimension" :class="{ selected: selectedDimensions.includes(dimension.id) }" @click="toggleDimension(dimension.id)">
            <strong>{{ compact(dimension.id) }} <small v-if="dimension.alias">— {{ dimension.alias }}</small></strong>
            <span>{{ selectedDimensions.includes(dimension.id) ? '✓ на карте · ' : '' }}{{ dimension.representation_type }} · {{ dimension.status }} · {{ dimension.utility.toFixed(2) }}</span>
          </button>
          <section v-if="dimensionDetail" class="dimension-detail">
            <h3>Проверка измерения</h3>
            <p>
              revision {{ dimensionDetail.metadata.revision }} ·
              {{ dimensionDetail.metadata.status }}
            </p>
            <dl>
              <dt>Semantic basis</dt>
              <dd>{{ dimensionDetail.semantic_basis.residual_feature || '—' }}</dd>
              <dt>Control features</dt>
              <dd>{{ dimensionDetail.control_features.join(', ') || 'нет' }}</dd>
              <dt>Support train / holdout / continual</dt>
              <dd>{{ dimensionDetail.metadata.train_support }} / {{ dimensionDetail.metadata.holdout_support }} / {{ dimensionDetail.metadata.continual_support }}</dd>
              <dt>Stability / lower bound</dt>
              <dd>{{ dimensionDetail.metadata.stability.toFixed(2) }} / {{ dimensionDetail.metadata.stability_lower_bound.toFixed(2) }}</dd>
              <dt>Retrieval / shadow gain</dt>
              <dd>{{ dimensionDetail.metadata.holdout_retrieval_gain.toFixed(2) }} / {{ dimensionDetail.metadata.shadow_retrieval_gain.toFixed(2) }}</dd>
              <dt>Validated utility</dt>
              <dd>{{ dimensionDetail.metadata.validated_answer_contribution_count }}</dd>
              <dt>Lineage</dt>
              <dd>{{ dimensionDetail.lineage?.lineage_reason || 'первая revision' }}</dd>
              <dt>Relations</dt>
              <dd>{{ dimensionDetail.related_dimensions.length }}</dd>
            </dl>
          </section>
          <p v-if="!dimensions.length" class="empty">Кандидаты появятся после повторяющихся наблюдений.</p>
        </template>
        <p v-if="error" class="error">{{ error }}</p>
      </aside>
    </section>

    <div v-if="resetOpen" class="reset-backdrop" role="presentation" @click.self="closeResetDialog">
      <section class="reset-dialog" role="dialog" aria-modal="true" aria-labelledby="reset-title">
        <div class="kicker">ТЕСТОВОЕ ОКРУЖЕНИЕ</div>
        <h2 id="reset-title">Очистить тестовое пространство</h2>
        <p>
          Будут удалены обученные источники, граф событий, семантическое поле,
          микровселенные, измерения, диалоги и тестовые трассы. Действие необратимо.
        </p>
        <label>
          Область очистки
          <select v-model="resetScope" :disabled="resetting">
            <option value="FULL_TEST_STATE">Полный тестовый сброс</option>
            <option value="DERIVED_SEMANTIC_SPACE">Только производное пространство</option>
          </select>
        </label>
        <p v-if="resetError" class="error">{{ resetError }}</p>
        <div class="reset-actions">
          <button type="button" class="cancel" :disabled="resetting" @click="closeResetDialog">Отмена</button>
          <button
            type="button"
            class="danger"
            :disabled="resetting"
            @click="resetTestSpace"
          >{{ resetting ? 'Очищаю…' : 'Очистить пространство' }}</button>
        </div>
      </section>
    </div>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { api } from '@/shared/api/client';
import { copyJsonToClipboard } from '@/shared/utils/clipboard';
import { useGraphStore } from '@/entities/graph/store';
import { useGraphTrainingStore } from '@/entities/graph/trainingStore';
import { useHiveStore } from '@/entities/hive/store';
import { useModelStore } from '@/entities/model/store';

type Universe = { id: string; name: string; entity_count: number; active_dimension_count: number };
type Entity = { id: string; label: string; mass: number; stability: number; base_position: number[] };
type Dimension = { id: string; alias?: string | null; representation_type: string; status: string; utility: number };
type DimensionDetail = {
  metadata: Dimension & {
    revision: number;
    train_support: number;
    holdout_support: number;
    continual_support: number;
    stability: number;
    stability_lower_bound: number;
    holdout_retrieval_gain: number;
    shadow_retrieval_gain: number;
    validated_answer_contribution_count: number;
  };
  semantic_basis: Record<string, string>;
  control_features: string[];
  lineage: { lineage_reason?: string } | null;
  related_dimensions: unknown[];
};
type Projection = { id: string; dimension_id: string; membership: number };
type WordForm = { id: string; surface: string; usage_count: number; morphological_features: Record<string, unknown> };
type Profile = {
  entity: Entity;
  stable_dimensions: Projection[];
  occurrence_distribution: unknown[];
  canonical_lemma?: string;
  usage_count?: number;
  word_forms?: WordForm[];
};
type ProjectionPoint = { id: string; label: string; x: number; y: number; projections: Record<string, number> };
type FieldForce = { force_type: string; source_cloud_id?: string | null; magnitude: number; vector: number[]; evidence_ids: string[]; payload: Record<string, unknown> };
type FieldCloud = {
  cloud_id: string; concept_id: string; canonical_lemma: string;
  bootstrap_center: number[]; learned_center: number[]; previous_revision_center: number[];
  proposed_center: number[]; center: number[]; display_center: number[];
  halo: number; mass: number; density: number; stability: number; permeability: number;
  active_dimensions: string[]; position_status: string; projection_status: string;
  validation: Record<string, unknown>; contextual_projections: Array<Record<string, unknown>>;
  supporting_source_ids: string[]; supporting_event_ids: string[]; force_breakdown: FieldForce[];
};
type SemanticField = { field_revision: number; projection_method: string; display_projection_warning: string; revision_history: Array<Record<string, unknown>>; clouds: FieldCloud[] };

type ResetScope = 'FULL_TEST_STATE' | 'DERIVED_SEMANTIC_SPACE';
type ResetReport = {
  reset: boolean;
  scope: ResetScope;
  mode: 'FRESH_SCHEMA' | 'CLEAR_DATA';
  database_generation_id: string;
  field_revision: number;
  invariants: Record<string, boolean>;
};

const universes = ref<Universe[]>([]);
const selectedUniverse = ref('words');
const base = ref<{ entities: Entity[] }>({ entities: [] });
const dimensions = ref<Dimension[]>([]);
const dimensionDetail = ref<DimensionDetail | null>(null);
const selectedEntity = ref<Profile | null>(null);
const selectedFieldCloud = ref<FieldCloud | null>(null);
const selectedDimensions = ref<string[]>([]);
const projectionPoints = ref<ProjectionPoint[]>([]);
const semanticField = ref<SemanticField>({ field_revision: 0, projection_method: 'none', display_projection_warning: 'Display projection — not semantic distance.', revision_history: [], clouds: [] });
const error = ref('');
const trainingText = ref('');
const training = ref(false);
const trainingStatus = ref('');
const exporting = ref(false);
const exportStatus = ref('');
const resetOpen = ref(false);
const resetting = ref(false);
const resetScope = ref<ResetScope>('FULL_TEST_STATE');
const resetError = ref('');
const graphStore = useGraphStore();
const trainingStore = useGraphTrainingStore();
const hiveStore = useHiveStore();
const modelStore = useModelStore();
const activeUniverse = computed(() => universes.value.find(item => item.id === selectedUniverse.value));
const fieldPoints = computed<ProjectionPoint[]>(() => semanticField.value.clouds.map(cloud => ({ id: cloud.cloud_id, label: cloud.canonical_lemma, x: (Math.max(-1, Math.min(1, cloud.display_center[0] || 0)) + 1) / 2, y: (Math.max(-1, Math.min(1, cloud.display_center[1] || 0)) + 1) / 2, projections: {} })));
const displayPoints = computed<Array<Entity | ProjectionPoint>>(() => selectedDimensions.value.length ? projectionPoints.value : fieldPoints.value);
const dimensionTitle = computed(() => selectedDimensions.value.map(compact).join(' × '));

function compact(value: string): string { return value.replace('dimension-', 'D-').slice(0, 14); }
function position(point: Entity | ProjectionPoint): Record<string, string> {
  if ('base_position' in point) {
    const [x = 0, y = 0] = point.base_position;
    return { left: `${18 + ((x + 1) / 2) * 64}%`, top: `${18 + ((y + 1) / 2) * 64}%`, fontSize: `${12 + Math.min(point.mass, 4) * 2}px` };
  }
  return { left: `${18 + point.x * 64}%`, top: `${18 + point.y * 64}%`, fontSize: '14px' };
}
async function loadUniverse(): Promise<void> {
  error.value = ''; selectedEntity.value = null; selectedFieldCloud.value = null; selectedDimensions.value = []; projectionPoints.value = []; dimensionDetail.value = null;
  try {
    const [space, fields, field] = await Promise.all([
      api.get<{ entities: Entity[] }>(`/api/universes/${selectedUniverse.value}/base-space`),
      api.get<{ dimensions: Dimension[] }>(`/api/universes/${selectedUniverse.value}/dimensions`),
      api.get<SemanticField>('/api/v2/semantic-field'),
    ]);
    base.value = space; dimensions.value = fields.dimensions; semanticField.value = field;
  } catch (cause) { error.value = cause instanceof Error ? cause.message : 'Не удалось загрузить пространство'; }
}
async function selectUniverse(id: string): Promise<void> { selectedUniverse.value = id; await loadUniverse(); }
async function toggleDimension(id: string): Promise<void> {
  selectedDimensions.value = selectedDimensions.value.includes(id)
    ? selectedDimensions.value.filter(value => value !== id)
    : [...selectedDimensions.value.slice(-1), id];
  if (!selectedDimensions.value.length) { projectionPoints.value = []; dimensionDetail.value = null; return; }
  try {
    const inspectedDimension = selectedDimensions.value[
      selectedDimensions.value.length - 1
    ];
    const [result, detail] = await Promise.all([
      api.post<{ points: ProjectionPoint[] }>('/api/visualization/project', {
        universe_id: selectedUniverse.value,
        space_type: 'dimensions',
        dimension_ids: selectedDimensions.value,
        projection_method: 'selected_dimensions',
      }),
      api.get<DimensionDetail>(`/api/dimensions/${inspectedDimension}`),
    ]);
    projectionPoints.value = result.points;
    dimensionDetail.value = detail;
  } catch (cause) { error.value = cause instanceof Error ? cause.message : 'Не удалось построить проекцию'; }
}
async function openEntity(id: string): Promise<void> {
  if (!selectedDimensions.value.length) {
    const cloud = semanticField.value.clouds.find(item => item.cloud_id === id);
    if (cloud) { selectedFieldCloud.value = cloud; selectedEntity.value = null; return; }
  }
  try { selectedEntity.value = await api.get<Profile>(`/api/entities/${id}/dimension-profile`); selectedFieldCloud.value = null; }
  catch (cause) { error.value = cause instanceof Error ? cause.message : 'Не удалось загрузить профиль'; }
}
async function refresh(): Promise<void> {
  universes.value = (await api.get<{ universes: Universe[] }>('/api/universes')).universes;
  if (!universes.value.some(item => item.id === selectedUniverse.value)) selectedUniverse.value = universes.value[0]?.id || '';
  if (selectedUniverse.value) await loadUniverse();
}
async function learn(): Promise<void> {
  const text = trainingText.value.trim();
  if (!text) return;
  training.value = true; error.value = ''; trainingStatus.value = '';
  try {
    const result = await api.post<{ status: string; universe_update?: { universes: string[] } }>('/api/v2/training/learn', {
      text,
      source_type: 'training',
    });
    trainingText.value = '';
    trainingStatus.value = result.status === 'CONFIRMED'
      ? `Готово: обновлено ${result.universe_update?.universes.length || 0} пространств.`
      : `Источник сохранён со статусом ${result.status}.`;
    await refresh();
  } catch (cause) { error.value = cause instanceof Error ? cause.message : 'Не удалось выполнить обучение'; }
  finally { training.value = false; }
}
function openResetDialog(): void {
  resetError.value = '';
  resetScope.value = 'FULL_TEST_STATE';
  resetOpen.value = true;
}
function closeResetDialog(): void {
  if (resetting.value) return;
  resetOpen.value = false;
  resetError.value = '';
}
function clearProjectStorage(): void {
  for (const storage of [window.localStorage, window.sessionStorage]) {
    const keys = Array.from({ length: storage.length }, (_, index) => storage.key(index))
      .filter((key): key is string => Boolean(key) && key!.startsWith('superai'));
    keys.forEach(key => storage.removeItem(key));
  }
}
function clearClientRuntime(): void {
  graphStore.clearLocalState();
  trainingStore.clearLocalState();
  hiveStore.clearRuntimeState();
  modelStore.clearLocalState();
  clearProjectStorage();
  selectedEntity.value = null;
  selectedFieldCloud.value = null;
  selectedDimensions.value = [];
  projectionPoints.value = [];
  dimensionDetail.value = null;
  trainingText.value = '';
}
async function resetTestSpace(): Promise<void> {
  resetting.value = true;
  resetError.value = '';
  error.value = '';
  try {
    const mode = resetScope.value === 'FULL_TEST_STATE' ? 'FRESH_SCHEMA' : 'CLEAR_DATA';
    const report = await api.post<ResetReport>('/api/v2/testing/reset', {
      scope: resetScope.value,
      mode,
    });
    if (!report.reset || Object.values(report.invariants).some(value => !value)) {
      throw new Error('Backend не подтвердил пустое и согласованное состояние');
    }
    clearClientRuntime();
    await api.get('/api/readiness');
    await refresh();
    trainingStatus.value = resetScope.value === 'FULL_TEST_STATE'
      ? 'Тестовое пространство очищено. Можно начинать новое обучение.'
      : 'Производное пространство очищено. Доступна явная пересборка из Event Graph.';
    resetOpen.value = false;
  } catch (cause) {
    resetError.value = cause instanceof Error ? cause.message : 'Не удалось очистить тестовое пространство';
  } finally {
    resetting.value = false;
  }
}

async function exportMemory(): Promise<void> {
  exporting.value = true; error.value = '';
  try {
    const copied = await copyJsonToClipboard(await api.get('/api/export/memory'));
    if (!copied) throw new Error('Браузер не предоставил доступ к буферу обмена');
    exportStatus.value = 'Скопировано';
    setTimeout(() => { exportStatus.value = ''; }, 1500);
  } catch (cause) { error.value = cause instanceof Error ? cause.message : 'Не удалось скопировать память'; }
  finally { exporting.value = false; }
}
onMounted(async () => {
  try {
    await refresh();
  } catch (cause) { error.value = cause instanceof Error ? cause.message : 'Не удалось загрузить микровселенные'; }
});
</script>

<style scoped>
.universe-page { min-height: 100vh; color: #e8edf5; background: #0c111b; font: 14px/1.4 Inter, ui-sans-serif, system-ui, sans-serif; }
.topbar { height: 62px; display: flex; align-items: center; justify-content: space-between; padding: 0 28px; border-bottom: 1px solid #233145; background: #101827; }
.brand { color: #f3f7fb; text-decoration: none; font-weight: 700; letter-spacing: .04em; }.brand small { color: #7d91ad; font-weight: 500; margin-left: 8px; }
nav { display: flex; align-items: center; gap: 18px; } nav a { color: #9db0c8; text-decoration: none; } nav a.router-link-exact-active { color: #8cceff; }.export-button { border: 1px solid #3e6688; border-radius: 6px; padding: 6px 9px; color: #b9e4fa; background: transparent; cursor: pointer; }.export-button:hover { background: #19334d; }.export-button:disabled { opacity: .5; cursor: default; }.reset-button { border: 1px solid #8f4954; border-radius: 6px; padding: 6px 9px; color: #ffb7bf; background: transparent; cursor: pointer; }.reset-button:hover { background: #47232d; }.reset-button:disabled { opacity: .5; cursor: default; }
.workspace { display: grid; grid-template-columns: 230px minmax(420px, 1fr) 310px; min-height: calc(100vh - 62px); gap: 1px; background: #233145; }.panel { background: #101827; padding: 20px; }
.navigator button, .dimension, .projection { display: flex; width: 100%; border: 0; border-radius: 8px; background: transparent; color: inherit; text-align: left; cursor: pointer; padding: 10px; margin: 4px 0; }.navigator button { flex-direction: column; }.navigator button span, .dimension span { color: #7d91ad; font-size: 12px; }.navigator button.selected, .navigator button:hover, .dimension:hover { background: #1a2a40; }
.kicker { color: #6cbefa; font-size: 10px; letter-spacing: .15em; font-weight: 700; }.panel-head { display: flex; align-items: flex-start; justify-content: space-between; }.panel-head h1 { margin: 3px 0; font-size: 24px; }.panel-head > span { color: #7d91ad; }.notice { color: #7d91ad; font-size: 12px; }
.training-form { display: grid; gap: 7px; margin: 14px 0; padding: 12px; border: 1px solid #29435d; border-radius: 10px; background: #142338; }.training-form label { color: #9db0c8; font-size: 12px; }.training-form textarea { resize: vertical; min-height: 56px; border: 1px solid #38536f; border-radius: 7px; padding: 8px; color: #e8edf5; background: #0c1624; font: inherit; }.training-actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; color: #8cccf5; font-size: 12px; }.training-actions div { display: flex; gap: 8px; }.training-actions button { border: 0; border-radius: 7px; padding: 8px 12px; color: #07121c; background: #6bc7f7; font-weight: 700; cursor: pointer; }.training-actions .clear-draft { color: #b9e4fa; background: transparent; border: 1px solid #38536f; }.training-actions button:disabled { opacity: .5; cursor: default; }
.map { position: relative; min-height: 680px; overflow: hidden; border: 1px solid #22334a; border-radius: 14px; background: radial-gradient(circle at 45% 42%, #183657 0, #101b2b 38%, #0e1520 75%); }.map::before { content: ''; position: absolute; inset: 10%; border: 1px solid #2b4867; border-radius: 50%; opacity: .4; }
.entity { position: absolute; transform: translate(-50%, -50%); border: 0; border-radius: 999px; padding: 7px 10px; color: #dff1ff; background: #1b5276cc; box-shadow: 0 0 0 1px #76ceff55, 0 0 20px #419bd044; cursor: pointer; }.entity:hover, .entity.active { background: #2b8fc8; box-shadow: 0 0 0 2px #b6e8ff; }.empty { position: absolute; inset: 45% 10%; text-align: center; color: #7d91ad; }
.inspector h2 { margin: 7px 0; }.inspector h3 { margin: 18px 0 7px; font-size: 12px; color: #9db0c8; }.inspector p { color: #9db0c8; }.show-dimensions, .show-word-forms { border: 1px solid #38536f; border-radius: 6px; padding: 7px 9px; color: #b9e4fa; background: transparent; cursor: pointer; }.word-form { display: flex; width: 100%; justify-content: space-between; border: 0; border-radius: 6px; padding: 5px 7px; color: #d8e8f4; background: #152438; text-align: left; cursor: pointer; }.word-form:hover { background: #1d3b59; }.word-form em { color: #91aac3; font-size: 11px; font-style: normal; }.show-word-forms { margin-top: 10px; }.dimension-help { font-size: 12px; }.dimension { flex-direction: column; }.dimension.selected { background: #1a2f48; box-shadow: inset 3px 0 #6bc7f7; }.dimension small { color: #9db0c8; font-weight: 400; }.projection { align-items: center; gap: 8px; padding: 5px 0; cursor: default; }.projection span { width: 92px; color: #9db0c8; }.projection i { flex: 1; height: 7px; background: #25344a; border-radius: 99px; overflow: hidden; }.projection b { display: block; height: 100%; background: #5bc0f6; }.projection em { width: 30px; color: #d8e8f4; font-style: normal; font-size: 12px; }.error { color: #ff9e9e !important; }
.dimension-detail { margin-top: 12px; padding: 10px; border: 1px solid #29435d; border-radius: 8px; background: #111e30; }.dimension-detail dl { display: grid; grid-template-columns: 1fr; gap: 2px; }.dimension-detail dt { margin-top: 7px; color: #7890ad; font-size: 10px; text-transform: uppercase; }.dimension-detail dd { margin: 0; color: #d3e2ef; font-size: 11px; overflow-wrap: anywhere; }
.cloud-detail { display: grid; gap: 3px; }.cloud-detail dt { margin-top: 7px; color: #7890ad; font-size: 10px; text-transform: uppercase; }.cloud-detail dd { margin: 0; color: #d3e2ef; font-size: 11px; overflow-wrap: anywhere; }.force-row { display: grid; gap: 2px; margin: 5px 0; padding: 7px; border: 1px solid #29435d; border-radius: 6px; background: #111e30; }.force-row strong { color: #b9e4fa; font-size: 10px; }.force-row span { color: #8fa7bd; font-size: 10px; }.empty-inline { color: #7d91ad; font-size: 11px; }
.reset-backdrop { position: fixed; inset: 0; z-index: 50; display: grid; place-items: center; padding: 20px; background: #040812cc; backdrop-filter: blur(4px); }.reset-dialog { width: min(520px, 100%); border: 1px solid #6e3540; border-radius: 14px; padding: 22px; background: #131b29; box-shadow: 0 24px 80px #000a; }.reset-dialog h2 { margin: 5px 0 10px; }.reset-dialog p { color: #aebed0; }.reset-dialog label { display: grid; gap: 6px; margin-top: 14px; color: #cbd8e5; font-size: 12px; }.reset-dialog select, .reset-dialog input { border: 1px solid #485d73; border-radius: 7px; padding: 9px; color: #eef5fb; background: #0c1522; font: inherit; }.reset-actions { display: flex; justify-content: flex-end; gap: 9px; margin-top: 20px; }.reset-actions button { border-radius: 7px; padding: 9px 12px; cursor: pointer; }.reset-actions .cancel { border: 1px solid #485d73; color: #c9d8e5; background: transparent; }.reset-actions .danger { border: 1px solid #a94755; color: #fff; background: #8f3040; font-weight: 700; }.reset-actions button:disabled { opacity: .45; cursor: default; }
@media (max-width: 960px) { .workspace { grid-template-columns: 190px 1fr; }.inspector { grid-column: 1 / -1; }.map { min-height: 430px; } }
</style>
