/** Model field store - navigation, clouds, spaces, placements. */

import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import { api } from '@/shared/api/client';
import type {
  CloudV2,
  PlacementV2,
  SpaceV2,
  NormalizedSpaceV2,
  StatsV2,
  StructureV2,
  SceneV2,
  TrainedModelSnapshotV2,
  SceneComponentV2,
} from '@/entities/model/types';

const spaceLabels: Record<SpaceV2['space_type'], string> = {
  global_field: 'Глобальное поле',
  scene_space: 'Пространство сцены',
  word_structure_space: 'Структура словоформы',
  morphology_space: 'Морфологическое пространство',
  sentence_frame_space: 'Пространство каркаса',
  concept_space: 'Пространство понятия',
  hive_space: 'Пространство улья',
  hive_subspace: 'Проекция улья',
};

export const useModelStore = defineStore('model', () => {
  const cloudsById = ref<Record<number, CloudV2>>({});
  const placementsById = ref<Record<number, PlacementV2>>({});
  const currentSpace = ref<SpaceV2 | null>(null);
  const stats = ref<StatsV2>({
    clouds_total: 0,
    clouds_by_type: {},
    spaces_total: 0,
    spaces_by_type: {},
    placements_total: 0,
    unique_word_forms: 0,
    scene_components_total: 0,
    structural_components_total: 0,
    concepts_total: 0,
    semantic_evidence_total: 0,
    concept_fogs_total: 0,
    concept_candidates_total: 0,
    semantic_backfill_scenes_total: 0,
  });
  const breadcrumb = ref<Array<{ space: SpaceV2; label: string }>>([]);
  const selectedPlacementId = ref<number | null>(null);
  const selectedStructure = ref<StructureV2 | null>(null);
  const currentStructure = ref<StructureV2 | null>(null);
  const selectedScene = ref<SceneV2 | null>(null);
  const lastTraining = ref<Record<string, any> | null>(null);
  const trainedModel = ref<TrainedModelSnapshotV2 | null>(null);
  const loading = ref(false);
  const error = ref('');

  const placements = computed(() => Object.values(placementsById.value))
  const selectedPlacement = computed(() =>
    selectedPlacementId.value ? placementsById.value[selectedPlacementId.value] ?? null : null
  );
  const selectedCloud = computed(() =>
    selectedPlacement.value ? cloudsById.value[selectedPlacement.value.cloud_id] ?? null : null
  );
  const selectedSceneComponent = computed<SceneComponentV2 | null>(() =>
    selectedPlacement.value && selectedScene.value
      ? selectedScene.value.components.find(
          item => item.placement_id === selectedPlacement.value?.id
        ) ?? null
      : null
  );

  function ingest(payload: NormalizedSpaceV2, pushBreadcrumb = true) {
    currentSpace.value = payload.space;
    cloudsById.value = Object.fromEntries(Object.values(payload.clouds).map(c => [c.id, c]));
    placementsById.value = Object.fromEntries(payload.placements.map(p => [p.id, p]));
    stats.value = payload.stats;
    selectedPlacementId.value = null;
    currentStructure.value = null;
    if (pushBreadcrumb) {
      const label = `${spaceLabels[payload.space.space_type]} #${payload.space.id}`;
      breadcrumb.value.push({ space: payload.space, label });
    }
  }

  async function run<T>(operation: () => Promise<T>): Promise<T> {
    loading.value = true;
    error.value = '';
    try {
      return await operation();
    } catch (cause) {
      error.value = cause instanceof Error ? cause.message : String(cause);
      throw cause;
    } finally {
      loading.value = false;
    }
  }

  async function loadField() {
    return run(async () => {
      const payload = await api.get<NormalizedSpaceV2>('/api/v2/field');
      breadcrumb.value = [];
      selectedScene.value = null;
      selectedStructure.value = null;
      ingest(payload);
    });
  }

  async function loadSpace(spaceId: number, pushBreadcrumb = true) {
    return run(async () => ingest(await api.get<NormalizedSpaceV2>(`/api/v2/spaces/${spaceId}`), pushBreadcrumb));
  }

  async function loadScene(cloudId: number) {
    const payload = await api.get<{ scene: SceneV2 }>(`/api/v2/scenes/${cloudId}`);
    selectedScene.value = payload.scene;
    return payload.scene;
  }

  async function loadStructure(cloudId: number) {
    selectedStructure.value = await api.get<StructureV2>(`/api/v2/clouds/${cloudId}/structure`);
    return selectedStructure.value;
  }

  async function selectPlacement(placementId: number | null) {
    selectedPlacementId.value = placementId;
    selectedStructure.value = null;
    const placement = placementId ? placementsById.value[placementId] : null;
    const cloud = placement ? cloudsById.value[placement.cloud_id] : null;
    if (cloud?.cloud_type === 'scene') await loadScene(cloud.id);
    if (cloud?.cloud_type === 'word_form') await loadStructure(cloud.id);
  }

  async function zoomIntoPlacement(placementId: number) {
    const placement = placementsById.value[placementId];
    const cloud = placement ? cloudsById.value[placement.cloud_id] : null;
    if (!cloud) return;
    if (cloud.cloud_type === 'scene') {
      const scene = await loadScene(cloud.id);
      await loadSpace(scene.scene_space_id);
      return;
    }
    if (cloud.cloud_type === 'word_form') {
      const structure = await loadStructure(cloud.id);
      if (!structure.structure_space) return;
      currentSpace.value = structure.structure_space;
      currentStructure.value = structure;
      cloudsById.value = Object.fromEntries(Object.values(structure.clouds).map(item => [item.id, item]));
      placementsById.value = {};
      breadcrumb.value.push({
        space: structure.structure_space,
        label: `${cloud.canonical_name} · структура`,
      });
      return;
    }
    if (cloud.cloud_type === 'concept') {
      const payload = await api.get<{ cloud: CloudV2; owned_spaces: SpaceV2[] }>(`/api/v2/clouds/${cloud.id}`);
      const fog = payload.owned_spaces.find(space => space.space_type === 'concept_space');
      if (fog) await loadSpace(fog.id);
    }
  }

  async function navigateTo(index: number) {
    const target = breadcrumb.value[index];
    if (!target) return;
    breadcrumb.value = breadcrumb.value.slice(0, index);
    selectedScene.value = target.space.space_type === 'scene_space' ? selectedScene.value : null;
    await loadSpace(target.space.id, false);
  }

  async function train(text: string) {
    return run(async () => {
      lastTraining.value = await api.post<Record<string, any>>('/api/v2/training/learn', { text });
      trainedModel.value = null;
      const payload = await api.get<NormalizedSpaceV2>('/api/v2/field');
      breadcrumb.value = [];
      ingest(payload);
      return lastTraining.value;
    });
  }

  async function tickPhysics() {
    if (!currentSpace.value) return;
    await run(async () => {
      await api.post(`/api/v2/spaces/${currentSpace.value!.id}/physics/tick`);
      await loadSpace(currentSpace.value!.id, false);
    });
  }

  async function loadTrainedModel() {
    return run(async () => {
      trainedModel.value = await api.get<TrainedModelSnapshotV2>('/api/v2/model');
      return trainedModel.value;
    });
  }

  async function clearModel() {
    return run(async () => {
      await api.delete('/api/v2/model');
      localStorage.removeItem('superai-v2-active-hive');
      localStorage.removeItem('superai-v2-chat-cache');
      lastTraining.value = null;
      trainedModel.value = null;
      selectedScene.value = null;
      selectedStructure.value = null;
      const payload = await api.get<NormalizedSpaceV2>('/api/v2/field');
      breadcrumb.value = [];
      ingest(payload);
    });
  }

  return {
    cloudsById,
    placementsById,
    currentSpace,
    currentStructure,
    selectedStructure,
    selectedScene,
    selectedSceneComponent,
    selectedPlacementId,
    selectedPlacement,
    selectedCloud,
    placements,
    stats,
    breadcrumb,
    lastTraining,
    trainedModel,
    loading,
    error,
    loadField,
    loadSpace,
    selectPlacement,
    zoomIntoPlacement,
    navigateTo,
    train,
    tickPhysics,
    loadTrainedModel,
    clearModel,
  };
});
