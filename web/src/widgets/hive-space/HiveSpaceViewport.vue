<template>
  <section class="space-panel" :class="`level-${current.type}`">
    <header class="space-head">
      <div>
        <div class="kicker">{{ current.type === 'hive' ? 'РАБОЧЕЕ ПРОСТРАНСТВО УЛЬЯ' : 'ЛОКАЛЬНОЕ ПРОСТРАНСТВО' }}</div>
        <h2>{{ current.title }}</h2>
      </div>
      <div class="space-actions">
        <button class="icon-button" :disabled="path.length === 1" title="Назад" @click="back">←</button>
        <button class="secondary" :class="{ selected: current.type === 'global' }" @click="openGlobal">Глобальное поле</button>
        <button v-if="current.type === 'global'" class="secondary" @click="openHive">Вернуться в улей</button>
        <button class="secondary" :disabled="path.length === 1" @click="up">На уровень выше</button>
        <button class="secondary" :class="{ selected: lockedCamera }" @click="lockedCamera = !lockedCamera">{{ lockedCamera ? 'Камера зафиксирована' : 'Зафиксировать камеру' }}</button>
      </div>
    </header>

    <nav class="breadcrumbs" aria-label="Навигация по пространствам">
      <button v-for="(item, index) in path" :key="`${item.type}-${index}`" :class="{ current: index === path.length - 1 }" @click="goTo(index)">{{ item.crumb }}</button>
    </nav>

    <div class="query-strip">
      <span>ЗАПРОС</span><strong>{{ hiveStore.goalText || 'Ожидание запроса' }}</strong>
      <em>{{ routeLabel }}</em>
    </div>

    <div class="viewport" :class="{ transitioning, 'hive-viewport': current.type === 'hive' }">
      <div class="energy-grid"></div>
      <HiveModePanel v-if="current.type === 'hive'" />
      <svg v-else class="space-svg" viewBox="0 0 1000 600" @wheel.prevent="onViewportWheel" @pointerdown="onViewportPointerDown" @pointermove="onViewportPointerMove" @pointerup="stopViewportPan" @pointerleave="stopViewportPan" @click.self="hiveStore.selectedCell = null">
        <defs>
          <filter id="spaceGlow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="7" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          <radialGradient id="spaceCore"><stop stop-color="#244c77"/><stop offset="1" stop-color="#081526"/></radialGradient>
        </defs>
        <g v-if="current.type === 'global'" class="global-field" :transform="globalTransform">
          <circle class="global-hive-boundary" :cx="hiveProjection.x" :cy="hiveProjection.y" :r="hiveProjection.r" />
          <text class="global-hive-label" :x="hiveProjection.x" :y="hiveProjection.y - hiveProjection.r - 12" text-anchor="middle">УЛЕЙ · {{ hiveStore.cells.length }} ЯЧЕЕК</text>
          <path v-for="flight in beeFlights" :key="flight.id" class="bee-flight" :d="flight.path" />
          <g v-for="node in globalNodes" :key="`global-${node.id}`" class="global-node" :class="{ carried: node.carried }" :transform="`translate(${node.x} ${node.y})`" @click.stop="selectGlobalNode(node)">
            <circle :r="node.radius" />
            <text text-anchor="middle" y="4">{{ node.short }}</text>
            <text class="global-label" text-anchor="middle" :y="node.radius + 14">{{ node.label }}</text>
          </g>
          <g v-for="flight in beeFlights" :key="`bee-${flight.id}`" class="bee-marker" :transform="`translate(${flight.x} ${flight.y})`"><text text-anchor="middle" y="4">✦</text></g>
          <g class="global-hive" :transform="`translate(${hiveProjection.x} ${hiveProjection.y})`" @click.stop="openHive">
            <path d="M -33 -28 L 0 -47 L 33 -28 V 28 L 0 47 L -33 28 Z" />
            <path class="hive-slat" d="M -16 -16 H 16 M -20 -3 H 20 M -16 10 H 16 M -10 23 H 10" />
            <text text-anchor="middle" y="69">ВОЙТИ В УЛЕЙ</text>
          </g>
        </g>
        <g v-else-if="String(current.type) === 'hive'">
          <path v-for="edge in hiveEdges" :key="edge.id" class="edge" :d="edge.path" :style="{ opacity: edge.weight }" />
          <g v-if="hiveStore.queryScene" class="query-scene-map">
            <path v-for="link in querySceneLinks" :key="link.id" class="query-link" :d="link.path" />
            <g v-for="slot in querySlots" :key="slot.id" class="query-slot" :class="slot.status" :transform="`translate(${slot.x} ${slot.y})`">
              <rect x="-72" y="-32" width="144" height="64" rx="15" />
              <text class="query-role" text-anchor="middle" y="-8">{{ slot.role }}</text>
              <text class="query-label" text-anchor="middle" y="14">{{ slot.label }}</text>
            </g>
            <g v-for="candidate in queryCandidateNodes" :key="candidate.id" class="query-candidate" :class="candidate.status" :transform="`translate(${candidate.x} ${candidate.y})`">
              <circle :r="candidate.radius" />
              <text class="candidate-type" text-anchor="middle" y="-5">КАНДИДАТ</text>
              <text text-anchor="middle" y="11">{{ candidate.label }}</text>
            </g>
            <g v-for="bridge in semanticBridgeNodes" :key="bridge.id" class="semantic-bridge" :transform="`translate(${bridge.x} ${bridge.y})`">
              <path :d="bridge.path" />
              <circle :r="bridge.radius" />
              <text class="bridge-type" text-anchor="middle" y="-6">МОСТ</text>
              <text text-anchor="middle" y="12">{{ bridge.label }}</text>
              <text class="bridge-detail" text-anchor="middle" :y="bridge.radius + 14">{{ bridge.detail }}</text>
            </g>
            <text class="query-step" x="500" y="548" text-anchor="middle">СЦЕНА ЗАПРОСА · ШАГ {{ hiveStore.vibrationHistory.length }} · {{ hiveStore.queryAnswer?.answer_mode || 'partial' }}</text>
          </g>
          <g v-for="node in hiveNodes" :key="node.cell.id" class="hive-node" :class="{ active: hiveStore.selectedCell?.id === node.cell.id }" :transform="`translate(${node.x} ${node.y})`" @click.stop="openSceneFor(node.cell)">
            <circle class="node-halo" :r="node.radius + 17" />
            <circle class="node-body" :r="node.radius" :style="{ '--energy': node.energy }" />
            <text class="node-type" text-anchor="middle" y="4">{{ node.type }}</text>
            <text class="node-label" text-anchor="middle" :y="node.radius + 19">{{ node.label }}</text>
            <text class="node-metric" text-anchor="middle" :y="node.radius + 33">{{ node.activation }}% · d{{ node.depth }}</text>
          </g>
        </g>

        <g v-else-if="current.type === 'scene'" class="scene-space">
          <path v-for="link in sceneLinks" :key="link.id" class="role-link" :d="link.path" />
          <g v-for="node in sceneNodes" :key="node.id" class="role-node" :class="node.role" :transform="`translate(${node.x} ${node.y})`" @click.stop="openLexeme(node)">
            <rect x="-105" y="-47" width="210" height="94" rx="18" />
            <text class="role-name" text-anchor="middle" y="-15">{{ roleName(node.role) }}</text>
            <text class="role-label" text-anchor="middle" y="12">{{ node.label }}</text>
            <text class="role-score" text-anchor="middle" y="31">{{ node.activation }}% activation</text>
          </g>
        </g>

        <g v-else-if="current.type === 'lexeme'" class="lexeme-space">
          <circle class="orbit" cx="500" cy="300" r="188" />
          <circle class="orbit second" cx="500" cy="300" r="112" />
          <path v-for="node in lexemeNodes" :key="`lexeme-link-${node.id}`" class="lexeme-link" :d="`M 500 300 L ${node.x} ${node.y}`" />
          <g class="lexeme-center"><circle cx="500" cy="300" r="70"/><text x="500" y="295" text-anchor="middle">{{ current.label }}</text><text x="500" y="318" text-anchor="middle">активная лексема</text></g>
          <g v-for="node in lexemeNodes" :key="node.id" class="lexeme-node" :transform="`translate(${node.x} ${node.y})`" @click.stop="openMorphology(node)"><circle :r="node.radius"/><text text-anchor="middle" y="4">{{ node.label }}</text><text class="score" text-anchor="middle" :y="node.radius + 15">{{ node.kind }}</text></g>
        </g>

        <g v-else-if="current.type === 'morphology'" class="morph-space">
          <rect class="morph-zone stem" x="88" y="220" width="270" height="190" rx="22"/><rect class="morph-zone operators" x="390" y="105" width="240" height="138" rx="22"/><rect class="morph-zone ending" x="662" y="220" width="250" height="190" rx="22"/><rect class="morph-zone hypotheses" x="330" y="445" width="340" height="92" rx="22"/>
          <text class="zone-label" x="112" y="252">ОСНОВА</text><text class="zone-label" x="414" y="137">ОПЕРАТОРЫ</text><text class="zone-label" x="686" y="252">ОКОНЧАНИЕ</text><text class="zone-label" x="354" y="476">ГИПОТЕЗЫ</text>
          <path class="morph-link" d="M 355 314 H 662"/><path class="morph-link dashed" d="M 500 244 V 445"/>
          <g class="morph-token" transform="translate(223 320)"><rect x="-74" y="-37" width="148" height="74" rx="13"/><text text-anchor="middle" y="5">{{ stem }}</text></g>
          <g v-for="(operator, index) in operators" :key="operator" class="operator-token" :transform="`translate(${450 + index * 120} 183)`"><circle r="34"/><text text-anchor="middle" y="4">{{ operator }}</text></g>
          <g class="morph-token ending-token" transform="translate(787 320)" @click.stop="openWordForm"><rect x="-74" y="-37" width="148" height="74" rx="13"/><text text-anchor="middle" y="5">{{ ending || '∅' }}</text><text class="small" text-anchor="middle" y="58">готовая форма</text></g>
          <text class="hypothesis" x="500" y="514" text-anchor="middle">{{ current.label }} · {{ current.score }}% · обратная проверка</text>
        </g>

        <g v-else-if="current.type === 'word_form'" class="form-space">
          <circle class="form-field" cx="500" cy="300" r="170"/>
          <g v-for="(node, index) in formNodes" :key="node.id" class="form-node" :class="{ selected: index === 0 }" :transform="`translate(${node.x} ${node.y})`" @click.stop="openCharacters(node)"><circle :r="node.radius"/><text text-anchor="middle" y="2">{{ node.label }}</text><text class="score" text-anchor="middle" :y="node.radius + 17">{{ node.score }}%</text><text class="details" text-anchor="middle" :y="node.radius + 30">{{ node.status }}</text></g>
        </g>

        <g v-else class="character-space">
          <path class="character-line" d="M 105 300 H 895"/>
          <g v-for="(letter, index) in letters" :key="`${letter}-${index}`" class="character-node" :transform="`translate(${characterX(index)} 300)`"><circle r="42"/><text text-anchor="middle" y="8">{{ letter }}</text><text class="position" text-anchor="middle" y="68">{{ index + 1 }}</text><path v-if="index < letters.length - 1" d="M 49 0 H 122"/></g>
        </g>
      </svg>
      <div v-if="current.type !== 'hive'" class="space-legend" :class="{ expanded: current.type === 'global' }">
        <template v-if="current.type === 'global'">
          <span><i class="field-concept"></i>понятие в поле</span>
          <span><i class="carried-concept"></i>унесено пчёлами в улей</span>
          <span><i class="bee-route"></i>траектория пчелы</span>
          <span><i class="hive-boundary"></i>локальная проекция улья</span>
          <span><i class="hive-entry"></i>войти в улей</span>
          <span>{{ globalNodes.length }} понятий поля</span>
        </template>
        <template v-else-if="String(current.type) === 'hive'">
          <span><i class="active-cell"></i>активная ячейка</span><span><i class="selected-cell"></i>выбранная ячейка</span><span><i class="cell-link"></i>смысловая связь</span><span><i class="cell-metric"></i>подпись: активация · глубина</span><span>{{ physicalCellCount }} активных ячеек</span>
        </template>
        <template v-else-if="current.type === 'scene'">
          <span><i class="role-area"></i>роль сцены</span><span><i class="role-object"></i>объект</span><span><i class="role-link-icon"></i>связь ролей</span><span><i class="role-score-icon"></i>подпись: активация</span>
        </template>
        <template v-else-if="current.type === 'lexeme'">
          <span><i class="lexeme-center-icon"></i>активная лексема</span><span><i class="lexeme-form-icon"></i>форма / сосед</span><span><i class="lexeme-link-icon"></i>локальная связь</span><span><i class="orbit-icon"></i>радиус близости</span>
        </template>
        <template v-else-if="current.type === 'morphology'">
          <span><i class="stem-icon"></i>основа</span><span><i class="operator-icon"></i>грамматический оператор</span><span><i class="ending-icon"></i>окончание / форма</span><span><i class="hypothesis-icon"></i>гипотеза</span>
        </template>
        <template v-else-if="current.type === 'word_form'">
          <span><i class="candidate-icon"></i>кандидат формы</span><span><i class="best-candidate-icon"></i>лучший кандидат</span><span><i class="form-score-icon"></i>подпись: score · статус</span>
        </template>
        <template v-else><span><i class="character-icon"></i>символ</span><span><i class="sequence-icon"></i>фиксированный порядок</span><span><i class="position-icon"></i>номер позиции</span></template>
      </div>
      <div v-if="current.type === 'global'" class="camera-controls"><button @click="zoomGlobal(1.2)">+</button><button @click="zoomGlobal(.8)">−</button><button @click="resetGlobalCamera">Обзор</button><span>{{ Math.round(globalZoom * 100) }}%</span></div>
      <button v-if="String(current.type) !== 'hive'" class="root-button" @click="goTo(0)">Вернуться в основной улей</button>
    </div>

    <footer class="timeline"><span>СОСТОЯНИЕ УЛЬЯ</span><b>{{ hiveStore.reasoningLoading ? 'выполняется' : 'стабильно' }}</b><span>энергия {{ averageEnergy }}%</span><button class="secondary" @click="router.push({ name: 'analytics', query: { level: current.type, label: current.label } })">Показать аналитику уровня</button></footer>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useRouter } from 'vue-router';
import { useHiveStore } from '@/entities/hive/store';
import { useModelStore } from '@/entities/model/store';
import HiveModePanel from './HiveModePanel.vue';

type Level = { type: 'global' | 'hive' | 'scene' | 'lexeme' | 'morphology' | 'word_form' | 'characters'; title: string; crumb: string; label: string; cell?: any; score?: number };
const hiveStore = useHiveStore();
const modelStore = useModelStore();
const router = useRouter();
const path = ref<Level[]>([{ type: 'hive', title: 'Улей', crumb: 'Улей', label: '' }]);
const transitioning = ref(false);
const lockedCamera = ref(false);
const globalZoom = ref(1);
const globalPan = ref({ x: 0, y: 0 });
const viewportPanning = ref(false);
const viewportPointer = ref({ x: 0, y: 0 });
const current = computed(() => path.value[path.value.length - 1]);
const routeLabel = computed(() => hiveStore.queryPipeline?.answer?.status === 'RESOLVED' ? 'ANSWER_READY' : hiveStore.queryPipeline?.vibration?.status === 'RUNNING' ? 'HIVE_REASONING' : hiveStore.queryPipeline?.query_scene?.status === 'RESOLVED' ? 'ROLE_RESOLVED' : hiveStore.queryPipeline?.memory_search?.status || hiveStore.queryPipeline?.token_resolution?.status || hiveStore.decision?.decision || 'ГОТОВ');
const averageEnergy = computed(() => Math.round(((hiveStore.hive as any)?.energy?.reasoning_cells_average || hiveStore.cells.reduce((sum, cell) => sum + (cell.local_activation || 0), 0) / Math.max(hiveStore.cells.length, 1)) * 100));
const stem = computed(() => current.value.label.slice(0, Math.max(1, current.value.label.length - 1)));
const ending = computed(() => current.value.label.slice(-1));
const operators = computed(() => current.value.label.endsWith('у') ? ['ACC', 'SG'] : ['KNOWN', 'FORM']);
const letters = computed(() => [...current.value.label]);

const querySlots = computed(() => {
  const slots = hiveStore.queryScene?.slots || [];
  const spaced = slots.slice(0, 4);
  return spaced.map((slot: any, index) => ({
    id: slot.id || `slot-${index}`, role: String(slot.role || '').toUpperCase(),
    label: String(slot.status || '').toLowerCase() === 'resolved' ? String(slot.value?.surface || slot.label) : String(slot.surface || slot.label || '…'),
    status: String(slot.status || 'fixed').toLowerCase(), x: 500 + (index - (spaced.length - 1) / 2) * 180, y: 300,
  }));
});
const querySceneLinks = computed(() => querySlots.value.slice(1).map((slot, index) => ({ id: `query-${index}`, path: `M ${querySlots.value[index].x + 73} 300 L ${slot.x - 73} 300` })));
const queryCandidateNodes = computed(() => {
  const target = querySlots.value.find(slot => slot.status === 'empty' || slot.status === 'resolved') || querySlots.value[0];
  if (!target) return [];
  return hiveStore.queryCandidates.slice(0, 7).map((candidate, index) => {
    const angle = index / Math.max(hiveStore.queryCandidates.length, 1) * Math.PI * 2 - Math.PI / 2;
    const score = candidate.scores.total || 0;
    return { id: candidate.id, label: candidate.surface, status: candidate.status, radius: 12 + score * 16, x: target.x + Math.cos(angle) * (105 + (1 - score) * 70), y: target.y + Math.sin(angle) * (105 + (1 - score) * 70) };
  });
});

const semanticBridgeNodes = computed(() => {
  const objectSlot = querySlots.value.find(slot => slot.role === 'OBJECT');
  if (!objectSlot) return [];
  return hiveStore.cells.filter(cell => cell.component_class === 'semantic_bridge').map((cell: any, index) => {
    const bridge = cell.metadata?.bridge || {};
    const surface = bridge.unknown_token?.surface || cell.metadata?.source_signal || 'форма';
    const hypothesis = bridge.unknown_token?.lemma_hypothesis || 'гипотеза';
    const global = bridge.global_candidate?.lexeme || cell.label || 'кандидат';
    return {
      id: cell.id, label: global, detail: `${Math.round((cell.local_activation || 0) * 100)}% · ${surface} → ${hypothesis} → ${global}`,
      radius: 34, x: objectSlot.x + index * 84, y: objectSlot.y + 118, path: `M ${objectSlot.x} ${objectSlot.y + 33} L ${objectSlot.x + index * 84} ${objectSlot.y + 84}`,
    };
  });
});
const hiveNodes = computed(() => hiveStore.cells.filter(cell => !['semantic_bridge', 'role_candidate'].includes(cell.component_class)).map((cell: any, index: number) => ({ cell, x: 135 + ((Number(cell.x) || index * 137) % 720), y: 110 + ((Number(cell.y) || index * 97) % 370), radius: 28 + Math.min(22, cell.retention * 24), label: (cell.label || 'ячейка').slice(0, 18), type: typeShort(cell.component_class), activation: Math.round((cell.local_activation || 0) * 100), energy: cell.local_activation || 0.2, depth: cell.subspaces?.length || 0 })));
const physicalCellCount = computed(() => hiveStore.workingCellCount || hiveNodes.value.length);
const hiveEdges = computed(() => hiveNodes.value.flatMap((left, index) => hiveNodes.value.slice(index + 1).map(right => ({ id: `${left.cell.id}-${right.cell.id}`, weight: similarity(left.cell.components || [], right.cell.components || []), path: `M ${left.x} ${left.y} L ${right.x} ${right.y}` })).filter(edge => edge.weight > .13)));
const carriedCloudIds = computed(() => new Set(hiveStore.cells.flatMap(cell => [cell.dominant_cloud_id, cell.source_cloud_id, ...(cell.components || []).map(component => component.cloud_id)])));
const globalNodes = computed(() => {
  const placements = modelStore.placements.slice(0, 72);
  if (!placements.length) return hiveNodes.value.map((node, index) => ({ id: node.cell.dominant_cloud_id || index, label: node.label, short: node.type, x: node.x, y: node.y, radius: node.radius * .66, carried: true, cell: node.cell }));
  const maxX = Math.max(...placements.map(placement => placement.x), 1);
  const maxY = Math.max(...placements.map(placement => placement.y), 1);
  return placements.map(placement => { const cloud = modelStore.cloudsById[placement.cloud_id]; return { id: placement.id, label: cloud?.canonical_name || `#${placement.cloud_id}`, short: typeShort(cloud?.cloud_type || ''), x: 70 + (placement.x / maxX) * 860, y: 70 + (placement.y / maxY) * 450, radius: 10 + Math.min(17, placement.radius * .34), carried: carriedCloudIds.value.has(placement.cloud_id), cell: hiveStore.cells.find(cell => cell.dominant_cloud_id === placement.cloud_id || cell.source_cloud_id === placement.cloud_id) }; });
});
const hiveProjection = computed(() => { const nodes = globalNodes.value.filter(node => node.carried); if (!nodes.length) return { x: 500, y: 300, r: 105 }; const x = nodes.reduce((sum, node) => sum + node.x, 0) / nodes.length; const y = nodes.reduce((sum, node) => sum + node.y, 0) / nodes.length; return { x, y, r: Math.max(85, ...nodes.map(node => Math.hypot(node.x - x, node.y - y) + 54)) }; });
const beeFlights = computed(() => globalNodes.value.filter(node => node.carried).map(node => ({ id: node.id, x: node.x + (hiveProjection.value.x - node.x) * .46, y: node.y + (hiveProjection.value.y - node.y) * .46, path: `M ${node.x} ${node.y} L ${hiveProjection.value.x} ${hiveProjection.value.y}` })));
const globalTransform = computed(() => `translate(${500 + globalPan.value.x} ${300 + globalPan.value.y}) scale(${globalZoom.value}) translate(-500 -300)`);
const selectedComponents = computed(() => current.value.cell?.components || []);
const sceneNodes = computed(() => {
  const components = selectedComponents.value.length ? selectedComponents.value : hiveStore.cells.slice(0, 4).map((cell: any) => ({ canonical_name: cell.label, role: cell.component_class, local_activation: cell.local_activation, cloud_id: cell.dominant_cloud_id }));
  const locations = [[245, 170], [500, 170], [755, 170], [500, 430], [245, 430], [755, 430]];
  return components.slice(0, 6).map((component: any, index: number) => ({ id: component.id || component.cloud_id || index, label: component.canonical_name || component.label || 'компонент', role: normalizeRole(component.role, index), activation: Math.round((component.local_activation || current.value.cell?.local_activation || .5) * 100), x: locations[index][0], y: locations[index][1] }));
});
const sceneLinks = computed(() => sceneNodes.value.slice(1).map((node: any) => ({ id: `scene-${node.id}`, path: `M ${sceneNodes.value[0]?.x || 500} ${sceneNodes.value[0]?.y || 300} L ${node.x} ${node.y}` })));
const lexemeNodes = computed(() => {
  const source = hiveStore.generationCandidates.length ? hiveStore.generationCandidates : selectedComponents.value;
  return source.slice(0, 12).map((item: any, index: number) => { const a = index / Math.max(source.length, 1) * Math.PI * 2 - Math.PI / 2; const distance = 135 + (index % 2) * 62; return { id: item.id || index, label: item.candidate_text || item.canonical_name || current.value.label, kind: item.status || 'известная форма', radius: 25 + (index % 3) * 4, x: 500 + Math.cos(a) * distance, y: 300 + Math.sin(a) * distance, score: Math.round((item.score_total || item.local_activation || .7) * 100) }; });
});
const formNodes = computed(() => {
  const source = hiveStore.generationCandidates.length ? hiveStore.generationCandidates : [{ candidate_text: current.value.label, score_total: .96, status: 'подтверждена' }];
  return source.slice(0, 8).map((item: any, index: number) => { const a = index / Math.max(source.length, 1) * Math.PI * 2 - Math.PI / 2; const distance = index === 0 ? 0 : 130 + (index % 2) * 42; return { id: item.id || index, label: item.candidate_text || current.value.label, score: Math.min(100, Math.round((item.score_total || .6) * 100)), status: item.status === 'SELECTED' ? 'выбрана' : item.status === 'GENERATED' ? 'гипотеза' : 'известная', radius: index === 0 ? 54 : 32, x: 500 + Math.cos(a) * distance, y: 300 + Math.sin(a) * distance }; });
});

function move(level: Level) { transitioning.value = true; window.setTimeout(() => { path.value.push(level); transitioning.value = false; }, 220); }
async function openGlobal() { await modelStore.loadField(); path.value = [{ type: 'global', title: 'Глобальное поле', crumb: 'Глобальное поле', label: '' }]; }
function openHive() { path.value = [{ type: 'hive', title: 'Улей', crumb: 'Улей', label: '' }]; }
function selectGlobalNode(node: any) { if (node.cell) { hiveStore.selectedCell = node.cell; openHive(); } }
function zoomGlobal(factor: number) { globalZoom.value = Math.max(.35, Math.min(5, globalZoom.value * factor)); }
function resetGlobalCamera() { globalZoom.value = 1; globalPan.value = { x: 0, y: 0 }; }
function onViewportWheel(event: WheelEvent) { if (current.value.type === 'global') zoomGlobal(Math.exp(-event.deltaY * .0015)); }
function onViewportPointerDown(event: PointerEvent) { if (current.value.type !== 'global' || (event.target as Element).closest('.global-node,.global-hive')) return; viewportPanning.value = true; viewportPointer.value = { x: event.clientX, y: event.clientY }; }
function onViewportPointerMove(event: PointerEvent) { if (!viewportPanning.value) return; const svg = event.currentTarget as SVGSVGElement; const box = svg.getBoundingClientRect(); globalPan.value = { x: globalPan.value.x + (event.clientX - viewportPointer.value.x) * 1000 / box.width, y: globalPan.value.y + (event.clientY - viewportPointer.value.y) * 600 / box.height }; viewportPointer.value = { x: event.clientX, y: event.clientY }; }
function stopViewportPan() { viewportPanning.value = false; }
function openSceneFor(cell: any) { hiveStore.selectedCell = cell; openScene(); }
function openScene() { const cell = hiveStore.selectedCell || hiveStore.cells[0]; if (cell) move({ type: 'scene', title: cell.label || 'Сцена', crumb: 'Сцена', label: cell.label || '', cell }); }
function openLexeme(node: any) { const cell = current.value.cell || hiveStore.selectedCell; move({ type: 'lexeme', title: `Лексема «${node.label}»`, crumb: `Лексема «${node.label}»`, label: node.label, cell }); if (cell) void hiveStore.expandCell(cell.id, 'lexeme'); }
function openMorphology(node: any) { move({ type: 'morphology', title: `Морфология «${node.label}»`, crumb: 'Морфология', label: node.label, cell: current.value.cell, score: node.score }); }
function openWordForm() { move({ type: 'word_form', title: `Словоформа «${current.value.label}»`, crumb: `Словоформа «${current.value.label}»`, label: current.value.label, cell: current.value.cell, score: current.value.score }); }
function openCharacters(node: any) { move({ type: 'characters', title: `Буквы «${node.label}»`, crumb: 'Буквы', label: node.label, cell: current.value.cell, score: node.score }); }
function back() { if (path.value.length > 1) path.value.pop(); }
function up() { goTo(Math.max(0, path.value.length - 2)); }
function goTo(index: number) { path.value = path.value.slice(0, index + 1); }
function characterX(index: number) { return 145 + index * Math.min(180, 700 / Math.max(letters.value.length - 1, 1)); }
function normalizeRole(role: string, index: number) { if (['subject', 'predicate', 'object', 'location', 'time', 'property', 'instrument', 'cause', 'target'].includes(role)) return role; return ['subject', 'predicate', 'object', 'location'][index % 4]; }
function roleName(role: string) { return ({ subject: 'АГЕНТ', predicate: 'ДЕЙСТВИЕ', object: 'ОБЪЕКТ', location: 'МЕСТО', time: 'ВРЕМЯ', property: 'СВОЙСТВО', instrument: 'ИНСТРУМЕНТ', cause: 'ПРИЧИНА', target: 'ЦЕЛЬ' } as Record<string, string>)[role] || 'КОНТЕКСТ'; }
function typeShort(type: string) { return ({ scene: 'СЦЕНА', memory_source: 'СЦЕНА-ИСТОЧНИК', word_form: 'ФОРМА', lexeme: 'ЛЕКСЕМА', concept: 'ПОНЯТИЕ', concept_candidate: 'ГИПОТЕЗА', semantic_bridge: 'МОСТ', role_candidate: 'КАНДИДАТ', reasoning_support: 'ПОДДЕРЖКА' } as Record<string, string>)[type] || 'ПОДДЕРЖКА'; }
function similarity(left: any[], right: any[]) { const rightIds = new Set(right.map(item => item.cloud_id || item.id)); return left.reduce((sum, item) => sum + (rightIds.has(item.cloud_id || item.id) ? Math.min(item.composition_share || .2, .7) : 0), 0); }

defineExpose({ openScene });
</script>

<style scoped lang="scss">
.space-panel{display:flex;min-width:0;min-height:0;flex-direction:column;overflow:hidden;border:1px solid rgba(132,180,236,.18);border-radius:16px;background:linear-gradient(145deg,rgba(14,31,58,.94),rgba(6,15,29,.98));box-shadow:0 20px 50px rgba(0,0,0,.24)}.space-head{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;padding:18px 20px 12px;border-bottom:1px solid rgba(162,189,225,.1)}.kicker{color:#73b0ff;font-size:10px;letter-spacing:.14em}.space-head h2{margin:3px 0 0;color:#e8f3ff;font-size:18px}.space-actions{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:6px}.secondary,.icon-button{border:1px solid rgba(126,233,208,.25);border-radius:7px;padding:7px 9px;color:#c3eee5;background:rgba(126,233,208,.07);font:10px system-ui;cursor:pointer}.secondary:disabled,.icon-button:disabled{opacity:.35;cursor:default}.secondary.selected{border-color:#ffc968;color:#ffe3a1}.icon-button{font-size:15px;line-height:12px}.breadcrumbs{display:flex;gap:4px;overflow:auto;padding:10px 20px 4px}.breadcrumbs button{padding:3px 0;border:0;color:#7f9dc5;background:none;font:10px system-ui;white-space:nowrap;cursor:pointer}.breadcrumbs button+button:before{content:'›';margin:0 8px;color:#465b7b}.breadcrumbs button.current{color:#e9f5ff}.query-strip{display:flex;align-items:center;gap:10px;margin:8px 18px 12px;padding:9px 11px;border:1px solid rgba(94,154,225,.18);border-radius:9px;color:#cbd9ec;background:rgba(46,81,126,.18);font-size:10px}.query-strip span{color:#7fa3d3;letter-spacing:.1em}.query-strip strong{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500}.query-strip em{margin-left:auto;color:#ffc968;font-style:normal}.viewport{position:relative;flex:1;min-height:390px;overflow:hidden;background:radial-gradient(circle at 50% 45%,rgba(53,93,155,.32),transparent 46%),#071321}.viewport.transitioning .space-svg{transform:scale(.88);opacity:.08}.space-svg{position:relative;z-index:1;width:100%;height:100%;transition:transform .22s ease,opacity .22s ease}.energy-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(112,163,229,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(112,163,229,.07) 1px,transparent 1px);background-size:45px 45px;mask-image:radial-gradient(circle,black,transparent 82%)}.edge{fill:none;stroke:#6a9ffc;stroke-width:1.5;stroke-dasharray:4 6}.hive-node{cursor:pointer}.hive-node .node-halo{fill:rgba(105,212,232,.04);stroke:rgba(105,212,232,.16);stroke-width:1}.hive-node .node-body{fill:url(#spaceCore);stroke:#76e6d1;stroke-width:2;filter:drop-shadow(0 0 calc(5px + var(--energy) * 11px) rgba(111,228,211,.56))}.hive-node.active .node-body{stroke:#ffc968;filter:drop-shadow(0 0 13px rgba(255,201,104,.8))}.node-type{fill:#8df2de;font:800 11px system-ui}.node-label,.node-metric{fill:#d5e4f6;font:11px system-ui;paint-order:stroke;stroke:#071321;stroke-width:3}.node-metric{fill:#8aa2c2;font-size:9px}.role-link{fill:none;stroke:#6198eb;stroke-width:1.5;stroke-dasharray:5 5}.role-node{cursor:pointer}.role-node rect{fill:rgba(26,56,92,.76);stroke:#73b0ff;stroke-width:1.5}.role-node.object rect{stroke:#b891ff;fill:rgba(82,44,128,.36)}.role-node.predicate rect{stroke:#7ee9d0;fill:rgba(24,100,91,.3)}.role-node.location rect{stroke:#ffc968;fill:rgba(111,78,26,.32)}.role-name{fill:#84aae0;font:10px system-ui;letter-spacing:.12em}.role-label{fill:#eef7ff;font:600 17px system-ui}.role-score{fill:#7f99ba;font:9px system-ui}.orbit{fill:none;stroke:rgba(180,133,255,.26);stroke-width:1.3;stroke-dasharray:4 7}.orbit.second{stroke:rgba(180,133,255,.15);stroke-dasharray:none}.lexeme-link{stroke:rgba(180,133,255,.42);stroke-width:1}.lexeme-center circle{fill:rgba(157,101,247,.22);stroke:#bd8aff;stroke-width:2;filter:url(#spaceGlow)}.lexeme-center text:first-of-type{fill:#f2eaff;font:600 22px system-ui}.lexeme-center text:last-of-type{fill:#b79bd9;font:9px system-ui}.lexeme-node{cursor:pointer}.lexeme-node circle{fill:rgba(119,65,192,.34);stroke:#c28fff;stroke-width:1.5}.lexeme-node text{fill:#f0e7ff;font:11px system-ui}.lexeme-node .score,.form-node .score{fill:#bca2dc;font-size:8px}.morph-zone{stroke-width:1.5}.morph-zone.stem{fill:rgba(49,196,229,.09);stroke:#5de6f4}.morph-zone.operators{fill:rgba(170,140,255,.1);stroke:#b69aff}.morph-zone.ending{fill:rgba(255,201,104,.1);stroke:#ffc968}.morph-zone.hypotheses{fill:rgba(121,231,208,.07);stroke:#78e7d0}.zone-label{fill:#9eb6d7;font:10px system-ui;letter-spacing:.12em}.morph-link{fill:none;stroke:#6ad6ee;stroke-width:2}.morph-link.dashed{stroke:#bb9afa;stroke-dasharray:5 6}.morph-token{cursor:default}.morph-token rect{fill:#0d2f47;stroke:#70e6f4;stroke-width:2}.morph-token text{fill:#dffaff;font:600 24px system-ui}.morph-token .small{fill:#8ebed0;font:9px system-ui}.ending-token{cursor:pointer}.ending-token rect{stroke:#ffc968}.operator-token circle{fill:rgba(159,122,239,.26);stroke:#c29bff;stroke-width:1.5}.operator-token text{fill:#ecdefe;font:10px system-ui}.hypothesis{fill:#a1e8db;font:12px system-ui}.form-field{fill:rgba(48,117,139,.08);stroke:rgba(120,231,208,.35);stroke-width:1.5;stroke-dasharray:5 7}.form-node{cursor:pointer}.form-node circle{fill:rgba(64,141,152,.35);stroke:#7ee9d0;stroke-width:1.5}.form-node.selected circle{fill:rgba(255,201,104,.22);stroke:#ffc968;stroke-width:2;filter:url(#spaceGlow)}.form-node text{fill:#e7faf5;font:600 13px system-ui}.form-node .details{fill:#8fb4af;font:8px system-ui}.character-line{stroke:rgba(255,201,104,.32);stroke-width:2}.character-node circle{fill:rgba(132,88,15,.42);stroke:#ffc968;stroke-width:2;filter:url(#spaceGlow)}.character-node text{fill:#fff3c8;font:600 29px system-ui}.character-node .position{fill:#b79854;font:9px system-ui}.character-node path{stroke:#ffc968;stroke-width:1.5;marker-end:none}.space-legend{position:absolute;z-index:2;bottom:14px;left:16px;display:flex;gap:12px;padding:7px 10px;border:1px solid rgba(144,181,223,.14);border-radius:8px;color:#91a9c9;background:rgba(5,14,27,.78);font-size:10px}.space-legend i{display:inline-block;width:7px;height:7px;margin-right:5px;border-radius:50%;background:#7ee9d0}.root-button{position:absolute;z-index:2;right:16px;bottom:14px;border:1px solid rgba(255,201,104,.35);border-radius:8px;padding:7px 10px;color:#ffdaa0;background:rgba(59,43,14,.58);font:10px system-ui;cursor:pointer}.timeline{display:flex;align-items:center;gap:14px;padding:11px 18px;border-top:1px solid rgba(162,189,225,.1);color:#849dbe;font-size:10px}.timeline>span:first-child{letter-spacing:.1em}.timeline b{color:#7ee9d0;font-weight:500}.timeline button{margin-left:auto}@media(max-width:760px){.space-head{flex-direction:column}.space-actions{justify-content:flex-start}.viewport{min-height:360px}.timeline{flex-wrap:wrap}}
.global-hive-boundary{fill:rgba(120,231,208,.06);stroke:#78e7d0;stroke-width:2;stroke-dasharray:7 6;filter:url(#spaceGlow)}.global-hive-label{fill:#8be9d8;font:10px system-ui;letter-spacing:.12em}.bee-flight{fill:none;stroke:#ffc968;stroke-width:1.3;stroke-dasharray:4 5;opacity:.72}.global-node{cursor:pointer}.global-node circle{fill:rgba(59,92,139,.36);stroke:#6d9ee0;stroke-width:1.2}.global-node.carried circle{fill:rgba(120,231,208,.24);stroke:#78e7d0;stroke-width:2;filter:url(#spaceGlow)}.global-node text{fill:#d7e8fa;font:700 9px system-ui}.global-node .global-label{fill:#a8bdd8;font:9px system-ui;font-weight:400;paint-order:stroke;stroke:#071321;stroke-width:3}.bee-marker text{fill:#ffc968;font:16px system-ui;filter:url(#spaceGlow)}
.global-hive{cursor:pointer}.global-hive>path:first-child{fill:rgba(255,201,104,.24);stroke:#ffc968;stroke-width:2;filter:url(#spaceGlow)}.global-hive .hive-slat{fill:none;stroke:#fff0be;stroke-width:3;stroke-linecap:round}.global-hive text{fill:#ffe6a2;font:700 9px system-ui;letter-spacing:.08em;paint-order:stroke;stroke:#071321;stroke-width:3}.camera-controls{position:absolute;z-index:3;top:15px;right:16px;display:flex;align-items:center;gap:5px;padding:5px;border:1px solid rgba(144,181,223,.18);border-radius:9px;background:rgba(5,14,27,.82)}.camera-controls button{min-width:27px;border:1px solid rgba(126,233,208,.2);border-radius:5px;padding:5px;color:#c6eee7;background:rgba(126,233,208,.08);font:11px system-ui;cursor:pointer}.camera-controls button:last-of-type{padding:5px 7px}.camera-controls span{min-width:32px;color:#8fa8c8;font:9px ui-monospace,Consolas,monospace;text-align:center}
.space-legend.expanded{max-width:calc(100% - 32px);row-gap:7px}.space-legend .field-concept{background:#557bb7}.space-legend .carried-concept{background:#78e7d0;box-shadow:0 0 7px #78e7d0}.space-legend .bee-route{width:15px;height:2px;border-radius:0;background:#ffc968}.space-legend .hive-boundary{width:15px;height:8px;border:1px dashed #78e7d0;border-radius:5px;background:transparent}.space-legend .hive-entry{width:11px;height:11px;border:1px solid #ffc968;border-radius:3px;background:rgba(255,201,104,.2);transform:rotate(30deg)}
.space-legend .active-cell{background:#78e7d0;box-shadow:0 0 7px #78e7d0}.space-legend .selected-cell{background:#ffc968;box-shadow:0 0 7px #ffc968}.space-legend .cell-link,.space-legend .role-link-icon,.space-legend .lexeme-link-icon,.space-legend .sequence-icon{width:15px;height:2px;border-radius:0;background:#6a9ffc}.space-legend .cell-metric,.space-legend .role-score-icon,.space-legend .form-score-icon,.space-legend .position-icon{width:8px;height:8px;border:1px solid #8fa8c8;background:transparent}.space-legend .role-area{border:1px solid #73b0ff;background:rgba(115,176,255,.35)}.space-legend .role-object{border:1px solid #b891ff;background:rgba(184,145,255,.35)}.space-legend .lexeme-center-icon{width:10px;height:10px;background:#bd8aff;box-shadow:0 0 7px #bd8aff}.space-legend .lexeme-form-icon{border:1px solid #c28fff;background:rgba(194,143,255,.4)}.space-legend .orbit-icon{width:12px;height:12px;border:1px dashed #b891ff;background:transparent}.space-legend .stem-icon{width:12px;height:8px;border:1px solid #5de6f4;border-radius:2px;background:rgba(93,230,244,.25)}.space-legend .operator-icon{background:#c29bff}.space-legend .ending-icon{width:12px;height:8px;border:1px solid #ffc968;border-radius:2px;background:rgba(255,201,104,.25)}.space-legend .hypothesis-icon{border:1px dashed #78e7d0;background:transparent}.space-legend .candidate-icon{border:1px solid #7ee9d0;background:rgba(126,233,208,.28)}.space-legend .best-candidate-icon{background:#ffc968;box-shadow:0 0 7px #ffc968}.space-legend .character-icon{width:10px;height:10px;border:1px solid #ffc968;background:rgba(255,201,104,.3)}
.query-link{stroke:rgba(120,231,208,.74);stroke-width:3;fill:none}.query-slot rect{fill:rgba(21,61,88,.92);stroke:#73b0ff;stroke-width:2}.query-slot.empty rect{fill:rgba(103,71,25,.92);stroke:#ffc968;filter:drop-shadow(0 0 10px rgba(255,201,104,.5))}.query-slot.resolved rect{fill:rgba(25,83,72,.94);stroke:#78e7d0}.query-role{fill:#8ca6c9;font:700 9px system-ui}.query-label{fill:#eef8ff;font:700 13px system-ui}.query-candidate{transition:transform .35s ease,opacity .35s ease}.query-candidate circle{fill:rgba(47,114,145,.9);stroke:#78e7d0;stroke-width:2}.query-candidate text{fill:#eaf7ff;font:10px system-ui;pointer-events:none}.query-candidate.strengthened circle,.query-candidate.stable circle{fill:rgba(39,144,122,.94)}.query-candidate.winner circle{fill:rgba(213,147,48,.95);stroke:#ffe1a1;filter:drop-shadow(0 0 11px rgba(255,201,104,.65))}.query-candidate.weakened,.query-candidate.evicted{opacity:.25}.query-candidate.conflict circle{fill:rgba(169,61,67,.9);stroke:#ff9b9b}.query-step{fill:#9ab0cc;font:10px system-ui;letter-spacing:.12em}
.query-candidate .candidate-type{fill:#8df2de;font-size:7px;font-weight:700;letter-spacing:.08em}.semantic-bridge{pointer-events:none}.semantic-bridge path{fill:none;stroke:#b891ff;stroke-width:2;stroke-dasharray:4 5}.semantic-bridge circle{fill:rgba(96,60,153,.48);stroke:#c69cff;stroke-width:2;filter:drop-shadow(0 0 10px rgba(190,144,255,.5))}.semantic-bridge text{fill:#f1e8ff;font:600 10px system-ui}.semantic-bridge .bridge-type{fill:#d8bfff;font-size:8px;letter-spacing:.1em}.semantic-bridge .bridge-detail{fill:#bda8d8;font:8px system-ui;paint-order:stroke;stroke:#071321;stroke-width:3}
.viewport.hive-viewport{overflow-x:hidden;overflow-y:auto}.viewport.hive-viewport .hive-mode-panel{min-height:max-content}
</style>
