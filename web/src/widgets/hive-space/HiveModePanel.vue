<template>
  <section class="hive-mode-panel">
    <nav class="mode-tabs" aria-label="Режим визуализации">
      <button v-for="tab in tabs" :key="tab.id" :class="{ active: mode === tab.id }" @click="setMode(tab.id)">{{ tab.label }}<i v-if="tab.id === 'answer' && visualization.answerBuild.reverseValidation.status === 'PASSED'">●</i></button>
    </nav>

    <div class="mode-status"><span>{{ isProbe ? 'РЕЗОНАНСНЫЙ СИГНАЛ' : 'АКТИВНЫЙ ЗАПРОС' }}</span><b>{{ isProbe ? resonanceProbe?.input : (visualization.scene.activeQuery || '—') }}</b><em>{{ stateLabel }}</em></div>

    <div v-if="mode === 'scene'" class="scene-view">
      <div class="scene-flow">
        <template v-for="(slot, index) in sceneSlots" :key="slot.id || slot.role">
          <article class="scene-slot" :class="slotClass(slot)" @click="inspect(slot)">
            <span class="slot-role">{{ slot.role }}</span>
            <strong>{{ slot.lemma }}</strong>
            <small v-if="slot.surface && slot.surface !== slot.lemma">форма запроса: «{{ slot.surface }}»</small>
            <small v-if="slot.status === 'EMPTY'">статус: EMPTY</small>
            <small v-else-if="slot.status !== 'FIXED'">статус: {{ slot.status }}</small>
            <small v-if="slot.status === 'RESOLVED'">ответ на роль «{{ slot.question_word || slot.secondary || slot.role }}» · уверенность {{ Math.round(Number(slot.confidence || candidateScore / 100) * 100) }}%</small>
            <div v-if="slot.role === 'OBJECT' && bridge" class="bridge-badge" @click.stop="bridgeOpen = !bridgeOpen">{{ bridge.surface }} → {{ bridge.lemma }} → {{ bridge.global }} · {{ bridge.confidence }}%</div>
            <div v-if="bridgeOpen && slot.role === 'OBJECT' && bridge" class="bridge-detail"><span>Форма пользователя: {{ bridge.surface }}</span><span>Гипотеза леммы: {{ bridge.lemma || '—' }}</span><span>Общая основа: {{ bridge.sharedBase || '—' }}</span><span>Тип связи: semantic bridge</span></div>
          </article>
          <span v-if="index < sceneSlots.length - 1" class="scene-arrow">→</span>
        </template>
      </div>
      <div v-if="candidatesVisible" class="candidate-strip"><div class="candidate-title">КАНДИДАТЫ РОЛИ · {{ requestedRoleLabel }}</div><button v-for="candidate in visualization.search.candidates" :key="candidate.id" :class="candidate.status" @click="inspect(candidate)"><b>{{ candidate.label }}</b><span>{{ candidate.score }}%</span><small>{{ candidate.selection_reason || candidate.status }}</small></button></div>
      <div class="scene-actions"><button @click="mode = 'search'">Показать ход поиска</button><button @click="sourcesOpen = !sourcesOpen">Показать источники</button><button @click="mode = 'answer'">Показать сборку ответа</button></div>
      <div v-if="sourcesOpen" class="source-list"><div class="section-title">СЦЕНЫ ПАМЯТИ · {{ visualization.search.sources.length }}</div><article v-for="source in visualization.search.sources" :key="source.id" class="source-card" :class="{ selected: source.selected, conflict: source.result_type === 'CONFLICT_HIT' }"><div><strong>{{ source.text }}</strong><em>{{ source.result_type }}</em></div><span>{{ sourceScoreSummary(source) }}</span><b>{{ source.selection_reason || 'сцена не дала опорного совпадения' }}</b></article></div>
    </div>

    <div v-else-if="mode === 'search'" class="search-view"><div class="search-columns"><div class="timeline"><div v-if="!traceStages.length" class="mission"><span>○</span><div><b>Поиск не запускался</b><small>Для активного запроса пока нет маршрута поиска.</small></div></div><details v-for="(stage, index) in traceStages" :key="stage.id" class="trace-stage"><summary><span>{{ stage.status === 'NO_MATCH' || stage.status === 'FAILED' ? '!' : index < traceStages.length - 1 ? '✓' : '→' }}</span><div><b>{{ stage.label }}</b><small>{{ stage.status }} · {{ stage.summary }}</small></div></summary><pre>{{ JSON.stringify(stage.raw, null, 2) }}</pre></details></div><div class="evidence"><div class="section-title">ИСТОЧНИКИ И ДОКАЗАТЕЛЬСТВА</div><article v-for="source in visualization.search.sources" :key="source.id" class="source-card" :class="{ selected: source.selected, conflict: source.result_type === 'CONFLICT_HIT' }"><div><strong>{{ source.text }}</strong><em>{{ source.result_type }}</em></div><span>{{ sourceScoreSummary(source) }}</span><b>{{ source.selection_reason || `общий вес: ${source.score}%` }}</b></article><div v-if="bridge" class="route">ФОРМА → {{ bridge.surface }} · ЛЕММА → {{ bridge.lemma || '—' }} · ОСНОВА → {{ bridge.sharedBase || '—' }} · ГЛОБАЛЬНАЯ ЛЕКСЕМА → {{ bridge.global || '—' }}</div></div></div></div>

    <div v-else-if="mode === 'structure'" class="structure-view">
      <div class="structure-summary"><section><span>СОСТАВ РАБОЧЕГО УЛЬЯ</span><b>Рабочие ячейки: {{ visualization.hiveStructure.placements.working_cells }}</b><b>Источники памяти: {{ visualization.hiveStructure.placements.memory_sources }}</b><b>Всего физических размещений: {{ visualization.hiveStructure.placements.total }}</b></section><section><span>СТРУКТУРНОЕ ПОГРУЖЕНИЕ</span><b>Выбранный объект: {{ activeWord || 'не выбран' }}</b><b>Лексема → морфология → морфемы → буквы</b></section></div>
      <div class="scale-switcher" aria-label="Масштаб внутреннего пространства">
        <button v-for="scale in structureScales" :key="scale.id" :class="{ active: structureScale === scale.id }" @click="structureScale = scale.id">{{ scale.label }} <b>{{ scale.count }}</b></button>
      </div>
      <div class="hive-structure" :class="`focus-${structureScale}`">
        <section class="scale-zone hive-zone">
          <span class="scale-caption">УЛЕЙ</span>
          <div class="hive-core"><small>РАБОЧАЯ ПАМЯТЬ</small><strong>{{ visualization.hiveStructure.placements.working_cells }}</strong><em>активных ячеек</em></div>
        </section>
        <span class="scale-link" aria-hidden="true">→</span>
        <section class="scale-zone words-zone">
          <span class="scale-caption">СЛОВА</span>
          <div class="honeycomb words-honeycomb">
            <button v-for="word in wordNodes" :key="word" class="hex word-hex" :class="{ selected: activeWord === word }" @click="focusedWord = word">{{ word }}</button>
          </div>
        </section>
        <span class="scale-link" aria-hidden="true">→</span>
        <section class="scale-zone morph-zone">
          <span class="scale-caption">МОРФЕМЫ · {{ focusedWord }}</span>
          <div v-if="activeWord" class="honeycomb morph-honeycomb"><span v-for="part in morphemeNodes" :key="part" class="hex morph-hex">{{ part }}</span></div><div v-else class="structure-empty">Выберите лексему</div>
        </section>
        <span class="scale-link" aria-hidden="true">→</span>
        <section v-if="activeWord" class="scale-zone letters-zone">
          <span class="scale-caption">БУКВЫ</span>
          <div class="letter-row"><span v-for="(letter, index) in letterNodes" :key="`${letter}-${index}`" class="letter-cell">{{ letter }}</span></div>
        </section>
      </div>
      <div class="structure-detail"><span>Выбрано: <b>{{ activeWord || '—' }}</b></span><span>{{ activeWord ? morphemeNodes.length : 0 }} морфемы</span><span>{{ activeWord ? letterNodes.length : 0 }} букв</span></div>
    </div>

    <div v-else-if="mode === 'dynamics'" class="dynamics-view">
      <div class="dynamics-toolbar"><span>ШАГ {{ hiveStore.dynamics?.step || 0 }}</span><button @click="hiveStore.runReasoningStep">Один шаг</button><button @click="showForces = !showForces">{{ showForces ? 'Скрыть силы' : 'Показать силы' }}</button><button @click="showTrajectories = !showTrajectories">{{ showTrajectories ? 'Скрыть следы' : 'Показать следы' }}</button></div>
      <svg class="dynamics-field" viewBox="0 0 1000 620" role="img" aria-label="Физическое поле рабочего улья">
        <circle cx="500" cy="310" r="92" class="zone-core" /><circle cx="500" cy="310" r="180" class="zone-active" /><circle cx="500" cy="310" r="270" class="zone-candidate" /><circle cx="500" cy="310" r="350" class="zone-eviction" />
        <g v-for="anchor in hiveStore.dynamics?.anchors || []" :key="String(anchor.anchor_id)" class="dynamics-anchor"><circle :cx="anchorX(anchor)" :cy="anchorY(anchor)" r="10" /><text :x="anchorX(anchor)" :y="anchorY(anchor) - 15">{{ String(anchor.role || '') }}</text></g>
        <g v-for="node in hiveStore.dynamics?.nodes || []" :key="node.cell_id" class="dynamics-node" :class="node.eviction_status" @click="selectDynamicsNode(node)">
          <path v-if="showForces" :d="forcePath(node)" class="net-arrow" /><polyline v-if="showTrajectories" :points="node.trajectory.map(point => `${point.x * 1000},${point.y * 620}`).join(' ')" class="trajectory-line" />
          <circle :cx="node.position.x * 1000" :cy="node.position.y * 620" :r="12 + node.mass.local * 15" :style="{ opacity: .35 + node.activation * .65 }" /><text :x="node.position.x * 1000" :y="node.position.y * 620 + 30">{{ node.label || node.cell_id }}</text>
        </g>
      </svg>
      <div v-if="selectedDynamicsNode" class="dynamics-inspector"><strong>{{ selectedDynamicsNode.label || selectedDynamicsNode.cell_id }}</strong><span>сила {{ selectedDynamicsNode.net_force.magnitude.toFixed(3) }}</span><span>статус {{ selectedDynamicsNode.eviction_status }} · зона {{ selectedDynamicsNode.zone }}</span><span>температура {{ Math.round((hiveStore.dynamics?.temperature.current || 0) * 100) }}%</span></div>
    </div>

    <div v-else-if="mode === 'resonance'" class="resonance-view">
      <div class="resonance-scope" role="group" aria-label="Область резонанса"><button :class="{ active: resonanceScope === 'LOCAL_ONLY' }" :disabled="hiveStore.loading" @click="setResonanceScope('LOCAL_ONLY')">Локальная память</button><button :class="{ active: resonanceScope === 'LOCAL_THEN_GLOBAL' }" :disabled="hiveStore.loading" @click="setResonanceScope('LOCAL_THEN_GLOBAL')">Глобальная память</button></div>
      <section class="probe-route"><b>Сигнал: {{ resonanceProbe?.input || visualization.resonance.probe_text || '—' }}</b><span>Тип: {{ resonanceProbe?.probe_type || '—' }}</span><span>Стадия: {{ visualization.resonance.status }}</span><span>Точная форма: {{ resonanceProbe?.local_search?.matches?.length || resonanceProbe?.global_search?.matches?.length ? 'найдена' : 'не найдена' }}</span><span>Структура: {{ resonanceProbe?.signature?.possible_stems?.join(', ') || '—' }}</span></section>
      <section v-if="resonanceMatches.length" class="resonance-results"><article v-for="match in resonanceMatches" :key="match.id" class="resonance-card"><div><strong>{{ match.value }}</strong><b>{{ Math.round(match.score * 100) }}%</b></div><span>тип: {{ match.type }}</span><span>совпадение: {{ match.match_type }}</span><span>общий фрагмент: {{ match.shared_structure }}</span><small>связанных сцен: {{ match.related_scenes?.length || 0 }}</small><button @click="hiveStore.importResonanceMatch(match.id)">Перенести в улей</button></article></section>
      <small v-else>Локальных и глобальных откликов пока нет.</small>
    </div>

    <div v-else class="answer-view">
      <div class="answer-tabs"><button :class="{ active: answerTab === 'short' }" @click="answerTab = 'short'">Короткий ответ</button><button :class="{ active: answerTab === 'full' }" @click="answerTab = 'full'">Полный ответ</button></div>
      <div v-if="activePlan.source_scene_text" class="plan-source"><span>СЦЕНА-ИСТОЧНИК</span><b>{{ activePlan.source_scene_text }}</b></div>
      <div v-if="activePlan.slots.length" class="plan-grid">
        <article v-for="slot in activePlan.slots" :key="slot.slot_id || slot.role" class="plan-slot">
          <span>{{ String(slot.role || '').toUpperCase() }}</span>
          <strong>{{ slot.lemma || slot.local_lemma || slot.surface }}</strong>
          <small v-if="slot.surface">форма: {{ slot.surface }}</small>
          <small v-if="formSourceLabel(slot)">источник: {{ formSourceLabel(slot) }}</small>
          <small v-if="featureSummary(slot.observed_features)">наблюдалось: {{ featureSummary(slot.observed_features) }}</small>
          <small v-if="featureSummary(slot.requested_features)">требовалось: {{ featureSummary(slot.requested_features) }}</small>
          <div v-if="slot.role === 'location' || slot.role === 'LOCATION'" class="operators">{{ [slot.preposition, slot.requested_features?.case, slot.requested_features?.number].filter(Boolean).join(' · ').toUpperCase() }}</div>
        </article>
      </div>
      <div v-if="assemblyLine" class="assembly-line">{{ assemblyLine }}</div>
      <div class="answer-result">{{ activeAnswer || answerMessage }}</div>
      <div class="validation"><div><span>ОБРАТНАЯ ПРОВЕРКА</span><b>{{ visualization.answerBuild.reverseValidation.status }}</b></div><small>{{ validationSummary }}</small><strong>{{ Math.round(Number(visualization.answerBuild.reverseValidation.score || 0) * 100) }}%</strong></div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useHiveStore } from '@/entities/hive/store';
import { mapVisualization, type HiveMode } from '@/features/reasoning/model/visualizationMapper';

const hiveStore = useHiveStore();
const mode = ref<HiveMode>('scene');
const answerTab = ref<'short' | 'full'>('short');
const structureScale = ref<'all' | 'words' | 'morphemes' | 'letters'>('all');
const bridgeOpen = ref(false);
const sourcesOpen = ref(false);
const showForces = ref(true);
const showTrajectories = ref(true);
const selectedDynamicsNode = ref<any>(null);
const selected = ref<unknown>(null);
const resonanceScope = ref<'LOCAL_ONLY' | 'LOCAL_THEN_GLOBAL'>('LOCAL_THEN_GLOBAL');
const tabs = computed(() => isProbe.value ? [{ id: 'resonance' as HiveMode, label: 'Резонанс' }, { id: 'structure' as HiveMode, label: 'Структура' }, { id: 'dynamics' as HiveMode, label: 'Импорт в улей' }] : [{ id: 'scene' as HiveMode, label: 'Сцена' }, { id: 'resonance' as HiveMode, label: 'Локальный резонанс' }, { id: 'structure' as HiveMode, label: 'Устройство улья' }, { id: 'dynamics' as HiveMode, label: 'Динамика' }, { id: 'search' as HiveMode, label: 'Поиск пчёл' }, { id: 'answer' as HiveMode, label: 'Сборка ответа' }]);
const visualization = computed(() => mapVisualization({ queryScene: hiveStore.queryScene, queryFrame: hiveStore.queryFrame, activeQuery: hiveStore.activeQuery, queryCandidates: hiveStore.queryCandidates, memoryScenes: hiveStore.memoryScenes, memorySources: hiveStore.memorySources, unknownTokenSearches: hiveStore.unknownTokenSearches, localResonance: hiveStore.localResonance, resonanceProbes: hiveStore.resonanceProbes, hiveStructure: (hiveStore as any).hiveStructure, vibrationHistory: hiveStore.vibrationHistory, sentencePlan: hiveStore.sentencePlan, fullSentencePlan: hiveStore.fullSentencePlan, generationCandidates: hiveStore.generationCandidates, morphologyTrace: hiveStore.morphologyTrace, reverseValidation: hiveStore.reverseValidation, queryAnswer: hiveStore.queryAnswer }));
const resonanceProbe = computed<any>(() => visualization.value.resonance.probe);
const isProbe = computed(() => Boolean(resonanceProbe.value));
const resonanceMatches = computed<any[]>(() => resonanceProbe.value?.local_results?.length ? resonanceProbe.value.local_results : resonanceProbe.value?.global_results || []);
const bridge = computed(() => visualization.value.scene.bridge);
const sceneSlots = computed(() => visualization.value.scene.slots);
const activePlan = computed(() => (answerTab.value === 'short' ? visualization.value.answerBuild.shortPlan : visualization.value.answerBuild.fullPlan) || { slots: [] });
const candidateScore = computed(() => visualization.value.search.candidates[0]?.score || 0);
const activeAnswer = computed(() => answerTab.value === 'short' ? visualization.value.answerBuild.answer : visualization.value.answerBuild.fullAnswer);
const assemblyLine = computed(() => activePlan.value.slots.map((slot: any) => [slot.preposition, slot.surface || slot.lemma || slot.local_lemma].filter(Boolean).join(' ')).filter(Boolean).join(' + '));
function featureSummary(features: Record<string, unknown> | null | undefined) {
  if (!features) return '';
  return [features.case, features.number, features.gender].filter(Boolean).join(' · ').toUpperCase();
}
function formSourceLabel(slot: any) {
  if (slot.form_provenance?.source_type === 'observed_training_form') return 'наблюдалась в обучающей сцене';
  if (slot.context_source === 'memory_scene') return 'сцена памяти';
  if (slot.context_source === 'query_frame' || slot.source_type === 'query_surface') return 'текущий запрос';
  if (slot.source_type === 'known_word_form') return 'известная словоформа памяти';
  return '';
}
const answerMessage = computed(() => {
  if (hiveStore.queryAnswer?.status === 'UNRESOLVED') return 'Подходящий ответ в доступной памяти не найден.';
  if (hiveStore.loading || hiveStore.reasoningLoading) return 'Улей подбирает ответ…';
  return (hiveStore.queryAnswer as any)?.status_message || 'Ответ появится после разрешения роли в сцене.';
});
const validationSummary = computed(() => {
  const validation = visualization.value.answerBuild.reverseValidation;
  if (validation.status === 'WAITING') return hiveStore.queryAnswer?.status === 'UNRESOLVED' ? 'Ответ не найден, обратная проверка не требуется.' : 'Проверка будет доступна после разрешения роли.';
  const checks = validation.checks || {};
  return Object.entries(checks).filter(([, value]) => value).map(([key]) => key.replace('_preserved', '').replace('_', ' ')).join(' · ') || 'Нет подтверждённых проверок.';
});
const candidatesVisible = computed(() => sceneSlots.value.some((slot: any) => ['EMPTY', 'SEARCHING', 'CANDIDATES_FOUND', 'RESOLVING'].includes(slot.status)));
const requestedRoleLabel = computed(() => String((hiveStore.queryFrame as any)?.requested_role || hiveStore.queryScene?.requested_role || '—').toUpperCase());
const stageLabels: Record<string, string> = {
  INTENT_CLASSIFICATION: 'Классификация входа', QUERY_FRAME: 'Формирование сцены запроса',
  MEMORY_SCENE_SEARCH: 'Поиск по сценам памяти', CANDIDATE_RANKING: 'Ранжирование кандидатов',
  SEMANTIC_BRIDGE: 'Семантический мост', VIBRATION: 'Вибрация кандидатов', ANSWER_ASSEMBLY: 'Сборка и обратная проверка',
};
const traceStages = computed(() => {
  const stages = ((hiveStore.reasoningTrace as any)?.stages || []).map((stage: any) => ({
    id: stage.id, label: stageLabels[stage.stage] || stage.stage, status: stage.status || 'WAITING',
    summary: traceSummary(stage), raw: stage,
  }));
  const missions = visualization.value.search.missions.map((mission: any[], index: number) => ({
    id: `mission-${index}-${mission[0]}`, label: `Пчела · ${mission[1]}`, status: 'COMPLETED', summary: mission[2], raw: mission,
  }));
  return [...stages, ...missions];
});
const answerStatus = computed(() => String((hiveStore.queryPipeline as any)?.answer?.status || (hiveStore.queryAnswer as any)?.status || ''));
const stateLabel = computed(() => isProbe.value ? visualization.value.resonance.status : hiveStore.loading || hiveStore.reasoningLoading ? 'HIVE_REASONING' : answerStatus.value === 'RESOLVED' ? 'ANSWER_READY' : hiveStore.queryScene?.status === 'RESOLVED' ? 'ROLE_RESOLVED' : 'SEARCHING');
const wordNodes = computed(() => {
  const words = [...sceneSlots.value.map((slot: any) => String(slot.lemma || slot.surface || '')), ...hiveStore.cells.map(cell => String(cell.label || ''))]
    .filter(Boolean);
  return [...new Set(words)].slice(0, 7);
});
const focusedWord = ref('');
const activeWord = computed(() => wordNodes.value.includes(focusedWord.value) ? focusedWord.value : '');
const morphemeNodes = computed(() => splitMorphemes(activeWord.value));
const letterNodes = computed(() => Array.from(activeWord.value.replace(/[^а-яёa-z]/gi, '')));
const structureScales = computed(() => [
  { id: 'all' as const, label: 'Весь улей', count: hiveStore.cells.length || wordNodes.value.length },
  { id: 'words' as const, label: 'Слова', count: wordNodes.value.length },
  { id: 'morphemes' as const, label: 'Морфемы', count: morphemeNodes.value.length },
  { id: 'letters' as const, label: 'Буквы', count: letterNodes.value.length },
]);
function slotClass(slot: any) { return String(slot.status || 'EMPTY').toLowerCase(); }
function inspect(item: unknown) { selected.value = item; }
function selectDynamicsNode(node: unknown) { selectedDynamicsNode.value = node; }
function forcePath(node: any) { const x = node.position.x * 1000; const y = node.position.y * 620; return `M ${x} ${y} L ${x + node.net_force.x * 20} ${y + node.net_force.y * 20}`; }
function anchorX(anchor: any) { return Number(anchor.position?.x || .5) * 1000; }
function anchorY(anchor: any) { return Number(anchor.position?.y || .5) * 620; }
function setMode(nextMode: HiveMode) { mode.value = nextMode; if (nextMode === 'structure') void hiveStore.hierarchy(); }
async function setResonanceScope(scope: 'LOCAL_ONLY' | 'LOCAL_THEN_GLOBAL') { resonanceScope.value = scope; await hiveStore.rerunLocalResonance(scope); }
function traceSummary(stage: any) {
  if (stage.stage === 'MEMORY_SCENE_SEARCH') return `${Array.isArray(stage.output) ? stage.output.filter((item: any) => item.result_type !== 'NO_HIT').length : 0} релевантных сцен`;
  if (stage.stage === 'CANDIDATE_RANKING') return `${Array.isArray(stage.output) ? stage.output.length : 0} кандидатов`;
  if (stage.stage === 'VIBRATION') return `шаг ${stage.step || 0}${stage.output?.winner ? ' · победитель найден' : ''}`;
  if (stage.stage === 'ANSWER_ASSEMBLY') return stage.output?.short_answer || 'проверка ответа';
  if (stage.stage === 'SEMANTIC_BRIDGE') return `${stage.output?.surface || 'форма'} → ${stage.output?.selected_candidate?.candidate_lexeme || 'нет связи'}`;
  return stage.output?.requested_role ? `роль ответа: ${stage.output.requested_role}` : stage.input || 'готово';
}
function sourceScoreSummary(source: any) {
  const roles = Object.entries(source.scores?.role_matches || {}).map(([role, score]) => `${role}: ${Math.round(Number(score) * 100)}%`);
  return [...roles, `итог: ${source.score}%`].join(' · ');
}
function splitMorphemes(word: string) {
  const normalized = word.toLowerCase().replace(/[^а-яёa-z]/gi, '');
  if (!normalized) return [];
  if (normalized.endsWith('ть') && normalized.length > 3) return [normalized.slice(0, -2), 'ть'];
  if (normalized.length >= 5) return [normalized.slice(0, -2), normalized.slice(-2, -1), normalized.slice(-1)];
  if (normalized.length >= 3) return [normalized.slice(0, -1), normalized.slice(-1)];
  return [normalized];
}
</script>

<style scoped lang="scss">
.hive-mode-panel{position:relative;z-index:2;display:flex;min-height:max-content;box-sizing:border-box;flex-direction:column;padding:20px 20px 32px;background:radial-gradient(circle at 50% 35%,rgba(61,100,166,.18),transparent 55%),#071321;color:#dceaff}.mode-tabs{display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr));gap:6px;margin-bottom:14px}.mode-tabs button,.answer-tabs button{min-width:0;border:1px solid rgba(125,176,237,.22);border-radius:8px;padding:9px 8px;color:#8da7c8;background:rgba(10,27,48,.76);font:600 10px system-ui;cursor:pointer}.mode-tabs button.active,.answer-tabs button.active{border-color:#73b0ff;color:#f1f7ff;background:rgba(60,104,170,.34)}.mode-tabs i{margin-left:7px;color:#78e7d0;font-style:normal}.mode-status{display:flex;align-items:center;gap:12px;padding:9px 12px;border:1px solid rgba(120,231,208,.18);border-radius:8px;color:#86a2c5;font-size:10px}.mode-status b{min-width:0;overflow:hidden;color:#78e7d0;text-overflow:ellipsis;white-space:nowrap}.mode-status em{margin-left:auto;color:#7187a7;font-style:normal;white-space:nowrap}.scene-view,.search-view,.answer-view,.structure-view{flex:1;padding-top:30px}.scene-flow{display:flex;align-items:center;justify-content:center;gap:18px;min-height:235px}.scene-slot{position:relative;width:180px;min-height:105px;padding:16px;border:1px solid #73b0ff;border-radius:12px;background:rgba(26,63,105,.66);cursor:pointer}.scene-slot.resolved{border-color:#78e7d0;background:rgba(25,83,72,.68)}.scene-slot.empty,.scene-slot.searching,.scene-slot.candidates_found,.scene-slot.resolving{border-color:#ffc968;background:rgba(90,62,22,.6)}.slot-role,.section-title{display:block;color:#8fb3e5;font-size:10px;letter-spacing:.12em}.scene-slot strong{display:block;margin:8px 0;color:#f1f7ff;font-size:20px}.scene-slot small{display:block;color:#9cb2d0;font-size:10px}.scene-arrow{color:#73b0ff;font-size:26px}.bridge-badge{margin-top:10px;padding-top:8px;border-top:1px solid rgba(189,138,255,.25);color:#d5b9ff;font-size:10px}.bridge-detail{display:grid;gap:4px;margin-top:8px;padding:8px;border-radius:7px;color:#dccbfa;background:rgba(77,45,130,.42);font-size:9px}.candidate-strip,.source-list{margin:12px auto;max-width:760px;padding:13px;border:1px solid rgba(255,201,104,.2);border-radius:10px;background:rgba(45,31,13,.35)}.candidate-title{margin-bottom:8px;color:#ffc968;font-size:10px;letter-spacing:.1em}.candidate-strip button{display:inline-grid;grid-template-columns:auto auto;gap:3px 9px;margin:4px;padding:8px 10px;border:1px solid #ffc968;border-radius:12px;color:#ffe6ad;background:rgba(137,91,19,.35);cursor:pointer}.candidate-strip button span{color:#ffc968}.candidate-strip button small{grid-column:1/-1;max-width:240px;overflow:hidden;color:#a99977;font-size:8px;text-overflow:ellipsis;white-space:nowrap}.scene-actions{display:flex;justify-content:center;gap:8px;margin:20px 0}.scene-actions button{border:0;border-radius:7px;padding:8px 11px;color:#b9efe3;background:rgba(120,231,208,.1);font:10px system-ui;cursor:pointer}.source-list{display:grid;gap:8px}.source-card{display:grid;gap:6px;padding:11px;border-left:3px solid #526985;background:rgba(17,53,66,.5);font-size:10px}.source-card.selected{border-left-color:#78e7d0}.source-card.conflict{border-left-color:#e56b6f}.source-card>div{display:flex;align-items:center;justify-content:space-between;gap:10px}.source-card strong{color:#effaff;font-size:12px}.source-card em{color:#7f96b4;font:8px ui-monospace,Consolas,monospace}.source-card span{color:#8bb7bc}.source-card b{color:#78e7d0;font-weight:500}.source-card.conflict b{color:#ff9f9f}.search-columns{display:grid;grid-template-columns:minmax(250px,.8fr) minmax(280px,1fr);gap:24px;max-width:900px;margin:auto}.timeline{display:grid;align-content:start;gap:5px}.mission{display:flex;gap:12px;padding:12px;border-left:2px solid #78e7d0;background:rgba(17,45,66,.42)}.mission>span{color:#78e7d0}.mission b,.mission small{display:block}.mission b{color:#eef7ff;font-size:12px}.mission small{margin-top:4px;color:#9cb2d0;font-size:10px}.trace-stage{border-left:2px solid #78e7d0;background:rgba(17,45,66,.42)}.trace-stage summary{display:flex;gap:12px;padding:12px;list-style:none;cursor:pointer}.trace-stage summary::-webkit-details-marker{display:none}.trace-stage summary>span{color:#78e7d0}.trace-stage summary div{min-width:0}.trace-stage b,.trace-stage small{display:block}.trace-stage b{color:#eef7ff;font-size:12px}.trace-stage small{margin-top:4px;color:#9cb2d0;font-size:10px}.trace-stage pre{max-height:240px;margin:0;padding:12px;overflow:auto;border-top:1px solid rgba(120,231,208,.12);color:#9ed6c9;background:rgba(3,12,23,.72);font:9px/1.45 ui-monospace,Consolas,monospace;white-space:pre-wrap}.evidence{display:grid;align-content:start;gap:9px}.route{padding:14px;border:1px dashed #b891ff;border-radius:8px;color:#d6bdff;font-size:11px;line-height:1.8}.plan-source{display:grid;gap:5px;margin-top:12px;padding:10px 12px;border:1px solid rgba(120,231,208,.2);border-radius:8px;background:rgba(20,62,61,.28)}.plan-source span{color:#78e7d0;font-size:9px;letter-spacing:.1em}.plan-source b{color:#dff7f1;font-size:11px;font-weight:500}.plan-grid{display:flex;gap:10px;overflow:auto;padding:16px 0}.plan-slot{min-width:150px;padding:13px;border:1px solid #73b0ff;border-radius:10px;background:rgba(25,56,93,.55)}.plan-slot span,.plan-slot strong,.plan-slot small{display:block}.plan-slot span{color:#8fb3e5;font-size:9px;letter-spacing:.1em}.plan-slot strong{margin:8px 0;color:#f4f8ff;font-size:16px}.plan-slot small{margin-top:3px;color:#a2b5cf;font-size:10px}.operators{margin-top:10px;color:#c9b0ff;font-size:9px}.assembly-line{margin:12px 0;color:#cdb8ff;font-size:14px;text-align:center}.answer-result{padding:18px;border:1px solid #78e7d0;border-radius:10px;color:#f3fffc;background:rgba(29,105,90,.25);font-size:26px;text-align:center}.validation{display:grid;gap:7px;margin-top:16px;padding:13px;border:1px solid rgba(120,231,208,.25);border-radius:9px}.validation div{display:flex;justify-content:space-between;color:#78e7d0;font-size:10px;letter-spacing:.1em}.validation small{color:#a5c0bc;font-size:10px}.validation>strong{color:#78e7d0;font-size:18px}

.structure-summary{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-bottom:18px}.structure-summary section{display:grid;align-content:start;gap:6px;min-width:0;padding:12px;border:1px solid rgba(115,176,255,.18);border-radius:9px;background:rgba(13,35,59,.46)}.structure-summary span{color:#89a9d0;font-size:9px;letter-spacing:.12em}.structure-summary b{color:#dceaff;font-size:11px;font-weight:500;line-height:1.35}.scale-switcher{display:flex;flex-wrap:wrap;justify-content:center;gap:6px;margin-bottom:22px}.scale-switcher button{border:1px solid rgba(125,176,237,.22);border-radius:999px;padding:7px 11px;color:#8da7c8;background:rgba(10,27,48,.76);font:10px system-ui;cursor:pointer}.scale-switcher button.active{border-color:#78e7d0;color:#e5fffa;background:rgba(48,129,115,.3)}.scale-switcher b{margin-left:4px;color:#ffc968;font-weight:500}.hive-structure{display:grid;grid-template-columns:minmax(105px,.75fr) auto minmax(170px,1.1fr) auto minmax(130px,.9fr) auto minmax(120px,.7fr);align-items:center;gap:12px;min-height:270px;transition:opacity .18s ease}.scale-zone{min-width:0;transition:opacity .18s ease,transform .18s ease}.scale-caption{display:block;margin-bottom:12px;color:#89a9d0;font-size:9px;letter-spacing:.12em;text-align:center}.scale-link{color:#5d9df4;font-size:22px}.hive-core{display:grid;place-content:center;justify-items:center;width:112px;height:128px;margin:auto;clip-path:polygon(25% 2%,75% 2%,100% 50%,75% 98%,25% 98%,0 50%);color:#e8fbff;background:linear-gradient(145deg,rgba(69,123,189,.88),rgba(24,66,116,.88));box-shadow:inset 0 0 0 1px #7ab3ff,0 0 24px rgba(89,147,235,.26)}.hive-core small{color:#b9d6fb;font-size:7px;letter-spacing:.06em}.hive-core strong{margin:3px 0;color:#fff;font-size:28px}.hive-core em{color:#a8c9ee;font-size:9px;font-style:normal}.honeycomb{display:flex;flex-wrap:wrap;justify-content:center;align-content:center;gap:4px}.hex{display:grid;place-items:center;width:68px;height:58px;border:0;clip-path:polygon(25% 2%,75% 2%,100% 50%,75% 98%,25% 98%,0 50%);font:600 10px system-ui}.word-hex{color:#ddf7f2;background:rgba(40,126,117,.72);cursor:pointer}.word-hex.selected{color:#fff3cf;background:rgba(197,134,37,.92);box-shadow:0 0 0 2px #ffd67f}.morph-honeycomb{max-width:165px}.morph-hex{width:55px;height:48px;color:#f1e8ff;background:rgba(108,71,166,.75);font-size:9px}.letter-row{display:flex;flex-wrap:wrap;justify-content:center;gap:5px;max-width:150px;margin:auto}.letter-cell{display:grid;place-items:center;width:31px;height:35px;clip-path:polygon(25% 2%,75% 2%,100% 50%,75% 98%,25% 98%,0 50%);color:#fff0c3;background:rgba(183,126,36,.76);font:600 13px system-ui}.focus-words .hive-zone,.focus-words .morph-zone,.focus-words .letters-zone,.focus-morphemes .hive-zone,.focus-morphemes .words-zone,.focus-morphemes .letters-zone,.focus-letters .hive-zone,.focus-letters .words-zone,.focus-letters .morph-zone{opacity:.3;transform:scale(.94)}.structure-detail{display:flex;justify-content:center;gap:18px;margin-top:24px;color:#91a8c8;font-size:10px}.structure-detail b{color:#f0f8ff;font-weight:500}@media(max-width:960px){.hive-structure{grid-template-columns:repeat(2,minmax(130px,1fr));gap:18px}.scale-link{display:none}}@media(max-width:760px){.scene-flow{flex-direction:column;gap:7px}.scene-arrow{transform:rotate(90deg)}.search-columns{grid-template-columns:1fr}.mode-status em{display:none}.hive-structure,.structure-summary{grid-template-columns:1fr}.structure-detail{flex-wrap:wrap;gap:8px 14px}}
.resonance-view{display:grid;gap:12px;max-width:580px;margin:26px auto;padding:20px;border:1px solid rgba(115,176,255,.24);border-radius:12px;color:#cfe0f8;background:rgba(10,27,48,.5)}.resonance-view small{color:#8da7c8}.resonance-view b{color:#78e7d0}.resonance-scope{display:flex;gap:6px;padding:4px;border:1px solid rgba(115,176,255,.2);border-radius:9px;background:#091827}.resonance-scope button{flex:1;border:0;border-radius:6px;padding:9px;color:#91a8c8;background:transparent;font:600 11px system-ui;cursor:pointer}.resonance-scope button.active{color:#f0fffc;background:rgba(54,139,122,.48)}.resonance-scope button:disabled{opacity:.55;cursor:wait}
.probe-route{display:grid;gap:5px;padding:12px;border:1px solid rgba(120,231,208,.22);border-radius:9px;color:#9cb2d0;font-size:11px}.probe-route b{font-size:13px}.resonance-results{display:grid;gap:8px}.resonance-card{display:grid;gap:5px;padding:12px;border:1px solid rgba(115,176,255,.24);border-radius:9px;background:rgba(25,56,93,.45);font-size:10px}.resonance-card>div{display:flex;justify-content:space-between;align-items:center}.resonance-card strong{color:#f1f7ff;font-size:16px}.resonance-card span{color:#9cb2d0}.resonance-card button{justify-self:start;margin-top:4px;border:1px solid rgba(120,231,208,.35);border-radius:6px;padding:6px 9px;color:#dffff8;background:rgba(58,139,122,.25);font:10px system-ui;cursor:pointer}
.dynamics-view{display:grid;gap:12px;max-width:1000px;margin:18px auto}.dynamics-toolbar{display:flex;justify-content:flex-end;align-items:center;gap:7px;color:#94a9c7;font-size:10px}.dynamics-toolbar span{margin-right:auto;color:#ffc968}.dynamics-toolbar button{border:1px solid rgba(120,231,208,.25);border-radius:6px;padding:7px 9px;color:#bceee4;background:rgba(120,231,208,.08);font:10px system-ui;cursor:pointer}.dynamics-field{width:100%;min-height:390px;border:1px solid rgba(255,201,104,.18);border-radius:12px;background:radial-gradient(circle at center,rgba(255,201,104,.12),rgba(5,14,28,.7) 58%)}.zone-core,.zone-active,.zone-candidate,.zone-eviction{fill:none;stroke-width:1;stroke-dasharray:5 6}.zone-core{stroke:#78e7d0}.zone-active{stroke:#6ca2ff;stroke-opacity:.6}.zone-candidate{stroke:#ffc968;stroke-opacity:.5}.zone-eviction{stroke:#dc6b75;stroke-opacity:.55}.dynamics-anchor circle{fill:#ffc968;stroke:#fff1c2;stroke-width:2}.dynamics-anchor text{fill:#ffd98b;font:12px system-ui;text-anchor:middle}.dynamics-node{cursor:pointer}.dynamics-node circle{fill:#78e7d0;stroke:#effffb;stroke-width:2;transition:r .2s ease}.dynamics-node.WEAKENING circle{fill:#ffc968}.dynamics-node.DRIFTING_OUT circle,.dynamics-node.AT_BOUNDARY circle{fill:#e56b6f}.dynamics-node.PINNED circle{fill:#fff1bc}.dynamics-node text{fill:#dff7f1;font:12px system-ui;text-anchor:middle;pointer-events:none}.net-arrow{stroke:#fff;stroke-width:2;marker-end:url(#arrow);opacity:.8}.trajectory-line{fill:none;stroke:#78e7d0;stroke-width:2;stroke-opacity:.3}.dynamics-inspector{display:flex;flex-wrap:wrap;gap:16px;padding:12px;border:1px solid rgba(120,231,208,.25);border-radius:9px;color:#9bb0cc;font-size:10px}.dynamics-inspector strong{color:#f0f7ff}.dynamics-inspector span{color:#78e7d0}
</style>
