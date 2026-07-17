<template>
  <section class="whole-hive">
    <header class="whole-toolbar">
      <div class="aggregation" aria-label="Агрегация слов">
        <button :class="{ active: aggregation === 'lexeme' }" @click="setAggregation('lexeme')">Леммы</button>
        <button :class="{ active: aggregation === 'word_form' }" @click="setAggregation('word_form')">Формы слов</button>
      </div>
      <label><input v-model="showScenes" type="checkbox"> Сцены</label>
      <label><input v-model="showQuery" type="checkbox"> Запрос</label>
      <label><input v-model="showZones" type="checkbox"> Зоны</label>
      <button class="refresh" :disabled="hiveStore.snapshotLoading" @click="refresh()">{{ hiveStore.snapshotLoading ? 'Загрузка…' : 'Обновить' }}</button>
    </header>

    <div v-if="!snapshot" class="whole-empty">Статическая проекция улья загружается…</div>
    <template v-else>
      <div class="whole-summary">
        <span>Сцены <b>{{ snapshot.summary.scene_count }}</b></span><span>Слова <b>{{ snapshot.summary.word_count }}</b></span>
        <span>Активно <b>{{ snapshot.summary.active_word_count }}</b></span><span>Ёмкость <b>{{ snapshot.hive.occupied_cells }} / {{ snapshot.hive.capacity }}</b></span><span>Ячейки <b>{{ snapshot.diagnostics.counts.working_cells_total }} / {{ snapshot.diagnostics.counts.projected_cells_total }}</b></span>
        <span>Энергия <b>{{ Math.round(snapshot.hive.energy * 100) }}%</b></span><span>Резонанс <b>{{ resonanceLabel }}</b></span>
      </div>
      <div class="whole-layout">
        <aside class="scene-list">
          <button v-for="scene in snapshot.scenes" :key="scene.id" :class="{ selected: selected?.id === scene.id }" @click="selected = scene">
            <strong>{{ scene.text }}</strong><small>{{ scene.source }} · {{ scene.id }}</small><span>Акт. {{ scene.physics.local_activation.toFixed(2) }} · Грав. {{ scene.physics.local_gravity.toFixed(2) }}</span>
          </button>
        </aside>
        <div class="canvas-wrap" @wheel.prevent="zoomAt">
          <svg class="whole-canvas" viewBox="0 0 1000 620" role="img" aria-label="Полное поле памяти улья" @pointerdown="startPan" @pointermove="pan" @pointerup="stopPan" @pointerleave="stopPan">
            <defs>
              <filter id="soft"><feGaussianBlur stdDeviation="18" /></filter>
              <filter id="glow"><feGaussianBlur stdDeviation="5" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
            </defs>
            <g :transform="transform">
              <g v-if="showZones" class="zones"><circle :cx="center.x" :cy="center.y" r="80" /><circle :cx="center.x" :cy="center.y" r="170" /><circle :cx="center.x" :cy="center.y" r="275" /></g>
              <g v-if="showScenes" class="scene-regions">
                <ellipse v-for="scene in snapshot.scenes" :key="scene.id" :cx="x(scene.position.x)" :cy="y(scene.position.y)" :rx="sceneRadius(scene)" :ry="sceneRadius(scene) * .56" :class="{ related: isRelated(scene) }" :style="{ opacity: .09 + scene.physics.retention * .13 }" />
              </g>
              <g v-if="selectedWord" class="influence-lines"><line v-for="scene in contributedScenes" :key="scene.id" :x1="x(selectedWord.position.render_x)" :y1="y(selectedWord.position.render_y)" :x2="x(scene.position.x)" :y2="y(scene.position.y)" /></g>
              <g v-if="showQuery && snapshot.query_overlay?.anchors?.length" class="query-overlay"><circle :cx="center.x" :cy="center.y" r="235" /><text :x="center.x" :y="center.y - 248">ЗАПРОС · {{ snapshot.query_overlay.reconstructed_text || snapshot.query_overlay.source_text }}</text></g>
              <g v-for="cell in projectedCells" :key="`cell-${cell.id}`" class="projected-cell" :style="{ opacity: .25 + cell.physics.retention * .65 }">
                <circle :cx="x(cell.position.x)" :cy="y(cell.position.y)" :r="8 + cell.physics.local_activation * 14" :style="{ fill: cellColor(cell) }" />
                <text :x="x(cell.position.x)" :y="y(cell.position.y) - 16">{{ cell.label }}</text>
              </g>
              <g v-for="word in snapshot.words" :key="word.id" class="word-node" :class="{ selected: selected?.id === word.id, related: isRelatedWord(word) }" :style="{ opacity: .24 + word.local.retention * .76 }" @click.stop="selected = word">
                <circle class="gravity" :cx="x(word.position.render_x)" :cy="y(word.position.render_y)" :r="halo(word)" :style="{ opacity: .07 + word.local.gravity * .15 }" />
                <circle class="energy" :cx="x(word.position.render_x)" :cy="y(word.position.render_y)" :r="core(word) + 8 + word.local.energy * 8" :style="{ animationDuration: `${1.8 - word.local.energy * 1.1}s` }" />
                <circle class="core" :cx="x(word.position.render_x)" :cy="y(word.position.render_y)" :r="core(word)" :style="{ fill: color(word), strokeWidth: 1.5 + word.local.stored_strength * 6 }" filter="url(#glow)" />
                <text :x="x(word.position.render_x)" :y="y(word.position.render_y) + core(word) + 15">{{ word.lemma }}</text>
              </g>
            </g>
          </svg>
          <div class="legend"><span class="core-key"></span>масса <span class="halo-key"></span>гравитация <span class="ring-key"></span>сила хранения <span class="pulse-key"></span>энергия</div>
        </div>
        <aside class="inspector">
          <template v-if="selectedWord">
            <div class="inspector-head"><small>СЛОВО · {{ selectedWord.roles.join(', ') }}</small><strong>{{ selectedWord.lemma }}</strong></div>
            <div class="metrics"><span>Масса <b>{{ selectedWord.global.mass.toFixed(2) }}</b></span><span>Гравитация <b>{{ selectedWord.local.gravity.toFixed(2) }}</b></span><span>Активация <b>{{ selectedWord.local.activation.toFixed(2) }}</b></span><span>Сила <b>{{ selectedWord.local.stored_strength.toFixed(2) }}</b></span><span>Удержание <b>{{ selectedWord.local.retention.toFixed(2) }}</b></span><span>Энергия <b>{{ selectedWord.local.energy.toFixed(2) }}</b></span></div>
            <div class="contributions"><small>ИСТОЧНИКИ · {{ selectedWord.scene_support_count }}</small><button v-for="contribution in selectedWord.contributions" :key="`${contribution.scene_id}-${contribution.role}`" @click="selected = sceneById(contribution.scene_id)">{{ contribution.surface }} · {{ contribution.role }}<b>{{ contribution.scene_id }}</b></button></div>
          </template>
          <template v-else-if="selectedScene">
            <div class="inspector-head"><small>СЦЕНА ПАМЯТИ</small><strong>{{ selectedScene.text }}</strong></div>
            <div class="metrics"><span>Активация <b>{{ selectedScene.physics.local_activation.toFixed(2) }}</b></span><span>Гравитация <b>{{ selectedScene.physics.local_gravity.toFixed(2) }}</b></span><span>Статус <b>{{ selectedScene.status.candidate_status }}</b></span></div>
            <div class="contributions"><small>РОЛИ</small><span v-for="(value, role) in selectedScene.roles" :key="role">{{ role }} <b>{{ value.surface || value.lemma }}</b></span></div>
          </template>
          <template v-else><div class="inspector-head"><small>ВЕСЬ УЛЕЙ</small><strong>Выберите слово или сцену</strong></div><p>Размер ядра — глобальная масса, ореол — локальная гравитация, яркость — активация.</p></template>
          <button v-if="selected" class="close" @click="selected = null">Снять выбор</button>
          <p v-for="warning in snapshot.diagnostics.warnings" :key="warning.code" class="warning">{{ warning.message }}</p>
        </aside>
      </div>
      <footer v-if="snapshot.timeline.length" class="timeline"><button v-for="item in snapshot.timeline" :key="item.step" :class="{ active: timelineStep === item.step }" @click="timelineStep = item.step; refresh(String(item.step))">Шаг {{ item.step }}</button></footer>
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useHiveStore } from '@/entities/hive/store';
import type { HiveSnapshotCellV2, HiveSnapshotSceneV2, HiveSnapshotWordV2 } from '@/entities/hive/types';

const hiveStore = useHiveStore();
const aggregation = ref<'lexeme' | 'word_form'>('lexeme');
const showScenes = ref(true);
const showQuery = ref(true);
const showZones = ref(false);
const selected = ref<HiveSnapshotWordV2 | HiveSnapshotSceneV2 | null>(null);
const timelineStep = ref<number | null>(null);
const camera = ref({ x: 0, y: 0, zoom: 1 });
const pointer = ref({ active: false, x: 0, y: 0 });
const snapshot = computed(() => hiveStore.hiveSnapshot);
const center = computed(() => ({ x: 500, y: 310 }));
const transform = computed(() => `translate(${camera.value.x} ${camera.value.y}) scale(${camera.value.zoom}) translate(${500 - 500 / camera.value.zoom} ${310 - 310 / camera.value.zoom})`);
const selectedWord = computed(() => selected.value && 'contributions' in selected.value ? selected.value : null);
const selectedScene = computed(() => selected.value && 'physics' in selected.value ? selected.value : null);
const projectedCells = computed(() => snapshot.value?.cells || [] as HiveSnapshotCellV2[]);
const resonanceLabel = computed(() => snapshot.value?.summary.resonance_status === 'IDLE' ? 'не запущен' : snapshot.value?.summary.resonance_status || '—');
const x = (value: number) => value * 1000;
const y = (value: number) => value * 620;
const core = (word: HiveSnapshotWordV2) => Math.max(10, Math.min(38, 10 + Math.sqrt(Math.max(0, word.global.mass)) * 9));
const halo = (word: HiveSnapshotWordV2) => core(word) + 16 + word.local.gravity * 54;
const color = (word: HiveSnapshotWordV2) => `hsl(${190 - word.local.activation * 42} 75% ${42 + word.local.activation * 20}%)`;
const cellColor = (cell: HiveSnapshotCellV2) => cell.component_class === 'semantic_bridge' ? '#b891ff' : cell.component_class === 'role_candidate' ? '#ffc968' : '#59b8d8';
const sceneById = (id: string) => snapshot.value?.scenes.find(scene => scene.id === id) || null;
const contributedScenes = computed(() => selectedWord.value?.contributions.map(item => sceneById(item.scene_id)).filter(Boolean) as HiveSnapshotSceneV2[] || []);
const isRelated = (scene: HiveSnapshotSceneV2) => Boolean(selectedWord.value?.contributions.some(item => item.scene_id === scene.id) || selectedScene.value?.id === scene.id);
const isRelatedWord = (word: HiveSnapshotWordV2) => Boolean(selectedScene.value && word.contributions.some(item => item.scene_id === selectedScene.value?.id));
const sceneRadius = (scene: HiveSnapshotSceneV2) => 62 + scene.physics.local_gravity * 58;
async function setAggregation(next: 'lexeme' | 'word_form') { aggregation.value = next; selected.value = null; await hiveStore.loadSnapshot(next); }
async function refresh(step = 'current') { await hiveStore.loadSnapshot(aggregation.value); if (step !== 'current') timelineStep.value = Number(step); }
function zoomAt(event: WheelEvent) { camera.value.zoom = Math.max(.55, Math.min(2.6, camera.value.zoom * Math.exp(-event.deltaY * .0015))); }
function startPan(event: PointerEvent) { if ((event.target as Element).closest('.word-node')) return; pointer.value = { active: true, x: event.clientX, y: event.clientY }; }
function pan(event: PointerEvent) { if (!pointer.value.active) return; const svg = event.currentTarget as SVGSVGElement; const box = svg.getBoundingClientRect(); camera.value.x += (event.clientX - pointer.value.x) * 1000 / box.width; camera.value.y += (event.clientY - pointer.value.y) * 620 / box.height; pointer.value.x = event.clientX; pointer.value.y = event.clientY; }
function stopPan() { pointer.value.active = false; }
</script>

<style scoped lang="scss">
.whole-hive{display:grid;gap:10px;min-height:540px}.whole-toolbar,.whole-summary,.legend,.timeline{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.whole-toolbar{justify-content:space-between}.whole-toolbar label,.whole-summary span{color:#91a8c8;font-size:10px}.aggregation{display:flex;padding:3px;border:1px solid rgba(115,176,255,.2);border-radius:8px}.aggregation button,.refresh,.timeline button,.close{border:0;border-radius:6px;padding:7px 9px;color:#9cb9dc;background:transparent;font:10px system-ui;cursor:pointer}.aggregation button.active,.refresh,.timeline button.active{color:#e8fffa;background:rgba(71,155,137,.42)}.whole-summary{padding:9px 12px;border:1px solid rgba(120,231,208,.14);border-radius:9px;background:rgba(11,34,53,.42)}.whole-summary b{margin-left:3px;color:#e9f7ff}.whole-layout{display:grid;grid-template-columns:180px minmax(360px,1fr) 205px;min-height:455px;border:1px solid rgba(115,176,255,.17);border-radius:12px;overflow:hidden;background:#071523}.scene-list,.inspector{overflow:auto;padding:10px;background:rgba(6,17,31,.72)}.scene-list{border-right:1px solid rgba(115,176,255,.14)}.scene-list button{display:grid;gap:5px;width:100%;margin-bottom:6px;border:1px solid rgba(115,176,255,.15);border-left:3px solid #598fe1;border-radius:7px;padding:8px;color:#9bb4d4;background:rgba(20,48,79,.36);font:9px system-ui;text-align:left;cursor:pointer}.scene-list button.selected{border-left-color:#78e7d0;background:rgba(35,98,89,.4)}.scene-list strong{color:#e9f4ff;font-size:10px}.scene-list small{color:#7189ab}.canvas-wrap{position:relative;min-width:0;background:radial-gradient(circle at 50% 50%,rgba(44,113,141,.18),transparent 58%),#06111d}.whole-canvas{width:100%;height:100%;min-height:455px;touch-action:none}.zones circle{fill:none;stroke:#6ca2ff;stroke-dasharray:4 8;opacity:.25}.scene-regions ellipse{fill:#7a62bc;filter:url(#soft);transition:opacity .2s}.scene-regions ellipse.related{fill:#78e7d0;opacity:.35!important}.influence-lines line{stroke:#78e7d0;stroke-width:2;stroke-opacity:.65}.query-overlay circle{fill:none;stroke:#ffc968;stroke-width:1;stroke-dasharray:6 7;opacity:.45}.query-overlay text{fill:#ffd98b;font:10px system-ui;text-anchor:middle}.word-node{cursor:pointer}.word-node.related .core,.word-node.selected .core{stroke:#fff4ca}.gravity{fill:#65d6e0}.energy{fill:none;stroke:#a2fff0;stroke-width:1;opacity:.22;transform-origin:center;animation:pulse 1.2s ease-out infinite}.core{stroke:#e8fff9}.word-node text{fill:#eaf5ff;font:11px system-ui;text-anchor:middle;pointer-events:none}.legend{position:absolute;bottom:7px;left:12px;color:#90a9ca;font-size:9px}.legend i,.core-key,.halo-key,.ring-key,.pulse-key{display:inline-block;width:9px;height:9px;border-radius:50%;background:#6cd8e1}.halo-key{background:rgba(108,216,225,.22);box-shadow:0 0 0 4px rgba(108,216,225,.18)}.ring-key{border:2px solid #ffc968;background:transparent}.pulse-key{border:1px solid #b0fff2;background:transparent}.inspector{border-left:1px solid rgba(115,176,255,.14);color:#9bb3ce;font-size:10px}.inspector-head{display:grid;gap:5px;padding-bottom:10px;border-bottom:1px solid rgba(115,176,255,.14)}.inspector-head small,.contributions small{color:#81a8d8;letter-spacing:.1em}.inspector-head strong{color:#eff8ff;font-size:16px}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin:12px 0}.metrics span,.contributions span{display:flex;justify-content:space-between;gap:7px}.metrics b,.contributions b{color:#78e7d0}.contributions{display:grid;gap:7px}.contributions button{display:grid;grid-template-columns:1fr auto;gap:5px;border:0;border-left:2px solid #5d99dc;padding:6px;color:#bcd1e8;background:rgba(20,52,79,.42);font:9px system-ui;text-align:left;cursor:pointer}.close{margin-top:12px;border:1px solid rgba(120,231,208,.25)}.warning{padding:7px;border-left:2px solid #ffc968;color:#d7bd83;background:rgba(109,78,30,.22);font-size:9px}.timeline{justify-content:center}.whole-empty{display:grid;place-items:center;min-height:420px;color:#91a8c8;border:1px dashed rgba(115,176,255,.28);border-radius:10px}@keyframes pulse{to{transform:scale(1.38);opacity:0}}@media(max-width:900px){.whole-layout{grid-template-columns:150px minmax(280px,1fr)}.inspector{grid-column:1/-1;border-top:1px solid rgba(115,176,255,.14);border-left:0}.whole-canvas{min-height:390px}}@media(max-width:620px){.whole-layout{grid-template-columns:1fr}.scene-list{max-height:150px;border-right:0;border-bottom:1px solid rgba(115,176,255,.14)}}
.projected-cell{pointer-events:none}.projected-cell text{fill:#9fcee4;font:8px system-ui;text-anchor:middle}
</style>
