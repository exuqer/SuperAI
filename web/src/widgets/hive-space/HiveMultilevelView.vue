<template>
  <section class="multilevel-view">
    <header class="level-ribbon">
      <button v-for="level in levels" :key="level.id" :class="{ active: activeLevel === level.id }" @click="activeLevel = level.id; view = 'spaces'">
        <span>{{ level.label }}</span><b>{{ level.count }}</b><small>{{ level.dimensions }} измерений</small>
      </button>
    </header>

    <nav class="analysis-tabs">
      <button v-for="tab in tabs" :key="tab.id" :class="{ active: view === tab.id }" @click="view = tab.id">{{ tab.label }}</button>
      <button class="refresh" :disabled="store.multilevelLoading" @click="store.loadMultilevelViews()">{{ store.multilevelLoading ? '…' : '↻' }}</button>
    </nav>

    <div v-if="!state || !views" class="empty">Многомерная трасса появится после первого запроса.</div>

    <section v-else-if="view === 'spaces' || view === 'global'" class="space-view">
      <svg viewBox="0 0 1000 520" role="img" :aria-label="view === 'global' ? 'Глобальное поле понятий' : levelName(activeLevel)">
        <defs><filter id="cloud-glow"><feGaussianBlur stdDeviation="7" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter></defs>
        <line v-for="(edge, index) in graphEdges" :key="index" :x1="nodeX(edge.source)" :y1="nodeY(edge.source)" :x2="nodeX(edge.target)" :y2="nodeY(edge.target)" />
        <g v-for="node in graphNodes" :key="node.id" @click="selected = node">
          <ellipse :cx="node.x * 1000" :cy="node.y * 520" :rx="26 + Number(node.halo || 0) * 38" :ry="17 + Number(node.halo || 0) * 25" class="cloud-halo" />
          <circle :cx="node.x * 1000" :cy="node.y * 520" :r="10 + Number(node.density || 0) * 14" filter="url(#cloud-glow)" />
          <text :x="node.x * 1000" :y="node.y * 520 + 34">{{ node.label }}</text>
        </g>
      </svg>
      <aside><span>{{ view === 'global' ? 'ГЛОБАЛЬНОЕ ПОЛЕ' : levelName(activeLevel).toUpperCase() }}</span><b>{{ graphNodes.length }} облаков</b><small>{{ graphDimensions.join(' · ') }}</small><div><i v-for="route in scoutRoutes" :key="route.task_id">{{ route.target_space }} · {{ route.fragment }}</i></div></aside>
    </section>

    <section v-else-if="view === 'hive'" class="layer-grid">
      <article v-for="layer in memoryLayers" :key="layer.id" :class="['memory-layer', layer.id]">
        <header><span>{{ layer.label }}</span><b>{{ layer.items.length }}</b></header>
        <div class="layer-items">
          <button v-for="item in layer.items" :key="item.item_id" @click="selected = item">
            <strong>{{ item.content.text || item.content.summary || item.item_id }}</strong>
            <small>{{ item.topics.join(' · ') }}</small>
            <i :style="{ width: `${item.temperature * 100}%` }" />
            <em>T {{ percent(item.temperature) }} · A {{ percent(item.activation) }} · D {{ percent(item.depth) }}</em>
          </button>
        </div>
      </article>
    </section>

    <section v-else-if="view === 'topics'" class="topic-grid">
      <article v-for="cluster in topicClusters" :key="cluster.cluster_id" :style="{ '--heat': cluster.temperature }">
        <span>{{ cluster.state }}</span><strong>{{ cluster.topics.join(' · ') || cluster.cluster_id }}</strong>
        <small>{{ cluster.member_ids.length }} объектов · масса {{ number(cluster.mass) }}</small>
        <div class="thermometer"><i :style="{ width: `${cluster.temperature * 100}%` }" /></div>
      </article>
    </section>

    <section v-else-if="view === 'vertical_transition'" class="vertical-view">
      <div class="vertical-levels">
        <template v-for="(level, index) in vertical.levels" :key="level.id">
          <article><span>{{ levelName(level.id) }}</span><b>{{ level.object_count }}</b></article>
          <i v-if="index < vertical.levels.length - 1">↓ ↑</i>
        </template>
      </div>
      <div class="transition-list">
        <article v-for="(transition, index) in vertical.transitions" :key="index" :class="transition.direction">
          <b>{{ levelName(transition.from) }} {{ transition.direction === 'up' ? '↑' : '↓' }} {{ levelName(transition.to) }}</b>
          <span>{{ transition.reason }}</span><small>{{ transition.fragment }}</small>
        </article>
        <p v-if="!vertical.transitions.length">Нижние уровни не понадобились: ответ собирается на верхнем уровне.</p>
      </div>
    </section>

    <section v-else-if="view === 'tick_timeline'" class="tick-timeline">
      <article v-for="tick in timeline.ticks" :key="`${tick.turn}-${tick.tick}`">
        <b>Ход {{ tick.turn }} · такт {{ tick.tick }}</b>
        <span>{{ tick.active_spaces.map(levelName).join(' → ') || 'без активации' }}</span>
        <small>{{ tick.packet_count }} пакетов · бюджет {{ tick.budget.spent }}/{{ tick.budget.total }} · {{ tick.answer.mode }}</small>
      </article>
    </section>

    <section v-else-if="view === 'retention'" class="retention-view">
      <table>
        <thead><tr><th>Объект</th><th>Слой</th><th>T</th><th>A</th><th>Масса</th><th>Глубина</th><th>Retention</th><th>Eviction</th></tr></thead>
        <tbody><tr v-for="row in retention.rows" :key="row.id"><td>{{ row.id }}</td><td><span :class="['layer-pill', row.layer]">{{ row.layer }}</span></td><td>{{ percent(row.temperature) }}</td><td>{{ percent(row.activation) }}</td><td>{{ number(row.mass) }}</td><td>{{ percent(row.depth) }}</td><td>{{ percent(row.retention) }}</td><td>{{ percent(row.eviction_score) }}</td></tr></tbody>
      </table>
      <div class="event-stream"><span v-for="(event, index) in retention.events.slice(-20).reverse()" :key="index"><b>{{ event.event_type }}</b>{{ event.target_id }}<small>ход {{ event.turn }}</small></span></div>
    </section>

    <section v-else-if="view === 'explanation'" class="explanation-view">
      <header><span>РЕЖИМ ОТВЕТА</span><b>{{ explanation.answer.mode || 'unknown' }}</b><strong>{{ explanation.answer.status || 'UNVERIFIED' }}</strong></header>
      <div class="explanation-columns">
        <article><span>НАЙДЕНО</span><b>{{ explanation.retrieved.length }}</b><small>пакетов активации из памяти и пространств</small></article>
        <article><span>СОБРАНО</span><b>{{ explanation.composed?.surface || '—' }}</b><small>{{ explanation.composed?.status || 'нижний спуск не требовался' }}</small></article>
        <article><span>РЕАКТИВИРОВАНО</span><b>{{ explanation.reactivated.length }}</b><small>старых тематических объектов</small></article>
        <article><span>ОТБРОШЕНО</span><b>{{ explanation.rejected.length }}</b><small>конфликтных или слабых сцен</small></article>
      </div>
      <div class="guard">Факты с нижних уровней: <b>{{ explanation.fact_guard.lower_levels_created_fact ? 'созданы' : 'не создавались' }}</b></div>
    </section>

    <pre v-else class="json-view">{{ JSON.stringify({ state, views }, null, 2) }}</pre>

    <aside v-if="selected" class="item-inspector"><button @click="selected = null">×</button><pre>{{ JSON.stringify(selected, null, 2) }}</pre></aside>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useHiveStore } from '@/entities/hive/store';

const store = useHiveStore();
const view = ref('hive');
const activeLevel = ref('event_space');
const selected = ref<any>(null);
const state = computed(() => store.multilevel);
const views = computed<any>(() => store.multilevelViews);
const tabs = [
  { id: 'global', label: 'Глобальное поле' }, { id: 'spaces', label: 'Пространства' }, { id: 'hive', label: 'Улей' }, { id: 'topics', label: 'Острова' },
  { id: 'vertical_transition', label: 'Вертикальный переход' }, { id: 'tick_timeline', label: 'Reasoning ticks' },
  { id: 'retention', label: 'Охлаждение' }, { id: 'explanation', label: 'Объяснение' }, { id: 'json', label: 'JSON' },
];
const levelLabels: Record<string, string> = { event_space: 'События', concept_space: 'Понятия', word_space: 'Слова', morpheme_space: 'Морфемы', symbol_space: 'Буквы' };
const levels = computed(() => Object.entries(state.value?.spaces || {}).map(([id, value]: [string, any]) => ({ id, label: levelName(id), count: value.object_count, dimensions: value.dimensions.length })));
const memoryLayers = computed(() => {
  const layers = views.value?.hive?.layers || {};
  return [
    { id: 'hot', label: 'ГОРЯЧИЙ', items: layers.hot || [] }, { id: 'warm', label: 'ТЁПЛЫЙ', items: layers.warm || [] },
    { id: 'cold', label: 'ХОЛОДНЫЙ', items: layers.cold || [] }, { id: 'archive', label: 'АРХИВ', items: layers.archive || [] },
  ];
});
const topicClusters = computed<any[]>(() => views.value?.topics?.clusters || []);
const activeSpaceViewId = computed(() => ({ event_space: 'events', concept_space: 'concepts', word_space: 'words', morpheme_space: 'morphemes', symbol_space: 'symbols' } as Record<string, string>)[activeLevel.value] || 'events');
const graph = computed<any>(() => view.value === 'global' ? views.value?.global || {} : views.value?.[activeSpaceViewId.value] || {});
const graphNodes = computed<any[]>(() => graph.value.nodes || graph.value.clouds || []);
const graphEdges = computed<any[]>(() => (graph.value.edges || graph.value.relations || []).filter((edge: any) => graphNodes.value.some(node => node.id === edge.source) && graphNodes.value.some(node => node.id === edge.target)));
const graphDimensions = computed<string[]>(() => graph.value.dimensions || state.value?.spaces?.[activeLevel.value]?.dimensions || []);
const scoutRoutes = computed<any[]>(() => views.value?.global?.scout_routes || []);
const vertical = computed<any>(() => views.value?.vertical_transition || { levels: [], transitions: [] });
const timeline = computed<any>(() => views.value?.tick_timeline || { ticks: [] });
const retention = computed<any>(() => views.value?.retention || { rows: [], events: [] });
const explanation = computed<any>(() => views.value?.explanation || { answer: {}, retrieved: [], reactivated: [], rejected: [], fact_guard: {} });
function levelName(value: string) { return levelLabels[value] || value; }
function percent(value: number) { return `${Math.round(Number(value || 0) * 100)}%`; }
function number(value: number) { return Number(value || 0).toFixed(2); }
function graphNode(id: string) { return graphNodes.value.find(node => node.id === id) || { x: .5, y: .5 }; }
function nodeX(id: string) { return graphNode(id).x * 1000; }
function nodeY(id: string) { return graphNode(id).y * 520; }
</script>

<style scoped lang="scss">
.multilevel-view{position:relative;display:grid;gap:14px;padding-top:16px;color:#dceaff}.level-ribbon{display:grid;grid-template-columns:repeat(5,minmax(100px,1fr));gap:7px}.level-ribbon button{display:grid;gap:3px;border:1px solid rgba(115,176,255,.2);border-radius:9px;padding:9px;color:#8da7c8;background:rgba(10,27,48,.7);text-align:left;cursor:pointer}.level-ribbon button.active{border-color:#78e7d0;background:rgba(35,91,84,.38)}.level-ribbon b{color:#eef8ff;font-size:18px}.level-ribbon small{font-size:8px}.analysis-tabs{display:flex;gap:5px;overflow:auto}.analysis-tabs button{flex:0 0 auto;border:1px solid rgba(115,176,255,.18);border-radius:7px;padding:7px 9px;color:#8ea7c7;background:rgba(9,25,43,.66);font:9px system-ui;cursor:pointer}.analysis-tabs button.active{border-color:#ffc968;color:#ffe8b7;background:rgba(103,72,25,.36)}.analysis-tabs .refresh{margin-left:auto;font-size:14px}.empty{display:grid;place-items:center;min-height:360px;border:1px dashed rgba(115,176,255,.25);border-radius:10px;color:#7188a8}.layer-grid{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:8px;min-height:380px}.memory-layer{overflow:hidden;border:1px solid rgba(115,176,255,.18);border-radius:10px;background:rgba(6,17,31,.55)}.memory-layer>header{display:flex;justify-content:space-between;padding:10px;color:#9cb5d6;font-size:9px;letter-spacing:.1em}.memory-layer.hot>header{color:#ff9e8b}.memory-layer.warm>header{color:#ffc968}.memory-layer.cold>header{color:#78e7d0}.memory-layer.archive>header{color:#8696b3}.layer-items{display:grid;gap:6px;padding:7px}.layer-items button{display:grid;gap:5px;border:0;border-left:3px solid #6ca2ff;border-radius:6px;padding:8px;color:#a8bdd8;background:rgba(18,43,70,.55);text-align:left;cursor:pointer}.layer-items strong{overflow:hidden;color:#edf7ff;font-size:10px;text-overflow:ellipsis;white-space:nowrap}.layer-items small{overflow:hidden;font-size:8px;text-overflow:ellipsis;white-space:nowrap}.layer-items i{height:3px;border-radius:3px;background:linear-gradient(90deg,#6ca2ff,#ffc968,#e56b6f)}.layer-items em{font:8px ui-monospace,Consolas,monospace;font-style:normal}.topic-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}.topic-grid article{display:grid;gap:7px;min-height:110px;padding:13px;border:1px solid rgba(120,231,208,.2);border-radius:50% 45% 48% 42%;background:radial-gradient(circle,rgba(120,231,208,.14),rgba(11,29,48,.72));box-shadow:0 0 calc(30px * var(--heat)) rgba(120,231,208,.18)}.topic-grid span{color:#78e7d0;font-size:8px}.topic-grid strong{font-size:11px}.topic-grid small{color:#8fa7c7;font-size:9px}.thermometer{height:4px;border-radius:4px;background:#142941}.thermometer i{display:block;height:100%;background:linear-gradient(90deg,#6ca2ff,#ffc968,#e56b6f)}.vertical-view{display:grid;grid-template-columns:minmax(180px,.45fr) 1fr;gap:18px}.vertical-levels{display:grid;gap:5px}.vertical-levels article{display:flex;align-items:center;justify-content:space-between;padding:12px;border:1px solid rgba(115,176,255,.23);border-radius:9px;background:rgba(20,50,82,.48)}.vertical-levels i{text-align:center;color:#78e7d0;font-style:normal}.transition-list{display:grid;align-content:start;gap:7px}.transition-list article{display:grid;gap:4px;padding:11px;border-left:3px solid #ffc968;background:rgba(73,51,18,.36)}.transition-list article.up{border-color:#78e7d0;background:rgba(24,73,63,.36)}.transition-list span,.transition-list small{color:#92a8c6;font-size:9px}.transition-list p{padding:18px;border:1px dashed rgba(120,231,208,.25);color:#8ea8c8}.tick-timeline{display:grid;gap:7px}.tick-timeline article{display:grid;grid-template-columns:150px 1fr auto;gap:12px;padding:11px;border-left:2px solid #78e7d0;background:rgba(15,42,62,.48)}.tick-timeline span{color:#a9c5e9}.tick-timeline small{color:#8aa0bd}.retention-view{display:grid;grid-template-columns:minmax(500px,1fr) 230px;gap:12px}.retention-view table{width:100%;border-collapse:collapse;font-size:9px}.retention-view th,.retention-view td{padding:7px;border-bottom:1px solid rgba(115,176,255,.1);text-align:left}.retention-view th{color:#7791b2}.retention-view td:first-child{max-width:180px;overflow:hidden;color:#e6f1ff;text-overflow:ellipsis;white-space:nowrap}.layer-pill{padding:3px 5px;border-radius:6px;background:#283b55}.layer-pill.hot{color:#ff9e8b}.layer-pill.warm{color:#ffc968}.layer-pill.cold{color:#78e7d0}.event-stream{display:grid;align-content:start;gap:5px;max-height:430px;overflow:auto}.event-stream span{display:grid;grid-template-columns:auto 1fr auto;gap:6px;padding:6px;background:rgba(18,43,70,.48);font-size:8px}.event-stream b{color:#78e7d0}.event-stream small{color:#7186a3}.explanation-view{display:grid;gap:14px}.explanation-view>header{display:flex;align-items:center;gap:12px;padding:14px;border:1px solid rgba(120,231,208,.24);border-radius:10px}.explanation-view>header span{color:#82a5d0;font-size:9px}.explanation-view>header b{color:#78e7d0;font-size:18px}.explanation-view>header strong{margin-left:auto;color:#ffc968}.explanation-columns{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.explanation-columns article{display:grid;gap:7px;padding:13px;border:1px solid rgba(115,176,255,.18);border-radius:9px;background:rgba(16,40,68,.45)}.explanation-columns span{color:#82a5d0;font-size:8px}.explanation-columns b{color:#eef8ff;font-size:16px}.explanation-columns small{color:#8fa4c1;font-size:9px}.guard{padding:12px;border-left:3px solid #78e7d0;background:rgba(26,77,67,.34);font-size:10px}.guard b{color:#78e7d0}.json-view{max-height:520px;overflow:auto;padding:14px;border:1px solid rgba(115,176,255,.18);border-radius:9px;color:#9dd3c8;background:#06111d;font:9px/1.5 ui-monospace,Consolas,monospace}.item-inspector{position:absolute;inset:82px 10px 10px auto;width:min(380px,80%);z-index:3;overflow:auto;padding:14px;border:1px solid #78e7d0;border-radius:9px;background:#071523;box-shadow:0 15px 45px #000}.item-inspector button{float:right;border:0;color:#fff;background:none;cursor:pointer}.item-inspector pre{color:#9dd3c8;font:9px/1.5 ui-monospace,Consolas,monospace;white-space:pre-wrap}@media(max-width:900px){.level-ribbon{grid-template-columns:repeat(3,1fr)}.layer-grid,.explanation-columns{grid-template-columns:repeat(2,1fr)}.retention-view,.vertical-view{grid-template-columns:1fr}}@media(max-width:620px){.level-ribbon,.layer-grid,.explanation-columns{grid-template-columns:1fr}.tick-timeline article{grid-template-columns:1fr}.retention-view{overflow:auto}}
.space-view{display:grid;grid-template-columns:minmax(420px,1fr) 210px;min-height:420px;border:1px solid rgba(115,176,255,.18);border-radius:10px;overflow:hidden;background:radial-gradient(circle at 45% 50%,rgba(44,113,141,.18),transparent 55%),#06111d}.space-view svg{width:100%;height:100%;min-height:420px}.space-view line{stroke:#6ca2ff;stroke-width:1;stroke-opacity:.35}.space-view g{cursor:pointer}.space-view .cloud-halo{fill:#78e7d0;fill-opacity:.1}.space-view circle{fill:#4d9fc7;stroke:#dffff8;stroke-width:1.5}.space-view text{fill:#e9f6ff;font:10px system-ui;text-anchor:middle;pointer-events:none}.space-view aside{display:grid;align-content:start;gap:8px;padding:14px;border-left:1px solid rgba(115,176,255,.16);background:rgba(7,20,35,.72)}.space-view aside>span{color:#78e7d0;font-size:9px;letter-spacing:.1em}.space-view aside>b{color:#eff8ff;font-size:20px}.space-view aside>small{color:#8098b8;font-size:8px;line-height:1.5}.space-view aside div{display:grid;gap:5px;margin-top:8px}.space-view aside i{padding:6px;border-left:2px solid #ffc968;color:#b9cae0;background:rgba(80,57,22,.25);font-size:8px;font-style:normal}@media(max-width:760px){.space-view{grid-template-columns:1fr}.space-view aside{border-top:1px solid rgba(115,176,255,.16);border-left:0}}
</style>
