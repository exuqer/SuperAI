<template>
  <section class="panel field-panel">
    <div class="panel-head">
      <div>
        <div class="kicker">ROUTING DECISION</div>
        <h2>{{ modeLabel }}</h2>
      </div>
      <span class="run-state" :class="mode">{{ decision?.decision || 'ожидание' }}</span>
    </div>
    <div class="goal-strip">
      <div class="goal-icon">⌘</div>
      <div>
        <span class="tiny-label">Текущий запрос</span>
        <strong>{{ goalText || 'Отправьте сообщение для активации контекста' }}</strong>
      </div>
      <div v-if="metrics" class="goal-metrics">
        <span>пчёлы <b>{{ metrics.bees }}</b></span>
        <span>итерации <b>{{ metrics.iterations }}</b></span>
      </div>
    </div>
    <div class="field-wrap">
      <svg class="field-svg" viewBox="0 0 1000 700" role="img" :aria-label="modeLabel">
        <defs>
          <radialGradient id="fieldGlow">
            <stop stop-color="#78e7d0" stop-opacity=".24" />
            <stop offset="1" stop-color="#78e7d0" stop-opacity="0" />
          </radialGradient>
          <filter id="soft">
            <feGaussianBlur stdDeviation="6" />
          </filter>
        </defs>
        <circle class="field-ring" cx="500" cy="350" r="245" />
        <circle class="field-ring ring-2" cx="500" cy="350" r="165" />
        <circle class="field-ring ring-3" cx="500" cy="350" r="82" />
        <g v-if="hasSwarmMap" class="swarm-map">
          <line
            v-for="source in sources"
            :key="`flight-${source.id}`"
            class="flight-line"
            x1="500"
            y1="350"
            :x2="sourcePoint(source).x"
            :y2="sourcePoint(source).y"
          />
          <circle
            v-for="source in sources"
            :key="`halo-${source.id}`"
            class="source-halo"
            :cx="sourcePoint(source).x"
            :cy="sourcePoint(source).y"
            :r="22 + source.fitness * 40"
          />
          <g
            v-for="source in sources"
            :key="source.id"
            class="source"
            :transform="`translate(${sourcePoint(source).x} ${sourcePoint(source).y})`"
          >
            <circle :r="8 + source.fitness * 14" />
            <text y="-26" text-anchor="middle">{{ source.label }}</text>
          </g>
          <text
            v-for="(bee, index) in bees"
            :key="bee.id"
            class="bee"
            :x="470 + (index % 5) * 18"
            :y="325 + Math.floor(index / 5) * 18"
          >✦</text>
        </g>
        <g v-else-if="hiveStore.cells.length" class="resonance-map">
          <line
            v-for="node in hiveGraphNodes"
            :key="`line-${node.cell.id}`"
            x1="500"
            y1="350"
            :x2="node.x"
            :y2="node.y"
            :class="{ active: activeCellIds.has(node.cell.id) }"
          />
          <g
            v-for="node in hiveGraphNodes"
            :key="node.cell.id"
            :class="['resonance-node', { active: activeCellIds.has(node.cell.id) }]"
            :transform="`translate(${node.x} ${node.y})`"
            @click="hiveStore.selectedCell = node.cell"
          >
            <circle :r="node.radius + 11" class="node-halo" />
            <circle :r="node.radius" class="node-core" />
            <text text-anchor="middle" y="4">{{ node.label }}</text>
<text text-anchor="middle" :y="node.radius + 17" class="node-value">{{ formatPercent(node.cell.retention) }}</text>
          </g>
        </g>
        <g class="field-center">
          <circle cx="500" cy="350" r="34" fill="url(#fieldGlow)" />
          <circle cx="500" cy="350" r="22" />
          <text x="500" y="347" text-anchor="middle">{{ hasSwarmMap ? 'РОЙ' : 'УЛЕЙ' }}</text>
          <text x="500" y="361" text-anchor="middle">{{ hasSwarmMap ? 'поиск' : 'резонанс' }}</text>
        </g>
      </svg>
      <div class="field-legend">
        <span v-if="hasSwarmMap"><i class="dot scout"></i>ограниченный рой</span>
        <span v-else><i class="dot local"></i>локальная активация</span>
        <span><i class="dot nectar"></i>сильный контекст</span>
      </div>
      <div v-if="decision" class="field-caption">
        {{ decision.external_search_required
          ? 'Показан только поиск недостающих компонентов.'
          : 'Внешний поиск пропущен: улей использовал сохранённый контекст.' }}
      </div>
    </div>
    <div class="event-log">
      <div class="log-title">Лента решения</div>
      <div v-for="event in eventLog" :key="event.id" class="event">
        <span class="event-dot" :class="event.kind"></span>
        <span>{{ event.text }}</span>
        <time>{{ event.time }}</time>
      </div>
      <div v-if="!eventLog.length" class="empty-log">События появятся после сообщения</div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useHiveStore } from '@/entities/hive/store';
import { formatTime } from '@/shared/utils/time';
import { formatPercent } from '@/shared/utils/formatters';

const hiveStore = useHiveStore();

const mode = computed(() =>
  !hiveStore.decision ? 'idle'
    : !hiveStore.decision.external_search_required ? 'local'
    : hiveStore.decision.decision === 'PARTIAL_HIT' ? 'partial'
    : 'external'
);

const modeLabel = computed(() =>
  mode.value === 'local' ? 'Локальный резонанс'
    : mode.value === 'partial' ? 'Частичный поиск'
    : mode.value === 'external' ? 'Внешний поиск'
    : 'Улей готов'
);

const hasSwarmMap = computed(() =>
  hiveStore.decision?.external_search_required && hiveStore.externalSearch?.sources?.length
);

const activeCellIds = computed(() =>
  new Set(((hiveStore.decision as any)?.matches || []).map((m: any) => m.cell_id))
);

const sources = computed(() => hiveStore.externalSearch?.sources || []);
const bees = computed(() => hiveStore.externalSearch?.bees || []);
const metrics = computed(() => hiveStore.metrics);
const goalText = computed(() => hiveStore.goalText);
const decision = computed(() => hiveStore.decision);
const eventLog = computed(() => {
  if (!hiveStore.decision) return [];
  const now = formatTime(new Date().toISOString());
  const events = [
    { id: 'route', kind: 'goal', text: `Маршрутизатор: ${hiveStore.decision.decision}`, time: now },
  ];
  if (hiveStore.decision.external_search_required) {
    events.push({
      id: 'search',
      kind: 'bee',
      text: `Рой получил ${hiveStore.metrics?.bees || 0} пчёл для недостающего контекста`,
      time: now,
    });
  } else {
    events.push({
      id: 'local',
      kind: 'done',
      text: `Активировано ${hiveStore.metrics?.activated_cells || 0} ячеек без внешнего поиска`,
      time: now,
    });
  }
  if ((hiveStore.metrics?.merged_cells || 0) > 0) {
    events.push({
      id: 'merge',
      kind: 'nectar',
      text: 'Новый нектар объединён с локальной памятью',
      time: now,
    });
  }
  return events;
});

const hiveGraphNodes = computed(() =>
  hiveStore.cells.map((cell: any, index: number) => ({
    cell,
    x: 105 + ((cell.x || index * 73) % 650),
    y: 90 + ((cell.y || index * 107) % 330),
    radius: 24 + cell.retention * 24,
    label: cell.label.slice(0, 14),
  }))
);

function sourcePoint(source: any) {
  return {
    x: 105 + (Math.abs(source.x || 0) % 790),
    y: 95 + (Math.abs(source.y || 0) % 510),
  };
}
</script>

<style scoped lang="scss">
.field-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
  border: 1px solid rgba(162, 189, 225, 0.15);
  border-radius: 14px;
  background: rgba(14, 28, 48, 0.76);
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.16);
}

.panel-head {
  display: flex;
  justify-content: space-between;
  padding: 20px 20px 14px;
  border-bottom: 1px solid rgba(162, 189, 225, 0.1);
}

.kicker {
  color: #73b0ff;
  font-size: 10px;
  letter-spacing: 0.13em;
}

.panel h2 {
  margin: 2px 0 0;
  font-size: 18px;
}

.run-state {
  color: #798daa;
  font-size: 11px;
}

.run-state i {
  display: inline-block;
  width: 7px;
  height: 7px;
  margin-right: 6px;
  border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 10px currentColor;
}

.goal-strip {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 16px 18px 0;
  padding: 10px 12px;
  border: 1px solid rgba(115, 176, 255, 0.17);
  border-radius: 10px;
  background: rgba(35, 63, 100, 0.3);
}

.goal-icon {
  display: grid;
  place-items: center;
  width: 29px;
  height: 29px;
  border-radius: 8px;
  color: #ffc968;
  background: rgba(255, 201, 104, 0.12);
  font-size: 22px;
}

.goal-strip strong {
  display: block;
  max-width: 310px;
  overflow: hidden;
  color: #dae7f8;
  font-size: 12px;
  font-weight: 500;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.goal-metrics {
  display: flex;
  gap: 14px;
  margin-left: auto;
  color: #7689a8;
  font-size: 10px;
}

.goal-metrics b {
  color: #d9e7fa;
  font-weight: 500;
}

.field-wrap {
  position: relative;
  flex: 1;
  min-height: 260px;
}

.field-svg {
  width: 100%;
  height: 100%;
  display: block;
}

.field-ring {
  fill: none;
  stroke: rgba(105, 152, 214, 0.22);
  stroke-width: 1;
}

.ring-2 {
  stroke-dasharray: 3 7;
}

.ring-3 {
  stroke: rgba(126, 233, 208, 0.35);
}

.flight-line {
  stroke: #73b0ff;
  stroke-width: 1.5;
  opacity: 0.45;
  stroke-dasharray: 5 6;
}

.source-halo {
  fill: rgba(120, 231, 208, 0.08);
  stroke: rgba(120, 231, 208, 0.2);
}

.source circle {
  fill: #ffc968;
  stroke: #ffe2a0;
  stroke-width: 1.5;
  filter: drop-shadow(0 0 5px #ffc968);
}

.source text {
  fill: #b9c9dd;
  font: 10px system-ui;
}

.bee {
  fill: #78e7d0;
  font-size: 16px;
}

.resonance-map line {
  stroke: #78e7d0;
  stroke-width: 1.5;
  opacity: 0.19;
}

.resonance-map line.active {
  stroke: #ffc968;
  opacity: 0.75;
}

.resonance-node {
  cursor: pointer;
}

.resonance-node .node-halo {
  fill: rgba(120, 231, 208, 0.08);
  stroke: rgba(120, 231, 208, 0.2);
}

.resonance-node .node-core {
  fill: rgba(120, 231, 208, 0.17);
  stroke: #78e7d0;
  stroke-width: 1.5;
}

.resonance-node.active .node-core {
  fill: rgba(255, 201, 104, 0.2);
  stroke: #ffc968;
  filter: drop-shadow(0 0 10px rgba(255, 201, 104, 0.6));
}

.resonance-node text {
  fill: #d9f6ef;
  font: 10px system-ui;
}

.resonance-node .node-value {
  fill: #ffc968;
  font-size: 9px;
}

.field-center circle {
  fill: rgba(126, 233, 208, 0.16);
  stroke: #78e7d0;
  stroke-width: 1;
}

.field-center text {
  fill: #98eadc;
  font: 8px system-ui;
  letter-spacing: 0.1em;
}

.field-legend {
  position: absolute;
  left: 18px;
  bottom: 14px;
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  color: #8296b4;
  font-size: 10px;
}

.dot {
  display: inline-block;
  width: 7px;
  height: 7px;
  margin-right: 4px;
  border-radius: 50%;
}

.dot.scout,
.dot.local {
  background: #78e7d0;
}

.dot.nectar {
  background: #ffc968;
}

.field-caption {
  position: absolute;
  right: 18px;
  bottom: 14px;
  max-width: 350px;
  color: #93b7b0;
  font-size: 10px;
  text-align: right;
}

.event-log {
  flex: 0 0 auto;
  padding: 12px 18px 14px;
  border-top: 1px solid rgba(162, 189, 225, 0.1);
}

.log-title {
  margin-bottom: 9px;
  color: #7e95b6;
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.event {
  display: flex;
  gap: 7px;
  margin: 7px 0;
  color: #aabbd2;
  font-size: 10px;
}

.event time {
  margin-left: auto;
  color: #5d7190;
  font-size: 9px;
}

.event-dot {
  width: 5px;
  height: 5px;
  flex: 0 0 auto;
  border-radius: 50%;
  background: #7087a5;
}

.event-dot.bee {
  background: #73b0ff;
}

.event-dot.nectar {
  background: #ffc968;
}

.event-dot.goal,
.event-dot.done {
  background: #78e7d0;
}

.empty-log {
  color: #607590;
  font-size: 10px;
}
</style>