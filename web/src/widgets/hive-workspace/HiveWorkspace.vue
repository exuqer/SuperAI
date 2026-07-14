<template>
  <aside class="panel hive-panel">
    <div class="panel-head">
      <div>
        <div class="kicker">РАБОЧАЯ ПАМЯТЬ</div>
        <h2>Улей</h2>
      </div>
      <span class="hive-count">{{ hiveStore.cells.length }} / {{ hiveStore.hive?.max_cells || 24 }} ячеек</span>
    </div>
    <div class="hive-summary">
      <div>
        <span class="tiny-label">Активация</span>
        <strong>{{ formatPercent(hiveStore.averageActivation) }}</strong>
      </div>
      <div>
        <span class="tiny-label">Удержание</span>
        <strong>{{ formatPercent(hiveStore.averageRetention) }}</strong>
      </div>
    </div>
    <section class="reasoning-controls">
      <div class="reasoning-head">
        <span class="tiny-label">ВИБРАЦИЯ УЛЬЯ</span>
        <span>{{ reasoningStatus }}</span>
      </div>
      <label>Шаги <input v-model.number="hiveStore.reasoningSteps" type="number" min="0" max="32" /></label>
      <div class="reasoning-actions">
        <button class="secondary" :disabled="hiveStore.reasoningLoading || !hiveStore.hive" @click="hiveStore.runReasoningStep">Один шаг</button>
        <button class="primary" :disabled="hiveStore.reasoningLoading || !hiveStore.hive" @click="hiveStore.runReasoning">{{ hiveStore.reasoningLoading ? 'Вибрация…' : 'Запустить' }}</button>
        <button class="secondary" disabled>Стоп</button>
      </div>
      <div v-if="hiveStore.runResult" class="reasoning-meta">run {{ hiveStore.runResult.run.id }} · {{ hiveStore.runResult.completed_steps }} шагов · {{ hiveStore.runResult.stop_reason }}</div>
      <div class="json-actions">
        <button class="secondary" @click="hiveStore.openJson('current')">Показать JSON</button>
        <button class="secondary" @click="hiveStore.copyJson('current')">{{ hiveStore.copyStatus || 'Копировать JSON' }}</button>
      </div>
    </section>
    <div class="hive-map">
      <svg v-if="hiveGraphNodes.length" class="hive-graph" viewBox="0 0 850 500">
        <line
          v-for="link in hiveGraphLinks"
          :key="link.id"
          :x1="link.left.x"
          :y1="link.left.y"
          :x2="link.right.x"
          :y2="link.right.y"
          :stroke-width="1 + link.similarity * 7"
        />
        <g
          v-for="node in hiveGraphNodes"
          :key="node.cell.id"
          class="hive-node"
          :class="{ selected: hiveStore.selectedCell?.id === node.cell.id, active: hiveStore.activeCellIds.has(node.cell.id) }"
          :transform="`translate(${node.x} ${node.y})`"
          @click="hiveStore.selectedCell = node.cell"
        >
          <circle :r="node.radius" />
          <circle class="node-core" :r="Math.max(10, node.radius - 10)" />
          <text class="node-label" text-anchor="middle" y="-2">{{ node.label }}</text>
          <text class="node-gravity" text-anchor="middle" :y="node.radius + 15">{{ formatPercent(node.cell.retention) }}</text>
        </g>
      </svg>
      <div v-else class="hive-empty">Улей ожидает нектар</div>
    </div>
    <div class="hive-legend">
      <span><i></i> ячейка памяти</span>
      <span><b></b> общий состав</span>
    </div>
    <section v-if="hiveStore.selectedCell" class="hive-inspector">
      <div class="inspector-head">
        <div>
          <span class="tiny-label">Выбранная ячейка</span>
          <strong>{{ hiveStore.selectedCell.label }}</strong>
        </div>
        <button class="ghost" @click="hiveStore.selectedCell = null">×</button>
      </div>
      <div class="node-stats">
        <span>активация <b>{{ formatPercent(hiveStore.selectedCell.local_activation) }}</b></span>
        <span>удержание <b>{{ formatPercent(hiveStore.selectedCell.retention) }}</b></span>
      </div>
      <div class="component-list">
        <span
          v-for="component in hiveStore.selectedCell.components || []"
          :key="component.id || component.cloud_id"
          class="component"
        >
          <span>{{ component.canonical_name }}</span>
          <b>{{ formatPercent(component.composition_share) }}</b>
        </span>
      </div>
    </section>
  </aside>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useHiveStore } from '@/entities/hive/store';
import { formatPercent } from '@/shared/utils/formatters';

const hiveStore = useHiveStore();

const hiveGraphNodes = computed(() =>
  hiveStore.cells.map((cell: any, index: number) => ({
    cell,
    x: 105 + ((cell.x || index * 73) % 650),
    y: 90 + ((cell.y || index * 107) % 330),
    radius: 24 + cell.retention * 24,
    label: cell.label.slice(0, 14),
  }))
);

const hiveGraphLinks = computed(() => {
  const links: Array<{ id: string; left: typeof hiveGraphNodes.value[number]; right: typeof hiveGraphNodes.value[number]; similarity: number }> = [];
  const nodes = hiveGraphNodes.value;
  for (let left = 0; left < nodes.length; left++) {
    for (let right = left + 1; right < nodes.length; right++) {
      const similarity = compositionSimilarity(
        nodes[left].cell.components || [],
        nodes[right].cell.components || []
      );
      if (similarity >= 0.18) {
        links.push({
          id: `${nodes[left].cell.id}-${nodes[right].cell.id}`,
          left: nodes[left],
          right: nodes[right],
          similarity,
        });
      }
    }
  }
  return links;
});

function compositionSimilarity(left: any[], right: any[]) {
  const leftMap = new Map(left.map(item => [item.cloud_id || item.id || 0, item.composition_share]));
  const rightMap = new Map(right.map(item => [item.cloud_id || item.id || 0, item.composition_share]));
  let overlap = 0;
  for (const [key, value] of leftMap) {
    overlap += Math.min(value, rightMap.get(key) || 0);
  }
  return overlap;
}

const reasoningStatus = computed(() =>
  hiveStore.reasoningLoading ? 'выполняется'
    : hiveStore.runResult?.stop_reason || 'готов'
);
</script>

<style scoped lang="scss">
.hive-panel {
  overflow: auto;
}

.hive-count {
  color: #ffc968;
  font-size: 10px;
}

.hive-summary {
  display: flex;
  justify-content: space-around;
  padding: 15px;
  border-bottom: 1px solid rgba(162, 189, 225, 0.1);
}

.hive-summary div {
  display: grid;
  gap: 3px;
}

.hive-summary strong {
  color: #f2f7ff;
  font-size: 19px;
  font-weight: 500;
}

.reasoning-controls {
  margin: 0 16px 16px;
  padding: 12px;
  border: 1px solid rgba(120, 231, 208, 0.2);
  border-radius: 10px;
  background: rgba(5, 14, 28, 0.4);
}

.reasoning-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
  color: #8ca2c1;
  font-size: 10px;
}

.reasoning-controls label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: #8ca2c1;
  font-size: 10px;
}

.reasoning-controls input,
.json-toolbar input,
.json-toolbar select {
  width: 70px;
  border: 1px solid rgba(160, 190, 225, 0.2);
  border-radius: 5px;
  padding: 4px;
  color: #e7f0ff;
  background: #081421;
  font: 11px system-ui;
}

.reasoning-actions {
  display: flex;
  gap: 7px;
  margin-top: 8px;
}

.reasoning-actions button {
  flex: 1;
}

.primary,
.secondary {
  border: 0;
  border-radius: 6px;
  padding: 7px;
  color: #07111f;
  background: #78e7d0;
  cursor: pointer;
  font: 10px system-ui;
}

.secondary {
  border: 1px solid rgba(126, 233, 208, 0.25);
  color: #bceee4;
  background: rgba(126, 233, 208, 0.08);
}

.reasoning-meta {
  margin-top: 8px;
  overflow: hidden;
  color: #7187a7;
  font: 9px ui-monospace, Consolas, monospace;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.json-actions {
  margin-top: 8px;
}

.hive-map {
  position: relative;
  min-height: 260px;
  height: 330px;
  margin: 16px;
  overflow: hidden;
  border: 1px solid rgba(255, 201, 104, 0.15);
  border-radius: 12px;
  background: radial-gradient(circle at center, rgba(133, 92, 31, 0.16), transparent 60%), rgba(7, 17, 31, 0.56);
}

.hive-map::before {
  content: '';
  position: absolute;
  inset: 18px;
  background-image: linear-gradient(30deg, rgba(255, 201, 104, 0.08) 12%, transparent 12.5%, transparent 87%, rgba(255, 201, 104, 0.08) 87.5%), linear-gradient(150deg, rgba(255, 201, 104, 0.08) 12%, transparent 12.5%, transparent 87%, rgba(255, 201, 104, 0.08) 87.5%);
  background-size: 34px 58px;
  clip-path: polygon(25% 6%, 75% 6%, 100% 50%, 75% 94%, 25% 94%, 0 50%);
}

.hive-graph {
  position: relative;
  width: 100%;
  height: 100%;
  display: block;
}

.hive-graph line {
  stroke: #6ca2ff;
  stroke-opacity: 0.42;
}

.hive-node {
  cursor: pointer;
}

.hive-node > circle:first-child {
  fill: rgba(120, 231, 208, 0.16);
  stroke: rgba(120, 231, 208, 0.82);
  stroke-width: 2;
}

.hive-node .node-core {
  fill: rgba(15, 38, 60, 0.88);
  stroke: rgba(255, 201, 104, 0.42);
  stroke-width: 1;
}

.hive-node.selected > circle:first-child,
.hive-node.active > circle:first-child,
.hive-node:hover > circle:first-child {
  fill: rgba(255, 201, 104, 0.2);
  stroke: #ffc968;
  filter: drop-shadow(0 0 9px rgba(255, 201, 104, 0.55));
}

.node-label {
  fill: #d9f6ef;
  font: 11px system-ui;
  pointer-events: none;
}

.node-gravity {
  fill: #ffc968;
  font: 10px system-ui;
  pointer-events: none;
}

.hive-empty {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: #607590;
  font-size: 11px;
}

.hive-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  padding: 0 18px 12px;
  color: #8ca2c1;
  font-size: 10px;
}

.hive-legend i,
.hive-legend b {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-right: 4px;
  border-radius: 50%;
  background: #78e7d0;
}

.hive-legend b {
  width: 15px;
  height: 2px;
  border-radius: 0;
  background: #6ca2ff;
  vertical-align: middle;
}

.hive-inspector {
  margin: 0 18px 16px;
  padding: 14px;
  border: 1px solid rgba(120, 231, 208, 0.24);
  border-radius: 11px;
  background: rgba(5, 14, 28, 0.46);
}

.inspector-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
}

.inspector-head strong {
  display: block;
  max-width: 250px;
  margin-top: 3px;
  color: #ecf7ff;
  font-size: 13px;
  line-height: 1.35;
}

.node-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 10px 0;
  color: #8fa7c7;
  font-size: 10px;
}

.node-stats span {
  padding: 5px 7px;
  border-radius: 5px;
  background: rgba(108, 162, 255, 0.1);
}

.node-stats b {
  color: #dff7f1;
}

.component-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.component {
  display: flex;
  align-items: center;
  gap: 7px;
  max-width: 100%;
  padding: 6px 8px;
  border: 1px solid rgba(108, 162, 255, 0.2);
  border-radius: 6px;
  color: #c7d5e9;
  background: rgba(21, 41, 68, 0.7);
  font: 11px system-ui;
}

.component b {
  color: #ffc968;
  font-weight: 500;
}

@media (max-width: 1180px) {
  .hive-panel {
    grid-column: 1 / -1;
    min-height: 360px;
  }
}

@media (max-width: 760px) {
  .hive-panel {
    grid-column: 1 / -1;
    min-height: 420px;
  }
}
</style>
