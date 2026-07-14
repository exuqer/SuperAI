<template>
  <div v-if="jsonOpen" class="json-modal" role="dialog" aria-modal="true">
    <div class="json-card">
      <div class="json-card-head">
        <strong>JSON улья</strong>
        <button class="ghost" @click="jsonOpen = false">×</button>
      </div>
      <div class="json-toolbar">
        <select v-model="jsonMode" @change="loadJson">
          <option value="current">Текущее состояние</option>
          <option value="initial">Исходное состояние</option>
          <option value="snapshot">Шаг рассуждения</option>
          <option value="trace">Полный журнал</option>
        </select>
        <input v-if="jsonMode === 'snapshot' || jsonMode === 'initial'" v-model.number="jsonStep" type="number" min="0" @change="loadJson" />
        <select v-model="jsonFormat">
          <option value="formatted">Форматированный</option>
          <option value="raw">Сырой</option>
        </select>
        <button class="secondary" @click="copyJson(jsonMode)">Копировать</button>
        <button class="secondary" @click="downloadJson">Скачать</button>
      </div>
      <pre>{{ jsonText }}</pre>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useHiveExport } from '../model/useHiveExport';

const {
  jsonOpen,
  jsonMode,
  jsonFormat,
  jsonStep,
  jsonValue,
  loadJson,
  copyJson,
  downloadJson,
} = useHiveExport();

const jsonText = computed(() =>
  jsonValue.value == null
    ? 'Загрузка…'
    : jsonFormat.value === 'raw'
      ? JSON.stringify(jsonValue.value)
      : JSON.stringify(jsonValue.value, null, 2)
);
</script>

<style scoped lang="scss">
.json-modal {
  position: fixed;
  inset: 0;
  z-index: 20;
  display: grid;
  place-items: center;
  padding: 20px;
  background: rgba(2, 8, 18, 0.76);
}

.json-card {
  width: min(900px, 96vw);
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  padding: 16px;
  border: 1px solid rgba(126, 233, 208, 0.28);
  border-radius: 12px;
  background: #0d1d31;
  box-shadow: 0 18px 70px rgba(0, 0, 0, 0.5);
}

.json-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}

.json-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-bottom: 10px;
}

.json-toolbar select,
.json-toolbar input {
  width: auto;
  border: 1px solid rgba(160, 190, 225, 0.2);
  border-radius: 5px;
  padding: 4px;
  color: #e7f0ff;
  background: #081421;
  font: 11px system-ui;
}

.json-toolbar select {
  width: auto;
}

.secondary {
  border: 1px solid rgba(126, 233, 208, 0.27);
  border-radius: 7px;
  padding: 6px 8px;
  color: #bceee4;
  background: rgba(126, 233, 208, 0.08);
  cursor: pointer;
  font: 10px system-ui;
}

.json-card pre {
  min-height: 300px;
  margin: 10px 0 0;
  padding: 12px;
  overflow: auto;
  border: 1px solid rgba(160, 190, 225, 0.12);
  border-radius: 8px;
  color: #c9d9ef;
  background: #06111f;
  font: 10px/1.45 ui-monospace, Consolas, monospace;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
