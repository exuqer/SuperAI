<template>
  <button type="button" class="info-tooltip" :aria-label="`Описание: ${label}`">
    <span aria-hidden="true">i</span>
    <span class="tooltip-text" role="tooltip"><b>{{ label }}</b>{{ text }}</span>
  </button>
</template>

<script setup lang="ts">
defineProps<{ label: string; text: string }>();
</script>

<style scoped>
.info-tooltip {
  position: relative;
  display: inline-grid;
  place-items: center;
  width: 16px;
  height: 16px;
  margin-left: 5px;
  border: 1px solid rgba(115, 176, 255, .6);
  border-radius: 50%;
  padding: 0;
  color: #9bc4ff;
  background: transparent;
  cursor: help;
  font: 700 10px/1 system-ui;
}

.tooltip-text {
  position: absolute;
  z-index: 5;
  bottom: calc(100% + 8px);
  left: 50%;
  width: min(270px, 70vw);
  transform: translateX(-50%);
  visibility: hidden;
  padding: 9px 10px;
  border: 1px solid rgba(162, 189, 225, .25);
  border-radius: 8px;
  color: #dce8f8;
  background: #0b1a2d;
  box-shadow: 0 8px 24px rgba(0, 0, 0, .35);
  font: 11px/1.4 system-ui;
  text-align: left;
  opacity: 0;
  pointer-events: none;
  transition: opacity .12s ease, visibility .12s ease;
}

.tooltip-text b { display: block; margin-bottom: 3px; color: #78e7d0; }
.info-tooltip:hover .tooltip-text,
.info-tooltip:focus-visible .tooltip-text { visibility: visible; opacity: 1; }
</style>
