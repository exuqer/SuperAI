<template>
  <div class="training-page">
    <ModelWorkspace />
  </div>
</template>

<script setup lang="ts">
import ModelWorkspace from '@/widgets/model-workspace/ModelWorkspace.vue';
</script>

<style scoped>
.training-page {
  min-height: 100vh;
  color: #dce8f8;
  background: #07111f;
}
</style>