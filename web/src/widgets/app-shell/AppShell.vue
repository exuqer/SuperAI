<template>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">
        <strong>semantic_ants</strong>
        <span v-if="runtime.config" class="muted">{{ runtime.config.state_dir }}</span>
      </div>
      <nav>
        <RouterLink v-for="item in nav" :key="item.path" :to="item.path">
          {{ item.title }}
        </RouterLink>
      </nav>
      <div v-if="runtime.error" class="error">{{ runtime.error }}</div>
    </aside>
    <main class="content">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import { RouterLink, RouterView } from 'vue-router';
import { useRuntimeStore } from '@/app/stores/runtime';

const runtime = useRuntimeStore();
const nav = [
  { path: '/chat', title: 'Диалог' },
  { path: '/understand', title: 'Пониматель' },
  { path: '/training', title: 'Обучение' },
  { path: '/layers', title: 'Слои' },
  { path: '/concepts', title: 'Понятия' },
  { path: '/graph', title: 'Граф' },
  { path: '/memory', title: 'Память' },
  { path: '/system', title: 'Система' },
];

onMounted(() => {
  runtime.loadConfig().catch(() => undefined);
});
</script>

<style scoped lang="scss">
.shell {
  display: grid;
  grid-template-columns: 250px minmax(0, 1fr);
  min-height: 100vh;
}

.sidebar {
  position: sticky;
  top: 0;
  display: grid;
  grid-template-rows: auto 1fr auto;
  gap: 18px;
  height: 100vh;
  border-right: 1px solid var(--line);
  background: var(--surface);
  padding: 18px;
}

.brand {
  display: grid;
  gap: 4px;
}

.brand strong {
  font-size: 18px;
}

nav {
  display: grid;
  gap: 6px;
  align-content: start;
}

nav a {
  border-radius: var(--radius);
  padding: 10px 12px;
  color: var(--text);
  text-decoration: none;
}

nav a.router-link-active {
  background: var(--accent);
  color: white;
}

.content {
  min-width: 0;
  padding: 22px;
}

.error {
  border: 1px solid rgba(215, 47, 47, 0.4);
  border-radius: var(--radius);
  padding: 10px;
  color: var(--signal);
  font-size: 13px;
}

@media (max-width: 820px) {
  .shell {
    grid-template-columns: 1fr;
  }

  .sidebar {
    position: static;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }

  nav {
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
  }
}
</style>
