<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { RouterLink, useRoute } from 'vue-router'

import type { ApiMode } from '@/shared/api/service'
import { useRuntimeStore } from '@/shared/model/runtime-store'
import StatusBadge from './StatusBadge.vue'

const runtime = useRuntimeStore()
const route = useRoute()
const navigationOpen = ref(false)

const navigation = [
  { name: 'run', label: 'Запуск', hint: 'Новая задача' },
  { name: 'traces', label: 'Трассы', hint: 'Причины и spans' },
  { name: 'hive', label: 'Улей', hint: 'Рабочая память' },
  { name: 'storage', label: 'Хранилище', hint: 'Артефакты' },
  { name: 'cosmos', label: 'Космос', hint: 'Claims и provenance' },
  { name: 'system', label: 'Система', hint: 'Health и очередь' },
]

function changeMode(event: Event) {
  const value = (event.target as HTMLSelectElement).value as ApiMode
  runtime.setMode(value)
  void runtime.bootstrap()
}

function closeNavigation() {
  navigationOpen.value = false
}

onMounted(() => {
  void runtime.bootstrap()
})
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <RouterLink class="brand" :to="{ name: 'run' }" @click="closeNavigation">
        <span class="brand__mark" aria-hidden="true">S</span>
        <span>
          <strong>SuperAI</strong>
          <small>trace-first client</small>
        </span>
      </RouterLink>

      <div class="topbar__status">
        <StatusBadge
          v-if="runtime.system"
          :status="runtime.system.health.status"
          :label="runtime.system.health.status === 'ok' ? 'система готова' : undefined"
        />
        <span v-else class="muted">проверка состояния…</span>
        <label v-if="runtime.isModeToggleAvailable" class="mode-switch">
          <span>источник</span>
          <select :value="runtime.mode" aria-label="Источник данных" @change="changeMode">
            <option value="mock">mock</option>
            <option value="live">live</option>
          </select>
        </label>
      </div>

      <button
        class="menu-toggle"
        type="button"
        :aria-expanded="navigationOpen"
        aria-controls="main-navigation"
        @click="navigationOpen = !navigationOpen"
      >
        Навигация
      </button>
    </header>

    <div class="workspace">
      <aside id="main-navigation" class="sidebar" :class="{ 'sidebar--open': navigationOpen }">
        <nav aria-label="Основная навигация">
          <RouterLink
            v-for="item in navigation"
            :key="item.name"
            :to="{ name: item.name }"
            class="nav-link"
            :class="{ 'nav-link--active': route.name === item.name }"
            @click="closeNavigation"
          >
            <span>{{ item.label }}</span>
            <small>{{ item.hint }}</small>
          </RouterLink>
        </nav>
        <div class="sidebar__note">
          <strong>Версия контрактов</strong>
          <span>{{ runtime.system?.meta.schemaVersion ?? '—' }}</span>
          <small>{{ runtime.system?.meta.apiVersion ?? 'пока без backend' }}</small>
        </div>
      </aside>

      <main class="main-content">
        <slot />
      </main>
    </div>
  </div>
</template>

<style scoped lang="scss">
.app-shell {
  min-height: 100vh;
}

.topbar {
  position: sticky;
  z-index: 10;
  top: 0;
  display: flex;
  align-items: center;
  gap: 1rem;
  min-height: 4.25rem;
  padding: 0.75rem clamp(1rem, 4vw, 2.25rem);
  border-bottom: 1px solid rgba(168, 190, 228, 0.15);
  background: rgba(9, 17, 31, 0.88);
  backdrop-filter: blur(16px);
}

.brand {
  display: inline-flex;
  align-items: center;
  gap: 0.7rem;
  color: #f3f7ff;
  text-decoration: none;

  strong,
  small {
    display: block;
  }

  strong {
    letter-spacing: -0.025em;
  }

  small {
    color: #8fa2c1;
    font-size: 0.68rem;
  }
}

.brand__mark {
  display: grid;
  place-items: center;
  width: 2.15rem;
  height: 2.15rem;
  border-radius: 0.66rem;
  color: #061325;
  background: linear-gradient(135deg, #76e8cc, #6fa4ff);
  font-weight: 900;
}

.topbar__status {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-left: auto;
}

.mode-switch {
  display: flex;
  align-items: center;
  gap: 0.42rem;
  color: #aabbd5;
  font-size: 0.75rem;

  select {
    border: 1px solid rgba(168, 190, 228, 0.23);
    border-radius: 0.45rem;
    color: #eaf2ff;
    background: #101e36;
    padding: 0.28rem 0.4rem;
  }
}

.menu-toggle {
  display: none;
  border: 1px solid rgba(168, 190, 228, 0.25);
  border-radius: 0.55rem;
  color: #e8effd;
  background: #13223b;
  padding: 0.45rem 0.6rem;
}

.workspace {
  display: grid;
  grid-template-columns: 15rem minmax(0, 1fr);
  min-height: calc(100vh - 4.25rem);
}

.sidebar {
  position: sticky;
  top: 4.25rem;
  align-self: start;
  min-height: calc(100vh - 4.25rem);
  padding: 1.15rem 0.85rem;
  border-right: 1px solid rgba(168, 190, 228, 0.12);
  background: rgba(10, 20, 35, 0.45);

  nav {
    display: grid;
    gap: 0.22rem;
  }
}

.nav-link {
  display: grid;
  gap: 0.08rem;
  border-radius: 0.62rem;
  color: #c5d3ea;
  padding: 0.65rem 0.74rem;
  text-decoration: none;

  &:hover {
    background: rgba(106, 153, 236, 0.11);
  }

  &--active {
    color: #f4f8ff;
    background: linear-gradient(90deg, rgba(55, 112, 209, 0.29), rgba(55, 112, 209, 0.08));
  }

  span {
    font-size: 0.89rem;
    font-weight: 720;
  }

  small {
    color: #8194b3;
    font-size: 0.7rem;
  }
}

.sidebar__note {
  display: grid;
  gap: 0.25rem;
  margin: 1.3rem 0.3rem;
  color: #aec2e2;
  font-size: 0.8rem;

  small {
    color: #8195b4;
  }
}

.main-content {
  width: min(100%, 90rem);
  padding: clamp(1rem, 3vw, 2.25rem);
}

@media (max-width: 720px) {
  .topbar {
    gap: 0.65rem;
  }

  .topbar__status {
    margin-left: auto;
  }

  .mode-switch span {
    display: none;
  }

  .menu-toggle {
    display: block;
  }

  .workspace {
    display: block;
  }

  .sidebar {
    position: fixed;
    z-index: 9;
    top: 4.25rem;
    bottom: 0;
    left: 0;
    width: min(18rem, 88vw);
    min-height: 0;
    transform: translateX(-105%);
    transition: transform 160ms ease;
    box-shadow: 18px 0 40px rgba(0, 0, 0, 0.35);

    &--open {
      transform: translateX(0);
    }
  }
}

@media (max-width: 460px) {
  .topbar__status :deep(.status-badge) {
    display: none;
  }
}
</style>
