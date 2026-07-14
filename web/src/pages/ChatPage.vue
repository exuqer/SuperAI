<template>
  <div class="chat-page">
    <header class="topbar">
      <div class="brand">
        <span class="brand-mark">✦</span>
        <div>
          <div class="brand-title">SuperAI <span>· чат с ульем</span></div>
          <div class="brand-sub">локальная память → ограниченный поиск</div>
        </div>
      </div>
      <div class="top-actions">
        <RouterLink class="nav-link" to="/field">Обучение поля</RouterLink>
        <span class="status" :class="mode">{{ modeLabel }}</span>
        <button class="ghost" @click="hiveStore.resetHive">Новый улей</button>
        <span class="avatar">AI</span>
      </div>
    </header>
    <main class="workspace">
      <ChatPanel />
      <RoutingField />
      <HiveWorkspace />
    </main>
    <JsonExportDialog />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue';
import ChatPanel from '@/widgets/chat-panel/ChatPanel.vue';
import RoutingField from '@/widgets/routing-field/RoutingField.vue';
import HiveWorkspace from '@/widgets/hive-workspace/HiveWorkspace.vue';
import JsonExportDialog from '@/features/hive-export/ui/JsonExportDialog.vue';
import { useHiveStore } from '@/entities/hive/store';

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

onMounted(() => { void hiveStore.restoreHive(); });
</script>

<style scoped lang="scss">
.chat-page {
  min-height: 100vh;
  color: #e7f0ff;
  background: #07111f;
}

.topbar {
  height: 72px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 32px;
  border-bottom: 1px solid rgba(162, 189, 225, 0.15);
  background: rgba(8, 18, 33, 0.94);
}

.brand,
.top-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.top-actions { gap: 18px; }
.brand-mark { display: grid; place-items: center; width: 34px; height: 34px; border-radius: 10px; color: #07111f; background: linear-gradient(135deg, #78e7d0, #6ca2ff); font-weight: 800; }
.brand-title { font-size: 18px; font-weight: 700; }
.brand-title span { color: #8496b5; font-weight: 400; }
.brand-sub { color: #7689a8; font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; }
.nav-link { color: #78e7d0; text-decoration: none; }
.status { color: #8496b5; font-size: 12px; }
.status.local { color: #7ee9d0; }
.status.partial { color: #ffc968; }
.status.external { color: #73b0ff; }
.ghost { border: 0; color: #9db0ce; background: transparent; cursor: pointer; }
.avatar { display: grid; place-items: center; width: 30px; height: 30px; border-radius: 50%; color: #07111f; background: #ffc968; font-weight: 700; }

.workspace {
  display: grid;
  grid-template-columns: minmax(260px, 320px) minmax(430px, 1fr) minmax(340px, 430px);
  gap: 14px;
  max-width: 1780px;
  height: calc(100vh - 72px);
  margin: auto;
  padding: 18px;
}

@media (max-width: 1180px) {
  .workspace { height: auto; min-height: calc(100vh - 72px); grid-template-columns: minmax(230px, 0.8fr) minmax(400px, 1.5fr); }
  .workspace :deep(.hive-panel) { grid-column: 1 / -1; min-height: 360px; }
}

@media (max-width: 760px) {
  .topbar { height: auto; min-height: 68px; padding: 12px 16px; }
  .top-actions .status,
  .top-actions .ghost { display: none; }
  .workspace { display: flex; flex-direction: column; height: auto; padding: 10px; }
}
</style>
