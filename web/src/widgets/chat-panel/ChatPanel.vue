<template>
  <section class="panel chat-panel">
    <div class="panel-head">
      <div>
        <div class="kicker">CONVERSATION</div>
        <h1>Диалог</h1>
      </div>
      <span class="run-state" :class="{ running: hiveStore.loading }">
        {{ hiveStore.loading ? 'обработка…' : `${hiveStore.messages.length} сообщений` }}
      </span>
    </div>
    <div class="messages">
      <article
        v-for="message in hiveStore.messages"
        :key="message.id"
        class="message"
        :class="message.role"
      >
        <div class="message-meta">
          <span>{{ message.role === 'user' ? 'Вы' : 'Улей' }}</span>
          <time>{{ formatTime(message.created_at) }}</time>
        </div>
        <div class="bubble">{{ message.text }}</div>
      </article>
      <div v-if="!hiveStore.messages.length" class="empty-chat">
        Контекст этого улья сохранится при переходе к обучению поля.
      </div>
    </div>
    <form class="composer" @submit.prevent="sendMessage">
      <textarea
        v-model="draft"
        rows="3"
        placeholder="Например: Кот ест рыбу"
        :disabled="hiveStore.loading"
      />
      <div class="composer-foot">
        <span v-if="hiveStore.error" class="connection-error">{{ hiveStore.error }}</span>
        <span v-else>Улей сначала проверит локальную память</span>
        <button class="send" :disabled="hiveStore.loading || !draft.trim()">
          {{ hiveStore.loading ? 'Ищу…' : 'Отправить' }}
        </button>
      </div>
    </form>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useHiveStore } from '@/entities/hive/store';
import { formatTime } from '@/shared/utils/time';

const hiveStore = useHiveStore();
const draft = ref('');

async function sendMessage() {
  const text = draft.value.trim();
  if (!text || hiveStore.loading || !hiveStore.hive) return;
  await hiveStore.query(text);
  draft.value = '';
}
</script>

<style scoped lang="scss">
.chat-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
  border: 1px solid rgba(162, 189, 225, 0.15);
  border-radius: 14px;
  background: rgba(14, 28, 48, 0.76);
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.16);
}

.messages {
  flex: 1;
  min-height: 0;
  padding: 18px 16px;
  overflow: auto;
}

.message {
  margin-bottom: 16px;
}

.message-meta {
  display: flex;
  justify-content: space-between;
  margin-bottom: 6px;
  color: #91a5c4;
  font-size: 11px;
}

.message-meta time {
  color: #5f7393;
  font-size: 10px;
}

.bubble {
  max-width: 92%;
  padding: 10px 12px;
  border: 1px solid rgba(162, 189, 225, 0.13);
  border-radius: 11px 11px 11px 3px;
  color: #c7d5e9;
  background: rgba(21, 41, 68, 0.65);
  font-size: 13px;
  line-height: 1.5;
}

.message.user .message-meta {
  justify-content: flex-end;
  gap: 8px;
  color: #ffc968;
}

.message.user .bubble {
  margin-left: auto;
  border-color: rgba(255, 201, 104, 0.18);
  border-radius: 11px 11px 3px 11px;
  color: #f3dfb5;
  background: rgba(93, 71, 35, 0.35);
}

.empty-chat {
  margin-top: 46px;
  color: #607a9d;
  font-size: 12px;
  line-height: 1.55;
}

.composer {
  flex: 0 0 auto;
  padding: 14px;
  border-top: 1px solid rgba(162, 189, 225, 0.1);
}

.composer textarea {
  width: 100%;
  resize: none;
  box-sizing: border-box;
  border: 1px solid rgba(162, 189, 225, 0.16);
  border-radius: 9px;
  padding: 10px;
  color: #e7f0ff;
  background: rgba(5, 14, 28, 0.55);
  outline: none;
  font: inherit;
  font-size: 12px;
}

.composer textarea:focus {
  border-color: #78e7d0;
}

.composer-foot {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  margin-top: 8px;
  color: #637895;
  font-size: 10px;
}

.connection-error {
  color: #ffad91;
}

.send {
  min-width: 88px;
  height: 30px;
  border: 0;
  border-radius: 8px;
  color: #07111f;
  background: #78e7d0;
  cursor: pointer;
  font: inherit;
  font-size: 11px;
}

.send:disabled {
  opacity: 0.35;
}
</style>