<script setup lang="ts">
import { computed, ref, onMounted, nextTick } from 'vue'
import { useRuntimeStore } from '@/shared/model/runtime-store'
import { useRouter, useRoute } from 'vue-router'
import StatusBadge from '@/widgets/app-shell/StatusBadge.vue'
import JsonViewer from '@/widgets/inspectors/JsonViewer.vue'

const runtime = useRuntimeStore()
const router = useRouter()
const route = useRoute()

const messages = computed(() => runtime.conversationMessages(conversationId.value))
const inputMessage = ref('')
const isLoading = ref(false)
const conversationId = ref(route.params.conversationId as string || 'conversation-local')
const projectId = ref(route.params.projectId as string || '')
const showTrace = ref(false)
const selectedTraceId = ref<string>()

const canSend = computed(() => inputMessage.value.trim().length > 0 && !isLoading.value)

async function sendMessage() {
  if (!canSend.value) return
  
  const message = inputMessage.value.trim()
  inputMessage.value = ''
  isLoading.value = true
  
  // Add user message
  runtime.appendConversationMessage(conversationId.value, {
    role: 'user',
    content: message,
    timestamp: new Date().toISOString(),
  })
  
  await nextTick()
  scrollToBottom()
  
  try {
    const task = await runtime.runTask({
      schema_version: '1.0',
      tenant_id: 'local',
      user_id: 'user-local',
      message,
      conversation_id: conversationId.value,
      project_id: projectId.value || undefined,
      budget: {
        schema_version: '1.0',
        time_ms: 30000,
        step_limit: 32,
        memory_bytes: 16384,
        event_limit: 20,
      },
    })
    
    if (task?.answer?.text) {
      runtime.appendConversationMessage(conversationId.value, {
        role: 'assistant',
        content: task.answer.text,
        timestamp: new Date().toISOString(),
        taskId: task.id,
        traceId: task.traceId,
      })
      
      if (task.traceId) {
        selectedTraceId.value = task.traceId
      }
    } else {
      runtime.appendConversationMessage(conversationId.value, {
        role: 'system',
        content: `Ошибка: ${task?.error?.message ?? runtime.runError?.message ?? 'Неизвестная ошибка'}`,
        timestamp: new Date().toISOString(),
      })
    }
  } catch (error) {
    runtime.appendConversationMessage(conversationId.value, {
      role: 'system',
      content: `Ошибка: ${error instanceof Error ? error.message : 'Неизвестная ошибка'}`,
      timestamp: new Date().toISOString(),
    })
  } finally {
    isLoading.value = false
    await nextTick()
    scrollToBottom()
  }
}

function scrollToBottom() {
  const container = document.querySelector('.messages-container')
  if (container) {
    container.scrollTop = container.scrollHeight
  }
}

function openTrace(traceId: string) {
  selectedTraceId.value = traceId
  showTrace.value = true
}

function closeTrace() {
  showTrace.value = false
  selectedTraceId.value = undefined
}

function clearConversation() {
  runtime.clearConversationMessages(conversationId.value)
  conversationId.value = `conv-${Date.now()}`
  window.localStorage.setItem('superai.activeConversationId', conversationId.value)
  if (route.params.conversationId) {
    router.push({ name: 'dialogue', params: { conversationId: conversationId.value, projectId: projectId.value } })
  }
}

function formatTime(date: Date) {
  return date.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
}

function addNewline(event: KeyboardEvent) {
  if (event.shiftKey) return
  event.preventDefault()
  void sendMessage()
}

function formatMessage(content: string): string {
  return content
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>')
}

onMounted(() => {
  // The messages are restored from the runtime store/localStorage.
  if (route.params.conversationId) {
    conversationId.value = route.params.conversationId as string
  } else {
    conversationId.value = window.localStorage.getItem('superai.activeConversationId') || 'conversation-local'
  }
  window.localStorage.setItem('superai.activeConversationId', conversationId.value)
  if (route.params.projectId) {
    projectId.value = route.params.projectId as string
  }
})
</script>

<template>
  <div class="dialogue-page">
    <header class="page-heading">
      <div>
        <p class="eyebrow">Режим диалога</p>
        <h1>Чат с агентом</h1>
        <p>Непрерывный диалог с агентом SuperAI. История сохраняется в рамках Conversation ID.</p>
      </div>
      <div class="header-actions">
        <StatusBadge
          :status="runtime.mode === 'mock' ? 'verified' : 'running'"
          :label="runtime.mode === 'mock' ? 'mock DTO' : 'live API'"
        />
        <button class="button button--quiet" @click="clearConversation">
          Новый диалог
        </button>
      </div>
    </header>

    <div class="dialogue-layout">
      <main class="messages-panel surface">
        <div class="messages-container" ref="messagesContainer">
          <div v-if="messages.length === 0" class="empty-state">
            <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            <h3>Начните диалог</h3>
            <p>Введите сообщение ниже, чтобы начать общение с агентом.</p>
          </div>
          
          <div v-else class="messages-list">
            <div
              v-for="(message, index) in messages"
              :key="index"
              :class="['message', `message--${message.role}`]"
            >
              <div class="message-header">
                <span class="message-role">
                  <span v-if="message.role === 'user'">👤 Вы</span>
                  <span v-else-if="message.role === 'assistant'">🤖 Агент</span>
                  <span v-else>⚙️ Система</span>
                </span>
                <span class="message-time">{{ formatTime(new Date(message.timestamp)) }}</span>
                <span v-if="message.taskId" class="message-task-id">{{ message.taskId.slice(0, 8) }}…</span>
              </div>
              <div class="message-content" v-html="formatMessage(message.content)"></div>
              <div v-if="message.role === 'assistant' && message.taskId" class="message-actions">
                <button v-if="message.traceId" class="button button--quiet button--small" @click="openTrace(message.traceId)">
                  Открыть трассу
                </button>
                <RouterLink
                  v-if="runtime.task?.hiveId"
                  class="button button--quiet button--small"
                  :to="{ name: 'hive', params: { hiveId: runtime.task.hiveId } }"
                >
                  Открыть Улей
                </RouterLink>
              </div>
            </div>
          </div>
        </div>
        
        <div class="input-area">
          <div class="conversation-info">
            <span class="info-item">
              <strong>Conversation ID:</strong> {{ conversationId }}
            </span>
            <span class="info-item" v-if="projectId">
              <strong>Project ID:</strong> {{ projectId }}
            </span>
          </div>
          <div class="input-row">
            <textarea
              v-model="inputMessage"
              @keydown.enter.exact="sendMessage"
              @keydown.enter.shift="addNewline"
              placeholder="Введите сообщение… (Enter — отправить, Shift+Enter — новая строка)"
              :disabled="isLoading"
              rows="3"
              class="message-input"
            />
            <button
              class="button button--primary send-button"
              :disabled="!canSend"
              @click="sendMessage"
            >
              <span v-if="isLoading">Отправка…</span>
              <span v-else>Отправить</span>
            </button>
          </div>
        </div>
      </main>

      <aside class="sidebar surface" v-if="runtime.task || runtime.hive || runtime.activeTrace">
        <div class="sidebar-section" v-if="runtime.task">
          <h3>Текущая задача</h3>
          <StatusBadge :status="runtime.task.status" />
          <p class="task-id">{{ runtime.task.id }}</p>
          <p v-if="runtime.task.answer" class="task-answer">{{ runtime.task.answer.text }}</p>
        </div>
        
        <div class="sidebar-section" v-if="runtime.hive">
          <h3>Улей</h3>
          <p class="hive-id">{{ runtime.hive.id }}</p>
          <p>Состояние: {{ runtime.hive.state }}</p>
          <RouterLink class="button button--secondary button--small" :to="{ name: 'hive', params: { hiveId: runtime.hive.id } }">
            Открыть Улей
          </RouterLink>
        </div>

        <div class="sidebar-section" v-if="runtime.activeTrace">
          <h3>Активная трасса</h3>
          <p class="trace-id">{{ runtime.activeTrace.id }}</p>
          <RouterLink class="button button--secondary button--small" :to="{ name: 'traces', params: { traceId: runtime.activeTrace.id } }">
            Открыть трассу
          </RouterLink>
        </div>
      </aside>
    </div>

    <!-- Trace Modal -->
    <div v-if="showTrace" class="modal-overlay" @click="closeTrace">
      <div class="modal-content" @click.stop>
        <header class="modal-header">
          <h3>Трасса выполнения</h3>
          <button class="button button--quiet button--icon" @click="closeTrace">✕</button>
        </header>
        <div class="modal-body">
          <JsonViewer :data="runtime.traces[selectedTraceId!]" />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
.dialogue-page {
  display: grid;
  grid-template-rows: auto 1fr;
  height: 100vh;
  overflow: hidden;
}

.page-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1.5rem;
  padding: 1.25rem 1.5rem;
  border-bottom: 1px solid rgba(168, 190, 228, 0.12);
  background: rgba(10, 20, 35, 0.8);
  backdrop-filter: blur(8px);
  
  .eyebrow {
    margin: 0 0 0.25rem;
    color: #73a0e8;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  
  h1 {
    margin: 0 0 0.25rem;
    font-size: 1.35rem;
    font-weight: 600;
    color: #eaf2ff;
  }
  
  p {
    margin: 0;
    color: #8fa1bd;
    font-size: 0.85rem;
    line-height: 1.4;
  }
  
  .header-actions {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-shrink: 0;
  }
}

.dialogue-layout {
  display: grid;
  grid-template-columns: 1fr 280px;
  gap: 1rem;
  padding: 1rem;
  overflow: hidden;
  
  @media (max-width: 1000px) {
    grid-template-columns: 1fr;
  }
}

.messages-panel {
  display: grid;
  grid-template-rows: 1fr auto;
  overflow: hidden;
  border-radius: 0.85rem;
  border: 1px solid rgba(168, 190, 228, 0.12);
  background: #0a1423;
}

.messages-container {
  overflow-y: auto;
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  min-height: 300px;
  color: #6b8ab8;
  text-align: center;
  padding: 2rem;
  
  .empty-icon {
    width: 64px;
    height: 64px;
    margin-bottom: 1rem;
    opacity: 0.5;
  }
  
  h3 {
    margin: 0 0 0.5rem;
    color: #9ab0d6;
    font-size: 1.1rem;
  }
  
  p {
    margin: 0;
    font-size: 0.9rem;
  }
}

.message {
  display: grid;
  gap: 0.4rem;
  max-width: 85%;
  
  &.message--user {
    margin-left: auto;
    text-align: right;
  }
  
  &.message--assistant {
    margin-right: auto;
  }
  
  &.message--system {
    margin: 0 auto;
    max-width: 90%;
    opacity: 0.7;
    font-size: 0.8rem;
  }
}

.message-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.7rem;
  color: #73a0e8;
  
  .message--user & {
    justify-content: flex-end;
  }
  
  .message-role {
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  
  .message-time {
    color: #5a7ab8;
  }
  
  .message-task-id {
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.65rem;
    color: #4a6fa5;
    background: rgba(74, 111, 165, 0.15);
    padding: 0.1rem 0.4rem;
    border-radius: 0.3rem;
  }
}

.message-content {
  padding: 0.75rem 1rem;
  border-radius: 0.75rem;
  line-height: 1.6;
  font-size: 0.9rem;
  white-space: pre-wrap;
  word-wrap: break-word;
  
  .message--user & {
    background: linear-gradient(135deg, rgba(115, 160, 232, 0.15), rgba(69, 130, 224, 0.1));
    border: 1px solid rgba(115, 160, 232, 0.2);
    color: #eaf2ff;
    border-bottom-right-radius: 0.25rem;
  }
  
  .message--assistant & {
    background: rgba(15, 25, 40, 0.8);
    border: 1px solid rgba(168, 190, 228, 0.12);
    color: #d4e4f7;
    border-bottom-left-radius: 0.25rem;
  }
  
  .message--system & {
    background: rgba(251, 128, 145, 0.1);
    border: 1px solid rgba(251, 128, 145, 0.2);
    color: #ffabb6;
    font-style: italic;
    font-size: 0.8rem;
  }
}

.message-actions {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.35rem;
  padding: 0 0.25rem;
  
  .message--user & {
    justify-content: flex-end;
  }
}

.input-area {
  padding: 1rem 1.25rem;
  border-top: 1px solid rgba(168, 190, 228, 0.1);
  background: rgba(10, 20, 35, 0.6);
  border-bottom-left-radius: 0.85rem;
  border-bottom-right-radius: 0.85rem;
}

.conversation-info {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-bottom: 0.75rem;
  font-size: 0.7rem;
  color: #6b8ab8;
  
  .info-item {
    font-family: "SFMono-Regular", Consolas, monospace;
  }
}

.input-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 0.75rem;
  align-items: end;
}

.message-input {
  width: 100%;
  min-height: 60px;
  max-height: 160px;
  padding: 0.7rem 0.9rem;
  border: 1px solid rgba(168, 190, 228, 0.18);
  border-radius: 0.6rem;
  background: #08101d;
  color: #eaf2ff;
  font-family: inherit;
  font-size: 0.9rem;
  line-height: 1.5;
  resize: vertical;
  transition: border-color 0.15s, box-shadow 0.15s;
  
  &:focus {
    outline: none;
    border-color: #73a0e8;
    box-shadow: 0 0 0 3px rgba(115, 160, 232, 0.15);
  }
  
  &::placeholder {
    color: #5a7ab8;
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
}

.send-button {
  height: fit-content;
  padding: 0.7rem 1.25rem;
  font-weight: 600;
  white-space: nowrap;
}

.sidebar {
  display: grid;
  gap: 1rem;
  overflow-y: auto;
  padding: 1rem;
  border-radius: 0.85rem;
  border: 1px solid rgba(168, 190, 228, 0.12);
  background: #0a1423;
  
  @media (max-width: 1000px) {
    display: none;
  }
}

.sidebar-section {
  padding: 0.75rem;
  border-radius: 0.6rem;
  background: rgba(15, 25, 40, 0.6);
  border: 1px solid rgba(168, 190, 228, 0.08);
  
  h3 {
    margin: 0 0 0.5rem;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #73a0e8;
  }
  
  p {
    margin: 0.25rem 0;
    font-size: 0.8rem;
    color: #b4c3db;
  }
}

.task-id, .hive-id, .trace-id {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.7rem;
  color: #73a0e8;
  word-break: break-all;
}

.task-answer {
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: rgba(115, 160, 232, 0.08);
  border-radius: 0.4rem;
  font-size: 0.8rem;
  color: #cbd8eb;
  white-space: pre-wrap;
  max-height: 120px;
  overflow-y: auto;
}

.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 1rem;
}

.modal-content {
  width: 100%;
  max-width: 900px;
  max-height: 85vh;
  background: #0d1828;
  border: 1px solid rgba(168, 190, 228, 0.15);
  border-radius: 1rem;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid rgba(168, 190, 228, 0.12);
  
  h3 {
    margin: 0;
    font-size: 1rem;
    color: #eaf2ff;
  }
}

.modal-body {
  flex: 1;
  overflow: auto;
  padding: 1rem;
}
</style>
