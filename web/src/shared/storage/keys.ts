/** Storage keys - single source of truth for localStorage keys. */

export const STORAGE_KEYS = {
  // Hive/chat persistence
  ACTIVE_HIVE: 'superai-v2-active-hive',
  CHAT_CACHE: 'superai-v2-chat-cache',
  CONVERSATION_ID: 'superai-v2-conversation',

  // Model/field persistence
  MODEL_VIEW_STATE: 'superai-v2-model-view-state',
  SELECTED_PLACEMENT: 'superai-v2-selected-placement',
  BREADCRUMB: 'superai-v2-breadcrumb',

  // Training
  LAST_TRAINING_TEXT: 'superai-v2-last-training-text',
  SHOW_MODEL_DATA: 'superai-v2-show-model-data',

  // UI preferences
  THEME: 'superai-v2-theme',
  PANEL_SIZES: 'superai-v2-panel-sizes',
} as const;

export type StorageKey = (typeof STORAGE_KEYS)[keyof typeof STORAGE_KEYS];