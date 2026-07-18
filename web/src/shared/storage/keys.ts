/** Storage keys - single source of truth for localStorage keys. */

export const STORAGE_KEYS = {
  // Hive/chat persistence
  ACTIVE_HIVE: 'superai-v27-active-hive',
  CHAT_CACHE: 'superai-v27-chat-cache',
  CONVERSATION_ID: 'superai-v27-conversation',

  // Model/field persistence
  MODEL_VIEW_STATE: 'superai-v27-model-view-state',
  SELECTED_PLACEMENT: 'superai-v27-selected-placement',
  BREADCRUMB: 'superai-v27-breadcrumb',

  // Training
  LAST_TRAINING_TEXT: 'superai-v27-last-training-text',
  SHOW_MODEL_DATA: 'superai-v27-show-model-data',

  // UI preferences
  THEME: 'superai-v27-theme',
  PANEL_SIZES: 'superai-v27-panel-sizes',
} as const;

export type StorageKey = (typeof STORAGE_KEYS)[keyof typeof STORAGE_KEYS];
