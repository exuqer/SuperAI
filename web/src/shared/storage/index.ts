/** Unified storage API with typed access. */

import { STORAGE_KEYS, type StorageKey } from './keys';

function isClient(): boolean {
  return typeof window !== 'undefined' && typeof localStorage !== 'undefined';
}

export function getStorageItem<T>(key: StorageKey, defaultValue: T): T {
  if (!isClient()) return defaultValue;
  try {
    const item = localStorage.getItem(key);
    if (!item) return defaultValue;
    try {
      return JSON.parse(item) as T;
    } catch {
      return item as T;
    }
  } catch {
    return defaultValue;
  }
}

export function setStorageItem<T>(key: StorageKey, value: T): void {
  if (!isClient()) return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Ignore quota exceeded, etc.
  }
}

export function removeStorageItem(key: StorageKey): void {
  if (!isClient()) return;
  localStorage.removeItem(key);
}

function setRawStorageItem(key: StorageKey, value: string): void {
  if (!isClient()) return;
  localStorage.setItem(key, value);
}

export function clearStorage(): void {
  if (!isClient()) return;
  Object.values(STORAGE_KEYS).forEach(key => localStorage.removeItem(key));
}

// Typed helpers for specific keys
export const storage = {
  // Hive/chat
  getActiveHive: () => getStorageItem<string | null>(STORAGE_KEYS.ACTIVE_HIVE, null),
  setActiveHive: (value: string | null) => value === null
    ? removeStorageItem(STORAGE_KEYS.ACTIVE_HIVE)
    : setRawStorageItem(STORAGE_KEYS.ACTIVE_HIVE, value),
  removeActiveHive: () => removeStorageItem(STORAGE_KEYS.ACTIVE_HIVE),

  getChatCache: () => getStorageItem<any>(STORAGE_KEYS.CHAT_CACHE, null),
  setChatCache: (value: any) => setStorageItem(STORAGE_KEYS.CHAT_CACHE, value),
  removeChatCache: () => removeStorageItem(STORAGE_KEYS.CHAT_CACHE),

  getConversationId: () => getStorageItem<string>(STORAGE_KEYS.CONVERSATION_ID, ''),
  setConversationId: (value: string) => setRawStorageItem(STORAGE_KEYS.CONVERSATION_ID, value),

  // Model view
  getModelViewState: () => getStorageItem<any>(STORAGE_KEYS.MODEL_VIEW_STATE, null),
  setModelViewState: (value: any) => setStorageItem(STORAGE_KEYS.MODEL_VIEW_STATE, value),

  getSelectedPlacement: () => getStorageItem<number | null>(STORAGE_KEYS.SELECTED_PLACEMENT, null),
  setSelectedPlacement: (value: number | null) => setStorageItem(STORAGE_KEYS.SELECTED_PLACEMENT, value),

  getBreadcrumb: () => getStorageItem<any[]>(STORAGE_KEYS.BREADCRUMB, []),
  setBreadcrumb: (value: any[]) => setStorageItem(STORAGE_KEYS.BREADCRUMB, value),

  // Training
  getLastTrainingText: () => getStorageItem<string>(STORAGE_KEYS.LAST_TRAINING_TEXT, ''),
  setLastTrainingText: (value: string) => setStorageItem(STORAGE_KEYS.LAST_TRAINING_TEXT, value),

  getShowModelData: () => getStorageItem<boolean>(STORAGE_KEYS.SHOW_MODEL_DATA, false),
  setShowModelData: (value: boolean) => setStorageItem(STORAGE_KEYS.SHOW_MODEL_DATA, value),

  // UI
  getTheme: () => getStorageItem<'light' | 'dark'>(STORAGE_KEYS.THEME, 'dark'),
  setTheme: (value: 'light' | 'dark') => setStorageItem(STORAGE_KEYS.THEME, value),

  getPanelSizes: () => getStorageItem<number[]>(STORAGE_KEYS.PANEL_SIZES, []),
  setPanelSizes: (value: number[]) => setStorageItem(STORAGE_KEYS.PANEL_SIZES, value),
};
