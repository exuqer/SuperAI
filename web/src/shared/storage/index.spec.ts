import { beforeEach, describe, expect, it } from 'vitest';

import { STORAGE_KEYS } from './keys';
import { storage } from './index';


describe('storage compatibility', () => {
  beforeEach(() => localStorage.clear());

  it('reads legacy raw identifiers', () => {
    localStorage.setItem(STORAGE_KEYS.ACTIVE_HIVE, 'hive-legacy');
    localStorage.setItem(STORAGE_KEYS.CONVERSATION_ID, 'conversation-legacy');

    expect(storage.getActiveHive()).toBe('hive-legacy');
    expect(storage.getConversationId()).toBe('conversation-legacy');
  });

  it('keeps identifiers in the original raw format', () => {
    storage.setActiveHive('hive-current');
    storage.setConversationId('conversation-current');

    expect(localStorage.getItem(STORAGE_KEYS.ACTIVE_HIVE)).toBe('hive-current');
    expect(localStorage.getItem(STORAGE_KEYS.CONVERSATION_ID)).toBe('conversation-current');
  });
});
