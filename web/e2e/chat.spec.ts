import { expect, test } from '@playwright/test';


const hive = {
  id: 'hive-e2e',
  space_id: 2,
  query_text: '',
  query_json: {},
  max_cells: 24,
  created_at: '2026-07-14T12:00:00Z',
  updated_at: '2026-07-14T12:00:00Z',
};

const state = { hive, cells: [], messages: [] };

test.describe('Hive chat', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => localStorage.clear());
    await page.route('**/api/v2/hives', route => route.fulfill({ json: state }));
    await page.route('**/api/v2/hives/hive-e2e/query', route => route.fulfill({
      json: {
        ...state,
        message_id: 'message-e2e',
        decision: {
          decision: 'MISS',
          external_search_required: true,
          reasons: ['empty local memory'],
          matches: [],
        },
        resonance_events: [],
        external_search: { sources: [], bees: [], iterations: 0, anchors: [] },
        merge_results: [],
        metrics: { bees: 0, iterations: 0, activated_cells: 0, merged_cells: 0 },
      },
    }));
    await page.route('**/api/v2/hives/hive-e2e/reasoning/export**', route => route.fulfill({
      json: { schema_version: 2, hive: { id: 'hive-e2e' } },
    }));
    await page.goto('/');
  });

  test('creates a session and sends a message', async ({ page }) => {
    await page.locator('textarea').fill('Кот ест рыбу');
    await page.getByRole('button', { name: 'Отправить' }).click();

    await expect(page.locator('.message.user .bubble')).toHaveText('Кот ест рыбу');
    await expect(page.getByText(/Улей активировал известные компоненты/)).toBeVisible();
  });

  test('opens the JSON export dialog', async ({ page }) => {
    await page.getByRole('button', { name: 'Показать JSON' }).click();

    await expect(page.getByRole('dialog')).toBeVisible();
    await expect(page.getByRole('dialog')).toContainText('"schema_version": 2');
  });
});
