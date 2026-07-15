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

const analytics = {
  hive_id: 'hive-e2e',
  runs: [
    { id: 'run-new', hive_id: 'hive-e2e', status: 'COMPLETED', reasoning_steps: 1, completed_steps: 1, stop_reason: 'COMPLETED', random_seed: 1, created_at: '2026-07-14T12:01:00Z', completed_at: '2026-07-14T12:01:01Z', query: { terms: ['кто', 'ловит', 'рыбу'], roles: ['subject', 'predicate', 'object'] }, config: {} },
    { id: 'run-old', hive_id: 'hive-e2e', status: 'COMPLETED', reasoning_steps: 1, completed_steps: 1, stop_reason: 'COMPLETED', random_seed: 1, created_at: '2026-07-14T12:00:00Z', completed_at: '2026-07-14T12:00:01Z', query: { terms: ['кто', 'ловит', 'рыбу'], roles: ['subject', 'predicate', 'object'] }, config: {} },
  ],
  primary: null,
  comparison: null,
};

function analyticsRun(run: typeof analytics.runs[number], answer: string) {
  const snapshot = (step: number) => ({
    step, phase: step ? 'AFTER_SETTLE' : 'INITIAL', created_at: run.created_at, temperature: step ? .72 : 1,
    metrics: { average_activation: .62, average_retention: .47, total_energy: 4.4, active_nodes: 1, weakening_nodes: 0, evicted_nodes: 0 },
    nodes: [{ placement_id: 1, cell_id: 'cell-fisher', cloud_id: 20, node_type: 'scene', label: 'рыбак ловит рыбу', local_activation: .62, local_gravity: .64, retention: .47, energy: .73, eviction_status: 'ACTIVE' }],
    candidates: [{ placement_id: 1, cell_id: 'cell-fisher', scene_cloud_id: 20, scene_label: 'рыбак ловит рыбу', answer, matched_components: [], answer_components: [], semantic_score: 1, dynamic_score: .57, viability: 1, candidate_score: .87, eviction_status: 'ACTIVE', explanation: 'ответ извлечён из роли подлежащее' }], delta: {}, events: [],
  });
  return { run, query_components: [{ term: 'кто', role: 'subject', word_form_cloud_id: null }], snapshots: [snapshot(0), snapshot(1)], events: [], clusters: [] };
}

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
    await page.route('**/api/v2/hives/hive-e2e/hierarchy', route => route.fulfill({
      json: {
        schema_version: 2,
        hive,
        projection: { space_type: 'hive', scope: 'bounded_field_projection', source_space_id: 2, parent_projection_id: null, parent_node_id: null, depth: 0, capacity: 24 },
        cells: [],
        subspaces: [],
        generation_candidates: [],
        inspection_projections: [],
      },
    }));
    await page.route('**/api/v2/hives/hive-e2e/analytics**', route => route.fulfill({
      json: { ...analytics, primary: analyticsRun(analytics.runs[0], 'рыбак'), comparison: analyticsRun(analytics.runs[1], 'кот') },
    }));
    await page.goto('/');
  });

  test('creates a session and sends a message', async ({ page }) => {
    await page.locator('textarea').fill('Кот ест рыбу');
    await page.getByRole('button', { name: 'Отправить' }).click();

    await expect(page.locator('.message.user .bubble')).toHaveText('Кот ест рыбу');
    await expect(page.getByText('Подходящий ответ в доступной памяти не найден.')).toBeVisible();
  });

  test('puts the full resolved answer into chat', async ({ page }) => {
    await page.route('**/api/v2/hives/hive-e2e/query', route => route.fulfill({
      json: {
        ...state,
        message_id: 'message-full-answer',
        resolved_mode: 'NEW_QUERY',
        decision: { decision: 'ROLE_HIT', external_search_required: false, reasons: [], matches: [] },
        resonance_events: [],
        external_search: { sources: [], bees: [], iterations: 0, anchors: [] },
        merge_results: [],
        metrics: {},
        answer: {
          status: 'RESOLVED', answer_mode: 'exact', confidence: .99,
          surface_answer: 'Рыбу.', full_surface_answer: 'Продают рыбу на рынке.',
        },
      },
    }));

    await page.locator('textarea').fill('Что там на рынке?');
    await page.getByRole('button', { name: 'Отправить' }).click();

    await expect(page.locator('.message.assistant .bubble')).toHaveText('Продают рыбу на рынке.');
  });

  test('opens the JSON export dialog', async ({ page }) => {
    await page.getByRole('button', { name: 'Показать JSON' }).click();

    await expect(page.getByRole('dialog')).toBeVisible();
    await expect(page.getByRole('dialog')).toContainText('"schema_version": 2');
  });

  test('opens analytics and drills into a hive cell', async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('superai-v2-active-hive', 'hive-e2e'));
    await page.route('**/api/v2/hives/hive-e2e', route => route.fulfill({
      json: {
        hive,
        messages: [],
        cells: [{
          id: 'cell-fisher', hive_id: 'hive-e2e', dominant_cloud_id: 20, hive_placement_id: 1,
          source_cloud_id: 20, source_placement_id: 1, source_space_id: 1, source_scene_cloud_id: 20,
          stored_strength: .8, retention: .47, local_activation: .62, component_class: 'core', metadata: {},
          created_at: hive.created_at, updated_at: hive.updated_at, label: 'рыбак ловит рыбу', x: 100, y: 100, gravity: .64, components: [],
        }],
      },
    }));
    await page.goto('/analytics');
    await expect(page.getByRole('heading', { name: 'Лаборатория улья' })).toBeVisible();
    await expect(page.locator('.candidate-row').first()).toContainText('рыбак');
    await page.getByRole('button', { name: 'Описание: Температура' }).first().hover();
    await expect(page.getByText(/Уровень исследования и шума при встряске/)).toBeVisible();
    await page.locator('.candidate-row').first().click();
    await expect(page).toHaveURL(/\/?cell=cell-fisher/);
    await expect(page.locator('.hive-inspector')).toContainText('рыбак ловит рыбу');
  });
});
