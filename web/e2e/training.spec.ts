import { expect, test } from '@playwright/test'

const emptyField = {
  space: { id: 1, space_type: 'global_field', owner_cloud_id: null, parent_space_id: null, random_seed: 1337 },
  clouds: {},
  placements: [],
  stats: {
    clouds_total: 0,
    clouds_by_type: {},
    spaces_total: 1,
    spaces_by_type: { global_field: 1 },
    placements_total: 0,
    unique_word_forms: 0,
    scene_components_total: 0,
    structural_components_total: 0,
    concepts_total: 0,
  },
}

const emptyModel = {
  schema_version: 1,
  stats: emptyField.stats,
  model: {
    clouds: [], spaces: [], cloud_placements: [], structural_components: [], lexemes: [],
    word_forms: [], semantic_memberships: [], scenes: [], scene_components: [],
    training_runs: [], training_observations: [], training_change_events: [],
  },
}

test.describe('V2 Training View', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/v2/field', route => route.fulfill({ json: emptyField }))
    await page.route('**/api/v2/model', route => route.fulfill({ json: emptyModel }))
    await page.goto('/field')
  })

  test('loads the placement field', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Обучение модели' })).toBeVisible()
    await expect(page.getByRole('button', { name: /^Глобальное поле/ })).toBeVisible()
  })

  test('has V2 train and destructive reset controls', async ({ page }) => {
    await expect(page.locator('textarea')).toBeVisible()
    await expect(page.getByRole('button', { name: 'Обучить' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Очистить данные' })).toBeVisible()
  })

  test('shows a copyable trained-model object', async ({ page }) => {
    await page.getByRole('button', { name: 'Показать' }).click()
    await expect(page.getByLabel('Объект обученной модели')).toContainText('"schema_version": 1')
    await expect(page.getByRole('button', { name: 'Копировать JSON' })).toBeVisible()
  })
})
