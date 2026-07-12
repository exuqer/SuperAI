import { test, expect } from '@playwright/test'

test.describe('Training View', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('loads the training page', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('SuperAI')
  })

  test('has textarea for input', async ({ page }) => {
    await expect(page.locator('textarea')).toBeVisible()
  })

  test('has train and reset buttons', async ({ page }) => {
    await expect(page.locator('button:has-text("Обучить")')).toBeVisible()
    await expect(page.locator('button:has-text("Сбросить")')).toBeVisible()
  })
})