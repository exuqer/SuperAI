import { expect, test } from '@playwright/test'

test.skip('mock task opens its trace and linked Hive view', async ({ page }) => {
  await page.goto('/run')
  await page.getByRole('button', { name: 'Запустить задачу' }).click()

  await expect(page.getByText('task-success-001')).toBeVisible()
  await page.getByRole('link', { name: 'Открыть трассу' }).click()
  await expect(page.getByRole('heading', { name: 'Трассы выполнения' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'trace-success-001' })).toBeVisible()

  await page.getByRole('link', { name: 'Улей' }).first().click()
  await expect(page.getByRole('heading', { name: 'Улей' })).toBeVisible()
})
