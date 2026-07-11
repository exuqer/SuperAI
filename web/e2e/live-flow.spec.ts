import { expect, test } from '@playwright/test'

const liveBaseUrl = process.env.SUPERAI_LIVE_BASE_URL

test.skip(!liveBaseUrl, 'Set SUPERAI_LIVE_BASE_URL to run against a local API.')

test('live task uses the same TaskSubmission and TaskView transport shape', async ({ page }) => {
  await page.goto('/run')
  const healthProbe = await page.evaluate(async (baseUrl) => {
    try {
      const response = await fetch(baseUrl + '/api/v1/health')
      return { ok: response.ok, status: response.status }
    } catch (error) {
      return { ok: false, error: String(error) }
    }
  }, liveBaseUrl!)
  expect(healthProbe.ok).toBe(true)

  await page.getByLabel('Сообщение пользователя').fill('Сформируй краткий ответ из разрешённого контекста.')
  await page.getByLabel('Conversation ID').fill('conversation-live-browser')
  await page.getByLabel('Project ID (необязательно)').fill('project-live-browser')
  await page.getByRole('button', { name: 'Запустить задачу' }).click()

  await expect(page.getByText('Task result')).toBeVisible()
  await expect(page.getByRole('link', { name: 'Открыть трассу' })).toBeVisible()
  await page.getByRole('link', { name: 'Открыть трассу' }).click()
  await expect(page.getByRole('heading', { name: 'Трассы выполнения' })).toBeVisible()
})
