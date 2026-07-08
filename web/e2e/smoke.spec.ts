import { expect, test } from '@playwright/test';

test('chat bubbles and training preview smoke', async ({ page }) => {
  let chatSessions: Array<Record<string, unknown>> = [];

  await page.route('**/api/config', async (route) => {
    await route.fulfill({
      json: {
        state_dir: '.semantic_ants',
        allow_network: false,
        autoload_builtin: true,
        checkpoint_version: 3,
        examples_seen: 0,
        last_result_id: null,
      },
    });
  });

  await page.route('**/api/jobs', async (route) => {
    await route.fulfill({ json: [] });
  });

  await page.route('**/api/jobs/*', async (route) => {
    const jobId = route.request().url().split('/').pop() ?? 'job-1';
    await route.fulfill({
      json: {
        job_id: jobId,
        name: 'learn',
        status: 'completed',
        created_at: 1_710_000_000,
        started_at: 1_710_000_001,
        finished_at: 1_710_000_002,
        result: { examples: 1 },
        error: null,
        traceback: null,
      },
    });
  });

  await page.route('**/api/chat/sessions', async (route) => {
    await route.fulfill({ json: chatSessions });
  });

  await page.route('**/api/chat/message', async (route) => {
    const payload = route.request().postDataJSON() as Record<string, unknown>;
    const text = String(payload.text ?? '');
    const response = 'Ассистент отвечает';
    const resultId = 'result-1';

    chatSessions = [
      {
        session_id: 'default',
        turn_count: 2,
        updated_at: 1_710_000_001,
        turns: [
          {
            role: 'user',
            text,
            result_id: resultId,
            concepts: ['/m/top/dialogue'],
            created_at: 1_710_000_000,
          },
          {
            role: 'assistant',
            text: response,
            result_id: resultId,
            concepts: ['/m/top/dialogue'],
            created_at: 1_710_000_001,
          },
        ],
      },
    ];

    await route.fulfill({
      json: {
        result: {
          result_id: resultId,
          input_text: text,
          lang: String(payload.lang ?? 'auto'),
          tokens: [text],
          activated_concepts: [
            {
              uri: '/m/top/dialogue',
              label: 'Общение',
              language: 'ru',
              layer: 0,
              layers: [0],
              active_layers: [0],
              score: 1.2,
              sources: ['input'],
            },
          ],
          routes: [],
          summary: 'Краткая сводка',
          response,
          sources: ['input'],
          session_id: 'default',
          context_turns: [],
          semantic_vector: {
            items: [{ uri: '/m/top/dialogue', label: 'Общение', layer: 0, layers: [0], active_layers: [0], score: 1.2 }],
            strength_vector: [3],
          },
          signal_trace: [],
        },
        graph: {
          nodes: [],
          edges: [],
          stats: { nodes: 0, edges: 0, signal_nodes: 0, signal_edges: 0 },
        },
        trace_interpretation: {
          summary: {},
          chains: [],
          active_edge_ids: [],
        },
      },
    });
  });

  await page.goto('/chat');
  await expect(page.getByText('session_id=default')).toBeVisible();
  await expect(page.getByText('Чат пуст.')).toBeVisible();

  await page.getByPlaceholder('Напишите сообщение...').fill('hello');
  await page.getByRole('button', { name: 'Отправить' }).click();

  await expect(page.locator('.bubble--user').first()).toContainText('hello');
  await expect(page.locator('.bubble--assistant').first()).toContainText('Ассистент отвечает');
  await expect(page.getByRole('heading', { name: 'Ответ' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Semantic vector' })).toBeVisible();

  await page.goto('/training');
  await expect(page.getByText('Конструктор примера')).toBeVisible();
  await expect(page.locator('pre')).toContainText('/m/top/dialogue');
  await expect(page.locator('pre')).toContainText('"kind":"qa_with_layers"');
});
