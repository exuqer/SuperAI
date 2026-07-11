import { describe, expect, it } from 'vitest'

import {
  toUiCosmosConcepts,
  toUiHive,
  toUiTask,
  toUiTrace,
} from './ui-models'
import { getFixture } from '@/shared/mocks/fixtures'

describe('transport to UI mappers', () => {
  it('reads nested TaskView.contract instead of invented task-level fields', () => {
    const fixture = getFixture('success')
    const task = toUiTask(fixture.task!)

    expect(task.conversationId).toBe('conversation-unity-001')
    expect(task.budget?.timeLimitMs).toBe(4_000)
    expect(task.answer?.text).toContain('Is Trigger')
  })

  it('uses TraceSpan summaries and events from the real trace payload', () => {
    const fixture = getFixture('retry')
    const trace = toUiTrace(fixture.trace!, toUiTask(fixture.task!))

    expect(trace.spans[0]?.input).toEqual({})
    expect(trace.events[0]?.kind).toBe('CommandRetryScheduled')
    expect(trace.status).toBe('failed')
  })

  it('derives Hive stores and warm eviction view from HiveView.entries', () => {
    const fixture = getFixture('success')
    const hive = toUiHive(fixture.hive!)

    expect(hive.stores.find((store) => store.storeId === 'GoalStore')?.protectedCount).toBe(1)
    expect(hive.evictedItems[0]?.destination).toBe('warm')
  })

  it('joins concepts and claims without treating an index as canonical data', () => {
    const fixture = getFixture('success')
    const concepts = toUiCosmosConcepts(fixture.concepts ?? [], fixture.claims ?? [])

    expect(concepts[0]?.claims[0]?.sourceArtifactId).toBe('artifact-unity-docs-001')
  })
})
