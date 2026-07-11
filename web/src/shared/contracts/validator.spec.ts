import { describe, expect, it } from 'vitest'

import {
  ContractValidationError,
  parseArtifactRefDto,
  parseClaimDtos,
  parseConceptDtos,
  parseHiveViewDto,
  parseTaskSubmissionDto,
  parseTaskViewDto,
  parseTraceDto,
} from './validator'
import { getFixture } from '@/shared/mocks/fixtures'

describe('public API DTO validator', () => {
  it('accepts the v1 success fixture using actual live endpoint shapes', () => {
    const fixture = getFixture('success')

    expect(parseTaskSubmissionDto(fixture.request).message).toContain('Unity')
    expect(parseTaskViewDto(fixture.task).contract?.budget.time_ms).toBe(4_000)
    expect(parseTraceDto(fixture.trace).events).toHaveLength(2)
    expect(parseHiveViewDto(fixture.hive).entries[0]?.store_name).toBe('GoalStore')
    expect(parseArtifactRefDto(fixture.artifacts?.[0]).access_scope.visibility).toBe('project')
    expect(parseConceptDtos(fixture.concepts)).toHaveLength(2)
    expect(parseClaimDtos(fixture.claims)).toHaveLength(1)
  })

  it('keeps accepted forward-compatible fields out of the UI contract decision', () => {
    const fixture = getFixture('success')
    const task = {
      ...fixture.task,
      future_server_field: { safe_to_ignore: true },
    }

    expect(parseTaskViewDto(task).task_id).toBe('task-success-001')
  })

  it('rejects an unknown schema major before it reaches a page', () => {
    const fixture = getFixture('incompatible')

    expect(() => parseTaskViewDto(fixture.task)).toThrow(ContractValidationError)
    expect(() => parseTraceDto(fixture.trace)).toThrow(/несовместима/)
  })
})
