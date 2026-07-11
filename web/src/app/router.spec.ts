import { describe, expect, it } from 'vitest'

import router from './router'

describe('application routes', () => {
  it('exposes every client-first diagnostic page', () => {
    const names = new Set(router.getRoutes().map((route) => route.name))

    for (const name of ['run', 'traces', 'hive', 'storage', 'cosmos', 'system']) {
      expect(names.has(name)).toBe(true)
    }
  })
})
