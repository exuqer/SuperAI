import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import StatusBadge from './StatusBadge.vue'

describe('StatusBadge', () => {
  it('renders a readable status and semantic variant', () => {
    const wrapper = mount(StatusBadge, {
      props: { status: 'dead_letter' },
    })

    expect(wrapper.text()).toContain('dead letter')
    expect(wrapper.classes()).toContain('status-badge--dead-letter')
  })
})
