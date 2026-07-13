import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import SpaceVisualization from './SpaceVisualization.vue'

describe('SpaceVisualization', () => {
  it('renders without crashing when no words', () => {
    const wrapper = mount(SpaceVisualization, {
      props: {
        concepts: [],
        width: 800,
        height: 500,
      },
    })
    expect(wrapper.find('svg').exists()).toBe(true)
  })

  it('renders word nodes when words provided', () => {
    const concepts = [
      { id: 1, token: 'test', mass: 1.0, radius: 34, activation: 1, position: [100, 100] },
      { id: 2, token: 'hello', mass: 2.0, radius: 39, activation: 1, position: [200, 200] },
    ]
    const wrapper = mount(SpaceVisualization, {
      props: {
        concepts,
        width: 800,
        height: 500,
      },
    })
    const nodes = wrapper.findAll('.concept-node')
    expect(nodes).toHaveLength(2)
    expect(wrapper.findAll('line')).toHaveLength(0)
  })

  it('shows legend', () => {
    const wrapper = mount(SpaceVisualization, {
      props: {
        concepts: [],
        width: 800,
        height: 500,
      },
    })
    expect(wrapper.find('.legend').exists()).toBe(true)
  })
})
