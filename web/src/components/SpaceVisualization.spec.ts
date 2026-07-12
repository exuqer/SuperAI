import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import SpaceVisualization from './SpaceVisualization.vue'

describe('SpaceVisualization', () => {
  it('renders without crashing when no words', () => {
    const wrapper = mount(SpaceVisualization, {
      props: {
        words: [],
        width: 800,
        height: 500,
      },
    })
    expect(wrapper.find('svg').exists()).toBe(true)
  })

  it('renders word nodes when words provided', () => {
    const words = [
      { word: 'test', mass: 1.0, x: 100, y: 100 },
      { word: 'hello', mass: 2.0, x: 200, y: 200 },
    ]
    const wrapper = mount(SpaceVisualization, {
      props: {
        words,
        width: 800,
        height: 500,
      },
    })
    const nodes = wrapper.findAll('.word-node')
    expect(nodes).toHaveLength(2)
  })

  it('shows legend', () => {
    const wrapper = mount(SpaceVisualization, {
      props: {
        words: [],
        width: 800,
        height: 500,
      },
    })
    expect(wrapper.find('.legend').exists()).toBe(true)
  })
})