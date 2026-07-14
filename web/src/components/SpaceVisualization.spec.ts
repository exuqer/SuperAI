import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import SpaceVisualization from './SpaceVisualization.vue'

const space = { id: 1, space_type: 'scene_space' as const, owner_cloud_id: 7, parent_space_id: 2, random_seed: 1 }
const cloud = { id: 3, cloud_type: 'word_form' as const, canonical_name: 'кот', mass: 1, density: 1, stability: .2, base_activation: 0, observation_count: 1, metadata_json: '{}' }
const placement = { id: 11, cloud_id: 3, space_id: 1, x: 100, y: 120, z: null, radius: 20, local_activation: 1, local_density: 1, local_gravity: 0, local_stability_modifier: 0, metadata_json: '{}' }

describe('SpaceVisualization', () => {
  it('renders normalized placements', () => {
    const wrapper = mount(SpaceVisualization, { props: { space, clouds: { 3: cloud }, placements: [placement] } })
    expect(wrapper.findAll('.placement-node')).toHaveLength(1)
    expect(wrapper.findAll('.density-field circle')).toHaveLength(1)
    expect(wrapper.find('radialGradient[id="continuum-word_form"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('кот')
  })

  it('emits placement identity instead of cloud identity', async () => {
    const wrapper = mount(SpaceVisualization, { props: { space, clouds: { 3: cloud }, placements: [placement] } })
    await wrapper.find('.placement-node').trigger('click')
    expect(wrapper.emitted('select-placement')?.[0]).toEqual([11])
  })
})
