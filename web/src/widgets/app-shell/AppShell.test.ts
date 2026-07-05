import { shallowMount, RouterLinkStub } from '@vue/test-utils';
import { reactive } from 'vue';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import AppShell from './AppShell.vue';

const { runtimeState } = vi.hoisted(() => ({
  runtimeState: {
    config: { state_dir: '.semantic_ants' },
    error: null as string | null,
    loadConfig: vi.fn(async () => undefined),
  },
}));

const runtimeMock = reactive(runtimeState);

vi.mock('@/app/stores/runtime', () => ({
  useRuntimeStore: () => runtimeMock,
}));

describe('AppShell', () => {
  beforeEach(() => {
    runtimeState.loadConfig.mockReset();
  });

  it('shows the understand page in the sidebar menu', () => {
    const wrapper = shallowMount(AppShell, {
      global: {
        stubs: {
          RouterLink: RouterLinkStub,
        },
      },
    });

    expect(wrapper.text()).toContain('Пониматель');
    const links = wrapper.findAllComponents(RouterLinkStub);
    expect(links.some((link) => link.props('to') === '/understand')).toBe(true);
  });
});
