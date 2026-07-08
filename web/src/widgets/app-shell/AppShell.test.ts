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

  it('shows only the experiment pages in the sidebar menu', () => {
    const wrapper = shallowMount(AppShell, {
      global: {
        stubs: {
          RouterLink: RouterLinkStub,
        },
      },
    });

    expect(wrapper.text()).toContain('Диалог');
    expect(wrapper.text()).toContain('Обучение');
    expect(wrapper.text()).toContain('Граф');
    const links = wrapper.findAllComponents(RouterLinkStub);
    expect(links.map((link) => link.props('to'))).toEqual(['/chat', '/training', '/graph']);
  });
});
