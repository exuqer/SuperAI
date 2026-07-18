/** Vue Router configuration with lazy-loaded routes. */

import { createRouter, createWebHistory } from 'vue-router';

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'chat',
      component: () => import('@/pages/ChatPage.vue'),
    },
    {
      path: '/space',
      name: 'universe',
      component: () => import('@/pages/UniversePage.vue'),
    },
    {
      path: '/field',
      redirect: { name: 'universe' },
    },
    {
      path: '/analytics',
      redirect: { name: 'universe' },
    },
  ],
});

export default router;
