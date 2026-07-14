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
      path: '/field',
      name: 'training',
      component: () => import('@/pages/TrainingPage.vue'),
    },
  ],
});

export default router;