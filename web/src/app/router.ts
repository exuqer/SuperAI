import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: { name: 'run' },
  },
  {
    path: '/run',
    name: 'run',
    component: () => import('@/pages/run/RunPage.vue'),
    meta: { title: 'Запуск' },
  },
  {
    path: '/dialogue/:conversationId?/:projectId?',
    name: 'dialogue',
    component: () => import('@/pages/dialogue/DialoguePage.vue'),
    meta: { title: 'Диалог' },
  },
  {
    path: '/traces/:traceId?',
    name: 'traces',
    component: () => import('@/pages/traces/TracesPage.vue'),
    meta: { title: 'Трассы' },
  },
  {
    path: '/hives/:hiveId?',
    name: 'hive',
    component: () => import('@/pages/hive/HivePage.vue'),
    meta: { title: 'Улей' },
  },
  {
    path: '/storage/:artifactId?',
    name: 'storage',
    component: () => import('@/pages/storage/StoragePage.vue'),
    meta: { title: 'Хранилище' },
  },
  {
    path: '/cosmos',
    name: 'cosmos',
    component: () => import('@/pages/cosmos/CosmosPage.vue'),
    meta: { title: 'Космос' },
  },
  {
    path: '/system',
    name: 'system',
    component: () => import('@/pages/system/SystemPage.vue'),
    meta: { title: 'Система' },
  },
  {
    path: '/ai-model',
    name: 'ai-model',
    component: () => import('@/pages/ai-model/AIModelPage.vue'),
    meta: { title: 'Модель ИИ' },
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: { name: 'run' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior: () => ({ top: 0 }),
})

router.afterEach((to) => {
  const title = typeof to.meta.title === 'string' ? to.meta.title : 'SuperAI'
  document.title = title + ' · SuperAI'
})

export default router
