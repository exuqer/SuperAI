import { createRouter, createWebHistory } from 'vue-router';
import ChatPage from '@/pages/chat/ChatPage.vue';
import TrainingPage from '@/pages/training/TrainingPage.vue';
import GraphPage from '@/pages/graph/GraphPage.vue';

export const routes = [
  { path: '/', redirect: '/chat' },
  { path: '/chat', component: ChatPage, meta: { title: 'Диалог' } },
  { path: '/training', component: TrainingPage, meta: { title: 'Обучение' } },
  { path: '/graph', component: GraphPage, meta: { title: 'Граф' } },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});
