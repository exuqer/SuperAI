import { createRouter, createWebHistory } from 'vue-router';
import ChatPage from '@/pages/chat/ChatPage.vue';
import TrainingPage from '@/pages/training/TrainingPage.vue';
import LayersPage from '@/pages/layers/LayersPage.vue';
import ConceptsPage from '@/pages/concepts/ConceptsPage.vue';
import GraphPage from '@/pages/graph/GraphPage.vue';
import MemoryPage from '@/pages/memory/MemoryPage.vue';
import SystemPage from '@/pages/system/SystemPage.vue';

export const routes = [
  { path: '/', redirect: '/chat' },
  { path: '/chat', component: ChatPage, meta: { title: 'Диалог' } },
  { path: '/training', component: TrainingPage, meta: { title: 'Обучение' } },
  { path: '/layers', component: LayersPage, meta: { title: 'Слои' } },
  { path: '/concepts', component: ConceptsPage, meta: { title: 'Понятия' } },
  { path: '/graph', component: GraphPage, meta: { title: 'Граф' } },
  { path: '/memory', component: MemoryPage, meta: { title: 'Память' } },
  { path: '/system', component: SystemPage, meta: { title: 'Система' } },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});
