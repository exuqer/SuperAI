import { createRouter, createWebHistory } from 'vue-router'
import TrainingView from '@/views/TrainingView.vue'
import ChatView from '@/views/ChatView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'chat',
      component: ChatView,
    },
    {
      path: '/field',
      name: 'training',
      component: TrainingView,
    },
  ],
})

export default router
