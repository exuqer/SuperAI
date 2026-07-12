import { createRouter, createWebHistory } from 'vue-router'
import TrainingView from '@/views/TrainingView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'training',
      component: TrainingView,
    },
  ],
})

export default router