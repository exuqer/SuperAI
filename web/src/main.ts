import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './app/App.vue'
import router from './app/router'
import './shared/styles/main.scss'

createApp(App).use(createPinia()).use(router).mount('#app')
