import { createRouter, createWebHistory } from 'vue-router'
import UploadPhotoView from '@/views/UploadPhotoView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: UploadPhotoView,
    },
  ],
})

export default router
