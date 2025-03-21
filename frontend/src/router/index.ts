import { createRouter, createWebHistory } from 'vue-router'
import UploadPhotoView from '@/views/UploadPhotoView.vue'
import PlantInformations from '@/components/PlantInformations.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: UploadPhotoView,
    },
    {
      path: '/plant/:id',
      name: 'Plant',
      component: PlantInformations,
    },
  ],
})

export default router
