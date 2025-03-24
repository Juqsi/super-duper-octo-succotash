import { createRouter, createWebHistory } from 'vue-router'
import UploadPhotoView from '@/views/UploadPhotoView.vue'
import PlantInformationView from '@/views/PlantInformationView.vue'
import HomeView from '@/views/HomeView.vue'
import HistoryView from '@/views/HistoryView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
    },
    {
      path: '/history',
      name: 'history',
      component: HistoryView,
    },
    {
      path: '/upload',
      name: 'upload',
      component: UploadPhotoView,
    },
    {
      path: '/last/:number',
      name: 'Plant',
      component: PlantInformationView,
    },
  ],
})

export default router
