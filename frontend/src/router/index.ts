import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'

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
      component: () => import('@/views/HistoryView.vue'),
    },
    {
      path: '/upload',
      name: 'upload',
      component: () => import('@/views/UploadPhotoView.vue'),
    },
    {
      path: '/last/:number',
      name: 'Plant',
      component: () => import('@/views/PlantInformationView.vue'),
    },
    {
      path: '/search',
      name: 'SearchPlant',
      component: () => import('@/views/SearchPlantView.vue'),
    },
    {
      path: '/:pathMatch(.*)*',
      name: 'NotFound',
      component: () => import('@/views/404View.vue'),
    },
  ],
})

export default router
