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
      component: () => import(HistoryView),
    },
    {
      path: '/upload',
      name: 'upload',
      component: () => import(UploadPhotoView),
    },
    {
      path: '/last/:number',
      name: 'Plant',
      component: () => import(PlantInformationView),
    },
    {
      path: '/:pathMatch(.*)*',
      name: 'NotFound',
      component: () => import('../views/404view.vue'),
    },
  ],
})

export default router
