<script lang="ts" setup>
import { computed, ref } from 'vue'
import { usePlantHistory } from '@/stores/usePlantHistory'
import PlantInformationCard from '@/components/PlantInformationCard.vue'
import { useRoute } from 'vue-router'
import NavBar from '@/components/NavBar.vue'
import EmptyState from '@/components/EmptyState.vue'

const plantHistory = usePlantHistory()
const route = useRoute()
const length = Number(route.params.number) || 1
const recognizedImages = computed(() => plantHistory.history.slice(0, length))

const selectedImageIndex = ref(0)

const getImage = (imageKey: string) => {
  return localStorage.getItem(imageKey) || ''
}

const selectedImage = computed(() => recognizedImages.value[selectedImageIndex.value])
</script>

<template>
  <NavBar />
  <div class="min-h-screen p-4 bg-gradient-to-br from-green-100 to-blue-100">
    <h1 class="mx-auto text-3xl font-bold mb-4 max-w-2xl">Recognized plants</h1>

    <div class="flex flex-wrap justify-start gap-4 mb-6 overflow-x-auto max-w-2xl mx-auto">
      <div
        v-for="(entry, index) in recognizedImages"
        :key="entry.timestamp"
        :class="{
          'border-primary': selectedImageIndex === index,
          'border-foreground': selectedImageIndex !== index,
        }"
        class="w-20 h-20 border-2 rounded-md overflow-hidden cursor-pointer"
        @click="selectedImageIndex = index"
      >
        <img
          :src="getImage(entry.imageKey)"
          alt="Scanned plant"
          class="w-full h-full object-cover"
        />
      </div>
    </div>

    <div v-if="selectedImage" class="grid gap-6 justify-items-center">
      <h2 class="text-xl font-semibold mb-2 w-full max-w-2xl">
        Image from {{ new Date(selectedImage.timestamp).toLocaleString() }}
      </h2>
      <div class="grid gap-4 w-full justify-items-center max-w-full">
        <template v-for="(rec, index) in selectedImage.recognitions" :key="index">
          <PlantInformationCard :recognition="rec" />
        </template>
        <empty-state
          :condition="selectedImage.recognitions.length === 0"
          img-src="/undraw_flowers_171u.svg"
          subtitle="try again with a different picture"
          title="No plant recognized"
        />
      </div>
    </div>
  </div>
</template>

<style scoped></style>
