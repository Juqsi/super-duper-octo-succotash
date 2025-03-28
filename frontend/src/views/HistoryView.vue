<script lang="ts" setup>
import { computed } from 'vue'
import { usePlantHistory } from '@/stores/usePlantHistory'
import NavBar from '@/components/NavBar.vue'
import EmptyState from '@/components/EmptyState.vue'

const plantHistory = usePlantHistory()
const recognizedImages = computed(() => plantHistory.history)

const clearHistory = () => {
  plantHistory.clearHistory()
}

const getImage = (imageKey: string) => {
  return localStorage.getItem(imageKey) || ''
}

const formatTimestamp = (timestamp: number) => {
  return new Date(timestamp).toLocaleString()
}
</script>

<template>
  <div>
    <NavBar />

    <div class="min-h-screen p-4 bg-gradient-to-br from-green-100 to-blue-100">
      <div class="flex justify-between items-center mb-6 max-w-2xl mx-auto">
        <h1 class="text-3xl font-bold">History</h1>
        <button
          class="bg-red-500 text-white px-4 py-2 rounded-full hover:bg-red-600 transition"
          @click="clearHistory"
        >
          Delete history
        </button>
      </div>

      <div v-if="recognizedImages.length" class="grid gap-6 justify-items-center">
        <div
          v-for="entry in recognizedImages"
          :key="entry.timestamp"
          class="border p-4 rounded-lg bg-white shadow w-full max-w-2xl"
        >
          <div class="flex items-center mb-4">
            <img
              :src="getImage(entry.imageKey)"
              alt="Bild Thumbnail"
              class="w-16 h-16 rounded mr-4 object-cover"
            />
            <div>
              <p class="text-sm text-gray-600">
                {{ formatTimestamp(entry.timestamp) }}
              </p>
            </div>
          </div>
          <empty-state
            :condition="entry.recognitions.length === 0"
            img-src="/undraw_flowers_171u.svg"
            subtitle="try again with a different picture"
            title="No plant recognized"
          />
          <div v-if="entry.recognitions && entry.recognitions.length > 0">
            <h2 class="text-lg font-bold mb-2">Recognized plants:</h2>
            <div class="space-y-2">
              <div
                v-for="(rec, index) in entry.recognitions"
                :key="index"
                class="border p-2 rounded"
              >
                <router-link
                  :to="'/search?name=' + (rec.plant?.scientific_name ?? rec.name.replace('_', ' '))"
                >
                  <p><strong>Name:</strong> {{ rec.name.replace('_', ' ') }}</p>
                  <p>
                    <strong>Probability:</strong>
                    {{ rec.probability ? rec.probability.toFixed(2) + '%' : '' }}
                  </p>
                  <div v-if="rec.plant">
                    <p><strong>Common Name:</strong> {{ rec.plant.common_name }}</p>
                    <p>
                      <strong>Scientific Name:</strong>
                      {{ rec.plant.scientific_name }}
                    </p>
                  </div>
                  <div v-else>
                    <p>Plant information not available.</p>
                  </div>
                </router-link>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div v-else class="text-center text-gray-600">No history available.</div>
    </div>
  </div>
</template>

<style scoped></style>
