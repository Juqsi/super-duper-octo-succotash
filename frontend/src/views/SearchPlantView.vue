<script lang="ts" setup>
import { onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import NavBar from '@/components/NavBar.vue'
import SearchBar from '@/components/SearchBar.vue'
import PlantInformationCard from '@/components/PlantInformationCard.vue'
import { useSearch } from '@/composable/useSearch.ts'
import { type Plant, type Recognition } from '@/stores/usePlantHistory'

const route = useRoute()

const searchQuery = ref<string>('')

const { searchPlant, isLoading, error } = useSearch()

const searchResults = ref<Plant[]>([])

const submitSearch = async () => {
  searchQuery.value = (route.query.name as string) || ''
  if (searchQuery.value && searchQuery.value.length > 0) {
    const data = await searchPlant(searchQuery.value)
    searchResults.value = data || []
  }
}

onMounted(async () => {
  await submitSearch()
})
</script>

<template>
  <NavBar />
  <div class="w-full max-w-2xl mx-auto px-4">
    <SearchBar class="max-w-xl" @submit="submitSearch" />

    <div class="py-4">
      <h1 class="text-3xl font-bold mb-4">Search Results: {{ searchQuery }}</h1>

      <div v-if="isLoading" class="text-center">Loading...</div>
      <div v-if="error" class="text-center text-red-600">{{ error }}</div>
      <div v-if="!isLoading && searchResults.length === 0" class="text-center text-gray-600">
        No results found.
      </div>
      <div v-else class="grid gap-6">
        <div v-for="(rec, index) in searchResults" :key="index">
          <PlantInformationCard
            :recognition="
              {
                plant: rec,
                name: rec.scientific_name,
                wikipedia: '',
              } as Recognition
            "
          />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped></style>
