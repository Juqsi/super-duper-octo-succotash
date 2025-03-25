<script lang="ts" setup>
import { Search } from 'lucide-vue-next'
import { ref } from 'vue'
import router from '@/router'

const name = ref<string>('')

const emit = defineEmits<{
  (e: 'submit'): void
}>()

const submitSearch = () => {
  router.push({ name: 'SearchPlant', query: { name: name.value } }).then(() => {
    emit('submit')
    name.value = ''
  })
}
</script>

<template>
  <div
    class="w-full max-w-xl flex items-center bg-white/80 backdrop-blur rounded-full overflow-hidden shadow"
  >
    <input
      v-model="name"
      class="flex-grow px-5 py-3 text-gray-800 bg-transparent placeholder-gray-600 focus:outline-none"
      placeholder="Search for a plant..."
      type="text"
      @keyup.enter="submitSearch"
    />
    <button
      class="bg-primary hover:bg-green-700 text-white p-3 rounded-full m-1 transition"
      @click="submitSearch"
    >
      <Search />
    </button>
  </div>
</template>

<style scoped></style>
