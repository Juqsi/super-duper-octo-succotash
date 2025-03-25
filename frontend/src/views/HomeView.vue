<script lang="ts" setup>
import SearchBar from '@/components/SearchBar.vue'
</script>
<template>
  <div
    class="min-h-svh bg-cover bg-center"
    style="background-image: url('/cactus-7856967_1920.jpg')"
  >
    <div class="absolute inset-0 bg-black/50 backdrop-blur-sm"></div>

    <div class="relative z-10 flex justify-between items-center p-4 md:p-6">
      <router-link to="/">
        <h1 class="text-white text-xl font-semibold">PlantAI</h1>
      </router-link>
      <div class="flex gap-4">
        <router-link to="/history">
          <button class="text-white rounded-full px-4 py-2">History</button>
        </router-link>
        <router-link to="/upload">
          <button class="bg-white text-black px-4 py-2 rounded-full hover:bg-gray-200 transition">
            Upload
          </button>
        </router-link>
      </div>
    </div>

    <div
      class="relative z-10 flex flex-col items-center justify-center text-center px-4 pt-20 sm:pt-32"
    >
      <h2 class="text-4xl sm:text-5xl font-bold text-white mb-4 drop-shadow">
        Discover Plants Instantly
      </h2>
      <p class="text-lg sm:text-xl text-white mb-8 drop-shadow">
        Identify plants with a photo or search by name
      </p>

      <SearchBar class="max-w-md" />
    </div>

    <div class="absolute bottom-2 right-4 text-xs text-white/80 z-10">
      Photo by
      <a
        class="underline hover:text-white"
        href="https://pixabay.com/de/users/ignartonosbg-21428489/"
        target="_blank"
        >ignartonosbg</a
      >
      on Pixabay
    </div>
  </div>
</template>
