<script lang="ts" setup></script>

<template>
  <nav class="relative z-10 flex justify-between items-center p-4 md:p-6">
    <router-link to="/">
      <h1 class="text-black text-xl font-semibold">PlantAI</h1>
    </router-link>
    <div class="flex gap-4">
      <router-link to="/history">
        <button class="text-black rounded-full px-4 py-2">History</button>
      </router-link>
      <router-link to="/upload">
        <button class="bg-white text-black px-4 py-2 rounded-full hover:bg-gray-200 transition">
          Upload
        </button>
      </router-link>
    </div>
  </nav>
</template>

<style scoped></style>
