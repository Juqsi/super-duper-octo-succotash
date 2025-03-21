<script lang="ts" setup>
import { RouterView } from 'vue-router'
import { Toaster } from 'vue-sonner'
</script>

<template>
  <div class="w-screen p-2 flex flex-col items-center justify-center mt-4">
    <RouterView />
  </div>

  <Toaster position="top-center" richColors />
</template>
