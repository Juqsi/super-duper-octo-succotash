<script lang="ts" setup>
import { RouterView } from 'vue-router'
import { Toaster } from 'vue-sonner'
</script>

<template>
  <div class="min-h-svh bg-gradient-to-br from-green-100 to-blue-100">
    <RouterView />
  </div>

  <Toaster position="bottom-center" richColors />
</template>
