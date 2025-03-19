<script lang="ts" setup>
import { RouterView } from 'vue-router'
import { Toaster } from 'vue-sonner'
</script>

<template>
  <RouterView class="w-full p-2" />
  <Toaster position="top-center" richColors />
</template>
