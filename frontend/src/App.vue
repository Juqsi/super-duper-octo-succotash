<script setup lang="ts">
import { RouterView } from 'vue-router'
import ImageUpload from "@/components/imageUpload.vue";
import {Toaster} from "vue-sonner";
</script>

<template>
  <ImageUpload />
  <RouterView />
  <Toaster rich-colors />
</template>
