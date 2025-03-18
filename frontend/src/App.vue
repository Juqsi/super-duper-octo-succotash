<script setup lang="ts">
import { RouterView } from 'vue-router'
import ImageUpload from "@/components/imageUpload.vue";
import {Toaster} from "vue-sonner";
import CapturePhotoComponent from "@/components/CapturePhotoComponent.vue";
</script>

<template>
  <ImageUpload />
  <RouterView />
  <CapturePhotoComponent />
  <Toaster rich-colors />
</template>
