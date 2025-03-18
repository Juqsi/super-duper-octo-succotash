<script setup>
import { ref } from 'vue';
import { useImageUpload } from '@/composable/useImageUpload'
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const images = ref([]);
const imageFiles = ref([]);
const { uploadImages, isUploading, error } = useImageUpload("http://localhost:8000/upload");

const handleFileChange = (event) => {
  const files = event.target.files;
  if (files.length) {
    Array.from(files).forEach((file) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        images.value.push(e.target.result);
      };
      reader.readAsDataURL(file);
      imageFiles.value.push(file);
    });
  }
};

const submitImages = async () => {
  const response = await uploadImages(imageFiles.value);
};
</script>

<template>
  <div class="flex flex-col items-center gap-4 p-6">
    <Card class="w-full max-w-md p-4 border border-gray-200 shadow-md rounded-2xl">
      <CardContent class="flex flex-col gap-4">
        <input type="file" accept="image/*" multiple @change="handleFileChange" class="hidden" id="file-upload" />
        <label for="file-upload" class="cursor-pointer p-4 border border-dashed border-gray-300 rounded-xl text-gray-600 text-center hover:bg-gray-50">Bilder hochladen</label>
        <div v-if="images.length" class="grid grid-cols-3 gap-2">
          <img v-for="(image, index) in images" :key="index" :src="image" class="w-24 h-24 object-cover rounded-lg border" alt="hochgeladenes Foto"/>
        </div>
        <Button @click="submitImages" :disabled="isUploading" class="w-full">
          {{ isUploading ? "Hochladen..." : "Bilder absenden" }}
        </Button>
        <p v-if="error" class="text-red-500 text-sm mt-2">{{ error }}</p>
      </CardContent>
    </Card>
  </div>
</template>
