<script setup>
import { ref } from 'vue';
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const images = ref([]);

const handleFileChange = (event) => {
  const files = event.target.files;
  if (files.length) {
    Array.from(files).forEach((file) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        images.value.push(e.target.result);
      };
      reader.readAsDataURL(file);
    });
  }
};

const submitImages = () => {
  console.log(images.value);
};
</script>

<template>
  <div class="flex flex-col items-center gap-4 p-6">
    <Card class="w-full max-w-md p-4 border border-gray-200 shadow-md rounded-2xl">
      <CardContent class="flex flex-col gap-4">
        <input type="file" accept="image/*" multiple @change="handleFileChange" class="hidden" id="file-upload" />
        <label for="file-upload" class="cursor-pointer p-4 border border-dashed border-gray-300 rounded-xl text-gray-600 text-center hover:bg-gray-50">Bilder hochladen</label>
        <div v-if="images.length" class="grid grid-cols-3 gap-2">
          <img v-for="(image, index) in images" :key="index" :src="image" class="w-24 h-24 object-cover rounded-lg border"  alt="hochgeladenes Foto"/>
        </div>
        <Button @click="submitImages" class="w-full">Bilder absenden</Button>
      </CardContent>
    </Card>
  </div>
</template>

<style scoped>
</style>
