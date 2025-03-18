<script setup>
import { ref } from 'vue';
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {toast} from 'vue-sonner'

const images = ref([]);
const imageFiles = ref([]);

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
  if (!imageFiles.value.length) {
    alert("Bitte lade zuerst Bilder hoch.");
    return;
  }

  const formData = new FormData();
  imageFiles.value.forEach((file, index) => {
    formData.append(`images`, file);
  });

  try {
    toast.loading("send Image")
    const response = await fetch("http://localhost:8000/uploads", {
      method: "POST",
      body: formData,
      headers: {
        Authorization: "Bearer dein-token",
      },
    });

    if (!response.ok) {
      throw new Error(`Fehler: ${response.statusText}`);
    }

    const data = await response.json();
    console.log("Upload erfolgreich:", data);
    toast.success("Bilder erfolgreich hochgeladen!");
  } catch (error) {
    console.error("Upload fehlgeschlagen:", error);
    toast.error("Fehler beim Hochladen!");
  }
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
