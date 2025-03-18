<script setup>
import { ref } from 'vue';
import { useImageUpload } from '@/composable/useImageUpload';
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { TrashIcon } from "lucide-vue-next";

const images = ref([]);
const imageFiles = ref([]);
const { uploadImages, isUploading, error } = useImageUpload("http://localhost:8000/upload");

const handleFileChange = (event) => {
  const files = Array.from(event.target.files);
  if (!files.length) return;

  files.forEach((file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      images.value.push({ name: file.name, src: e.target.result });
    };
    reader.readAsDataURL(file);
    imageFiles.value.push(file);
  });
};

const removeFile = (index) => {
  images.value.splice(index, 1);
  imageFiles.value.splice(index, 1);
};

const submitImages = async () => {
  if (imageFiles.value.length === 0) return;

  const response = await uploadImages(imageFiles.value);
  if (response) {
    // Nach erfolgreichem Upload Liste leeren
    images.value = [];
    imageFiles.value = [];
  }

  if (error.value) {
    console.error("Upload-Fehler:", error.value);
  }
};
</script>

<template>
  <div class="flex flex-col items-center gap-4 p-6">
    <Card class="w-full max-w-md p-4 border border-gray-200 shadow-md rounded-2xl">
      <CardContent class="flex flex-col gap-4">
        <input type="file" accept="image/*" multiple @change="handleFileChange" class="hidden" id="file-upload" />
        <label for="file-upload" class="cursor-pointer p-4 border border-dashed border-gray-300 rounded-xl text-gray-600 text-center hover:bg-gray-50">
          Bilder hochladen
        </label>

        <div v-if="images.length" class="grid grid-cols-3 gap-2 relative">
          <div v-for="(image, index) in images" :key="index" class="relative">
            <img :src="image.src" class="w-24 h-24 object-cover rounded-lg border" alt="hochgeladenes Foto"/>
            <Button size="icon" variant="ghost" class="absolute top-1 right-1 bg-white p-1 rounded-full shadow"
                    @click="removeFile(index)">
              <TrashIcon class="w-4 h-4 text-destructive" />
            </Button>
          </div>
        </div>

        <Button @click="submitImages" class="w-full" :disabled="images.length === 0 || isUploading">
          {{ isUploading ? "Wird hochgeladen..." : "Bilder absenden" }}
        </Button>
      </CardContent>
    </Card>
  </div>
</template>
