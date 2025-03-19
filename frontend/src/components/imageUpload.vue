<script lang="ts" setup>
import { ref } from 'vue'
import { useImageUpload } from '@/composable/useImageUpload'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { TrashIcon } from 'lucide-vue-next'

// Typ für gespeicherte Bilder
interface ImagePreview {
  name: string
  src: string
}

const images = ref<ImagePreview[]>([])
const imageFiles = ref<File[]>([])
const { uploadImages, isUploading, error } = useImageUpload('http://localhost:8000/upload')

// Datei hochladen
const handleFileChange = (event: Event) => {
  const input = event.target as HTMLInputElement
  if (!input.files) return

  const files = Array.from(input.files) as File[]
  if (files.length === 0) return

  files.forEach((file) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      if (e.target?.result) {
        images.value.push({ name: file.name, src: e.target.result as string })
        imageFiles.value.push(file)
      }
    }
    reader.readAsDataURL(file)
  })
}

// Datei entfernen
const removeFile = (index: number) => {
  images.value.splice(index, 1)
  imageFiles.value.splice(index, 1)
}

// Bilder hochladen
const submitImages = async () => {
  if (imageFiles.value.length === 0) return

  const response = await uploadImages(imageFiles.value)
  if (response) {
    images.value = []
    imageFiles.value = []
  }

  if (error.value) {
    console.error('Upload-Fehler:', error.value)
  }
}
</script>

<template>
  <div class="flex flex-col items-center gap-4 p-6">
    <Card class="w-full max-w-md p-4 border border-gray-200 shadow-md rounded-2xl">
      <CardContent class="flex flex-col gap-4">
        <input
          id="file-upload"
          accept="image/*"
          class="hidden"
          multiple
          type="file"
          @change="handleFileChange"
        />
        <label
          class="cursor-pointer p-4 border border-dashed border-gray-300 rounded-xl text-gray-600 text-center hover:bg-gray-50"
          for="file-upload"
        >
          Bilder hochladen
        </label>

        <div v-if="images.length" class="grid grid-cols-3 gap-2 relative">
          <div v-for="(image, index) in images" :key="index" class="relative">
            <img
              :src="image.src"
              alt="hochgeladenes Foto"
              class="w-24 h-24 object-cover rounded-lg border"
            />
            <Button
              class="absolute top-1 right-1 bg-white p-1 rounded-full shadow"
              size="icon"
              variant="ghost"
              @click="removeFile(index)"
            >
              <TrashIcon class="w-4 h-4 text-destructive" />
            </Button>
          </div>
        </div>

        <Button :disabled="images.length === 0 || isUploading" class="w-full" @click="submitImages">
          {{ isUploading ? 'Wird hochgeladen...' : 'Bilder absenden' }}
        </Button>
      </CardContent>
    </Card>
  </div>
</template>
