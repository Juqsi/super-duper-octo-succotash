<script lang="ts" setup>
import { ref } from 'vue'
import { useImageUpload } from '@/composable/useImageUpload'
import { Button } from '@/components/ui/button'
import { TrashIcon } from 'lucide-vue-next'

interface ImagePreview {
  name: string
  src: string
}

const images = ref<ImagePreview[]>([])
const imageFiles = ref<File[]>([])
const { uploadImages, isUploading } = useImageUpload()

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

const removeFile = (index: number) => {
  images.value.splice(index, 1)
  imageFiles.value.splice(index, 1)
}

const submitImages = async () => {
  if (imageFiles.value.length === 0) return

  const response = await uploadImages(imageFiles.value)
  if (response) {
    images.value = []
    imageFiles.value = []
  }
}
</script>

<template>
  <div class="flex flex-col items-center gap-4">
    <input
      id="file-upload"
      accept="image/*"
      class="hidden"
      multiple
      type="file"
      @change="handleFileChange"
    />
    <label
      class="cursor-pointer p-4 border border-dashed border-gray-300 rounded-xl text-gray-600 text-center hover:bg-gray-50 transition-colors duration-200 w-full"
      for="file-upload"
    >
      Upload image
    </label>

    <div v-if="images.length" class="grid grid-cols-3 gap-2">
      <div v-for="(image, index) in images" :key="index" class="relative group">
        <img
          :src="image.src"
          alt="hochgeladenes Foto"
          class="w-24 h-24 object-cover rounded-lg border transition-transform duration-200 group-hover:scale-105"
        />
        <Button
          class="absolute top-1 right-1 bg-white p-1 rounded-full shadow opacity-0 group-hover:opacity-100 transition-opacity duration-200"
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
  </div>
</template>
