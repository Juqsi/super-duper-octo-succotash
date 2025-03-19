<script lang="ts" setup>
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { CameraIcon, TrashIcon } from 'lucide-vue-next'
import { nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useImageUpload } from '@/composable/useImageUpload'

const { uploadImages, error } = useImageUpload('http://localhost:8000/upload')

const camera = ref(null)
const canvas = ref(null)
const isCameraActive = ref(false)
const photos = ref([])
let stream = null

const startCamera = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true })
    if (camera.value) {
      camera.value.srcObject = stream
      await nextTick()
      camera.value.play()
    }
    isCameraActive.value = true
  } catch (error) {
    console.error('Kamera-Fehler:', error)
    toast.error('Fehler: Zugriff auf Kamera verweigert oder nicht verfügbar.')
  }
}

watch(
  camera,
  (newVal) => {
    if (newVal) {
      if (stream) {
        newVal.srcObject = stream
        newVal.onloadedmetadata = () => {
          newVal.play()
        }
      }
    }
  },
  { immediate: true },
)

const stopCamera = () => {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop())
    stream = null
    isCameraActive.value = false
  }
}

onBeforeUnmount(() => {
  if (stream) {
    stopCamera()
  }
})

const capturePhoto = () => {
  const ctx = canvas.value.getContext('2d')
  canvas.value.width = camera.value.videoWidth
  canvas.value.height = camera.value.videoHeight
  ctx.drawImage(camera.value, 0, 0, canvas.value.width, canvas.value.height)
  photos.value.push(canvas.value.toDataURL('image/png'))
}

const removePhoto = (index) => {
  photos.value.splice(index, 1)
}

const emitPhotos = async () => {
  if (!photos.value.length) {
    toast.error('Keine Bilder zum Hochladen.')
    return
  }
  const response = await uploadImages([...photos.value])
  if (response) {
    photos.value = []
  }
  if (error.value) {
    console.error('Upload-Fehler:', error.value)
  }
}

const onVideoError = (event) => {
  console.error('Video-Fehler:', event)
  toast.error('Video error')
}
</script>

<template>
  <Card class="max-w-md mx-auto p-6">
    <CardContent>
      <div v-if="!isCameraActive" class="text-center flex flex-col gap-6">
        <div class="flex items-center gap-4">
          <div class="size-12 border border-muted rounded-full flex items-center justify-center">
            <CameraIcon class="w-6 h-6 text-muted-foreground" />
          </div>
          <div>
            <h2 class="text-lg font-medium">Take a Photo</h2>
            <p class="text-muted-foreground text-sm">Capture product feedback photos.</p>
          </div>
        </div>
        <Button class="w-full" @click="startCamera">Start Camera</Button>
      </div>

      <div v-else>
        <div class="flex justify-between items-center mb-4">
          <Button variant="outline" @click="stopCamera">Exit</Button>
          <Button
            :disabled="photos.length === 0"
            class="disabled:bg-muted disabled:text-muted-foreground"
            @click="emitPhotos"
            >Use Photos
          </Button>
        </div>

        <video
          ref="camera"
          autoplay
          class="w-full rounded-lg mb-4"
          muted
          playsinline
          @error="onVideoError"
        ></video>
        <div class="grid grid-cols-3 gap-3">
          <div v-for="(photo, index) in photos" :key="index" class="relative">
            <img :src="photo" alt="Aufgenommenes Bild" class="rounded-lg w-full" />
            <Button
              class="absolute top-2 right-2"
              size="icon"
              variant="ghost"
              @click="removePhoto(index)"
            >
              <TrashIcon class="w-4 h-4 text-destructive" />
            </Button>
          </div>
        </div>
        <div class="flex gap-2 mt-4">
          <Button :disabled="photos.length >= 3" @click="capturePhoto">Capture</Button>
        </div>
      </div>

      <canvas ref="canvas" class="hidden"></canvas>
    </CardContent>
  </Card>
</template>
