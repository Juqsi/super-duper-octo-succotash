<script lang="ts" setup>
import { Button } from '@/components/ui/button'
import { CameraIcon, SendHorizontal, TrashIcon } from 'lucide-vue-next'
import { nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useImageUpload } from '@/composable/useImageUpload'
import { toast } from 'vue-sonner'
import router from '@/router'

const { uploadImages } = useImageUpload()
const camera = ref<HTMLVideoElement | null>(null)
const canvas = ref<HTMLCanvasElement | null>(null)
const isCameraActive = ref(false)
const photos = ref<string[]>([])
let stream: MediaStream | null = null

const startCamera = async () => {
  try {
    const constraints = {
      video: { facingMode: { ideal: 'environment' } },
    }
    stream = await navigator.mediaDevices.getUserMedia(constraints)

    if (camera.value) {
      camera.value.srcObject = stream
      await nextTick()
      camera.value.play()
    }

    isCameraActive.value = true
  } catch (err) {
    console.error('Kamera-Fehler:', err)
    toast.error('Fehler: Zugriff auf Kamera verweigert oder nicht verfügbar.')
  }
}

watch(
  camera,
  (newVal) => {
    if (newVal && stream) {
      newVal.srcObject = stream
      newVal.onloadedmetadata = () => newVal.play()
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
  stopCamera()
})

const capturePhoto = () => {
  if (!canvas.value || !camera.value) {
    toast.error('Fehler: Kamera oder Canvas nicht verfügbar.')
    return
  }

  const ctx = canvas.value.getContext('2d')
  if (!ctx) {
    toast.error('Fehler: Canvas-Kontext konnte nicht erstellt werden.')
    return
  }

  canvas.value.width = camera.value.videoWidth
  canvas.value.height = camera.value.videoHeight
  ctx.drawImage(camera.value, 0, 0, canvas.value.width, canvas.value.height)

  const imageData = canvas.value.toDataURL('image/png')
  photos.value.push(imageData)
}

const removePhoto = (index: number) => {
  photos.value.splice(index, 1)
}

const emitPhotos = async () => {
  if (!photos.value.length) {
    toast.error('Keine Bilder zum Hochladen.')
    return
  }

  const response = await uploadImages([...photos.value])
  const length = photos.value.length
  if (response) {
    photos.value = []
  }
  router.push('/last/' + length)
}

const onVideoError = (event: Event) => {
  console.error('Video-Fehler:', event)
  toast.error('Video-Fehler: Kamera funktioniert nicht. Bitte prüfe die Kameraeinstellungen.')
}
</script>

<template>
  <div v-if="!isCameraActive" class="text-center flex flex-col gap-6">
    <div class="flex items-center gap-4">
      <div class="size-12 border border-muted rounded-full flex items-center justify-center">
        <CameraIcon class="w-6 h-6 text-muted-foreground" />
      </div>
      <div>
        <h2 class="text-lg font-medium">Capture Photo</h2>
        <p class="text-muted-foreground text-sm">Take a picture of your plant</p>
      </div>
    </div>
    <Button class="w-full" @click="startCamera">Start Camera</Button>
  </div>

  <div v-else>
    <video
      ref="camera"
      autoplay
      class="w-full rounded-lg mb-4"
      muted
      playsinline
      @error="onVideoError"
    ></video>
    <div class="flex justify-between items-center mb-4">
      <Button variant="outline" @click="stopCamera">Exit</Button>
      <Button :disabled="photos.length >= 3" @click="capturePhoto">Capture</Button>
    </div>

    <!-- Hover-Animation für Vorschaubilder -->
    <div class="grid grid-cols-3 gap-3">
      <div v-for="(photo, index) in photos" :key="index" class="relative group">
        <img
          :src="photo"
          alt="Aufgenommenes Bild"
          class="rounded-lg w-full transition-transform duration-200 group-hover:scale-105"
        />
        <Button
          class="absolute top-1 right-1 bg-white p-1 rounded-full shadow opacity-0 group-hover:opacity-100 transition-opacity duration-200"
          size="icon"
          variant="ghost"
          @click="removePhoto(index)"
        >
          <TrashIcon class="w-4 h-4 text-destructive" />
        </Button>
      </div>
    </div>
    <div class="flex gap-2 my-4 w-full justify-end">
      <Button
        :disabled="photos.length === 0"
        class="disabled:bg-muted disabled:text-muted-foreground"
        @click="emitPhotos"
      >
        <SendHorizontal />
      </Button>
    </div>
  </div>

  <canvas ref="canvas" class="hidden"></canvas>
</template>
