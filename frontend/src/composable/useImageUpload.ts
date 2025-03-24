import { ref } from 'vue'
import { toast } from 'vue-sonner'
import { usePlantHistory } from '@/stores/usePlantHistory.ts'

export const BASE_PATH = import.meta.env.VITE_API_BASE || ''

export function useImageUpload(apiUrl = BASE_PATH) {
  const plantHistory = usePlantHistory()
  const isUploading = ref(false)
  const error = ref<string | null>(null)
  const MAX_FILE_SIZE_MB = 5

  const uploadImages = async (imageSources) => {
    if (!imageSources.length) {
      toast.error('Bitte wähle zuerst Bilder aus.')
      return
    }

    let base64Images: string[] = []

    if (imageSources[0] instanceof File) {
      const oversizedFiles = imageSources.filter(
        (file) => file.size > MAX_FILE_SIZE_MB * 1024 * 1024,
      )
      if (oversizedFiles.length) {
        toast.warning(`Einige Dateien sind zu groß (max. ${MAX_FILE_SIZE_MB} MB).`)
        return
      }

      base64Images = await Promise.all(
        imageSources.map(
          (file) =>
            new Promise<string>((resolve, reject) => {
              const reader = new FileReader()
              reader.onload = () => resolve(reader.result as string)
              reader.onerror = reject
              reader.readAsDataURL(file)
            }),
        ),
      )
    } else {
      base64Images = imageSources
    }

    const payload = { images: base64Images }
    const toastId = toast.loading('Bilder werden hochgeladen...')

    isUploading.value = true
    error.value = null

    try {
      const response = await fetch(apiUrl + '/recognizePlant', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        throw new Error(`Fehler: ${response.statusText}`)
      }

      const data = await response.json()

      data.results.forEach((result) => {
        plantHistory.addImageRecognition({
          image: result.image,
          recognitions: result.recognitions,
        })
      })

      toast.success('Bilder erfolgreich hochgeladen!', { id: toastId })
      return data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unbekannter Fehler'
      console.error('Upload fehlgeschlagen:', errorMessage)
      error.value = errorMessage
      toast.error(`Fehler: ${errorMessage}`, { id: toastId })
    } finally {
      isUploading.value = false
    }
  }

  return { uploadImages, isUploading, error }
}
