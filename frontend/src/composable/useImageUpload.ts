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
      toast.error('Please select images first.')
      return
    }

    let base64Images: string[] = []

    if (imageSources[0] instanceof File) {
      const oversizedFiles = imageSources.filter(
        (file) => file.size > MAX_FILE_SIZE_MB * 1024 * 1024,
      )
      if (oversizedFiles.length) {
        toast.warning(`Image is too large (max. ${MAX_FILE_SIZE_MB} MB).`)
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
    const toastId = toast.loading('Upload image...')

    isUploading.value = true
    error.value = null

    try {
      const response = await fetch(apiUrl + '/uploads', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`)
      }

      const data = await response.json()

      data.results.forEach((result) => {
        plantHistory.addImageRecognition({
          image: result.image,
          recognitions: result.recognitions,
        })
      })

      toast.success('Upload successfully completed', { id: toastId })
      return data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown Error'
      console.error('Upload error:', errorMessage)
      error.value = errorMessage
      toast.error(`Error: ${errorMessage}`, { id: toastId })
    } finally {
      isUploading.value = false
    }
  }

  return { uploadImages, isUploading, error }
}
