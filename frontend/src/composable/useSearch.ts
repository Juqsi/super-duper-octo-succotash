import { ref } from 'vue'
import { toast } from 'vue-sonner'

export const BASE_PATH = import.meta.env.VITE_API_BASE || ''

export function useSearch(apiUrl: string = BASE_PATH) {
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  const searchPlant = async (name: string) => {
    const toastId = toast.loading('Search plant...')
    isLoading.value = true
    error.value = null

    const payload = { name }
    try {
      const response = await fetch(apiUrl + '/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`)
      }

      const data = await response.json()
      toast.success('Search successful!', { id: toastId })
      return data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      console.error('Search failed:', errorMessage)
      error.value = errorMessage
      toast.error(`Error: ${errorMessage}`, { id: toastId })
    } finally {
      isLoading.value = false
    }
  }

  return { searchPlant, isLoading, error }
}
