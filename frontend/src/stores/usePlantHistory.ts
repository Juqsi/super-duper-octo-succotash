import { defineStore } from 'pinia'

export interface Plant {
  plant_id: number
  common_name: string | undefined
  scientific_name: string | undefined
  cycle: string | undefined
  watering_short: string | undefined
  hardiness_zone: string | undefined
  sun: string | undefined
  cones: string | undefined
  leaf: string | undefined
  leaf_color: string | undefined
  growth_rate: string | undefined
  care_level: string | undefined
  watering_extended: string | undefined
  sunlight_extended: string | undefined
  pruning_extended: string | undefined
}

export interface Recognition {
  name: string
  plant: Plant | null
  wikipedia: string
  probability?: number
}

export interface RecognizedImage {
  // Speichere hier nur den Key, mit dem du das Bild abrufen kannst
  imageKey: string
  recognitions: Recognition[]
  timestamp: number
}

export const usePlantHistory = defineStore('plantHistory', {
  state: () => ({
    history: [] as RecognizedImage[],
  }),
  actions: {
    /**
     * entry.image enthält den Base64-String des Bildes.
     * Wir speichern diesen in einem eigenen localStorage-Eintrag,
     * und speichern im Store nur den key als Referenz.
     */
    addImageRecognition(
      entry: Omit<RecognizedImage, 'timestamp' | 'imageKey'> & { image: string },
    ) {
      const timestamp = Date.now()
      const imageKey = `plantImage_${timestamp}`
      try {
        localStorage.setItem(imageKey, entry.image)
      } catch (e) {
        console.error('Fehler beim Speichern des Bildes:', e)
      }
      const recognizedImage: RecognizedImage = {
        imageKey,
        recognitions: entry.recognitions,
        timestamp,
      }
      this.history.unshift(recognizedImage)

      if (this.history.length > 10) {
        const removedEntries = this.history.splice(10)
        removedEntries.forEach((entry) => {
          localStorage.removeItem(entry.imageKey)
        })
      }
    },

    removeImageRecognition(imageKey: string) {
      localStorage.removeItem(imageKey)
      this.history = this.history.filter((entry) => entry.imageKey !== imageKey)
    },

    clearHistory() {
      this.history.forEach((entry) => {
        localStorage.removeItem(entry.imageKey)
      })
      this.history = []
    },

    getImageRecognition(imageKey: string): RecognizedImage | undefined {
      return this.history.find((entry) => entry.imageKey === imageKey)
    },
  },
  persist: true,
})
