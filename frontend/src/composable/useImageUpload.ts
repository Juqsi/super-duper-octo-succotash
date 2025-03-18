import { ref } from 'vue';
import { toast } from 'vue-sonner';

export function useImageUpload(apiUrl) {
  const isUploading = ref(false);
  const error = ref<string | null>(null);
  const MAX_FILE_SIZE_MB = 5; // Maximale Dateigröße in MB

  const uploadImages = async (imageFiles) => {
    if (!imageFiles.length) {
      toast.error("Bitte lade zuerst Bilder hoch.");
      return;
    }

    const oversizedFiles = imageFiles.filter((file) => file.size > MAX_FILE_SIZE_MB * 1024 * 1024);
    if (oversizedFiles.length) {
      toast.warning(`Einige Dateien sind zu groß (max. ${MAX_FILE_SIZE_MB} MB).`);
      return;
    }

    const formData = new FormData();
    imageFiles.forEach((file) => {
      formData.append("images", file);
    });

    const toastId = toast.loading("Bilder werden hochgeladen...");

    isUploading.value = true;
    error.value = null;

    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Fehler: ${response.statusText}`);
      }

      const data = await response.json();

      toast.success("Bilder erfolgreich hochgeladen!", { id: toastId });
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unbekannter Fehler";
      console.error("Upload fehlgeschlagen:", errorMessage);
      error.value = errorMessage;

      toast.error(`Fehler: ${errorMessage}`, { id: toastId });
    } finally {
      isUploading.value = false;
    }
  };

  return { uploadImages, isUploading, error };
}
