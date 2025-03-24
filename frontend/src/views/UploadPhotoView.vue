<script lang="ts" setup>
import CapturePhotoComponent from '@/components/CapturePhotoComponent.vue'
import ImageUpload from '@/components/ImageUpload.vue'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import NavBar from '@/components/NavBar.vue'
</script>

<template>
  <NavBar />

  <div class="flex justify-center items-center p-4 mt-4">
    <div
      class="w-[400px] max-w-full bg-background bg-opacity-80 backdrop-blur-md p-6 rounded-xl shadow-lg"
    >
      <Tabs default-value="capturePhoto">
        <TabsList class="grid w-full grid-cols-2 mb-4">
          <TabsTrigger value="capturePhoto"> Bildaufnehmen</TabsTrigger>
          <TabsTrigger value="UploadImage"> Bild Hochladen</TabsTrigger>
        </TabsList>
        <TabsContent value="capturePhoto">
          <CapturePhotoComponent />
        </TabsContent>
        <TabsContent value="UploadImage">
          <ImageUpload />
        </TabsContent>
      </Tabs>
    </div>
  </div>
</template>

<style scoped></style>
