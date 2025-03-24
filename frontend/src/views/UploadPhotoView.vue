<script lang="ts" setup>
import CapturePhotoComponent from '@/components/CapturePhotoComponent.vue'
import ImageUpload from '@/components/ImageUpload.vue'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
</script>

<template>
  <Tabs class="w-[400px] max-w-full" default-value="capturePhoto">
    <TabsList class="grid w-full grid-cols-2">
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
</template>

<style scoped></style>
