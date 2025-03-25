<script lang="ts" setup>
import { Button } from '@/components/ui/button'
import EmptyState from '@/components/EmptyState.vue'
</script>

<template>
  <EmptyState
    :condition="true"
    img-src="/svg/undraw_page-not-found_6wni.svg"
    subtitle="The requested page does not exist."
    title="404 - Page Not Found"
  />
  <router-link to="/">
    <Button variant="link">Zurück</Button>
  </router-link>
</template>

<style scoped></style>
