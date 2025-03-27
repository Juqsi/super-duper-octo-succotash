<script lang="ts" setup>
import EmptyState from '@/components/EmptyState.vue'
import NavBar from '@/components/NavBar.vue'
</script>

<template>
  <NavBar />
  <EmptyState
    :condition="true"
    img-src="/undraw_page-not-found_6wni.svg"
    subtitle="The requested page does not exist."
    title="404 - Page Not Found"
  />
</template>

<style scoped></style>
