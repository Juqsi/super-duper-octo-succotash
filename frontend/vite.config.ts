import vue from '@vitejs/plugin-vue'
import tailwind from 'tailwindcss'
import { defineConfig } from 'vite'
import { fileURLToPath, URL } from 'node:url'
import mkcert from 'vite-plugin-mkcert'

export default defineConfig({
  css: {
    postcss: {
      plugins: [tailwind()],
    },
  },
  plugins: [vue(), mkcert()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
})
