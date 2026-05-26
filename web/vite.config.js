import { defineConfig } from 'vite'

export default defineConfig({
  root: '.',
  publicDir: 'public',
  server: { host: '127.0.0.1', port: 5173, open: true },
  build: { outDir: 'dist', sourcemap: true },
})
