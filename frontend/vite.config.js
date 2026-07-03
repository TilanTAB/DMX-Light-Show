import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Proxy /api requests to the FastAPI backend during development.
    // In production, the React build is served directly from FastAPI,
    // so no proxy is needed — relative URLs hit the same origin.
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})

