import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    // This prevents "ReferenceError: process is not defined" if any library tries to access process.env
    'process.env': {}
  }
})