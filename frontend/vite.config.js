import { defineConfig } from "vite";

// Optional Vite config. The prototype UI works WITHOUT this (FastAPI serves the
// static files directly). If you prefer hot-reload while developing the UI, run
// `npm install && npm run dev` inside frontend/ and Vite will proxy /api to the
// FastAPI server.
export default defineConfig({
  root: ".",
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
