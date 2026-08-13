import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // listen on all interfaces, so it's reachable from your phone on the same WiFi
    port: 5173,
  },
});
