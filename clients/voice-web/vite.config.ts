import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Built to a static dist served by the voice proxy (aiohttp) at `/` + `/assets`.
// For `pnpm dev`, /api is proxied (ws included) to a running proxy backend —
// set VITE_BACKEND to point at one (default: localhost:8999, e.g. when dev'ing
// on hypatia). Mic capture needs a secure context; localhost qualifies.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backend = env.VITE_BACKEND || "http://127.0.0.1:8999";
  return {
    plugins: [react()],
    build: { target: "esnext", assetsDir: "assets" },
    server: {
      host: "0.0.0.0",
      proxy: {
        "/api": { target: backend, changeOrigin: true, ws: true },
      },
    },
  };
});
