import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./frontend/meta-tests",
  workers: 1,
  use: { baseURL: "http://127.0.0.1:8032", browserName: "chromium" },
  webServer: {
    command: "uv run python -m backend.tests.meta_apps_demo --port 8032",
    url: "http://127.0.0.1:8032/api/health",
    timeout: 120000,
    reuseExistingServer: false,
  },
});
