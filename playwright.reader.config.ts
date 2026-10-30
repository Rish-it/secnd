import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "frontend/reader-tests",
  workers: 1,
  use: { baseURL: "http://127.0.0.1:8034", browserName: "chromium" },
  webServer: {
    command: "uv run python -m backend.tests.reader_demo",
    url: "http://127.0.0.1:8034/api/health",
    reuseExistingServer: false,
    timeout: 60000,
  },
});
