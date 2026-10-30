import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./frontend/tests",
  workers: 1,
  use: { baseURL: "http://127.0.0.1:8031", browserName: "chromium" },
  webServer: {
    command: "uv run python -m backend.tests.inbox_demo --port 8031",
    url: "http://127.0.0.1:8031/api/health",
    timeout: 120000,
    reuseExistingServer: false,
  },
});
