import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  testMatch: "harness.spec.ts",
  workers: 1,
  outputDir: "./test-results",
  use: {
    baseURL: "http://secnd.test",
    browserName: "chromium",
    colorScheme: "light",
    serviceWorkers: "block",
  },
});
