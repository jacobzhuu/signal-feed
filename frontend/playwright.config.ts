import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  timeout: 90000,
  use: {
    baseURL: process.env.SIGNALFEED_BASE_URL || "http://127.0.0.1:8766",
    headless: true,
    viewport: { width: 1440, height: 1000 },
  },
  reporter: [
    ["list"],
    [
      "json",
      {
        outputFile:
          process.env.SIGNALFEED_BROWSER_REPORT ||
          "test-results/browser-tests.json",
      },
    ],
  ],
});
