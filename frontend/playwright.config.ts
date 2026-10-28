import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  timeout: 90000,
  use: {
    baseURL: "http://127.0.0.1:8766",
    headless: true,
    viewport: { width: 1440, height: 1000 },
  },
  reporter: [
    ["list"],
    [
      "json",
      { outputFile: "/data1/zhuzy/signalfeed/reports/browser-tests.json" },
    ],
  ],
});
