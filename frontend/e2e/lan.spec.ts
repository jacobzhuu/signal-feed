import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
import path from "node:path";

test("LAN HTTP supports exposure and bookmark retries without external assets", async ({ page }) => {
  const errors: string[] = [];
  const external: string[] = [];
  const events: { kind: string; event_id: string }[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== "http://signalfeed.test") {
      external.push(url.href);
      return route.abort();
    }
    if (url.pathname.startsWith("/api/")) {
      if (url.pathname === "/api/events") {
        const event = route.request().postDataJSON();
        events.push(event);
        if (event.kind === "bookmark" && events.filter((e) => e.kind === "bookmark").length === 1)
          return route.abort("failed");
        return route.fulfill({ json: { accepted: true } });
      }
      return route.fulfill({ json:
        url.pathname === "/api/interests" ? { topics: [] } :
        url.pathname === "/api/stats" ? { impressions: 0, ctr: null } : {
          request_id: "lan-test",
          items: [{ id: "lan-article", title: "LAN article", url: "https://example.com",
            domain: "example.com", published: 1, score: 1, bookmarked: false,
            position: 0, reason: "fixture" }],
        },
      });
    }
    const dist = path.resolve("dist");
    const file = path.resolve(dist, "." + (url.pathname === "/" ? "/index.html" : url.pathname));
    if (!file.startsWith(dist + path.sep)) return route.abort();
    return route.fulfill({ body: await readFile(file), contentType:
      file.endsWith(".js") ? "application/javascript" :
      file.endsWith(".css") ? "text/css" : "text/html" });
  });
  await page.goto("http://signalfeed.test/");
  expect(await page.evaluate(() => window.isSecureContext)).toBe(false);
  expect(await page.evaluate(() => typeof crypto.randomUUID)).toBe("undefined");
  await expect.poll(() => events.some((e) => e.kind === "impression")).toBe(true);
  await page.getByRole("button", { name: "收藏文章", exact: true }).click();
  await expect(page.getByRole("button", { name: "取消收藏", exact: true })).toBeVisible();
  const bookmarks = events.filter((event) => event.kind === "bookmark");
  expect(bookmarks).toHaveLength(2);
  expect(bookmarks[0].event_id).toBe(bookmarks[1].event_id);
  expect(bookmarks[0].event_id).not.toBe(events.find((e) => e.kind === "impression")!.event_id);
  for (const event of events)
    expect(event.event_id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  expect(errors).toEqual([]);
  expect(external).toEqual([]);
});
