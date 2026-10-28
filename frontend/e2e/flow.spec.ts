import { test, expect } from "@playwright/test";
test("real HN content: exposure, bookmark persistence, search, click", async ({
  page,
  context,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.locator("article").first()).toBeVisible({ timeout: 60000 });
  await expect(page.locator("article")).toHaveCount(30);
  const first = page.locator("article").first();
  const title = await first.locator(".story").textContent();
  await page.waitForTimeout(1500);
  await first.getByRole("button", { name: "收藏文章", exact: true }).click();
  await expect(first.getByRole("button", { name: "取消收藏" })).toBeVisible();
  await page.getByRole("button", { name: "我的收藏", exact: true }).click();
  await expect(page.locator("article").first().locator(".story")).toHaveText(
    title!,
  );
  await page.reload();
  await page.getByRole("button", { name: "我的收藏", exact: true }).click();
  await expect(page.locator("article").first().locator(".story")).toHaveText(
    title!,
  );
  await page.getByRole("textbox", { name: "搜索资讯" }).fill("AI");
  await page.getByRole("button", { name: "搜索 ↗" }).click();
  await expect(page.getByRole("heading", { name: "搜索：AI" })).toBeVisible();
  await expect(page.locator("article").first()).toBeVisible();
  const clickRequest = page.waitForRequest(
    (req) =>
      req.url().endsWith("/api/events") && req.postDataJSON()?.kind === "click",
  );
  await page.locator("article").first().locator(".story").click();
  await clickRequest;
  await page.screenshot({
    path: "/data1/zhuzy/signalfeed/reports/website.png",
    fullPage: true,
  });
  const stats = await (await page.request.get("/api/stats")).json();
  expect(stats.impressions).toBe(0);
  expect(stats.ctr).toBeNull();
  expect(errors).toEqual([]);
});
test("mobile layout remains usable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.locator("article").first()).toBeVisible({ timeout: 60000 });
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  );
  expect(overflow).toBe(false);
  await page.screenshot({
    path: "/data1/zhuzy/signalfeed/reports/website-mobile.png",
    fullPage: true,
  });
});
