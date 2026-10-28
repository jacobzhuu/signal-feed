import { test, expect } from "@playwright/test";

test("ColdFuse interests, real read, update and isolated feedback", async ({ page, context }) => {
  await page.goto("/");
  await expect(page.locator("article").first()).toBeVisible({ timeout: 60000 });
  await page.getByRole("button", { name: "兴趣加速（实验）" }).click();
  await expect(page.getByRole("heading", { name: "兴趣加速", exact: true })).toBeVisible();
  await expect(page.locator("article").first()).toBeVisible();
  const interests = page.waitForResponse(r => r.url().endsWith("/api/interests") && r.request().method() === "PUT");
  await page.getByRole("button", { name: "Artificial Intelligence", exact: true }).click();
  await interests;
  await expect(page.locator("article").first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "兴趣加速", exact: true })).toBeVisible();
  const title = await page.locator("article").first().locator(".story").textContent();
  const event = page.waitForResponse(r => r.url().endsWith("/api/events") && r.request().postDataJSON()?.kind === "click");
  const popup = context.waitForEvent("page");
  await page.locator("article").first().locator(".story").click();
  await event;
  await (await popup).close();
  await page.getByRole("button", { name: "根据最新阅读更新" }).click();
  await expect(page.locator("article").first()).toBeVisible();
  expect(await page.locator(".story").allTextContents()).not.toContain(title);
  const health = await (await page.request.get("/api/health")).json();
  expect(health.test_events).toBe(true);
  expect((await (await page.request.get("/api/stats")).json()).ctr).toBeNull();
});
