import { test, expect } from "@playwright/test";

test("slow ColdFuse retains the previous list and uses the new request when ready", async ({ page }) => {
  const listing = (name: string) => ({request_id:name,items:[{
    id:name,title:name,url:"https://example.com",domain:"example.com",published:1,
    score:1,bookmarked:false,position:0,reason:"fixture",
  }]});
  await page.route("**/api/**", route => {
    const path = new URL(route.request().url()).pathname;
    return route.fulfill({json:path==="/api/interests"?{topics:[]}:
      path==="/api/stats"?{impressions:0,ctr:null}:path==="/api/events"?{accepted:true}:listing("Previous article")});
  });
  let release!: () => Promise<void>;
  await page.route("**/api/feed?cold=true", route => {
    release=()=>route.fulfill({json:{...listing("Updated article"),experiment:{status:"timeout-fallback"}}});
  });
  await page.goto("/");
  await expect(page.locator(".story")).toHaveText(/Previous article/);
  await page.getByRole("button",{name:"兴趣加速（实验）",exact:true}).click();
  await expect(page.locator(".loading")).toBeVisible();
  await expect(page.locator(".story")).toHaveText(/Previous article/);
  await expect.poll(()=>typeof release).toBe("function");
  await release();
  await expect(page.locator(".story")).toHaveText(/Updated article/);
  await expect(page.locator(".loading")).toHaveCount(0);
  const saved=page.waitForRequest(r=>r.url().endsWith("/api/events") && r.postDataJSON()?.kind==="bookmark");
  await page.getByRole("button",{name:"收藏文章",exact:true}).click();
  expect((await saved).postDataJSON().request_id).toBe("Updated article");
});
