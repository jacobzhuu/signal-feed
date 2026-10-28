import { test, expect, type Page, type Route } from "@playwright/test";

function listing(title: string) {
  return {
    request_id: title,
    items: [
      {
        id: title,
        title,
        url: "https://example.com",
        domain: "example.com",
        published: 1,
        score: 1,
        bookmarked: false,
        position: 0,
        reason: "fixture",
      },
    ],
  };
}
async function mockApi(page: Page) {
  await page.route("**/api/**", (route) => {
    const url = new URL(route.request().url());
    const json =
      url.pathname === "/api/interests"
        ? { topics: [] }
        : url.pathname === "/api/stats"
          ? { impressions: 0, ctr: null }
          : url.pathname === "/api/events"
            ? { accepted: true, inserted: true }
            : listing(
                url.pathname === "/api/bookmarks"
                  ? "Saved article"
                  : "Initial article",
              );
    return route.fulfill({ json });
  });
}
async function flushResponse(
  page: Page,
  route: Route,
  status: number,
  title: string,
) {
  const received = page.waitForResponse(
    (r) => r.url() === route.request().url(),
  );
  await route.fulfill({ status, json: listing(title) });
  await (await received).finished();
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      ),
  );
}

test("slow feed cannot overwrite a newer bookmarks page", async ({ page }) => {
  await mockApi(page);
  let release!: (route: Route) => void;
  const pending = new Promise<Route>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/feed", (route) => {
    release(route);
  });
  await page.goto("/");
  const old = await pending;
  await page.getByRole("button", { name: "我的收藏", exact: true }).click();
  await expect(page.locator(".story")).toContainText("Saved article");
  await flushResponse(page, old, 200, "Stale feed");
  await expect(page.getByRole("heading", { name: "你的收藏" })).toBeVisible();
  await expect(page.locator(".story")).toContainText("Saved article");
});

for (const oldStatus of [200, 500]) {
  test(`superseded search ${oldStatus} preserves newer loading, errors and results`, async ({
    page,
  }) => {
    await mockApi(page);
    const routes = new Map<string, Route>();
    await page.route("**/api/search?*", (route) => {
      routes.set(new URL(route.request().url()).searchParams.get("q")!, route);
    });
    await page.goto("/");
    await expect(page.locator(".story")).toContainText("Initial article");
    for (const query of ["older", "newer"]) {
      await page.getByRole("textbox", { name: "搜索资讯" }).fill(query);
      await page.getByRole("button", { name: "搜索 ↗" }).click();
      await expect.poll(() => routes.has(query)).toBeTruthy();
    }
    await flushResponse(page, routes.get("older")!, oldStatus, "Stale search");
    await expect(page.locator(".loading")).toBeVisible();
    await expect(page.getByRole("alert")).toHaveCount(0);
    await flushResponse(page, routes.get("newer")!, 200, "Newest search");
    await expect(page.locator(".story")).toContainText("Newest search");
    await expect(page.locator(".loading")).toHaveCount(0);
  });
}

test("late search error cannot replace the newer error and retry stays on its query", async ({
  page,
}) => {
  await mockApi(page);
  const routes = new Map<string, Route>();
  await page.route("**/api/search?*", (route) => {
    routes.set(new URL(route.request().url()).searchParams.get("q")!, route);
  });
  await page.goto("/");
  for (const query of ["old", "current"]) {
    await page.getByRole("textbox", { name: "搜索资讯" }).fill(query);
    await page.getByRole("button", { name: "搜索 ↗" }).click();
    await expect.poll(() => routes.has(query)).toBeTruthy();
  }
  await flushResponse(page, routes.get("current")!, 503, "unused");
  await expect(page.getByRole("alert")).toContainText("503");
  await flushResponse(page, routes.get("old")!, 500, "unused");
  await expect(page.getByRole("alert")).toContainText("503");
  routes.delete("current");
  await page.getByRole("button", { name: "重试", exact: true }).click();
  await expect.poll(() => routes.has("current")).toBeTruthy();
  await flushResponse(page, routes.get("current")!, 200, "Retried current");
  await expect(page.locator(".story")).toContainText("Retried current");
});

test("interest update finishing after navigation does not return to feed", async ({
  page,
}) => {
  await mockApi(page);
  let release!: (route: Route) => void;
  const pending = new Promise<Route>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/interests", (route) => {
    if (route.request().method() === "PUT") release(route);
    else return route.fulfill({ json: { topics: [] } });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Science", exact: true }).click();
  const update = await pending;
  await page.getByRole("button", { name: "我的收藏", exact: true }).click();
  await expect(page.locator(".story")).toContainText("Saved article");
  await flushResponse(page, update, 200, "unused");
  await expect(page.locator(".story")).toContainText("Saved article");
  await expect(page.getByRole("heading", { name: "你的收藏" })).toBeVisible();
});
