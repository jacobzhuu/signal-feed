// Campus acceptance uses a direct browser connection, independently of host proxies.
// This affects only the browser launched by this script, not any existing browser.
export function directBrowserOptions(environment = process.env) {
  const env = Object.fromEntries(Object.entries(environment).filter(([key]) => !/proxy/i.test(key)));
  return { headless: true, args: ['--no-proxy-server'], env };
}

export async function openSignalFeed(page, base) {
  const response = await page.goto(base, { waitUntil: 'domcontentloaded', timeout: 10000 });
  if (!response?.ok()) {
    throw Error(`Page navigation failed: HTTP ${response?.status() ?? 'no response'}; ColdFuse was not measured`);
  }
  await page.locator('article').first().waitFor({ timeout: 10000 });
}
