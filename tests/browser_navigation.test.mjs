import assert from 'node:assert/strict';
import http from 'node:http';
import os from 'node:os';
import { test } from 'node:test';
import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import { directBrowserOptions, openSignalFeed } from '../scripts/browser_navigation.mjs';

test('campus browser bypasses inherited proxy and rejects HTTP errors before waiting for articles', async () => {
  let proxyRequests = 0;
  const origin = http.createServer((req, res) => {
    res.writeHead(req.url === '/broken' ? 502 : 200, { 'Content-Type': 'text/html' });
    res.end(req.url === '/broken' ? 'Bad gateway' : '<article>Isolated fixture</article>');
  });
  const proxy = http.createServer((req, res) => {
    proxyRequests++;
    res.writeHead(502);
    res.end('Proxy must not handle the campus request');
  });
  await new Promise(r => origin.listen(0, '0.0.0.0', r));
  await new Promise(r => proxy.listen(0, '127.0.0.1', r));
  let browser;
  try {
    const host = Object.values(os.networkInterfaces()).flat().find(a => a.family === 'IPv4' && !a.internal)?.address || '127.0.0.1';
    const base = `http://${host}:${origin.address().port}`;
    const proxyUrl = `http://127.0.0.1:${proxy.address().port}`;
    browser = await chromium.launch(directBrowserOptions({
      ...process.env, HTTP_PROXY: proxyUrl, http_proxy: proxyUrl,
      HTTPS_PROXY: proxyUrl, https_proxy: proxyUrl, ALL_PROXY: proxyUrl,
      all_proxy: proxyUrl, NO_PROXY: '', no_proxy: '',
    }));
    const page = await browser.newPage();
    await openSignalFeed(page, base);
    assert.equal(await page.locator('article').innerText(), 'Isolated fixture');
    assert.equal(proxyRequests, 0);
    const start = performance.now();
    await assert.rejects(openSignalFeed(page, base + '/broken'), /HTTP 502; ColdFuse was not measured/);
    assert.ok(performance.now() - start < 3000);
  } finally {
    if (browser) await browser.close();
    origin.closeAllConnections();
    proxy.closeAllConnections();
    await Promise.all([new Promise(r => origin.close(r)), new Promise(r => proxy.close(r))]);
  }
});
