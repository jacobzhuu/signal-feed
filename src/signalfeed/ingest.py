import asyncio
import json
import time
from urllib.parse import urlparse
import httpx
from .store import db, initialize
from .settings import path

API = "https://hacker-news.firebaseio.com/v0"


async def sync(limit=1000):
    initialize()
    start = time.time()
    errors = 0
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:

        async def fetch(endpoint):
            for attempt in range(3):
                try:
                    r = await client.get(f"{API}/{endpoint}.json")
                    r.raise_for_status()
                    return r.json()
                except (httpx.HTTPError, ValueError):
                    if attempt == 2:
                        raise
                    await asyncio.sleep(0.5 * 2**attempt)

        top, new = await asyncio.gather(fetch("topstories"), fetch("newstories"))
        with db() as c:
            existing = [
                r[0]
                for r in c.execute("SELECT id FROM articles ORDER BY updated LIMIT 200")
            ]
        ids = list(dict.fromkeys([str(x) for x in top + new] + existing))[:limit]
        semaphore = asyncio.Semaphore(12)

        async def item(i):
            async with semaphore:
                try:
                    return await fetch(f"item/{i}")
                except Exception:
                    return {"fetch_error": i}

        items = await asyncio.gather(*(item(i) for i in ids))
    count = 0
    with db() as c:
        for row in items:
            if not row:
                continue
            if "fetch_error" in row:
                errors += 1
                continue
            iid = str(row["id"])
            if row.get("deleted") or row.get("dead"):
                c.execute(
                    "UPDATE articles SET deleted=1,updated=? WHERE id=?",
                    (time.time(), iid),
                )
                c.execute("DELETE FROM article_fts WHERE id=?", (iid,))
                continue
            if row.get("type") != "story" or not row.get("title"):
                continue
            url = row.get("url") or f"https://news.ycombinator.com/item?id={iid}"
            parsed = urlparse(url)
            if parsed.scheme not in {"https", "http"}:
                continue
            domain = parsed.netloc
            title = row["title"]
            c.execute(
                "INSERT INTO articles VALUES(?,?,?,?,?,?,0,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,url=excluded.url,domain=excluded.domain,score=excluded.score,deleted=0,updated=excluded.updated",
                (
                    iid,
                    title,
                    url,
                    domain,
                    row["time"],
                    row.get("score", 0),
                    time.time(),
                ),
            )
            c.execute("DELETE FROM article_fts WHERE id=?", (iid,))
            c.execute("INSERT INTO article_fts VALUES(?,?,?)", (iid, title, domain))
            count += 1
        c.execute(
            "INSERT INTO ingest_runs VALUES(?,?,?,?)",
            (start, time.time(), count, errors),
        )
    report = dict(source=API, count=count, errors=errors, seconds=time.time() - start)
    (path("reports") / "hn_ingest.json").write_text(json.dumps(report, indent=2))
    print(report)
    return report
