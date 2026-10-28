"""Read-only HTTP latency benchmark; does not emit fake exposure/click events."""

import asyncio
import json
import time
import statistics
from pathlib import Path
import os
import httpx


async def run():
    url = "http://127.0.0.1:8765"
    result = {}
    async with httpx.AsyncClient(trust_env=False, timeout=90) as client:
        tick = time.perf_counter()
        response = await client.get(url + "/api/feed")
        response.raise_for_status()
        result["initial_request_ms"] = (time.perf_counter() - tick) * 1000
        for concurrency in [1, 8]:
            semaphore = asyncio.Semaphore(concurrency)
            latencies = []

            async def request(i):
                async with semaphore:
                    start = time.perf_counter()
                    r = await client.get(
                        url
                        + ("/api/search?q=machine+learning" if i % 2 else "/api/feed")
                    )
                    r.raise_for_status()
                    latencies.append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            await asyncio.gather(*(request(i) for i in range(80)))
            duration = time.perf_counter() - start
            ordered = sorted(latencies)
            result[str(concurrency)] = dict(
                requests=80,
                p50_ms=statistics.median(ordered),
                p95_ms=ordered[int(0.95 * (len(ordered) - 1))],
                requests_per_second=80 / duration,
            )
    result["note"] = (
        "Local HTTP mixed feed/search; no exposure events; shared CPU/GPU host, not production capacity claim"
    )
    path = (
        Path(os.environ.get("SIGNALFEED_HOME", "/data1/zhuzy/signalfeed"))
        / "reports"
        / "latency.json"
    )
    path.write_text(json.dumps(result, indent=2))
    print(result)


asyncio.run(run())
