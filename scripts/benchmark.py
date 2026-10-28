"""Mixed multi-user benchmark. Requests write snapshots; use an isolated service."""

import asyncio
import json
import os
import time
from pathlib import Path
import uuid
import httpx
import numpy as np
from signalfeed.provenance import code_identity


async def run():
    url = os.environ.get("SIGNALFEED_BASE_URL", "http://127.0.0.1:8767")
    root = Path(os.environ["SIGNALFEED_HOME"])
    root.joinpath("reports").mkdir(parents=True, exist_ok=True)
    clients = [
        httpx.AsyncClient(base_url=url, trust_env=False, timeout=90) for _ in range(16)
    ]
    start = time.perf_counter()
    try:
        health = (await clients[0].get("/api/health")).json()
        if not health.get("test_events"):
            raise ValueError(
                "Benchmark requires SIGNALFEED_TEST_EVENTS=1 on an isolated server"
            )
        for i, client in enumerate(clients):
            r = await client.get("/api/interests")
            r.raise_for_status()
            if i % 3:
                r = await client.put(
                    "/api/interests",
                    json={
                        "topics": ["Artificial Intelligence" if i % 2 else "Security"]
                    },
                )
                r.raise_for_status()
            if i % 3 == 2:
                feed = (await client.get("/api/feed")).json()
                for item in feed["items"][:5]:
                    r = await client.post(
                        "/api/events",
                        json={
                            "event_id": str(uuid.uuid4()),
                            "request_id": feed["request_id"],
                            "item_id": item["id"],
                            "position": item["position"],
                            "kind": "click",
                        },
                    )
                    r.raise_for_status()
        result = {
            "run_id": root.name,
            "provenance": code_identity(),
            "preparation_seconds": time.perf_counter() - start,
            "users": len(clients),
            "health_before": health,
            "loads": {},
        }
        queries = [
            "machine learning",
            "security",
            "open source",
            "databases",
            "compilers",
            "robotics",
            "startups",
            "science",
        ]
        for concurrency in [1, 8, 32]:
            semaphore = asyncio.Semaphore(concurrency)
            latencies, event_latencies, errors = [], [], []

            async def request(i):
                async with semaphore:
                    client = clients[i % len(clients)]
                    tick = time.perf_counter()
                    try:
                        response = (
                            await client.get(
                                "/api/search", params={"q": queries[i % len(queries)]}
                            )
                            if i % 2
                            else await client.get("/api/feed")
                        )
                        response.raise_for_status()
                        data = response.json()
                        latencies.append((time.perf_counter() - tick) * 1000)
                        if i % 4 == 0 and data["items"]:
                            item = data["items"][0]
                            tick = time.perf_counter()
                            event = {
                                "event_id": str(uuid.uuid4()),
                                "request_id": data["request_id"],
                                "item_id": item["id"],
                                "position": item["position"],
                                "kind": "impression",
                            }
                            r = await client.post("/api/events", json=event)
                            r.raise_for_status()
                            event_latencies.append((time.perf_counter() - tick) * 1000)
                    except Exception as exc:
                        errors.append(type(exc).__name__ + ": " + str(exc))

            tick = time.perf_counter()
            await asyncio.gather(*(request(i) for i in range(160)))
            seconds = time.perf_counter() - tick
            result["loads"][str(concurrency)] = {
                "requests": 160,
                "successful": len(latencies),
                "seconds": seconds,
                "requests_per_second": len(latencies) / seconds,
                "p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
                "p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
                "event_p95_ms": float(np.percentile(event_latencies, 95))
                if event_latencies
                else None,
                "errors": errors,
            }
        result["health_after"] = (await clients[0].get("/api/health")).json()
        result["note"] = (
            "Isolated test service; mixed cold/interest/warm users, search/feed, request snapshots and test exposure writes. Shared host; not production capacity."
        )
        (root / "reports/latency.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        if any(x["errors"] for x in result["loads"].values()):
            raise RuntimeError("Benchmark requests failed; inspect saved evidence")
    finally:
        await asyncio.gather(*(c.aclose() for c in clients))


if __name__ == "__main__":
    asyncio.run(run())
