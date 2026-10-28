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
    cold = os.environ.get("SIGNALFEED_BENCH_COLD") == "1"
    feed_url = "/api/feed?cold=true" if cold else "/api/feed"
    root.joinpath("reports").mkdir(parents=True, exist_ok=True)
    clients = [
        httpx.AsyncClient(base_url=url, trust_env=False, timeout=90) for _ in range(16)
    ]
    start = time.perf_counter()
    profiles = []
    try:
        health = (await clients[0].get("/api/health")).json()
        if not health.get("test_events"):
            raise ValueError(
                "Benchmark requires SIGNALFEED_TEST_EVENTS=1 on an isolated server"
            )
        for i, client in enumerate(clients):
            topics, history = [], []
            r = await client.get("/api/interests")
            r.raise_for_status()
            if i % 3:
                topics = ["Artificial Intelligence" if i % 2 else "Security"]
                r = await client.put(
                    "/api/interests",
                    json={"topics": topics},
                )
                r.raise_for_status()
            if i % 3 == 2:
                feed = (await client.get("/api/feed")).json()
                items = feed["items"]
                chosen = [
                    items[(i * 7 + j) % len(items)] for j in range(min(5, len(items)))
                ]
                for item in chosen:
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
                    history.append(item["id"])
            profiles.append(dict(client=i, topics=topics, history=history))
        result = {
            "run_id": root.name,
            "provenance": code_identity(),
            "preparation_seconds": time.perf_counter() - start,
            "users": len(clients),
            "profiles": profiles,
            "unique_profiles": len(
                {json.dumps([p["topics"], p["history"]]) for p in profiles}
            ),
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
            statuses = {}
            client_methods = [dict(feed=0, search=0) for _ in clients]

            async def request(i):
                async with semaphore:
                    client = clients[i % len(clients)]
                    # Alternate by complete client rounds, not parity of the
                    # client ID: every cohort must exercise both endpoints.
                    search = (i // len(clients)) % 2 == 1
                    client_methods[i % len(clients)][
                        "search" if search else "feed"
                    ] += 1
                    tick = time.perf_counter()
                    try:
                        response = (
                            await client.get(
                                "/api/search", params={"q": queries[i % len(queries)]}
                            )
                            if search
                            else await client.get(feed_url)
                        )
                        response.raise_for_status()
                        data = response.json()
                        status = (data.get("experiment") or {}).get("status", "search")
                        statuses[status] = statuses.get(status, 0) + 1
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
                "statuses": statuses,
                "client_methods": client_methods,
            }
        result["health_after"] = (await clients[0].get("/api/health")).json()
        result["note"] = (
            "Isolated test service; every client exercises search and feed; distinct rotated warm histories; stable profiles favor cache reuse. Request snapshots and test exposure writes included. Shared host; co-located work may affect latency; not production capacity."
        )
        (
            root / "reports" / ("cold_latency.json" if cold else "latency.json")
        ).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        if any(x["errors"] for x in result["loads"].values()):
            raise RuntimeError("Benchmark requests failed; inspect saved evidence")
    finally:
        await asyncio.gather(*(c.aclose() for c in clients))


if __name__ == "__main__":
    asyncio.run(run())
