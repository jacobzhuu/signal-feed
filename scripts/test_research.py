"""Exercise real experimental inference against the isolated browser-test server."""

import json
import time
import uuid
from pathlib import Path
import httpx

with httpx.Client(
    base_url="http://127.0.0.1:8766", trust_env=False, timeout=90
) as client:
    feed = client.get("/api/feed").json()
    item = feed["items"][0]
    event = {
        "event_id": str(uuid.uuid4()),
        "request_id": feed["request_id"],
        "item_id": item["id"],
        "position": 0,
        "kind": "click",
    }
    client.post("/api/events", json=event).raise_for_status()
    runs = []
    for _ in range(8):
        start = time.perf_counter()
        response = client.get("/api/feed?research=true")
        response.raise_for_status()
        r = response.json()
        runs.append(
            {
                "strategy": r["strategy"],
                "seconds": time.perf_counter() - start,
                "items": len(r["items"]),
            }
        )
        if r["strategy"] in {"research-cache", "research-mind-transfer"}:
            break
        time.sleep(2)
    assert runs[-1]["strategy"] in {"research-cache", "research-mind-transfer"}, runs
    Path("/data1/zhuzy/signalfeed/reports/research-inference.json").write_text(
        json.dumps(runs, indent=2)
    )
    print(runs)
