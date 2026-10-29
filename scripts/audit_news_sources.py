"""Bounded metadata inspection of exactly two official news data sources."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zlib
import httpx

SOURCES = {
    "ebnerd-home.html": "https://recsys.eb.dk/",
    "ebnerd-description.html": "https://recsys.eb.dk/dataset/",
    "ebnerd-terms.pdf": "https://recsys.eb.dk/assets/pdf/general_license_terms.pdf",
    "adressa-home.html": "https://reclab.idi.ntnu.no/dataset/",
    "adressa-description.pdf": "https://reclab.idi.ntnu.no/dataset/cxdataset_v5.pdf",
    "adressa-license.html": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
}


def fetch(client, url, file, cap, *, prefix=False):
    headers = (
        {"Range": f"bytes=0-{cap - 1}", "Accept-Encoding": "identity"} if prefix else {}
    )
    read, stopped = 0, False
    with client.stream("GET", url, headers=headers) as response, file.open("xb") as out:
        response.raise_for_status()
        for chunk in response.iter_bytes(65536):
            keep = chunk[: cap - read]
            out.write(keep)
            read += len(keep)
            if read == cap:
                stopped = True
                break
        meta = dict(
            url=url,
            status=response.status_code,
            final_url=str(response.url),
            bytes=read,
            stopped_at_cap=stopped,
            etag=response.headers.get("etag"),
            content_range=response.headers.get("content-range"),
        )
    meta["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    if stopped and not prefix:
        raise ValueError("Metadata document exceeds cap")
    return meta


def first_tar_lines(compressed, limit=500):
    """Read bounded complete JSON lines from a truncated gzip/tar prefix."""
    raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(
        compressed, 16 * 1024 * 1024
    )
    offset = 0
    while offset + 512 <= len(raw):
        header = raw[offset : offset + 512]
        if not header.strip(b"\0"):
            break
        size = int(header[124:136].strip(b"\0 ") or b"0", 8)
        name = header[:100].split(b"\0")[0].decode("utf-8")
        if header[156:157] in (b"0", b"\0"):
            body = raw[offset + 512 : min(offset + 512 + size, len(raw))]
            lines = body.split(b"\n")[:-1]
            rows = [json.loads(line) for line in lines[:limit] if line.strip()]
            return name, rows
        offset += 512 + ((size + 511) // 512) * 512
    raise ValueError("No regular member in bounded tar prefix")


def audit(output, previous):
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        for name, url in SOURCES.items():
            files[name] = fetch(client, url, output / name, 4 * 1024 * 1024)
        prefix = output / "adressa-prefix.tar.gz.partial"
        files[prefix.name] = fetch(
            client,
            "https://reclab.idi.ntnu.no/dataset/one_week.tar.gz",
            prefix,
            2 * 1024 * 1024,
            prefix=True,
        )
    member, rows = first_tar_lines(prefix.read_bytes())
    old_news = json.loads((previous / "dataset/news.json").read_text())
    old_titles = {n["title"].strip().casefold() for n in old_news}
    fields = Counter(k for row in rows for k in row)
    dates = Counter(
        datetime.fromtimestamp(r["time"], timezone.utc).date().isoformat()
        for r in rows
        if isinstance(r.get("time"), (float, int))
    )
    report = dict(
        reviewed_sources=["EB-NeRD", "Adressa"],
        files=files,
        ebnerd=dict(
            downloaded_behavior_rows=0,
            access="No dataset downloaded: official terms require name/email/affiliation and prior acceptance; none submitted",
            primary_natural_cold_start_eligible=False,
            reasons=[
                "Future activity-selected population",
                "Static 21-day histories must not be called complete lifetime history",
                "Official test omits click labels",
                "Danish vs current English-oriented encoder and HN domain",
            ],
        ),
        adressa_prefix=dict(
            member=member,
            rows=len(rows),
            fields=dict(fields),
            dates=dict(dates),
            title_rows=sum(bool(r.get("title")) for r in rows),
            exact_title_matches_prior_catalog=sum(
                r.get("title", "").strip().casefold() in old_titles for r in rows
            ),
            prior_news_sha256=hashlib.sha256(
                (previous / "dataset/news.json").read_bytes()
            ).hexdigest(),
            impression_list_present=any(
                "article_ids_inview" in r or "impressions" in r for r in rows
            ),
            sampling="First complete JSON lines of first regular archive member; non-random, not a population sample",
            primary_natural_cold_start_eligible=False,
            reasons=[
                "Visited-page events do not provide displayed non-click labels",
                "Window start left-censors user history; no registration time",
                "Norwegian texts; incomplete historical content versions",
            ],
        ),
        independence="Different publisher/date/ID namespaces are not proof of person-level non-overlap; never join anonymized IDs across releases",
        decision="No new efficacy dataset admitted; no full download or model fitting",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    with (output / "audit.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous", type=Path, default=Path("artifacts/adaptation-v2"))
    a = p.parse_args()
    audit(a.output, a.previous)
