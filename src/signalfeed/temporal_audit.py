"""Read-only feedback feasibility, not recommendation efficacy or publication."""

from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import zlib
import numpy as np
from .snapshots import unpack


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def day(value):
    return datetime.fromtimestamp(value, timezone.utc).date().isoformat()


def load_protocol(file):
    p = json.loads(Path(file).read_text())
    if p["version"] != "temporal-readiness-v4" or p["label_window_seconds"] != 86400:
        raise ValueError("Unsupported collection protocol")
    if (
        p["efficacy_authorized"]
        or p["automatic_training"]
        or p["automatic_promotion"]
        or p["algorithm_registered"]
    ):
        raise ValueError("Readiness collection cannot authorize training or efficacy")
    previous_end, buckets = None, set()
    for phase in ("select", "guard", "test"):
        window = p["phases"][phase]
        begin, end = stamp(window["start"]), stamp(window["end"])
        if begin >= end or (
            previous_end is not None
            and begin < previous_end + p["label_window_seconds"]
        ):
            raise ValueError("Phases overlap their label windows")
        if buckets.intersection(window["buckets"]):
            raise ValueError("User buckets overlap")
        buckets.update(window["buckets"])
        previous_end = end
    if (
        buckets != set(range(10))
        or stamp(p["close_at"]) < previous_end + p["label_window_seconds"]
    ):
        raise ValueError("Incomplete user partition or label maturity")
    return p


def readonly(file):
    c = sqlite3.connect(
        Path(file).resolve().as_uri() + "?mode=ro", uri=True, timeout=15
    )
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA query_only=ON")
    return c


def backup(source, destination):
    """SQLite backup includes committed WAL without copying a live file pair."""
    if destination.exists():
        raise FileExistsError(destination)
    destination.touch(mode=0o600, exist_ok=False)
    src = readonly(source)
    try:
        with sqlite3.connect(destination) as dest:
            src.backup(dest)
    finally:
        src.close()


def band(n):
    return (
        "0"
        if n == 0
        else "1-5"
        if n <= 5
        else "6-9"
        if n < 10
        else "10"
        if n == 10
        else "11+"
    )


def phase_for(user, created, protocol):
    bucket = (
        int(
            hashlib.sha256((protocol["user_split_salt"] + user).encode()).hexdigest(),
            16,
        )
        % 10
    )
    for phase, window in protocol["phases"].items():
        if bucket in window["buckets"] and stamp(window["start"]) <= created < stamp(
            window["end"]
        ):
            return phase
    return None


def snapshot_status(request, items, prior, protocol):
    if request["payload"] is None:
        return "missing_snapshot", None
    try:
        s = unpack(request["payload"])
        if (
            s.get("feature_version") != protocol["snapshot_feature_version"]
            or s.get("created") != request["created"]
            or s.get("model_version") != request["model_version"]
        ):
            return "incompatible_snapshot", None
        history = s["history"]
        if (
            not isinstance(history, list)
            or any(not isinstance(i, str) for i in history)
            or len(history) > 20
            or len(set(history)) != len(history)
            or not set(history).issubset(prior)
        ):
            return "invalid_snapshot_history", None
        if not items:
            return "empty_returned_list", None
        for item in items:
            features = np.asarray(s["features"][item], dtype="float32")
            score = s["content_scores"][item]
            if (
                features.shape != (1158,)
                or not np.isfinite(features).all()
                or not np.isfinite(score)
                or not np.isclose(features[-3], len(history) / 20, atol=1e-7)
                or not np.isclose(features[-1], score, atol=1e-6)
            ):
                return "invalid_snapshot_features", None
        return "compatible", s
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
        AttributeError,
        zlib.error,
        struct.error,
    ):
        return "corrupt_snapshot", None


def audit(database, as_of, protocol):
    from .feedback_policy import load_feedback_policy

    policy = load_feedback_policy()
    if not np.isfinite(as_of):
        raise ValueError("Invalid audit time")
    c = readonly(database)
    try:
        c.execute("BEGIN")
        inventory = {
            name: c.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            for name in ("users", "requests", "events", "request_snapshots")
        }
        requests = {
            r["id"]: dict(r)
            for r in c.execute(
                "SELECT * FROM requests WHERE created<=? ORDER BY created,id", (as_of,)
            )
        }
        events = [
            dict(e)
            for e in c.execute(
                "SELECT * FROM events WHERE created<=? ORDER BY created,id", (as_of,)
            )
        ]
        item_lists = {}
        malformed_requests = set()
        for rid, request in requests.items():
            try:
                items = json.loads(request["items"])
                if (
                    not isinstance(items, list)
                    or any(not isinstance(i, str) for i in items)
                    or len(set(items)) != len(items)
                ):
                    raise ValueError("Invalid returned items")
                item_lists[rid] = items
            except (ValueError, TypeError):
                malformed_requests.add(rid)
        event_counts = Counter()
        per_request, history = defaultdict(list), defaultdict(list)
        test_users = {
            e["user_id"]
            for e in events
            if e["is_test"] or e["user_id"] in policy.user_ids
        } | {r["user_id"] for r in requests.values() if r["user_id"] in policy.user_ids}
        seen_event_keys = set()
        delays = defaultdict(list)
        for e in events:
            event_counts["all_visible_events"] += 1
            if e["is_test"] or e["user_id"] in policy.user_ids:
                event_counts["test_events"] += 1
                continue
            event_counts["non_test_" + e["kind"]] += 1
            r = requests.get(e["request_id"])
            items = item_lists.get(e["request_id"], [])
            if (
                r is None
                or r["user_id"] != e["user_id"]
                or e["created"] < r["created"]
                or not 0 <= e["position"] < len(items)
                or items[e["position"]] != e["item_id"]
            ):
                event_counts["invalid_event_identity_position_time"] += 1
                continue
            if e["kind"] not in ("click", "impression", "bookmark", "unbookmark"):
                event_counts["unknown_kind"] += 1
                continue
            key = (e["request_id"], e["item_id"], e["kind"])
            if e["kind"] in ("impression", "click") and key in seen_event_keys:
                event_counts["duplicate_exposure_click"] += 1
                continue
            seen_event_keys.add(key)
            if e["kind"] in ("click", "bookmark"):
                history[e["user_id"]].append((e["created"], e["item_id"]))
            if e["kind"] in ("click", "impression"):
                latency = e["created"] - r["created"]
                delays[e["kind"]].append(latency)
                if latency >= protocol["label_window_seconds"]:
                    event_counts["outside_24h_" + e["kind"]] += 1
                else:
                    per_request[e["request_id"]].append(e)
        timestamps = {u: [t for t, _ in h] for u, h in history.items()}
        daily, quality = defaultdict(Counter), Counter()
        selected, selected_users = [], set()
        all_mature_users, all_clicked_users = set(), set()
        payloads = c.execute(
            "SELECT r.*,s.payload FROM requests r LEFT JOIN request_snapshots s ON s.request_id=r.id WHERE r.created<=? ORDER BY r.created,r.id",
            (as_of,),
        )
        for request in payloads:
            rid, user = request["id"], request["user_id"]
            if (
                request["query"]
                or request["strategy"] != protocol["baseline_strategy"]
                or request["model_version"] != protocol["baseline_strategy"]
            ):
                quality["non_baseline_feed_requests"] += 1
                continue
            quality["baseline_feed_requests"] += 1
            end = request["created"] + protocol["label_window_seconds"]
            mature = end <= as_of
            quality["mature_feed_requests" if mature else "immature_feed_requests"] += 1
            if user in test_users:
                quality["test_contaminated_user_requests"] += 1
                continue
            if rid in malformed_requests:
                quality["invalid_returned_items"] += 1
                continue
            before = bisect_left(timestamps.get(user, []), request["created"])
            prior = [i for _, i in history.get(user, [])[:before]]
            raw_n, unique_n = len(prior), len(set(prior))
            group = band(raw_n)
            d = daily[day(request["created"]), group]
            d["requests"] += 1
            d["mature_requests"] += int(mature)
            status, snapshot = snapshot_status(
                request, item_lists[rid], set(prior), protocol
            )
            quality[status] += 1
            d[status] += 1
            if status != "compatible":
                continue
            if raw_n != unique_n:
                quality["requests_with_repeated_positive_actions"] += 1
            if len(snapshot["history"]) != unique_n:
                quality["history_capped_or_catalog_filtered"] += 1
            feedback = per_request[rid]
            views = {e["item_id"] for e in feedback if e["kind"] == "impression"}
            clicks = {e["item_id"] for e in feedback if e["kind"] == "click"}
            exposed = views | clicks
            if mature:
                d["usable_exposures"] += len(exposed)
                d["viewable_exposures"] += len(views)
                d["click_inferred_exposures"] += len(clicks - views)
                d["clicked_exposures"] += len(clicks)
                if exposed:
                    d["usable_requests"] += 1
                    all_mature_users.add(user)
                if clicks:
                    all_clicked_users.add(user)
            phase = phase_for(user, request["created"], protocol)
            # Reserve before looking at feedback/maturity. Never replace an
            # unexposed first request by a later clicked request from this user.
            if (
                phase
                and group in protocol["primary_bands"]
                and user not in selected_users
            ):
                selected_users.add(user)
                selected.append(
                    dict(
                        user_hash=hashlib.sha256(
                            ("temporal-v4/private/" + user).encode()
                        ).hexdigest(),
                        request_hash=hashlib.sha256(rid.encode()).hexdigest(),
                        phase=phase,
                        band=group,
                        date=day(request["created"]),
                        created=request["created"],
                        label_end=end,
                        raw_positive_actions=raw_n,
                        unique_items=unique_n,
                        snapshot_history=len(snapshot["history"]),
                        mature=mature,
                        exposures=len(exposed) if mature else None,
                        clicks=len(clicks) if mature else None,
                    )
                )
        phase_counts = {}
        minimum = protocol["minimum_pilot"]
        for phase in ("select", "guard", "test"):
            phase_counts[phase] = {}
            for group in protocol["primary_bands"]:
                reserved = [
                    r for r in selected if r["phase"] == phase and r["band"] == group
                ]
                usable = [r for r in reserved if r["mature"] and r["exposures"]]
                dates = sorted({r["date"] for r in usable})
                clicked = sum(bool(r["clicks"]) for r in usable)
                enough = (
                    len(usable) >= minimum["users_per_band_per_phase"]
                    and len(dates) >= minimum["dates_per_band_per_phase"]
                    and clicked >= minimum["clicked_users_per_band_per_phase"]
                )
                phase_counts[phase][group] = dict(
                    reserved_users=len(reserved),
                    mature_exposed_users=len(usable),
                    dates=dates,
                    clicked_users=clicked,
                    pilot_minimum_met=enough,
                )
        enough = all(
            v["pilot_minimum_met"]
            for groups in phase_counts.values()
            for v in groups.values()
        )
        closed = as_of >= stamp(protocol["close_at"])
        status = (
            "pilot_minimum_met_replay_missing"
            if enough
            else "closed_insufficient_feedback"
            if closed
            else "collecting_insufficient_feedback"
        )
        return dict(
            version=protocol["version"],
            as_of=as_of,
            as_of_utc=datetime.fromtimestamp(as_of, timezone.utc).isoformat(),
            inventory=inventory,
            events=dict(event_counts),
            requests=dict(quality),
            source_test_users=len(test_users),
            feedback_policy=policy.metadata(),
            mature_exposed_users=len(all_mature_users),
            mature_clicked_users=len(all_clicked_users),
            daily={f"{d}/{g}": dict(v) for (d, g), v in sorted(daily.items())},
            phases=phase_counts,
            feedback_intervals_seconds={
                k: dict(
                    n=len(v),
                    p50=float(np.percentile(v, 50)),
                    p95=float(np.percentile(v, 95)),
                )
                for k, v in delays.items()
            },
            readiness=dict(
                status=status,
                feedback_pilot_minimum_met=enough,
                full_catalog_replay_supported=False,
                efficacy_authorized=False,
                reason="Returned-item snapshots lack a request-time full catalog and complete replay inputs",
            ),
            notes=[
                "Counts marked non-test do not prove human activity or signup status",
                "Anonymous user table has no creation time; inventory is capture-time only",
                "Daily CTR is observational on logged exposure support, not counterfactual policy gain",
                "Feedback interval is request-to-server-receipt, not client/network reporting delay",
                "Strict whole-request snapshot validation; training may retain individual valid feature rows",
            ],
        ), selected
    finally:
        c.rollback()
        c.close()
