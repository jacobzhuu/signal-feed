"""Immutable, read-only feedback capture; optional bounded daily observation."""

import argparse
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from signalfeed import temporal_audit, snapshots
from signalfeed.temporal_audit import audit, backup, load_protocol, stamp


def checksum(file):
    with Path(file).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(file, obj):
    with file.open("x") as stream:
        json.dump(obj, stream, indent=2)


def source_identity(protocol):
    return {
        name: checksum(file)
        for name, file in (
            ("audit_feedback.py", __file__),
            ("temporal_audit.py", temporal_audit.__file__),
            ("snapshots.py", snapshots.__file__),
            ("protocol.json", protocol),
        )
    }


def collect(database, output, protocol_file, as_of):
    if as_of > time.time():
        raise ValueError("Cannot mature future feedback")
    protocol = load_protocol(protocol_file)
    source = source_identity(protocol_file)
    start = time.perf_counter()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(output, 0o700)
    try:
        backup(database, output / "source.sqlite")
        report, reserved = audit(output / "source.sqlite", as_of, protocol)
        report["capture"] = dict(
            source_database=str(database.resolve()),
            database_sha256=checksum(output / "source.sqlite"),
            protocol=protocol,
            algorithm=source,
            elapsed_seconds=time.perf_counter() - start,
            source_open_mode="ro; query_only; SQLite backup",
            snapshot_contains_private_anonymous_activity=True,
        )
        write(output / "audit.json", report)
        with (output / "reserved-users.jsonl").open("x") as stream:
            for row in reserved:
                stream.write(json.dumps(row) + "\n")
        os.chmod(output / "reserved-users.jsonl", 0o600)
        if source_identity(protocol_file) != source:
            raise ValueError("Observer source changed during audit")
        write(
            output / "completed.json",
            dict(
                status="complete",
                audit_sha256=checksum(output / "audit.json"),
                reserved_sha256=checksum(output / "reserved-users.jsonl"),
            ),
        )
        print(
            json.dumps(
                dict(
                    output=str(output),
                    readiness=report["readiness"],
                    events=report["events"],
                    mature_exposed_users=report["mature_exposed_users"],
                )
            ),
            flush=True,
        )
        return report
    except Exception as exc:
        write(
            output / "failure.json",
            dict(
                error=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            ),
        )
        raise


def next_tick(now):
    current = datetime.fromtimestamp(now, timezone.utc)
    tick = current.replace(hour=0, minute=10, second=0, microsecond=0)
    if tick.timestamp() <= now:
        tick += timedelta(days=1)
    return tick.timestamp()


def watch(database, output, protocol_file):
    protocol = load_protocol(protocol_file)
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(output, 0o700)
    with (output / "observer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = source_identity(protocol_file)
        record = output / "observer.json"
        if record.exists():
            if json.loads(record.read_text())["algorithm"] != frozen:
                raise ValueError("Changed observer/protocol; use a fresh output root")
        else:
            write(
                record,
                dict(
                    algorithm=frozen,
                    close_at=protocol["close_at"],
                    schedule="00:10 UTC daily, immediate first capture",
                    training=False,
                    promotion=False,
                ),
            )
        end = stamp(protocol["close_at"])
        while True:
            now = time.time()
            if source_identity(protocol_file) != frozen:
                raise ValueError("Observer source changed; refusing unsealed execution")
            name = datetime.fromtimestamp(now, timezone.utc).strftime(
                "%Y%m%dT%H%M%S.%fZ"
            )
            try:
                collect(database, output / name, protocol_file, now)
            except Exception:
                traceback.print_exc()
            if now >= end:
                print(
                    "Observation window closed; no training or automatic extension.",
                    flush=True,
                )
                break
            target = min(next_tick(now), end)
            print(
                json.dumps(
                    dict(
                        next_capture_utc=datetime.fromtimestamp(
                            target, timezone.utc
                        ).isoformat()
                    )
                ),
                flush=True,
            )
            while time.time() < target:
                time.sleep(min(60, max(0, target - time.time())))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=Path("configs/temporal_v4.json"))
    p.add_argument("--as-of", type=float)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--log", type=Path)
    args = p.parse_args()
    with ExitStack() as stack:
        if args.log:
            stream = stack.enter_context(args.log.open("a", buffering=1))
            stack.enter_context(redirect_stdout(stream))
            stack.enter_context(redirect_stderr(stream))
        if args.watch:
            if args.as_of is not None:
                p.error("--as-of cannot be combined with --watch")
            watch(args.db, args.output, args.protocol)
        else:
            collect(
                args.db,
                args.output,
                args.protocol,
                time.time() if args.as_of is None else args.as_of,
            )
