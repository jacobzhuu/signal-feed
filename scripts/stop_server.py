"""Stop only the recorded server and its verified children; fail closed on timeout."""

import argparse
import os
from pathlib import Path
import signal
import time


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def descendants(pid):
    result = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) == pid:
                child = int(entry.name)
                result[child] = identity(child)
                result.update(descendants(child))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def stop(file, timeout):
    if not file.exists():
        return
    pid = int(file.read_text().strip())
    stamp = identity(pid)
    if stamp is None:
        file.unlink()
        return
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    if not (b"signalfeed serve" in command or b"signalfeed.cli serve" in command):
        raise RuntimeError("PID does not identify SignalFeed; preserving PID file")
    children = descendants(pid)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while identity(pid) == stamp and time.monotonic() < deadline:
        children.update(descendants(pid))
        time.sleep(0.1)
    if identity(pid) == stamp:
        raise RuntimeError(
            "Server did not exit; preserving PID file. Do not start another server"
        )
    # A worker/resource tracker can outlive its parent. Never signal a reused PID.
    for child, child_stamp in children.items():
        if child_stamp is not None and identity(child) == child_stamp:
            try:
                os.kill(child, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not any(s is not None and identity(p) == s for p, s in children.items()):
            file.unlink()
            return
        time.sleep(0.1)
    raise RuntimeError(
        "Server child still running; preserving PID file for investigation"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pidfile", type=Path)
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    stop(args.pidfile, args.timeout)
