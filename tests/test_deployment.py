"""Release operations use synthetic databases/processes only."""

import importlib.util
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import pytest


def script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backup_includes_uncheckpointed_wal_and_refuses_overwrite(tmp_path):
    module = script("backup_database")
    source, target = tmp_path / "source.sqlite", tmp_path / "backup.sqlite"
    with sqlite3.connect(source) as db:
        db.execute("PRAGMA journal_mode=WAL")
        for table in (
            "users",
            "requests",
            "events",
            "bookmarks",
            "request_snapshots",
            "articles",
        ):
            db.execute(f"CREATE TABLE {table}(id TEXT)")
            db.execute(f"INSERT INTO {table} VALUES('preserved')")
        db.commit()
        assert Path(str(source) + "-wal").stat().st_size > 0
        result = module.backup(source, target)
        assert result["integrity"] == "ok" and set(result["counts"].values()) == {1}
        with pytest.raises(FileExistsError):
            module.backup(source, target)


def test_stop_timeout_preserves_pid_and_retry_waits_for_exit(tmp_path):
    module = script("stop_server")
    ready = tmp_path / "ready"
    code = (
        "import signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path(%r).touch();time.sleep(30)"
        % str(ready)
    )
    proc = subprocess.Popen([sys.executable, "-c", code, "signalfeed.cli", "serve"])
    pidfile = tmp_path / "server.pid"
    pidfile.write_text(str(proc.pid))
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        with pytest.raises(RuntimeError, match="did not exit"):
            module.stop(pidfile, 0.1)
        assert pidfile.exists() and proc.poll() is None
    finally:
        proc.kill()
        proc.wait(timeout=3)
    module.stop(pidfile, 0.1)
    assert not pidfile.exists()


def test_stop_rejects_unrelated_pid(tmp_path):
    module = script("stop_server")
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    pidfile = tmp_path / "server.pid"
    pidfile.write_text(str(proc.pid))
    try:
        with pytest.raises(RuntimeError, match="does not identify"):
            module.stop(pidfile, 0.1)
        assert pidfile.exists() and proc.poll() is None
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=3)
