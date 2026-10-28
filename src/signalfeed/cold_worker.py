"""Single persistent research worker; bounded queue, cache and hard deadline."""

import hashlib
import json
import multiprocessing
import threading
import time
import numpy as np
from .settings import path
from .provenance import sha256

MAX_RUNTIME = 30
_lock = threading.Lock()
_process = _connection = _pending = _timer = None
_version = None
_cache = {}


def model_identity():
    files = [path("mind") / "quantizer.npz"]
    for name in ("sid-42", "coldfuse-42"):
        files += [
            path("models/" + name) / "model.pt",
            path("models/" + name) / "bundle.json",
        ]
    files += [path("models/coldfuse-42") / "train.json"]
    gate = path("models") / "coldfuse.json"
    if gate.exists():
        files.append(gate)
    return hashlib.sha256(repr([sha256(f) for f in files]).encode()).hexdigest()


def cache_key(payload, version):
    h = hashlib.sha256(version.encode())
    for key in sorted(payload):
        value = payload[key]
        h.update(key.encode())
        h.update(
            value.tobytes() if isinstance(value, np.ndarray) else repr(value).encode()
        )
    return h.hexdigest()


def _worker(connection):
    try:
        import torch
        from threadpoolctl import threadpool_limits
        from .coldfuse import ColdFuse, Config
        from .cold_training import load
        from .generator import Generator
        from .sid import ResidualCodes

        torch.set_num_threads(2)
        threadpool_limits(limits=1, user_api="blas")
        generator, ranker = Generator(), load()
        quantizer = ResidualCodes.load(path("mind") / "quantizer.npz")
        gate = path("models") / "coldfuse.json"
        config = (
            Config(**json.loads(gate.read_text())["config"])
            if gate.exists()
            else Config()
        )
        catalog_key, codes = None, None
        while True:
            payload = connection.recv()
            x = payload["vectors"]
            key = hashlib.sha256(x.tobytes()).hexdigest()
            if key != catalog_key:
                codes, catalog_key = quantizer.encode(x), key
            core = ColdFuse(x, payload["popularity"], codes=codes, generator=generator)
            result = core.recommend(
                payload["history"],
                payload["eligible"],
                payload["explicit"],
                config,
                ranker,
                cf_scores=payload["cf_scores"],
            )
            connection.send((True, result))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        connection.send((False, type(exc).__name__ + ": " + str(exc)))
    finally:
        connection.close()


def _stop():
    global _process, _connection, _pending, _timer, _version
    if _timer is not None:
        _timer.cancel()
    if _process is not None:
        if _process.is_alive():
            _process.terminate()
        _process.join(0.2)
        if _process.is_alive():
            _process.kill()
            _process.join(0.2)
    if _connection is not None:
        _connection.close()
    _process = _connection = _pending = _timer = _version = None


def _collect():
    global _pending
    if _pending is None or not _pending["done"].is_set():
        return
    job = _pending
    ok, result = job["result"]
    if ok:
        _cache[job["key"]] = (job["finished"], result)
        _timer.cancel()
        _pending = None
    else:
        _stop()


def _expire(job):
    with _lock:
        if _pending is job and _timer is threading.current_thread():
            if job["done"].is_set():
                _collect()
            else:
                _stop()


def _exchange(connection, payload, job):
    # Neither a large send nor a partial response can block the request thread
    # or prevent the deadline thread from terminating this child.
    try:
        connection.send(payload)
        job["result"] = connection.recv()
    except (OSError, EOFError, TypeError) as exc:
        job["result"] = (False, str(exc))
    job["finished"] = time.monotonic()
    job["done"].set()
    with _lock:
        if _pending is job:
            _collect()


def shutdown():
    with _lock:
        _stop()
        _cache.clear()


def recommend(payload, timeout=2):
    global _process, _connection, _pending, _timer, _version
    if not _lock.acquire(blocking=False):
        return None, "busy-fallback", None
    try:
        version = model_identity()
        key = cache_key(payload, version)
        if _version is not None and _version != version:
            _stop()
            _cache.clear()
        _collect()
        now = time.monotonic()
        if key in _cache and now - _cache[key][0] < 300:
            return _cache[key][1], "cache", version
        if _pending is not None:
            return None, "busy-fallback", version
        if _process is None or not _process.is_alive():
            _stop()
            context = multiprocessing.get_context("spawn")
            _connection, child = context.Pipe()
            _process = context.Process(target=_worker, args=(child,), daemon=True)
            _process.start()
            child.close()
            _version = version
        if len(_cache) >= 128:
            _cache.pop(next(iter(_cache)))
        job = dict(key=key, done=threading.Event())
        _pending = job
        _timer = threading.Timer(MAX_RUNTIME, _expire, args=(job,))
        _timer.daemon = True
        _timer.start()

        threading.Thread(
            target=_exchange, args=(_connection, payload, job), daemon=True
        ).start()
        if job["done"].wait(min(timeout, MAX_RUNTIME)):
            _collect()
            if key in _cache:
                return _cache[key][1], "research", version
            return None, "model-error-fallback", version
        return None, "timeout-fallback", version
    except Exception:
        _stop()
        return None, "unavailable-fallback", None
    finally:
        _lock.release()
