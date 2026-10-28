"""Isolated MIND-to-HN research inference; never automatically promoted."""

import multiprocessing
import hashlib
import time
import threading
import numpy as np
from .settings import path
from .provenance import sha256

_cache = {}
_pending = None
_lock = threading.Lock()
MAX_RUNTIME = 30


def research_rank(payload):
    import torch
    from .generator import Generator
    from .ranker import load_ranker
    from .sid import ResidualCodes
    from .settings import path

    torch.set_num_threads(2)
    items, vectors, history = payload
    codes = ResidualCodes.load(path("mind") / "quantizer.npz").encode(vectors)
    index = {a["id"]: i for i, a in enumerate(items)}
    h = [index[i] for i in history if i in index][-20:]
    if not h:
        return []
    profile = vectors[h].mean(0)
    profile /= max(np.linalg.norm(profile), 1e-8)
    similarity = vectors @ profile
    generated = Generator().generate(
        history, index, codes, valid_codes=set(map(tuple, codes))
    )
    candidates = list(np.argsort(-similarity)[:80])
    weights = {}
    for code, score in generated:
        bucket = np.flatnonzero(np.all(codes[:, :2] == np.array(code[:2]), axis=1))
        chosen = bucket[np.argsort(-similarity[bucket])[:5]]
        candidates.extend(chosen.tolist())
        for i in chosen:
            weights[int(i)] = float(score)
    ids = list(dict.fromkeys(candidates))
    x = vectors[ids]
    attention = x @ vectors[h].T * 5
    attention = np.exp(attention - attention.max(1, keepdims=True))
    attention /= attention.sum(1, keepdims=True)
    interests = attention @ vectors[h]
    features = np.concatenate(
        [
            x,
            interests,
            x * interests,
            np.stack(
                [
                    x @ profile,
                    x @ vectors[h[-1]],
                    np.zeros(len(ids)),
                    np.full(len(ids), len(h) / 20),
                    np.zeros(len(ids)),
                    np.zeros(len(ids)),
                ],
                1,
            ),
        ],
        1,
    ).astype("float32")
    with torch.inference_mode():
        scores = load_ranker()(torch.from_numpy(features)).numpy()
    return [items[ids[i]]["id"] for i in np.argsort(-scores)]


def cache_key(payload):
    items, vectors, history = payload
    models = [
        sha256(path("mind") / "quantizer.npz"),
        sha256(path("models/sid-42") / "model.pt"),
        sha256(path("models/ranker-42") / "model.pt"),
    ]
    h = hashlib.sha256(
        repr(([(a["id"], a["title"]) for a in items], history, models)).encode()
    )
    h.update(np.asarray(vectors, dtype="float32").tobytes())
    return h.hexdigest()


def _worker(connection, payload):
    try:
        connection.send((True, research_rank(payload)))
    except Exception as exc:
        connection.send((False, type(exc).__name__ + ": " + str(exc)))
    finally:
        connection.close()


def _stop(pending):
    pending["timer"].cancel()
    process = pending["process"]
    if process.is_alive():
        process.terminate()
    process.join(timeout=0.2)
    if process.is_alive():
        process.kill()
        process.join(timeout=0.2)
    pending["connection"].close()


def _expire(process):
    global _pending
    with _lock:
        if _pending is not None and _pending["process"] is process:
            _stop(_pending)
            _pending = None


def shutdown():
    global _pending
    with _lock:
        if _pending is not None:
            _stop(_pending)
            _pending = None
        _cache.clear()


def _collect():
    global _pending
    if _pending is None:
        return
    pending = _pending
    if pending["connection"].poll():
        try:
            ok, result = pending["connection"].recv()
            if ok:
                _cache[pending["key"]] = (time.monotonic(), result)
        except (EOFError, OSError):
            pass
        _stop(pending)
        _pending = None
    elif (
        not pending["process"].is_alive()
        or time.monotonic() - pending["started"] >= MAX_RUNTIME
    ):
        _stop(pending)
        _pending = None


def _recommend(payload, timeout=2):
    global _pending
    key = cache_key(payload)
    now = time.monotonic()
    _collect()
    if key in _cache and now - _cache[key][0] < 300:
        return _cache[key][1], "research-cache"
    if _pending is not None:
        return [], "research-busy-fallback"
    if len(_cache) > 128:
        _cache.clear()
    context = multiprocessing.get_context("spawn")
    connection, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child, payload), daemon=True)
    process.start()
    child.close()
    timer = threading.Timer(MAX_RUNTIME, _expire, args=(process,))
    timer.daemon = True
    _pending = dict(
        key=key, process=process, connection=connection, started=now, timer=timer
    )
    timer.start()
    if connection.poll(min(timeout, MAX_RUNTIME)):
        _collect()
        if key in _cache:
            return _cache[key][1], "research-mind-transfer"
        return [], "research-unavailable-fallback"
    return [], "research-timeout-fallback"


def recommend(payload, timeout=2):
    if not _lock.acquire(blocking=False):
        return [], "research-busy-fallback"
    try:
        return _recommend(payload, timeout)
    except Exception:
        return [], "research-unavailable-fallback"
    finally:
        _lock.release()
