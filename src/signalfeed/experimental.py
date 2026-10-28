"""Isolated MIND-to-HN research inference; never automatically promoted."""

from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing
import hashlib
import time
import threading
import numpy as np

_pool = None
_cache = {}
_pending = None
_lock = threading.Lock()


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


def _recommend(payload, timeout=2):
    global _pool, _pending
    items, vectors, history = payload
    key = hashlib.sha256(repr(([a["id"] for a in items], history)).encode()).hexdigest()
    now = time.monotonic()
    if key in _cache and now - _cache[key][0] < 300:
        return _cache[key][1], "research-cache"
    if _pending is not None:
        old_key, future = _pending
        if future.done():
            try:
                _cache[old_key] = (now, future.result())
            except Exception:
                pass
            _pending = None
            if key in _cache:
                return _cache[key][1], "research-cache"
        else:
            return [], "research-busy-fallback"
    if len(_cache) > 128:
        _cache.clear()
    if _pool is None:
        _pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    future = _pool.submit(research_rank, payload)
    _pending = (key, future)
    try:
        result = future.result(timeout=timeout)
        _cache[key] = (now, result)
        _pending = None
        return result, "research-mind-transfer"
    except TimeoutError:
        return [], "research-timeout-fallback"
    except Exception:
        _pending = None
        return [], "research-unavailable-fallback"


def recommend(payload, timeout=2):
    if not _lock.acquire(blocking=False):
        return [], "research-busy-fallback"
    try:
        return _recommend(payload, timeout)
    finally:
        _lock.release()
