"""Explicit offline inference from persisted MIND checkpoints."""

from collections import defaultdict
import numpy as np
import torch
from .features import Features
from .generator import Generator
from .ranker import load_ranker
from .evaluation import top, fuse


def recommend(history, seed=42, limit=10):
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1, user_api="blas")
    torch.set_num_threads(2)
    features = Features().load()
    unknown = [i for i in history if i not in features.index]
    if unknown:
        raise ValueError(f"Unknown MIND IDs: {unknown[:5]}")
    request = {"history": history[-20:]}
    eligible = np.ones(len(features.news), dtype=bool)
    eligible[features.history(request)] = False
    sim, cf = features.scores(request)
    buckets = defaultdict(list)
    for i, code in enumerate(features.codes):
        if eligible[i]:
            buckets[tuple(code)].append(i)
    generated = Generator(seed).generate(
        history, features.index, features.codes, valid_codes=buckets
    )
    prefix_candidates = []
    for code, score in generated:
        matches = np.flatnonzero(
            eligible & np.all(features.codes[:, :2] == np.array(code[:2]), axis=1)
        )
        prefix_candidates.extend(matches[np.argsort(-sim[matches])[:5]].tolist())
    candidates = fuse(
        {
            "content": top(sim, eligible, 100),
            "itemcf": top(cf, eligible, 100),
            "popular": top(features.pop, eligible, 100),
            "generated": list(dict.fromkeys(prefix_candidates)),
        },
        weights={"generated": 0.5},
    )
    model = load_ranker(seed)
    with torch.inference_mode():
        scores = model(
            torch.from_numpy(features.matrix(request, candidates, stats=False))
        ).numpy()
    ordered = np.argsort(-scores)[:limit]
    return {
        "scope": "MIND offline inference over loaded catalog; not historical time-travel evaluation",
        "model_version": f"sid-{seed}/ranker-{seed}",
        "score_kind": "uncalibrated ranking logit",
        "items": [
            {
                "id": features.news[candidates[i]]["id"],
                "title": features.news[candidates[i]]["title"],
                "score": float(scores[i]),
            }
            for i in ordered
        ],
    }
