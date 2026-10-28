"""Explicit offline inference from persisted MIND checkpoints."""

import numpy as np
import torch
from .features import Features
from .generator import Generator
from .ranker import load_ranker
from .retrieval import RetrievalPolicy


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
    policy = RetrievalPolicy(features, Generator(seed))
    candidates = policy.candidates(request, eligible)["adaptive_fusion"]
    model = load_ranker(seed)
    with torch.inference_mode():
        scores = model(
            torch.from_numpy(features.matrix(request, candidates, stats=False))
        ).numpy()
    ordered = np.argsort(-scores, kind="stable")[:limit]
    return {
        "scope": "MIND offline inference over loaded catalog; not historical time-travel evaluation",
        "fusion_gate": policy.gate,
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
