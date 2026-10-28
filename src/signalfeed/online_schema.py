"""Online feature and composed-policy contract, independent of MIND models."""

import numpy as np
import torch
from .settings import ENCODER_MODEL, ENCODER_REVISION

FEATURE_VERSION = "online-text-v2"
LABEL_WINDOW = 86400
MIN_HISTORY = 5
CONTRACT = {
    "feature_version": FEATURE_VERSION,
    "input_dim": 1158,
    "encoder_model": ENCODER_MODEL,
    "encoder_revision": ENCODER_REVISION,
    "architecture": "Ranker(1158,128,64,1)",
    "policy_version": "cold-fallback-v1",
    "minimum_history": MIN_HISTORY,
}


def online_matrix(vectors, candidates, history, explicit=None, content_scores=None):
    h = history[-20:]
    x = vectors[candidates]
    explicit = (
        np.zeros(vectors.shape[1], dtype="float32") if explicit is None else explicit
    )
    profile = (
        np.average(vectors[h], axis=0, weights=np.linspace(0.5, 1, len(h)))
        if h
        else np.zeros(vectors.shape[1])
    )
    profile = profile + explicit
    profile /= max(np.linalg.norm(profile), 1e-8)
    if h:
        att = x @ vectors[h].T * 5
        att = np.exp(att - att.max(1, keepdims=True))
        att /= att.sum(1, keepdims=True)
        interest = att @ vectors[h] + explicit
        recent = x @ vectors[h[-1]]
    else:
        interest = np.broadcast_to(explicit, x.shape).copy()
        recent = np.zeros(len(x))
    interest /= np.maximum(np.linalg.norm(interest, axis=1, keepdims=True), 1e-8)
    extras = np.stack(
        [
            x @ profile,
            recent,
            x @ explicit,
            np.full(len(x), len(h) / 20),
            np.isin(candidates, h).astype(float),
            np.zeros(len(x)) if content_scores is None else content_scores,
        ],
        1,
    )
    return np.concatenate([x, interest, x * interest, extras], 1).astype("float32")


def policy_scores(model, features, content_scores):
    """Cold/sparse users retain the complete interest-aware content policy."""
    scores = np.asarray(content_scores, dtype="float32").copy()
    warm = features[:, -3] >= MIN_HISTORY / 20
    if model is not None and warm.any():
        with torch.inference_mode():
            scores[warm] = model(torch.from_numpy(features[warm])).numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite policy scores")
    return scores
