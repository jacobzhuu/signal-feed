"""Lightweight few-shot interests and validation-guarded residual fusion.

No model fitting or labels in this module. Mixed interests are not automatically
treated as unreliable users. Popularity is only a zero-signal fallback.
"""

from dataclasses import dataclass, asdict
import itertools
import numpy as np
from .coldfuse import unit, state
from .retrieval import top

VERSION = "interest-adaptation-v1"


@dataclass(frozen=True)
class AdaptConfig:
    multi: float = 0.5
    recent: float = 0.0
    side: float = 0.25

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all() or any(
            not 0 <= v <= 1 for v in asdict(self).values()
        ):
            raise ValueError("Invalid adaptation weights")


GRID = tuple(
    AdaptConfig(*v) for v in itertools.product((0, 0.5, 1), (0, 0.25), (0, 0.25, 0.5))
)


def signals(vectors, history, prior=None):
    s = state(vectors, history, prior)
    h = s.history
    if h:
        similarities = vectors @ vectors[h].T
        mean = np.average(vectors[h], axis=0, weights=np.linspace(0.5, 1, len(h)))
        centroid = vectors @ unit(mean)
        coherence = float(np.clip(np.linalg.norm(mean), 0, 1))
        agreement = (
            float(np.clip(unit(mean) @ s.explicit, 0, 1)) if s.explicit.any() else 0.0
        )
        maximum, recent = similarities.max(1), similarities[:, -1]
    else:
        centroid = maximum = recent = np.zeros(len(vectors), dtype="float32")
        coherence, agreement = 0.0, 0.0
    return dict(
        state=s,
        centroid=centroid,
        maximum=maximum,
        recent=recent,
        prior=vectors @ s.explicit,
        coherence=coherence,
        agreement=agreement,
    )


def interest_scores(sig, config, ablation=None):
    s = sig["state"]
    n = len(s.history)
    if not n:
        return sig["prior"].copy()
    multi = 0.0 if ablation == "no_multi" else config.multi
    # Max similarity retains a secondary theme; recency has a separate control.
    score = (1 - multi) * sig["centroid"] + multi * sig["maximum"]
    score = (1 - config.recent) * score + config.recent * sig["recent"]
    if s.explicit.any() and ablation != "no_prior":
        prior_weight = 3 / (n + 3)
        if ablation != "no_reliability":
            prior_weight *= sig["agreement"]
        score = (1 - prior_weight) * score + prior_weight * sig["prior"]
    return score


def safe_content(vectors, history, prior, eligible, popularity):
    s = state(vectors, history, prior)
    return top(vectors @ s.profile if s.profile.any() else popularity, eligible, 100)


def fuse_interest(sig, content, channels, config, ablation=None):
    """Return up to 100 candidates; never allow an unsupported side channel."""
    scores = {i: 1 / (61 + rank) for rank, i in enumerate(content)}
    gate = {"content": 1.0}
    if not sig["state"].history and not sig["state"].explicit.any():
        return content, scores, gate
    for name in ("itemcf", "generated"):
        ids = list(dict.fromkeys(channels.get(name, [])))
        if ablation == "no_side" or (name == "generated" and ablation == "no_gen"):
            ids = []
        strength = config.side
        if ablation not in ("fixed_side", "no_reliability"):
            # Agreement is observable support, not calibrated correctness.
            overlap = len(set(ids) & set(content)) / max(1, min(len(ids), len(content)))
            strength *= overlap
        gate[name] = strength if ids else 0.0
        if strength <= 0:
            continue
        for rank, i in enumerate(ids):
            scores[i] = scores.get(i, 0.0) + strength / (61 + rank)
    pred = sorted(scores, key=lambda i: (-scores[i], i))[:100]
    return pred, scores, gate


def cohort_key(history, prior):
    n = len(set(history))
    scenario = "prior" if np.any(prior) else "no_prior"
    band = "0" if n == 0 else "1" if n == 1 else "2-5" if n <= 5 else "6+"
    return scenario + "/" + band


class GuardedAdapter:
    """Require domain-matched evidence; a rejected group skips expensive work.

    The service can call this shared policy without adding a frontend flow.
    MIND evidence deliberately does not authorize deployment in the HN domain.
    """

    def __init__(self, decision, *, domain):
        if decision.get("version") != VERSION or decision.get("domain") != domain:
            raise ValueError("Adaptation evidence does not match policy/domain")
        self.config = AdaptConfig(**decision["config"])
        self.accepted = {
            k for k, v in decision["groups"].items() if v.get("accepted") is True
        }

    def recommend(
        self, vectors, popularity, history, prior, eligible, channel_provider=None
    ):
        group = cohort_key(history, prior)
        if group not in self.accepted:
            return dict(
                ids=safe_content(vectors, history, prior, eligible, popularity),
                status="content-fallback",
                group=group,
                channels_called=False,
            )
        sig = signals(vectors, history, prior)
        personal = bool(sig["state"].history or sig["state"].explicit.any())
        score = interest_scores(sig, self.config) if personal else popularity
        content = top(score, eligible, 100)
        call = channel_provider is not None and self.config.side > 0 and personal
        channels = channel_provider() if call else {}
        channels = {
            name: [i for i in ids if 0 <= i < len(eligible) and eligible[i]]
            for name, ids in channels.items()
        }
        ids, _, gates = fuse_interest(sig, content, channels, self.config)
        return dict(
            ids=ids, status="adaptation", group=group, gates=gates, channels_called=call
        )
