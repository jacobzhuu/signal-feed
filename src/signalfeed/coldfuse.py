"""Shared label-free ColdFuse policy. All histories are oldest to newest."""

from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from .retrieval import top
from .settings import ENCODER_MODEL, ENCODER_REVISION

VERSION = "coldfuse-v1"
FEATURE_VERSION = "cold-semantic-v1"
CONTRACT = dict(
    feature_version=FEATURE_VERSION,
    input_dim=1158,
    encoder_model=ENCODER_MODEL,
    encoder_revision=ENCODER_REVISION,
    architecture="Ranker(1158,128,64,1)",
    policy_version=VERSION,
    minimum_history=0,
)
BUDGETS = (0, 1, 3, 5, 10)


def unit(x):
    x = np.asarray(x, dtype="float32")
    return x / max(float(np.linalg.norm(x)), 1e-8)


@dataclass(frozen=True)
class Config:
    tau: float = 3
    generation: float = 0.75
    dnn: float = 0.25

    def __post_init__(self):
        if not (
            np.isfinite([self.tau, self.generation, self.dnn]).all()
            and self.tau > 0
            and self.generation >= 0
            and 0 <= self.dnn <= 1
        ):
            raise ValueError("Invalid ColdFuse configuration")


@dataclass
class State:
    history: list
    explicit: np.ndarray
    profile: np.ndarray
    reliability: float
    shrinkage: float


def state(vectors, history, explicit=None, tau=3, shrink=True):
    # Repeated views must not manufacture independent evidence.
    h = list(dict.fromkeys(reversed(history)))[::-1][-20:]
    if any(i < 0 or i >= len(vectors) for i in h):
        raise ValueError("History outside the catalog")
    prior = unit(np.zeros(vectors.shape[1]) if explicit is None else explicit)
    n = len(h)
    mean = (
        np.average(vectors[h], axis=0, weights=np.linspace(0.5, 1, n))
        if h
        else np.zeros(vectors.shape[1])
    )
    coherence = float(np.clip(np.linalg.norm(mean), 0, 1))
    reliability = n / (n + tau) * coherence if n else 0.0
    alpha = n / (n + tau) if n else 0.0
    if not shrink and n:
        alpha = 1.0
    profile = (
        unit(alpha * unit(mean) + (1 - alpha) * prior) if prior.any() else unit(mean)
    )
    return State(h, prior, profile, reliability, alpha)


def matrix(vectors, ids, s):
    x = vectors[ids]
    if s.history and len(ids):
        att = x @ vectors[s.history].T * 5
        att = np.exp(att - att.max(1, keepdims=True))
        att /= att.sum(1, keepdims=True)
        interest = s.shrinkage * (att @ vectors[s.history])
        interest += (1 - s.shrinkage) * s.explicit
        if not s.explicit.any():
            interest = att @ vectors[s.history]
        recent = x @ vectors[s.history[-1]]
    else:
        interest = np.broadcast_to(s.explicit, x.shape).copy()
        recent = np.zeros(len(ids))
    interest /= np.maximum(np.linalg.norm(interest, axis=1, keepdims=True), 1e-8)
    extras = np.stack(
        [
            x @ s.profile,
            recent,
            x @ s.explicit,
            np.full(len(ids), len(s.history) / 20),
            np.full(len(ids), s.reliability),
            np.full(len(ids), s.shrinkage),
        ],
        1,
    )
    return np.concatenate([x, interest, x * interest, extras], 1).astype("float32")


def beam_concentration(beams):
    if not beams:
        return 0.0
    p = np.array([score for _, score in beams], dtype=float)
    p = np.exp(p - p.max())
    p /= p.sum()
    # Concentration among the retained beams, NOT calibrated correctness.
    return float(np.sum(p * p))


def weights(s, cf_support, concentration, config=Config(), fixed=False):
    if fixed:
        return dict(popular=1.0, content=1.0, itemcf=1.0, generated=config.generation)
    evidence = max(s.reliability, 0.5 if s.explicit.any() else 0)
    return dict(
        popular=1 - 0.6 * evidence,
        content=2 * evidence,
        itemcf=s.reliability * cf_support,
        generated=config.generation * evidence * (0.25 + 0.75 * concentration),
    )


def fusion(lists, gate, budget=100):
    scores = defaultdict(float)
    scale = max(sum(gate.values()) / 61, 1e-8)
    for name, ids in lists.items():
        if gate.get(name, 0) <= 0:
            continue
        for rank, i in enumerate(dict.fromkeys(ids)):
            scores[int(i)] += gate[name] / (61 + rank) / scale
    ids = sorted(scores, key=lambda i: (-scores[i], i))[:budget]
    return ids, np.array([scores[i] for i in ids], dtype="float32")


def blend(features, baseline, logits=None, strength=0.25):
    from scipy.special import expit

    baseline = np.asarray(baseline, dtype="float32")
    if logits is None or not len(baseline):
        return baseline.copy()
    prior = np.linalg.norm(features[:, 384:768], axis=1) > 0
    # A prior is evidence, but an empty/no-prior user keeps the popularity policy.
    evidence = np.maximum(features[:, -2], prior.astype(float) * 0.25)
    amount = strength * evidence
    # Squashing is just a bounded ranking score, not a calibrated CTR.
    result = (1 - amount) * baseline + amount * expit(logits)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite ColdFuse score")
    return result


def policy_scores(model, features, baseline):
    import torch

    if model is None:
        return np.asarray(baseline).copy()
    with torch.inference_mode():
        logits = model(torch.from_numpy(np.asarray(features, dtype="float32"))).numpy()
    return blend(features, baseline, logits, strength=0.5)


class ColdFuse:
    def __init__(self, vectors, popularity, cf=None, codes=None, generator=None):
        self.x, self.pop, self.cf = vectors, popularity, cf
        self.codes, self.generator = codes, generator
        self.buckets, self.prefix = defaultdict(list), defaultdict(list)
        if codes is not None:
            for i, code in enumerate(codes):
                self.buckets[tuple(code)].append(i)
                self.prefix[tuple(code[:2])].append(i)

    def channels(self, s, eligible, cf_scores=None, beams=None):
        sim = self.x @ s.profile
        if cf_scores is None:
            cf_scores = (
                np.asarray(self.cf[s.history].sum(0)).ravel()
                if s.history and self.cf is not None
                else np.zeros(len(self.x))
            )
        lists = dict(
            popular=top(self.pop, eligible, 100),
            content=top(sim, eligible, 100) if s.profile.any() else [],
            itemcf=top(cf_scores, eligible & (cf_scores > 0), 100) if s.history else [],
        )
        anchors = []
        if beams is None:
            beams = []
            if self.generator is not None and (s.history or s.explicit.any()):
                # Semantic anchors are catalog content, never added to history.
                anchors = (
                    top(self.x @ s.explicit, eligible, 2) if s.explicit.any() else []
                )
                condition = list(dict.fromkeys(anchors + s.history))[-20:]
                valid = {
                    code for code, ids in self.buckets.items() if np.any(eligible[ids])
                }
                if valid:
                    beams = self.generator.generate(
                        condition,
                        {i: i for i in condition},
                        self.codes,
                        beams=20,
                        valid_codes=valid,
                    )
        exact, expanded = [], []
        within = sim if s.profile.any() else self.pop
        for code, _ in beams:
            pool = [i for i in self.buckets.get(tuple(code), []) if eligible[i]]
            exact.extend(sorted(pool, key=lambda i: (-self.pop[i], i))[:5])
            pool = [i for i in self.prefix.get(tuple(code[:2]), []) if eligible[i]]
            expanded.extend(sorted(pool, key=lambda i: (-within[i], i))[:5])
        lists["generated"] = list(dict.fromkeys(expanded))[:100]
        return dict(
            lists=lists,
            exact=list(dict.fromkeys(exact))[:100],
            concentration=beam_concentration(beams),
            anchors=anchors,
            cf_support=min(1.0, len(lists["itemcf"]) / 100),
            beams=beams,
        )

    def recommend(
        self,
        history,
        eligible,
        explicit=None,
        config=Config(),
        model=None,
        ablation=None,
        channels=None,
        cf_scores=None,
    ):
        s = state(
            self.x,
            history,
            None if ablation == "no_prior" else explicit,
            config.tau,
            shrink=ablation != "no_shrinkage",
        )
        ch = self.channels(s, eligible, cf_scores) if channels is None else channels
        gate = weights(
            s,
            ch["cf_support"],
            ch["concentration"],
            config,
            fixed=ablation == "fixed_gates",
        )
        lists = {
            k: v
            for k, v in ch["lists"].items()
            if ablation != "no_gen" or k != "generated"
        }
        ids, base = fusion(lists, gate)
        features = matrix(self.x, ids, s)
        logits = None
        if model is not None and ablation != "no_dnn" and ids:
            import torch

            with torch.inference_mode():
                logits = model(torch.from_numpy(features)).numpy()
        scores = blend(features, base, logits, config.dnn)
        order = np.lexsort((np.asarray(ids), -scores))
        return dict(
            ids=[ids[i] for i in order],
            scores=scores[order],
            baseline=base[order],
            features=features[order],
            state=s,
            gates=gate,
            channels=ch,
        )
