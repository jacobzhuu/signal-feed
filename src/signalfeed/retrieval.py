"""One deterministic retrieval policy shared by evaluation and inference."""

import json
from collections import defaultdict
import numpy as np
from .settings import path

POLICY_VERSION = "retrieval-v2"


def top(scores, eligible, k):
    scores = np.asarray(scores)
    ids = np.flatnonzero(np.asarray(eligible) & np.isfinite(scores))
    if k <= 0 or not len(ids):
        return []
    # Resolve boundary ties explicitly; argpartition alone picks arbitrary IDs.
    n = min(k, len(ids))
    cutoff = np.partition(scores[ids], len(ids) - n)[len(ids) - n]
    above = ids[scores[ids] > cutoff]
    tied = ids[scores[ids] == cutoff][: n - len(above)]
    chosen = np.concatenate([above, tied])
    return chosen[np.lexsort((chosen, -scores[chosen]))].tolist()


def fuse(lists, budget=100, weights=None):
    scores = defaultdict(float)
    for name, ids in lists.items():
        weight = (weights or {}).get(name, 1)
        if weight <= 0:
            continue
        for rank, i in enumerate(dict.fromkeys(ids)):
            scores[i] += weight / (61 + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))[:budget]


def load_gate():
    file = path("models") / "fusion.json"
    gate = (
        json.loads(file.read_text())
        if file.exists()
        else {
            "threshold": 5,
            "weight": 1.5,
            "source": "untuned default",
            "policy_version": POLICY_VERSION,
        }
    )
    if gate.get("policy_version") != POLICY_VERSION:
        raise ValueError(
            "Fusion policy is stale; evaluate internal validation and tune again"
        )
    return gate


def fusion_weights(history_length, gate):
    long = history_length >= gate["threshold"]
    return {
        "popular": 0.5 if long else 1.5,
        "content": 1.0,
        "itemcf": 1.0,
        "generated": gate["weight"] if long else 0.5,
    }


def adaptive(lists, history_length, gate, *, generated=True):
    # The ablation changes exactly one channel, never the remaining weights.
    active = {k: v for k, v in lists.items() if generated or k != "generated"}
    return fuse(active, weights=fusion_weights(history_length, gate))


class RetrievalPolicy:
    def __init__(self, features, generator, gate=None):
        self.f = features
        self.generator = generator
        self.gate = load_gate() if gate is None else gate
        self.buckets, self.prefix_buckets = defaultdict(list), defaultdict(list)
        for i, code in enumerate(features.codes):
            self.buckets[tuple(code)].append(i)
            self.prefix_buckets[tuple(code[:2])].append(i)

    def candidates(self, request, eligible):
        f = self.f
        history = f.history(request)
        sim, cf = f.scores(request)
        pop = top(f.pop, eligible, 100)
        content = top(sim, eligible, 100) if history else []
        itemcf = top(cf, eligible & (cf > 0), 100) if history else []
        buckets = {
            code: [i for i in ids if eligible[i]] for code, ids in self.buckets.items()
        }
        buckets = {code: ids for code, ids in buckets.items() if ids}
        generated = self.generator.generate(
            request["history"], f.index, f.codes, beams=20, valid_codes=buckets.keys()
        )
        exact, prefix = [], []
        for code, _ in generated:
            exact.extend(
                sorted(buckets.get(code, []), key=lambda i: (-f.pop[i], i))[:5]
            )
            pool = [i for i in self.prefix_buckets[code[:2]] if eligible[i]]
            # With no content evidence, use the explicit train-popularity prior.
            within = sim if history else f.pop
            prefix.extend(sorted(pool, key=lambda i: (-within[i], i))[:5])
        exact, prefix = (
            list(dict.fromkeys(exact))[:100],
            list(dict.fromkeys(prefix))[:100],
        )
        lists = {
            "popular": pop,
            "content": content,
            "itemcf": itemcf,
            "generated": prefix,
        }
        return {
            "popularity": pop,
            "content": content,
            "itemcf": itemcf,
            "generative": exact,
            "generative_prefix": prefix,
            "traditional": fuse({k: v for k, v in lists.items() if k != "generated"}),
            "fixed_fusion": fuse(lists),
            "adaptive_fusion": adaptive(lists, len(history), self.gate),
            "adaptive_no_gen": adaptive(
                lists, len(history), self.gate, generated=False
            ),
        }
