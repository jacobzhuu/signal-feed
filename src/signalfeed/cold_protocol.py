"""Preregistered cold simulation. No current-impression labels enter inputs."""

from collections import Counter
import numpy as np
from .coldfuse import unit


def prototypes(f):
    return {
        int(c): unit(f.x[(f.categories == c) & (f.first < f.cutoff)].mean(0))
        for c in np.unique(f.categories[f.first < f.cutoff])
    }


def simulate(f, row, k, scenario, category_vectors):
    history = f.history(row)
    observed_pool = history[-10:]
    older = [i for i in history[:-10] if i not in set(observed_pool)]
    observed = observed_pool[-k:] if k else []
    categories = Counter(int(f.categories[i]) for i in older)
    chosen = sorted(categories, key=lambda c: (-categories[c], c))[:2]
    prior = (
        unit(
            sum(
                (category_vectors[c] for c in chosen if c in category_vectors),
                np.zeros(f.x.shape[1]),
            )
        )
        if scenario == "proxy_explicit"
        else np.zeros(f.x.shape[1], dtype="float32")
    )
    return observed, prior, older


def sample(rows, f, count, natural=False):
    rows = [
        r
        for r in rows
        if (len(f.history(r)) <= 5 if natural else len(f.history(r)) >= 15)
    ]
    if len(rows) > count:
        ids = np.random.default_rng(42).choice(len(rows), count, replace=False)
        rows = [rows[i] for i in sorted(ids)]
    return rows
