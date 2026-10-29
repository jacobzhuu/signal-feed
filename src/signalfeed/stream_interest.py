"""Bounded, evidence-shrunk interests. No labels, model fitting or catalog cache."""

from dataclasses import dataclass
import numpy as np
from .coldfuse import canonical_history, unit

VERSION = "stream-interest-v3"


@dataclass(frozen=True)
class InterestConfig:
    alpha: float = 0.25
    tau: float = 3.0

    def __post_init__(self):
        if not np.isfinite([self.alpha, self.tau]).all() or not (
            0 <= self.alpha <= 1 and self.tau > 0
        ):
            raise ValueError("Invalid interest configuration")


GRID = tuple(InterestConfig(a, t) for a in (0.25, 0.5) for t in (1.0, 3.0))


class InterestState:
    """Append distinct items cheaply; rebuild only on repeats/window eviction.

    Cluster support excludes its strongest single observation. Thus an isolated
    theme is not declared reliable merely because its vector has unit norm.
    Catalog vectors are shared immutable inputs, never copied into user state.
    """

    def __init__(self, vectors, *, decay=0.85):
        if not 0 < decay <= 1:
            raise ValueError("Invalid decay")
        self.vectors = vectors
        self.decay = decay
        self.history = []
        self._clear()

    def _clear(self):
        d = self.vectors.shape[1]
        self.total = np.zeros(d, dtype="float64")
        self.position = np.zeros(d, dtype="float64")
        self.sums = np.zeros((3, d), dtype="float64")
        self.mass = np.zeros(3)
        self.largest = np.zeros(3)
        self.clusters = 0
        self.n = 0

    def _append(self, item):
        x = self.vectors[item]
        self.n += 1
        self.total += x
        self.position += self.n * x.astype("float64")
        if self.clusters:
            centers = self.sums[: self.clusters]
            centers = centers / np.maximum(
                np.linalg.norm(centers, axis=1, keepdims=True), 1e-8
            )
            similarity = centers @ x
            slot = int(similarity.argmax())
            if similarity[slot] < 0.65 and self.clusters < 3:
                slot = self.clusters
                self.clusters += 1
        else:
            slot = 0
            self.clusters = 1
        self.sums *= self.decay
        self.mass *= self.decay
        self.largest *= self.decay
        self.sums[slot] += x
        self.mass[slot] += 1
        self.largest[slot] = max(self.largest[slot], 1)

    def update(self, item):
        if not isinstance(item, (int, np.integer)) or not 0 <= item < len(self.vectors):
            raise ValueError("History outside catalog")
        rebuild = item in self.history or len(self.history) == 20
        self.history = canonical_history(self.history + [item])
        if rebuild:
            self._clear()
            for entry in self.history:
                self._append(entry)
        else:
            self._append(item)
        return rebuild

    @classmethod
    def from_history(cls, vectors, history, *, decay=0.85):
        result = cls(vectors, decay=decay)
        for item in canonical_history(history):
            result.update(item)
        return result

    @property
    def global_vector(self):
        # linspace(.5, 1, n) == (n-2 + position)/(2*(n-1)).
        if self.n <= 1:
            return unit(self.total)
        return unit((self.n - 2) * self.total + self.position)

    def project(self):
        centers = [unit(s) for s in self.sums[: self.clusters]]
        queries = np.column_stack([self.global_vector] + centers)
        return self.vectors @ queries  # Exactly one full-catalog multiplication.

    def scores(self, projected, config, ablation=None):
        base = projected[:, 0]
        if not self.n or ablation == "no_multi":
            return base.copy()
        support = np.maximum(
            0, self.mass[: self.clusters] - self.largest[: self.clusters]
        )
        reliability = support / (support + config.tau)
        if ablation == "no_reliability":
            reliability[:] = 1
        shrunk = base[:, None] + reliability * (projected[:, 1:] - base[:, None])
        return (1 - config.alpha) * base + config.alpha * shrunk.max(axis=1)
