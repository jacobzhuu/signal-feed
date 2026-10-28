"""Point-in-time shared features. Fit statistics never consume holdout labels."""

import json
from collections import defaultdict
import numpy as np
from scipy import sparse
from .settings import path
from .data import load_rows


class Features:
    def __init__(self):
        root = path("mind")
        self.news = json.loads((root / "news.json").read_text())
        self.index = {n["id"]: i for i, n in enumerate(self.news)}
        self.x = np.load(root / "embeddings.npy")
        self.codes = np.load(root / "codes.npy")
        cats = {c: i for i, c in enumerate(sorted({n["category"] for n in self.news}))}
        self.categories = np.array([cats[n["category"]] for n in self.news])
        self.first = np.array([n["first_seen"] for n in self.news])
        self.pop = np.zeros(len(self.news), dtype="float32")
        self.cf = None
        self.cutoff = json.loads((root / "manifest.json").read_text())["cutoff"]

    def fit(self, rows):
        self.pop.fill(0)
        users = defaultdict(set)
        for r in rows:
            for iid in r["history"]:
                if iid in self.index:
                    users[r["user"]].add(self.index[iid])
            for iid, y in zip(r["items"], r["labels"]):
                if y and iid in self.index:
                    self.pop[self.index[iid]] += 1
                    users[r["user"]].add(self.index[iid])
        rr = []
        cc = []
        for u, items in enumerate(users.values()):
            for i in items:
                rr.append(u)
                cc.append(i)
        mat = sparse.csr_matrix(
            (np.ones(len(rr), dtype="float32"), (rr, cc)),
            shape=(len(users), len(self.news)),
        )
        norms = np.sqrt(np.asarray(mat.sum(0)).ravel()).clip(1)
        # Bound intermediate memory by row blocks, not a full item-item product.
        self.cf = bounded_itemcf(mat, norms)
        np.save(path("mind") / "pop.npy", self.pop)
        sparse.save_npz(path("mind") / "cf.npz", self.cf)

    def load(self):
        self.pop = np.load(path("mind") / "pop.npy")
        self.cf = sparse.load_npz(path("mind") / "cf.npz")
        return self

    def history(self, r):
        return [self.index[i] for i in r["history"] if i in self.index]

    def vector(self, r):
        h = self.history(r)
        if not h:
            return np.zeros(self.x.shape[1], dtype="float32")
        weights = np.linspace(0.5, 1, len(h))
        v = np.average(self.x[h], axis=0, weights=weights)
        return (v / max(np.linalg.norm(v), 1e-8)).astype("float32")

    def scores(self, r):
        h = self.history(r)
        v = self.vector(r)
        sim = self.x @ v
        cf = np.asarray(self.cf[h].sum(0)).ravel() if h else np.zeros(len(self.news))
        return sim, cf

    def matrix(self, r, items, personal=True, stats=True):
        h = self.history(r) if personal else []
        v = self.vector(r) if personal else np.zeros(self.x.shape[1], dtype="float32")
        x = self.x[items]
        sim = x @ v
        recent = x @ self.x[h[-1]] if h else np.zeros(len(items))
        attention = x @ self.x[h].T if h else np.zeros((len(items), 1))
        att = attention * 5
        att = np.exp(att - att.max(1, keepdims=True))
        att /= att.sum(1, keepdims=True)
        interests = att @ self.x[h] if h else np.zeros_like(x)
        cat = np.array(
            [
                np.mean(self.categories[h] == self.categories[i]) if h else 0
                for i in items
            ]
        )
        pop = np.log1p(self.pop[items]) if stats else np.zeros(len(items))
        cf = (
            np.asarray(self.cf[h][:, items].sum(0)).ravel()
            if h and stats
            else np.zeros(len(items))
        )
        extra = np.stack(
            [
                sim,
                recent,
                cat,
                np.full(len(items), len(h) / 20),
                pop / 10,
                np.log1p(cf) / 5,
            ],
            1,
        )
        return np.concatenate([x, interests, x * interests, extra], 1).astype("float32")


def build_statistics():
    f = Features()
    f.fit(load_rows("train"))
    print({"items": len(f.news), "cf_edges": f.cf.nnz})


def bounded_itemcf(mat, norms=None, block_size=128, neighbors=100):
    norms = np.sqrt(np.asarray(mat.sum(0)).ravel()).clip(1) if norms is None else norms
    blocks = []
    for begin in range(0, mat.shape[1], block_size):
        co = (mat[:, begin : begin + block_size].T @ mat).tocoo()
        co.data /= norms[co.row + begin] * norms[co.col]
        co.data[co.col == co.row + begin] = 0
        block = co.tocsr()
        block.eliminate_zeros()
        for row in range(block.shape[0]):
            a, b = block.indptr[row : row + 2]
            if b - a > neighbors:
                order = np.lexsort((block.indices[a:b], -block.data[a:b]))
                block.data[a:b][order[neighbors:]] = 0
        block.eliminate_zeros()
        blocks.append(block)
    return sparse.vstack(blocks, format="csr") if blocks else sparse.csr_matrix((0, 0))
