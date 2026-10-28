"""Residual semantic codes and stateless constrained decoding.

Design derived from MiniOneRec's residual SID + prefix constraint pipeline.
No model-name-dependent offsets or cross-request mutable decoding state.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from transformers import LogitsProcessor
import torch


class ResidualCodes:
    def __init__(self, levels=3, size=32, seed=42):
        self.levels = levels
        self.size = size
        self.seed = seed
        self.centers = []

    def fit(self, x):
        residual = x.copy()
        self.centers = []
        for level in range(self.levels):
            km = MiniBatchKMeans(
                n_clusters=self.size,
                random_state=self.seed + level,
                batch_size=4096,
                n_init=3,
            ).fit(residual)
            self.centers.append(km.cluster_centers_)
            residual -= km.cluster_centers_[km.predict(residual)]
        return self

    def encode(self, x):
        residual = x.copy()
        codes = []
        for centers in self.centers:
            dist = (
                (residual**2).sum(1)[:, None]
                + (centers**2).sum(1)[None, :]
                - 2 * residual @ centers.T
            )
            c = dist.argmin(1)
            codes.append(c)
            residual -= centers[c]
        return np.stack(codes, 1)

    def save(self, file):
        np.savez(file, centers=np.array(self.centers))

    @classmethod
    def load(cls, file):
        centers = np.load(file)["centers"]
        obj = cls(len(centers), len(centers[0]))
        obj.centers = list(centers)
        return obj


def code_text(code):
    return "".join(f"<s{level}_{int(value)}>" for level, value in enumerate(code))


class Trie:
    def __init__(self, sequences, eos):
        self.root = {}
        self.eos = eos
        for seq in sequences:
            node = self.root
            for token in list(seq) + [eos]:
                node = node.setdefault(int(token), {})

    def allowed(self, prefix):
        node = self.root
        for token in prefix:
            if int(token) == self.eos:
                return [self.eos]
            node = node.get(int(token))
            if node is None:
                return [self.eos]
        return sorted(node) or [self.eos]


class TrieProcessor(LogitsProcessor):
    def __init__(self, trie, prompt_length):
        self.trie = trie
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for i, seq in enumerate(input_ids):
            mask[i, self.trie.allowed(seq[self.prompt_length :].tolist())] = 0
        return scores + mask
