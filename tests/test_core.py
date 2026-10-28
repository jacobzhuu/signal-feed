import numpy as np
import pytest
import torch
from signalfeed.sid import Trie, TrieProcessor, ResidualCodes
from signalfeed.evaluation import ranking_metrics, recall_metrics


def test_constrained_decode_is_stateless_and_batch_safe():
    trie = Trie([[1, 3], [1, 4], [2, 5]], eos=6)
    processor = TrieProcessor(trie, 2)
    scores = processor(torch.tensor([[9, 9, 1], [8, 8, 2]]), torch.zeros(2, 7))
    assert torch.isfinite(scores[0]).nonzero().flatten().tolist() == [3, 4]
    assert torch.isfinite(scores[1]).nonzero().flatten().tolist() == [5]
    again = processor(torch.tensor([[9, 9, 1], [8, 8, 2]]), torch.zeros(2, 7))
    assert torch.equal(scores, again)
    assert trie.allowed([1, 3]) == [6]


def test_metrics_use_all_positives_and_mind_mrr():
    got = ranking_metrics([1, 0, 1], [3, 2, 1])
    assert got["auc"] == 0.5
    assert got["mrr"] == pytest.approx((1 + 1 / 3) / 2)
    assert got["ndcg5"] == pytest.approx((1 + 1 / 2) / (1 + 1 / np.log2(3)))
    assert recall_metrics([3, 7], [3, 9])["recall20"] == 0.5
    assert ranking_metrics([0, 0], [1, 2])["auc"] is None


def test_frozen_quantizer_preserves_old_codes(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 8)).astype("float32")
    q = ResidualCodes(size=4).fit(x)
    before = q.encode(x)
    q.save(tmp_path / "q.npz")
    loaded = ResidualCodes.load(tmp_path / "q.npz")
    loaded.encode(rng.normal(size=(10, 8)).astype("float32"))
    np.testing.assert_array_equal(before, loaded.encode(x))
