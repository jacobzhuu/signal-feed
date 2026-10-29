"""Synthetic contract regressions; these are never quality evidence."""

from dataclasses import asdict
import json
import numpy as np
import pytest
from signalfeed.adaptation import (
    AdaptConfig,
    GuardedAdapter,
    VERSION,
    signals,
    interest_scores,
    safe_content,
    fuse_interest,
    cohort_key,
)
from signalfeed.adaptation_experiments import paired_test, holm
from signalfeed.retrieval import top


def test_heterogeneous_history_preserves_secondary_interest():
    x = np.array([[1, 0], [0, 1], [0.707, 0.707], [0.99, 0.1]], dtype="float32")
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    sig = signals(x, [0, 1])
    centroid = interest_scores(sig, AdaptConfig(multi=0))
    multiple = interest_scores(sig, AdaptConfig(multi=1))
    assert centroid[2] > centroid[0]
    assert multiple[0] > multiple[2] and multiple[1] > multiple[2]
    assert np.allclose(
        multiple, interest_scores(signals(x, [0, 0, 1]), AdaptConfig(multi=1))
    )


def test_conflicting_prior_is_shrunk_without_erasing_click():
    x = np.eye(3, dtype="float32")
    sig = signals(x, [0], x[1])
    score = interest_scores(sig, AdaptConfig(multi=0))
    unguarded = interest_scores(sig, AdaptConfig(multi=0), "no_reliability")
    assert score[0] > score[1]
    assert unguarded[1] > unguarded[0]
    assert np.array_equal(interest_scores(signals(x, [], x[1]), AdaptConfig()), x[1])


def test_no_side_agreement_has_zero_weight_and_ablation_is_controlled():
    x = np.eye(4, dtype="float32")
    sig = signals(x, [0])
    pred, _, gates = fuse_interest(sig, [1, 2], {"generated": [3]}, AdaptConfig())
    assert pred == [1, 2] and gates["generated"] == 0
    pred, _, gates = fuse_interest(
        sig, [1, 2], {"generated": [3]}, AdaptConfig(), "fixed_side"
    )
    assert 3 in pred and gates["generated"] == 0.25
    no_gen, _, _ = fuse_interest(
        sig, [1, 2], {"generated": [3]}, AdaptConfig(), "no_gen"
    )
    assert no_gen == [1, 2]


def test_fallback_precedes_expensive_generation_and_matches_content():
    x = np.eye(4, dtype="float32")
    prior = np.zeros(4, dtype="float32")
    eligible = np.array([False, True, True, True])
    guard = GuardedAdapter(
        dict(version=VERSION, domain="test", config=asdict(AdaptConfig()), groups={}),
        domain="test",
    )

    def forbidden():
        raise AssertionError("Rejected groups must not call generation")

    result = guard.recommend(x, np.arange(4), [0], prior, eligible, forbidden)
    assert result["ids"] == safe_content(x, [0], prior, eligible, np.arange(4))
    assert result["status"] == "content-fallback" and not result["channels_called"]
    assert 0 not in result["ids"]
    empty = guard.recommend(x, np.arange(4), [], prior, eligible, forbidden)
    assert empty["ids"] == [3, 2, 1]


def test_guard_domain_rejection_and_invalid_config():
    with pytest.raises(ValueError, match="domain"):
        GuardedAdapter(dict(version=VERSION, domain="mind"), domain="hn")
    for value in (float("nan"), -1, 2):
        with pytest.raises(ValueError):
            AdaptConfig(side=value)


def test_accepted_shared_policy_filters_unavailable_candidates():
    x = np.eye(4, dtype="float32")
    prior = np.zeros(4, dtype="float32")
    mask = np.array([False, True, True, False])
    cfg = AdaptConfig(side=0.5)
    sig = signals(x, [0], prior)
    content = top(interest_scores(sig, cfg), mask, 100)
    expected, _, _ = fuse_interest(sig, content, {"generated": [1, 2]}, cfg)
    guard = GuardedAdapter(
        dict(
            version=VERSION,
            domain="test",
            config=asdict(cfg),
            groups={cohort_key([0], prior): dict(accepted=True)},
        ),
        domain="test",
    )
    result = guard.recommend(
        x, np.arange(4), [0], prior, mask, lambda: {"generated": [0, 1, 2, 3, -1, 99]}
    )
    assert result["ids"] == expected
    assert result["channels_called"]
    assert (
        guard.recommend(x, np.arange(4), [0], prior, np.zeros(4, dtype=bool))["ids"]
        == []
    )


def test_zero_differences_and_multiple_comparison_correction():
    zero = paired_test([0] * 20, repeats=200)
    assert zero["p"] == 1 and zero["ci"] == [0, 0]
    positive = paired_test([1] * 20, repeats=1000)
    assert positive["ci"][0] == 1 and positive["p"] < 0.01
    cases = dict(a=dict(p=0.01), b=dict(p=0.02), c=dict(p=0.9))
    holm(cases)
    assert cases["a"]["p_holm"] == pytest.approx(0.03)
    assert cases["b"]["p_holm"] == pytest.approx(0.04)


def test_guard_counts_users_not_seeds_or_budget_instances(tmp_path, monkeypatch):
    import signalfeed.adaptation_experiments as exp

    records = []
    for seed in (42, 43, 44):
        for u in range(40):
            for k in (3, 5):
                records.append(
                    (
                        seed,
                        dict(
                            user=str(u),
                            k=k,
                            guard_group="no_prior/2-5",
                            metrics=dict(
                                adaptation=dict(recall50=1),
                                content_safe=dict(recall50=0),
                            ),
                        ),
                    )
                )
    monkeypatch.setattr(exp, "read_records", lambda *args: iter(records))
    monkeypatch.setattr(exp, "sha256", lambda file: "hash")
    (tmp_path / "selection.json").write_text(
        json.dumps(dict(config=asdict(AdaptConfig())))
    )
    exp.guard(tmp_path)
    decision = json.loads((tmp_path / "guard.json").read_text())
    assert decision["groups"]["no_prior/2-5"]["n"] == 40
    assert not decision["groups"]["no_prior/2-5"]["accepted"]
