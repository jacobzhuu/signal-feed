"""Semantic, incremental and admission regressions, never quality evidence."""

from dataclasses import asdict
import json
import numpy as np
import pytest
from signalfeed.coldfuse import state, canonical_history
from signalfeed.stream_interest import InterestState, InterestConfig
from signalfeed.interest_experiments import admission, records, write_json


def vectors():
    rng = np.random.default_rng(901)
    x = rng.normal(size=(80, 12)).astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_singletons_cannot_manufacture_evidence():
    x = np.eye(5, dtype="float32")
    s = InterestState.from_history(x, [0, 1, 2])
    p = s.project()
    assert s.clusters == 3
    np.testing.assert_array_equal(s.mass - s.largest, 0)
    np.testing.assert_allclose(s.scores(p, InterestConfig()), p[:, 0])
    assert not np.allclose(s.scores(p, InterestConfig(), "no_reliability"), p[:, 0])
    duplicate = InterestState.from_history(x, [0, 0, 0])
    assert duplicate.n == 1 and duplicate.mass.sum() == 1


def test_supported_secondary_theme_survives_with_bounded_state():
    x = np.array(
        [[1, 0], [0, 1], [1, 0.01], [0.01, 1], [0.707, 0.707]], dtype="float32"
    )
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    s = InterestState.from_history(x, [0, 1, 2, 3])
    assert s.clusters == 2 and np.all(s.mass[:2] - s.largest[:2] > 0)
    p = s.project()
    score = s.scores(p, InterestConfig(alpha=0.5, tau=1))
    assert score[0] > p[0, 0] and score[1] > p[1, 0]
    assert np.allclose(s.scores(p, InterestConfig(), "no_multi"), p[:, 0])


def test_incremental_matches_batch_every_prefix_repeats_and_eviction():
    x = vectors()
    history = list(range(28)) + [1, 27, 1, 29]
    stream = InterestState(x)
    for n, i in enumerate(history):
        stream.update(i)
        batch = InterestState.from_history(x, history[: n + 1])
        assert stream.history == batch.history == canonical_history(history[: n + 1])
        assert stream.clusters == batch.clusters <= 3
        for field in ("mass", "largest", "sums", "total", "position"):
            np.testing.assert_allclose(
                getattr(stream, field), getattr(batch, field), rtol=1e-12, atol=1e-12
            )
        np.testing.assert_allclose(
            stream.global_vector, state(x, history[: n + 1]).profile, atol=1e-7
        )
        np.testing.assert_allclose(
            stream.scores(stream.project(), InterestConfig()),
            batch.scores(batch.project(), InterestConfig()),
        )
    assert stream.update(2)  # Full history triggers bounded rebuild.


def test_invalid_history_and_config():
    s = InterestState(vectors())
    for i in (-1, 80, 1.5):
        with pytest.raises(ValueError):
            s.update(i)
    for kwargs in ({"alpha": float("nan")}, {"tau": 0}, {"alpha": 2}):
        with pytest.raises(ValueError):
            InterestConfig(**kwargs)
    assert asdict(InterestConfig()) == dict(alpha=0.25, tau=3)


def admission_rows():
    return [
        dict(
            user=str(i),
            band=band,
            period="early" if i < 200 else "late",
            metrics={
                "interest": dict(recall50=0.2, end_to_end_ndcg10=0.1),
                "content_safe": dict(recall50=0.1, end_to_end_ndcg10=0.1),
            },
        )
        for band in ("1-5", "10")
        for i in range(400)
    ]


def test_admission_requires_significance_time_stability_and_ndcg():
    rows = admission_rows()
    result = admission(rows)
    assert all(v["accepted"] for v in result.values())
    for r in rows:
        if r["band"] == "10":
            r["metrics"]["interest"]["end_to_end_ndcg10"] = 0.09
        elif r["period"] == "late":
            r["metrics"]["interest"]["recall50"] = 0.09
    result = admission(rows)
    assert not any(v["accepted"] for v in result.values())


def test_exclusive_outputs_and_changed_predictions_fail_closed(tmp_path):
    out = tmp_path / "test.json"
    write_json(out, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json(out, {"a": 2})
    assert json.loads(out.read_text()) == {"a": 1}
    dest = tmp_path / "reports/select"
    dest.mkdir(parents=True)
    (dest / "metrics.json").write_text(json.dumps(dict(algorithm={})))
    with pytest.raises(ValueError, match="changed"):
        records(tmp_path, "select")
