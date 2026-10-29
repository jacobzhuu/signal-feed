"""Validate independent metric verification against hand-checkable examples."""

import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module(name="report_adaptation"):
    file = Path(__file__).resolve().parents[1] / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location("adaptation_report", file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_cross_release_ids_are_not_article_identity():
    prep = module("prepare_adaptation")
    old = [
        dict(
            id="N1",
            category="a",
            subcategory="b",
            title="old",
            abstract="text",
            url="url",
        )
    ]
    large = [dict(old[0], id="N99"), dict(old[0], title="different")]
    mapping = prep.news_mapping(old, large)
    assert mapping == {"N99": "N1", "N1": "large:N1"}
    with pytest.raises(ValueError, match="reconcile"):
        prep.news_mapping(old, [large[1]])


def test_exclusion_uses_canonical_events_not_user_ids_or_labels(tmp_path):
    import json
    from datetime import datetime, timezone

    prep = module("prepare_adaptation")
    base = tmp_path / "baseline"
    (base / "mind").mkdir(parents=True)
    stamp = "11/15/2019 8:00:00 AM"
    time = (
        datetime.strptime(stamp, "%m/%d/%Y %I:%M:%S %p")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    r = dict(user="U1", time=time, history=["N1"], items=["N1", "N2"], labels=[1, 0])
    for split in ("train", "valid", "test"):
        (base / f"mind/{split}.jsonl").write_text(json.dumps(r) + "\n")
    for split in ("train", "dev"):
        dest = tmp_path / f"raw/{split}"
        dest.mkdir(parents=True)
        # Different user ID, candidate order and labels, same exposed event.
        (dest / "behaviors.tsv").write_text(f"999\tU900\t{stamp}\tL1\tL2-1 L1-0\n")
    excluded, audit = prep.canonical_exclusions(
        tmp_path, base, {"L1": "N1", "L2": "N2"}
    )
    assert excluded == {"U900"}
    assert audit["exposed_event_signatures"] == audit["matched_event_signatures"] == 1


def test_independent_metric_ties_and_multiple_positives():
    report = module()
    m = report.independent_metrics([2, 1], [1, 3], [0, 1, 1, 0], [0, 0, 0, 0])
    assert m["auc"] == 0.5
    assert m["mrr"] == pytest.approx((1 / 2 + 1 / 3) / 2)
    assert m["recall50"] == 0.5
    assert m["end_to_end_ndcg5"] == pytest.approx(
        (1 / np.log2(3)) / (1 + 1 / np.log2(3))
    )
    assert report.independent_metrics([], [1], [1, 1], [0, 1])["auc"] is None
    with pytest.raises(ValueError):
        report.independent_metrics([1, 1], [1], [1, 0], [1, 0])
    with pytest.raises(ValueError):
        report.independent_metrics([1], [1], [1, 0], [float("nan"), 0])


def test_cf_exposure_audit_matches_unique_history_and_extended_catalog():
    from scipy.sparse import csr_matrix

    report = module()
    cf = csr_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    scores = report.unique_cf_scores(cf, [0, 0, 1, 5], [1, 2, 4])
    assert scores.tolist() == [1.0, 5.0, 0.0]
    assert report.unique_cf_scores(cf, [5], [1, 4]).tolist() == [0.0, 0.0]
