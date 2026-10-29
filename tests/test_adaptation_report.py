"""Validate independent metric verification against hand-checkable examples."""

import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    file = Path(__file__).resolve().parents[1] / "scripts/report_adaptation.py"
    spec = importlib.util.spec_from_file_location("adaptation_report", file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


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
