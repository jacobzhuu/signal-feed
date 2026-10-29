"""Research evidence cannot be excused as routine website status."""

import importlib.util
import json
from pathlib import Path
import pytest
from signalfeed.provenance import sha256


def reporter():
    spec = importlib.util.spec_from_file_location(
        "interest_report",
        Path(__file__).resolve().parents[1] / "scripts/report_interest.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unchanged_research_and_disclosed_operational_refresh(tmp_path):
    research = tmp_path / "predictions.jsonl"
    research.write_text("immutable")
    operational = tmp_path / "hn_ingest.json"
    operational.write_text(
        json.dumps(
            dict(
                source="https://hacker-news.firebaseio.com/v0",
                count=99,
                errors=0,
                seconds=1,
            )
        )
    )
    files = {
        str(research): sha256(research),
        str(operational): "previous-operational-hash",
    }
    result = reporter().audit_previous(files)
    assert result["verified_research_files"] == 1
    record = result["operational_status"][str(operational)]
    assert record["changed"] and record["frozen_sha256"] == "previous-operational-hash"
    assert record["current_sha256"] == sha256(operational)
    research.write_text("modified")
    with pytest.raises(ValueError, match="Old evidence changed"):
        reporter().audit_previous(files)


def test_arbitrary_json_is_not_an_operational_exception(tmp_path):
    file = tmp_path / "hn_ingest.json"
    file.write_text(json.dumps(dict(predictions=[1, 2, 3])))
    with pytest.raises(ValueError, match="schema"):
        reporter().audit_previous({str(file): "old"})
