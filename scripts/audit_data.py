"""Check real processed data invariants; fail on temporal/schema violations."""

import json
from pathlib import Path
import os
import numpy as np
from signalfeed.data import load_rows

root = Path(os.environ.get("SIGNALFEED_HOME", "/data1/zhuzy/signalfeed"))
train = load_rows("train")
valid = load_rows("valid")
test = load_rows("test")
assert max(r["time"] for r in train) < min(r["time"] for r in valid)
assert max(r["time"] for r in valid) < min(r["time"] for r in test)
news = json.loads((root / "mind/news.json").read_text())
ids = {n["id"] for n in news}
missing = set()
for rows in [train, valid, test]:
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert len(r["items"]) == len(r["labels"])
        assert all(y in [0, 1] for y in r["labels"])
        assert len(r["history"]) <= 20
        missing.update(set(r["items"] + r["history"]) - ids)
assert not missing, missing
x = np.load(root / "mind/embeddings.npy")
codes = np.load(root / "mind/codes.npy")
assert x.shape == (len(news), 384)
assert np.isfinite(x).all()
assert codes.shape == (len(news), 3) and codes.min() >= 0 and codes.max() < 32
report = {
    "status": "passed",
    "train_impressions": len(train),
    "internal_valid_impressions": len(valid),
    "heldout_impressions": len(test),
    "items": len(news),
    "missing_metadata_ids": len(missing),
    "embedding_shape": list(x.shape),
    "time_order": "train < internal valid < heldout",
}
(root / "reports/data-audit.json").write_text(json.dumps(report, indent=2))
print(report)
