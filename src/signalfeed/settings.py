import os
from pathlib import Path

ROOT = Path(os.environ.get("SIGNALFEED_HOME", "/data1/zhuzy/signalfeed"))
ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(ROOT / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")


def path(name: str) -> Path:
    p = ROOT / name
    p.mkdir(parents=True, exist_ok=True)
    return p


ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B"
QWEN_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
