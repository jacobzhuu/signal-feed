"""Content-addressed dependencies for offline models and measured runs."""

import hashlib
import json
import subprocess
from pathlib import Path
from .settings import path, ENCODER_REVISION

FEATURE_VERSION = "mind-text-v1"
DEPENDENCIES = (
    "news.json",
    "embeddings.npy",
    "codes.npy",
    "quantizer.npz",
    "pop.npy",
    "cf.npz",
    "manifest.json",
    "train.jsonl",
    "valid.jsonl",
)


def sha256(file):
    h = hashlib.sha256()
    with open(file, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def data_identity():
    return {
        "feature_version": FEATURE_VERSION,
        "encoder_revision": ENCODER_REVISION,
        "files": {name: sha256(path("mind") / name) for name in DEPENDENCIES},
    }


def bind_model(directory, *, origin="training"):
    manifest = {
        **data_identity(),
        "model_sha256": sha256(directory / "model.pt"),
        "origin": origin,
    }
    (directory / "bundle.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_model(directory):
    file = directory / "bundle.json"
    if not file.exists():
        raise ValueError(
            f"Unbound checkpoint: {directory}; retrain or explicitly attest audited legacy artifacts"
        )
    manifest = json.loads(file.read_text())
    identity = data_identity()
    if any(manifest.get(k) != v for k, v in identity.items()):
        raise ValueError(f"Model/data/encoder mismatch: {directory}")
    if manifest.get("model_sha256") != sha256(directory / "model.pt"):
        raise ValueError(f"Checkpoint checksum mismatch: {directory}")
    return manifest


def code_identity():
    root = Path(__file__).resolve().parents[2]
    files = []
    for directory in [
        "src",
        "scripts",
        "tests",
        "frontend/src",
        "frontend/e2e",
        "configs",
    ]:
        files.extend(
            file
            for file in (root / directory).rglob("*")
            if file.is_file()
            and file.suffix in {".py", ".sh", ".ts", ".tsx", ".css", ".json"}
        )
    files += [
        root / "pyproject.toml",
        root / "frontend/package.json",
        root / "frontend/package-lock.json",
    ]
    digest = hashlib.sha256()
    for file in sorted(files):
        digest.update(str(file.relative_to(root)).encode())
        digest.update(file.read_bytes())
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    return {
        "git_commit": commit,
        "worktree_dirty": dirty,
        "source_sha256": digest.hexdigest(),
    }


def run_identity(seed, split):
    return {
        **code_identity(),
        "data": data_identity(),
        "seed": seed,
        "split": split,
        "test_sha256": sha256(path("mind") / f"{split}.jsonl"),
        "models": {
            kind: sha256(path(f"models/{kind}-{seed}") / "model.pt")
            for kind in ("sid", "ranker")
        },
    }
