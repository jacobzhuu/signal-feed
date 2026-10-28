"""Separate semantic DNN; history masking never changes the target labels."""

import json
import time
import numpy as np
import torch
from .coldfuse import CONTRACT, BUDGETS, state, matrix
from .cold_protocol import prototypes, simulate
from .features import Features
from .data import load_rows
from .ranker import Ranker
from .settings import path
from .provenance import bind_model, verify_model, code_identity


def arrays(f, rows, seed, category_vectors):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for r in rows:
        pos = [i for i, y in enumerate(r["labels"]) if y]
        neg = [i for i, y in enumerate(r["labels"]) if not y]
        if not pos:
            continue
        chosen = pos + list(rng.choice(neg, min(len(neg), 4 * len(pos)), replace=False))
        k = int(rng.choice(BUDGETS))
        scenario = "proxy_explicit" if rng.random() < 0.5 else "no_explicit"
        h, prior, _ = simulate(f, r, k, scenario, category_vectors)
        xs.append(
            matrix(f.x, [f.index[r["items"][i]] for i in chosen], state(f.x, h, prior))
        )
        ys.extend(r["labels"][i] for i in chosen)
    return np.concatenate(xs), np.asarray(ys, dtype="float32")


def train(seed=42, limit=30000, epochs=3):
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1, user_api="blas")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    start, source = time.time(), code_identity()
    f = Features().load()
    pv = prototypes(f)
    rows = load_rows("train", limit)
    x, y = arrays(f, rows, seed, pv)
    vx, vy = arrays(f, load_rows("valid", 2000), 42, pv)
    model = Ranker()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    dest = path(f"models/coldfuse-{seed}")
    if (dest / "model.pt").exists():
        raise FileExistsError("Use a fresh run; never overwrite experiment checkpoints")
    best, logs = float("inf"), []
    for epoch in range(epochs):
        model.train()
        order, losses = rng.permutation(len(y)), []
        for begin in range(0, len(y), 1024):
            ids = order[begin : begin + 1024]
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(torch.from_numpy(x[ids])), torch.from_numpy(y[ids])
            )
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        total = 0.0
        with torch.inference_mode():
            for b in range(0, len(vy), 1024):
                total += float(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        model(torch.from_numpy(vx[b : b + 1024])),
                        torch.from_numpy(vy[b : b + 1024]),
                        reduction="sum",
                    )
                )
        row = dict(
            epoch=epoch + 1,
            train_loss=float(np.mean(losses)),
            valid_loss=total / len(vy),
        )
        logs.append(row)
        print(row, flush=True)
        if row["valid_loss"] < best:
            best = row["valid_loss"]
            torch.save(model.state_dict(), dest / "model.pt")
    if source["source_sha256"] != code_identity()["source_sha256"]:
        raise RuntimeError("Source changed during training")
    report = dict(
        seed=seed,
        **CONTRACT,
        impressions=len(rows),
        examples=len(y),
        seconds=time.time() - start,
        epochs=logs,
        provenance=source,
        device="cpu",
        parameters=sum(p.numel() for p in model.parameters()),
        augmentation="one seeded uniform budget and scenario per training impression; train-only category prototypes; no CF/pop features",
        calibrated_ctr=False,
    )
    (dest / "train.json").write_text(json.dumps(report, indent=2))
    bind_model(dest)
    return report


def load(seed=42):
    directory = path(f"models/coldfuse-{seed}")
    verify_model(directory)
    report = json.loads((directory / "train.json").read_text())
    if any(report.get(k) != v for k, v in CONTRACT.items()):
        raise ValueError("ColdFuse feature/model contract mismatch")
    model = Ranker()
    model.load_state_dict(
        torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    )
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Non-finite ColdFuse model")
    return model.eval()
