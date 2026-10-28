import json
import time
import numpy as np
import torch
from torch import nn
from .settings import path
from .features import Features
from .data import load_rows


class Ranker(nn.Module):
    def __init__(self, dim=1158):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def training_arrays(f, rows, seed, personal=True):
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    # Population and CF aggregates use only prior training rows. Full train
    # aggregates would leak labels when reused as features for those same rows.
    for r in rows:
        pos = [i for i, y in enumerate(r["labels"]) if y]
        neg = [i for i, y in enumerate(r["labels"]) if not y]
        if not pos:
            continue
        selected = pos + list(
            rng.choice(neg, min(len(neg), 4 * len(pos)), replace=False)
        )
        ids = [f.index[r["items"][i]] for i in selected]
        features = f.matrix(r, ids, stats=False, personal=personal)
        # Same-timestamp rows must not see one another's labels: no pop/CF
        # training features in v1. Serving uses the identical masked columns.
        xs.append(features)
        ys.extend(r["labels"][i] for i in selected)
    return np.concatenate(xs), np.asarray(ys, dtype="float32")


def train(seed=42, epochs=3, limit=0, personal=True):
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    f = Features().load()
    dest = path(
        f"models/ranker-{seed}" if personal else f"models/ranker-no-personal-{seed}"
    )
    start = time.time()
    x, y = training_arrays(f, load_rows("train", limit), seed, personal)
    vx, vy = training_arrays(f, load_rows("valid", 2000), seed, personal)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Ranker(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best = float("inf")
    logs = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(y))
        losses = []
        for begin in range(0, len(y), 2048):
            idx = order[begin : begin + 2048]
            xx = torch.from_numpy(x[idx]).to(device)
            yy = torch.from_numpy(y[idx]).to(device)
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(xx), yy)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        losses_v = []
        with torch.no_grad():
            for b in range(0, len(vy), 2048):
                pred = model(torch.from_numpy(vx[b : b + 2048]).to(device))
                losses_v.append(
                    nn.functional.binary_cross_entropy_with_logits(
                        pred, torch.from_numpy(vy[b : b + 2048]).to(device)
                    ).item()
                )
        row = dict(
            epoch=epoch + 1,
            train_loss=float(np.mean(losses)),
            valid_loss=float(np.mean(losses_v)),
        )
        logs.append(row)
        print(row, flush=True)
        if row["valid_loss"] < best:
            best = row["valid_loss"]
            torch.save(model.state_dict(), dest / "model.pt")
    meta = dict(
        seed=seed,
        personal=personal,
        examples=len(y),
        seconds=time.time() - start,
        history=logs,
        negative_sampling="up to 4 real unclicked exposures per positive",
        calibrated_ctr=False,
        masked_features=["population", "itemcf"],
    )
    (dest / "train.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_ranker(seed=42, personal=True):
    directory = (
        f"models/ranker-{seed}" if personal else f"models/ranker-no-personal-{seed}"
    )
    model = Ranker()
    model.load_state_dict(
        torch.load(path(directory) / "model.pt", map_location="cpu", weights_only=True)
    )
    return model.eval()
