"""Small autoregressive SID recommender using MiniOneRec's sequence SFT objective.

A bounded-vocabulary local profile makes full-data experiments affordable; the
Qwen/upstream adapter in upstream_smoke.py verifies pretrained LLM compatibility.
"""

import json
import time
import numpy as np
import torch
from torch import nn
from .settings import path
from .data import load_rows
from .provenance import bind_model, verify_model
from .sid import ResidualCodes

SIZE = 32
LEVELS = 3
PAD = 96
BOS = 97
SEP = 98


class SIDTransformer(nn.Module):
    def __init__(self, dim=128, layers=3):
        super().__init__()
        self.embed = nn.Embedding(99, dim, padding_idx=PAD)
        self.pos = nn.Embedding(80, dim)
        layer = nn.TransformerEncoderLayer(
            dim, 4, dim * 4, 0.1, batch_first=True, norm_first=True
        )
        self.net = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(dim, 96)

    def forward(self, x):
        h = self.embed(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        mask = torch.triu(
            torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), 1
        )
        return self.head(self.net(h, mask=mask, src_key_padding_mask=x.eq(PAD)))


def token_history(history, index, codes):
    seq = [BOS]
    for iid in history[-20:]:
        if iid in index:
            seq.extend((codes[index[iid]] + np.arange(3) * SIZE).tolist())
    return seq + [SEP]


def prepare_codes():
    out = path("mind")
    news = json.loads((out / "news.json").read_text())
    x = np.load(out / "embeddings.npy")
    cutoff = json.loads((out / "manifest.json").read_text())["cutoff"]
    eligible = np.array([n["first_seen"] < cutoff for n in news])
    quant = ResidualCodes().fit(x[eligible])
    quant.save(out / "quantizer.npz")
    codes = quant.encode(x)
    np.save(out / "codes.npy", codes)
    unique = len(set(map(tuple, codes[eligible])))
    (out / "sid_manifest.json").write_text(
        json.dumps(
            dict(
                levels=3,
                size=32,
                fit_items=int(eligible.sum()),
                unique_codes=unique,
                collision_rate=1 - unique / int(eligible.sum()),
                new_item_policy="frozen codebook; bucket mapping preserves all items",
            ),
            indent=2,
        )
    )


def examples(rows, index, codes):
    result = []
    for r in rows:
        h = token_history(r["history"], index, codes)
        for iid, label in zip(r["items"], r["labels"]):
            if label and iid in index:
                target = (codes[index[iid]] + np.arange(3) * SIZE).tolist()
                result.append((h, target))
    return result


def batch_examples(examples, device):
    width = max(len(h) + 2 for h, t in examples)
    x = torch.full((len(examples), width), PAD, dtype=torch.long, device=device)
    y = torch.full_like(x, -100)
    for j, (h, t) in enumerate(examples):
        tokens = h + t
        x[j, : len(tokens) - 1] = torch.tensor(tokens[:-1], device=device)
        y[j, len(h) - 1 : len(h) + 2] = torch.tensor(t, device=device)
    return x, y


def train(seed=42, epochs=3, limit=0):
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = path("mind")
    dest = path(f"models/sid-{seed}")
    start = time.time()
    news = json.loads((out / "news.json").read_text())
    index = {n["id"]: i for i, n in enumerate(news)}
    codes = np.load(out / "codes.npy")
    train_data = examples(load_rows("train", limit), index, codes)
    valid = examples(load_rows("valid", 2000), index, codes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SIDTransformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best = float("inf")
    log = []
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_data)
        losses = []
        for begin in range(0, len(train_data), 256):
            x, y = batch_examples(train_data[begin : begin + 256], device)
            opt.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 96), y.reshape(-1), ignore_index=-100
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            losses.append(loss.item())
        model.eval()
        v = []
        with torch.no_grad():
            for begin in range(0, len(valid), 256):
                x, y = batch_examples(valid[begin : begin + 256], device)
                v.append(
                    nn.functional.cross_entropy(
                        model(x).reshape(-1, 96), y.reshape(-1), ignore_index=-100
                    ).item()
                )
        row = dict(
            epoch=epoch + 1,
            train_loss=float(np.mean(losses)),
            valid_loss=float(np.mean(v)),
            elapsed=time.time() - start,
        )
        log.append(row)
        print(row, flush=True)
        if row["valid_loss"] < best:
            best = row["valid_loss"]
            torch.save(model.state_dict(), dest / "model.pt")
    meta = dict(
        profile="compact SID Transformer; not Qwen",
        seed=seed,
        epochs=epochs,
        train_examples=len(train_data),
        parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.time() - start,
        history=log,
    )
    (dest / "train.json").write_text(json.dumps(meta, indent=2))
    bind_model(dest)
    return meta


class Generator:
    def __init__(self, seed=42, device="cpu"):
        verify_model(path(f"models/sid-{seed}"))
        self.device = device
        self.model = SIDTransformer().to(device)
        self.model.load_state_dict(
            torch.load(
                path(f"models/sid-{seed}") / "model.pt",
                map_location=device,
                weights_only=True,
            )
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, history, index, codes, beams=20, valid_codes=None):
        h = token_history(history, index, codes)
        active = [([], 0.0)]
        prefixes = None
        if valid_codes is not None:
            prefixes = {}
            for code in valid_codes:
                tok = tuple((np.array(code) + np.arange(3) * SIZE).tolist())
                for j in range(3):
                    prefixes.setdefault(tok[:j], set()).add(tok[j])
        for level in range(3):
            x = torch.tensor([h + p for p, s in active], device=self.device)
            scores = self.model(x)[:, -1].log_softmax(-1).cpu().numpy()
            expanded = []
            for row, (prefix, score) in zip(scores, active):
                allowed = (
                    range(level * SIZE, (level + 1) * SIZE)
                    if prefixes is None
                    else sorted(prefixes.get(tuple(prefix), []))
                )
                expanded.extend(
                    (prefix + [int(t)], score + float(row[t])) for t in allowed
                )
            active = sorted(expanded, key=lambda z: z[1], reverse=True)[:beams]
            if not active:
                return []
        return [
            (tuple(int(t - j * SIZE) for j, t in enumerate(p)), s) for p, s in active
        ]
