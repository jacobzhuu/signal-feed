"""Exercise adapted upstream SidSFTDataset and a real Qwen checkpoint on Amazon."""

import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    LogitsProcessorList,
)
from peft import LoraConfig, get_peft_model
from .settings import path, QWEN_MODEL, QWEN_REVISION
from .sid import Trie, TrieProcessor
from .vendor.minionerec.data import SidSFTDataset


def run(steps=32):
    torch.set_num_threads(8)
    torch.manual_seed(42)
    start = time.time()
    upstream = (
        Path(
            os.environ.get(
                "SIGNALFEED_UPSTREAM",
                str(Path(__file__).resolve().parents[3] / "MiniOneRec"),
            )
        )
        / "data"
        / "Amazon"
    )
    out = path("upstream-smoke")
    category = "Industrial_and_Scientific"
    suffix = f"{category}_5_2016-10-2018-11.csv"
    for split, count in [("train", 256), ("valid", 32), ("test", 32)]:
        frame = pd.read_csv(upstream / split / suffix).sample(n=count, random_state=42)
        frame.to_csv(out / f"{split}.csv", index=False)
    index = json.loads((upstream / "index" / f"{category}.index.json").read_text())
    tokens = sorted({token for sid in index.values() for token in sid})
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, revision=QWEN_REVISION)
    tokenizer.add_tokens(tokens)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, revision=QWEN_REVISION, torch_dtype=torch.bfloat16
    )
    model.resize_token_embeddings(len(tokenizer))
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    data = SidSFTDataset(str(out / "train.csv"), tokenizer=tokenizer, max_len=512)
    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, return_tensors="pt", pad_to_multiple_of=8
    )
    loader = torch.utils.data.DataLoader(
        data, batch_size=2, shuffle=True, collate_fn=collator
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-4
    )
    model.train()
    losses = []
    for step, batch in enumerate(loader):
        if step >= steps:
            break
        batch = {k: v[:, -512:].cuda() for k, v in batch.items()}
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print({"step": step + 1, "loss": loss.item()}, flush=True)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    # Reload the saved adapter to verify a real persisted inference path.
    from peft import PeftModel

    del optimizer, model
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, revision=QWEN_REVISION, torch_dtype=torch.bfloat16
    )
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, out / "adapter").cuda().eval()
    evaldata = SidSFTDataset(
        str(out / "test.csv"), tokenizer=tokenizer, max_len=512, test=True
    )
    sid_strings = {"".join(sid) for sid in index.values()}
    seqs = [tokenizer.encode(s, add_special_tokens=False) for s in sorted(sid_strings)]
    trie = Trie(seqs, tokenizer.eos_token_id)
    results = []
    model.config.use_cache = True
    with torch.inference_mode():
        for i in range(len(evaldata)):
            x = torch.tensor([evaldata[i]["input_ids"]], device="cuda")
            y = model.generate(
                x,
                attention_mask=torch.ones_like(x),
                num_beams=10,
                num_return_sequences=10,
                max_new_tokens=8,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([TrieProcessor(trie, x.shape[1])]),
            )
            predictions = [
                tokenizer.decode(s[x.shape[1] :], skip_special_tokens=True).strip()
                for s in y
            ]
            target = str(evaldata.data.iloc[i]["item_sid"])
            results.append(dict(target=target, predictions=predictions))
    hr = np.mean([r["target"] in r["predictions"] for r in results])
    ndcg = np.mean(
        [
            1 / np.log2(r["predictions"].index(r["target"]) + 2)
            if r["target"] in r["predictions"]
            else 0
            for r in results
        ]
    )
    invalid = sum(p not in sid_strings for r in results for p in r["predictions"])
    report = dict(
        kind="real Amazon smoke; not paper reproduction",
        model="Qwen/Qwen2.5-0.5B + LoRA + trainable embeddings",
        train_examples=256,
        train_steps=len(losses),
        losses=losses,
        test_examples=len(results),
        hr10=float(hr),
        ndcg10=float(ndcg),
        invalid=invalid,
        seconds=time.time() - start,
    )
    (out / "predictions.json").write_text(json.dumps(results))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(report)
