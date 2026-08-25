"""LoRA fine-tuning of a protein language model, one configuration per run.

Parameter-efficient fine-tuning over a grid of rank, alpha, learning rate, target
modules and class weighting. The classification head is trained jointly with the
adapters (modules_to_save), validation is carved from the training split only,
and early stopping is on validation macro-F1.

Two practical warnings learned from running this:

1. Training budget can dominate the hyperparameter grid. If most configurations
   peak at or near the epoch cap, the grid is measuring how fast each one learns,
   not which is better. Raise the cap and the patience until the peak epochs sit
   comfortably inside it, then report those numbers.

2. Short patience on a small validation fold manufactures variance -- a run can
   stop early on noise and look far worse than it is.

Rows are written incrementally, and each run should be given its own --out path;
parallel runs sharing one path will overwrite each other.

    python src/plm_lora.py --config-id 1 --corpus dedup --split 0 \
        --max-epochs 80 --patience 10 --out results/plm/lora_s0.csv
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C
import corpora as D

REPO = "facebook/esm2_t33_650M_UR50D"
MAXLEN = 1022

# A deliberately compact grid: 12 configs spanning capacity, learning rate,
# adapter placement and class weighting, rather than a 72-cell product we could
# not afford to run honestly.
CONFIGS = [
    dict(rank=8,  alpha=16, lr=1e-4, targets="qv",   weighted=False),
    dict(rank=8,  alpha=16, lr=3e-4, targets="qv",   weighted=False),
    dict(rank=16, alpha=32, lr=1e-4, targets="qv",   weighted=False),
    dict(rank=16, alpha=32, lr=3e-4, targets="qv",   weighted=False),
    dict(rank=16, alpha=32, lr=1e-3, targets="qv",   weighted=False),
    dict(rank=32, alpha=32, lr=3e-4, targets="qv",   weighted=False),
    dict(rank=16, alpha=32, lr=3e-4, targets="qkvd", weighted=False),
    dict(rank=32, alpha=32, lr=3e-4, targets="qkvd", weighted=False),
    dict(rank=16, alpha=32, lr=3e-4, targets="qv",   weighted=True),
    dict(rank=16, alpha=32, lr=1e-3, targets="qv",   weighted=True),
    dict(rank=32, alpha=32, lr=3e-4, targets="qkvd", weighted=True),
    dict(rank=8,  alpha=16, lr=3e-4, targets="qv",   weighted=True),
]
TARGETS = {"qv": ["query", "value"],
           "qkvd": ["query", "key", "value", "dense"]}


class SeqDS(Dataset):
    def __init__(self, seqs, labels, tok):
        self.s, self.y, self.tok = seqs, labels, tok

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i], int(self.y[i])


def collate(batch, tok):
    seqs, ys = zip(*batch)
    enc = tok(list(seqs), return_tensors="pt", padding=True, truncation=True,
              max_length=MAXLEN + 2)
    enc["labels"] = torch.tensor(ys)
    return enc


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    preds, trues, probs = [], [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.float()
        probs.append(torch.softmax(logits, -1).cpu().numpy())
        preds.append(logits.argmax(-1).cpu().numpy())
        trues.append(labels.numpy())
    y_pred = np.concatenate(preds); y_true = np.concatenate(trues)
    return C.compute_metrics(y_true, y_pred, np.concatenate(probs), n_classes,
                             all_auc_variants=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", type=int, required=True)
    ap.add_argument("--corpus", default="dedup")
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.06)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/plm_lora_lora.csv")
    args = ap.parse_args()

    cfg = CONFIGS[args.config_id]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_cosine_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model

    S, y, names, info = D.get_corpus(args.corpus)
    n_classes = len(names)
    splits = C.stratified_splits(y, n_splits=args.n_splits, test_size=0.30,
                                 seed=args.seed, dataset=args.corpus).splits
    tr_all, te = splits[args.split]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(tr_all)
    n_val = int(len(perm) * args.val_frac)
    va, tr = perm[:n_val], perm[n_val:]
    print(f"[cfg {args.config_id}] {json.dumps(cfg)}", flush=True)
    print(f"[data] corpus={args.corpus} n={len(S)} classes={n_classes} "
          f"train={len(tr)} val={len(va)} test={len(te)} split={args.split}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(REPO)
    base = AutoModelForSequenceClassification.from_pretrained(
        REPO, num_labels=n_classes, torch_dtype=torch.float32)
    lcfg = LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=0.1,
                      bias="none", task_type="SEQ_CLS",
                      target_modules=TARGETS[cfg["targets"]],
                      modules_to_save=["classifier"])
    model = get_peft_model(base, lcfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable={trainable:,} / {total:,} "
          f"({100*trainable/total:.3f}%)  head trained JOINTLY with adapters",
          flush=True)

    coll = lambda b: collate(b, tok)
    dl_tr = DataLoader(SeqDS([S[i] for i in tr], y[tr], tok), batch_size=args.batch,
                       shuffle=True, collate_fn=coll, num_workers=2)
    dl_va = DataLoader(SeqDS([S[i] for i in va], y[va], tok), batch_size=args.batch * 2,
                       shuffle=False, collate_fn=coll, num_workers=2)
    dl_te = DataLoader(SeqDS([S[i] for i in te], y[te], tok), batch_size=args.batch * 2,
                       shuffle=False, collate_fn=coll, num_workers=2)

    if cfg["weighted"]:
        counts = np.bincount(y[tr], minlength=n_classes).astype(np.float64)
        w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32,
                                                          device=device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"], weight_decay=args.weight_decay)
    steps = max(1, (len(dl_tr) // args.accum)) * args.max_epochs
    sched = get_cosine_schedule_with_warmup(opt, int(steps * args.warmup_frac), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, bad = -1.0, -1, 0
    rows = []
    for epoch in range(1, args.max_epochs + 1):
        model.train(); t0 = time.perf_counter(); running = 0.0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(dl_tr):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(device == "cuda")):
                logits = model(**batch).logits
                loss = loss_fn(logits.float(), labels) / args.accum
            scaler.scale(loss).backward()
            running += loss.item() * args.accum
            if (step + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sched.step()
        vm = evaluate(model, dl_va, device, n_classes)
        row = dict(config_id=args.config_id, **cfg, corpus=args.corpus,
                   split=args.split, epoch=epoch,
                   train_loss=running / max(1, len(dl_tr)),
                   val_accuracy=vm["accuracy"], val_f1_macro=vm["f1_macro"],
                   val_balanced_accuracy=vm["balanced_accuracy"],
                   trainable_params=trainable, seconds=time.perf_counter() - t0,
                   batch=args.batch, accum=args.accum,
                   weight_decay=args.weight_decay, warmup_frac=args.warmup_frac,
                   optimizer="AdamW", scheduler="cosine", max_epochs=args.max_epochs,
                   head_joint=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(str(out).replace(".csv", f"_cfg{args.config_id}.csv"),
                                  index=False)
        print(f"  epoch {epoch:2d}  loss={row['train_loss']:.4f}  "
              f"val_acc={vm['accuracy']:.4f}  val_f1m={vm['f1_macro']:.4f}  "
              f"({row['seconds']:.0f}s)", flush=True)
        if vm["f1_macro"] > best + 1e-4:
            best, best_epoch, bad = vm["f1_macro"], epoch, 0
            tm = evaluate(model, dl_te, device, n_classes)
            best_test = {f"test_{k}": v for k, v in tm.items()
                         if isinstance(v, (int, float))}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at epoch {epoch} "
                      f"(best {best:.4f} @ {best_epoch})", flush=True)
                break

    final = dict(rows[-1]); final.update(best_val_f1_macro=best,
                                         best_epoch=best_epoch, **best_test)
    pd.DataFrame([final]).to_csv(
        str(out).replace(".csv", f"_cfg{args.config_id}_final.csv"), index=False)
    print(f"[done] cfg {args.config_id}: best val macro-F1 {best:.4f} at epoch "
          f"{best_epoch}; test macro-F1 {best_test.get('test_f1_macro'):.4f}, "
          f"test acc {best_test.get('test_accuracy'):.4f}", flush=True)


if __name__ == "__main__":
    main()
