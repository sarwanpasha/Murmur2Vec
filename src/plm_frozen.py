"""Frozen protein language model baselines.

Embeds the corpus with several ESM-2 checkpoints and evaluates the embeddings
with the same classifiers and splits as every other method here. Three axes:

    model size    a ladder of checkpoints, to show whether the comparison is an
                  artifact of using a small model
    regime        truncate (feed the first context-window residues), window
                  (tile the whole sequence and pool across windows), and a
                  length-matched sketch that sees only what the model sees
    pooling       mean, CLS and max

The length-matched arm matters: if the sequences are longer than the model's
context window, an unmatched comparison gives the sketch strictly more input, and
that objection has to be answered with data rather than argument.

Only distinct sequences are embedded; duplicates are indexed into the result.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C
import corpora as D

MODELS = {
    "esm2_8M":   "facebook/esm2_t6_8M_UR50D",
    "esm2_35M":  "facebook/esm2_t12_35M_UR50D",
    "esm2_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_650M": "facebook/esm2_t33_650M_UR50D",
}
MAXLEN = 1022


def windows(s, size=MAXLEN, overlap=256):
    step = size - overlap
    out, i = [], 0
    while i < len(s):
        out.append(s[i:i + size])
        if i + size >= len(s):
            break
        i += step
    return out


@torch.no_grad()
def embed(model, tok, seqs, device, batch=8, pooling=("mean", "cls", "max")):
    """Returns {pooling: array}.  Padding is masked out of mean and max."""
    out = {p: [] for p in pooling}
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXLEN + 2)
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model(**enc).last_hidden_state            # B x T x H
        mask = enc["attention_mask"].unsqueeze(-1).bool()
        if "mean" in out:
            summed = (h * mask).sum(1)
            out["mean"].append((summed / mask.sum(1).clamp(min=1)).float().cpu())
        if "cls" in out:
            out["cls"].append(h[:, 0].float().cpu())
        if "max" in out:
            out["max"].append(h.masked_fill(~mask, -1e4).max(1).values.float().cpu())
    return {p: torch.cat(v).numpy() for p, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--regimes", nargs="+", default=["truncate", "window"])
    ap.add_argument("--corpora", nargs="+", default=["dedup", "raw"])
    ap.add_argument("--classifiers", nargs="+", default=["LR", "RF", "DT"])
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--collision", type=float, default=0.06)
    ap.add_argument("--out", default="results/plm_frozen_esm2.csv")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} torch={torch.__version__}", flush=True)

    S_all, Y_all = D.load_covid_raw()
    uniq = sorted(set(S_all))
    u_index = {s: i for i, s in enumerate(uniq)}
    print(f"[data] {len(S_all)} sequences, {len(uniq)} distinct "
          f"-> embedding only the distinct ones", flush=True)
    print(f"[data] length audit: {json.dumps(D.corpus_stats(S_all, Y_all))}",
          flush=True)

    trunc = [s[:MAXLEN] for s in uniq]
    rows = []

    # ---- length-matched sketch control (no GPU needed) --------------------
    for corpus in args.corpora:
        S, y, names, _ = D.get_corpus(corpus)
        n_classes = len(names)
        splits = C.stratified_splits(y, n_splits=args.splits, test_size=0.30,
                                     seed=args.seed, dataset=corpus).splits
        for tag, seqs in (("m2v_full", S), ("m2v_trunc", [s[:MAXLEN] for s in S])):
            per_seq, vocab = C.corpus_kmer_counts(seqs, args.k)
            m = C.m_for_collision(sorted(vocab), args.collision, seed=args.seed)
            X = C.murmur2vec(per_seq, m, signed=False, seed=args.seed)
            for clf in args.classifiers:
                for si, (tr, te) in enumerate(splits):
                    r = C.fit_eval(clf, X[tr], X[te], y[tr], y[te], n_classes,
                                   seed=args.seed)
                    r.update(corpus=corpus, model="Murmur2Vec", regime=tag,
                             pooling="-", split=si, dim=m)
                    rows.append(r)
            sel = [r for r in rows if r.get("regime") == tag and r["corpus"] == corpus]
            print(f"  {corpus:<6} {tag:<10} acc={np.mean([s['accuracy'] for s in sel]):.4f} "
                  f"f1m={np.mean([s['f1_macro'] for s in sel]):.4f}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    # ---- frozen ESM-2 ------------------------------------------------------
    for mname in args.models:
        repo = MODELS[mname]
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo, torch_dtype=torch.float16).to(device).eval()
        print(f"\n[model] {mname} loaded in {time.perf_counter()-t0:.0f}s", flush=True)

        for regime in args.regimes:
            t0 = time.perf_counter()
            if regime == "truncate":
                E = embed(model, tok, trunc, device, args.batch)
            else:
                flat, owner = [], []
                for i, s in enumerate(uniq):
                    for w in windows(s):
                        flat.append(w); owner.append(i)
                Ew = embed(model, tok, flat, device, args.batch)
                owner = np.asarray(owner)
                E = {}
                for p, arr in Ew.items():
                    agg = np.zeros((len(uniq), arr.shape[1]), dtype=np.float32)
                    cnt = np.zeros(len(uniq), dtype=np.float32)
                    np.add.at(agg, owner, arr)
                    np.add.at(cnt, owner, 1.0)
                    E[p] = agg / cnt[:, None]
            print(f"  [embed] {mname}/{regime} {time.perf_counter()-t0:.0f}s "
                  f"dim={next(iter(E.values())).shape[1]}", flush=True)

            for corpus in args.corpora:
                S, y, names, _ = D.get_corpus(corpus)
                n_classes = len(names)
                idx = np.array([u_index[s] for s in S])
                splits = C.stratified_splits(y, n_splits=args.splits,
                                             test_size=0.30, seed=args.seed,
                                             dataset=corpus).splits
                for pool, arr in E.items():
                    X = arr[idx]
                    for clf in args.classifiers:
                        got = []
                        for si, (tr, te) in enumerate(splits):
                            try:
                                r = C.fit_eval(clf, X[tr], X[te], y[tr], y[te],
                                               n_classes, seed=args.seed,
                                               standardize=True)
                            except Exception as e:
                                print(f"   ! {mname}/{regime}/{pool}/{clf}: {e}",
                                      flush=True); continue
                            r.update(corpus=corpus, model=mname, regime=regime,
                                     pooling=pool, split=si, dim=X.shape[1])
                            rows.append(r); got.append(r)
                        if got:
                            print(f"    {corpus:<6} {mname:<9} {regime:<8} {pool:<4} "
                                  f"{clf:<3} acc={np.mean([g['accuracy'] for g in got]):.4f} "
                                  f"f1m={np.mean([g['f1_macro'] for g in got]):.4f}",
                                  flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\n=== summary (best classifier per cell) ===", flush=True)
    print(df.groupby(["corpus", "model", "regime", "pooling"])
          [["accuracy", "f1_macro"]].max().round(4).to_string(), flush=True)
    print(f"\n[done] {len(df)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
