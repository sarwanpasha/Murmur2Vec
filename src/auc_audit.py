"""Diagnose disagreement between AUC definitions on an imbalanced problem.

Two methods within half a point of each other on accuracy cannot legitimately
differ by a wide margin on AUC. When they appear to, the cause is usually that
the two numbers were computed under different conventions. This script recomputes
both embeddings through one code path and prints every AUC definition side by
side -- one-vs-rest macro and weighted, one-vs-one macro, and the pathological
hard-label variant -- so the discrepancy can be attributed instead of guessed at.

On a corpus with a dominant class, OvR-weighted can sit far above OvR-macro. Pick
one, say which, and report the others.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--collision", type=float, default=0.06)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/stats_auc_audit.csv")
    ap.add_argument("--classifiers", nargs="+", default=list(C.CLASSIFIERS))
    args = ap.parse_args()

    t0 = time.time()
    seqs, y, names = C.load_covid()
    n_classes = len(names)
    print(f"[data] {len(seqs)} sequences, {n_classes} lineages", flush=True)
    print(f"[data] length stats: {json.dumps(C.length_stats(seqs))}", flush=True)

    per_seq, vocab = C.corpus_kmer_counts(seqs, args.k)
    U = C.unique_kmer_count(seqs, args.k)
    print(f"[kmers] k={args.k}  unique k-mers U={U}  vocab={len(vocab)}", flush=True)

    m = C.m_for_collision(sorted(vocab), args.collision, seed=args.seed)
    achieved = C.collision_pct(sorted(vocab), m, seed=args.seed)
    print(f"[calib] target {args.collision:.0%} -> m={m} (achieved {achieved:.4f})",
          flush=True)
    print(f"[calib] heuristic m>=(U-1)/(2c) would give m={C.m_heuristic(U, args.collision)}",
          flush=True)

    embeddings = {}
    t = time.perf_counter()
    embeddings["Murmur2Vec"] = C.murmur2vec(per_seq, m, signed=False, seed=args.seed)
    print(f"[embed] Murmur2Vec  m={m}  {time.perf_counter()-t:.2f}s", flush=True)

    t = time.perf_counter()
    Xspec, keys = C.full_spectrum(per_seq, vocab)
    embeddings["Spike2Vec"] = Xspec
    print(f"[embed] Spike2Vec   D={Xspec.shape[1]}  {time.perf_counter()-t:.2f}s",
          flush=True)

    splits = C.stratified_splits(y, n_splits=args.splits, test_size=0.30,
                                 seed=args.seed, dataset="covid7000")

    audit = C.duplicate_audit(seqs, splits.splits[0])
    print(f"[leak] split 0: {audit['n_exact_dup']}/{audit['n_test']} test sequences "
          f"({audit['frac_exact_dup']:.1%}) are exact duplicates of a training sequence",
          flush=True)

    rows = []
    for emb_name, X in embeddings.items():
        for clf_name in args.classifiers:
            for si, (tr, te) in enumerate(splits.splits):
                try:
                    res = C.fit_eval(clf_name, X[tr], X[te], y[tr], y[te],
                                     n_classes, seed=args.seed,
                                     all_auc_variants=True)
                except Exception as e:
                    print(f"  ! {emb_name}/{clf_name}/split{si}: {e}", flush=True)
                    continue
                res.update(embedding=emb_name, split=si, m=m, k=args.k)
                rows.append(res)
            done = [r for r in rows if r["embedding"] == emb_name
                    and r["classifier"] == clf_name]
            if done:
                acc = np.mean([r["accuracy"] for r in done])
                a_mac = np.nanmean([r["roc_auc_ovr_macro"] for r in done])
                a_wt = np.nanmean([r.get("roc_auc_ovr_weighted", np.nan) for r in done])
                a_hw = np.nanmean([r.get("roc_auc_hardlabel_ovr_weighted", np.nan) for r in done])
                a_hm = np.nanmean([r.get("roc_auc_hardlabel_ovr_macro", np.nan) for r in done])
                print(f"  {emb_name:<11} {clf_name:<4} acc={acc:.4f}  "
                      f"AUC ovr-macro={a_mac:.4f}  ovr-weighted={a_wt:.4f}  "
                      f"hard-weighted={a_hw:.4f}  hard-macro={a_hm:.4f}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    keep = ["accuracy", "f1_macro", "roc_auc_ovr_macro", "roc_auc_ovr_weighted",
            "roc_auc_ovo_macro", "roc_auc_hardlabel_ovr_weighted",
            "roc_auc_hardlabel_ovr_macro"]
    summary = df.groupby(["embedding", "classifier"])[keep].mean().round(4)
    print("\n=== mean over splits ===", flush=True)
    print(summary.to_string(), flush=True)
    summary.to_csv(str(out).replace(".csv", "_summary.csv"))

    print(f"\n[done] {len(df)} rows in {time.time()-t0:.1f}s -> {out}", flush=True)
    print("Compare against submitted Table 5: Spike2Vec 0.859/0.842, "
          "Murmur2Vec 0.854/0.990", flush=True)


if __name__ == "__main__":
    main()
