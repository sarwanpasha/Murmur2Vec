"""Sweep k and the sketch size m: how much does the collision rate cost?

For each k, calibrates m to a range of target collision rates, then measures
accuracy, macro-F1 and peak memory. Two questions:

1. Does the closed form m >= ceil((U-1)/(2c)) reproduce the binary search? It
   does to within a few percent for c <= 10%, drifting to about 1.4x at c = 40%,
   which turns m selection into one O(N) pass to count unique k-mers instead of a
   grid search.

2. Is a compressive setting free? Accuracy is flat across a very wide range of m,
   so the smallest m meeting a loose collision target is usually as good as a
   much larger one -- and at small k the sketch can easily be LARGER than simply
   storing the observed spectrum, which is worth checking before claiming
   compression.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C


def hashed_keys(keys, family="murmur", seed=0):
    h = C.hash_fn(family, seed)
    return np.fromiter((h(w) for w in keys), dtype=np.uint64, count=len(keys))


def collision_from_hashes(H, m):
    return 1.0 - np.unique(H % np.uint64(m)).size / H.size


def calibrate(H, target, lo=None, hi=None):
    U = H.size
    lo = lo or max(2, U // 500)
    hi = hi or max(lo * 2, 600 * U)
    if collision_from_hashes(H, hi) > target:
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if collision_from_hashes(H, mid) <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--collisions", type=float, nargs="+",
                    default=[0.40, 0.30, 0.20, 0.10, 0.06, 0.04, 0.02, 0.01])
    ap.add_argument("--eval-ks", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--eval-collisions", type=float, nargs="+",
                    default=[0.40, 0.20, 0.06, 0.01])
    ap.add_argument("--classifiers", nargs="+", default=["LR", "RF", "DT"])
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    seqs, y, names = C.load_covid()
    n_classes = len(names)
    splits = C.stratified_splits(y, n_splits=args.splits, test_size=0.30,
                                 seed=args.seed, dataset="covid7000")
    print(f"[data] {len(seqs)} seqs, {n_classes} classes", flush=True)

    calib_rows, eval_rows = [], []
    for k in args.ks:
        t0 = time.perf_counter()
        per_seq, vocab = C.corpus_kmer_counts(seqs, k)
        keys = sorted(vocab)
        U = len(keys)
        D_theory = 21 ** k
        H = hashed_keys(keys, seed=args.seed)
        print(f"\n[k={k}] U={U}  21^k={D_theory}  "
              f"kmer pass {time.perf_counter()-t0:.1f}s", flush=True)

        for c in args.collisions:
            t = time.perf_counter()
            m = calibrate(H, c)
            achieved = collision_from_hashes(H, m)
            row = dict(k=k, target_collision=c, m=m,
                       achieved_collision=float(achieved),
                       U=U, D_theory=D_theory,
                       m_heuristic=C.m_heuristic(U, c),
                       m_exact_model=C.m_exact(U, c),
                       compression_vs_theory=m / D_theory,
                       compression_vs_vocab=m / U,
                       calib_seconds=time.perf_counter() - t)
            row["heuristic_ratio"] = row["m_heuristic"] / m
            calib_rows.append(row)
            print(f"  c={c:>5.0%} m={m:<10d} achieved={achieved:.4f} "
                  f"heuristic={row['m_heuristic']:<10d} "
                  f"(x{row['heuristic_ratio']:.2f})  m/21^k={m/D_theory:.4g}",
                  flush=True)

        if k in args.eval_ks:
            for c in args.eval_collisions:
                m = next(r["m"] for r in calib_rows
                         if r["k"] == k and r["target_collision"] == c)
                t = time.perf_counter()
                X = C.murmur2vec(per_seq, m, signed=False, seed=args.seed)
                embed_s = time.perf_counter() - t
                mem_mb = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6
                for clf in args.classifiers:
                    for si, (tr, te) in enumerate(splits.splits):
                        try:
                            r = C.fit_eval(clf, X[tr], X[te], y[tr], y[te],
                                           n_classes, seed=args.seed)
                        except Exception as e:
                            print(f"   ! k={k} c={c} {clf} split{si}: {e}", flush=True)
                            continue
                        r.update(k=k, target_collision=c, m=m, split=si,
                                 embed_seconds=embed_s, sketch_mb=mem_mb,
                                 compression_vs_theory=m / (21 ** k))
                        eval_rows.append(r)
                sel = [r for r in eval_rows if r["k"] == k and r["target_collision"] == c]
                if sel:
                    print(f"  eval k={k} c={c:.0%} m={m}: "
                          f"acc={np.mean([r['accuracy'] for r in sel]):.4f} "
                          f"f1m={np.mean([r['f1_macro'] for r in sel]):.4f} "
                          f"({embed_s:.1f}s embed, {mem_mb:.1f} MB)", flush=True)
        del per_seq, vocab, keys, H

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(calib_rows).to_csv(out / "calibration_grid_calibration.csv", index=False)
    pd.DataFrame(eval_rows).to_csv(out / "calibration_grid_eval.csv", index=False)
    print(f"\n[done] -> {out}/calibration_grid_calibration.csv, {out}/calibration_grid_eval.csv", flush=True)


if __name__ == "__main__":
    main()
