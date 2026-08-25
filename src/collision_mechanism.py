"""Why do collisions not destroy accuracy? An interventional test.

Observing that a sketch works is not an explanation. This script intervenes.

1. Adversarial collisions. Select the top-N k-mers by mutual information with the
   label (computed on training folds only) and force them into a single bucket.
   Compare against forcing an equal number of FREQUENCY-MATCHED but
   non-discriminative k-mers into a bucket. If accuracy depends on discriminative
   k-mers avoiding collisions, the first collapses performance and the second
   does nothing. Sweeping N gives a dose-response curve.

   The frequency-matched control is what makes this an argument: without it, any
   damage could be attributed to removing frequent features rather than
   informative ones.

2. Gaussian random projection at matched dimension, as a negative control. This
   separates two effects that are easy to conflate: hashing loses little relative
   to the full spectrum, and hashing beats a dense projection -- but the second
   holds only for axis-aligned classifiers, and vanishes for a rotation-invariant
   linear model, exactly as Johnson-Lindenstrauss theory predicts.

3. Signal-to-noise as a function of k-mer frequency.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C
import corpora as D


def discriminative_kmers(Xspec, y, keys, tr, top_n):
    """Top-n k-mers by mutual information with the label, fit on train rows only."""
    from sklearn.feature_selection import mutual_info_classif
    sub = Xspec[tr]
    var = np.asarray(sub.power(2).mean(axis=0) - np.square(sub.mean(axis=0))).ravel()
    cand = np.argsort(-var)[:min(1500, len(keys))]          # prefilter, MI is slow
    mi = mutual_info_classif(sub[:, cand].toarray(), y[tr],
                             discrete_features=True, random_state=0)
    order = cand[np.argsort(-mi)]
    return [keys[i] for i in order[:top_n]], mi


def frequency_matched(keys, freqs, chosen, rng):
    """Pick controls with the same corpus-frequency profile as `chosen`."""
    chosen_set = set(chosen)
    idx = {w: i for i, w in enumerate(keys)}
    pool = np.array([i for i, w in enumerate(keys) if w not in chosen_set])
    pool_f = freqs[pool]
    out = []
    used = set()
    for w in chosen:
        target = freqs[idx[w]]
        d = np.abs(pool_f - target)
        for j in np.argsort(d):
            if pool[j] not in used:
                used.add(pool[j]); out.append(keys[pool[j]]); break
    return out


def collapse_map(kmers, n_buckets, m, seed):
    """Force the given k-mers into a handful of shared buckets."""
    rng = np.random.default_rng(seed)
    targets = rng.choice(m, size=n_buckets, replace=False)
    return {w: int(targets[i % n_buckets]) for i, w in enumerate(kmers)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ms", type=int, nargs="+", default=[2000, 8000])
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpora", nargs="+", default=["dedup", "raw"])
    ap.add_argument("--classifiers", nargs="+", default=["LR", "RF", "DT"])
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--collapse-buckets", type=int, default=4)
    ap.add_argument("--out", default="results/collision_mechanism_mechanism.csv")
    args = ap.parse_args()

    rows, snr_rows = [], []
    for corpus in args.corpora:
        S, y, names, info = D.get_corpus(corpus)
        n_classes = len(names)
        per_seq, vocab = C.corpus_kmer_counts(S, args.k)
        keys = sorted(vocab)
        Xspec, keys = C.full_spectrum(per_seq, vocab)
        freqs = np.asarray(Xspec.sum(axis=0)).ravel()
        splits = C.stratified_splits(y, n_splits=args.splits, test_size=0.30,
                                     seed=args.seed, dataset=corpus)
        print(f"\n===== {corpus} n={len(S)} D={Xspec.shape[1]} =====", flush=True)

        # ---- 2. SNR vs frequency, measured -------------------------------
        m_snr = args.ms[-1]
        Xm = C.murmur2vec(per_seq, m_snr, signed=True, seed=args.seed)
        h = C.hash_fn("murmur", args.seed)
        Xd = Xspec[:200].toarray()
        Xmd = Xm[:200].toarray()
        for i, w in enumerate(keys):
            j = h(w) % m_snr
            true = Xd[:, i]
            if true.sum() == 0:
                continue
            interference = Xmd[:, j] - true
            snr_rows.append(dict(corpus=corpus, kmer=w, freq=float(freqs[i]),
                                 m=m_snr,
                                 signal=float(np.mean(true ** 2)),
                                 noise=float(np.mean(interference ** 2))))
        print(f"[snr] measured interference for {len(keys)} k-mers at m={m_snr}",
              flush=True)

        for m in args.ms:
            variants = {
                "Murmur2Vec": C.murmur2vec(per_seq, m, signed=False, seed=args.seed),
                "GaussianJL": C.gaussian_sketch(Xspec, m, seed=args.seed),
                "FullSpectrum": Xspec,
            }
            for vname, X in variants.items():
                for clf in args.classifiers:
                    got = []
                    for si, (tr, te) in enumerate(splits.splits):
                        try:
                            r = C.fit_eval(clf, X[tr], X[te], y[tr], y[te],
                                           n_classes, seed=args.seed)
                        except Exception as e:
                            print(f"  ! {vname}/{clf}/{si}: {e}", flush=True); continue
                        r.update(corpus=corpus, variant=vname, m=m, split=si,
                                 dim=X.shape[1], arm="baseline")
                        rows.append(r); got.append(r)
                    if got:
                        print(f"  {corpus:<6} m={m:<6} {vname:<13} {clf:<3} "
                              f"acc={np.mean([g['accuracy'] for g in got]):.4f} "
                              f"f1m={np.mean([g['f1_macro'] for g in got]):.4f}",
                              flush=True)

            # ---- 3. intervention ----------------------------------------
            rng = np.random.default_rng(args.seed)
            for si, (tr, te) in enumerate(splits.splits):
                disc, _ = discriminative_kmers(Xspec, y, keys, tr, args.top_n)
                ctrl = frequency_matched(keys, freqs, disc, rng)
                arms = {
                    "collapse_discriminative": collapse_map(disc, args.collapse_buckets, m, args.seed),
                    "collapse_control": collapse_map(ctrl, args.collapse_buckets, m, args.seed),
                }
                for arm, fmap in arms.items():
                    Xi = C.murmur2vec(per_seq, m, signed=False, seed=args.seed,
                                      force_collide=fmap)
                    for clf in args.classifiers:
                        try:
                            r = C.fit_eval(clf, Xi[tr], Xi[te], y[tr], y[te],
                                           n_classes, seed=args.seed)
                        except Exception as e:
                            print(f"  ! {arm}/{clf}/{si}: {e}", flush=True); continue
                        r.update(corpus=corpus, variant="Murmur2Vec", m=m,
                                 split=si, dim=Xi.shape[1], arm=arm,
                                 n_collapsed=len(fmap))
                        rows.append(r)
            sel = pd.DataFrame([r for r in rows
                                if r.get("m") == m and r["corpus"] == corpus])
            if not sel.empty:
                print(sel.groupby(["arm", "classifier"])[["accuracy", "f1_macro"]]
                      .mean().round(4).to_string(), flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(snr_rows).to_csv(str(out).replace(".csv", "_snr.csv"), index=False)
    print(f"\n[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
