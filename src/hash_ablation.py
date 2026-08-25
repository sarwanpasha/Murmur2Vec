"""Is the method about one hash function, or about feature hashing?

Runs one hash family per invocation so families can run in parallel, each writing
its own file. Two parts.

A. CLASSIFICATION. Sketch the corpus with this family at the m calibrated for a
   target collision rate, signed and unsigned, over several hash seeds and
   several splits. Each family calibrates its OWN m, so the comparison is at
   matched collision rate rather than at one family's m.

B. HASH DIAGNOSTICS. Collision rate, bucket-load uniformity (chi-square),
   avalanche, throughput, and two measures of pairwise independence.

Two measurement details that are easy to get wrong:

    Avalanche must be normalised by the MEASURED output width, not by an assumed
    64 bits. MurmurHash3 returns 32 bits, so dividing by 64 makes a perfect hash
    look like it flips a quarter of its bits.

    Pairwise independence must be measured at a SMALL number of buckets. At the
    sketch's own m almost every key lands in its own bucket under both seeds, so
    the mutual information between two seeds approaches 1 for every family
    including a cryptographic hash -- a statistic that measures the load factor
    rather than the hash. Measured at m = 64 the labellings are genuinely coarse.

The interesting comparison is a family with weak avalanche but sound pairwise
independence against a cryptographic hash: if the two classify the same, the task
needs only 2-wise independence, and the choice of hash is a speed decision.

    python src/hash_ablation.py --family fnv1a --seeds 0 1 2 --splits 5 \
        --out results/hash_ablation/fnv1a.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C
import corpora as D


def append_row(path: Path, row: dict):
    """Write one row immediately; header only on first write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


# ---------------------------------------------------------------- diagnostics
def bucket_loads(keys, m, family, seed):
    f = C.hash_fn(family, seed)
    idx = np.fromiter((f(k) % m for k in keys), dtype=np.int64, count=len(keys))
    return idx


def uniformity(idx, m):
    """Chi-square of bucket occupancy against uniform. Returns (chi2/df, p)."""
    from scipy import stats
    counts = np.bincount(idx, minlength=m)
    n = counts.sum()
    exp = n / m
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    df = m - 1
    p = float(stats.chi2.sf(chi2, df))
    return chi2 / df, p


def hash_width(family, seed, keys, n=2000):
    """Effective output width: how many bit positions actually vary.

    mmh3.hash returns 32 bits, xxh64/fnv1a/sha256-truncated return 64, and the
    tabulation construction here masks to 63. Normalising avalanche by a fixed 64
    would make a perfect 32-bit hash look like it only flips half its bits, so
    measure the width instead of assuming it.
    """
    f = C.hash_fn(family, seed)
    ref = f(keys[0])
    acc = 0
    for kk in keys[:n]:
        acc |= (f(kk) ^ ref)
    return max(1, bin(acc).count("1"))


def avalanche(family, seed, keys, n=4000, rng=None):
    """Mean fraction of output bits that flip when one input character changes.

    Ideal for a well-mixing hash is 0.5. A weak family sits well below it.
    Normalised by the measured output width, not by an assumed 64.
    """
    rng = rng or np.random.default_rng(0)
    f = C.hash_fn(family, seed)
    width = hash_width(family, seed, keys)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    picks = rng.choice(len(keys), size=min(n, len(keys)), replace=False)
    fracs = []
    for i in picks:
        s_ = keys[i]
        pos = int(rng.integers(0, len(s_)))
        repl = alphabet[int(rng.integers(0, len(alphabet)))]
        if repl == s_[pos]:
            continue
        t = s_[:pos] + repl + s_[pos + 1:]
        x = f(s_) ^ f(t)
        fracs.append(bin(x).count("1") / width)
    return float(np.mean(fracs)), float(np.std(fracs)), int(width)


def pair_independence(keys, family, seed_a, seed_b, m_small=64):
    """Normalised mutual information between bucket index under two seeds.

    Measured at a DELIBERATELY SMALL m (default 64), not at the sketch's m.
    At the sketch's m (~27,000 buckets for 3,492 keys) almost every key gets its
    own bucket under both seeds, so both labelings are nearly bijective with key
    identity and NMI is ~0.99 for every family including cryptographic ones --
    a degenerate statistic that measures the load factor, not the hash. With 64
    buckets and 3,492 keys the labelings are genuinely coarse and NMI ~0 is the
    signal that the two seeds are behaving independently.
    """
    from sklearn.metrics import normalized_mutual_info_score
    a = bucket_loads(keys, m_small, family, seed_a)
    b = bucket_loads(keys, m_small, family, seed_b)
    return float(normalized_mutual_info_score(a, b))


def pairwise_collision(keys, family, seed, m_small=64, n_pairs=200000, rng=None):
    """Empirical P(h(x) == h(y) mod m) over random distinct key pairs.

    This is the quantity 2-wise independence actually promises: it should equal
    1/m regardless of the key distribution. Reported as the ratio to 1/m, so
    1.00 is ideal and a family that clusters keys shows > 1.
    """
    rng = rng or np.random.default_rng(12345)
    idx = bucket_loads(keys, m_small, family, seed)
    n = len(idx)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    rate = float((idx[i[keep]] == idx[j[keep]]).mean())
    return rate, rate * m_small


def throughput(family, seed, keys, repeats=3):
    f = C.hash_fn(family, seed)
    best = None
    for _ in range(repeats):
        t = time.perf_counter()
        for k in keys:
            f(k)
        dt = time.perf_counter() - t
        best = dt if best is None else min(best, dt)
    return len(keys) / best


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(C.HASH_FAMILIES))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--collision", type=float, default=0.06)
    ap.add_argument("--corpus", default="dedup")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--classifiers", nargs="+", default=["LR", "RF", "DT"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-spectrum", action="store_true")
    args = ap.parse_args()

    fam = args.family
    out = Path(args.out)
    if out.exists():
        out.unlink()
    diag_path = Path(str(out).replace(".csv", "_diag.csv"))
    if diag_path.exists():
        diag_path.unlink()

    t0 = time.time()
    seqs, y, names, info = D.get_corpus(args.corpus)
    n_classes = len(names)
    print(f"[data] corpus={args.corpus} n={len(seqs)} classes={n_classes}", flush=True)

    per_seq, vocab = C.corpus_kmer_counts(seqs, args.k)
    keys = sorted(vocab)
    U = len(keys)
    print(f"[kmers] k={args.k} U={U}", flush=True)

    # m is calibrated PER FAMILY, so every family is compared at its own 6%
    # collision point rather than at murmur's m. That is the fair comparison:
    # we are asking whether the family matters once collision rate is matched.
    m = C.m_for_collision(keys, args.collision, hash_family=fam, seed=0)
    achieved = C.collision_pct(keys, m, hash_family=fam, seed=0)
    m_murmur = C.m_for_collision(keys, args.collision, hash_family="murmur", seed=0)
    print(f"[calib] {fam}: m={m} (achieved {achieved:.4f}); murmur would use {m_murmur}",
          flush=True)

    # ---------------- part B: diagnostics (cheap, do first so they always land)
    print("[diag] running hash diagnostics", flush=True)
    for seed in args.seeds:
        idx = bucket_loads(keys, m, fam, seed)
        chi2_df, chi2_p = uniformity(idx, m)
        av_mean, av_sd, width = avalanche(fam, seed, keys)
        occ = len(set(idx.tolist()))
        row = dict(family=fam, seed=seed, m=m, U=U, k=args.k,
                   m_murmur_equiv=m_murmur,
                   collision_pct=C.collision_pct(keys, m, hash_family=fam, seed=seed),
                   occupied_buckets=occ, load_factor=U / m,
                   chi2_over_df=chi2_df, chi2_p=chi2_p,
                   avalanche_mean=av_mean, avalanche_sd=av_sd,
                   output_bits=width,
                   max_bucket_load=int(np.bincount(idx).max()),
                   hashes_per_sec=throughput(fam, seed, keys))
        pc_rate, pc_ratio = pairwise_collision(keys, fam, seed)
        row["pairwise_collision_rate_m64"] = pc_rate
        row["pairwise_collision_ratio"] = pc_ratio
        row["nmi_vs_seed0"] = (
            np.nan if seed == args.seeds[0]
            else pair_independence(keys, fam, args.seeds[0], seed))
        append_row(diag_path, row)
        print(f"  seed {seed}: coll={row['collision_pct']:.4f} "
              f"chi2/df={chi2_df:.3f} (p={chi2_p:.3g}) "
              f"avalanche={av_mean:.4f} ({width}b) "
              f"maxload={row['max_bucket_load']} "
              f"{row['hashes_per_sec']:,.0f} hash/s", flush=True)

    # ---------------- part A: classification
    splits = C.stratified_splits(y, n_splits=args.splits, test_size=0.30,
                                 seed=0, dataset=f"{args.corpus}-hash_ablation")

    arms = []
    for seed in args.seeds:
        for signed in (False, True):
            arms.append((f"{fam}{'-signed' if signed else ''}", seed, signed))

    for label, seed, signed in arms:
        t = time.perf_counter()
        X = C.murmur2vec(per_seq, m, signed=signed, hash_family=fam,
                         seed=seed, sign_seed=seed + 1000)
        embed_s = time.perf_counter() - t
        for clf in args.classifiers:
            for si, (tr, te) in enumerate(splits.splits):
                try:
                    r = C.fit_eval(clf, X[tr], X[te], y[tr], y[te], n_classes, seed=0)
                except Exception as e:                       # noqa: BLE001
                    print(f"  ! {label}/{clf}/split{si}: {e}", flush=True)
                    continue
                r.update(family=fam, arm=label, signed=signed, hash_seed=seed,
                         split=si, m=m, k=args.k, corpus=args.corpus,
                         embed_seconds=embed_s)
                append_row(out, r)
        done = pd.read_csv(out)
        cur = done[(done.arm == label)]
        print(f"  {label:<18} seed={seed} acc={cur.accuracy.mean():.4f} "
              f"f1m={cur.f1_macro.mean():.4f}  (embed {embed_s:.1f}s)", flush=True)

    # ---------------- reference: full spectrum, same splits, once
    if not args.skip_spectrum:
        Xs, _ = C.full_spectrum(per_seq, vocab)
        for clf in args.classifiers:
            for si, (tr, te) in enumerate(splits.splits):
                try:
                    r = C.fit_eval(clf, Xs[tr], Xs[te], y[tr], y[te], n_classes, seed=0)
                except Exception as e:                       # noqa: BLE001
                    print(f"  ! spectrum/{clf}/split{si}: {e}", flush=True)
                    continue
                r.update(family="__spectrum__", arm="Spike2Vec", signed=False,
                         hash_seed=-1, split=si, m=Xs.shape[1], k=args.k,
                         corpus=args.corpus, embed_seconds=np.nan)
                append_row(out, r)
        print("  Spike2Vec reference done", flush=True)

    D_ = pd.read_csv(out)
    print("\n=== mean over splits and seeds ===", flush=True)
    print(D_.groupby(["arm", "classifier"])[["accuracy", "f1_macro"]]
          .agg(["mean", "std"]).round(4).to_string(), flush=True)
    print(f"\n[done] {len(D_)} rows in {time.time()-t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
