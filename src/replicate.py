"""Replicate the head-to-head comparison on a different corpus.

A sketch evaluated only on a set of near-identical sequences is barely tested:
when almost every k-mer is shared between any two sequences, almost nothing
depends on how collisions are resolved. This script re-runs the core measurements
on any FASTA, so the claims can be checked on corpora of genuinely divergent
sequences of different lengths.

Four blocks, all written incrementally:

1. Corpus facts, including the exact-duplicate fraction and the length spread.
2. m calibration: binary search vs the closed form m >= (U-1)/(2c), at six
   collision targets. If the heuristic is a general property rather than a fit to
   one dataset, it has to hold here too.
3. Sketch vs full spectrum under stratified splits.
4. The same under similarity-aware grouped splits.

Identity here is k-mer Jaccard, not the Hamming distance used by
similarity_splits.py, because these sequences are variable length and cannot be
compared position by position. The clustering is greedy over a randomised order,
the same prefilter idea used by MMseqs2 and CD-HIT. The two measures are not
interchangeable and their thresholds should not be compared.

    python src/replicate.py --fasta data/corpus.fasta --name mycorpus \
        --labels data/labels.tsv --out results/replication/mycorpus.csv

--labels is optional and only needed when the FASTA defline carries no label.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C


def append_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def read_fasta(path):
    """Yield (id, label, sequence). Label is the second whitespace token."""
    ident = label = None
    buf = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if ident is not None:
                    yield ident, label, "".join(buf)
                buf = []
                parts = line[1:].split()
                ident = parts[0]
                label = parts[1] if len(parts) > 1 else "NA"
            else:
                buf.append(line.strip())
    if ident is not None:
        yield ident, label, "".join(buf)


def jaccard_clusters(seqs, k, threshold, rng=None):
    """Greedy single-pass clustering by k-mer Jaccard similarity.

    similarity_splits could use exact Hamming distance because every SARS-CoV-2 spike was
    1,274 aa and in register. These corpora are variable length, so identity is
    approximated by Jaccard over k-mer sets -- the same prefilter idea MMseqs2
    and CD-HIT use. Greedy assignment to the first centroid within threshold;
    order is randomised so the result does not depend on input order.
    """
    rng = rng or np.random.default_rng(0)
    sets = [frozenset(s[i:i + k] for i in range(len(s) - k + 1)) for s in seqs]
    order = rng.permutation(len(seqs))
    centroids = []          # (index, kmer set)
    assign = np.empty(len(seqs), dtype=np.int64)
    for i in order:
        si = sets[i]
        best = -1
        for ci, (idx, sc) in enumerate(centroids):
            inter = len(si & sc)
            if inter == 0:
                continue
            if inter / (len(si) + len(sc) - inter) >= threshold:
                best = ci
                break
        if best < 0:
            centroids.append((i, si))
            best = len(centroids) - 1
        assign[i] = best
    return assign, len(centroids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--labels", default=None,
                    help="optional id<TAB>label TSV; use when the FASTA defline "
                         "carries no label (e.g. the UniProt Pfam export)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--collision", type=float, default=0.06)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--jaccard", type=float, nargs="+", default=[0.9, 0.7, 0.5])
    ap.add_argument("--classifiers", nargs="+", default=["LR", "RF", "DT"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        out.unlink()
    meta_path = Path(str(out).replace(".csv", "_meta.csv"))
    if meta_path.exists():
        meta_path.unlink()
    t0 = time.time()

    # ---------------------------------------------------------- 1. corpus facts
    recs = list(read_fasta(args.fasta))
    if args.labels:
        lab = {}
        with open(args.labels) as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    lab[parts[0]] = parts[1]
        before = len(recs)
        recs = [(i, lab[i], s_) for i, _, s_ in recs if i in lab]
        print(f"[labels] joined {len(recs):,} of {before:,} records from "
              f"{args.labels}", flush=True)
    seqs_all = [s for _, _, s in recs]
    labs_all = [l for _, l, _ in recs]
    n_all = len(seqs_all)
    uniq = len(set(seqs_all))
    dup_frac = 1.0 - uniq / n_all
    lens = np.array([len(s) for s in seqs_all])
    print(f"[corpus] {args.name}: n={n_all:,}  unique={uniq:,}  "
          f"exact-duplicate fraction={dup_frac:.4f}", flush=True)
    print(f"[corpus] length min={lens.min()} median={int(np.median(lens))} "
          f"max={lens.max()} sd={lens.std():.1f}", flush=True)
    print(f"[corpus] classes: {dict(Counter(labs_all))}", flush=True)

    # deduplicate, majority label -- the same policy as the SARS-CoV-2 primary
    by_seq = defaultdict(list)
    for s, l in zip(seqs_all, labs_all):
        by_seq[s].append(l)
    seqs, labs, n_conflict = [], [], 0
    for s, ls in by_seq.items():
        c = Counter(ls)
        if len(c) > 1:
            n_conflict += 1
        seqs.append(s)
        labs.append(c.most_common(1)[0][0])
    names = sorted(set(labs))
    y = np.array([names.index(l) for l in labs])
    n_classes = len(names)
    print(f"[dedup] n={len(seqs):,}  conflicting-label sequences={n_conflict}",
          flush=True)
    print(f"[dedup] class counts: "
          f"{ {names[i]: int((y == i).sum()) for i in range(n_classes)} }",
          flush=True)

    per_seq, vocab = C.corpus_kmer_counts(seqs, args.k)
    keys = sorted(vocab)
    U = len(keys)
    print(f"[kmers] k={args.k} U={U:,}  (21^k = {21 ** args.k:,})", flush=True)

    append_row(meta_path, dict(
        corpus=args.name, n_raw=n_all, n_unique=uniq, dup_fraction=dup_frac,
        n_dedup=len(seqs), n_conflict=n_conflict, n_classes=n_classes,
        len_min=int(lens.min()), len_median=int(np.median(lens)),
        len_max=int(lens.max()), len_sd=float(lens.std()), U=U, k=args.k))

    # --------------------------------------------------- 2. m calibration (R4)
    print("\n[calib] target   binary-search m   heuristic (U-1)/(2c)   ratio",
          flush=True)
    for c in (0.40, 0.20, 0.10, 0.06, 0.02, 0.01):
        m_bin = C.m_for_collision(keys, c, seed=args.seed)
        m_heu = C.m_heuristic(U, c)
        print(f"[calib] {c:>5.0%}   {m_bin:>15,}   {m_heu:>20,}   "
              f"{m_heu / m_bin:>5.2f}", flush=True)
        append_row(meta_path, dict(corpus=args.name, calib_target=c,
                                   m_binary_search=m_bin, m_heuristic=m_heu,
                                   heuristic_ratio=m_heu / m_bin, U=U, k=args.k))

    m = C.m_for_collision(keys, args.collision, seed=args.seed)
    print(f"[calib] using m={m:,} at c={args.collision:.0%} "
          f"(achieved {C.collision_pct(keys, m, seed=args.seed):.4f})", flush=True)

    X_m2v = C.murmur2vec(per_seq, m, signed=False, seed=args.seed)
    X_spec, _ = C.full_spectrum(per_seq, vocab)
    embeddings = {"Murmur2Vec": X_m2v, "Spike2Vec": X_spec}
    print(f"[embed] Murmur2Vec D={X_m2v.shape[1]:,}  "
          f"Spike2Vec D={X_spec.shape[1]:,}", flush=True)

    # ------------------------------------------- 3. stratified head-to-head
    splits = C.stratified_splits(y, n_splits=args.splits, test_size=0.30,
                                 seed=args.seed, dataset=f"{args.name}-replicate")
    for emb, X in embeddings.items():
        for clf in args.classifiers:
            for si, (tr, te) in enumerate(splits.splits):
                try:
                    r = C.fit_eval(clf, X[tr], X[te], y[tr], y[te], n_classes,
                                   seed=args.seed)
                except Exception as e:                       # noqa: BLE001
                    print(f"  ! {emb}/{clf}/{si}: {e}", flush=True)
                    continue
                r.update(corpus=args.name, embedding=emb, protocol="stratified",
                         jaccard=np.nan, n_clusters=np.nan, split=si, m=m,
                         k=args.k)
                append_row(out, r)
        d = pd.read_csv(out)
        d = d[(d.embedding == emb) & (d.protocol == "stratified")]
        print(f"  [stratified] {emb:<11} acc={d.accuracy.mean():.4f} "
              f"f1m={d.f1_macro.mean():.4f}", flush=True)

    # -------------------------------------- 4. similarity-aware grouped splits
    for thr in args.jaccard:
        t = time.perf_counter()
        groups, ncl = jaccard_clusters(seqs, args.k, thr,
                                       rng=np.random.default_rng(args.seed))
        print(f"\n  [cluster] jaccard>={thr:.2f}: {ncl:,} clusters "
              f"({time.perf_counter() - t:.0f}s)", flush=True)
        if ncl < 10:
            print("  [cluster] too few clusters to split on; skipping", flush=True)
            continue
        gs = C.grouped_splits(y, groups, n_splits=args.splits, test_size=0.30,
                              seed=args.seed)
        for emb, X in embeddings.items():
            for clf in args.classifiers:
                for si, (tr, te) in enumerate(gs.splits):
                    if len(np.unique(y[tr])) < 2:
                        continue
                    try:
                        r = C.fit_eval(clf, X[tr], X[te], y[tr], y[te],
                                       n_classes, seed=args.seed)
                    except Exception as e:                   # noqa: BLE001
                        print(f"  ! {emb}/{clf}/{si}: {e}", flush=True)
                        continue
                    r.update(corpus=args.name, embedding=emb,
                             protocol="jaccard_group", jaccard=thr,
                             n_clusters=ncl, split=si, m=m, k=args.k)
                    append_row(out, r)
            d = pd.read_csv(out)
            d = d[(d.embedding == emb) & (d.protocol == "jaccard_group")
                  & (d.jaccard == thr)]
            if len(d):
                print(f"    {emb:<11} acc={d.accuracy.mean():.4f} "
                      f"f1m={d.f1_macro.mean():.4f}", flush=True)

    D = pd.read_csv(out)
    print("\n=== summary ===", flush=True)
    print(D.groupby(["protocol", "jaccard", "embedding"], dropna=False)
          [["accuracy", "f1_macro"]].agg(["mean", "std"]).round(4).to_string(),
          flush=True)
    print(f"\n[done] {len(D)} rows in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
