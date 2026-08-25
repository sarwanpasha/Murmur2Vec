"""Similarity-aware evaluation: what the benchmark is worth without leakage.

A random train/test split over a corpus of closely related sequences measures
partly the ability to recognise near-copies. This script re-evaluates under
progressively stricter protocols:

    identity_group   cluster sequences by sequence identity and keep whole
                     clusters on one side of the split, at several thresholds
    few_shot         hold out a rare class and expose only n labelled examples
    novelty          leave one class out entirely and score how well an
                     out-of-distribution signal (1 - max predicted probability)
                     separates the unseen class from the known ones

Clustering is exact single-linkage on Hamming distance with union-find, which is
valid only when all sequences are the same length and in register; for variable
length corpora use the Jaccard clustering in replicate.py instead, and do not
compare the thresholds across the two.

Accuracy is not monotone across thresholds, because grouping changes the test-set
class mix. Macro-F1 is, and is the metric to report under these protocols.

Results are checkpointed after each block, so a job stopped early keeps its work.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C
import corpora as D


def seq_matrix(S):
    L = len(S[0])
    assert all(len(s) == L for s in S), "identity clustering needs equal lengths"
    return np.frombuffer("".join(S).encode(), dtype=np.uint8).reshape(len(S), L)


def cluster_by_identity(M, threshold, block=512):
    """Single-linkage clusters at >= threshold identity, via union-find."""
    n, L = M.shape
    max_mismatch = int(np.floor((1.0 - threshold) * L))
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i0 in range(0, n, block):
        A = M[i0:i0 + block]
        for j0 in range(i0, n, block):
            B = M[j0:j0 + block]
            mism = (A[:, None, :] != B[None, :, :]).sum(axis=2)
            ii, jj = np.where(mism <= max_mismatch)
            for a, b in zip(ii + i0, jj + j0):
                if a < b:
                    union(int(a), int(b))
    labels = np.array([find(i) for i in range(n)])
    _, comp = np.unique(labels, return_inverse=True)
    return comp


def evaluate(X, y, splits, n_classes, clfs, seed, **tags):
    rows = []
    for clf in clfs:
        for si, (tr, te) in enumerate(splits):
            if len(np.unique(y[tr])) < 2:
                continue
            try:
                r = C.fit_eval(clf, X[tr], X[te], y[tr], y[te], n_classes, seed=seed)
            except Exception as e:
                print(f"   ! {tags} {clf} split{si}: {e}", flush=True)
                continue
            r.update(split=si, **tags)
            rows.append(r)
    return rows


def novelty_auroc(X, y, held_class, n_classes, clf_name, seed):
    """Train without one lineage; score test rows by 1 - max class probability."""
    from sklearn.metrics import roc_auc_score
    idx = np.arange(len(y))
    known = idx[y != held_class]
    novel = idx[y == held_class]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(known)
    cut = int(0.7 * len(perm))
    tr, te_known = perm[:cut], perm[cut:]
    clf = C.make_classifier(clf_name, seed=seed)
    Xtr = X[tr]
    if clf_name in C.DENSE_REQUIRED and hasattr(Xtr, "toarray"):
        Xtr = Xtr.toarray()
    clf.fit(Xtr, y[tr])
    te = np.concatenate([te_known, novel])
    Xte = X[te]
    if clf_name in C.DENSE_REQUIRED and hasattr(Xte, "toarray"):
        Xte = Xte.toarray()
    proba = clf.predict_proba(Xte)
    score = 1.0 - proba.max(axis=1)
    is_novel = np.concatenate([np.zeros(len(te_known)), np.ones(len(novel))])
    try:
        return float(roc_auc_score(is_novel, score)), len(novel)
    except Exception:
        return float("nan"), len(novel)


def _checkpoint(rows, novel_rows, out):
    """Flush whatever has been computed so far.

    This script used to write its CSVs only at the very end, so a job killed on
    its walltime lost every result it had already computed -- which is exactly
    what happened to m2v_similarity_splits on 2026-08-23 after nine hours of work. Checkpoint
    after each block instead; the cost is milliseconds.
    """
    if rows:
        pd.DataFrame(rows).to_csv(out, index=False)
    if novel_rows:
        pd.DataFrame(novel_rows).to_csv(
            str(out).replace(".csv", "_novelty.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--collision", type=float, default=0.06)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpora", nargs="+", default=["dedup", "raw"])
    ap.add_argument("--identities", type=float, nargs="+",
                    default=[1.0, 0.999, 0.99, 0.97, 0.95])
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--classifiers", nargs="+", default=["LR", "RF", "DT"])
    ap.add_argument("--out", default="results/similarity_splits_splits.csv")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    rows, novel_rows = [], []
    for corpus in args.corpora:
        S, y, names, info = D.get_corpus(corpus)
        n_classes = len(names)
        per_seq, vocab = C.corpus_kmer_counts(S, args.k)
        keys = sorted(vocab)
        m = C.m_for_collision(keys, args.collision, seed=args.seed)
        emb = {"Murmur2Vec": C.murmur2vec(per_seq, m, signed=False, seed=args.seed),
               "Spike2Vec": C.full_spectrum(per_seq, vocab)[0]}
        print(f"\n===== {corpus} n={len(S)} m={m} =====", flush=True)

        # --- protocol 0: random stratified baseline ------------------------
        base = C.stratified_splits(y, n_splits=args.splits, test_size=0.30,
                                   seed=args.seed, dataset=corpus).splits
        for name, X in emb.items():
            rows += evaluate(X, y, base, n_classes, args.classifiers, args.seed,
                             corpus=corpus, embedding=name, protocol="stratified",
                             identity=np.nan)

        # --- protocol 1: identity-aware group splits -----------------------
        M = seq_matrix(S)
        for thr in args.identities:
            t0 = time.perf_counter()
            comp = cluster_by_identity(M, thr)
            ncl = len(np.unique(comp))
            gs = C.grouped_splits(y, comp, n_splits=args.splits, test_size=0.30,
                                  seed=args.seed, dataset=corpus,
                                  kind=f"identity_{thr}").splits
            _checkpoint(rows, novel_rows, out)
            print(f"  [cluster] identity>={thr:.3f}: {ncl} clusters "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
            for name, X in emb.items():
                got = evaluate(X, y, gs, n_classes, args.classifiers, args.seed,
                               corpus=corpus, embedding=name,
                               protocol="identity_group", identity=thr,
                               n_clusters=ncl)
                rows += got
                if got:
                    print(f"    {name:<11} acc={np.mean([g['accuracy'] for g in got]):.4f} "
                          f"f1m={np.mean([g['f1_macro'] for g in got]):.4f}", flush=True)

        # --- protocol 2: few-shot rare lineages ----------------------------
        counts = np.bincount(y, minlength=n_classes)
        rare = np.argsort(counts)[:3]
        for cls in rare:
            for n_shot in args.shots:
                if counts[cls] <= n_shot:
                    continue
                fs = C.few_shot_splits(y, int(cls), n_shot, n_repeats=5,
                                       seed=args.seed, dataset=corpus).splits
                for name, X in emb.items():
                    rows += evaluate(X, y, fs, n_classes, args.classifiers,
                                     args.seed, corpus=corpus, embedding=name,
                                     protocol=f"few_shot", identity=np.nan,
                                     target_class=names[cls], n_shots=n_shot,
                                     class_support=int(counts[cls]))
        print(f"  [few-shot] done for {[names[c] for c in rare]}", flush=True)
        _checkpoint(rows, novel_rows, out)

        # --- protocol 3: novel-lineage detection ---------------------------
        for cls in rare:
            for name, X in emb.items():
                auroc, nnov = novelty_auroc(X, y, int(cls), n_classes, "RF", args.seed)
                novel_rows.append(dict(corpus=corpus, embedding=name,
                                       held_lineage=names[cls], n_novel=nnov,
                                       detection_auroc=auroc))
                _checkpoint(rows, novel_rows, out)
            print(f"  [novelty] {name:<11} hold out {names[cls]:<12} "
                      f"AUROC={auroc:.4f} (n={nnov})", flush=True)

    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(novel_rows).to_csv(str(out).replace(".csv", "_novelty.csv"), index=False)
    df = pd.DataFrame(rows)
    print("\n=== accuracy / macro-F1 by protocol ===", flush=True)
    print(df.groupby(["corpus", "protocol", "identity", "embedding"])
          [["accuracy", "f1_macro"]].mean().round(4).to_string(), flush=True)
    print(f"\n[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
