"""Head-to-head comparison: hashed sketch vs full k-mer spectrum.

Computes both embeddings through one code path and evaluates them on identical
splits with identical classifiers, emitting every AUC definition so the metric
choice is explicit rather than implicit. This is the table the statistics in
stats.py consume.

    python src/head_to_head.py --splits 20 --corpora dedup raw \
        --out results/head_to_head/audit20.csv

For many splits, shard by seed and merge with merge_shards.py: --seed changes
both the split RNG and the hash seed, so N shards of S splits give N*S replicates
that vary the partition AND the hash, which is the stronger design for a
randomised sketch.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C
import corpora as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--collision", type=float, default=0.06)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpora", nargs="+", default=["raw", "dedup"])
    ap.add_argument("--classifiers", nargs="+", default=list(C.CLASSIFIERS))
    ap.add_argument("--out", default="results/head_to_head_audit.csv")
    args = ap.parse_args()

    rows, meta = [], {}
    for corpus in args.corpora:
        S, y, names, info = D.get_corpus(corpus)
        meta[corpus] = info
        n_classes = len(names)
        print(f"\n===== corpus={corpus} n={len(S)} classes={n_classes} =====", flush=True)
        print(json.dumps(info), flush=True)

        per_seq, vocab = C.corpus_kmer_counts(S, args.k)
        keys = sorted(vocab)
        U = len(keys)
        m = C.m_for_collision(keys, args.collision, seed=args.seed)
        heur = C.m_heuristic(U, args.collision)
        print(f"[calib] U={U} m={m} (achieved {C.collision_pct(keys,m,seed=args.seed):.4f}) "
              f"heuristic={heur} ratio={heur/m:.3f}  21^k={21**args.k}", flush=True)
        meta[corpus].update(U=U, m=m, m_heuristic=heur, D_theory=21 ** args.k)

        emb = {"Murmur2Vec": C.murmur2vec(per_seq, m, signed=False, seed=args.seed),
               "Spike2Vec": C.full_spectrum(per_seq, vocab)[0]}

        splits = C.stratified_splits(y, n_splits=args.splits, test_size=0.30,
                                     seed=args.seed, dataset=corpus)
        leak = [D.exact_duplicate_leakage(S, s) for s in splits.splits]
        fl = float(np.mean([l["frac_exact_dup"] for l in leak]))
        print(f"[leak] mean exact-duplicate fraction in test: {fl:.1%}", flush=True)
        meta[corpus]["mean_test_exact_dup_frac"] = fl

        for name, X in emb.items():
            for clf in args.classifiers:
                t0 = time.perf_counter()
                got = []
                for si, (tr, te) in enumerate(splits.splits):
                    try:
                        r = C.fit_eval(clf, X[tr], X[te], y[tr], y[te], n_classes,
                                       seed=args.seed, all_auc_variants=True)
                    except Exception as e:
                        print(f"  ! {corpus}/{name}/{clf}/{si}: {e}", flush=True)
                        continue
                    r.update(corpus=corpus, embedding=name, split=si, m=m,
                             k=args.k, dim=X.shape[1])
                    rows.append(r); got.append(r)
                if got:
                    g = lambda f: float(np.nanmean([x.get(f, np.nan) for x in got]))
                    print(f"  {corpus:<6} {name:<11} {clf:<4} "
                          f"acc={g('accuracy'):.4f} balacc={g('balanced_accuracy'):.4f} "
                          f"f1m={g('f1_macro'):.4f} mcc={g('mcc'):.4f} | "
                          f"AUC macro={g('roc_auc_ovr_macro'):.4f} "
                          f"wtd={g('roc_auc_ovr_weighted'):.4f} "
                          f"ovo={g('roc_auc_ovo_macro'):.4f} "
                          f"hard-wtd={g('roc_auc_hardlabel_ovr_weighted'):.4f} "
                          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out, index=False)
    with open(str(out).replace(".csv", "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    keep = ["accuracy", "balanced_accuracy", "f1_macro", "mcc",
            "roc_auc_ovr_macro", "roc_auc_ovr_weighted", "roc_auc_ovo_macro",
            "roc_auc_hardlabel_ovr_weighted"]
    s = df.groupby(["corpus", "embedding", "classifier"])[keep].mean().round(4)
    print("\n=== mean over splits ===", flush=True)
    print(s.to_string(), flush=True)
    s.to_csv(str(out).replace(".csv", "_summary.csv"))
    print(f"\n[done] {len(df)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
