"""Paired significance and equivalence testing over a head-to-head table.

For every pair of (embedding, classifier) cells it reports a Nadeau-Bengio
corrected paired t-test, the naive uncorrected p-value alongside it, a
Holm-Bonferroni family-wise correction, and a TOST equivalence test at a
pre-registered margin. Each comparison ends in one of three verdicts:

    different            the corrected test rejects equality
    equivalent (TOST)    the difference is inside the margin with 90% confidence
    inconclusive         neither -- the study cannot distinguish the two and
                         cannot establish that they are the same

The third verdict is the point of the script. Reporting a large p-value as if it
showed equivalence is a common error; with few splits a design can easily produce
a large p-value next to a large effect size, which is underpowering rather than a
tie. Margins default to 0.01 accuracy and 0.02 macro-F1 and should be fixed
before looking at the data.

    python src/stats.py --audit results/head_to_head/audit20.csv \
        --out results/head_to_head/stats20.csv
"""
from __future__ import annotations

import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C

MARGINS = {"accuracy": 0.01, "f1_macro": 0.02}


def paired_frame(df, key_cols, value_col, split_col="split"):
    """Wide table: one row per split, one column per method."""
    d = df.copy()
    d["_key"] = d[key_cols].astype(str).agg(" | ".join, axis=1)
    return d.pivot_table(index=split_col, columns="_key", values=value_col)


def compare_all(wide, metric, test_frac=0.30):
    rows = []
    names = list(wide.columns)
    for a, b in itertools.combinations(names, 2):
        pair = wide[[a, b]].dropna()
        if len(pair) < 3:
            continue
        t_corr = C.paired_t(pair[a].values, pair[b].values, name=f"{a} vs {b}",
                            corrected_for_resampling=True, test_frac=test_frac)
        t_naive = C.paired_t(pair[a].values, pair[b].values,
                             corrected_for_resampling=False, test_frac=test_frac)
        eq = C.tost(pair[a].values, pair[b].values, MARGINS[metric],
                    test_frac=test_frac)
        r = t_corr.as_dict()
        r.update(metric=metric, method_a=a, method_b=b,
                 mean_a=float(pair[a].mean()), sd_a=float(pair[a].std(ddof=1)),
                 mean_b=float(pair[b].mean()), sd_b=float(pair[b].std(ddof=1)),
                 p_naive=t_naive.p, t_naive=t_naive.t,
                 tost_margin=eq["margin"], p_tost=eq["p_tost"],
                 equivalent=eq["equivalent"],
                 ci90_low=eq["ci90_low"], ci90_high=eq["ci90_high"])
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    corrected, reject = C.holm_bonferroni(out["p"].values)
    out["p_corrected"] = corrected
    out["significant_holm"] = reject
    out["correction"] = "Holm-Bonferroni"
    # A comparison is only reported as a TIE if TOST says so AND the difference
    # test fails to reject. Reporting either one alone is the error to avoid.
    out["verdict"] = np.where(
        out["significant_holm"], "different",
        np.where(out["equivalent"], "equivalent (TOST)", "inconclusive"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="results/head_to_head_audit.csv")
    ap.add_argument("--corpus", default="dedup")
    ap.add_argument("--metrics", nargs="+", default=["accuracy", "f1_macro"])
    ap.add_argument("--out", default="results/stats_stats.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.audit)
    df = df[df["corpus"] == args.corpus]
    print(f"[data] {len(df)} rows, corpus={args.corpus}, "
          f"{df['split'].nunique()} splits", flush=True)

    allrows = []
    for metric in args.metrics:
        wide = paired_frame(df, ["embedding", "classifier"], metric)
        res = compare_all(wide, metric)
        if res.empty:
            continue
        allrows.append(res)
        print(f"\n=== {metric} (margin {MARGINS[metric]}) ===", flush=True)
        show = res[["method_a", "method_b", "mean_a", "mean_b", "mean_diff",
                    "t", "df", "p", "p_corrected", "cohens_d",
                    "ci_low", "ci_high", "p_tost", "verdict"]]
        print(show.round(4).to_string(index=False), flush=True)

    if allrows:
        out = pd.concat(allrows, ignore_index=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        n_eq = int((out["verdict"] == "equivalent (TOST)").sum())
        n_diff = int((out["verdict"] == "different").sum())
        n_inc = int((out["verdict"] == "inconclusive").sum())
        print(f"\n[summary] {n_diff} different, {n_eq} equivalent, "
              f"{n_inc} inconclusive (of {len(out)} comparisons)", flush=True)
        print("NOTE: 'inconclusive' is the honest label for a comparison that "
              "neither rejects equality nor establishes equivalence. The "
              "submitted paper reported these as ties.", flush=True)
        print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
