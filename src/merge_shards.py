"""Merge sharded head-to-head runs into one table.

A long run of N splits can be sharded into several shorter jobs, which schedule
far more easily on a shared machine. Each shard numbers its splits from zero, so
this renumbers them to shard * splits_per_shard + split before concatenating;
without that, the paired tests downstream would treat different splits as the
same observation.

Note what the shards vary. The seed drives the hash seed as well as the split
RNG, so the merged replicates vary both the train/test partition and the hash --
a stronger test than fixing the hash and resampling splits, since the reported
variance then includes hash-seed variance. The full spectrum baseline is
unaffected by the hash, and both embeddings see identical splits within a shard,
so the pairing stays valid.

    python src/merge_shards.py results/head_to_head/audit20.csv
"""
import argparse
import glob as globmod
import re
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Merge sharded audit runs into one table, renumbering splits.")
    ap.add_argument("--shards", default="results/head_to_head_audit20_s*.csv",
                    help="glob for the shard CSVs; shard index is read from the "
                         "_s<N> suffix")
    ap.add_argument("--out", default="results/head_to_head_audit20.csv",
                    help="merged output path")
    ap.add_argument("--splits-per-shard", type=int, default=5,
                    help="splits each shard ran; used to offset the split index")
    args = ap.parse_args()

    cand = globmod.glob(args.shards)
    files = sorted([f for f in cand if re.search(r"_s\d+\.csv$", f)],
                   key=lambda s: int(re.search(r"_s(\d+)\.csv$", s).group(1)))
    if not files:
        sys.exit(f"no shard files matched {args.shards!r} "
                 f"(expected names ending _s0.csv, _s1.csv, ...)")

    frames = []
    for f in files:
        s = int(re.search(r"_s(\d+)\.csv$", f).group(1))
        d = pd.read_csv(f)
        d["shard"] = s
        d["split"] = s * args.splits_per_shard + d["split"]
        frames.append(d)
        print(f"{f}: {len(d)} rows, splits {sorted(d.split.unique())}")

    D = pd.concat(frames, ignore_index=True)
    D.to_csv(args.out, index=False)
    n = D.groupby(["corpus", "embedding", "classifier"]).size()
    print(f"\nmerged {len(D)} rows -> {args.out}")
    print(f"splits per cell: min {n.min()} max {n.max()} "
          f"(want {len(files) * args.splits_per_shard})")


if __name__ == "__main__":
    main()
