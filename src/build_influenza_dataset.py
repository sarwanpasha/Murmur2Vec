"""Build an influenza protein-type dataset from an NCBI Datasets package.

Classifies influenza A proteins by product -- hemagglutinin, neuraminidase,
matrix, nucleocapsid, the polymerase subunits, NS1, NEP. The product name is on
every FASTA defline, so no external metadata is required.

This is deliberately not an HA-subtype dataset. Recent NCBI virus packages carry
no serotype field in the data report, and a literal scan finds a subtype string in
well under one percent of records, so a subtype benchmark would be two classes
drawn from a tiny slice of the corpus. Protein type is also the better test for a
k-mer sketch: the classes are different proteins with different folds and lengths
spanning roughly an order of magnitude, rather than variants of one protein.

Mature-peptide fragments (signal peptides, HA1/HA2, PA-X, PB1-F2) are dropped so
every class is a whole chain, and duplicates are removed BEFORE balancing, so the
balanced set is not quietly padded with copies of the same sequence.

    python src/build_influenza_dataset.py --faa protein.faa --out-dir data \
        --max-per-class 1500 --min-per-class 200
"""
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# canonical influenza A products -> class label
CANON = {
    "hemagglutinin": "HA",
    "neuraminidase": "NA",
    "matrix protein 1": "M1",
    "matrix protein 2": "M2",
    "nucleocapsid protein": "NP",
    "nonstructural protein 1": "NS1",
    "nuclear export protein": "NEP",
    "polymerase PA": "PA",
    "polymerase PB1": "PB1",
    "polymerase PB2": "PB2",
}
# fragments of a larger chain -- excluded so classes are whole proteins
FRAGMENTS = ("sig_peptide", "HA1", "HA2", "PA-X", "PB1-F2", "mature peptide")


def parse(faa):
    """Yield (accession, product, sequence)."""
    acc = prod = None
    buf = []
    with open(faa) as f:
        for line in f:
            if line.startswith(">"):
                if acc is not None:
                    yield acc, prod, "".join(buf)
                buf = []
                head = line[1:].rstrip()
                acc = head.split(":", 1)[0].split()[0]
                # product = text between the accession token and the first [tag=
                m = re.match(r"\S+\s+(.*?)\s*\[", head)
                prod = m.group(1).strip() if m else ""
            else:
                buf.append(line.strip())
    if acc is not None:
        yield acc, prod, "".join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faa", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-per-class", type=int, default=1500)
    ap.add_argument("--min-per-class", type=int, default=200)
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    by_class = defaultdict(list)
    raw_products = Counter()
    n_seen = n_frag = 0

    for acc, prod, seq in parse(args.faa):
        n_seen += 1
        raw_products[prod] += 1
        if any(fr in prod for fr in FRAGMENTS):
            n_frag += 1
            continue
        label = CANON.get(prod)
        if label is None:
            continue
        if not (args.min_len <= len(seq) <= args.max_len):
            continue
        if set(seq) - set("ACDEFGHIKLMNPQRSTVWYXBZJUO"):
            continue
        by_class[label].append((acc, seq))

    print(f"[scan] {n_seen:,} FASTA records, {n_frag:,} dropped as fragments")
    print("[scan] most common raw products:")
    for p, c in raw_products.most_common(14):
        print(f"    {c:>8,}  {p[:70]}")

    keep = {k: v for k, v in by_class.items() if len(v) >= args.min_per_class}
    print(f"\n[classes] {len(keep)} of {len(by_class)} pass min-per-class="
          f"{args.min_per_class}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fa = out / "flu_prottype_sequences.fasta"
    tsv = out / "flu_prottype_labels.tsv"
    total = 0
    with open(fa, "w") as ff, open(tsv, "w") as tf:
        tf.write("id\tlabel\n")
        for label in sorted(keep, key=lambda k: -len(keep[k])):
            items = keep[label]
            # dedupe identical sequences BEFORE subsampling, so the balanced set
            # is not silently padded with copies (the exact failure mode this
            # audit found in a corpus of near-duplicate sequences)
            seen, uniq = set(), []
            for acc, seq in items:
                if seq not in seen:
                    seen.add(seq)
                    uniq.append((acc, seq))
                if len(uniq) >= args.max_per_class * 4:
                    break
            if len(uniq) > args.max_per_class:
                idx = rng.choice(len(uniq), size=args.max_per_class, replace=False)
                uniq = [uniq[i] for i in sorted(idx)]
            lens = [len(s) for _, s in uniq]
            print(f"    {label:<5} n={len(uniq):>5}  raw={len(items):>7,}  "
                  f"unique={len(seen):>6,}  len {min(lens)}-{max(lens)} "
                  f"(median {int(np.median(lens))})")
            for acc, seq in uniq:
                ff.write(f">{acc} {label}\n{seq}\n")
                tf.write(f"{acc}\t{label}\n")
                total += 1
    print(f"\n[done] {total:,} sequences across {len(keep)} classes")
    print(f"       {fa}\n       {tsv}")


if __name__ == "__main__":
    main()
