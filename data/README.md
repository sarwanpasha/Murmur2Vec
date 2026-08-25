# Data

No sequence data is committed to this repository. This file describes what the
scripts expect and how to build it.

## Expected layout

```
data/
  sequences.npy        object array of amino-acid strings
  labels.npy           object array of class labels, same length and order
```

`src/corpora.py` reads these and derives three protocols from them:

| name | what it is |
|---|---|
| `raw` | the corpus exactly as loaded |
| `dedup` | exact duplicates collapsed, majority label per distinct sequence |
| `dedup_strict` | as `dedup`, but distinct sequences carrying conflicting labels are dropped |

Use `dedup` as the primary protocol and report `raw` alongside it, so that any
memorisation effect is visible and attributed rather than hidden. `corpora.py`
prints the exact-duplicate fraction when it loads a corpus; if that number is
large, a random-split score is partly measuring the ability to recognise copies.

## Other corpora

`src/replicate.py` works on any FASTA and does not use the `.npy` layout:

```bash
python src/replicate.py --fasta data/mycorpus.fasta --name mycorpus \
    --out results/replication/mycorpus.csv
```

The class label is taken from the second whitespace-separated token of each
defline. If your deflines carry no label, pass a two-column `id<TAB>label` file
with `--labels`.

A sketch tested only on near-identical sequences is barely tested at all: when
almost every k-mer is shared between any two sequences, almost nothing depends on
how collisions are resolved. Replicating on a corpus of divergent sequences of
differing lengths is the informative check.

## Building an influenza protein-type corpus

`src/build_influenza_dataset.py` turns an NCBI Datasets virus package into a
balanced, deduplicated protein-type dataset. Fetch the package with the NCBI
`datasets` command-line tool, then:

```bash
python src/build_influenza_dataset.py \
    --faa <package>/ncbi_dataset/data/protein.faa \
    --out-dir data --max-per-class 1500 --min-per-class 200
```

It writes `flu_prottype_sequences.fasta` and `flu_prottype_labels.tsv`, which
`replicate.py` reads directly. Classes are the protein products -- hemagglutinin,
neuraminidase, matrix, nucleocapsid, the polymerase subunits, NS1, NEP -- taken
from the FASTA deflines, so no external metadata is needed.

Note that recent NCBI virus packages carry no serotype field, so an HA-subtype
dataset cannot be built from the data report alone; the script documents this.

## Protein family corpora

Any FASTA of family-labelled sequences works with `replicate.py`. Reviewed
entries from a protein sequence database, filtered to a handful of families with
a cap per family, give a corpus in the right shape.
