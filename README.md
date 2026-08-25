# Murmur2Vec

Fixed-size embeddings of biological sequences by feature hashing of the k-mer
spectrum, plus the experiments used to test when that works and why.

A sequence is turned into counts over its k-mers; each k-mer is hashed to one of
`m` buckets and its count added there. The result is an `m`-dimensional vector
whose size you choose in advance, with no vocabulary pass, no dictionary to store
or ship, `O(1)` work per k-mer, and the ability to embed a k-mer that was never
seen at fit time. Buckets collide, and the question this repository is built to
answer is what those collisions cost.

## What is here

```
src/         library and one script per experiment
results/     the CSV outputs of those experiments, grouped by topic
scripts/     driver that runs the pipeline end to end
data/        how to obtain the corpora (no sequence data is committed)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. The protein-language-model scripts additionally need `torch`,
`transformers` and `peft`, and a GPU is strongly recommended for those two; every
other script runs on CPU.

## Quickstart

```python
import sys; sys.path.insert(0, "src")
import core as C

seqs = ["MFVFLVLLPLVSSQCVNLT", "MFVFLVLLPLVSSQCVNLS"]
counts, vocab = C.corpus_kmer_counts(seqs, k=3)

# choose m for a target collision rate, then embed
m = C.m_for_collision(sorted(vocab), 0.06)
X = C.murmur2vec(counts, m)          # (n_sequences, m)
```

`C.m_heuristic(U, c)` gives the same `m` from a closed form, `ceil((U-1)/(2c))`,
without the search: accurate to a few percent for `c <= 10%`, drifting to about
1.4x at `c = 40%`.

## The scripts

Every script has a module docstring explaining what it measures and why, and
`--help` for its arguments. All of them import `src/core.py`, so the embedding,
the metrics and the statistics are computed one way throughout.

| script | question it answers |
|---|---|
| `core.py` | library: hashing, sketches, baselines, splits, metrics, statistics |
| `corpora.py` | corpus construction and the exact-duplicate audit |
| `head_to_head.py` | sketch vs full k-mer spectrum, all AUC definitions |
| `stats.py` | corrected paired t + TOST equivalence over a head-to-head table |
| `auc_audit.py` | why two AUC numbers disagree on an imbalanced problem |
| `calibration_grid.py` | sweep k and m: what does the collision rate cost? |
| `similarity_splits.py` | identity-aware splits, few-shot, novel-class detection |
| `collision_mechanism.py` | adversarial collisions and a random-projection control |
| `plm_frozen.py` | frozen protein language model baselines |
| `plm_lora.py` | LoRA fine-tuning of a protein language model |
| `replicate.py` | re-run the comparison on any other corpus |
| `hash_ablation.py` | is it this hash function, or feature hashing in general? |
| `build_influenza_dataset.py` | build an influenza protein-type dataset from NCBI |
| `merge_shards.py` | merge sharded runs into one table |

## Reproducing

```bash
bash scripts/run_all.sh          # everything except the language-model arms
bash scripts/run_all.sh --plm    # include those (needs a GPU)
```

Or run one at a time:

```bash
python src/head_to_head.py --splits 20 --corpora dedup raw \
    --out results/head_to_head/audit20.csv
python src/stats.py --audit results/head_to_head/audit20.csv \
    --out results/head_to_head/stats20.csv
```

Long runs shard cleanly. `--seed` drives the hash seed as well as the split RNG,
so four shards of five splits give twenty replicates that vary both the partition
and the hash; `merge_shards.py` renumbers and concatenates them.

## Methodological notes

These are the things that changed conclusions during development, collected here
because they are easy to get wrong and cheap to get right.

**Deduplicate before you split.** A sequence corpus can be mostly exact
duplicates. Under a random split those duplicates put copies of training rows in
the test set, and part of the reported score is then memorisation. `corpora.py`
reports the duplicate fraction so the size of the effect can be stated rather
than assumed.

**Random splits are the easy case.** Grouping by sequence identity so that
related sequences cannot straddle the split is harder and more informative.
Accuracy is not monotone across identity thresholds, because grouping changes the
test-set class mix; macro-F1 is, and is the metric to report under those
protocols.

**Say which AUC you mean.** On an imbalanced problem, one-vs-rest weighted AUC
can sit far above one-vs-rest macro. `head_to_head.py` emits every definition.

**A large p-value is not equivalence.** Repeated random splits reuse data, so the
naive paired t-test is anti-conservative; `core.paired_t` applies the
Nadeau-Bengio correction. And to claim two methods are the same you need a TOST
equivalence test at a margin fixed in advance, which `stats.py` reports next to
the difference test. Comparisons that are neither different nor equivalent are
labelled `inconclusive` rather than quietly called ties.

**Compression is a claim you have to check.** At small k the sketch at a low
collision rate can be considerably larger than simply storing the observed k-mer
spectrum. Compare `m` against the number of k-mers you actually see, not against
the alphabet size raised to k.

**Give every parallel job its own output path.** Scripts here append rows as they
compute them, so an interrupted run keeps its work; two runs sharing one path
will overwrite each other.

## Results

`results/` holds the CSV outputs, grouped by topic: `head_to_head/`,
`calibration/`, `mechanism/`, `similarity/`, `plm/`, `replication/`,
`hash_ablation/`. They are the raw per-split rows, not summary tables, so any
aggregation or test can be recomputed from them. See `results/README.md`.

## License

MIT. See `LICENSE`.
