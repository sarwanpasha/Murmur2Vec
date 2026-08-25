# Results

Raw per-split rows from the experiments, not summary tables, so that any
aggregation or statistical test can be recomputed from them.

| directory | produced by | contents |
|---|---|---|
| `head_to_head/` | `head_to_head.py`, `stats.py`, `auc_audit.py` | sketch vs full spectrum across classifiers and splits, all AUC definitions, and the paired-t + TOST tables computed from them |
| `calibration/` | `calibration_grid.py` | k x m sweep: collision rate, accuracy, macro-F1, peak memory, and the closed-form m against binary search |
| `mechanism/` | `collision_mechanism.py` | adversarial-collision intervention with frequency-matched controls, and the Gaussian random-projection control |
| `similarity/` | `similarity_splits.py` | identity-grouped splits at several thresholds, few-shot curves, novel-class detection AUROC |
| `plm/` | `plm_frozen.py`, `plm_lora.py` | frozen protein-language-model baselines across sizes, regimes and poolings; LoRA fine-tuning grid and multi-split runs |
| `replication/` | `replicate.py` | the same comparison on other corpora; `*_meta.csv` holds corpus facts and the m-calibration table |
| `hash_ablation/` | `hash_ablation.py` | five hash families at matched collision rate, signed and unsigned; `*_diag.csv` holds avalanche, bucket uniformity, pairwise independence and throughput |

Column names are shared across files wherever the quantity is the same, because
every script scores through `core.compute_metrics`. Files ending `_meta.csv` hold
corpus-level facts rather than per-split scores, `_diag.csv` holds hash
diagnostics rather than classification results, and `_summary.csv` holds a
group-by over the per-split rows in the file beside it.
