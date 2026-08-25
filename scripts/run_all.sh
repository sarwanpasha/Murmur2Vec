#!/usr/bin/env bash
# Reproduce the experiments end to end.
#
#   bash scripts/run_all.sh          core experiments (CPU)
#   bash scripts/run_all.sh --plm    also the protein-language-model arms (GPU)
#
# Run from the repository root. Every step appends rows as it computes them, so
# an interrupted run keeps its work and can be restarted step by step.
set -euo pipefail

RUN_PLM=0
[[ "${1:-}" == "--plm" ]] && RUN_PLM=1

PY=${PYTHON:-python}
SPLITS=${SPLITS:-10}
SEED=${SEED:-0}
K=${K:-3}
COLLISION=${COLLISION:-0.06}

mkdir -p results/{head_to_head,calibration,mechanism,similarity,plm,replication,hash_ablation}

step() { echo; echo "=== $* ==="; }

step "head-to-head: sketch vs full spectrum"
$PY src/head_to_head.py --k "$K" --collision "$COLLISION" \
    --splits "$SPLITS" --seed "$SEED" --corpora dedup raw \
    --out results/head_to_head/audit.csv

step "statistics: corrected paired t + TOST equivalence"
$PY src/stats.py --audit results/head_to_head/audit.csv \
    --out results/head_to_head/stats.csv

step "AUC definitions side by side"
$PY src/auc_audit.py --k "$K" --collision "$COLLISION" --splits 5 \
    --out results/head_to_head/auc_audit.csv

step "calibration: k x m sweep"
$PY src/calibration_grid.py --out results/calibration/grid.csv

step "mechanism: adversarial collisions and the projection control"
$PY src/collision_mechanism.py --out results/mechanism/mechanism.csv

step "similarity-aware splits, few-shot and novel-class detection"
$PY src/similarity_splits.py --corpora dedup --splits "$SPLITS" \
    --out results/similarity/splits.csv

step "hash-family ablation"
for fam in murmur xxhash fnv1a tabulation sha256; do
    $PY src/hash_ablation.py --family "$fam" --seeds 0 1 2 --splits 5 \
        --out "results/hash_ablation/${fam}.csv"
done

if [[ -f data/flu_prottype_sequences.fasta ]]; then
    step "replication on another corpus"
    $PY src/replicate.py --fasta data/flu_prottype_sequences.fasta \
        --name flu_prottype --splits "$SPLITS" \
        --out results/replication/flu.csv
else
    echo
    echo "[skip] replication: data/flu_prottype_sequences.fasta not found."
    echo "       See data/README.md to build it."
fi

if [[ $RUN_PLM -eq 1 ]]; then
    step "frozen protein language model baselines"
    $PY src/plm_frozen.py --out results/plm/frozen.csv

    step "LoRA fine-tuning, one file per split"
    for s in 0 1 2 3 4; do
        $PY src/plm_lora.py --config-id 1 --corpus dedup --split "$s" \
            --max-epochs 80 --patience 10 \
            --out "results/plm/lora_s${s}.csv"
    done
else
    echo
    echo "[skip] protein language model arms. Re-run with --plm (needs a GPU"
    echo "       and the optional dependencies in requirements.txt)."
fi

echo
echo "=== done. Outputs are in results/ ==="
