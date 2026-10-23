#!/usr/bin/env bash
# One QLoRA adapter per MoleculeNet dataset: train, then score on the scaffold test split.
# Run from the DFS repo root after setup.sh:
#   bash moleculenet/train_eval_moleculenet.sh                     # all datasets
#   DATASETS="bbbp delaney" bash moleculenet/train_eval_moleculenet.sh
#   BASELINE=1 bash moleculenet/train_eval_moleculenet.sh          # also score the untuned model

wandb login        # or export WANDB_API_KEY=... beforehand

set -euo pipefail

ALL="delaney freesolv lipo bace_regression clearance"

#All
#"bbbp bace_classification clintox sider tox21 hiv delaney freesolv lip bace_regression clearance"

#small
#"bbbp bace_classification clintox delaney freesolv lipo bace_regression clearance"
#large
# sider hiv tox21

DATASETS=${DATASETS:-$ALL}

python moleculenet/prepare_moleculenet.py --datasets "$(echo $DATASETS | tr ' ' ',')"

export WANDB_PROJECT=molnet-qlora
export WANDB_LOG_MODEL=false       # don't upload 27B checkpoints

SAVE_ROOT=saves/qwen3.8-27b/qlora/molnet

for d in $DATASETS; do
    case $d in
        hiv|tox21) EPOCHS=1 ;;   # >=30k examples each
        *)               EPOCHS=2 ;;
    esac
    echo "== $d ($EPOCHS epochs) =="
    # train.py = llamafactory-cli train + val ROC-AUC (cls) and trainable-param logging
    python moleculenet/train.py moleculenet/qwen_qlora_moleculenet.yaml \
        dataset="${d}_train" eval_dataset="${d}_val" \
        output_dir="$SAVE_ROOT/$d" run_name="qwen3.8-27b-qlora-$d" \
        num_train_epochs="$EPOCHS"
    python moleculenet/score_moleculenet.py --dataset "$d"
    if [ "${BASELINE:-0}" = 1 ]; then
        python moleculenet/score_moleculenet.py --dataset "$d" --base_only
    fi
done

column -s, -t "$SAVE_ROOT/results.csv"
