#!/usr/bin/env bash
# Train the QLoRA adapter, then score it on the held-out test split and the 250-molecule benchmark.
# Run from the DFS repo root after setup.sh: bash finetune/train_eval.sh
set -euo pipefail

source "$HOME/envs/finetune/bin/activate"

# wandb logs train/eval loss and lr; GPU util/memory/power show up in the run's "System" tab.
export WANDB_PROJECT=dfs-retro-qlora
export WANDB_LOG_MODEL=false       # don't upload 27B checkpoints

SAVE_DIR=saves/qwen3.8-27b/qlora/uspto50k

llamafactory-cli train finetune/qwen_qlora_uspto50k.yaml

for split in uspto50k_retro_test uspto50k_bench250; do
    llamafactory-cli train finetune/qwen_qlora_predict.yaml \
        eval_dataset="$split" output_dir="$SAVE_DIR/predict_$split"
    echo "== $split =="
    python finetune/score_predictions.py "$SAVE_DIR/predict_$split/generated_predictions.jsonl"
done
