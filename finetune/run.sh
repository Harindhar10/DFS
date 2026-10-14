#!/usr/bin/env bash
# Run from the DFS repo root on the GPU box, step by step.
set -euo pipefail

# env
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e "./LLaMA-Factory[torch,metrics]" bitsandbytes deepchem rdkit wandb
uv pip install -U transformers       # a brand-new Qwen architecture usually needs the latest transformers
# optional: uv pip install flash-attn --no-build-isolation   (H100)

# wandb: GPU utilization, memory, power and temperature are logged automatically under the run's "System" tab
wandb login                          # or: export WANDB_API_KEY=...
export WANDB_PROJECT=dfs-retro-qlora
export WANDB_LOG_MODEL=false         # don't upload 27B checkpoints

# data (from the DFS repo root, with DeepRetro/ cloned beside it)
python finetune/prepare_uspto50k.py

# train
llamafactory-cli train finetune/qwen_qlora_uspto50k.yaml

# predict + score
llamafactory-cli train finetune/qwen_qlora_predict.yaml
python finetune/score_predictions.py saves/qwen3.8-27b/qlora/uspto50k/predict_test/generated_predictions.jsonl
# for bench250, set eval_dataset to uspto50k_bench250 and output_dir to .../predict_bench250, then run again
