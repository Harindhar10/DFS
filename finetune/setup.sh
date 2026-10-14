#!/usr/bin/env bash
# One-time setup on the GPU box. Run from the DFS repo root: bash finetune/setup.sh
set -euo pipefail

ENV_DIR="$HOME/envs/finetune"
MODEL_ID="Qwen/Qwen3.8-27B"

[ -d LLaMA-Factory ] || git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

python3 -m venv /venv/llama

uv pip install --python /venv/llama/bin/python --upgrade pip

uv pip install --python /venv/llama/bin/python \
-e "./LLaMA-Factory" bitsandbytes deepchem rdkit wandb
# LLaMA-Factory rejects transformers outside this range at import time.
uv pip install --python /venv/llama/bin/python \
"transformers>=4.55.0,<=5.8.0,!=4.57.0,!=5.6.0"
# optional (H100): uv pip install flash-attn --no-build-isolation

# Fail now, not hours later, if this transformers version can't read the model's config.
python -c "from transformers import AutoConfig; print('model_type:', AutoConfig.from_pretrained('$MODEL_ID').model_type)"

