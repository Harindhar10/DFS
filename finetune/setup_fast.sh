set -euo pipefail

ENV_DIR="$HOME/envs/finetune"
MODEL_ID="Qwen/Qwen3.8-27B"

[ -d LLaMA-Factory ] || git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

uv pip install -e "./LLaMA-Factory" deepchem rdkit wandb
# LLaMA-Factory rejects transformers outside this range at import time.

python -c "from transformers import AutoConfig; print('model_type:', AutoConfig.from_pretrained('$MODEL_ID').model_type)"
