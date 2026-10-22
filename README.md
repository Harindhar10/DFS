# DFS

QLoRA fine-tuning of `Qwen/Qwen3.8-27B` with LLaMA-Factory. Run everything from the repo root.

| Folder | Contents |
|---|---|
| `finetune/` | Shared: environment setup (`setup.sh`, `setup_fast.sh`) and `profile_train.py` |
| `retrosynthesis/` | USPTO-50k one-step retrosynthesis: zero-shot benchmark notebook, SFT data prep, train/predict configs, scorer |
| `moleculenet/` | MoleculeNet property prediction: DeepChem scaffold splits (`datasets/`), SFT data prep, one adapter per dataset, scorer |

```bash
bash finetune/setup.sh                                   # once, on the GPU box

bash retrosynthesis/train_eval.sh                        # USPTO-50k

python moleculenet/prepare_data.py --datasets tox21      # (re)build a DeepChem split
DATASETS="bbbp delaney" bash moleculenet/train_eval_moleculenet.sh   # omit DATASETS for all 11
```

Adapters and predictions go to `saves/qwen3.8-27b/qlora/{uspto50k,molnet}/`.
