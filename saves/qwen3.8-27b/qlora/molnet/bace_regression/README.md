---
library_name: peft
license: other
base_model: Qwen/Qwen3.8-27B
tags:
- base_model:adapter:Qwen/Qwen3.8-27B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: bace_regression
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bace_regression

This model is a fine-tuned version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on the bace_regression_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9150

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.03
- num_epochs: 2

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.9745        | 1.0   | 38   | 0.9408          |
| 0.8948        | 2.0   | 76   | 0.9150          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.8.0
- Pytorch 2.10.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2