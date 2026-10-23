"""`llamafactory-cli train` plus validation ROC-AUC and trainable-parameter logging.

Usage (from the DFS repo root, on the GPU box) -- same yaml and key=value overrides as llamafactory-cli:
    python moleculenet/train.py moleculenet/qwen_qlora_moleculenet.yaml dataset=bbbp_train eval_dataset=bbbp_val ...

- ParamCountCallback: at train start, prints trainable / total / trainable% parameters and writes them
  to the wandb summary + config (params/trainable, params/total, params/trainable_pct). Counts come from
  LLaMA-Factory's count_parameters, which unpacks bnb 4-bit weights (total ~27.8B, not the packed size).
- ValAucCallback (classification datasets only): after every evaluation, scores <dataset>_val.json with
  the current adapter using the same P(Yes) code as score_moleculenet.py, logs val/roc_auc (and per task)
  to wandb, and puts eval_roc_auc into the eval metrics, so metric_for_best_model=eval_roc_auc picks the
  checkpoint with the best validation AUC. Regression datasets keep eval_loss for checkpoint selection.
- In wandb, all Trainer validation metrics appear as val/* (val/loss, ...). The run id is saved to
  <output_dir>/wandb_run_id so score_moleculenet.py can log test metrics into the same run.
"""
import json, os, sys

import torch
import transformers.integrations.integration_utils as hf_integrations
from transformers import TrainerCallback

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.misc import count_parameters
from llamafactory.hparams import DataArguments, read_args
from llamafactory.train.tuner import run_exp

from score_moleculenet import mean_roc_auc, p_yes  # moleculenet/ is on sys.path as the script dir

# Show validation metrics as val/* in wandb. Only the wandb names change: the Trainer's own keys stay
# eval_* because metric_for_best_model lookup and LLaMA-Factory's loss plot depend on them.
# WandbCallback.on_log calls this module-level function at log time, so wrapping it is enough.
_rewrite_logs = hf_integrations.rewrite_logs
hf_integrations.rewrite_logs = lambda d: {
    ("val/" + k[len("eval/"):] if k.startswith("eval/") else k): v for k, v in _rewrite_logs(d).items()
}


def wandb_run(args):
    if "wandb" not in (args.report_to or []):
        return None
    import wandb  # WandbCallback runs before ours, so the run already exists
    return wandb.run


class ParamCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None or not state.is_world_process_zero:
            return
        trainable, total = count_parameters(model)
        stats = {"params/trainable": trainable, "params/total": total,
                 "params/trainable_pct": 100 * trainable / total}
        print(f"trainable params: {trainable:,} || all params: {total:,} || "
              f"trainable%: {stats['params/trainable_pct']:.4f}")
        run = wandb_run(args)
        if run is not None:
            run.summary.update(stats)
            run.config.update(stats, allow_val_change=True)
            # score_moleculenet.py reopens this run to log the test metrics next to training.
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "wandb_run_id"), "w") as f:
                f.write(run.id)


class ValAucCallback(TrainerCallback):
    def __init__(self, val_recs, template_name, batch_size):
        self.val_recs, self.template_name, self.batch_size = val_recs, template_name, batch_size
        self.template = None

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if self.template is None:
            self.template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template=self.template_name))

        was_training = model.training
        model.eval()
        preds = p_yes(model, tokenizer, self.template, self.val_recs, self.batch_size, desc="val roc-auc")
        if was_training:
            model.train()

        mean, aucs, skipped = mean_roc_auc(self.val_recs, preds)
        if metrics is not None:
            metrics["eval_roc_auc"] = mean  # same dict the Trainer reads for metric_for_best_model
        print(f"step {state.global_step}: val ROC-AUC = {mean:.4f}"
              + (f" (skipped single-class tasks: {skipped})" if skipped else ""))
        run = wandb_run(args)
        if run is not None and state.is_world_process_zero:
            run.log({"val/roc_auc": mean, **{f"val/roc_auc/{t}": a for t, a in aucs.items()},
                     "train/global_step": state.global_step})
        torch.cuda.empty_cache()


args = read_args()  # parses sys.argv: yaml in argv[1] plus key=value overrides -> dict
if not isinstance(args, dict):
    sys.exit("usage: python moleculenet/train.py <config.yaml> [key=value ...]")

callbacks = [ParamCountCallback()]
eval_name = args.get("eval_dataset")
if eval_name:
    val_path = os.path.join(args.get("dataset_dir", "moleculenet/data"), f"{eval_name}.json")
    val_recs = json.load(open(val_path))
    if val_recs and val_recs[0]["output"] in ("Yes", "No"):
        callbacks.append(ValAucCallback(val_recs, args["template"],
                                        int(args.get("per_device_eval_batch_size", 32))))
        if args.get("metric_for_best_model", "eval_loss") == "eval_loss":  # keep an explicit override
            args.update(metric_for_best_model="eval_roc_auc", greater_is_better=True)
        print(f"val ROC-AUC on {val_path} ({len(val_recs)} examples); "
              f"best checkpoint by {args['metric_for_best_model']}")

run_exp(args, callbacks=callbacks)
