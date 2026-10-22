"""Score a MoleculeNet QLoRA adapter (or the untuned base model) on its test split.

Usage (from the DFS repo root, on the GPU box):
    python moleculenet/score_moleculenet.py --dataset bbbp               # fine-tuned adapter
    python moleculenet/score_moleculenet.py --dataset bbbp --base_only   # zero-shot baseline

Classification: P(Yes) = softmax over the Yes/No logits at the first answer token, so ROC-AUC is
computed from a continuous score (generated Yes/No text would only give 0/1 points).
Regression: greedy-decode a few tokens and parse the first number; unparseable answers fall back
to the train-split mean for that task and are counted.

The model, quantization, template and adapter are loaded through LLaMA-Factory itself, so the
prompt tokens are exactly what the adapter was trained on.
"""
import argparse, csv, json, math, os, re
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score
from tqdm.auto import tqdm

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer

MODEL_ID = "Qwen/Qwen3.8-27B"
TEMPLATE = "qwen3_5_nothink"     # must match qwen_qlora_moleculenet.yaml
SAVE_ROOT = "saves/qwen3.8-27b/qlora/molnet"
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def load(adapter):
    model_args, data_args, finetuning_args, _ = get_infer_args(dict(
        model_name_or_path=MODEL_ID,
        adapter_name_or_path=adapter,
        finetuning_type="lora",
        template=TEMPLATE,
        trust_remote_code=True,
        quantization_bit=4,
        quantization_method="bnb",
        double_quantization=True,
        infer_dtype="bfloat16",
    ))
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    model.eval()
    return tokenizer, template, model


def prompt_ids(tokenizer, template, rec):
    # Same message LLaMA-Factory's Alpaca converter builds: instruction + "\n" + input.
    messages = [{"role": "user", "content": rec["instruction"] + "\n" + rec["input"]},
                {"role": "assistant", "content": ""}]
    ids, _ = template.encode_oneturn(tokenizer, messages, system=rec["system"])
    return ids


@torch.inference_mode()
def generate(model, tokenizer, batch_ids, max_new_tokens):
    """Left-pad and run generate(); generate() derives position ids from the padding mask."""
    width = max(len(x) for x in batch_ids)
    pad = tokenizer.pad_token_id
    input_ids = torch.tensor([[pad] * (width - len(x)) + x for x in batch_ids], device=model.device)
    attn = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in batch_ids], device=model.device)
    return model.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                          do_sample=False, output_logits=True, return_dict_in_generate=True,
                          pad_token_id=pad), width


def single_token(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{word!r} is not a single token: {ids}")
    return ids[0]


def p_yes(model, tokenizer, template, recs, batch_size, desc=None):
    """P(Yes) per record: softmax over the Yes/No logits at the first answer token."""
    yes, no = single_token(tokenizer, "Yes"), single_token(tokenizer, "No")
    preds = []
    for i in tqdm(range(0, len(recs), batch_size), desc=desc):
        chunk = recs[i:i + batch_size]
        out, _ = generate(model, tokenizer, [prompt_ids(tokenizer, template, r) for r in chunk], 1)
        logits = out.logits[0].float()                       # (batch, vocab) at the first answer token
        preds.extend(torch.softmax(logits[:, [yes, no]], dim=-1)[:, 0].tolist())
    return preds


def mean_roc_auc(recs, preds):
    """Mean ROC-AUC over tasks -> (mean, {task: auc}, [tasks skipped for having one class])."""
    by_task = defaultdict(lambda: ([], []))
    for r, p in zip(recs, preds):
        by_task[r["task"]][0].append(r["label"])
        by_task[r["task"]][1].append(p)
    aucs, skipped = {}, []
    for t, (y, p) in by_task.items():
        if len(set(y)) < 2:
            skipped.append(t)        # AUC undefined when the split has one class
            continue
        aucs[t] = float(roc_auc_score(y, p))
    mean = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    return mean, aucs, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", default="moleculenet/data")
    ap.add_argument("--adapter", default=None, help=f"default: {SAVE_ROOT}/<dataset>")
    ap.add_argument("--base_only", action="store_true", help="score the untuned model (zero-shot)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--results_csv", default=f"{SAVE_ROOT}/results.csv")
    args = ap.parse_args()

    test = json.load(open(os.path.join(args.data_dir, f"{args.dataset}_test.json")))
    train = json.load(open(os.path.join(args.data_dir, f"{args.dataset}_train.json")))
    is_cls = test[0]["output"] in ("Yes", "No")

    adapter = None if args.base_only else (args.adapter or f"{SAVE_ROOT}/{args.dataset}")
    tokenizer, template, model = load(adapter)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    preds = []
    if is_cls:
        preds = p_yes(model, tokenizer, template, test, args.batch_size, desc=args.dataset)
    else:
        train_mean = defaultdict(list)
        for r in train:
            train_mean[r["task"]].append(r["label"])
        train_mean = {t: float(np.mean(v)) for t, v in train_mean.items()}
        raw, failed = [], 0
        for i in tqdm(range(0, len(test), args.batch_size), desc=args.dataset):
            chunk = test[i:i + args.batch_size]
            out, width = generate(model, tokenizer, [prompt_ids(tokenizer, template, r) for r in chunk], 16)
            texts = tokenizer.batch_decode(out.sequences[:, width:], skip_special_tokens=True)
            for r, text in zip(chunk, texts):
                m = NUMBER.search(text)
                val = float(m.group(0)) if m else math.nan
                if not math.isfinite(val):
                    failed += 1
                    val = train_mean[r["task"]]
                raw.append(text)
                preds.append(val)

    metrics = {"dataset": args.dataset, "model": "base" if args.base_only else adapter, "n": len(test)}
    if is_cls:
        score, aucs, skipped = mean_roc_auc(test, preds)
        y_all = np.array([r["label"] for r in test])
        metrics.update(
            metric="roc_auc",
            score=score,
            accuracy=float(np.mean((np.array(preds) >= 0.5) == (y_all >= 0.5))),
            per_task=aucs,
            skipped_single_class=skipped,
        )
    else:
        y, p = np.array([r["label"] for r in test]), np.array(preds)
        metrics.update(
            metric="rmse",
            score=float(np.sqrt(np.mean((y - p) ** 2))),
            mae=float(mean_absolute_error(y, p)),
            parse_fail_pct=100 * failed / len(test),
        )
        if args.dataset == "freesolv":
            metrics["note"] = "freesolv labels are in the DeepChem split's units, not raw kcal/mol"

    out_dir = f"{SAVE_ROOT}/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    tag = "base" if args.base_only else "adapter"
    with open(f"{out_dir}/test_metrics_{tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{out_dir}/test_predictions_{tag}.jsonl", "w") as f:
        for k, (r, p) in enumerate(zip(test, preds)):
            rec = {"smiles": r["input"], "task": r["task"], "label": r["label"], "pred": p}
            if not is_cls:
                rec["raw"] = raw[k]
            f.write(json.dumps(rec) + "\n")

    new = not os.path.exists(args.results_csv)
    with open(args.results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["dataset", "model", "metric", "score", "n"])
        w.writerow([args.dataset, tag, metrics["metric"], f"{metrics['score']:.4f}", len(test)])

    print(json.dumps({k: v for k, v in metrics.items() if k != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
