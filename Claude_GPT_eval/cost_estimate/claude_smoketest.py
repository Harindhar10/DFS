"""
Pilot run: Claude Opus 5.5 on a sample of MoleculeNet validation molecules.

Measures what the cost estimator can't know in advance:
  * thinking tokens per request (Opus 5.5 can't turn thinking off)
  * how often the max_tokens cap cuts off the answer
  * cache hit rate inside a batch
  * whether the JSON answers parse

Uses the VALIDATION split only, so the pilot never touches test data.

Usage:
    pip install anthropic pandas
    export ANTHROPIC_API_KEY=...
    python pilot_opus.py --dry-run      # build prompts, show one, count tokens; sends nothing
    python pilot_opus.py                # submit the batch, wait, analyse
    python pilot_opus.py --resume msgbatch_...   # analyse a batch you already submitted
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path

import anthropic
import pandas as pd

# =============================================================================
# CONFIG — edit these
# =============================================================================

MODEL = "claude-sonnet-5-5"
EFFORT = "low"                      # lowest effort; Opus 5.5 can't disable thinking
MAX_TOKENS = 2000                   # hard cap on thinking + answer
K = 10                              # fixed shots per dataset (same for every query -> cached)
N_PER_DATASET = 2            # pilot molecules per dataset (from the valid split)
SEED = 0
CACHE_TTL = "1h"                    # batches can take > 5 min

DATA_DIR = Path("Claude_GPT_eval/datasets")             # expects data/<dataset>/{train,valid}.csv
SMILES_COL = "smiles"               # every other column is treated as a label
OUT_DIR = Path("pilot_results")

# Batch prices for Opus 5.5 (USD per million tokens) = standard x 0.5
PRICE = {"input": 2.00, "output": 10.00, "cache_write_1h": 4.00, "cache_read": 0.10}

# Short task descriptions. Edit freely: they change the prompt, not the code.
DATASET_INFO = {
    "bace_classification": ("cls", "Inhibition of human beta-secretase 1 (BACE-1)."),
    "bace_regression":     ("reg", "Binding affinity to human BACE-1, as pIC50."),
    "bbbp":                ("cls", "Blood-brain barrier penetration."),
    "clearance":           ("reg", "Clearance, in the same units as the examples."),
    "clintox":             ("cls", "FDA approval status and failure in clinical trials for toxicity."),
    "delaney":             ("reg", "Aqueous solubility, as log10(mol/L)."),
    "freesolv":            ("reg", "Hydration free energy in water, in kcal/mol."),
    "hiv":                 ("cls", "Ability to inhibit HIV replication."),
    "lipo":                ("reg", "Lipophilicity, as octanol/water distribution coefficient logD at pH 7.4."),
    "sider":               ("cls", "Marketed-drug side effects, grouped by system organ class."),
    "tox21":               ("cls", "Toxicity in 12 Tox21 nuclear-receptor and stress-response assays."),
}

SYSTEM_TEMPLATE = """You are an expert medicinal chemist predicting molecular properties from SMILES.

Task: {description}
Targets: {targets}

{answer_rule}

Respond with a single JSON object and nothing else, using exactly these keys: {keys}

Labelled examples (null = label not measured):
{shots}"""

CLS_RULE = ("For each target, give the probability (0 to 1) that the molecule is positive. "
            "Use two decimal places.")
REG_RULE = "For each target, give your best numeric estimate. Use two decimal places."


# =============================================================================
# PROMPT BUILDING
# =============================================================================

def load(name: str, split: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name / f"{split}.csv")

def fmt_label(v, typ: str):
    if pd.isna(v):
        return None
    return int(v) if typ == "cls" else round(float(v), 3)

def build_system(name: str, train: pd.DataFrame, label_cols: list[str]) -> str:
    typ, desc = DATASET_INFO[name]
    shots_df = train.sample(min(K, len(train)), random_state=SEED)
    shots = "\n".join(
        f"SMILES: {row[SMILES_COL]}\nAnswer: "
        + json.dumps({c: fmt_label(row[c], typ) for c in label_cols})
        for _, row in shots_df.iterrows()
    )
    return SYSTEM_TEMPLATE.format(
        description=desc,
        targets=", ".join(label_cols),
        answer_rule=CLS_RULE if typ == "cls" else REG_RULE,
        keys=json.dumps(label_cols),
        shots=shots,
    )

def build_requests(datasets: list[str] | None = None) -> tuple[list[dict], dict]:
    requests, meta = [], {}
    for name in (datasets or DATASET_INFO):
        train, valid = load(name, "train"), load(name, "valid")
        label_cols = [c for c in train.columns if c != SMILES_COL]
        system = build_system(name, train, label_cols)
        sample = valid.sample(min(N_PER_DATASET, len(valid)), random_state=SEED)
        for i, (_, row) in enumerate(sample.iterrows()):
            cid = f"{name}-{i}"
            meta[cid] = {"dataset": name, "smiles": row[SMILES_COL], "label_cols": label_cols,
                         "truth": {c: fmt_label(row[c], DATASET_INFO[name][0]) for c in label_cols}}
            requests.append({
                "custom_id": cid,
                "params": {
                    "model": MODEL,
                    "max_tokens": MAX_TOKENS,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": EFFORT},
                    # The breakpoint sits on the system block (instructions + shots), which is
                    # identical for every molecule in the dataset, so it is what gets cached.
                    "system": [{"type": "text", "text": system,
                                "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}}],
                    "messages": [{"role": "user", "content": f"SMILES: {row[SMILES_COL]}"}],
                },
            })
    return requests, meta


# =============================================================================
# RUN
# =============================================================================

def submit_and_wait(client: anthropic.Anthropic, requests: list[dict]) -> str:
    batch = client.messages.batches.create(requests=requests)
    print(f"Submitted batch {batch.id} with {len(requests)} requests.")
    (OUT_DIR / "batch_id.txt").write_text(batch.id)
    while True:
        b = client.messages.batches.retrieve(batch.id)
        c = b.request_counts
        print(f"  {b.processing_status}: {c.succeeded} ok, {c.errored} errored, "
              f"{c.processing} processing")
        if b.processing_status == "ended":
            return batch.id
        time.sleep(60)

def parse_answer(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def visible_tokens(client: anthropic.Anthropic, text: str, baseline: int) -> int:
    """Tokens in the visible answer, so thinking = output_tokens - visible."""
    if not text:
        return 0
    n = client.messages.count_tokens(model=MODEL, messages=[{"role": "user", "content": text}])
    return max(0, n.input_tokens - baseline)

def collect(client: anthropic.Anthropic, batch_id: str, meta: dict) -> pd.DataFrame:
    baseline = client.messages.count_tokens(
        model=MODEL, messages=[{"role": "user", "content": "."}]).input_tokens - 1
    rows = []
    for entry in client.messages.batches.results(batch_id):
        m = meta.get(entry.custom_id, {})
        row = {"custom_id": entry.custom_id, "dataset": m.get("dataset"), "status": entry.result.type}
        if entry.result.type == "succeeded":
            msg = entry.result.message
            u = msg.usage
            text = "".join(b.text for b in msg.content if b.type == "text")
            vis = visible_tokens(client, text, baseline)
            answer = parse_answer(text)
            row.update({
                "stop_reason": msg.stop_reason,
                "input_tokens": u.input_tokens,
                "cache_write_tokens": u.cache_creation_input_tokens or 0,
                "cache_read_tokens": u.cache_read_input_tokens or 0,
                "output_tokens": u.output_tokens,
                "visible_tokens": vis,
                "thinking_tokens": max(0, u.output_tokens - vis),
                "parsed": answer is not None
                          and set(m.get("label_cols", [])) <= set(answer or {}),
                "answer": json.dumps(answer),
                "truth": json.dumps(m.get("truth")),
                "text": text,
            })
        else:
            row["error"] = str(getattr(entry.result, "error", entry.result.type))
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# ANALYSIS
# =============================================================================

def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(q * len(xs)) - 1)] if xs else 0

def analyse(df: pd.DataFrame) -> None:
    ok = df[df["status"] == "succeeded"]
    print(f"\n{len(ok)}/{len(df)} requests succeeded.\n")
    print(f"{'dataset':<20}{'n':>4}{'think med':>11}{'p95':>7}{'max':>7}"
          f"{'truncated':>11}{'parse ok':>10}{'cache hit':>11}")
    for name, g in ok.groupby("dataset"):
        t = g["thinking_tokens"].tolist()
        cached = g["cache_write_tokens"].gt(0) | g["cache_read_tokens"].gt(0)
        hit = g["cache_read_tokens"].gt(0).sum() / max(1, cached.sum()) if cached.any() else float("nan")
        print(f"{name:<20}{len(g):>4}{statistics.median(t):>11.0f}{pct(t, .95):>7}{max(t):>7}"
              f"{(g['stop_reason'] == 'max_tokens').sum():>11}{g['parsed'].mean():>10.0%}"
              f"{hit:>11.0%}")

    t_all = ok["thinking_tokens"].tolist()
    cached = ok["cache_write_tokens"].gt(0) | ok["cache_read_tokens"].gt(0)
    hit_rate = ok.loc[cached, "cache_read_tokens"].gt(0).mean() if cached.any() else float("nan")
    cost = (ok["input_tokens"].sum() * PRICE["input"]
            + ok["cache_write_tokens"].sum() * PRICE["cache_write_1h"]
            + ok["cache_read_tokens"].sum() * PRICE["cache_read"]
            + ok["output_tokens"].sum() * PRICE["output"]) / 1e6

    print(f"\nThinking tokens per request: median {statistics.median(t_all):.0f}, "
          f"p95 {pct(t_all, .95)}, max {max(t_all)}")
    print(f"Truncated by max_tokens:     {(ok['stop_reason'] == 'max_tokens').sum()} of {len(ok)}")
    print(f"Answers parsed:              {ok['parsed'].mean():.0%}")
    print(f"Cache hit rate:              {hit_rate:.0%} (requests with a cacheable prefix)")
    print(f"Pilot cost:                  ${cost:.2f}")
    print("\nFor moleculenet_cost.py:")
    print(f'  MODEL_CONFIG["{MODEL}"]["thinking_tokens"] = {statistics.median(t_all):.0f}   # median')
    print(f"  CACHE_HIT_RATE = {hit_rate:.2f}")
    print(f"  Suggested MAX_TOKENS for the full run: ~{int(pct(t_all, .99) * 1.5) + 500}")


# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build prompts and count tokens only")
    ap.add_argument("--resume", metavar="BATCH_ID", help="analyse an already-submitted batch")
    ap.add_argument("--datasets", nargs="+", choices=list(DATASET_INFO), metavar="NAME",
                    help="run only these datasets (default: all). "
                         f"Choices: {', '.join(DATASET_INFO)}")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    client = anthropic.Anthropic()
    # With --resume, pass the same --datasets you submitted with so results match their metadata.
    requests, meta = build_requests(args.datasets)
    n_ds = len(args.datasets) if args.datasets else len(DATASET_INFO)
    print(f"Built {len(requests)} requests across {n_ds} dataset(s).")

    if args.dry_run:
        first = requests[0]["params"]
        print("\n--- example system prompt ---\n" + first["system"][0]["text"])
        print("--- example user message ---\n" + first["messages"][0]["content"])
        print("\nPrompt tokens per dataset (first request):")
        seen = set()
        for r in requests:
            name = meta[r["custom_id"]]["dataset"]
            if name in seen:
                continue
            seen.add(name)
            p = r["params"]
            n = client.messages.count_tokens(model=MODEL, system=p["system"],
                                             messages=p["messages"]).input_tokens
            note = "" if n >= 512 else "  (below the 512-token cache minimum: not cached)"
            print(f"  {name:<20}{n:>7,}{note}")
        return

    batch_id = args.resume or submit_and_wait(client, requests)
    df = collect(client, batch_id, meta)
    out = OUT_DIR / f"{batch_id}.csv"
    df.to_csv(out, index=False)
    print(f"Saved per-request results to {out}")
    analyse(df)


if __name__ == "__main__":
    main()