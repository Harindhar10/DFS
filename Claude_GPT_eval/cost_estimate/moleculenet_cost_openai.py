"""
MoleculeNet x OpenAI (GPT): cost & token budget estimator.

Mirrors moleculenet_cost.py (the Claude version) so results are comparable:
  * Batch API (50% off everything; the official Batch table also halves
    cached-input and cache-write prices, so the discounts stack).
  * Prompt caching on the shared prefix (instructions [+ few-shot block]).
  * Lowest available reasoning effort, one test molecule per request.
  * Multi-task datasets: one request predicts ALL labels as JSON.
  * Classification -> probability in [0, 1]; regression -> a number.

OpenAI-specific differences from the Claude script:
  * Cache: 1,024-token minimum, cache writes 1.25x input, fixed 30-minute TTL.
  * Long context: a prompt over 272,000 input tokens (cached tokens included)
    pays long-context rates for the WHOLE request.
  * gpt-6-astra and gpt-6.1-sol cannot run with reasoning effort "none";
    their minimum is "low", so set reasoning_tokens to what you measure.

Prices from https://developers.openai.com/api/docs/pricing (October 2026).
Re-check before you spend money.
"""

from __future__ import annotations

import math
import warnings

# =============================================================================
# 1. HYPERPARAMETERS — edit these
# =============================================================================

MODEL = "gpt-6.1-sol"              # key into MODEL_CONFIG
K = 10                             # shots: int, "all" (full train set), or {dataset: int|"all"}
SHOT_STRATEGY = "fixed"            # "fixed": same K shots for every query -> cacheable prefix
                                   # "per_query": shots chosen per molecule (e.g. kNN) -> not cacheable
SPLITS = ("valid", "test")         # valid = picking prompt / K, test = final numbers
N_SEEDS = 1                        # repeats with different shot samples
DATASETS_TO_RUN = None             # None = all, or e.g. ["bbbp", "tox21"]

USE_BATCH = True
BATCH_MULTIPLIER = 0.5
USE_CACHE = True
CACHE_HIT_RATE = 0.8               # cache hits are best-effort, not guaranteed

# --- token-shape assumptions (calibrate with calibrate_chars_per_token()) ---
CHARS_PER_TOKEN = 1.01              # SMILES tokenize worse than English (~4 chars/token)
SYSTEM_PROMPT_TOKENS = 250         # generic instructions + output-format spec
DATASET_DESC_TOKENS = 100          # per-dataset task description
EXAMPLE_OVERHEAD_TOKENS = 8        # "SMILES: ... \nLabels: ...\n" scaffolding per shot
QUERY_OVERHEAD_TOKENS = 20         # wrapper around the test SMILES
LABEL_SEP_TOKENS = 3               # quotes / colon / comma per JSON entry
LABEL_WRAPPER_TOKENS = 2           # { }
CLS_LABEL_IN_TOKENS = 1            # "0" / "1" in a shot
REG_LABEL_IN_TOKENS = 4            # "-3.27" in a shot
CLS_PROB_OUT_TOKENS = 3            # "0.73" in the answer
REG_VALUE_OUT_TOKENS = 4           # "-3.27" in the answer
OUTPUT_WRAPPER_TOKENS = 5          # braces, stop, stray whitespace


# =============================================================================
# 2. MODEL CONFIG — USD per million tokens (standard, pre-batch prices)
# =============================================================================

LONG_CONTEXT_THRESHOLD = 272_000   # prompts above this pay long-context rates

MODEL_CONFIG = {
    "gpt-6-astra": {
        "input": 10.00, "cached_input": 1.00, "cache_write": 12.50, "output": 50.00,
        "long": {"input": 20.00, "cached_input": 2.00, "cache_write": 25.00, "output": 75.00},
        "context_window": 1_050_000, "min_cache_tokens": 1024,
        "min_reasoning_effort": "low",
        "reasoning_tokens": 500,     # can't disable reasoning: set to measured tokens/request
    },
    "gpt-6.1-sol": {
        "input": 2.00, "cached_input": 0.10, "cache_write": 2.50, "output": 10.00,
        "long": {"input": 4.00, "cached_input": 0.20, "cache_write": 5.00, "output": 15.00},
        "context_window": 1_050_000, "min_cache_tokens": 1024,
        "min_reasoning_effort": "low",
        "reasoning_tokens": 500,     # can't disable reasoning: set to measured tokens/request
    },
    "gpt-6-luna": {
        "input": 0.10, "cached_input": 0.01, "cache_write": 0.125, "output": 0.50,
        "long": {"input": 0.20, "cached_input": 0.02, "cache_write": 0.25, "output": 0.75},
        "context_window": 1_050_000, "min_cache_tokens": 1024,
        "min_reasoning_effort": "none",
        "reasoning_tokens": 0,
    },
}


# =============================================================================
# 3. DATASETS — sizes, task structure, mean SMILES length per split
# =============================================================================

def round10(x: float) -> int:
    """Round to the nearest multiple of 10 (minimum 10)."""
    return max(10, int(math.floor(x / 10 + 0.5)) * 10)

# (n_train, n_valid, n_test), type, n_tasks, avg tokens per task name,
# mean SMILES length (train, valid, test) from your stats table.
_RAW = {
    #                      sizes                 type     tasks name_tok  mean len (train, valid, test)
    "bace_classification": ((1210, 151, 152),    "cls",   1,    1,  (64.0, 66.1, 74.3)),
    "bace_regression":     ((1210, 151, 152),    "reg",   1,    1,  (64.0, 66.1, 74.3)),
    "bbbp":                ((1640, 203, 189),    "cls",   1,    1,  (48.7, 68.8, 49.5)),
    "clearance":           ((668, 84, 84),       "reg",   1,    1,  (52.6, 52.8, 50.5)),
    "clintox":             ((1163, 142, 143),    "cls",   2,    5,  (54.5, 70.9, 47.3)),
    "delaney":             ((902, 113, 113),     "reg",   1,    1,  (20.1, 31.6, 32.7)),
    "freesolv":            ((513, 64, 65),       "reg",   1,    1,  (13.6, 23.2, 21.5)),
    "hiv":                 ((32840, 4103, 4102), "cls",   1,    1,  (44.4, 49.9, 44.3)),
    "lipo":                ((3359, 419, 420),    "reg",   1,    1,  (47.2, 47.8, 48.6)),
    "sider":               ((1106, 133, 127),    "cls",   27,   8,  (51.7, 54.3, 70.9)),
    "tox21":               ((6248, 774, 762),    "cls",   12,   5,  (30.7, 50.8, 49.1)),
}

DATASETS = {
    name: {
        "n": dict(zip(("train", "valid", "test"), sizes)),
        "type": typ,
        "n_tasks": n_tasks,
        "name_tok": name_tok,
        "smiles_len": dict(zip(("train", "valid", "test"), (round10(x) for x in lens))),
    }
    for name, (sizes, typ, n_tasks, name_tok, lens) in _RAW.items()
}


# =============================================================================
# 4. TOKEN MODEL (identical to the Claude script)
# =============================================================================

def smiles_tokens(n_chars: int) -> int:
    return math.ceil(n_chars / CHARS_PER_TOKEN)

def label_block_tokens(ds: dict, value_tokens: int) -> int:
    return ds["n_tasks"] * (ds["name_tok"] + value_tokens + LABEL_SEP_TOKENS) + LABEL_WRAPPER_TOKENS

def shot_tokens(ds: dict) -> int:
    v = CLS_LABEL_IN_TOKENS if ds["type"] == "cls" else REG_LABEL_IN_TOKENS
    return EXAMPLE_OVERHEAD_TOKENS + smiles_tokens(ds["smiles_len"]["train"]) + label_block_tokens(ds, v)

def instruction_tokens(ds: dict) -> int:
    return SYSTEM_PROMPT_TOKENS + DATASET_DESC_TOKENS + ds["n_tasks"] * ds["name_tok"]

def query_tokens(ds: dict, split: str) -> int:
    return QUERY_OVERHEAD_TOKENS + smiles_tokens(ds["smiles_len"][split])

def output_tokens(ds: dict, model_cfg: dict) -> int:
    """Visible answer + reasoning tokens (reasoning is billed as output)."""
    v = CLS_PROB_OUT_TOKENS if ds["type"] == "cls" else REG_VALUE_OUT_TOKENS
    body = ds["n_tasks"] * (ds["name_tok"] + v + LABEL_SEP_TOKENS) + OUTPUT_WRAPPER_TOKENS
    return body + model_cfg.get("reasoning_tokens", 0)

def resolve_k(name: str, ds: dict, k) -> int:
    k_ds = k.get(name, 0) if isinstance(k, dict) else k
    n_train = ds["n"]["train"]
    if k_ds == "all":
        return n_train
    if k_ds > n_train:
        warnings.warn(f"[{name}] K={k_ds} > train size {n_train}; clamping to {n_train}.")
        return n_train
    return int(k_ds)

def max_k_that_fits(name: str, ds: dict, model_cfg: dict, split: str) -> int:
    budget = (model_cfg["context_window"] - instruction_tokens(ds)
              - query_tokens(ds, split) - output_tokens(ds, model_cfg))
    return max(0, min(ds["n"]["train"], budget // shot_tokens(ds)))


# =============================================================================
# 5. COST MODEL
# =============================================================================

def _prices(model_cfg: dict, prompt_tokens: int) -> dict:
    """Long-context rates apply to the whole request once the prompt crosses the threshold."""
    return model_cfg["long"] if prompt_tokens > LONG_CONTEXT_THRESHOLD else model_cfg

def estimate(model: str = MODEL, k=K, shot_strategy: str = SHOT_STRATEGY,
             splits=SPLITS, n_seeds: int = N_SEEDS, datasets=DATASETS_TO_RUN,
             cache_hit_rate: float = CACHE_HIT_RATE, use_cache: bool = USE_CACHE,
             use_batch: bool = USE_BATCH) -> list[dict]:
    """Return one row per (dataset, split) with token counts and USD cost."""
    cfg = MODEL_CONFIG[model]
    batch_mult = BATCH_MULTIPLIER if use_batch else 1.0
    rows = []

    for name in (datasets or DATASETS):
        ds = DATASETS[name]
        k_ds = resolve_k(name, ds, k)
        shots = k_ds * shot_tokens(ds)

        for split in splits:
            n = ds["n"][split]
            if shot_strategy == "fixed":
                prefix = instruction_tokens(ds) + shots
                suffix = query_tokens(ds, split)
            elif shot_strategy == "per_query":
                prefix = instruction_tokens(ds)
                suffix = shots + query_tokens(ds, split)
            else:
                raise ValueError(f"unknown SHOT_STRATEGY {shot_strategy!r}")
            out = output_tokens(ds, cfg)
            prompt = prefix + suffix

            fits = prompt + out <= cfg["context_window"]
            if not fits:
                warnings.warn(
                    f"[{model} | {name}/{split}] K={k_ds} -> {prompt + out:,} tokens exceeds the "
                    f"{cfg['context_window']:,}-token context window. Largest K that fits: "
                    f"{max_k_that_fits(name, ds, cfg, split):,}."
                )

            p = _prices(cfg, prompt)
            cacheable = use_cache and prefix >= cfg["min_cache_tokens"]
            n_req = n * n_seeds

            if cacheable:
                # Per seed: the first request writes; afterwards misses re-write, hits read.
                writes = n_seeds * (1 + (n - 1) * (1 - cache_hit_rate))
                reads = n_seeds * (n - 1) * cache_hit_rate
                in_cost = (writes * prefix * p["cache_write"]
                           + reads * prefix * p["cached_input"]
                           + n_req * suffix * p["input"])
            else:
                in_cost = n_req * prompt * p["input"]
            out_cost = n_req * out * p["output"]
            nocache_cost = (n_req * prompt * p["input"] + out_cost) * batch_mult / 1e6

            rows.append({
                "dataset": name, "split": split, "K": k_ds, "requests": n_req,
                "prefix_tok": prefix, "suffix_tok": suffix, "prompt_tok": prompt, "output_tok": out,
                "total_in_tok": n_req * prompt, "total_out_tok": n_req * out,
                "cached": cacheable, "fits": fits, "long_ctx": prompt > LONG_CONTEXT_THRESHOLD,
                # Requests that exceed the context window can't run, so they cost 0
                # and are left out of totals.
                "cost_usd": (in_cost + out_cost) * batch_mult / 1e6 if fits else 0.0,
                "cost_no_cache_usd": nocache_cost if fits else 0.0,
            })
    return rows


# =============================================================================
# 6. REPORTING
# =============================================================================

def report(model: str = MODEL, k=K, **kw) -> float:
    cfg = MODEL_CONFIG[model]
    print(f"\n=== {model} | K={k} | shots={kw.get('shot_strategy', SHOT_STRATEGY)} | "
          f"splits={kw.get('splits', SPLITS)} | seeds={kw.get('n_seeds', N_SEEDS)} ===")
    if cfg["min_reasoning_effort"] != "none" and cfg.get("reasoning_tokens", 0) == 0:
        print(f"NOTE: {model} can't disable reasoning (min effort '{cfg['min_reasoning_effort']}'); "
              f"reasoning_tokens=0 underestimates output cost.")
    rows = estimate(model=model, k=k, **kw)
    hdr = (f"{'dataset':<20}{'split':<6}{'K':>7}{'reqs':>7}{'prompt tok':>12}{'out tok':>8}"
           f"{'cache':>6}{'long':>5}{'cost $':>11}{'no-cache $':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cost = f"{r['cost_usd']:>11.2f}{r['cost_no_cache_usd']:>12.2f}" if r["fits"] else f"{'TOO LONG':>23}"
        print(f"{r['dataset']:<20}{r['split']:<6}{r['K']:>7,}{r['requests']:>7,}{r['prompt_tok']:>12,}"
              f"{r['output_tok']:>8,}{'yes' if r['cached'] else 'no':>6}{'yes' if r['long_ctx'] else '':>5}{cost}")
    ok = [r for r in rows if r["fits"]]
    total = sum(r["cost_usd"] for r in ok)
    print("-" * len(hdr))
    print(f"{'TOTAL (runnable)':<20}{'':<6}{'':>7}{sum(r['requests'] for r in ok):>7,}"
          f"{sum(r['total_in_tok'] for r in ok) / 1e6:>10.1f}M"
          f"{sum(r['total_out_tok'] for r in ok) / 1e6:>7.2f}M{'':>11}{total:>11.2f}"
          f"{sum(r['cost_no_cache_usd'] for r in ok):>12.2f}")
    skipped = sorted({f"{r['dataset']}/{r['split']}" for r in rows if not r["fits"]})
    if skipped:
        print(f"Excluded (exceeds context window): {', '.join(skipped)}")
    return total

def sweep(models=tuple(MODEL_CONFIG), k_values=(0, 1, 5, 10, 50, 100, 500, "all"), **kw) -> None:
    """Total cost (USD) for every model x K combination."""
    print(f"\n=== Total cost (USD), shots={kw.get('shot_strategy', SHOT_STRATEGY)}, "
          f"splits={kw.get('splits', SPLITS)}, seeds={kw.get('n_seeds', N_SEEDS)} ===")
    print(f"{'K':>6}" + "".join(f"{m:>16}" for m in models))
    for k in k_values:
        cells = []
        for m in models:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # report() shows the context warnings
                rows = estimate(model=m, k=k, **kw)
            cost = sum(r["cost_usd"] for r in rows)
            cells.append(f"{cost:,.2f}" + ("*" if not all(r["fits"] for r in rows) else ""))
        print(f"{str(k):>6}" + "".join(f"{c:>16}" for c in cells))
    print("* some datasets exceed the context window at this K and are excluded from the total")


# =============================================================================
# 7. OPTIONAL: calibrate CHARS_PER_TOKEN against the real tokenizer
# =============================================================================

def calibrate_chars_per_token(smiles: list[str], model: str = MODEL) -> float:
    """Count real tokens for a sample of SMILES (needs OPENAI_API_KEY).

    Uses OpenAI's token-counting endpoint (POST /v1/responses/input_tokens),
    which reflects the model's actual tokenizer.
    """
    from openai import OpenAI
    client = OpenAI()
    count = lambda text: client.responses.input_tokens.count(model=model, input=text).input_tokens
    joined = "\n".join(smiles)
    n_tok = count(joined) - count(".")  # subtract fixed message overhead
    return len(joined) / n_tok


if __name__ == "__main__":
    warnings.simplefilter("always")
    # Print warnings to stdout so they appear next to the table they belong to.
    warnings.showwarning = lambda msg, *a, **k: print(f"WARNING: {msg}")
    report()                     # detailed breakdown for MODEL / K above
    report(k="all")              # full-training-set condition (triggers context warnings)
    sweep()                      # every model x K