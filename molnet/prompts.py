"""Data loading, in-context example selection, prompt construction and response parsing.

Messages are built in the OpenAI/litellm chat format. The system message is a list of
text parts: [instructions, in-context examples]. Everything shared across queries sits
in the system message so the prompt prefix is identical for every molecule of a
(dataset, k) group and can be cached; the query SMILES comes last.
"""
import json
import math
import re
from functools import cache

import numpy as np
import pandas as pd

from .config import CLASSIFICATION_UNIT, DATA_DIR, DATASETS


@cache
def load_split(dataset, split):
    """Read datasets/<dataset>/<split>.csv (cached; treat the result as read-only)."""
    return pd.read_csv(DATA_DIR / dataset / f'{split}.csv')


def task_keys(dataset):
    """Short JSON keys (t1..tN) used in place of long column names to save output tokens."""
    return [f't{i + 1}' for i in range(len(DATASETS[dataset]['tasks']))]


def output_schema(dataset):
    """Strict JSON schema of the answer: one required number per task key."""
    keys = task_keys(dataset)
    return {
        'type': 'object',
        'properties': {k: {'type': 'number'} for k in keys},
        'required': keys,
        'additionalProperties': False,
    }


@cache
def select_examples(dataset, k, seed):
    """k random train examples, the same for every query of a dataset (cacheable prefix).

    Single-task classification draws balanced classes so rare positives (e.g. HIV) appear.
    The draw does not depend on k, so the 5-shot examples are a subset of the 20-shot ones.
    """
    train = load_split(dataset, 'train')
    if k == 0:
        return train.iloc[:0]
    cols = list(DATASETS[dataset]['tasks'])
    pool = train.dropna(subset=cols, how='all')
    rng = np.random.default_rng(seed)
    if DATASETS[dataset]['type'] == 'classification' and len(cols) == 1:
        pos = rng.permutation(pool.index[pool[cols[0]] == 1].to_numpy())
        neg = rng.permutation(pool.index[pool[cols[0]] == 0].to_numpy())
        n_pos = min(len(pos), k // 2)
        idx = rng.permutation(np.concatenate([pos[:n_pos], neg[:k - n_pos]]))
    else:
        idx = rng.permutation(pool.index.to_numpy())[:k]
    return pool.loc[idx].reset_index(drop=True)


def build_messages(dataset, split, k, row, seed):
    """OpenAI-format messages asking for the properties of test molecule `row`, with k examples.

    The result depends only on its arguments, so any request can be rebuilt from its custom_id.
    """
    ds = DATASETS[dataset]
    keys = task_keys(dataset)
    is_cls = ds['type'] == 'classification'

    lines = [
        'You are an expert medicinal and computational chemist. You predict molecular '
        'properties from SMILES strings.',
        '',
        'For the molecule given by the user, predict:',
        *(f'- {key}: {desc}' for key, desc in zip(keys, ds['tasks'].values())),
        '',
        f'Units: every value is the {CLASSIFICATION_UNIT}.' if is_cls else f"Units: {ds['unit']}.",
        f'Respond with only a JSON object with the keys {", ".join(keys)} and numeric values, '
        f'for example {json.dumps({key: 0.5 if is_cls else 0.0 for key in keys[:2]})}.',
    ]
    system_parts = [{'type': 'text', 'text': '\n'.join(lines)}]

    examples = select_examples(dataset, k, seed)
    if len(examples):
        header = 'Examples with measured values from the training set'
        header += ' (1 = true, 0 = false, null = not measured):' if is_cls else ':'
        blocks = [header]
        for _, ex in examples.iterrows():
            labels = {key: None if pd.isna(ex[col]) else int(ex[col]) if is_cls else round(float(ex[col]), 3)
                      for key, col in zip(keys, ds['tasks'])}
            blocks.append(f"SMILES: {ex['smiles']}\nAnswer: {json.dumps(labels)}")
        system_parts.append({'type': 'text', 'text': '\n\n'.join(blocks)})

    smiles = load_split(dataset, split).loc[row, 'smiles']
    return [
        {'role': 'system', 'content': system_parts},
        {'role': 'user', 'content': f'SMILES: {smiles}'},
    ]


def system_text(messages):
    """The system message's text parts joined into one string."""
    return '\n\n'.join(p['text'] for p in messages[0]['content'])


_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def parse_response(dataset, text):
    """Map a model's JSON answer back to ({column: float or None}, error or None).

    Classification probabilities are clipped to [0, 1].
    """
    cols = list(DATASETS[dataset]['tasks'])
    empty = {c: None for c in cols}
    if not text:
        return empty, 'empty response'
    match = _JSON_RE.search(text)
    if not match:
        return empty, 'no JSON object in response'
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return empty, f'invalid JSON: {e}'

    is_cls = DATASETS[dataset]['type'] == 'classification'
    preds, missing = {}, []
    for key, col in zip(task_keys(dataset), cols):
        try:
            v = float(obj.get(key))
        except (TypeError, ValueError):
            v = math.nan
        if math.isnan(v) or math.isinf(v):
            preds[col] = None
            missing.append(key)
        else:
            preds[col] = min(max(v, 0.0), 1.0) if is_cls else v
    return preds, (f'missing keys: {missing}' if missing else None)
