"""Scores collected predictions with DeepChem's metric functions.

Classification: ROC-AUC per task (dc.metrics.roc_auc_score), averaged over tasks.
Regression: RMSE per task (dc.metrics.rms_score), averaged over tasks.
Missing or unparsable predictions are filled with the train-set prior (positive rate
or mean) so every model is scored on the same molecules; the fill rate is reported.
"""
import json

import numpy as np
import pandas as pd

from config import DATASETS
from prompts import load_split


def load_results(run_dir):
    """All records of run_dir/results.jsonl as a DataFrame, keeping the latest record per (model, custom_id)."""
    path = run_dir / 'results.jsonl'
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        df = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    return df.drop_duplicates(subset=['model', 'custom_id'], keep='last').reset_index(drop=True)


def evaluate_run(run_dir, split):
    """Score every (model, dataset, k) group of a run, print a summary and return the metrics DataFrame.

    Each row holds the metric value, request count, parse-failure and fill rates, token totals and cost.
    Returns an empty DataFrame if the run has no results yet.
    """
    import deepchem as dc

    results = load_results(run_dir)
    if results.empty:
        print('No results yet.')
        return pd.DataFrame()

    rows = []
    for (model, dataset, k), g in results.groupby(['model', 'dataset', 'k']):
        cols = list(DATASETS[dataset]['tasks'])
        is_cls = DATASETS[dataset]['type'] == 'classification'
        prior = load_split(dataset, 'train')[cols].mean().to_numpy(dtype=float)
        y_true = load_split(dataset, split).loc[g['row'].to_numpy(), cols].to_numpy(dtype=float)
        y_pred = np.array([[np.nan if (p or {}).get(c) is None else p[c] for c in cols]
                           for p in g['predictions']], dtype=float)
        missing = np.isnan(y_pred)
        y_pred = np.where(missing, prior, y_pred)

        scores = []
        for j in range(len(cols)):
            mask = ~np.isnan(y_true[:, j])
            yt, yp = y_true[mask, j], y_pred[mask, j]
            if is_cls and len(np.unique(yt)) == 2:  # AUC needs both classes
                scores.append(dc.metrics.roc_auc_score(yt, yp))
            elif not is_cls and len(yt):
                scores.append(dc.metrics.rms_score(yt, yp))

        cost = g['cost_usd'].dropna()
        rows.append({
            'model': model, 'dataset': dataset, 'k': int(k), 'metric': 'roc_auc' if is_cls else 'rmse',
            'value': float(np.mean(scores)) if scores else float('nan'),
            'n': len(g), 'tasks_scored': len(scores),
            'parse_fail_rate': float((g['status'] != 'ok').mean()),
            'fill_rate': float(missing.mean()),
            'input_tokens': int(g['input_tokens'].sum()),
            'cache_read_tokens': int(g['cache_read_tokens'].sum()),
            'output_tokens': int(g['output_tokens'].sum()),
            'cost_usd': round(float(cost.sum()), 6) if len(cost) else None,
        })
    metrics = pd.DataFrame(rows).sort_values(['dataset', 'model', 'k']).reset_index(drop=True)

    pd.set_option('display.width', 200)
    for metric, direction in [('roc_auc', 'higher is better'), ('rmse', 'lower is better')]:
        m = metrics[metrics['metric'] == metric]
        if len(m):
            print(f'\n=== {metric.upper()} ({direction}) ===')
            print(m.pivot_table(index='dataset', columns=['model', 'k'], values='value').round(3).to_string())
    print('\n=== Cost (USD) and parse failures ===')
    summary = metrics.groupby(['model', 'k']).agg(
        cost_usd=('cost_usd', 'sum'), requests=('n', 'sum'),
        input_tokens=('input_tokens', 'sum'), cache_read_tokens=('cache_read_tokens', 'sum'),
        output_tokens=('output_tokens', 'sum'), parse_fail_rate=('parse_fail_rate', 'mean'))
    print(summary.round(4).to_string())
    print(f"\nTotal cost: ${metrics['cost_usd'].sum():.4f}")
    return metrics
