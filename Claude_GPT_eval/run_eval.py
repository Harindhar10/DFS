"""Evaluate Claude and GPT models on MoleculeNet property prediction.

Zero-shot baseline (k=0) and in-context learning (k>0 random train examples).

Batch workflow (default, 50% cheaper):
    python run_eval.py submit --shots 0,5,20 --dry-run   # request count + cost estimate
    python run_eval.py submit --shots 0,5,20             # prints a run id
    python run_eval.py status   --run-id <id>
    python run_eval.py collect  --run-id <id>            # download, parse, cost, log to Langfuse
    python run_eval.py evaluate --run-id <id>

Realtime smoke test:
    python run_eval.py submit --mode realtime --limit 3 --datasets bbbp,delaney --shots 0,5
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from dotenv import load_dotenv

from config import DATASETS, DEFAULT_MODELS, DEFAULT_SHOTS, ROOT, RUNS_DIR
from prompts import build_messages, load_split, parse_response, system_text

load_dotenv(ROOT / '.env')

import backends  # noqa: E402  (imports litellm and langfuse; after .env is loaded)
from evaluate import evaluate_run, load_results  # noqa: E402

CUSTOM_ID_RE = re.compile(r'^(?P<dataset>.+)-k(?P<k>\d+)-r(?P<row>\d+)$')


# ---------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------

def load_manifest(run_id):
    """Read runs/<run_id>/manifest.json: the run's config and its submitted batches."""
    with open(RUNS_DIR / run_id / 'manifest.json') as f:
        return json.load(f)


def save_manifest(manifest):
    """Write the manifest back to runs/<run_id>/manifest.json (creating the run directory)."""
    path = RUNS_DIR / manifest['run_id'] / 'manifest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)


def append_results(run_id, records):
    """Append result records to runs/<run_id>/results.jsonl."""
    with open(RUNS_DIR / run_id / 'results.jsonl', 'a') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')


def done_keys(run_id):
    """(model, custom_id) pairs that already have a usable (non-API-error) result."""
    res = load_results(RUNS_DIR / run_id)
    if res.empty:
        return set()
    res = res[res['status'] != 'api_error']
    return set(zip(res['model'], res['custom_id']))


def make_record(run_id, model, custom_id, text, usage, cost, error, stop_reason, mode):
    """One results.jsonl record: parsed predictions, status ('ok', 'parse_error' or 'api_error'), tokens and cost."""
    m = CUSTOM_ID_RE.match(custom_id)
    dataset, k, row = m['dataset'], int(m['k']), int(m['row'])
    if error:
        predictions, status = {}, 'api_error'
    else:
        predictions, error = parse_response(dataset, text)
        status = 'ok' if error is None else 'parse_error'
    return {
        'run_id': run_id, 'model': model, 'provider': backends.provider_of(model), 'mode': mode,
        'dataset': dataset, 'k': k, 'row': row, 'custom_id': custom_id,
        'text': text, 'predictions': predictions, 'status': status, 'error': error,
        'stop_reason': stop_reason, **(usage or backends.normalize_usage({})), 'cost_usd': cost,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_submit(args):
    """Build every (model, dataset, k, molecule) request and run it: dry-run cost estimate, realtime, or batch.

    With --run-id, resumes a run: only requests without a usable result and not in an uncollected batch are sent.
    """
    if args.run_id:
        manifest = load_manifest(args.run_id)
        cfg = manifest['config']
    else:
        run_id = datetime.now().strftime('%Y%m%d-%H%M%S') + ('-dry' if args.dry_run else '')
        cfg = {'models': args.models.split(','), 'datasets': args.datasets.split(','),
               'shots': [int(s) for s in args.shots.split(',')], 'split': args.split,
               'seed': args.seed, 'limit': args.limit, 'mode': args.mode}
        unknown = [d for d in cfg['datasets'] if d not in DATASETS]
        if unknown:
            raise SystemExit(f'Unknown datasets: {unknown}. Choose from {list(DATASETS)}')
        manifest = {'run_id': run_id, 'created': datetime.now(timezone.utc).isoformat(),
                    'config': cfg, 'batches': []}
    run_id = manifest['run_id']

    # (dataset, k, row, custom_id, messages) for every query; --limit samples molecules per dataset.
    requests = []
    for dataset in cfg['datasets']:
        test = load_split(dataset, cfg['split'])
        rows = test.index
        if cfg['limit']:
            rows = test.sample(n=min(cfg['limit'], len(test)), random_state=cfg['seed']).index
        requests += [(dataset, k, int(row), f'{dataset}-k{k}-r{int(row)}',
                      build_messages(dataset, cfg['split'], k, int(row), cfg['seed']))
                     for k in cfg['shots'] for row in rows]

    skip = set() if args.dry_run or not args.run_id else done_keys(run_id)
    skip |= {(b['model'], cid) for b in manifest['batches'] if not b.get('collected') for cid in b['custom_ids']}
    todo = {model: [r for r in requests if (model, r[3]) not in skip] for model in cfg['models']}

    if args.dry_run:
        batch = cfg['mode'] == 'batch'
        total = 0.0
        print(f"Mode: {cfg['mode']} | split: {cfg['split']} | shots: {cfg['shots']}")
        for model, reqs in todo.items():
            in_tok = sum(backends.estimate_tokens(system_text(m) + m[1]['content']) for *_, m in reqs)
            out_tok = sum(8 + 8 * len(DATASETS[d]['tasks']) for d, *_ in reqs)
            cost = backends.estimate_cost(model, in_tok, out_tok, batch)
            total += cost or 0
            cost_s = f'${cost:.2f}' if cost is not None else 'unknown price'
            print(f'  {model}: {len(reqs)} requests, ~{in_tok / 1e6:.2f}M input / ~{out_tok / 1e6:.2f}M output tokens, '
                  f'~{cost_s}')
        print(f'Estimated total (no caching, excludes reasoning tokens): ~${total:.2f}')
        return

    save_manifest(manifest)
    print(f'Run id: {run_id}')

    if cfg['mode'] == 'realtime':
        def one(model, dataset, k, row, cid, messages):
            metadata = {'session_id': run_id, 'trace_name': f'{dataset}/k{k}', 'generation_name': 'predict',
                        'tags': [model, dataset, f'k={k}', 'realtime']}
            try:
                text, usage, cost, stop = backends.call_realtime(model, dataset, messages, metadata)
                return make_record(run_id, model, cid, text, usage, cost, None, stop, 'realtime')
            except Exception as e:
                return make_record(run_id, model, cid, None, None, None, repr(e), None, 'realtime')

        jobs = [(model, *r) for model, reqs in todo.items() for r in reqs]
        print(f'Running {len(jobs)} realtime requests with {args.workers} workers')
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(one, *j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                append_results(run_id, [rec])  # main thread only, so no lock needed
                if rec['status'] != 'ok':
                    print(f"  [{rec['status']}] {rec['model']} {rec['custom_id']}: {rec['error']}")
                if i % 50 == 0 or i == len(jobs):
                    print(f'  {i}/{len(jobs)} done')
        print(f'Results: {RUNS_DIR / run_id / "results.jsonl"}\nNext: python run_eval.py evaluate --run-id {run_id}')
        return

    for model, reqs in todo.items():
        for start in range(0, len(reqs), args.max_batch_requests):
            chunk = reqs[start:start + args.max_batch_requests]
            body = [{'custom_id': cid, 'body': backends.batch_body(model, dataset, k, messages)}
                    for dataset, k, row, cid, messages in chunk]
            input_path = RUNS_DIR / run_id / f"batch_input_{len(manifest['batches']):03d}.jsonl"
            batch_id = backends.submit_batch(model, body, input_path, run_id)
            manifest['batches'].append({'model': model, 'batch_id': batch_id, 'n': len(chunk),
                                        'collected': False, 'custom_ids': [r[3] for r in chunk]})
            save_manifest(manifest)
            print(f'  submitted {model}: {len(chunk)} requests -> batch {batch_id}')


def cmd_status(args):
    """Print the progress of every batch of a run."""
    manifest = load_manifest(args.run_id)
    for b in manifest['batches']:
        if b.get('collected'):
            print(f"  {b['model']} {b['batch_id']}: collected ({b['n']} requests)")
            continue
        s = backends.batch_status(b['model'], b['batch_id'])
        print(f"  {b['model']} {b['batch_id']}: {s['status']} - succeeded {s['succeeded']}, "
              f"failed {s['failed']}, pending {s['pending']} (of {b['n']})")


def cmd_collect(args):
    """Download finished batches, parse and cost each result, append to results.jsonl and log to Langfuse."""
    manifest = load_manifest(args.run_id)
    cfg, run_id = manifest['config'], manifest['run_id']
    done = done_keys(run_id)

    for b in manifest['batches']:
        if b.get('collected'):
            continue
        model = b['model']
        status = backends.batch_status(model, b['batch_id'])
        if not status['done']:
            print(f"  {model} {b['batch_id']}: {status['status']} (pending {status['pending']}) - not ready")
            continue
        records, seen = [], set()
        for res in backends.batch_results(model, b['batch_id'], status):
            seen.add(res['custom_id'])
            if (model, res['custom_id']) not in done:
                records.append(make_record(run_id, model, res['custom_id'], res['text'], res['usage'],
                                           res['cost'], res['error'], res['stop_reason'], 'batch'))
        # Requests the provider never returned (e.g. an expired OpenAI batch) are marked as API errors.
        for cid in set(b['custom_ids']) - seen:
            records.append(make_record(run_id, model, cid, None, None, None, 'no result returned', None, 'batch'))
        append_results(run_id, records)
        backends.log_generations([(r, build_messages(r['dataset'], cfg['split'], r['k'], r['row'], cfg['seed']))
                                  for r in records])
        b['collected'] = True
        save_manifest(manifest)
        n_bad = sum(r['status'] != 'ok' for r in records)
        print(f"  {model} {b['batch_id']}: collected {len(records)} results ({n_bad} not ok)")


def cmd_evaluate(args):
    """Score a run, save runs/<id>/metrics.csv and post the scores to Langfuse."""
    manifest = load_manifest(args.run_id)
    metrics = evaluate_run(RUNS_DIR / args.run_id, manifest['config']['split'])
    if metrics.empty:
        return
    out = RUNS_DIR / args.run_id / 'metrics.csv'
    metrics.to_csv(out, index=False)
    print(f'\nSaved {out}')
    if not args.no_langfuse:
        backends.log_scores(args.run_id, metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    s = sub.add_parser('submit', help='build prompts and submit batches (or run realtime)')
    s.add_argument('--models', default=','.join(DEFAULT_MODELS),
                   help='comma-separated; claude-* -> Anthropic, others -> OpenAI (or prefix provider/)')
    s.add_argument('--datasets', default=','.join(DATASETS))
    s.add_argument('--shots', default=','.join(map(str, DEFAULT_SHOTS)), help='comma-separated k values; 0 = zero-shot')
    s.add_argument('--split', default='test')
    s.add_argument('--seed', type=int, default=0, help='seed for ICL example selection and --limit sampling')
    s.add_argument('--limit', type=int, default=None, help='random sample of N molecules per dataset')
    s.add_argument('--mode', choices=['batch', 'realtime'], default='batch')
    s.add_argument('--workers', type=int, default=8, help='realtime concurrency')
    s.add_argument('--max-batch-requests', type=int, default=5000)
    s.add_argument('--run-id', help='resume an existing run: submit only requests without a usable result')
    s.add_argument('--dry-run', action='store_true', help='print request counts and estimated cost only')
    s.set_defaults(func=cmd_submit)

    for name, func, helptext in [('status', cmd_status, 'show batch progress'),
                                 ('collect', cmd_collect, 'download finished batches, log to Langfuse')]:
        c = sub.add_parser(name, help=helptext)
        c.add_argument('--run-id', required=True)
        c.set_defaults(func=func)

    e = sub.add_parser('evaluate', help='compute ROC-AUC / RMSE and costs')
    e.add_argument('--run-id', required=True)
    e.add_argument('--no-langfuse', action='store_true', help='do not post scores to Langfuse')
    e.set_defaults(func=cmd_evaluate)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
