"""LLM access: realtime calls, batch jobs, cost accounting and Langfuse logging.

- Realtime calls go through litellm.completion; litellm's `langfuse_otel` callback
  logs each call (with litellm's cost) to Langfuse.
- Batch jobs use each provider's Batch API (50% cheaper). OpenAI batches go through
  litellm's batch API. litellm cannot create Anthropic batches, so those use the
  anthropic SDK directly. Batch results bypass litellm's callbacks, so their cost is
  computed with litellm's price table and they are logged to Langfuse here.
"""
import json
import os

import anthropic
import litellm
from langfuse import get_client, propagate_attributes

from config import ANTHROPIC_CACHE_MIN_TOKENS, BATCH_DISCOUNT, DATASETS, MODEL_SETTINGS, PRICE_FALLBACK
from prompts import output_schema, system_text

litellm.drop_params = True  # drop params a model does not support instead of erroring
for _name, _info in PRICE_FALLBACK.items():
    if _name not in litellm.model_cost:
        litellm.register_model({_name: _info})

if os.environ.get('LANGFUSE_PUBLIC_KEY') and os.environ.get('LANGFUSE_SECRET_KEY'):
    LANGFUSE = get_client()
else:
    LANGFUSE = None
    print('WARNING: LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set - nothing will be logged to Langfuse')


# ---------------------------------------------------------------------------
# Models and request bodies
# ---------------------------------------------------------------------------

def provider_of(model):
    """Provider of a model: an explicit 'provider/' prefix, else anthropic for claude-*, else openai."""
    if '/' in model:
        return model.split('/', 1)[0]
    return 'anthropic' if model.startswith('claude') else 'openai'


def bare_model(model):
    """Model name without its 'provider/' prefix."""
    return model.split('/', 1)[1] if '/' in model else model


def max_tokens(model, dataset):
    """Output-token cap: ~12 tokens per task value plus the model's reasoning headroom (unused tokens are not billed)."""
    headroom = MODEL_SETTINGS.get(bare_model(model), {}).get('thinking_headroom', 0)
    return 32 + 12 * len(DATASETS[dataset]['tasks']) + headroom


def estimate_tokens(text):
    """Rough token count: SMILES-heavy text tokenises at ~3 characters per token."""
    return len(text) // 3 + 1


def use_anthropic_cache(model, messages):
    """Whether to mark the system prompt for Anthropic caching (only if it reaches the model's minimum cacheable length)."""
    if provider_of(model) != 'anthropic':
        return False
    return estimate_tokens(system_text(messages)) >= ANTHROPIC_CACHE_MIN_TOKENS.get(bare_model(model), 1024)


def json_response_format(dataset):
    """OpenAI-style strict structured-output format for the dataset's answer schema."""
    return {'type': 'json_schema',
            'json_schema': {'name': 'prediction', 'strict': True, 'schema': output_schema(dataset)}}


def batch_body(model, dataset, k, messages):
    """Request body for one batch request: Anthropic Messages API params or an OpenAI chat-completions body."""
    s = MODEL_SETTINGS.get(bare_model(model), {})
    if provider_of(model) == 'anthropic':
        system = [dict(p) for p in messages[0]['content']]
        if use_anthropic_cache(model, messages):
            # Batch requests run in no fixed order over a long time, so use the 1h cache.
            system[-1]['cache_control'] = {'type': 'ephemeral', 'ttl': '1h'}
        output_config = {'format': {'type': 'json_schema', 'schema': output_schema(dataset)}}
        if 'effort' in s:
            output_config['effort'] = s['effort']
        body = {'model': bare_model(model), 'max_tokens': max_tokens(model, dataset), 'system': system,
                'messages': [messages[1]], 'output_config': output_config}
    else:
        body = {'model': bare_model(model),
                'messages': [{'role': 'system', 'content': system_text(messages)}, messages[1]],
                'max_completion_tokens': max_tokens(model, dataset),
                'response_format': json_response_format(dataset),
                'prompt_cache_key': f'{dataset}-k{k}'}  # same-prefix requests share a cache
        if 'reasoning_effort' in s:
            body['reasoning_effort'] = s['reasoning_effort']
    if 'temperature' in s:
        body['temperature'] = s['temperature']
    return body


# ---------------------------------------------------------------------------
# Usage and cost
# ---------------------------------------------------------------------------

def normalize_usage(u):
    """Token counts from either an Anthropic-native or an OpenAI/litellm usage dict.

    input_tokens always includes cached tokens (cache reads and writes).
    """
    u = u or {}
    if 'input_tokens' in u and 'prompt_tokens' not in u:  # Anthropic Messages API
        cr = u.get('cache_read_input_tokens') or 0
        cw = u.get('cache_creation_input_tokens') or 0
        return {'input_tokens': (u.get('input_tokens') or 0) + cr + cw,
                'output_tokens': u.get('output_tokens') or 0,
                'cache_read_tokens': cr, 'cache_write_tokens': cw, 'reasoning_tokens': 0}
    ptd = u.get('prompt_tokens_details') or {}
    ctd = u.get('completion_tokens_details') or {}
    return {'input_tokens': u.get('prompt_tokens') or 0,
            'output_tokens': u.get('completion_tokens') or 0,
            'cache_read_tokens': ptd.get('cached_tokens') or u.get('cache_read_input_tokens') or 0,
            'cache_write_tokens': u.get('cache_creation_input_tokens') or 0,
            'reasoning_tokens': ctd.get('reasoning_tokens') or 0}


def estimate_cost(model, input_tokens, output_tokens, batch):
    """USD cost estimate from litellm's prices (no cache pricing); None if the model has no price."""
    try:
        p, c = litellm.cost_per_token(model=bare_model(model), custom_llm_provider=provider_of(model),
                                      prompt_tokens=input_tokens, completion_tokens=output_tokens)
    except Exception:
        return None
    return (p + c) * (BATCH_DISCOUNT if batch else 1.0)


# ---------------------------------------------------------------------------
# Realtime (litellm)
# ---------------------------------------------------------------------------

def call_realtime(model, dataset, messages, metadata):
    """One synchronous litellm call. Returns (text, normalized usage, cost_usd, stop_reason)."""
    if LANGFUSE and not litellm.callbacks:
        litellm.callbacks = ['langfuse_otel']
    s = MODEL_SETTINGS.get(bare_model(model), {})
    if use_anthropic_cache(model, messages):
        messages = [dict(messages[0], content=[dict(p) for p in messages[0]['content']]), messages[1]]
        messages[0]['content'][-1]['cache_control'] = {'type': 'ephemeral'}
    kwargs = {
        'model': f'{provider_of(model)}/{bare_model(model)}',
        'messages': messages,
        'max_tokens': max_tokens(model, dataset),
        'response_format': json_response_format(dataset),
        'metadata': metadata,
        'num_retries': 3,
    }
    kwargs.update({key: s[key] for key in ('temperature', 'reasoning_effort') if key in s})
    resp = litellm.completion(**kwargs)
    cost = resp._hidden_params.get('response_cost')
    if cost is None:
        cost = litellm.completion_cost(completion_response=resp)
    choice = resp.choices[0]
    return choice.message.content, normalize_usage(resp.usage.model_dump()), cost, choice.finish_reason


# ---------------------------------------------------------------------------
# Batch APIs
# ---------------------------------------------------------------------------

def submit_batch(model, requests, input_path, run_id):
    """Submit [{'custom_id', 'body'}] as one provider batch and return its batch id.

    OpenAI batches are uploaded from a JSONL file written to input_path.
    """
    if provider_of(model) == 'anthropic':
        batch = anthropic.Anthropic().messages.batches.create(
            requests=[{'custom_id': r['custom_id'], 'params': r['body']} for r in requests])
        return batch.id

    with open(input_path, 'w') as f:
        for r in requests:
            f.write(json.dumps({'custom_id': r['custom_id'], 'method': 'POST',
                                'url': '/v1/chat/completions', 'body': r['body']}) + '\n')
    with open(input_path, 'rb') as fh:
        file_obj = litellm.create_file(file=fh, purpose='batch', custom_llm_provider='openai')
    batch = litellm.create_batch(completion_window='24h', endpoint='/v1/chat/completions',
                                 input_file_id=file_obj.id, custom_llm_provider='openai',
                                 metadata={'run_id': run_id, 'model': model})
    return batch.id


def batch_status(model, batch_id):
    """Batch progress: {'status', 'done', 'succeeded', 'failed', 'pending'} (+ OpenAI output/error file ids)."""
    if provider_of(model) == 'anthropic':
        b = anthropic.Anthropic().messages.batches.retrieve(batch_id)
        c = b.request_counts
        return {'status': b.processing_status, 'done': b.processing_status == 'ended',
                'succeeded': c.succeeded, 'failed': c.errored + c.expired + c.canceled,
                'pending': c.processing}
    b = litellm.retrieve_batch(batch_id=batch_id, custom_llm_provider='openai')
    c = b.request_counts
    total = c.total if c else 0
    done_n = (c.completed + c.failed) if c else 0
    return {'status': b.status, 'done': b.status in ('completed', 'failed', 'expired', 'cancelled'),
            'succeeded': c.completed if c else 0, 'failed': c.failed if c else 0,
            'pending': total - done_n,
            'output_file_id': b.output_file_id, 'error_file_id': b.error_file_id}


def batch_results(model, batch_id, status):
    """Yield {'custom_id', 'text', 'usage', 'cost', 'error', 'stop_reason'} per request of a finished batch.

    usage is normalized; cost is litellm's price (including cache pricing) times the batch discount.
    """
    provider = provider_of(model)
    raw = []  # (custom_id, text, raw_usage, error, stop_reason)
    if provider == 'anthropic':
        for r in anthropic.Anthropic().messages.batches.results(batch_id):
            if r.result.type == 'succeeded':
                msg = r.result.message
                text = next((b.text for b in msg.content if b.type == 'text'), '')
                raw.append((r.custom_id, text, msg.usage.model_dump(exclude_none=True), None, msg.stop_reason))
            else:
                err = r.result.type
                if r.result.type == 'errored':
                    err += ': ' + json.dumps(r.result.error.model_dump())
                raw.append((r.custom_id, None, {}, err, None))
    else:
        for file_id in (status.get('output_file_id'), status.get('error_file_id')):
            if not file_id:
                continue
            content = litellm.file_content(file_id=file_id, custom_llm_provider='openai')
            for line in content.text.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                resp = rec.get('response') or {}
                body = resp.get('body') or {}
                if resp.get('status_code') == 200 and body.get('choices'):
                    choice = body['choices'][0]
                    raw.append((rec['custom_id'], choice['message'].get('content'), body.get('usage') or {},
                                None, choice.get('finish_reason')))
                else:
                    err = rec.get('error') or body.get('error') or f"status {resp.get('status_code')}"
                    raw.append((rec['custom_id'], None, {}, json.dumps(err), None))

    for custom_id, text, usage, error, stop_reason in raw:
        cost = None
        if usage:
            try:
                if provider == 'anthropic':
                    u = litellm.AnthropicConfig().calculate_usage(usage_object=usage, reasoning_content=None)
                else:
                    u = litellm.Usage(**usage)
                p, c = litellm.cost_per_token(model=bare_model(model), custom_llm_provider=provider, usage_object=u)
                cost = BATCH_DISCOUNT * (p + c)
            except Exception as e:  # unknown model price
                print(f'WARNING: no price for {model}: {e}')
        yield {'custom_id': custom_id, 'text': text, 'usage': normalize_usage(usage), 'cost': cost,
               'error': error, 'stop_reason': stop_reason}


# ---------------------------------------------------------------------------
# Langfuse (batch results and run-level scores)
# ---------------------------------------------------------------------------

def log_generations(items):
    """Log (record, messages) pairs of batch results to Langfuse as generations, then flush."""
    if LANGFUSE is None:
        return
    for record, messages in items:
        s = MODEL_SETTINGS.get(bare_model(record['model']), {})
        model_parameters = {'max_tokens': max_tokens(record['model'], record['dataset']),
                            **{k: s[k] for k in ('temperature', 'effort', 'reasoning_effort') if k in s}}
        usage = {'input': record['input_tokens'] - record['cache_read_tokens'] - record['cache_write_tokens'],
                 'input_cache_read': record['cache_read_tokens'],
                 'input_cache_creation': record['cache_write_tokens'],
                 'output': record['output_tokens']}
        cost = {'total': record['cost_usd']} if record['cost_usd'] is not None else None
        with propagate_attributes(
                session_id=record['run_id'],
                trace_name=f"{record['dataset']}/k{record['k']}",
                tags=[record['model'], record['dataset'], f"k={record['k']}", record['mode']],
                metadata={'dataset': record['dataset'], 'k': record['k'], 'row': record['row']}):
            with LANGFUSE.start_as_current_observation(
                    name='predict', as_type='generation', model=record['model'],
                    input=messages, output=record['text'], model_parameters=model_parameters,
                    usage_details=usage, cost_details=cost,
                    metadata={'status': record['status'], 'predictions': record['predictions'],
                              'stop_reason': record['stop_reason'], 'custom_id': record['custom_id']},
                    level='ERROR' if record['status'] == 'api_error' else 'DEFAULT',
                    status_message=record['error']):
                pass
    LANGFUSE.flush()


def log_scores(run_id, metrics):
    """Post each (metric, dataset, model, k) score of a metrics DataFrame to the run's Langfuse session, then flush."""
    if LANGFUSE is None:
        return
    for m in metrics.to_dict('records'):
        if m['value'] != m['value']:  # NaN
            continue
        LANGFUSE.create_score(
            name=f"{m['metric']}/{m['dataset']}/{m['model']}/k{m['k']}", value=float(m['value']),
            session_id=run_id, data_type='NUMERIC',
            comment=f"n={m['n']}, parse_fail_rate={m['parse_fail_rate']:.3f}, cost_usd={m['cost_usd']}")
    LANGFUSE.flush()
