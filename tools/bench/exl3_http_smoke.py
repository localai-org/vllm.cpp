#!/usr/bin/env python3
"""Check an explicitly launched EXL3 HTTP server; no model or engine imports."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import time
import traceback
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--url', required=True)
p.add_argument('--model', required=True)
p.add_argument('--mode', choices=['target', 'mtp3'], required=True)
p.add_argument('--out', required=True)
p.add_argument('--prompt-ids')
p.add_argument('--eos-id', type=int, default=248044)
p.add_argument('--drain-timeout', type=float, default=2.,
               help='Maximum seconds to observe scheduler teardown (0 < value <= 10)')
a = p.parse_args()
if not math.isfinite(a.drain_timeout) or not 0 < a.drain_timeout <= 10:
    p.error('--drain-timeout must be finite and in (0, 10]')
assert not Path(a.out).exists()
report = {'mode': a.mode, 'url': a.url, 'checks': [], 'drain_observations': [],
          'scope': 'Bounded HTTP behavior; simultaneous clients do not prove GPU overlap or physical slot identity. No Python-reference or maximum-context qualification.'}
SCHEDULER_GAUGES = ('vllm:num_requests_running', 'vllm:num_requests_waiting')

def check(name, **facts):
    report['checks'].append({'name': name, **facts})
    print(name, flush=True)

def open_req(path, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(a.url.rstrip('/')+path, data=data,
        headers={'Content-Type': 'application/json', 'Connection': 'close'})
    return urllib.request.urlopen(req, timeout=timeout)

def api(path, body=None):
    with open_req(path, body) as resp:
        assert resp.status == 200
        return json.load(resp)

def metrics(timeout=120):
    with open_req('/metrics', timeout=timeout) as resp:
        text = resp.read().decode()
    values = {}
    for line in text.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        name = line.split('{', 1)[0].split(' ', 1)[0]
        value = float(line.rsplit(' ', 1)[-1])
        if name in SCHEDULER_GAUGES:
            assert math.isfinite(value) and value >= 0, f'invalid scheduler gauge {name}: {value}'
        values[name] = values.get(name, 0.) + value
    for name in SCHEDULER_GAUGES:
        assert name in values, f'missing scheduler gauge {name}'
        assert math.isfinite(values[name]) and values[name] >= 0, f'invalid scheduler gauge {name}: {values[name]}'
    return values

def drained_metrics():
    started = time.monotonic()
    deadline = started+a.drain_timeout
    while True:
        remaining = deadline-time.monotonic()
        assert remaining > 0, 'scheduler drain deadline exceeded'
        values = metrics(timeout=remaining)
        report['drain_observations'].append({
            'unix_s': time.time(), 'elapsed_s': time.monotonic()-started,
            'gauges': {name: values[name] for name in SCHEDULER_GAUGES}})
        assert time.monotonic() <= deadline, 'scheduler drain deadline exceeded'
        if all(values[name] == 0 for name in SCHEDULER_GAUGES):
            return values
        remaining = deadline-time.monotonic()
        assert remaining > 0, 'scheduler drain deadline exceeded'
        time.sleep(min(.05, remaining))

def complete(prompt, cap=32, **extra):
    body = {'model': a.model, 'prompt': prompt, 'temperature': 0.,
            'max_tokens': cap, 'ignore_eos': True, 'logprobs': 0, **extra}
    started = time.monotonic()
    data = api('/v1/completions', body)
    assert data['model'] == a.model and len(data['choices']) == 1
    c = data['choices'][0]
    assert c['index'] == 0
    if body['ignore_eos']:
        assert data['usage']['completion_tokens'] == cap, data
        assert c['finish_reason'] == 'length', data
    assert len(c['logprobs']['tokens']) == data['usage']['completion_tokens'], data
    return {'response': data, 'wall_s': time.monotonic()-started}

def signature(result):
    c = result['response']['choices'][0]
    # MTP logprob positions may carry token_id placeholders; compare the
    # public text and usage, not those mixed display strings. Exact token-ID
    # continuity belongs to the separate native integration sentinel.
    return (c['text'], result['response']['usage']['completion_tokens'])

def stream(prompt, cap=32, cancel=False):
    body = {'model': a.model, 'prompt': prompt, 'temperature': 0.,
            'max_tokens': cap, 'ignore_eos': True, 'stream': True,
            'logprobs': 0, 'stream_options': {'include_usage': True}}
    started = time.monotonic()
    chunks, tokens, text, usage, done, first_s = [], [], '', None, False, None
    with open_req('/v1/completions', body) as resp:
        assert 'text/event-stream' in resp.headers.get('Content-Type', '')
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith('data: '):
                continue
            payload = line[6:]
            if payload == '[DONE]':
                done = True
                break
            d = json.loads(payload)
            assert d['model'] == a.model
            chunks.append(d)
            if d.get('usage') is not None:
                usage = d['usage']
            for c in d['choices']:
                assert c['index'] == 0
                text += c.get('text', '')
                t = (c.get('logprobs') or {}).get('tokens', [])
                tokens.extend(t)
                if c.get('text') and first_s is None:
                    first_s = time.monotonic()-started
                if cancel and c.get('text'):
                    return {'closed_after_first_text': True, 'first_s': first_s, 'generated_token_count': 'unavailable before terminal usage'}
    assert not cancel, 'cancel stream ended before any generated token'
    assert done and usage is not None, chunks
    assert usage['completion_tokens'] == cap, chunks
    if tokens:
        assert len(tokens) == cap, chunks
    assert any(c.get('finish_reason') == 'length' for d in chunks for c in d['choices']), chunks
    return {'text': text, 'tokens': tokens, 'usage': usage, 'sse_chunks': len(chunks),
            'first_s': first_s, 'wall_s': time.monotonic()-started}

try:
    with open_req('/health') as health:
        assert health.status == 200
    models = api('/v1/models')
    assert a.model in [x['id'] for x in models['data']]
    check('model-identification', models=models)
    before = metrics()
    prompts = [f'Request {name}. Continue this sentence with a short factual explanation of addition: one plus one equals' for name in ['alpha', 'bravo', 'charlie', 'delta']]
    baseline = [complete(prompt) for prompt in prompts]
    s = stream(prompts[0])
    assert s['text'] == signature(baseline[0])[0]
    assert s['usage']['completion_tokens'] == signature(baseline[0])[1]
    check('nonstream-stream-token-accounting', baseline=baseline[0], stream=s)
    for concurrency in [2, 3, 4]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            wave = list(pool.map(complete, prompts[:concurrency]))
        assert len({x['response']['id'] for x in wave}) == concurrency
        for actual, expected in zip(wave, baseline):
            assert signature(actual) == signature(expected), {'actual': actual, 'expected': expected}
        check(f'{concurrency}-concurrent-short-requests', wave=wave, same_greedy_text_and_usage_as_isolated=True)
    canceled = stream('Cancellation probe. Explain prime numbers in detail.', cap=256, cancel=True)
    assert canceled['closed_after_first_text']
    reused = complete(prompts[0])
    assert signature(reused) == signature(baseline[0])
    check('cancellation-and-followup-request', cancellation=canceled, followup=reused,
          physical_slot_identity='not observed by this HTTP client')
    eos = complete('End of sequence probe.', cap=16, ignore_eos=False,
                   allowed_token_ids=[a.eos_id])
    assert eos['response']['choices'][0]['finish_reason'] == 'stop', eos
    assert eos['response']['usage']['completion_tokens'] == 1, eos
    check('eos-stop', eos_id=a.eos_id, result=eos)
    after_eos = complete(prompts[0])
    assert signature(after_eos) == signature(baseline[0])
    check('eos-followup-request', result=after_eos)
    messages = [{'role': 'user', 'content': 'What is one plus one? Reply briefly.'}]
    templated = api('/tokenize', {'model': a.model, 'messages': messages,
                                 'chat_template_kwargs': {'enable_thinking': False}})
    assert templated['count'] > 0 and templated['tokens']
    chat = api('/v1/chat/completions', {'model': a.model, 'messages': messages,
        'temperature': 0., 'max_tokens': 32, 'ignore_eos': True,
        'chat_template_kwargs': {'enable_thinking': False}})
    assert chat['model'] == a.model and chat['choices'][0]['message']['role'] == 'assistant'
    assert chat['usage']['prompt_tokens'] == templated['count'], {'chat': chat, 'tokenized': templated}
    assert chat['usage']['completion_tokens'] == 32
    check('chat-template-real-model', tokenization=templated, response=chat)
    if a.prompt_ids:
        data = json.loads(Path(a.prompt_ids).read_text())
        ids = data['prompt_token_ids'] if 'prompt_token_ids' in data else data['cases'][0]['requests'][0]['prompt_ids']
        assert len(ids) == 4096
        prompt = api('/detokenize', {'model': a.model, 'tokens': ids})['prompt']
        encoded = api('/tokenize', {'model': a.model, 'prompt': prompt})
        assert encoded['tokens'] == ids, 'Provided 4K prompt does not round-trip through public tokenizer'
        m0 = metrics()
        cold = complete(prompt, cap=64)
        m1 = metrics()
        warm = complete(prompt, cap=64)
        m2 = metrics()
        assert signature(cold) == signature(warm)
        key = 'vllm:prompt_tokens_cached_total'
        if key not in m2:
            key = 'vllm:prompt_tokens_cached'
        cached = m2.get(key, 0.)-m1.get(key, 0.)
        assert cached > 0, {'before': m0, 'cold': m1, 'warm': m2}
        check('aligned-cold-warm-prefix', prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
              prompt_tokens=4096, cached_tokens_warm=cached, cold=cold, warm=warm)
    final = drained_metrics()
    draft_keys = {k for k in before.keys() | final.keys()
                  if k.startswith('vllm:spec_decode_num_draft_tokens')}
    for key in draft_keys:
        assert key in final, f'draft counter disappeared: {key}'
        assert all(math.isfinite(v) and v >= 0 for v in
                   (before.get(key, 0.), final[key])), f'invalid draft counter {key}'
        assert final[key] >= before.get(key, 0.), f'draft counter reset: {key}'
    draft_delta = sum(final[k]-before.get(k, 0.) for k in draft_keys)
    if a.mode == 'mtp3':
        assert draft_keys and draft_delta > 0, final
    else:
        assert draft_delta == 0, final
    check('speculation-counter-activity', requested_mode=a.mode, draft_token_delta=draft_delta,
          exact_draft_depth='not exposed by this server; verify the launch configuration separately',
          metrics_before=before, metrics_after=final)
    check('server-drained', metrics=final, deadline_s=a.drain_timeout)
    report['status'] = 'PASS'
except Exception as error:
    report['status'] = 'FAIL'
    report['error'] = repr(error)
    traceback.print_exc()
finally:
    with Path(a.out).open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
if report['status'] != 'PASS':
    raise SystemExit(1)
