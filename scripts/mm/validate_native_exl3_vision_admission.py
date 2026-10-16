#!/usr/bin/env python3
"""Bounded native vision admission/lifecycle check against an existing server.

Submit eight image/text clients to a C4 engine, witness queued clients, require
correct independent results and drain, then check two-image EOS and early
count refusal. No service management or inference fallback; stdlib only.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def run(args, report):
    url = args.url.rstrip('/')

    def api(path, body):
        request = urllib.request.Request(
            url + path, data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Connection': 'close'})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())

    def metrics():
        with urllib.request.urlopen(url + '/metrics', timeout=5) as response:
            raw = response.read().decode()
        values = {}
        for line in raw.splitlines():
            if line.startswith('#') or not line.strip():
                continue
            key = line.split('{', 1)[0].split(' ', 1)[0]
            value = float(line.rsplit(' ', 1)[-1])
            if not math.isfinite(value) or value < 0:
                raise RuntimeError('nonfinite or negative metric: ' + key)
            values[key] = values.get(key, 0) + value
        for key in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
            if key not in values:
                raise RuntimeError('missing drain gauge: ' + key)
        return values

    def drain():
        values = metrics()
        gauges = {key: values[key] for key in (
            'vllm:num_requests_running', 'vllm:num_requests_waiting')}
        if any(gauges.values()):
            raise RuntimeError('requests did not drain: ' + repr(gauges))
        return gauges

    fixtures = json.loads((args.fixtures / 'fixtures.json').read_text())
    selected = {f['file']: f for f in fixtures if f['file'] in ('orbit.png', 'comet.png')}
    if len(selected) != 2:
        raise RuntimeError('missing frozen image fixtures')
    parts = {}
    for name, fixture in selected.items():
        data = (args.fixtures / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != fixture['sha256']:
            raise RuntimeError('fixture changed: ' + name)
        parts[name] = {'type': 'image_url', 'image_url': {
            'url': 'data:' + fixture['media_type'] + ';base64,' + base64.b64encode(data).decode()}}
    report['fixtures'] = selected
    report['initial_gauges'] = drain()
    before = metrics()
    question = ('Read the large heading and number exactly. Then describe the color '
                'and shape on the left, followed by the color and shape on the right. '
                'Explain what is visible in a few complete English sentences.')
    cases = []
    for i in range(6):
        name = 'orbit.png' if i % 2 == 0 else 'comet.png'
        cases.append({'label': 'image-' + str(i), 'fixture': name,
                      'path': '/v1/chat/completions', 'body': {
                          'model': args.model, 'temperature': 0.0, 'max_tokens': 64,
                          'chat_template_kwargs': {'enable_thinking': False},
                          'messages': [{'role': 'user', 'content': [parts[name], {
                              'type': 'text', 'text': question}]}]}})
    for name, operation in (('alpha', 'addition'), ('beta', 'subtraction')):
        cases.append({'label': name, 'path': '/v1/completions', 'body': {
            'model': args.model, 'temperature': 0.0, 'max_tokens': 32,
            'ignore_eos': True, 'logprobs': 0,
            'prompt': 'Request ' + name + '. Give a short factual explanation of ' + operation + '.'}})
    barrier = threading.Barrier(len(cases))

    def execute(case):
        barrier.wait(timeout=10)
        response = api(case['path'], case['body'])
        choice = response['choices'][0]
        text = choice['message']['content'] if 'message' in choice else choice['text']
        result = {'label': case['label'], 'response': response, 'text': text}
        usage = response['usage']
        if usage['completion_tokens'] != case['body']['max_tokens']:
            raise RuntimeError('unexpected output length: ' + case['label'])
        if usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
            raise RuntimeError('incorrect usage: ' + case['label'])
        if 'fixture' in case:
            fixture = selected[case['fixture']]
            lower = text.lower()
            if any(word not in lower for word in fixture['expected']):
                raise RuntimeError('wrong image heading: ' + case['label'])
            for side, expected in fixture['sides'].items():
                start = lower.find(side)
                span = lower[start:start + 100] if start >= 0 else ''
                if any(word not in span for word in expected):
                    raise RuntimeError('wrong image side facts: ' + case['label'])
        elif not text.strip():
            raise RuntimeError('empty text result')
        return result

    report['observations'] = []
    report['cases'] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
        futures = [pool.submit(execute, case) for case in cases]
        deadline = time.monotonic() + 180
        while not all(f.done() for f in futures):
            if time.monotonic() > deadline:
                raise RuntimeError('bounded client observation deadline exceeded')
            snapshot = metrics()
            report['observations'].append({key: snapshot[key] for key in (
                'vllm:num_requests_running', 'vllm:num_requests_waiting')})
            time.sleep(0.02)  # Client telemetry sampling, not an engine wait.
        for future in futures:
            report['cases'].append(future.result())
    report['peak_running'] = max(s['vllm:num_requests_running'] for s in report['observations'])
    report['peak_waiting'] = max(s['vllm:num_requests_waiting'] for s in report['observations'])
    if report['peak_running'] != 4 or report['peak_waiting'] <= 0:
        raise RuntimeError('C4 queue admission was not witnessed')
    report['after_wave_gauges'] = drain()
    body = {'model': args.model, 'temperature': 0.0, 'max_tokens': 64,
            'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'user', 'content': [parts['orbit.png'], parts['comet.png'], {
                'type': 'text', 'text': "Read the large heading and number in each picture. List both in conversation order as 'First: ...' and 'Second: ...'."}]}]}
    eos = api('/v1/chat/completions', body)
    report['two_image_eos'] = eos
    choice = eos['choices'][0]
    if choice['finish_reason'] != 'stop' or not 0 < eos['usage']['completion_tokens'] < 64:
        raise RuntimeError('normal two-image EOS not witnessed')
    pairs = re.findall(r'\b(orbit|comet)\s*[:#-]?\s*(731|924)', choice['message']['content'].lower())
    if pairs != [('orbit', '731'), ('comet', '924')]:
        raise RuntimeError('two-image order incorrect')
    bad = json.loads(json.dumps(body))
    bad['messages'][0]['content'].insert(0, parts['orbit.png'])
    try:
        api('/v1/chat/completions', bad)
    except urllib.error.HTTPError as error:
        report['overlimit'] = {'http_status': error.code, 'body': error.read().decode()}
        if error.code != 400:
            raise RuntimeError('overlimit request did not return client error')
    else:
        raise RuntimeError('third image was accepted')
    # Retry through real generation, not merely a health check, after refusal.
    retry = api('/v1/chat/completions', body)
    report['retry'] = retry
    if retry['choices'] != eos['choices'] or retry['usage'] != eos['usage']:
        raise RuntimeError('retry changed ordered image response or usage')
    report['final_gauges'] = drain()
    after = metrics()
    report['draft_tokens'] = sum(after[k] - before.get(k, 0) for k in after
                                if k.startswith('vllm:spec_decode_num_draft_tokens'))
    if args.require_speculation and report['draft_tokens'] <= 0:
        raise RuntimeError('no actual speculative proposals')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[2] / 'tests/fixtures/native_vision_http')
    parser.add_argument('--require-speculation', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve evidence')
    report = {'status': 'FAIL', 'scope': 'eight-client C4 native image/text queue, two-image EOS/count/refusal/retry functional check; not exact greedy, numerical or performance parity'}
    result = 1
    try:
        run(args, report)
        report['status'] = 'PASS'
        result = 0
    except Exception as error:
        report['error'] = repr(error)
        print(report['error'], flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'peak_running', 'peak_waiting', 'draft_tokens', 'error')}), flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
