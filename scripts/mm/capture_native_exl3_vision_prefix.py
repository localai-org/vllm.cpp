#!/usr/bin/env python3
"""Submit frozen image-swap or conversation-history prefix prompts.

Use page/budget1600, prefix caching on, native trace2 and prefix snapshot trace.
Checks image semantics and the real prefix-hit counter; no service management or
Python learned inference. Actual restore/position proof belongs to the checker.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request

from capture_native_exl3_vision_max_eviction import CacheTrace, require
from native_vision_warm_owners import check_warm_sequence


def run(args, report):
    url = args.url.rstrip('/')
    fixtures = {f['name']: f for f in json.loads((args.fixtures / 'fixtures.json').read_text())}
    reference_file = args.fixtures / ('history-mrope-reference.json' if args.history else 'chunk-mrope-reference.json')
    reference_data = json.loads(reference_file.read_text())
    reference = next(c for c in reference_data['cases'] if c['offset'] == 1599)
    images = {}
    report['fixture_sha256'] = {}
    for name in ('orbit', 'comet'):
        fixture = fixtures[name]
        data = (args.fixtures / fixture['file']).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        require(digest == fixture['sha256'] and fixture['size'] == [512, 384], 'frozen fixture changed')
        report['fixture_sha256'][fixture['file']] = digest
        images[fixture['file']] = 'data:image/png;base64,' + base64.b64encode(data).decode()

    def api(path, body=None, raw=False, timeout=120):
        request = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Connection': 'close'})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode() if raw else json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError('HTTP ' + str(error.code) + ': ' + error.read().decode()) from error

    def metrics():
        wanted = ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:prefix_cache_hits',
                  'vllm:prefix_cache_queries', 'vllm:prompt_tokens_cached')
        values = {}
        for line in api('/metrics', raw=True, timeout=5).splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            key = line.split('{', 1)[0].split(' ', 1)[0].removesuffix('_total')
            if key in wanted:
                value = float(line.rsplit(' ', 1)[-1])
                require(math.isfinite(value) and value >= 0, 'invalid cache/lifecycle metric')
                values[key] = values.get(key, 0) + value
        require(set(values) == set(wanted), 'missing cache/drain metric')
        return values

    initial = metrics()
    require(initial['vllm:num_requests_running'] == initial['vllm:num_requests_waiting'] == 0,
            'unused drained server required')
    require(initial['vllm:prefix_cache_hits'] == initial['vllm:prompt_tokens_cached'] == 0, 'fresh cache counters required')
    report['initial_metrics'] = initial
    prefix = ' x' * 1595
    question = ('Read the large heading and number exactly. Then describe the color and shape on the left, '
                'followed by the color and shape on the right. Explain what is visible in a few complete English sentences.')
    tokenize_messages = [{'role': 'user', 'content': prefix + '<|vision_start|><|image_pad|><|vision_end|>' + question}]
    if args.history:
        prior = reference_data['assistant_text']
        require(hashlib.sha256(prior.encode()).hexdigest() == reference_data['assistant_text_sha256'], 'frozen history changed')
        tokenize_messages += [{'role': 'assistant', 'content': prior}, {'role': 'user', 'content': question}]
        report['history_assistant_text'] = prior
    tokens = api('/tokenize', {'model': args.model, 'messages': tokenize_messages,
        'add_special_tokens': True, 'chat_template_kwargs': {'enable_thinking': False}})['tokens']
    require(tokens.count(248056) == 1 and tokens.index(248056) == 1599, 'wrong frozen visual boundary')
    expanded = tokens[:1599] + [248056] * 192 + tokens[1600:]
    digest = hashlib.sha256(struct.pack('<' + str(len(expanded)) + 'i', *expanded)).hexdigest()
    require(digest == reference['tokens_sha256'] and len(expanded) == reference['prompt_tokens'],
            'prompt differs from executed CPU reference')
    report['expanded_prompt_sha256'] = digest
    report['tokenized_pre_expansion'] = tokens
    report['expected_prompt_tokens'] = reference['prompt_tokens']
    trace = CacheTrace(args.trace_log)
    try:
        require(not trace.read() and not trace.records, 'fresh unused server trace required')
        wave = (('history-cold', 'orbit', 0), ('history-warm', 'orbit', 1600)) if args.history else (
                ('A-cold', 'orbit', 0), ('A-warm', 'orbit', 1600), ('B-cold', 'comet', 0), ('A-warm-after-B', 'orbit', 1600))
        if args.warm_owner:
            wave += tuple((f'warm-wave-{repeat}-{phase}', name, 1600)
                          for repeat in range(4) for phase, name in enumerate(('orbit', 'orbit', 'comet', 'orbit')))
        for index, (label, name, hit) in enumerate(wave):
            fixture = fixtures[name]
            body = {'model': args.model, 'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': prefix}, {'type': 'image_url', 'image_url': {'url': images[fixture['file']]}},
                {'type': 'text', 'text': question}]}], 'temperature': 0., 'max_tokens': 64,
                'chat_template_kwargs': {'enable_thinking': False}}
            if args.history:
                body['messages'] += [{'role': 'assistant', 'content': prior}, {'role': 'user', 'content': question}]
            before = metrics()
            response = api('/v1/chat/completions', body)
            require(response['id'] == 'chatcmpl-' + str(index) and response['model'] == args.model and
                    len(response['choices']) == 1, 'wrong response owner/model/choices')
            require(response['usage'] == {'prompt_tokens': reference['prompt_tokens'], 'completion_tokens': 64,
                    'total_tokens': reference['prompt_tokens'] + 64} and
                    response['choices'][0]['finish_reason'] == 'length', 'wrong frozen response quota/finish')
            text = response['choices'][0]['message']['content']
            lower = text.lower()
            require(all(word in lower for word in fixture['expected']), 'incorrect image facts')
            for side, words in fixture['sides'].items():
                start = lower.find(side)
                require(start >= 0 and all(w in lower[start:start + 100] for w in words), 'incorrect image spatial facts')
            deadline = time.monotonic() + 15
            while True:
                idle = trace.read()
                after = metrics()
                require(len(idle) <= index + 1, 'unexpected concurrent release')
                if len(idle) == index + 1 and after['vllm:num_requests_running'] == after['vllm:num_requests_waiting'] == 0:
                    break
                require(time.monotonic() < deadline, 'missing drained device-owner release')
                time.sleep(.02)  # Diagnostic client observation, not a product wait.
            case = {'label': label, 'asset': fixture['file'], 'response': response, 'text': text,
                    'before': before, 'after': after, 'expected_cached': hit,
                    'hit_delta': after['vllm:prefix_cache_hits'] - before['vllm:prefix_cache_hits'],
                    'cached_delta': after['vllm:prompt_tokens_cached'] - before['vllm:prompt_tokens_cached']}
            report['cases'].append(case)
            require(case['hit_delta'] == hit, 'wrong observed prefix-hit tokens')
            if hit:
                base = report['cases'][0 if name == 'orbit' else 2]['response']
                require(response['choices'] == base['choices'] and response['usage'] == base['usage'],
                        'warm response changed')
            if index == 2:
                require(text != report['cases'][0]['text'], 'different same-shape image ignored')
            print(json.dumps({'label': label, 'hit_delta': case['hit_delta'], 'cached_delta': case['cached_delta']}), flush=True)
        report['idle'] = trace.read()
        require(len(report['idle']) == len(wave) and all(0 < e['backend_allocated_bytes'] <= e['backend_peak_allocated_bytes']
                for e in report['idle']), 'invalid/missing idle memory witnesses')
        if args.warm_owner:
            report['warm_owner_observations'] = check_warm_sequence(report['idle'], 4, 4)
    finally:
        trace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--trace-log', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[2] / 'tests/fixtures/native_vision_http')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history', action='store_true', help='cold/warm image in earlier message followed by pure text turn')
    parser.add_argument('--warm-owner', action='store_true',
                        help='image-swap only: establish a warm A/A/B/A wave and require three additional identical waves')
    args = parser.parse_args()
    if args.history and args.warm_owner:
        parser.error('warm-owner is bounded to the image-swap sequence')
    if args.output.exists():
        parser.error('output exists; preserve evidence')
    report = {'status': 'FAIL', 'cases': [], 'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'mode': 'history' if args.history else 'image-swap',
              'warm_owner': args.warm_owner,
              'scope': 'C1 target-only aligned image prefix restore/isolation; not eviction or numerical parity',
              'telemetry_limitation': 'output_processor generic cached-prompt counter is still deferred; prefix hits and actual restore metadata establish cache reuse'}
    code = 1
    try:
        run(args, report)
        report['status'], code = 'PASS', 0
    except Exception as error:
        report['error'] = repr(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'cases': len(report['cases']), 'error': report.get('error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
