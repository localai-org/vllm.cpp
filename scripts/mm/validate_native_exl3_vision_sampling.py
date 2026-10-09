#!/usr/bin/env python3
"""Bounded sampled image facts, SSE, usage and speculation on an existing server.

This stdlib client performs no service management. Seeds identify cases; it
does not require off/MTP3 or Python/native sampled token equality and does not
qualify RNG distributions. Use an otherwise idle native image server.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.request


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def score(text, fixture):
    # Score distinct image facts in named fields, not keywords anywhere in prose.
    clean = text.strip()
    if clean.startswith('```') and clean.endswith('```'):
        clean = re.sub(r'^```(?:json)?\s*', '', clean)[:-3].strip()
    answer = json.loads(clean)
    require(isinstance(answer, dict) and set(answer) == {'heading', 'left', 'right'},
            'expected three named image facts')
    require(all(isinstance(value, str) for value in answer.values()), 'nonstring image fact')
    heading = ' '.join(answer['heading'].lower().split())
    require(heading == ' '.join(fixture['expected'][:2]), 'incorrect heading: ' + heading)
    for side in ('left', 'right'):
        words = re.findall(r'[a-z]+', answer[side].lower())
        expected = fixture['sides'][side]
        other = fixture['sides']['right' if side == 'left' else 'left']
        require(all(word in words for word in expected), 'incorrect ' + side + ': ' + answer[side])
        require(other[0] not in words, 'contradictory color on ' + side)
    return answer


def run(args, report):
    url = args.url.rstrip('/')

    def metrics():
        with urllib.request.urlopen(url + '/metrics', timeout=5) as response:
            raw = response.read().decode()
        values = {}
        for line in raw.splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            key = line.split('{', 1)[0].split(' ', 1)[0]
            if key not in ('vllm:num_requests_running', 'vllm:num_requests_waiting') \
                    and not key.startswith('vllm:spec_decode_num_draft_tokens'):
                continue
            value = float(line.rsplit(' ', 1)[-1])
            require(math.isfinite(value) and value >= 0, 'invalid metric: ' + key)
            values[key] = values.get(key, 0) + value
        for key in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
            require(key in values and values[key] == 0, 'missing or nonzero drain gauge: ' + key)
        return values

    fixtures = json.loads((args.fixtures / 'fixtures.json').read_text())
    selected = [f for f in fixtures if f['name'] in ('orbit', 'comet', 'comet-unaligned')]
    require(len(selected) == 3, 'missing three frozen image fixtures')
    before = metrics()
    report['initial_metrics'] = before
    prompt = ('Read the heading exactly. Identify the color and shape on the left and on the '
              'right. Respond only with a JSON object with three string fields: heading, '
              'left, right. Do not add any other fields or explanation.')
    for fixture in selected:
        data = (args.fixtures / fixture['file']).read_bytes()
        require(hashlib.sha256(data).hexdigest() == fixture['sha256'], 'fixture changed')
        uri = 'data:' + fixture['media_type'] + ';base64,' + base64.b64encode(data).decode()
        for seed, stream in ((17, False), (29, True)):
            body = {'model': args.model, 'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': uri}},
                {'type': 'text', 'text': prompt}]}], 'temperature': 0.7,
                'top_p': 0.95, 'top_k': 20, 'seed': seed, 'max_tokens': 96,
                'stream': stream, 'chat_template_kwargs': {'enable_thinking': False}}
            if stream:
                body['stream_options'] = {'include_usage': True}
            case = {'fixture': fixture, 'seed': seed, 'stream': stream,
                    'sampling': {key: body[key] for key in ('temperature', 'top_p', 'top_k', 'seed', 'max_tokens')}}
            report['cases'].append(case)
            request = urllib.request.Request(url + '/v1/chat/completions',
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'Connection': 'close'})
            with urllib.request.urlopen(request, timeout=120) as response:
                require(response.status == 200, 'HTTP request failed')
                raw = response.read().decode()
                if stream:
                    require('text/event-stream' in response.headers.get('Content-Type', ''), 'wrong SSE media type')
            case['raw_response'] = raw
            if stream:
                require(raw.count('data: [DONE]') == 1, 'missing or duplicate SSE completion')
                parts = [json.loads(line[6:]) for line in raw.splitlines()
                         if line.startswith('data: ') and line[6:] != '[DONE]']
                require(parts and len({p['id'] for p in parts}) == 1, 'inconsistent SSE request identity')
                require(all(p['model'] == args.model for p in parts), 'wrong SSE model')
                choices = [c for part in parts for c in part.get('choices', [])]
                require(all(c['index'] == 0 for c in choices), 'wrong SSE choice index')
                text = ''.join(c['delta'].get('content', '') or '' for c in choices)
                finishes = [c['finish_reason'] for c in choices if c.get('finish_reason')]
                usages = [p['usage'] for p in parts if p.get('usage')]
                require(len(finishes) == len(usages) == 1, 'incorrect SSE finish/usage count')
                finish, usage = finishes[0], usages[0]
            else:
                response = json.loads(raw)
                require(response['model'] == args.model and len(response['choices']) == 1, 'wrong response model/choices')
                choice = response['choices'][0]
                require(choice['index'] == 0, 'wrong response choice index')
                text, finish, usage = choice['message']['content'], choice['finish_reason'], response['usage']
            case.update(text=text, finish_reason=finish, usage=usage)
            require(all(type(usage.get(key)) is int for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')),
                    'invalid usage types')
            require(usage['prompt_tokens'] > 192 and 0 < usage['completion_tokens'] <= 96, 'invalid token usage')
            require(usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'incorrect total usage')
            require(finish in ('stop', 'length'), 'abnormal finish')
            require(finish == 'stop' or usage['completion_tokens'] == 96, 'premature length finish')
            case['facts'] = score(text, fixture)
            case['metrics'] = metrics()
            print(json.dumps({'fixture': fixture['name'], 'seed': seed, 'stream': stream,
                              'facts': case['facts'], 'usage': usage}), flush=True)
    after = metrics()
    report['final_metrics'] = after
    keys = {key for key in before | after if key.startswith('vllm:spec_decode_num_draft_tokens')}
    require(all(after.get(key, 0) >= before.get(key, 0) for key in keys), 'speculation counter reset')
    report['draft_tokens'] = sum(after.get(key, 0) - before.get(key, 0) for key in keys)
    if args.spec_depth:
        require(keys and report['draft_tokens'] > 0, 'missing actual proposals')
    else:
        require(report['draft_tokens'] == 0, 'unexpected speculation in target-only run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--spec-depth', required=True, type=int, choices=(0, 3),
                        help='declared server recipe; positive counters do not independently prove depth')
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[2] /
                        'tests/fixtures/native_vision_http')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve previous evidence')
    report = {'status': 'FAIL', 'scope': 'C1 sampled image facts/SSE/usage; not RNG or numerical parity',
              'declared_spec_depth': args.spec_depth, 'cases': []}
    code = 1
    try:
        run(args, report)
        report['status'] = 'PASS'
        code = 0
    except Exception as error:
        report['error'] = repr(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'draft_tokens', 'error')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
