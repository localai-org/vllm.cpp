#!/usr/bin/env python3
"""Six frozen held-out image tasks and one repeated live SSE request.

Use a separately started otherwise idle native C1 image server. Stdlib only;
no service management or inference fallback. Reference answers are captured
first; this client scores specific named facts, not Python token equality.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import urllib.request
from native_exl3_vision_facts import qualification_inputs, evaluate


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(args, report):
    root = args.fixtures
    tasks, manifest_name, reference_name = qualification_inputs(root, args.resize_screenshot)
    reference = json.loads((root / reference_name).read_text())
    manifest_hash = hashlib.sha256((root / manifest_name).read_bytes()).hexdigest()
    require(reference['status'] == 'PASS' and reference['manifest_sha256'] == manifest_hash,
            'missing matching executed reference answers')
    require(len(reference['cases']) == len(tasks) and
            {c['name'] for c in reference['cases']} == {t['name'] for t in tasks}, 'incomplete reference matrix')
    require(all(c['finish_reason'] == 'stop'
                and evaluate(c['reference_text'], t)['pass']
                for t in tasks for c in reference['cases'] if c['name'] == t['name']), 'reference facts/hash failed')
    if args.resize_screenshot:
        require(reference['cases'][0]['image_sha256'] == tasks[0]['sha256'], 'reference resize image differs')
        actual = reference['actual_tower_input']
        require(actual['grid_thw'] == [1, 40, 54] and actual['pixels_shape'] == [2160, 1536]
                and actual['pixels_dtype'] == 'torch.float16' and actual['device'].startswith('xpu'),
                'missing actual oracle resize witness')
    report['manifest_sha256'] = manifest_hash
    report['reference_image'] = reference['reference_image']
    url = args.url.rstrip('/')

    def metrics():
        with urllib.request.urlopen(url + '/metrics', timeout=5) as response:
            raw = response.read().decode()
        result = {}
        for line in raw.splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            key = line.split('{', 1)[0].split(' ', 1)[0]
            if key not in ('vllm:num_requests_running', 'vllm:num_requests_waiting') \
                    and not key.startswith('vllm:spec_decode_num_draft_tokens'):
                continue
            value = float(line.rsplit(' ', 1)[-1])
            require(math.isfinite(value) and value >= 0, 'invalid metric: ' + key)
            result[key] = result.get(key, 0) + value
        require(all(k in result and result[k] == 0 for k in
                    ('vllm:num_requests_running', 'vllm:num_requests_waiting')), 'missing or nonzero drain gauges')
        return result

    before = metrics()
    report['initial_metrics'] = before
    dialog_index = 0 if args.resize_screenshot else 1
    dialog = tasks[dialog_index]
    for task, stream in [(t, False) for t in tasks] + [(dialog, True)]:
        pixels = (root / task['file']).read_bytes()
        uri = 'data:' + task['media_type'] + ';base64,' + base64.b64encode(pixels).decode()
        body = {'model': args.model, 'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': uri}}, {'type': 'text', 'text': task['prompt']}]}],
            'temperature': 0.0, 'max_tokens': 128, 'stream': stream,
            'chat_template_kwargs': {'enable_thinking': False}}
        if stream:
            body['stream_options'] = {'include_usage': True}
        case = {'name': task['name'], 'image_sha256': task['sha256'], 'stream': stream}
        report['cases'].append(case)
        request = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Connection': 'close'})
        with urllib.request.urlopen(request, timeout=120) as response:
            require(response.status == 200, 'HTTP request failed')
            raw = response.read().decode()
            if stream:
                require('text/event-stream' in response.headers.get('Content-Type', ''), 'wrong SSE content type')
        case['raw_response'] = raw
        if stream:
            require(raw.count('data: [DONE]') == 1, 'missing/duplicate SSE completion')
            parts = [json.loads(line[6:]) for line in raw.splitlines()
                     if line.startswith('data: ') and line[6:] != '[DONE]']
            require(parts and len({p['id'] for p in parts}) == 1, 'wrong SSE identity')
            require(all(p['model'] == args.model for p in parts), 'wrong SSE model')
            choices = [c for part in parts for c in part.get('choices', [])]
            require(all(c['index'] == 0 for c in choices), 'wrong SSE choice index')
            text = ''.join(c['delta'].get('content', '') or '' for c in choices)
            finishes = [c['finish_reason'] for c in choices if c.get('finish_reason')]
            usages = [p['usage'] for p in parts if p.get('usage')]
            require(len(finishes) == len(usages) == 1, 'wrong SSE finish/usage count')
            finish, usage, identity = finishes[0], usages[0], parts[0]['id']
        else:
            response = json.loads(raw)
            require(response['model'] == args.model and len(response['choices']) == 1, 'wrong model/choices')
            choice = response['choices'][0]
            require(choice['index'] == 0, 'wrong choice index')
            text, finish, usage, identity = choice['message']['content'], choice['finish_reason'], response['usage'], response['id']
        case.update(text=text, usage=usage, finish_reason=finish, request_id=identity)
        require(all(type(usage.get(k)) is int for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')), 'invalid usage types')
        require(usage['prompt_tokens'] > 192 and 0 < usage['completion_tokens'] <= 128
                and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'invalid token usage')
        require(finish == 'stop', 'expected complete finite fact answer/EOS')
        case['score'] = evaluate(text, task)
        case['metrics'] = metrics()
        print(json.dumps({'name': task['name'], 'stream': stream, 'text': text,
                          'score': case['score'], 'usage': usage}), flush=True)
    report['stream_matches_ordinary'] = report['cases'][dialog_index]['text'] == report['cases'][-1]['text'] \
        and report['cases'][dialog_index]['usage'] == report['cases'][-1]['usage']
    require(report['stream_matches_ordinary'], 'dialog repeat/SSE differs')
    after = metrics()
    report['final_metrics'] = after
    keys = {k for k in before | after if k.startswith('vllm:spec_decode_num_draft_tokens')}
    require(all(after.get(k, 0) >= before.get(k, 0) for k in keys), 'draft counter reset')
    report['draft_tokens'] = sum(after.get(k, 0) - before.get(k, 0) for k in keys)
    require(report['draft_tokens'] > 0 if args.spec_depth else report['draft_tokens'] == 0,
            'missing proposals or unexpected target-only speculation')
    require(all(case['score']['pass'] for case in report['cases']), 'held-out image facts failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--spec-depth', type=int, required=True, choices=(0, 3),
                        help='declared server recipe; counters alone do not certify depth')
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[2] /
                        'tests/fixtures/native_vision_qualification')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resize-screenshot', action='store_true', help='one ordinary and one SSE derived screenshot; original six-task default unchanged')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve prior evidence')
    report = {'status': 'FAIL', 'scope': 'six held-out native C1 image facts/SSE/usage; not tensor or performance parity',
              'declared_spec_depth': args.spec_depth, 'cases': []}
    if args.resize_screenshot:
        report['scope'] = 'one derived resize-required native C1 screenshot, ordinary/SSE facts/usage; not held-out or numerical parity'
    code = 1
    try:
        run(args, report)
        report['status'], code = 'PASS', 0
    except Exception as error:
        report['error'] = repr(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'draft_tokens', 'error')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
