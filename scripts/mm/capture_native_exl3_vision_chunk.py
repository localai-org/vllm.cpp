#!/usr/bin/env python3
"""Submit three frozen image-boundary requests to an unused C1 native server.

No server management or inference fallback. Actual positions/draft slices are
qualified separately by the trace checker. Trace wall times are not benchmarks.
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


def run(args, report):
    fixture = next(c for c in json.loads((args.fixtures / 'fixtures.json').read_text()) if c['name'] == 'orbit')
    image = (args.fixtures / fixture['file']).read_bytes()
    require(hashlib.sha256(image).hexdigest() == fixture['sha256'], 'frozen image changed')
    refs = {c['offset']: c for c in json.loads((args.fixtures / 'chunk-mrope-reference.json').read_text())['cases']}
    uri = 'data:image/png;base64,' + base64.b64encode(image).decode()
    url = args.url.rstrip('/')
    prompt = ('Read the large heading and number exactly. Then describe the color and shape on the left, '
              'followed by the color and shape on the right. Explain what is visible in a few complete English sentences.')

    def api(path, body):
        request = urllib.request.Request(url + path, data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Connection': 'close'})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError('HTTP ' + str(error.code) + ': ' + error.read().decode()) from error

    def metrics():
        with urllib.request.urlopen(url + '/metrics', timeout=5) as response:
            lines = response.read().decode().splitlines()
        values = {}
        for line in lines:
            if not line.strip() or line.startswith('#'):
                continue
            key = line.split('{', 1)[0].split(' ', 1)[0]
            if key in ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:spec_decode_num_draft_tokens_total'):
                value = float(line.rsplit(' ', 1)[-1])
                require(math.isfinite(value) and value >= 0, 'invalid metric')
                values[key] = values.get(key, 0) + value
        gauges = {k: values[k] for k in ('vllm:num_requests_running', 'vllm:num_requests_waiting') if k in values}
        require(len(gauges) == 2 and not any(gauges.values()), 'missing/nonzero drain gauges')
        return gauges, values.get('vllm:spec_decode_num_draft_tokens_total', 0)

    mode = {'budget': args.budget, 'cases': []}
    report['modes'].append(mode)
    mode['initial_gauges'], before_drafts = metrics()
    trace = CacheTrace(args.trace_log)
    try:
        require(not trace.read() and not trace.records, 'fresh unused server trace required')
        for index, offset in enumerate(range(args.boundary - 1, args.boundary + 2)):
            ref = refs[offset]
            count = offset - 4
            for _ in range(8):
                prefix = ' x' * count
                tokens = api('/tokenize', {'model': args.model, 'messages': [{'role': 'user',
                    'content': prefix + '<|vision_start|><|image_pad|><|vision_end|>' + prompt}],
                    'add_special_tokens': True, 'chat_template_kwargs': {'enable_thinking': False}})['tokens']
                observed = tokens.index(248056)
                if observed == offset:
                    break
                count += offset - observed
                require(count >= 0, 'invalid padding correction')
            else:
                raise RuntimeError('cannot reach frozen image offset')
            expanded = tokens[:offset] + [248056] * 192 + tokens[offset + 1:]
            digest = hashlib.sha256(struct.pack('<' + str(len(expanded)) + 'i', *expanded)).hexdigest()
            require(digest == ref['tokens_sha256'] and len(expanded) == ref['prompt_tokens'], 'frozen expanded prompt differs')
            body = {'model': args.model, 'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': prefix}, {'type': 'image_url', 'image_url': {'url': uri}},
                {'type': 'text', 'text': prompt}]}], 'temperature': 0., 'max_tokens': 64,
                'chat_template_kwargs': {'enable_thinking': False}}
            case = {'offset': offset, 'padding_items': count, 'tokenized_pre_expansion': tokens,
                    'expected_prompt_tokens': ref['prompt_tokens'], 'visual_placeholder_tokens': 192,
                    'request_sha256': hashlib.sha256(json.dumps(body).encode()).hexdigest()}
            mode['cases'].append(case)
            begin = time.monotonic()
            response = api('/v1/chat/completions', body)
            case.update(response=response, wall_s=time.monotonic() - begin)
            require(response['id'] == 'chatcmpl-' + str(index) and response['model'] == args.model, 'wrong fresh response identity/model')
            require(len(response['choices']) == 1 and response['choices'][0]['index'] == 0, 'wrong response choices')
            choice, usage = response['choices'][0], response['usage']
            case['text'] = choice['message']['content']
            require(usage['prompt_tokens'] == ref['prompt_tokens'] and 0 < usage['completion_tokens'] <= 64
                    and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'wrong image usage')
            require((choice['finish_reason'] == 'length' and usage['completion_tokens'] == 64) or
                    choice['finish_reason'] == 'stop', 'wrong completion finish')
            lower = case['text'].lower()
            case['missing_words'] = [word for word in fixture['expected'] if word not in lower]
            require(not case['missing_words'], 'incorrect image facts')
            for side, words in fixture['sides'].items():
                start = lower.find(side)
                require(start >= 0 and all(word in lower[start:start + 100] for word in words), 'incorrect image side')
            case['gauges'], after_drafts = metrics()
            deadline = time.monotonic() + 15
            while True:
                idle = trace.read()
                require(len(idle) <= index + 1, 'unexpected concurrent request')
                if len(idle) == index + 1:
                    case['device_idle'] = idle[-1]
                    break
                require(time.monotonic() < deadline, 'missing device-idle release')
                time.sleep(.05)  # Client-only observation after response, not a product wait.
            print(json.dumps({'case': index, 'budget': args.budget, 'offset': offset, 'usage': usage}), flush=True)
        report['draft_tokens'] = after_drafts - before_drafts
        require(report['draft_tokens'] > 0 if args.spec_depth else report['draft_tokens'] == 0, 'unexpected/missing actual speculation')
        mode['idle_cache'] = trace.read()
        require(len(mode['idle_cache']) == 3, 'missing release wave')
    finally:
        trace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--boundary', type=int, choices=[1600, 4096], required=True)
    parser.add_argument('--budget', type=int, choices=[1600, 4096], required=True, help='declared server prefill budget')
    parser.add_argument('--spec-depth', type=int, choices=[0, 3], required=True)
    parser.add_argument('--trace-log', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[2] / 'tests/fixtures/native_vision_http')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve prior evidence')
    report = {'status': 'FAIL', 'spec_depth': args.spec_depth, 'boundary': args.boundary, 'modes': [],
              'scope': 'three frozen C1 image boundary responses/release; independent trace/paired-ID checks required',
              'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    code = 1
    try:
        run(args, report)
        report['status'], code = 'PASS', 0
    except Exception as error:
        report['error'] = repr(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'draft_tokens': report.get('draft_tokens'), 'error': report.get('error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
