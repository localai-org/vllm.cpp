#!/usr/bin/env python3
"""Bounded trace-free C1 image SSE timings using existing server metrics.

One text primer, then a cold image and three repeats. This client never starts
services. Encoder invocation/phase times and allocator peak are not exposed
by the current untraced metrics, and remain explicitly unmeasured here.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request


GAUGES = ('vllm:num_requests_running', 'vllm:num_requests_waiting')
TIMINGS = ('vllm:time_to_first_token_seconds', 'vllm:request_prefill_time_seconds',
           'vllm:request_decode_time_seconds', 'vllm:e2e_request_latency_seconds')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(args, report):
    url = args.url.rstrip('/')

    def api(path, body):
        request = urllib.request.Request(url + path, data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Connection': 'close'})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)

    def metrics():
        with urllib.request.urlopen(url + '/metrics', timeout=5) as response:
            raw = response.read().decode()
        result = {}
        for line in raw.splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            key = line.split('{', 1)[0].split(' ', 1)[0]
            value = float(line.rsplit(' ', 1)[-1])
            require(math.isfinite(value) and value >= 0, 'invalid metric: ' + key)
            result[key] = result.get(key, 0) + value
        for key in GAUGES + ('vllm:prefix_cache_hits_total',):
            require(key in result, 'missing metric: ' + key)
        for key in TIMINGS:
            require(key + '_sum' in result and key + '_count' in result, 'missing timing histogram: ' + key)
        return result

    fixture = next(f for f in json.loads((args.fixtures / 'fixtures.json').read_text()) if f['name'] == 'orbit')
    pixels = (args.fixtures / fixture['file']).read_bytes()
    require(hashlib.sha256(pixels).hexdigest() == fixture['sha256'], 'fixture changed')
    uri = 'data:' + fixture['media_type'] + ';base64,' + base64.b64encode(pixels).decode()
    report['fixture'] = fixture
    prompt = ('Read the large heading and number exactly. Then describe the color and shape on the left, '
              'followed by the color and shape on the right. Explain what is visible in a few complete English sentences.')
    prompt = ' z' * args.post_image_padding + prompt
    count = 1595
    for _ in range(8):
        prefix = ' x' * count
        tokens = api('/tokenize', {'model': args.model, 'messages': [{'role': 'user',
            'content': prefix + '<|vision_start|><|image_pad|><|vision_end|>' + prompt}],
            'add_special_tokens': True, 'chat_template_kwargs': {'enable_thinking': False}})['tokens']
        offset = tokens.index(248056)
        if offset == 1599:
            break
        count += 1599 - offset
        require(count >= 0, 'invalid padding')
    else:
        raise RuntimeError('cannot reach aligned image offset')
    length = len(tokens) + 191
    require(length == 1840 + args.post_image_padding, 'unexpected frozen prompt length')
    require(args.expected_prefix_hit_tokens < length, 'prefix hit must leave a prompt suffix')
    report.update(prompt_tokens=length, visual_placeholder_tokens=192,
                  nonvisual_prompt_tokens=length - 192, image_offset=1599,
                  post_image_padding=args.post_image_padding,
                  expected_warm_prefix_hit_tokens=args.expected_prefix_hit_tokens)

    # Prime text quantization/decoder graph work without putting this image or
    # its prefix in either cache. This primer is not an image timing sample.
    primer = api('/v1/chat/completions', {'model': args.model, 'messages': [{'role': 'user',
        'content': ' y' * 1700 + ' Write a short neutral description.'}], 'temperature': 0.,
        'max_tokens': 32, 'ignore_eos': True, 'chat_template_kwargs': {'enable_thinking': False}})
    require(primer['usage']['completion_tokens'] == 32, 'incomplete text primer')
    report['text_primer_usage'] = primer['usage']
    before = metrics()
    require(all(before[k] == 0 for k in GAUGES), 'server must be otherwise idle')
    initial = before
    for index in range(4):
        body = {'model': args.model, 'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': prefix}, {'type': 'image_url', 'image_url': {'url': uri}},
            {'type': 'text', 'text': prompt}]}], 'temperature': 0., 'max_tokens': 64,
            'ignore_eos': True, 'stream': True, 'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'enable_thinking': False}}
        request = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Connection': 'close'})
        parts, content_times, done = [], [], 0
        begin = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as response:
            require(response.status == 200 and 'text/event-stream' in response.headers.get('Content-Type', ''), 'SSE request failed')
            for rawline in response:
                at = time.perf_counter()
                line = rawline.decode().strip()
                if not line.startswith('data: '):
                    continue
                if line[6:] == '[DONE]':
                    done += 1
                    continue
                part = json.loads(line[6:])
                parts.append(part)
                if any(c.get('delta', {}).get('content') for c in part.get('choices', [])):
                    content_times.append(at)
        end = time.perf_counter()
        require(done == 1 and len(content_times) > 1, 'missing streamed content or completion')
        require(parts and len({p['id'] for p in parts}) == 1 and all(p['model'] == args.model for p in parts), 'wrong SSE identity/model')
        choices = [c for p in parts for c in p.get('choices', [])]
        require(all(c['index'] == 0 for c in choices), 'wrong SSE choice')
        text = ''.join(c['delta'].get('content', '') or '' for c in choices)
        usages = [p['usage'] for p in parts if p.get('usage')]
        finishes = [c['finish_reason'] for c in choices if c.get('finish_reason')]
        require(len(usages) == 1 and finishes == ['length'], 'wrong usage/finish count')
        usage = usages[0]
        require(usage == {'prompt_tokens': length, 'completion_tokens': 64, 'total_tokens': length + 64}, 'wrong usage')
        lower = text.lower()
        require(all(word in lower for word in fixture['expected']), 'image heading/shape facts failed')
        for side, words in fixture['sides'].items():
            start = lower.find(side)
            require(start >= 0 and all(word in lower[start:start + 100] for word in words), 'incorrect image side')
        after = metrics()
        # Metrics publication can trail the SSE finish. Observe it outside the
        # measured interval; no wait is added to model execution.
        deadline = time.monotonic() + 5
        while any(after[k + '_count'] - before[k + '_count'] < 1 for k in TIMINGS):
            require(time.monotonic() < deadline, 'request timing metrics did not publish')
            time.sleep(0.02)
            after = metrics()
        require(all(after[k] == 0 for k in GAUGES), 'scheduler did not drain')
        durations = {}
        for key in TIMINGS:
            require(after[key + '_count'] - before[key + '_count'] == 1, 'unexpected concurrent timing sample')
            delta = after[key + '_sum'] - before[key + '_sum']
            require(delta > 0, 'missing/nonpositive timing: ' + key)
            durations[key] = delta
        hits = after['vllm:prefix_cache_hits_total'] - before['vllm:prefix_cache_hits_total']
        expected_hit = args.expected_prefix_hit_tokens if args.prefix_cache == 'enabled' and index else 0
        seconds = content_times[-1] - content_times[0]
        decode = durations['vllm:request_decode_time_seconds']
        case = {'index': index, 'label': 'cold_image_after_text_primer' if index == 0 else
                ('repeat_with_target_prefix' if hits else 'repeat_without_target_prefix'),
                'text': text, 'usage': usage, 'prefix_hit_tokens': hits,
                'expected_prefix_hit_tokens': expected_hit,
                'remaining_prompt_tokens_from_prefix_hits': length - hits,
                'client_ttft_s': content_times[0] - begin, 'client_total_s': end - begin,
                'client_post_first_content_s': seconds, 'sse_content_chunks': len(content_times),
                'client_post_first_token_rate': 63 / seconds,
                'server_post_first_token_rate': 63 / decode, 'server_duration_deltas': durations}
        report['cases'].append(case)
        require(hits == expected_hit,
                f'prefix hit differs from declared recipe: observed {hits}, expected {expected_hit}')
        require(index == 0 or text == report['cases'][0]['text'], 'repeat text differs')
        print(json.dumps({k: case[k] for k in ('label', 'client_ttft_s', 'client_total_s',
              'server_post_first_token_rate', 'prefix_hit_tokens')}), flush=True)
        before = after
    keys = {k for k in initial | after if k.startswith('vllm:spec_decode_num_draft_tokens')}
    report['draft_tokens'] = sum(after.get(k, 0) - initial.get(k, 0) for k in keys)
    require(report['draft_tokens'] > 0 if args.spec_depth else report['draft_tokens'] == 0, 'unexpected/missing speculation')
    report['warm_medians'] = {key: statistics.median(c[key] for c in report['cases'][1:]) for key in
        ('client_ttft_s', 'client_total_s', 'client_post_first_token_rate', 'server_post_first_token_rate')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--spec-depth', type=int, required=True, choices=(0, 3))
    parser.add_argument('--prefix-cache', required=True, choices=('enabled', 'disabled'))
    parser.add_argument('--post-image-padding', type=int, default=0,
                        help='Number of single-token z repetitions after the image; preserves its offset')
    parser.add_argument('--expected-prefix-hit-tokens', type=int, default=1600,
                        help='Exact required warm hit for the declared enabled-cache recipe')
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[2] / 'tests/fixtures/native_vision_http')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.post_image_padding < 0 or args.expected_prefix_hit_tokens <= 0:
        parser.error('padding must be nonnegative and expected warm hit positive')
    if args.output.exists():
        parser.error('output exists; preserve previous evidence')
    report = {'status': 'FAIL', 'scope': 'bounded C1 HTTP timings and existing combined engine phases; not full phase or Python-speed parity',
              'declared_spec_depth': args.spec_depth, 'declared_prefix_cache': args.prefix_cache,
              'unmeasured': ['host image decode/resize', 'isolated encoder duration', 'isolated language prefill',
                             'encoder invocation/hit counts', 'backend/driver peak memory'],
              'cases': []}
    code = 1
    try:
        run(args, report)
        report['status'], code = 'PASS', 0
    except Exception as error:
        report['error'] = repr(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'warm_medians', 'draft_tokens', 'error')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
