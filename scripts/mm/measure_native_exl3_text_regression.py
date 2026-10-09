#!/usr/bin/env python3
"""Bounded paired 4096-prompt/256-output text check; no service management.

Run once against a pinned baseline server, then a separate candidate server
with --baseline. Use C1, identical MTP/KV/chunk options, disabled prefix cache
and all trace/profiler flags off. Four samples follow a separate text primer.
The comparison checks text, usage and the per-token public logprobs spelling;
it does not claim raw token-ID or internal state equality from decoded strings.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request

from measure_native_exl3_vision_http import GAUGES, TIMINGS, require


def run(args, report):
    url = args.url.rstrip('/')

    def api(path, body):
        request = urllib.request.Request(url + path, data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Connection': 'close'})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'{path}: HTTP {error.code}: {error.read().decode()}') from error

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
            require(key + '_sum' in result and key + '_count' in result, 'missing histogram: ' + key)
        return result

    tail = '\nExplain why the sum of two even integers is even. Give a clear mathematical explanation:\n'
    count = 4070
    for _ in range(8):
        prompt = ' x' * count + tail
        tokens = api('/tokenize', {'model': args.model, 'prompt': prompt,
                                  'add_special_tokens': True})['tokens']
        if len(tokens) == 4096:
            break
        count += 4096 - len(tokens)
        require(count >= 0, 'invalid prompt padding')
    else:
        raise RuntimeError('cannot produce a 4096-token prompt')
    report['prompt_text_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
    report['prompt_token_ids'] = tokens
    primer = api('/v1/completions', {'model': args.model,
        'prompt': ' y' * 1700 + ' Write a short neutral description.',
        'temperature': 0., 'max_tokens': 32, 'ignore_eos': True})
    require(primer['usage']['completion_tokens'] == 32, 'incomplete primer')
    report['primer_usage'] = primer['usage']
    before = metrics()
    require(all(before[k] == 0 for k in GAUGES), 'server must be otherwise idle')
    initial = before
    for index in range(4):
        start = time.perf_counter()
        response = api('/v1/completions', {'model': args.model, 'prompt': prompt,
            'temperature': 0., 'max_tokens': 256, 'ignore_eos': True, 'logprobs': 0})
        wall = time.perf_counter() - start
        require(response['model'] == args.model and len(response['choices']) == 1, 'wrong response identity')
        choice = response['choices'][0]
        require(choice['index'] == 0 and choice['finish_reason'] == 'length', 'wrong finish')
        require(response['usage'] == {'prompt_tokens': 4096, 'completion_tokens': 256,
                                      'total_tokens': 4352}, 'wrong usage')
        require(choice['text'] and len(choice['logprobs']['tokens']) == 256, 'missing per-token output')
        after = metrics()
        deadline = time.monotonic() + 5
        while any(after[k + '_count'] - before[k + '_count'] < 1 for k in TIMINGS):
            require(time.monotonic() < deadline, 'timing metrics did not publish')
            time.sleep(.02)
            after = metrics()
        require(all(after[k] == 0 for k in GAUGES), 'scheduler did not drain')
        require(after['vllm:prefix_cache_hits_total'] == before['vllm:prefix_cache_hits_total'], 'prefix cache must be disabled')
        durations = {}
        for key in TIMINGS:
            require(after[key + '_count'] - before[key + '_count'] == 1, 'unexpected concurrent timing sample')
            durations[key] = after[key + '_sum'] - before[key + '_sum']
            require(durations[key] > 0, 'nonpositive timing')
        sample = {'index': index, 'text': choice['text'], 'token_representations': choice['logprobs']['tokens'],
                  'usage': response['usage'], 'client_total_s': wall,
                  'server_ttft_s': durations[TIMINGS[0]], 'server_prefill_s': durations[TIMINGS[1]],
                  'server_decode_s': durations[TIMINGS[2]], 'server_total_s': durations[TIMINGS[3]],
                  'server_post_first_token_rate': 255 / durations[TIMINGS[2]]}
        report['samples'].append(sample)
        require(index == 0 or all(sample[k] == report['samples'][0][k] for k in
                                 ('text', 'token_representations', 'usage')), 'repeat output differs')
        print(json.dumps({k: sample[k] for k in ('index', 'server_ttft_s',
                         'server_post_first_token_rate', 'client_total_s')}), flush=True)
        before = after
    keys = {k for k in initial | after if k.startswith('vllm:spec_decode_num_draft_tokens')}
    report['draft_tokens'] = sum(after.get(k, 0) - initial.get(k, 0) for k in keys)
    require(report['draft_tokens'] > 0 if args.spec_depth else report['draft_tokens'] == 0, 'unexpected/missing speculation')
    report['warm_medians'] = {k: statistics.median(s[k] for s in report['samples'][1:]) for k in
        ('client_total_s', 'server_ttft_s', 'server_prefill_s', 'server_decode_s',
         'server_total_s', 'server_post_first_token_rate')}
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        require(baseline['status'] == 'PASS' and baseline['declared_spec_depth'] == args.spec_depth,
                'baseline is incomplete or uses different speculation')
        require(baseline['prompt_token_ids'] == tokens, 'baseline prompt differs')
        require(len(baseline['samples']) == 4, 'baseline sample count differs')
        for prior, current in zip(baseline['samples'], report['samples']):
            require(all(prior[k] == current[k] for k in ('text', 'token_representations', 'usage')),
                    'candidate behavior differs from baseline')
        report['paired_behavior_equal'] = True
        report['baseline_sha256'] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
        report['candidate_baseline_duration_ratios'] = {k: v / baseline['warm_medians'][k]
            for k, v in report['warm_medians'].items() if k != 'server_post_first_token_rate'}
        report['performance_diagnostic_trigger'] = any(v > 1.05 for v in
            report['candidate_baseline_duration_ratios'].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--spec-depth', type=int, required=True, choices=(0, 3))
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve previous evidence')
    report = {'status': 'FAIL', 'declared_spec_depth': args.spec_depth,
              'scope': 'C1 trace-free 4K/O256 text behavior and timings; exact HTTP token representations, not raw-ID/state or Python parity',
              'samples': []}
    code = 1
    try:
        run(args, report)
        report['status'], code = 'PASS', 0
    except Exception as error:
        report['error'] = repr(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'warm_medians',
        'draft_tokens', 'paired_behavior_equal', 'performance_diagnostic_trigger', 'error')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
