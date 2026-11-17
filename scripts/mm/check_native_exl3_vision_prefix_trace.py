#!/usr/bin/env python3
"""Check native A/A/B/A or history requests and partial-image prefix restore.

Consumes an isolated C1 target-only request report and fresh diagnostic server
log. Checks actual restore events and submitted metadata, not numerical tower
parity. The generic cached-prompt counter remains deferred in output_processor.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from native_vision_warm_owners import check_warm_sequence
from native_vision_cache_trace import cache_record
import struct


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(values):
    return hashlib.sha256(struct.pack('<' + str(len(values)) + 'i', *values)).hexdigest()


def run(args, result):
    report = json.loads(args.requests.read_text())
    require(report['status'] == 'PASS', 'request run failed')
    execution = json.loads(args.execution.read_text()) if args.execution else report
    if args.execution:
        require(execution['exit_code'] == 0 and execution['server_removed'] and execution['production_stopped'],
                'missing actual execution/cleanup')
        require(execution['requests_sha256'] == hashlib.sha256(args.requests.read_bytes()).hexdigest(),
                'wrong supervised request report')
    cases = report['cases']
    history = args.history
    labels = ['history-cold', 'history-warm'] if history else ['A-cold', 'A-warm', 'B-cold', 'A-warm-after-B']
    assets = ['orbit.png'] * 2 if history else ['orbit.png', 'orbit.png', 'comet.png', 'orbit.png']
    warm_owner = report.get('warm_owner', False)
    require(type(warm_owner) is bool and not (history and warm_owner), 'invalid warm-owner sequence mode')
    if warm_owner:
        labels += [f'warm-wave-{repeat}-{phase}' for repeat in range(4) for phase in range(4)]
        assets *= 5
    require([c['label'] for c in cases] == labels, 'wrong ordered request wave')
    require([c['asset'] for c in cases] == assets,
            'unexpected images')
    raw = args.trace_log.read_text()
    frames, ids = [], []
    for line in raw.splitlines():
        match = re.search(r'INFO Received request (chatcmpl-\d+) endpoint=', line)
        if match:
            ids.append(match.group(1))
            frames.append([])
        if line.startswith('NATIVE_VISION_EMBED '):
            require(frames, 'unattributed metadata')
            frames[-1].append(json.loads(line.split(' ', 1)[1]))
    require(len(frames) == len(labels) and len(set(ids)) == len(labels), 'fresh isolated request log required')
    if history and args.reference.name == 'chunk-mrope-reference.json':
        args.reference = args.reference.with_name('history-mrope-reference.json')
    reference_data = json.loads(args.reference.read_text())
    ref = next(c for c in reference_data['cases'] if c['offset'] == 1599)
    if history and args.execution:
        require(report['mode'] == 'history' and hashlib.sha256(report['history_assistant_text'].encode()).hexdigest() ==
                reference_data['assistant_text_sha256'], 'frozen conversation history differs')
        tokens = report['tokenized_pre_expansion']
        require(tokens.count(248056) == 1 and tokens.index(248056) == 1599, 'wrong history marker')
        expanded = tokens[:1599] + [248056] * 192 + tokens[1600:]
        require(digest(expanded) == report['expanded_prompt_sha256'] == ref['tokens_sha256'],
                'history prompt differs from executed reference')
    cold_indices, warm_indices = ((0,), (1,)) if history else ((0, 2), (1, 3))
    if warm_owner:
        warm_indices += tuple(range(4, 20))
    for index in cold_indices:
        axes = [[], [], []]
        start = 0
        for frame in frames[index]:
            size = frame['tokens']
            require(0 < size <= 1600, 'wrong batch budget')
            end = start + size
            require(len(frame['mrope']) == 3 * size, 'wrong position shape')
            for axis in range(3):
                for column, value in enumerate(frame['mrope'][axis * size:(axis + 1) * size]):
                    logical = start + column
                    if logical < ref['prompt_tokens']:
                        axes[axis].append(value)
                    else:
                        require(value == logical + ref['delta'], 'wrong completion position')
            require(frame['image_mask'] == [int(1599 <= pos < 1791) for pos in range(start, end)],
                    'wrong visual mask')
            lo, hi = max(start, 1599), min(end, 1791)
            require(frame['source_slices'] == ([[lo - 1599, hi - lo]] if hi > lo else []),
                    'wrong encoder rows')
            start = end
        require(all(len(axis) == ref['prompt_tokens'] for axis in axes), 'incomplete prompt')
        require(digest([v for axis in axes for v in axis]) == ref['positions_sha256'],
                'positions differ from executed oracle')
        require(start == ref['prompt_tokens'] + cases[index]['response']['usage']['completion_tokens'] - 1,
                'wrong logical context')
    for index in warm_indices:
        base = 0 if assets[index] == 'orbit.png' else 2
        require(frames[index] == frames[base][1:], 'restored suffix metadata differs')
        require(frames[index][0]['source_slices'] == [[1, 191]], 'partial visual restore missing')
        require(cases[index]['text'] == cases[base]['text'] and
                cases[index]['response']['usage'] == cases[base]['response']['usage'],
                'warm native answer/usage changed')
    if not history:
        require(cases[0]['text'] != cases[2]['text'], 'different image ignored')
    for index, case in enumerate(cases):
        require(case['hit_delta'] == (1600 if index in warm_indices else 0), 'wrong prefix hit metric')
        require(all(math.isfinite(value) and value >= 0 for value in case['after'].values()),
                'invalid cache/drain metrics')
        if args.execution:
            require(case['hit_delta'] == case['after']['vllm:prefix_cache_hits'] - case['before']['vllm:prefix_cache_hits'],
                    'prefix-hit delta differs from executed counters')
            require(case['cached_delta'] == case['after']['vllm:prompt_tokens_cached'] - case['before']['vllm:prompt_tokens_cached'],
                    'cached-prompt diagnostic delta differs from executed counters')
            require(case['response']['id'] == ids[index], 'response/trace request identity mismatch')
        require(case['after']['vllm:num_requests_running'] == 0 and
                case['after']['vllm:num_requests_waiting'] == 0, 'scheduler not drained')
    enc = re.findall(r'NATIVE_VISION_ENCODE count=(\d+) hash=([0-9a-f]{64}) rows=(\d+)', raw)
    require(len(enc) == len(cold_indices) and [e[0] for e in enc] == [str(i + 1) for i in range(len(cold_indices))] and
            all(e[2] == '192' for e in enc) and len({e[1] for e in enc}) == len(cold_indices), 'encoder reuse/isolation failed')
    restored = re.findall(r'PREFIX_RESTORE request=([^ ]+) tokens=(\d+) slot=(\d+)', raw)
    require([(e[0], e[1]) for e in restored] == [(ids[i], '1600') for i in warm_indices],
            'actual recurrent restore events missing or attributed incorrectly')
    if args.execution:
        idle = [record for line in raw.splitlines()
                if (record := cache_record(line)) is not None and record['event'] == 'idle']
        require(idle == report['idle'] and len(idle) == len(labels), 'missing executed idle-owner witnesses')
        require(all(0 < e['backend_allocated_bytes'] <= e['backend_peak_allocated_bytes'] for e in idle),
                'invalid idle memory accounting')
        if warm_owner:
            observed = check_warm_sequence(idle, 4, 4)
            require(report['warm_owner_observations'] == observed, 'incorrect warm sequence phase report')
            result['warm_owner_observations'] = observed
    else:
        require(not warm_owner, 'warm sequence requires actual supervisor/cleanup receipt')
    result.update(encoder_submissions=enc, prefix_restores=restored,
                  request_mode='history' if history else 'image-swap',
                  cold_positions_exact=True, warm_suffix_exact=True,
                  execution_head=execution['head'], binary_sha256=execution['binary_sha256'],
                  telemetry_limitation='generic cached-prompt counter remains deferred; '
                    'prefix_cache_hits and actual restore/metadata events are checked')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--trace-log', type=Path, required=True)
    parser.add_argument('--execution', type=Path, help='actual supervisor/cleanup receipt bound to request-report SHA-256')
    parser.add_argument('--history', action='store_true', help='qualify the separate cold/warm image-history wave')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[2] /
                        'tests/fixtures/native_vision_http/chunk-mrope-reference.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve earlier evidence')
    result = {'status': 'FAIL', 'scope': 'native aligned prefix restore; not tower numerical parity'}
    code = 1
    try:
        run(args, result)
        result['status'] = 'PASS'
        code = 0
    except Exception as error:
        result['error'] = repr(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
