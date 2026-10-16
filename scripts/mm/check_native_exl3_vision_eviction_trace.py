#!/usr/bin/env python3
"""Check a bounded native encoder eviction/re-encode wave.

The 14-request C1 workload fills the existing 2048-row cache with fixed-size
192-row outputs. Trace level2 exposes actual cache ownership and backend USM
allocation counts; retained pools/weights are part of the resident baseline.
"""
import argparse
import json
from pathlib import Path
import re
import math


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def run(args, result):
    report = json.loads(args.requests.read_text())
    require(report['status'] == 'PASS', 'request wave failed')
    cases = report['cases']
    expected = ['orbit.png', 'orbit.png', 'comet.png'] + [f'eviction-comet-{i}.png' for i in range(9)] + ['orbit.png', 'comet.png']
    require([c['asset'] for c in cases] == expected, 'wrong bounded request wave')
    for case in cases:
        usage = case['response']['usage']
        require(usage == {'prompt_tokens': 245, 'completion_tokens': 64, 'total_tokens': 309}, 'wrong usage')
        require(set(case['gauges']) == {'vllm:num_requests_running', 'vllm:num_requests_waiting'} and
                all(math.isfinite(v) and v == 0 for v in case['gauges'].values()), 'missing/draining gauges')
        words = ['orbit', '731', 'red', 'circle', 'blue', 'square'] if case['asset'] == 'orbit.png' else ['comet', '924', 'green', 'square', 'yellow', 'circle']
        require(all(word in case['text'].lower() for word in words), 'image semantic check failed')
    for first, repeat in ((0, 1), (0, 12), (2, 13)):
        require(cases[first]['text'] == cases[repeat]['text'], 'answer changed after reuse/eviction')
    raw = args.trace_log.read_text()
    require(len(re.findall(r'INFO Received request chatcmpl-\d+ endpoint=', raw)) == 14, 'fresh 14-request log required')
    enc = re.findall(r'NATIVE_VISION_ENCODE count=(\d+) hash=([0-9a-f]{64}) rows=(\d+)', raw)
    require(len(enc) == 13 and [e[0] for e in enc] == [str(i) for i in range(1, 14)] and
            all(e[2] == '192' for e in enc), 'missing/incorrect encode observations')
    require(len({e[1] for e in enc[:11]}) == 11 and enc[11][1] == enc[0][1] and
            enc[12][1] == enc[1][1], 'wrong image cache identities')
    records = [json.loads(line.split(' ', 1)[1]) for line in raw.splitlines() if line.startswith('NATIVE_VISION_CACHE ')]
    records = [e for e in records if e['event'] in ('insert', 'evict')]
    require([e['event'] for e in records] == ['insert'] * 10 + ['evict', 'insert'] * 3, 'actual eviction missing')
    resident = set()
    for record in records:
        if record['event'] == 'insert':
            require(record['hash'] not in resident, 'double insertion')
            resident.add(record['hash'])
        else:
            require(record['hash'] in resident, 'eviction of absent output')
            resident.remove(record['hash'])
        require(record['entries'] == len(resident) <= 10, 'cache entry ownership leak')
        require(record['embedding_bytes'] == len(resident) * 192 * 5120 * 2, 'cache embedding byte leak')
        require(0 < record['backend_allocated_bytes'] <= record['backend_peak_allocated_bytes'], 'missing/bad memory telemetry')
    inserts = [record for record in records if record['event'] == 'insert']
    require([record['hash'] for record in inserts] == [e[1] for e in enc], 'encode/cache insertion differs')
    warm_bytes = inserts[-1]['backend_allocated_bytes']
    require(all(e['backend_allocated_bytes'] == warm_bytes for e in inserts[-3:]), 'warm backend allocations grow')
    result.update(encoder_submissions=13, actual_evictions=3, resident_entries=len(resident),
                  resident_embedding_bytes=len(resident) * 192 * 5120 * 2,
                  stable_warm_backend_bytes=warm_bytes,
                  peak_backend_bytes=max(e['backend_peak_allocated_bytes'] for e in records),
                  execution_head=report['head'], binary_sha256=report['binary_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--trace-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve earlier evidence')
    result = {'status': 'FAIL', 'scope': 'bounded native eviction/USM ownership; excludes driver memory and numerical tower parity'}
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
