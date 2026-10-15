#!/usr/bin/env python3
"""Check partial-image cancellation and three retries, or six for warm-owner mode.

Uses actual dev-mode abort/SSE results, submitted slices, idle ownership and
finite scheduler gauges. Does not qualify cancellation inside a specific GPU
kernel or asynchronous C4 overlap.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from native_vision_warm_owners import check_warm_owners
from native_vision_cache_trace import cache_record


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def gauges(values):
    require(set(values) == {'vllm:num_requests_running', 'vllm:num_requests_waiting'} and
            all(math.isfinite(v) and v == 0 for v in values.values()), 'missing/nonzero drain gauges')


def run(args, result):
    report = json.loads(args.requests.read_text())
    require(report['status'] == 'PASS', 'request run failed')
    execution = json.loads(args.execution.read_text()) if args.execution else report
    if args.execution:
        require(execution['exit_code'] == 0 and execution['server_removed'] and execution['production_stopped'],
                'missing actual execution/cleanup')
        require(execution['requests_sha256'] == hashlib.sha256(args.requests.read_bytes()).hexdigest(),
                'wrong supervised request report')
    require(report['abort'] == {'aborted': 1, 'status': 'aborted'}, 'abort missed live request')
    require(report['cancelled_usage'] == {'prompt_tokens': 1840, 'completion_tokens': 0, 'total_tokens': 1840},
            'cancellation happened after decode or lost usage')
    witness = report['pre_abort_log_tail']
    require('NATIVE_VISION_ENCODE count=1 ' in witness and 'status=done' not in witness,
            'encode/prefill witness missing')
    sse = report['cancelled_sse']
    require(sse.count('data: [DONE]') == 1, 'missing/duplicate cancelled SSE completion')
    frames = [json.loads(line[6:]) for line in sse.splitlines() if line.startswith('data: ') and line[6:] != '[DONE]']
    choices = [choice for frame in frames for choice in frame.get('choices', [])]
    require(all(not choice['delta'].get('content') for choice in choices), 'cancelled output contains generated text')
    require([c['finish_reason'] for c in choices if c.get('finish_reason')] == ['abort'], 'wrong finish event')
    gauges(report['post_abort_gauges'])
    contract = report.get('memory_contract', 'resident-exact')
    require(contract in ('resident-exact', 'graph-pool-owned', 'warm-owner'), 'unknown memory contract')
    retry_count = 6 if contract == 'warm-owner' else 3
    retries = report['followups']
    require(len(retries) == retry_count, 'wrong bounded successful retry count')
    for retry in retries:
        require(retry['response']['usage'] == {'prompt_tokens': 1840, 'completion_tokens': 64, 'total_tokens': 1904}, 'retry usage')
        require(all(word in retry['text'].lower() for word in ['orbit', '731', 'red', 'circle', 'blue', 'square']), 'retry image semantics')
        require(retry['text'] == retries[0]['text'], 'retry answer changed')
        gauges(retry['gauges'])
    raw = args.trace_log.read_text()
    groups = []
    for line in raw.splitlines():
        if re.search(r'INFO Received request chatcmpl-\d+ endpoint=', line):
            groups.append([])
        if line.startswith('NATIVE_VISION_EMBED '):
            require(groups, 'unattributed forward')
            groups[-1].append(json.loads(line.split(' ', 1)[1]))
    require(len(groups) == retry_count + 1 and len(groups[0]) == 1, 'cancelled request progressed beyond first chunk')
    require(groups[0][0]['tokens'] == 1600 and groups[0][0]['source_slices'] == [[0, 1]], 'wrong cancelled image slice')
    require(not re.search(r'INFO prefill id=chatcmpl-0 .*status=done', raw), 'cancelled prefill finished')
    require(len(re.findall(r'NATIVE_VISION_ENCODE count=\d+ ', raw)) == 1, 'retry encoded the cached image again')
    idle = [record for line in raw.splitlines()
            if (record := cache_record(line)) is not None and record['event'] == 'idle']
    require(len(idle) == retry_count + 1, 'idle owner release missing')
    require(all(e['entries'] == 1 and e['embedding_bytes'] == 1966080 and
                0 < e['backend_allocated_bytes'] <= e['backend_peak_allocated_bytes'] for e in idle), 'cache ownership/memory telemetry')
    active_bytes = None
    if contract == 'resident-exact':
        require(all(e['backend_allocated_bytes'] == idle[-1]['backend_allocated_bytes'] for e in idle[1:]), 'retry live memory grows')
    else:
        require(args.execution is not None and report['idle_records'] == idle, 'missing supervised pool observations')
        graph = [json.loads(line.split(' ', 1)[1]) for line in raw.splitlines()
                 if line.startswith('NATIVE_VISION_GRAPH ')]
        require(graph and all(g['requests'] == 1 and g['tokens'] == 1 for g in graph) and
                any(g['captured'] and g['replay_count'] > 1 for g in graph), 'missing actual C1 target graph replay')
        retries = idle[1:]
        for row in retries:
            require(isinstance(row['scratch_pool_retained_bytes'], int) and
                    0 <= row['scratch_pool_retained_bytes'] <= row['backend_allocated_bytes'], 'invalid free-pool accounting')
        active = [row['backend_allocated_bytes'] - row['scratch_pool_retained_bytes'] for row in retries]
        require(len(set(active)) == 1 and len({row['scratch_pool_live_blocks'] for row in retries}) == 1,
                'active retry owners grow')
        require(retries[0]['backend_graph_count'] > 0 and
                len({row['backend_graph_count'] for row in retries}) == 1 and
                len({row['backend_graph_device_bytes'] for row in retries}) == 1,
                'retry graph residency grows')
        for key in ('backend_allocated_bytes', 'scratch_pool_retained_bytes', 'scratch_pool_misses'):
            require(retries[-2][key] == retries[-1][key], 'warm total/free pool still grows')
        require(report['active_retry_backend_bytes'] == active[-1], 'incorrect owner baseline')
        active_bytes = active[-1]
        if contract == 'warm-owner':
            observed = check_warm_owners(retries)
            require(report['warm_owner_observations'] == observed, 'incorrect warm owner phase report')
            result['warm_owner_observations'] = observed
    result.update(encoder_submissions=1, cancelled_completion_tokens=0, successful_retries=retry_count,
                  stable_retry_backend_bytes=idle[-1]['backend_allocated_bytes'], resident_embedding_bytes=1966080,
                  peak_backend_bytes=max(e['backend_peak_allocated_bytes'] for e in idle),
                  execution_head=execution['head'], binary_sha256=execution['binary_sha256'])
    result.update(memory_contract=contract, active_retry_backend_bytes=active_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--trace-log', type=Path, required=True)
    parser.add_argument('--execution', type=Path, help='actual separate supervisor receipt for portable requests')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve earlier evidence')
    result = {'status': 'FAIL', 'scope': 'C1 cancellation/idle/retry; not MTP/C4/kernel-level abort parity'}
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
