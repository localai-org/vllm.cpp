#!/usr/bin/env python3
"""Check actual native MTP3 two-image prefill and mixed C4 decode cancellation.

Consumes the executed request receipt and trace; never runs a model or manages
services. Requires GPU-owner idle witnesses, unaffected siblings, cache reuse
and stable repeat memory. Does not certify kernel interruption or numerics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from native_vision_cache_trace import cache_record


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def gauges(values, running=0):
    require(set(values) == {'vllm:num_requests_running', 'vllm:num_requests_waiting'}, 'missing scheduler gauges')
    require(all(math.isfinite(v) for v in values.values()), 'nonfinite gauges')
    require(values['vllm:num_requests_running'] == running and values['vllm:num_requests_waiting'] == 0, 'incorrect running/waiting state')


def cancelled(case, decoded):
    raw = case['raw_sse']
    require(raw.count('data: [DONE]') == 1, 'missing/duplicate cancelled SSE done')
    frames = [json.loads(line[6:]) for line in raw.splitlines()
              if line.startswith('data: ') and line[6:] != '[DONE]']
    require(all(frame['id'] == case['id'] for frame in frames), 'cancelled stream request identity')
    choices = [choice for frame in frames for choice in frame.get('choices', [])]
    require([c['finish_reason'] for c in choices if c.get('finish_reason')] == ['abort'], 'wrong final cancellation reason')
    text = ''.join(c['delta'].get('content', '') or '' for c in choices)
    require(text == case['text'] and bool(text) == decoded, 'wrong cancelled stream content')
    usage = [frame['usage'] for frame in frames if frame.get('usage')]
    require(usage == [case['usage']], 'missing/duplicate or incorrect SSE usage')
    count = case['usage']['completion_tokens']
    require((0 < count < 64) if decoded else count == 0, 'incorrect cancellation output count')
    require(case['usage']['total_tokens'] == case['usage']['prompt_tokens'] + count, 'cancelled usage sum')


def ordered(case):
    response = case['response']
    text = response['choices'][0]['message']['content']
    require(text == case['text'], 'stored text differs from API response')
    require(re.findall(r'\b(orbit|comet)\s*[:#-]?\s*(731|924)', text.lower()) ==
            [('orbit', '731'), ('comet', '924')], 'ordered image facts')
    usage = response['usage']
    require(0 < usage['completion_tokens'] <= 64 and usage['total_tokens'] ==
            usage['prompt_tokens'] + usage['completion_tokens'], 'retry usage')
    gauges(case['gauges'])


def run(args, result):
    report = json.loads(args.requests.read_text())
    require(report['status'] == 'PASS', 'request execution failed')
    execution = json.loads(args.execution.read_text()) if args.execution else report
    if args.execution:
        require(execution['exit_code'] == 0 and execution['server_removed'] and execution['production_stopped'],
                'missing actual execution/cleanup')
        require(execution['requests_sha256'] == hashlib.sha256(args.requests.read_bytes()).hexdigest(),
                'wrong supervised request report')
    require(report['cold_abort'] == report['decode_abort'] == {'aborted': 1, 'status': 'aborted'}, 'abort missed live request')
    cancelled(report['cold_cancel'], False)
    cancelled(report['decode_cancel'], True)
    require(report['cold_cancel']['usage']['prompt_tokens'] == 2022, 'cold two-image prompt')
    gauges(report['cold_post_abort_gauges'])
    gauges(report['decode_pre_abort_gauges'], running=4)
    gauges(report['mixed_post_abort_gauges'])
    require(any(g['requests'] == 4 and g['captured'] for g in report['decode_pre_abort_graphs']), 'actual C4 graph execution missing before abort')
    raw = args.trace_log.read_text()
    cold_id = report['cold_cancel']['id']
    retry_id = report['cold_retry']['response']['id']
    require('INFO Received request ' + retry_id + ' endpoint=' in raw, 'retry trace attribution')
    cold = raw.split('INFO Received request ' + retry_id + ' endpoint=', 1)[0]
    require('INFO prefill id=' + cold_id + ' status=begin prompt_tokens=2022' in cold, 'cold prefill not observed')
    require(not re.search(r'INFO prefill id=' + re.escape(cold_id) + r' .*status=done', cold), 'cold prefill finished before abort')
    frames = [json.loads(line.split(' ', 1)[1]) for line in cold.splitlines()
              if line.startswith('NATIVE_VISION_EMBED ')]
    require(len(frames) == 1, 'cold request progressed beyond first chunk')
    frame = frames[0]
    require(frame['tokens'] == 1600 and frame['source_slices'] == [[0, 1]] and
            frame['image_mask'] == [0] * 1599 + [1], 'cold partial image slice')
    require(frame['mrope'] == list(range(1600)) * 3, 'cold partial-image positions')
    draft_frames = [json.loads(line.split(' ', 1)[1]) for line in cold.splitlines()
                    if line.startswith('NATIVE_VISION_DRAFT_EMBED ')]
    require(len(draft_frames) == 1 and draft_frames[0]['tokens'] == 1600 and
            draft_frames[0]['image_mask'] == [0] * 1598 + [1, 1] and
            draft_frames[0]['source_slices'] == [[0, 2]], 'shifted partial-image draft inputs')
    cold_idle = [record for line in cold.splitlines()
            if (record := cache_record(line)) is not None and record['event'] == 'idle']
    require(cold_idle == [report['cold_idle_release']], 'cold GPU idle-release witness missing')
    require(cold_idle[0]['entries'] == 1 and cold_idle[0]['embedding_bytes'] == 1966080, 'unexpected cold cache ownership')
    trace_graphs = [json.loads(line.split(' ', 1)[1]) for line in raw.splitlines()
                    if line.startswith('NATIVE_VISION_GRAPH ')]
    require(all(g in trace_graphs for g in report['decode_pre_abort_graphs']), 'C4 witness differs from executed trace')
    siblings = report['unaffected_siblings']
    require(len(siblings) == 3, 'missing unaffected siblings')
    for i, sibling in enumerate(siblings):
        usage = sibling['usage']
        require(usage['completion_tokens'] == (64 if i == 0 else 96), 'sibling output quota')
        require(usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'sibling usage')
    lower = siblings[0]['choices'][0]['message']['content'].lower()
    require(all(w in lower for w in ['comet', '924']), 'sibling heading facts')
    for side, words in {'left': ['green', 'square'], 'right': ['yellow', 'circle']}.items():
        start = lower.find(side)
        span = lower[start:start + 100] if start >= 0 else ''
        require(all(w in span for w in words), 'sibling spatial facts')
    ordered(report['cold_retry'])
    require(len(report['followups']) == 3, 'three reuse repetitions required')
    for retry in report['followups']:
        ordered(retry)
        require(retry['response']['choices'] == report['followups'][0]['response']['choices'] and
                retry['response']['usage'] == report['followups'][0]['response']['usage'], 'repeat response changed')
    require(len(re.findall(r'NATIVE_VISION_ENCODE count=\d+ ', raw)) == report['encodes'] == 2, 'image re-encoded after cancellation')
    idle = [record for line in raw.splitlines()
            if (record := cache_record(line)) is not None and record['event'] == 'idle']
    require(idle == report['idle'] and len(idle) >= 6, 'GPU idle release telemetry missing')
    require(report['mixed_idle_release'] in idle, 'mixed device-owner release missing')
    require(all(x['entries'] == 2 and x['embedding_bytes'] == 3932160 for x in idle[-4:]), 'unbounded image owners')
    require(all(0 < x['backend_allocated_bytes'] <= x['backend_peak_allocated_bytes'] for x in idle), 'invalid memory telemetry')
    require(idle[-1]['backend_allocated_bytes'] == idle[-2]['backend_allocated_bytes'], 'repeat memory grows')
    require(report['actual_draft_tokens'] > 0, 'actual speculation missing')
    require(execution['server_removed'] and execution['production_stopped'], 'isolated run cleanup failed')
    result.update(cold_completion_tokens=0, decode_completion_tokens=report['decode_cancel']['usage']['completion_tokens'],
                  unaffected_siblings=3, ordered_retries=4, encoder_submissions=2,
                  actual_draft_tokens=report['actual_draft_tokens'], stable_backend_bytes=idle[-1]['backend_allocated_bytes'],
                  peak_backend_bytes=max(x['backend_peak_allocated_bytes'] for x in idle),
                  execution_head=execution['head'], binary_sha256=execution['binary_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--trace-log', type=Path, required=True)
    parser.add_argument('--execution', type=Path, help='separate actual supervisor/cleanup receipt bound to request-report SHA-256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve evidence')
    result = {'status': 'FAIL', 'scope': 'actual MTP3 two-image partial-prefill/C4 decode cancellation, ownership/drain/retry; not numerical/RNG or inside-kernel interruption parity'}
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
