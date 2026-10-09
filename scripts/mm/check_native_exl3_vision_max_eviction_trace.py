#!/usr/bin/env python3
"""Check the bounded C1 MTP3 max-image/small-image eviction and reuse wave.

Requires ten fresh requests: max,max,small,small,small twice. Position digests
come from executed CPU reference code. No service management or inference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from native_vision_warm_owners import check_warm_sequence
from native_vision_cache_trace import cache_record
from native_vision_completion import committed_ids


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(values):
    return hashlib.sha256(struct.pack('<' + str(len(values)) + 'i', *values)).hexdigest()


def run(args, result):
    report = json.loads(args.requests.read_text())
    require(report['status'] == 'PASS' and report['spec_depth'] == 3, 'failed or wrong-depth run')
    execution = json.loads(args.execution.read_text()) if args.execution else report
    require(execution['production_stopped'] and execution['server_removed'], 'missing cleanup')
    if args.execution:
        require(execution['exit_code'] == 0 and execution['requests_sha256'] ==
                hashlib.sha256(args.requests.read_bytes()).hexdigest(), 'wrong/failed supervised request report')
    refs = json.loads(args.reference.read_text())
    reference = {case['file']: case for case in refs['cases']}
    expected = ['orbit-max-2048.png'] * 2 + ['comet.png'] * 3
    warm_owner = report.get('warm_owner', False)
    require(type(warm_owner) is bool, 'invalid warm-owner mode')
    waves = 5 if warm_owner else 2
    count = 5 * waves
    if warm_owner:
        require(args.execution is not None, 'warm sequence requires actual supervisor/cleanup receipt')
    require([case['file'] for case in report['cases']] == expected * waves, 'wrong bounded request wave')
    groups, current, cache, encodes, graphs = {}, None, [], [], []
    for line in args.trace_log.read_text().splitlines():
        match = re.search(r'INFO Received request chatcmpl-(\d+) endpoint=', line)
        if match:
            current = int(match.group(1))
            require(current not in groups, 'duplicate request')
            groups[current] = {'frames': [], 'drafts': {}, 'ids': []}
        if line.startswith('NATIVE_VISION_EMBED '):
            require(current is not None, 'unattributed target')
            frame = json.loads(line.split(' ', 1)[1])
            frame['committed_before'] = len(groups[current]['ids'])
            groups[current]['frames'].append(frame)
        elif line.startswith('NATIVE_VISION_DRAFT_EMBED '):
            require(current is not None and groups[current]['frames'], 'unattributed draft')
            step = len(groups[current]['frames']) - 1
            require(step not in groups[current]['drafts'], 'duplicate draft')
            groups[current]['drafts'][step] = json.loads(line.split(' ', 1)[1])
        elif line.startswith('NATIVE_VISION_SAMPLED '):
            record = json.loads(line.split(' ', 1)[1])
            require(current is not None and record['request_id'] == f'chatcmpl-{current}', 'wrong sampled owner')
            groups[current]['ids'].extend(record['token_ids'])
        elif (record := cache_record(line)) is not None:
            cache.append(record)
        elif line.startswith('NATIVE_VISION_GRAPH '):
            graphs.append(json.loads(line.split(' ', 1)[1]))
        elif line.startswith('NATIVE_VISION_ENCODE '):
            match = re.fullmatch(r'NATIVE_VISION_ENCODE count=(\d+) hash=([0-9a-f]{64}) rows=(\d+)', line)
            require(match is not None, 'malformed encoder observation')
            encodes.append(match.groups())
    require(set(groups) == set(range(count)), 'fresh isolated bounded-request trace required')
    require(len(encodes) == 2 * waves and [int(e[0]) for e in encodes] == list(range(1, 2 * waves + 1))
            and [int(e[2]) for e in encodes] == [4096, 192] * waves, 'missing eviction/re-encode')
    require(all(e[1] == encodes[index % 2][1] for index, e in enumerate(encodes))
            and encodes[0][1] != encodes[1][1], 'wrong cache identities')
    records = [c for c in cache if c['event'] in ('insert', 'evict')]
    require([c['event'] for c in records] == ['insert'] + ['evict', 'insert'] * (2 * waves - 1), 'missing actual cache evictions')
    sizes = {e[1]: int(e[2]) * 5120 * 2 for e in encodes}
    resident = {}
    for record in records:
        if record['event'] == 'insert':
            require(record['hash'] not in resident, 'double insertion')
            resident[record['hash']] = sizes[record['hash']]
        else:
            require(record['hash'] in resident, 'eviction of absent output')
            del resident[record['hash']]
        require(record['entries'] == len(resident) <= 1
                and record['embedding_bytes'] == sum(resident.values()), 'cache ownership byte mismatch')
    require([c['hash'] for c in records if c['event'] == 'insert'] == [e[1] for e in encodes], 'encoder/insert mismatch')
    idle = [c for c in cache if c['event'] == 'idle']
    require(len(idle) == count, 'missing device release witnesses')
    for c in cache:
        require(0 < c['backend_allocated_bytes'] <= c['backend_peak_allocated_bytes'], 'invalid backend telemetry')
    warm = idle[-1]['backend_allocated_bytes']
    require(all(c['backend_allocated_bytes'] == warm for c in idle[-3:]), 'final allocation grows')
    require(warm == idle[4]['backend_allocated_bytes'] and warm < idle[6]['backend_allocated_bytes'],
            'small workspace did not return to its prior resident baseline')
    if warm_owner:
        observed = check_warm_sequence(idle, 5, 5)
        require(report['warm_owner_observations'] == observed, 'incorrect warm sequence phase report')
        result['warm_owner_observations'] = observed
    sequences, checked, committed_count = {}, 0, 0
    for index, case in enumerate(report['cases']):
        ref = reference[case['file']]
        tokens = case['expanded_tokens']
        require(case['sha256'] == ref['image_sha256'] and digest(tokens) == ref['tokens_sha256'], 'wrong frozen recipe')
        require(case['visual_rows'] == ref['rows'] and case['offset'] == ref['offset'], 'wrong image span')
        length, offset, rows = ref['prompt_tokens'], ref['offset'], ref['rows']
        require(case['response']['id'] == f'chatcmpl-{index}', 'wrong response identity')
        ids = committed_ids(case['response'], length, groups[index]['ids'])
        require(set(case['gauges']) == {'vllm:num_requests_running', 'vllm:num_requests_waiting'}
                and all(math.isfinite(v) and v == 0 for v in case['gauges'].values()), 'missing drain gauges')
        require(case['device_idle'] == idle[index], 'wrong device release owner')
        text = case['response']['choices'][0]['message']['content'].lower()
        facts = ('orbit', '731', ('red', 'circle'), ('blue', 'square')) if rows == 4096 \
            else ('comet', '924', ('green', 'square'), ('yellow', 'circle'))
        require(facts[0] in text and facts[1] in text, 'incorrect image heading')
        for side, expected_words in [('left', facts[2]), ('right', facts[3])]:
            start = text.find(side)
            require(start >= 0 and all(word in text[start:start + 100] for word in expected_words), 'incorrect spatial facts')
        group = groups[index]
        committed_count += len(ids)
        if case['file'] in sequences:
            require(ids == sequences[case['file']], 'token sequence changed after reuse/eviction')
        sequences[case['file']] = ids
        cursor, axes = 0, [[], [], []]
        for step, frame in enumerate(group['frames']):
            if cursor >= length:
                cursor = length + frame['committed_before'] - 1
            size = frame['tokens']
            require(0 < size <= 1600 and len(frame['mrope']) == size * 3, 'wrong scheduled position shape')
            for axis in range(3):
                for column, value in enumerate(frame['mrope'][axis * size:(axis + 1) * size]):
                    position = cursor + column
                    if position < length:
                        axes[axis].append(value)
                    else:
                        require(value == position + ref['delta'], 'wrong continuation position')
            for shift, target in [(0, frame), (1, group['drafts'].get(step))]:
                begin, end = cursor + shift, cursor + shift + size
                first, last = max(begin, offset), min(end, offset + rows)
                slices = [[first - offset, last - first]] if last > first else []
                if shift and not slices:
                    require(target is None, 'unexpected draft image rows')
                    continue
                require(target is not None and target['tokens'] == size, 'missing target/draft source')
                require(target['source_slices'] == slices and target['image_mask'] ==
                        [int(offset <= pos < offset + rows) for pos in range(begin, end)], 'incorrect target/draft image slice')
            cursor += size
            checked += size * 3
        require(all(len(axis) == length for axis in axes) and digest(sum(axes, [])) == ref['positions_sha256'],
                'prompt positions differ from executed reference')
    captured = [g for g in graphs if g['captured']]
    require(captured and any(g['tokens'] == 4 for g in captured)
            and captured[-1]['replay_count'] > captured[0]['replay_count'], 'missing actual MTP3 graph progress')
    result.update(requests=count, encoder_submissions=2 * waves, actual_evictions=2 * waves - 1, committed_ids_checked=committed_count,
        positions_checked=checked, reference_source_sha256=refs['source_sha256'],
        stable_small_backend_bytes=warm, resident_embedding_bytes=sum(resident.values()),
        peak_backend_bytes=max(c['backend_peak_allocated_bytes'] for c in cache),
        binary_sha256=execution['binary_sha256'], execution_head=execution['head'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', required=True, type=Path)
    parser.add_argument('--trace-log', required=True, type=Path)
    parser.add_argument('--execution', type=Path,
                        help='separate actual supervisor cleanup/binary receipt for a portable client report')
    parser.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[2] /
                        'tests/fixtures/native_vision_http/max-eviction-mrope-reference.json')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve previous evidence')
    result = {'status': 'FAIL', 'scope': 'bounded native max/small MTP3 metadata, eviction and owner release; not tower/RNG/performance parity'}
    code = 1
    try:
        run(args, result)
        result['status'], code = 'PASS', 0
    except Exception as error:
        result['error'] = repr(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
