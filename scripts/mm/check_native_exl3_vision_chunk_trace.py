#!/usr/bin/env python3
"""Check native chunk slices/M-RoPE against compact executed oracle digests.

The caller supplies a fresh server log with VT_NATIVE_VISION_TRACE=2 and a
request report. No tensors are downloaded and no position formula is emulated.
This validates submitted metadata, not complete-tower numerical values.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(values):
    return hashlib.sha256(struct.pack('<' + str(len(values)) + 'i', *values)).hexdigest()


def run(args, result):
    reference = json.loads(args.reference.read_text())
    refs = {case['offset']: case for case in reference['cases']}
    execution = json.loads(args.requests.read_text())
    require(execution['status'] == 'PASS', 'request run failed')
    supervisor = json.loads(args.execution.read_text()) if args.execution else execution
    if args.execution:
        require(supervisor['exit_code'] == 0 and supervisor['server_removed'] and
                supervisor['production_stopped'], 'missing actual execution/cleanup')
        require(supervisor['requests_sha256'] == hashlib.sha256(args.requests.read_bytes()).hexdigest(),
                'wrong supervised request report')
    require(execution.get('spec_depth', 0) == args.spec_depth, 'capture depth differs')
    mode = next(item for item in execution['modes'] if item['budget'] == args.budget)
    require(len(mode['cases']) == 3, 'exactly three boundary cases required')
    log = args.trace_log.read_text()
    groups = {}
    current = None
    committed, draft_frames = {}, {}
    for line in log.splitlines():
        match = re.search(r'INFO Received request chatcmpl-(\d+) endpoint=', line)
        if match:
            current = int(match.group(1))
            groups[current] = []
            committed[current] = []
            draft_frames[current] = []
        if line.startswith('NATIVE_VISION_EMBED '):
            require(current is not None, 'unattributed metadata')
            frame = json.loads(line.split(' ', 1)[1])
            frame['committed_before'] = len(committed[current])
            groups[current].append(frame)
        elif line.startswith('NATIVE_VISION_DRAFT_EMBED '):
            require(current is not None and groups[current], 'unattributed draft')
            draft_frames[current].append((len(groups[current]) - 1, json.loads(line.split(' ', 1)[1])))
        elif line.startswith('NATIVE_VISION_SAMPLED '):
            record = json.loads(line.split(' ', 1)[1])
            require(record['request_id'] == 'chatcmpl-' + str(current), 'wrong sampled request')
            committed[current].extend(record['token_ids'])
    require(len(groups) == 3, 'use a fresh isolated server log')
    encode = re.findall(r'NATIVE_VISION_ENCODE count=(\d+) hash=([0-9a-f]{64}) rows=(\d+)', log)
    require(len(encode) == 1 and encode[0][0] == '1' and encode[0][2] == '192',
            'image must encode exactly once across all requests/chunks')
    result['encoder_submissions'] = encode
    result['reference_source_sha256'] = reference['source_sha256']
    result['execution_head'] = supervisor['head']
    result['binary_sha256'] = supervisor['binary_sha256']
    for index, case in enumerate(mode['cases']):
        ref = refs[case['offset']]
        unexpanded = case['tokenized_pre_expansion']
        require(unexpanded[ref['offset']] == 248056, 'wrong placeholder offset')
        tokens = unexpanded[:ref['offset']] + [248056] * ref['length'] + unexpanded[ref['offset'] + 1:]
        require(digest(tokens) == ref['tokens_sha256'], 'request does not match pinned reference')
        length = ref['prompt_tokens']
        require(length == case['expected_prompt_tokens'], 'prompt length differs')
        require(groups.get(index), 'missing request metadata')
        if args.execution:
            require(groups[index][0]['tokens'] == min(length, args.budget),
                    'fresh C1 first chunk differs from declared budget')
        prompt_axes = [[], [], []]
        start = 0
        visual_frames = []
        starts = []
        for step, frame in enumerate(groups[index]):
            if args.spec_depth and start >= length:
                start = length + frame['committed_before'] - 1
            starts.append(start)
            size = frame['tokens']
            require(size > 0 and size <= args.budget, 'invalid step size')
            end = start + size
            require(len(frame['mrope']) == 3 * size, 'wrong position shape')
            for axis in range(3):
                values = frame['mrope'][axis * size:(axis + 1) * size]
                for column, value in enumerate(values):
                    position = start + column
                    if position < length:
                        prompt_axes[axis].append(value)
                    else:
                        require(value == position + ref['delta'], 'incorrect completion position')
            mask = [int(ref['offset'] <= pos < ref['offset'] + ref['length'])
                    for pos in range(start, end)]
            require(frame['image_mask'] == mask, 'incorrect visual mask')
            first = max(start, ref['offset'])
            last = min(end, ref['offset'] + ref['length'])
            slices = [[first - ref['offset'], last - first]] if last > first else []
            require(frame['source_slices'] == slices, 'incorrect encoder source rows')
            if slices:
                visual_frames.append({'step': step, 'start': start, 'tokens': size, 'slices': slices})
            start = end
        require(all(len(axis) == length for axis in prompt_axes), 'incomplete prompt positions')
        require(digest([value for axis in prompt_axes for value in axis]) == ref['positions_sha256'],
                'prompt positions differ from executed Python oracle')
        if args.spec_depth:
            emitted = len(committed[index])
            usage = case['response']['usage']['completion_tokens']
            require(usage <= emitted <= usage + args.spec_depth, 'incorrect committed token count')
            by_step = dict(draft_frames[index])
            require(len(by_step) == len(draft_frames[index]), 'duplicate draft image merge')
            for step, frame in enumerate(groups[index]):
                shifted_start = starts[step] + 1
                shifted_end = shifted_start + frame['tokens']
                first = max(shifted_start, ref['offset'])
                last = min(shifted_end, ref['offset'] + ref['length'])
                expected = [[first - ref['offset'], last - first]] if last > first else []
                draft = by_step.get(step)
                # A missing output at the first lookahead-only image row uses
                # token lookup in the pinned V2 runner; target-covered rows must
                # always have their actual cached encoder slices.
                if expected and draft is None:
                    require(ref['offset'] == shifted_end - 1 and expected == [[0, 1]]
                            and not frame['source_slices'], 'missing admitted draft source')
                elif expected:
                    mask = [int(ref['offset'] <= pos < ref['offset'] + ref['length'])
                            for pos in range(shifted_start, shifted_end)]
                    require(draft['tokens'] == frame['tokens'] and draft['image_mask'] == mask
                            and draft['source_slices'] == expected, 'incorrect shifted draft source/mask')
                else:
                    require(draft is None, 'unexpected visual draft rows')
        else:
            require(start == length + case['response']['usage']['completion_tokens'] - 1,
                    'incorrect final logical context')
        result['cases'].append({'offset': ref['offset'], 'frames': len(groups[index]),
                               'positions_exact': True, 'mask_and_source_slices_exact': True,
                               'visual_frames': visual_frames})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--trace-log', type=Path, required=True)
    parser.add_argument('--execution', type=Path, help='actual separate supervisor receipt for a portable request report')
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--spec-depth', type=int, choices=[0, 1, 3], default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[2] /
                        'tests/fixtures/native_vision_http/chunk-mrope-reference.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve earlier evidence')
    result = {'status': 'FAIL', 'scope': 'native submitted metadata versus executed CPU oracle; '
              'not full tower/logit parity', 'cases': []}
    code = 1
    try:
        run(args, result)
        result['status'] = 'PASS'
        code = 0
    except Exception as error:
        result['error'] = repr(error)
        print(result['error'])
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'cases': len(result['cases'])}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
