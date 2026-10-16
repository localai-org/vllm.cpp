#!/usr/bin/env python3
"""Recheck the bounded Python self-control from complete unmasked head files.

This validates the recorded observation/restore chain, not native quality,
MTP, batch-invariant arithmetic, or a release-contract change.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


STATE_NAMES = {f'gdn{i}-{kind}' for i in range(48) for kind in ('conv', 'ssm')} | {
    f'attn{i}-{kind}' for i in range(16) for kind in ('k', 'v')}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head(root, entry):
    require(entry['shape'] == [248320] and entry['dtype'] == 'float16'
            and Path(entry['file']).name == entry['file'], 'invalid complete head')
    path = root / entry['file']
    require(path.stat().st_size == 248320 * 2 and digest(path) == entry['sha256'], 'head payload identity differs')
    values = struct.unpack('<248320e', path.read_bytes())
    require(all(math.isfinite(v) for v in values), 'nonfinite unmasked head')
    return values


def metrics(a, b):
    top = [sorted(range(len(v)), key=lambda i: (-v[i], i))[:2] for v in (a, b)]
    return {'elements': len(a), 'different_elements': sum(x != y for x, y in zip(a, b)),
            'max_abs': max(abs(x-y) for x, y in zip(a, b)),
            'rel_l2': math.sqrt(math.fsum((x-y)**2 for x, y in zip(a, b)) /
                               max(math.fsum(x*x for x in a), 1e-60)),
            'reference_top2_ids': top[0], 'candidate_top2_ids': top[1],
            'reference_margin': a[top[0][0]] - a[top[0][1]],
            'candidate_margin': b[top[1][0]] - b[top[1][1]]}


def states(entry):
    require(entry.keys() == STATE_NAMES and all(isinstance(v, str) and len(v) == 64
            and all(c in '0123456789abcdef' for c in v) for v in entry.values()), 'incomplete state fingerprints')


def check_metrics(actual, expected):
    require(actual.keys() == expected.keys(), 'metric schema differs')
    for key, value in expected.items():
        if key == 'rel_l2':
            require(math.isclose(actual[key], value, rel_tol=1e-11, abs_tol=1e-14), 'relative metric differs')
        else:
            require(actual[key] == value, 'complete head metric differs: ' + key)


def analyze(root, expected_receipt_sha256):
    receipt = root / 'result.json'
    require(digest(receipt) == expected_receipt_sha256, 'receipt identity differs')
    report = json.loads(receipt.read_text())
    require(report['status'] == 'DIAGNOSTIC', 'producer did not complete')
    spec, frames = report['specification'], report['frames']
    require(len(spec['teacher_tokens']) == 8 and len(frames['c1']) == len(frames['c4']) == 8,
            'incomplete fixed window')
    for phase, n in (('c1', 1), ('c4', 4)):
        pre = frames[phase + '_prefill']
        row, qsl = pre['row'], pre['query_start_loc']
        require(pre['rows'] == n and len(pre['seq_lens']) == n and len(qsl) == n + 1
                and qsl[row + 1] - qsl[row] == spec['context']
                and pre['input_token_ids'] == spec['expanded_prompt_ids'], 'prefill binding differs')
        head(root, pre['head'])
        for step, entry in enumerate(frames[phase]):
            row, context = entry['row'], spec['context'] + step
            require(entry['step'] == step and entry['rows'] == n and 0 <= row < n
                    and entry['context'] == context and entry['seq_lens'][row] == context + 1
                    and entry['input_token_ids'][row] == spec['teacher_tokens'][step]
                    and entry['query_start_loc'] == list(range(n + 1))
                    and [axis[row] for axis in entry['positions']] == [context + spec['mrope_delta']] * 3,
                    'decode row/token/position binding differs')
            states(entry['incoming']); states(entry['post'])
            if step:
                require(entry['incoming'] == frames[phase][step - 1]['post'], 'own history did not evolve from recorded post-state')
    comparisons = []
    for step, (c1, c4) in enumerate(zip(frames['c1'], frames['c4'])):
        require(c1['embedding_sha256'] == c4['embedding_sha256'], 'same-token embedding differs')
        require(c4['sibling_count_restored_per_control'] == 4, 'actual siblings not restored')
        a = head(root, c1['head'])
        item = {'step': step, 'own_incoming_different_states': sum(c1['incoming'][k] != c4['incoming'][k] for k in STATE_NAMES)}
        own = metrics(a, head(root, c4['head']))
        check_metrics(c4['evolving_separate_serving_seeds'], own)
        item['evolving_separate_serving_seeds'] = own
        controls = c4['controls']
        require(controls.keys() == {'identical_incoming_single_step', 'evolving_shared_seed'}, 'missing arithmetic controls')
        for label, control in controls.items():
            states(control['incoming']); states(control['post'])
            expected = (c1['incoming'] if label == 'identical_incoming_single_step' else
                        frames['c4'][step - 1]['controls'][label]['post'] if step else frames['c1'][0]['incoming'])
            require(control['incoming'] == expected, 'control incoming state chain differs')
            value = metrics(a, head(root, control['head']))
            check_metrics(control['logits'], value)
            item[label] = value
        comparisons.append(item)
    summaries = {}
    for label in ('identical_incoming_single_step', 'evolving_shared_seed', 'evolving_separate_serving_seeds'):
        summaries[label] = {'max_abs': max(v[label]['max_abs'] for v in comparisons),
            'peak_rel_l2': max(v[label]['rel_l2'] for v in comparisons),
            'top1_different_steps': [v['step'] for v in comparisons
                                    if v[label]['reference_top2_ids'][0] != v[label]['candidate_top2_ids'][0]]}
    return {'status': 'DIAGNOSTIC', 'scope': __doc__.strip(), 'case': report['case'],
            'receipt': {'file': str(receipt), 'sha256': expected_receipt_sha256},
            'states_per_frame': 128, 'window_steps': 8, 'full_vocabulary_size': 248320,
            'prefill_logits': metrics(head(root, frames['c1_prefill']['head']), head(root, frames['c4_prefill']['head'])),
            'summaries': summaries, 'comparisons': comparisons,
            'limitations': ['State fingerprints are observations, not saved replayable state arrays.',
                'Siblings retain actual reference generation histories; native diagnostic siblings use a different recipe.',
                'Teacher outputs are forced after unmasked observation and are not greedy-quality evidence.',
                'No MTP or chosen-head native qualification; no cross-runtime tensor equality claim.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'preserve prior analysis')
    result = {'status': 'FAIL'}
    try:
        result = analyze(args.capture, args.receipt_sha256)
    except Exception as error:
        result['error'] = repr(error)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'case', 'summaries', 'error')}))
    return 0 if result['status'] == 'DIAGNOSTIC' else 1


if __name__ == '__main__':
    raise SystemExit(main())
