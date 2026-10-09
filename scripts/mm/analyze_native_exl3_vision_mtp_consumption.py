#!/usr/bin/env python3
"""Check actual MTP emitted/consumed position transitions, not logit parity.

Speculative host ids are fresh: the runner vetoes async input combine. This
trace observes scheduled target inputs and actual sampler outputs without new
GPU copies. Independent full-head acceptance/retention is separate evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path


def require(value, message):
    if not value:
        raise RuntimeError(message)


def analyze(report):
    require(report['status'] == 'DIAGNOSTIC' and report['server_removed'] and report['production_stopped'],
            'incomplete native producer or teardown')
    result = []
    for response in report['responses']:
        public_id = response['response']['id']
        request_ids = {x['request_id'] for x in report['trace'] if x['request_id'] in (public_id, public_id + '-0')}
        require(len(request_ids) == 1, 'missing unique actual request trace')
        actual_id = request_ids.pop()
        entries = [x for x in report['trace'] if x['request_id'] == actual_id]
        require(entries and len(entries) <= 64 and all(x['token_ids'] for x in entries), 'empty or unbounded sampled trace')
        emitted = []
        previous = None
        transitions = []
        accepted = []
        for entry in entries:
            ids = entry['token_ids']
            md = entry.get('target_consumption')
            if md is not None:
                n, k, start = len(ids), md['drafts'], md['computed_before']
                require(md['fresh_host_ids'] is True and 1 <= k <= 3 and 1 <= n <= k + 1
                        and md['actual_num_sampled'] == n and md['committed_before'] == start + 1,
                        'wrong emitted versus consumed context')
                require(md['positions'] == list(range(start, start + k + 1)) and md['seq_len'] == start + k + 1
                        and len(md['input_ids']) == k + 1 and ids[:-1] == md['input_ids'][1:n],
                        'wrong target positions or accepted proposal prefix')
                require(previous is not None and md['input_ids'][0] == previous['token_ids'][-1]
                        and md['previous_num_sampled'] == len(previous['token_ids']),
                        'wrong previous bonus/correction token or accepted-state selector')
                prev_md = previous.get('target_consumption')
                if prev_md is not None:
                    expected = prev_md['computed_before'] + len(previous['token_ids'])
                    require(start == expected and md['committed_before'] == prev_md['committed_before'] + len(previous['token_ids'])
                            and md['positions'].count(expected) == 1,
                            'bonus/correction skipped or consumed at duplicate position')
                    transitions.append({'previous_computed': prev_md['computed_before'],
                        'previous_actual_accepted_drafts': len(previous['token_ids']) - 1,
                        'previous_committed_state_context': expected, 'next_computed': start,
                        'next_target_bonus_position': md['positions'][0],
                        'bonus_or_correction_id': md['input_ids'][0], 'previous_num_sampled': md['previous_num_sampled']})
                else:
                    require(len(previous['token_ids']) == 1 and start == response['response']['usage']['prompt_tokens'],
                            'first target consumption does not follow actual prompt prefill')
                accepted.append(n - 1)
            elif previous is not None:
                # Do not silently jump over an unobserved ordinary/clamped step.
                raise RuntimeError('ordinary step interrupts bounded MTP consumption witness')
            emitted.extend(ids)
            previous = entry
        require(len(transitions) >= 2, 'insufficient consecutive actual MTP transitions')
        requested = response['response']['usage']['completion_tokens']
        require(requested > 0 and len(emitted) >= requested and len(emitted) - requested <= 3,
                'sampler trace differs from returned token count')
        result.append({'label': response['label'], 'request_id': actual_id,
            'actual_accepted_lengths': sorted(set(accepted)), 'target_verify_packets': len(accepted),
            'checked_consecutive_transitions': len(transitions), 'transitions': transitions,
            'returned_completion_tokens': requested, 'sampler_emitted_tokens': len(emitted),
            'terminal_note': 'Final emitted bonus is not scheduled after completion; final sampler packet may be truncated by max_tokens.'})
    return {'status': 'PASS', 'scope': __doc__.strip(), 'requests': result,
        'limitations': ['Scheduled target consumption/selector seam only, not an independent logit acceptance recomputation.',
            'Full target retention uses the separate actual accepted0/1/2 snapshots and forced selector controls.',
            'No cross-shape greedy parity, all-prefix qualification or batch-contract approval.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'preserve prior analysis')
    result = {'status': 'FAIL'}
    try:
        require(hashlib.sha256(args.receipt.read_bytes()).hexdigest() == args.receipt_sha256, 'receipt checksum differs')
        result = analyze(json.loads(args.receipt.read_text()))
        result['receipt_sha256'] = args.receipt_sha256
    except Exception as error:
        result['error'] = repr(error)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'error', 'requests')}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
