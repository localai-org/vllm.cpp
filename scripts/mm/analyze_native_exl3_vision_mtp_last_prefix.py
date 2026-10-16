#!/usr/bin/env python3
"""Validate the actual last-common alpha prefix inside MTP verification packets.

This preserves the historical strict failure. Different committed prefixes are
never compared as equal states. Full packet heads independently bind acceptance.
"""
import argparse
import json
from pathlib import Path

from analyze_native_exl3_vision_adjacent_prefix import PREFIX
from analyze_native_exl3_vision_mtp_prefix import STATE_NAMES, compare, load
from analyze_native_exl3_vision_prefix import floats, metrics, require


def analyze(directory):
    before = [load(directory, f'prefix-{i}.json', packet_prefix=PREFIX) for i in range(3)]
    after = [load(directory, f'prefix-{i}-commit.json', packet_prefix=PREFIX) for i in range(3)]
    acceptance = []
    width = 248320
    for i, ((pre, blobs, binding), (post, committed, post_binding)) in enumerate(zip(before, after)):
        mtp, actual = pre['mtp'], post['mtp']
        require(pre['phase'] == 'post_target_before_sampling' and post['phase'] == 'post_verify_before_writeback' and
                post['unmasked_logits_metadata'] == f'prefix-{i}.json' and
                pre['request_id'] == post['request_id'] and pre['token_prefix'] == post['token_prefix'] and
                mtp['packet_token_prefix'] == actual['packet_token_prefix'] and
                mtp['verification_tokens'] == actual['verification_tokens'] and
                mtp['spec_state_indices'] == actual['spec_state_indices'], 'before/after packet binding differs')
        query = mtp['verification_tokens']
        column = mtp['selected_query_column']
        require(column == actual['selected_query_column'] and len(query) == 4, 'wrong selected packet column')
        head = blobs['verify-logits']
        selected = blobs['logits']
        raw = (directory / head['file']).read_bytes()
        require((directory / selected['file']).read_bytes() == raw[column * width * 4:(column + 1) * width * 4],
                'selected head differs from the full packet row')
        values = floats(directory, head)
        ids = [max(range(width), key=lambda v: values[row * width + v]) for row in range(4)]
        require(ids == mtp['unmasked_target_argmax_ids'] == actual['unmasked_target_argmax_ids'],
                'recorded packet argmax differs from actual full logits')
        accepted = 0
        while accepted < 3 and ids[accepted] == query[accepted + 1]:
            accepted += 1
        require(accepted >= column and actual['accepted_drafts'] == accepted and actual['num_sampled'] == accepted + 1 and
                actual['emitted_ids'] == query[1:1 + accepted] + [ids[accepted]],
                'actual greedy acceptance/correction differs from full packet logits')
        # The final state must be compared to a target snapshot at its own
        # accepted prefix, not to the earlier last-common-prefix snapshot.
        anchor_binding = binding
        target = before[i]
        if accepted != column:
            target = load(directory, f'prefix-{i}-target-anchor.json', packet_prefix=PREFIX, target_anchor=True)
            record, _, anchor_binding = target
            expected = dict(pre)
            expected.update(phase='post_target_greedy_anchor_before_sampling',
                            unmasked_logits_metadata=f'prefix-{i}.json', seq_len=post['seq_len'],
                            gdn_slot=post['gdn_slot'], blobs=record['blobs'], image_features=[])
            expected['mtp'] = dict(mtp, predicted_accepted_drafts=accepted,
                                   selected_ssm_slot=actual['selected_ssm_slot'],
                                   valid_conv_window_offset=accepted, state_token_prefix=actual['state_token_prefix'])
            require(record == expected, 'target anchor differs from bound target packet/actual selected state')
        anchor = compare(directory, target, directory, after[i], logits=False)
        require(anchor['all_compared_storage_exact'], 'committed states differ from the corresponding target snapshot')
        acceptance.append({'before': binding, 'after': post_binding, 'query_column': column,
                           'full_packet_argmax': ids, 'accepted_drafts': accepted,
                           'emitted_ids': actual['emitted_ids'], 'selected_context': pre['seq_len'],
                           'committed_context': post['seq_len'], 'committed_state_token_prefix': actual['state_token_prefix'],
                           'committed_states_vs_target_anchor': anchor, 'target_anchor_binding': anchor_binding,
                           'extended_prefix_anchor_required': accepted != column})
    comparisons = {}
    for i, label in ((0, 'warm_c2_vs_c1'), (1, 'mixed_c4_vs_c1')):
        comparisons[label] = {'storage': compare(directory, before[2], directory, before[i]),
                             'post_target_head': metrics(floats(directory, before[2][1]['logits']),
                                                         floats(directory, before[i][1]['logits'])),
                             'same_input_margin_bound': 'NOT_APPLICABLE: accumulated incoming histories are not restored'}
    top = [entry[0]['top2'] for entry in before]
    layouts = [{k: record[k] for k in ('row', 'num_reqs', 'actual_token_rows', 'positions', 'seq_lens',
                                      'query_start_loc', 'mrope_delta', 'own_image_features', 'gdn_slot')}
               for record, _, _ in before]
    return {'status': 'DIAGNOSTIC', 'scope': __doc__.strip(), 'metadata': [entry[2] for entry in before + after],
            'full_target_states_per_selected_snapshot': len(STATE_NAMES), 'top2': top, 'layouts': layouts,
            'acceptance': acceptance, 'comparisons': comparisons,
            'mixed_c4_strict_divergence_reproduced': top[1]['ids'][0] != top[2]['ids'][0],
            'warm_c2_strict_divergence_reproduced': top[0]['ids'][0] != top[2]['ids'][0],
            'limitations': ['Post-target snapshots at the common prefix already include accumulated shape differences.',
                            'Pre-sampler greedy previews are checked against actual rejection; they do not replace it.',
                            'This is not a same-input arithmetic comparison or approval of a numerical contract.',
                            'Original strict token failures and tower geometry limits remain unchanged.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'preserve prior analysis')
    report = {'status': 'FAIL'}
    try:
        report = analyze(args.capture)
    except Exception as error:
        report['error'] = repr(error)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'error', 'top2', 'mixed_c4_strict_divergence_reproduced')}))
    return 0 if report['status'] == 'DIAGNOSTIC' else 1


if __name__ == '__main__':
    raise SystemExit(main())
