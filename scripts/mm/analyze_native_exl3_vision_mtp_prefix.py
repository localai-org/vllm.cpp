#!/usr/bin/env python3
"""Bounded actual MTP prefix/accepted-state attribution, never a parity waiver."""
import argparse
import json
import math
import struct
from pathlib import Path

from analyze_native_exl3_vision_prefix import digest, floats, load as load_target, metrics, require

PROMPT_IDS = [1835, 8029, 13, 14569, 411, 11316, 440, 264, 2716, 57879, 15673, 314, 5089, 25, 799, 5346, 799, 16327]
TARGET_METADATA_SHA = [
    '3434c67d7a632ec4d461297fb1c2af532fa59a6373468fe3b511c8d9318e8328',
    '0e1d84a47a3773a4982236d743b0caf19c771c594faeca18a9add9168ef6c07c',
    'f81eb75e5bee92d3a182ee8a12273bfbcda213c73e97ccae6bfffbcf64ee427a',
    '9fc06f1d14307c75b87bb15a3a1a49e99fd1a657ca5a852cdedf3395b975392e']
STATE_NAMES = {f'gdn{i}-{kind}' for i in range(48) for kind in ('conv', 'ssm')} | {
    f'attn{i}-{kind}' for i in range(16) for kind in ('k', 'v')}


def load(directory, name, *, packet_prefix=None, target_anchor=False):
    path = directory / name
    require(path.stat().st_size <= 1024 * 1024, 'metadata bound exceeded')
    record = json.loads(path.read_text())
    require(packet_prefix is not None or not target_anchor, 'target anchor requires explicit packet prefix')
    require(record['phase'] == 'post_target_greedy_anchor_before_sampling' if target_anchor else
            record['phase'] in ('post_target_before_sampling', 'post_verify_before_writeback'), 'wrong phase')
    row, count = record['row'], record['num_reqs']
    require(0 <= row < count <= 4 and record['request_ids'][row] == record['request_id'], 'wrong request row')
    require(record['prompt_tokens'] == 18 and record['token_prefix'][:18] == PROMPT_IDS
            and (record['output_prefix'] in ([], [271]) if packet_prefix is None
                 else record['output_prefix'] == packet_prefix), 'wrong frozen prefix')
    require(record['token_prefix'][18:] == record['output_prefix'] and record['own_image_features'] == 0
            and record['mrope_delta'] == 0, 'selected request must be frozen pure-text alpha')
    mtp = record['mtp']
    query_column = 0
    packet_tokens = record['token_prefix']
    if packet_prefix is not None:
        query_column = mtp['selected_query_column']
        packet_tokens = mtp['packet_token_prefix']
        require(mtp['spec_row'] and type(query_column) is int and 0 <= query_column <= 3 and
                len(packet_tokens) == record['num_computed_before_step'] + 1 and
                record['token_prefix'] == packet_tokens + mtp['verification_tokens'][1:query_column + 1],
                'wrong selected packet prefix')
        require(mtp['selected_prefix_reachable_greedy'] is True, 'selected prefix was not reached greedily')
    offsets = record['query_start_loc']
    begin, end = offsets[row:row + 2]
    require(len(offsets) == count + 1 and offsets[0] == 0 and offsets[-1] == record['actual_token_rows']
            and all(a <= b for a, b in zip(offsets, offsets[1:])), 'wrong query offsets')
    require(mtp['verification_tokens'] == record['input_token_ids'][begin:end]
            and record['positions'][begin:end] == list(range(record['num_computed_before_step'], record['seq_lens'][row])),
            'verification token/position mismatch')
    require(mtp['physical_seq_len'] == record['seq_lens'][row] and 0 < mtp['physical_seq_len'] <= 1600,
            'wrong physical context')
    require(record['input_token_ids'][begin:begin + len(record['token_prefix']) - record['num_computed_before_step']]
            == record['token_prefix'][record['num_computed_before_step']:], 'query/committed prefix mismatch')
    page = record['kv_block_table'][row * record['kv_block_table_cols']]
    require(page >= 0 and record['kv_write_slots'][begin:end] ==
            [page * 1600 + t for t in range(record['num_computed_before_step'], mtp['physical_seq_len'])],
            'wrong KV write positions')
    if mtp['spec_row']:
        masks = mtp['spec_sequence_masks']
        require(len(masks) == count and masks[row] == 1 and mtp['drafts'] == 3
                and end - begin == 4 and mtp['spec_state_columns'] == 4, 'wrong MTP3 query geometry')
        spec_row = sum(masks[:row])
        slots = mtp['spec_state_indices'][spec_row * 4:(spec_row + 1) * 4]
        require(len(slots) == 4 and len(set(slots)) == 4 and min(slots) >= 0
                and mtp['owner_slot'] == slots[0], 'wrong MTP owner slots')
        selected = (mtp['predicted_accepted_drafts'] if target_anchor else
                    mtp['accepted_drafts'] if record['phase'] == 'post_verify_before_writeback' else query_column)
        require(type(selected) is int, 'non-integer selected column')
        require(0 <= selected <= 3 and record['gdn_slot'] == slots[selected]
                and mtp['selected_ssm_slot'] == slots[selected]
                and mtp['valid_conv_window_offset'] == selected and mtp['valid_conv_window_length'] == 3,
                'wrong selected committed state slot/window')
        require(record['seq_len'] == mtp['physical_seq_len'] - 3 + selected, 'wrong committed state position')
        if target_anchor:
            require(selected != query_column and selected >= query_column and
                    mtp['state_token_prefix'] == packet_tokens + mtp['verification_tokens'][1:selected + 1] and
                    len(mtp['state_token_prefix']) == record['seq_len'], 'wrong predicted target anchor prefix')
        if record['phase'] == 'post_verify_before_writeback':
            require(mtp['num_sampled'] == selected + 1 and len(mtp['emitted_ids']) == selected + 1,
                    'wrong acceptance count')
            require(mtp['emitted_ids'][:-1] == mtp['verification_tokens'][1:selected + 1]
                    and mtp['state_token_prefix'] == packet_tokens + mtp['emitted_ids'][:-1]
                    and len(mtp['state_token_prefix']) == record['seq_len'], 'wrong accepted state prefix')
            if packet_prefix is not None:
                require(mtp['selected_prefix_committed'] is True and selected >= query_column,
                        'actual rejection did not commit the selected prefix')
    else:
        require(record['output_prefix'] == [] and mtp['drafts'] == 0 and record['seq_len'] == 18
                and record['phase'] == 'post_target_before_sampling', 'unexpected ordinary MTP phase')
    blobs, total = {}, 0
    for entry in record['blobs']:
        name, file = entry['name'], entry['file']
        require(name not in blobs and Path(file).name == file, 'unsafe or duplicated payload')
        shape = entry['shape']
        require(shape and all(type(n) is int and n > 0 for n in shape), 'invalid payload shape')
        size = math.prod(shape) * {0: 4, 1: 2, 2: 2, 3: 1, 4: 4, 5: 8}[entry['dtype']]
        total += size
        require(size == entry['bytes'] and total <= 256 * 1024 * 1024, 'payload bound/shape mismatch')
        payload = directory / file
        require(payload.stat().st_size == size and digest(payload) == entry['sha256'], 'payload checksum mismatch')
        if name.endswith('-conv'):
            require(entry['dtype'] == 1 and shape == [10240, 6], 'wrong FP16 MTP Conv layout')
        elif name.endswith('-ssm'):
            require(entry['dtype'] == 0 and shape == [48, 128, 128], 'wrong FP32 MTP SSM layout')
        elif name.startswith('attn'):
            require(shape[0] == record['seq_len'] and len(shape) == 3, 'wrong valid KV prefix')
        blobs[name] = entry
    require(STATE_NAMES <= blobs.keys(), 'missing persistent target state')
    if target_anchor:
        require(blobs.keys() == STATE_NAMES and not record['image_features'] and
                'accepted_drafts' not in mtp and 'num_sampled' not in mtp and 'emitted_ids' not in mtp,
                'target anchor must contain only predicted persistent states')
    if record['phase'] == 'post_target_before_sampling':
        require('logits' in blobs and blobs['logits']['shape'] == [248320], 'missing complete target logit row')
        values = floats(directory, blobs['logits'])
        top = sorted(range(len(values)), key=lambda i: (-values[i], i))[:2]
        require(record['top2'] == {'ids': top, 'logits': [values[i] for i in top],
                'margin': values[top[0]] - values[top[1]]}, 'wrong top2')
        if mtp['spec_row']:
            require(blobs['verify-logits']['shape'] == [4, 248320] and blobs['verify-logits']['dtype'] == 0,
                    'missing complete verification logits')
    return record, blobs, {'file': path.name, 'sha256': digest(path), 'payload_bytes': total}


def canonical(directory, record, entry):
    raw = (directory / entry['file']).read_bytes()
    if entry['name'].endswith('-conv'):
        width = entry['shape'][1]
        offset = record.get('mtp', {}).get('valid_conv_window_offset', 0)
        return b''.join(raw[(c * width + offset) * 2:(c * width + offset + 3) * 2] for c in range(10240))
    return raw


def compare(left_dir, left, right_dir, right, logits=True):
    a, ab, _ = left
    b, bb, _ = right
    require(a['token_prefix'] == b['token_prefix'] and a['seq_len'] == b['seq_len'], 'different selected prefix')
    names = STATE_NAMES | ({'logits'} if logits else set())
    changed = []
    for name in sorted(names):
        x, y = ab[name], bb[name]
        require(x['dtype'] == y['dtype'], 'different storage dtype')
        if not name.endswith('-conv'):
            require(x['shape'] == y['shape'], 'different state shape')
        if name.startswith('attn'):
            require(all(x[k] == y[k] for k in ('fp8_kind', 'k_scale', 'v_scale')), 'different FP8 contract')
        if canonical(left_dir, a, x) != canonical(right_dir, b, y):
            changed.append(name)
    return {'compared_target_states': 128, 'logits_compared': logits, 'different': changed,
            'all_compared_storage_exact': not changed}


def analyze(directory, target):
    before = [load(directory, f'prefix-{i}.json') for i in range(4)]
    committed = [load(directory, f'prefix-{i}-commit.json') for i in (1, 3)]
    comparisons = {'mtp_prefill_mixed_vs_c1': compare(directory, before[2], directory, before[0]),
                   'mtp_first_verify_mixed_vs_c1': compare(directory, before[3], directory, before[1])}
    require(all(digest(target / f'prefix-{i}.json') == sha for i, sha in enumerate(TARGET_METADATA_SHA)),
            'frozen target metadata changed')
    target_pairs = [load_target(target, i) for i in range(4)]
    for i, label in ((0, 'mixed_prefill'), (1, 'mixed_first_verify'), (2, 'c1_prefill'), (3, 'c1_first_verify')):
        comparisons['mtp_vs_target_' + label] = compare(target, target_pairs[i], directory, before[i])
    selected_metrics = {}
    for i, label in ((1, 'mixed_mtp_vs_target'), (3, 'c1_mtp_vs_target')):
        a, ab, _ = target_pairs[i]
        b, bb, _ = before[i]
        selected_metrics[label] = {
            'gdn0-conv': metrics([x[0] for x in struct.iter_unpack('<e', canonical(target, a, ab['gdn0-conv']))],
                                [x[0] for x in struct.iter_unpack('<e', canonical(directory, b, bb['gdn0-conv']))]),
            'gdn0-ssm': metrics(floats(target, ab['gdn0-ssm']), floats(directory, bb['gdn0-ssm'])),
            'logits': metrics(floats(target, ab['logits']), floats(directory, bb['logits']))}
    images = {}
    for index, label in ((1, 'mixed_first_verify'),):
        target_images = {b['mm_hash']: b for b in target_pairs[index][1].values() if 'mm_hash' in b}
        native_images = {b['mm_hash']: b for b in before[index][1].values() if 'mm_hash' in b}
        require(target_images.keys() == native_images.keys(), 'different sibling image identities')
        images[label] = [{'mm_hash': key, 'storage_exact': target_images[key]['sha256'] == native_images[key]['sha256'],
                         'shape': native_images[key]['shape']} for key in sorted(native_images)]
    acceptance = []
    for index, pair in zip((1, 3), committed):
        post, blobs, binding = pair
        pre, pb, _ = before[index]
        require(post['unmasked_logits_metadata'] == f'prefix-{index}.json'
                and post['request_id'] == pre['request_id'] and post['token_prefix'] == pre['token_prefix'],
                'wrong before/after request binding')
        values = floats(directory, pb['verify-logits'])
        width = 248320
        ids = [max(range(width), key=lambda v: values[r * width + v]) for r in range(4)]
        drafts = pre['mtp']['verification_tokens'][1:]
        accepted = 0
        while accepted < 3 and ids[accepted] == drafts[accepted]:
            accepted += 1
        require(post['mtp']['accepted_drafts'] == accepted and
                post['mtp']['emitted_ids'] == drafts[:accepted] + [ids[accepted]],
                'greedy acceptance/correction disagrees with full unmasked target logits')
        require(accepted == 0, 'this frozen actual case is the rejection-zero control; do not generalize')
        states = compare(directory, before[index], directory, pair, logits=False)
        require(states['all_compared_storage_exact'], 'rejection-zero selected states differ from anchor snapshot')
        acceptance.append({'binding': binding, 'accepted_drafts': accepted, 'draft_ids': drafts,
            'unmasked_target_argmax_ids': ids, 'emitted_ids': post['mtp']['emitted_ids'],
            'committed_state_context': post['seq_len'], 'states_vs_anchor': states})
    return {'status': 'DIAGNOSTIC', 'scope': 'existing actual alpha first-MTP verification, rejection-zero only; full target caches captured, not all-state equality against matched-arithmetic target or last-prefix21 qualification',
            'selected_metrics': selected_metrics, 'sibling_images_vs_target': images,
            'metadata': [x[2] for x in before], 'acceptance': acceptance, 'comparisons': comparisons,
            'actual_matrix_rows': [x[0]['actual_token_rows'] for x in before],
            'first_logit_metrics_mixed_vs_c1': metrics(floats(directory, before[3][1]['logits']), floats(directory, before[1][1]['logits'])),
            'target_reference_metadata': [x[2] for x in target_pairs],
            'end_prefill_target_states_equal': comparisons['mtp_prefill_mixed_vs_c1']['all_compared_storage_exact']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--target-capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve previous evidence')
    report = {'status': 'FAIL'}
    try:
        report = analyze(args.capture, args.target_capture)
    except Exception as error:
        report['error'] = repr(error)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'error', 'actual_matrix_rows')}))
    return 0 if report['status'] == 'DIAGNOSTIC' else 1


if __name__ == '__main__':
    raise SystemExit(main())
