#!/usr/bin/env python3
"""Validate frozen actual alpha prefix20/21 pairs; no numerical qualification.

The prefix20 post-target state supplies the prefix21 incoming state. Admission
may differ from the original failing capture; record this rather than claiming
the original greedy gate passed. CPU-only and fail-closed under Python -O.
"""
import argparse
import json
from pathlib import Path

from analyze_native_exl3_vision_prefix import digest, floats, load, metrics, require
from analyze_native_exl3_vision_mtp_prefix import PROMPT_IDS, STATE_NAMES, compare
from native_vision_attribution_pins import load_attribution_pins

PREFIX = [271, 588, 5346, 799, 16327, 1330, 11, 864, 369, 279, 15346,
          1067, 314, 33041, 1330, 12103, 7896, 1083, 264, 3074, 2702]
# Filled from the actual immutable four-capture producer, never inferred shapes.
METADATA_SHA = [
    '11f8e82413271dd67413af5bea09df963913a647ac26c04fb886c8ecf711beb2',
    'f4e7440b1d4828fc551068b3e43b6b44408d5aa47871fb20651ef5fec638300f',
    '4e6b1e5929c7a91024865aa1047e6641aa27e599edbeaf6cb6c128c40e386532',
    'eaeab4a4f49a42db9da9d83af19fdc3f237ad2530e79e6c2672f9e8a80d0e409']


def analyze(directory, original):
    require(len(METADATA_SHA) == 4, 'missing frozen adjacent metadata identities')
    require(all(digest(directory / f'prefix-{i}.json') == sha for i, sha in enumerate(METADATA_SHA)),
            'frozen adjacent metadata changed')
    pairs = [load(directory, i) for i in range(4)]
    for i, (record, blobs, _) in enumerate(pairs):
        length = 20 + i % 2
        context = 18 + length
        require(record['prompt_tokens'] == 18 and record['output_prefix'] == PREFIX[:length]
                and record['token_prefix'] == PROMPT_IDS + PREFIX[:length]
                and record['num_computed_before_step'] == context - 1
                and record['seq_len'] == context and record['actual_token_rows'] == record['num_reqs']
                and record['query_start_loc'] == list(range(record['num_reqs'] + 1))
                and record['max_query_len'] == 1 and 'mtp' not in record,
                'wrong frozen selected prefix or ordinary decode geometry')
        require(STATE_NAMES | {'logits'} <= blobs.keys(), 'missing complete target states/logits')
        require(blobs['logits']['dtype'] == 0 and blobs['logits']['shape'] == [248320], 'partial head')
        for name in STATE_NAMES:
            dtype, shape = ((1, [10240, 3]) if name.endswith('-conv') else
                            (0, [48, 128, 128]) if name.endswith('-ssm') else (3, [context, 4, 256]))
            require(blobs[name]['dtype'] == dtype and blobs[name]['shape'] == shape, 'wrong state layout')
        values = floats(directory, blobs['logits'])
        top = sorted(range(248320), key=lambda v: (-values[v], v))[:2]
        require(record['top2'] == {'ids': top, 'logits': [values[v] for v in top],
                                  'margin': values[top[0]] - values[top[1]]}, 'wrong recorded logits/top2')
    for previous, current in ((0, 1), (2, 3)):
        a, b = pairs[previous][0], pairs[current][0]
        require(a['request_id'] == b['request_id'] and a['gdn_slot'] == b['gdn_slot']
                and a['token_prefix'] == b['token_prefix'][:-1]
                and a['seq_len'] == b['num_computed_before_step']
                and a['top2']['ids'][0] == b['token_prefix'][-1]
                and a['num_reqs'] == b['num_reqs'], 'adjacent request/state/sample binding differs')
        for record in (a, b):
            page = record['kv_block_table'][record['row'] * record['kv_block_table_cols']]
            require(page == a['kv_block_table'][a['row'] * a['kv_block_table_cols']], 'adjacent KV owner moved')
    pins = load_attribution_pins()['original_last_prefix_metadata']
    require(len(pins) == 3 and all(pin['file'] == f'prefix-{i}.json' and
            digest(original / pin['file']) == pin['sha256'] for i, pin in enumerate(pins)),
            'original prefix21 metadata changed')
    old = [load(original, i) for i in range(3)]
    c1_anchor = compare(original, old[2], directory, pairs[3])
    require(c1_anchor['all_compared_storage_exact'], 'C1 post21 differs from original actual anchor')
    comparisons = {'incoming38_mixed_vs_c1': compare(directory, pairs[2], directory, pairs[0]),
                   'outgoing39_mixed_vs_c1': compare(directory, pairs[3], directory, pairs[1]),
                   'new_mixed_vs_original_mixed': compare(original, old[1], directory, pairs[1]),
                   'new_c1_vs_original_c1': c1_anchor}
    selected_metrics = {label: {name: metrics(floats(directory, pairs[a][1][name]),
                                            floats(directory, pairs[b][1][name]))
                               for name in ('gdn0-conv', 'gdn0-ssm', 'logits')}
                        for label, a, b in (('incoming38', 2, 0), ('outgoing39', 3, 1))}
    images = {entry['mm_hash']: entry for entry in pairs[1][1].values() if 'mm_hash' in entry}
    old_images = {entry['mm_hash']: entry for entry in old[1][1].values() if 'mm_hash' in entry}
    require(images.keys() == old_images.keys() and len(images) == 2, 'wrong sibling image identities')
    image_checks = [{'hash': key, 'shape': images[key]['shape'],
                     'storage_exact': images[key]['sha256'] == old_images[key]['sha256']} for key in sorted(images)]
    require(all(x['storage_exact'] for x in image_checks), 'sibling encoder output changed')
    return {'status': 'DIAGNOSTIC', 'scope': __doc__.strip(), 'metadata': [x[2] for x in pairs],
            'comparisons': comparisons, 'selected_metrics': selected_metrics, 'sibling_images': image_checks,
            'first_divergence_reproduced': pairs[1][0]['top2']['ids'][0] != pairs[3][0]['top2']['ids'][0],
            'actual_geometry': [{k: rec[k] for k in ('row', 'actual_token_rows', 'positions', 'seq_lens',
                               'query_start_loc', 'input_token_ids', 'gdn_indices', 'top2')} for rec, _, _ in pairs],
            'original_mixed_geometry': {k: old[1][0][k] for k in ('row', 'actual_token_rows', 'positions', 'seq_lens', 'top2')},
            'same_input_margin_bound_applicable': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'original', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve previous evidence')
    report = {'status': 'FAIL'}
    try:
        report = analyze(args.capture, args.original)
    except Exception as error:
        report['error'] = repr(error)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'error', 'first_divergence_reproduced')}))
    return 0 if report['status'] == 'DIAGNOSTIC' else 1


if __name__ == '__main__':
    raise SystemExit(main())
