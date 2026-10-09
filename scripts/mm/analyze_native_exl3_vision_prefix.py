#!/usr/bin/env python3
"""Validate bounded last-identical-prefix dumps and describe their differences.

CPU-only, no numerical pass threshold. Post-target states may already contain
upstream differences; this comparison does not attribute their origin.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(directory, index):
    path = directory / f'prefix-{index}.json'
    record = json.loads(path.read_text())
    require(record['phase'] == 'post_target_before_sampling', 'wrong capture phase')
    require(1 <= record['num_reqs'] <= 4, 'unsupported batch')
    row = record['row']
    require(0 <= row < record['num_reqs'], 'invalid request row')
    require(record['request_ids'][row] == record['request_id'], 'request row mismatch')
    require(record['seq_len'] == len(record['token_prefix']) <= 1600,
            'context/prefix mismatch')
    require(record['token_prefix'][record['prompt_tokens']:] == record['output_prefix'],
            'committed output prefix mismatch')
    computed = record['num_computed_before_step']
    require(0 <= computed < record['seq_len'] and record['seq_lens'][row] == record['seq_len'],
            'invalid computed context')
    offsets = record['query_start_loc']
    begin, end = offsets[row:row + 2]
    require(len(offsets) == record['num_reqs'] + 1 and end - begin == record['seq_len'] - computed,
            'wrong selected query length')
    require(record['positions'][begin:end] == list(range(computed, record['seq_len']))
            and record['input_token_ids'][begin:end] == record['token_prefix'][computed:]
            and record['mrope_delta'] == 0 and record['own_image_features'] == 0,
            'this diagnostic requires the selected pure-text request')
    require(record['gdn_indices'][row] == record['gdn_slot'], 'recurrent slot mismatch')
    page = record['kv_block_table'][row * record['kv_block_table_cols']]
    require(record['kv_write_slots'][begin:end] ==
            [page * 1600 + t for t in range(computed, record['seq_len'])],
            'KV page/write position mismatch')
    blobs = {}
    total = 0
    for entry in record['blobs']:
        name, filename = entry['name'], entry['file']
        require(name not in blobs and Path(filename).name == filename,
                'duplicate blob or unsafe payload path')
        shape = entry['shape']
        require(shape and all(type(n) is int and n > 0 for n in shape), 'invalid shape')
        size = math.prod(shape) * {0: 4, 1: 2, 2: 2, 3: 1, 4: 4, 5: 8}[entry['dtype']]
        total += size
        require(size == entry['bytes'] and total <= 256 * 1024 * 1024,
                'payload byte bound/shape mismatch')
        payload = directory / filename
        require(payload.stat().st_size == size and digest(payload) == entry['sha256'],
                'payload size/checksum mismatch: ' + name)
        blobs[name] = entry
    require(len(blobs) >= 3 and 'logits' in blobs and 'gdn0-conv' in blobs
            and 'gdn0-ssm' in blobs, 'missing selected state/logits')
    return record, blobs, {'file': path.name, 'sha256': digest(path), 'payload_bytes': total}


def floats(directory, entry):
    raw = (directory / entry['file']).read_bytes()
    if entry['dtype'] == 0:
        values = array.array('f')
        values.frombytes(raw)
        if sys.byteorder != 'little':
            values.byteswap()
    elif entry['dtype'] == 1:
        values = [x[0] for x in struct.iter_unpack('<e', raw)]
    else:
        raise RuntimeError('float metrics require FP32/FP16, not raw FP8 storage')
    require(all(math.isfinite(x) for x in values), 'nonfinite captured value')
    return values


def metrics(reference, native):
    require(len(reference) == len(native), 'different operand lengths')
    squares = norm = maximum = 0.0
    changed = coordinate = 0
    for i, (a, b) in enumerate(zip(reference, native)):
        delta = abs(a - b)
        squares += delta * delta
        norm += a * a
        changed += delta != 0
        if delta > maximum:
            maximum, coordinate = delta, i
    return {'elements': len(reference), 'different_elements': changed,
            'max_abs': maximum, 'rel_l2': math.sqrt(squares / norm) if norm else None,
            'max_coordinate': coordinate, 'reference_at_max': reference[coordinate],
            'native_at_max': native[coordinate]}


def analyze(directory):
    loaded = [load(directory, i) for i in range(3)]
    records = [x[0] for x in loaded]
    require(all(x['token_prefix'] == records[2]['token_prefix'] for x in records),
            'not the last identical prefix')
    for record, blobs, _ in loaded:
        values = floats(directory, blobs['logits'])
        best = sorted(range(len(values)), key=lambda i: (-values[i], i))[:2]
        top = {'ids': best, 'logits': [values[i] for i in best],
               'margin': values[best[0]] - values[best[1]]}
        require(record['top2'] == top, 'recorded top2 differs from full logit row')
    reference = loaded[2][1]
    comparisons = {}
    for index, label in ((0, 'warm_c2_vs_c1'), (1, 'mixed_c4_vs_c1')):
        other = loaded[index][1]
        require(set(reference) <= set(other), 'missing canonical state')
        different = []
        for name, entry in reference.items():
            candidate = other[name]
            require((entry['dtype'], entry['shape']) == (candidate['dtype'], candidate['shape']),
                    'incompatible canonical state: ' + name)
            if name.startswith('attn'):
                require(all(entry[k] == candidate[k] for k in ('fp8_kind', 'k_scale', 'v_scale')),
                        'different FP8 storage contract')
            if entry['sha256'] != candidate['sha256']:
                different.append(name)
        detail = {name: metrics(floats(directory, reference[name]), floats(directory, other[name]))
                  for name in ('logits', 'gdn0-conv', 'gdn0-ssm')}
        logit_delta = detail['logits']['max_abs']
        comparisons[label] = {'common_blobs': len(reference), 'different_blobs': different,
            'selected_metrics': detail, 'reference_margin': records[2]['top2']['margin'],
            'margin_gt_2_post_target_logit_delta': records[2]['top2']['margin'] > 2 * logit_delta,
            'same_input_margin_bound_applicable': False}
    first_images = {e['mm_hash']: e for e in loaded[0][1].values() if 'mm_hash' in e}
    mixed_images = {e['mm_hash']: e for e in loaded[1][1].values() if 'mm_hash' in e}
    require(set(first_images) <= set(mixed_images), 'missing original image identity')
    images = {key: first_images[key]['sha256'] == mixed_images[key]['sha256'] for key in first_images}
    return {'status': 'DIAGNOSTIC', 'scope': 'post-target last-identical-prefix attribution; no parity pass',
            'metadata': [x[2] for x in loaded], 'same_full_token_prefix': True,
            'actual_batches': [x['num_reqs'] for x in records],
            'actual_token_rows': [x['actual_token_rows'] for x in records],
            'request_layout_checks': 'PASS', 'top2': [x['top2'] for x in records],
            'original_image_embeddings_storage_exact': images, 'comparisons': comparisons,
            'limits': ['Post-target states already differ; origin not established.',
                       'Physical rows do not establish padded matrix rows or kernel dispatch.',
                       'Target-only; no MTP commit/rejection qualification.',
                       'A failed margin bound does not establish harmlessness.']}


def analyze_paired(directory):
    """Compare prefill/first-decode states in the same C4 and C1 requests."""
    loaded = [load(directory, i) for i in range(4)]
    records = [x[0] for x in loaded]
    for before, after in ((records[0], records[1]), (records[2], records[3])):
        require(before['request_id'] == after['request_id']
                and before['matching_request_occurrence'] == after['matching_request_occurrence'],
                'paired phases belong to different requests')
        require(before['output_prefix'] == [] and after['output_prefix'] == [271]
                and before['num_computed_before_step'] == 0
                and after['num_computed_before_step'] == before['seq_len'],
                'expected full prefill then first decode')
        require(after['token_prefix'] == before['token_prefix'] + [271], 'paired token prefix mismatch')
    require(records[0]['token_prefix'] == records[2]['token_prefix']
            and records[1]['token_prefix'] == records[3]['token_prefix'], 'different arm prefixes')
    for record, blobs, _ in loaded:
        values = floats(directory, blobs['logits'])
        best = sorted(range(len(values)), key=lambda i: (-values[i], i))[:2]
        require(record['top2'] == {'ids': best, 'logits': [values[i] for i in best],
                'margin': values[best[0]] - values[best[1]]}, 'false paired top2')
    comparisons = {}
    for phase, i, j in (('prefill', 0, 2), ('first_decode', 1, 3)):
        native, reference = loaded[i][1], loaded[j][1]
        require(set(reference) <= set(native), 'missing paired canonical state')
        different = []
        for name, entry in reference.items():
            candidate = native[name]
            require((entry['dtype'], entry['shape']) == (candidate['dtype'], candidate['shape']),
                    'incompatible paired state: ' + name)
            if name.startswith('attn'):
                require(all(entry[k] == candidate[k] for k in ('fp8_kind', 'k_scale', 'v_scale')),
                        'different paired FP8 storage contract')
            if entry['sha256'] != candidate['sha256']:
                different.append(name)
        comparisons[phase] = {'common_blobs': len(reference), 'different_blobs': different,
            'selected_metrics': {name: metrics(floats(directory, reference[name]),
                                               floats(directory, native[name]))
                                 for name in ('logits', 'gdn0-conv', 'gdn0-ssm')}}
    rolls = {}
    for arm, i, j in (('mixed', 0, 1), ('c1', 2, 3)):
        before, after = loaded[i][1], loaded[j][1]
        layers = {}
        for name, entry in before.items():
            if not name.startswith('gdn') or not name.endswith('-conv'):
                continue
            require(entry['shape'][-1] == 3 and entry['dtype'] == 1
                    and after[name]['shape'] == entry['shape'], 'unsupported Conv temporal layout')
            a, b = floats(directory, entry), floats(directory, after[name])
            layers[name] = [sum(x != y for x, y in zip(a[t + 1::3], b[t::3])) for t in (0, 1)]
        require(layers, 'missing recurrent Conv states')
        rolls[arm] = {'retained_columns_storage_exact': all(v == [0, 0] for v in layers.values()),
                      'different_elements_per_retained_column': layers}
    fields = ('request_id', 'row', 'num_reqs', 'actual_token_rows', 'query_start_loc',
              'seq_lens', 'gdn_slot', 'matching_request_occurrence', 'top2')
    return {'status': 'DIAGNOSTIC', 'scope': 'same-instance prefill/first-decode state attribution',
            'metadata': [x[2] for x in loaded],
            'layouts': [{k: record[k] for k in fields} for record in records],
            'same_arm_prefixes': True, 'comparisons_mixed_vs_c1': comparisons,
            'same_instance_conv_roll': rolls,
            'limits': ['Correct Conv roll does not qualify SSM/KV arithmetic or MTP commits.',
                       'Capture changes admission timing; actual geometry is recorded.',
                       'Different prefill inputs must be separated from local decode error.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--paired', action='store_true', help='same-instance prefill/first-decode C4/C1 pairs')
    args = parser.parse_args()
    print(json.dumps(analyze_paired(args.directory) if args.paired else analyze(args.directory), indent=2))
