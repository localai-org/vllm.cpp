#!/usr/bin/env python3
"""Bounded teacher-forced native shape diagnosis; never a greedy-parity waiver."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

from replay_native_exl3_vision_first_gdn import CONFIG_SHA, BLOCK_SOURCE, validate_model_sources


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_plan(capture, path):
    plan = json.loads(path.read_text())
    require(plan['schema'] == 'native-target-trajectory-v1', 'wrong trajectory schema')
    tokens, context = plan['teacher_tokens'], plan['initial_context']
    require(type(context) is int and 0 < context < 1600 and isinstance(tokens, list)
            and 1 <= len(tokens) <= 32 and context + len(tokens) < 1600
            and all(type(t) is int and 0 <= t < 248320 for t in tokens), 'invalid teacher window')
    metadata = []
    for key in ('c1_checkpoint', 'c4_checkpoint'):
        pin = plan[key]
        require(Path(pin['file']).name == pin['file'], 'invalid checkpoint path')
        file = capture / pin['file']
        require(not file.is_symlink() and file.stat().st_size <= 1024 * 1024
                and digest(file) == pin['sha256'], 'checkpoint metadata identity differs')
        item = json.loads(file.read_text())
        require(item['seq_len'] == context and not item.get('mtp')
                and type(item['mrope_delta']) is int and context + item['mrope_delta'] >= 0
                and (plan.get('image_decode') is True or item['mrope_delta'] == 0),
                'unsupported seed context/position geometry')
        names, total = set(), 0
        for entry in item['blobs']:
            require(Path(entry['file']).name == entry['file'] and entry['name'] not in names,
                    'unsafe/duplicate checkpoint blob')
            blob = capture / entry['file']
            require(not blob.is_symlink() and 0 < blob.stat().st_size <= 4 * 1024**2
                    and blob.stat().st_size == entry['bytes'] and digest(blob) == entry['sha256'],
                    'checkpoint payload identity differs')
            names.add(entry['name'])
            total += entry['bytes']
        state_names = {f'gdn{i}-{kind}' for i in range(48) for kind in ('conv', 'ssm')}
        state_names |= {f'attn{i}-{kind}' for i in range(16) for kind in ('k', 'v')}
        require(state_names <= names and total <= 256 * 1024**2, 'incomplete/unbounded target checkpoint')
        metadata.append(item)
    require(metadata[0]['mrope_delta'] == metadata[1]['mrope_delta'], 'different completion positions')
    if plan.get('image_decode'):
        require(plan['image_decode'] is True and all(m['own_image_features'] == 1 for m in metadata),
                'missing selected image identity')
        images = []
        for m in metadata:
            selected = [f for f in m['image_features'] if f['request_id'] == m['request_id']]
            require(len(selected) == 1, 'wrong selected image attribution')
            payloads = [e for e in m['blobs'] if e.get('mm_hash') == selected[0]['hash']]
            require(len(payloads) == 1, 'missing selected image tensor')
            images.append((selected[0]['hash'], selected[0]['offset'], selected[0]['length'], payloads[0]['sha256']))
        require(images[0] == images[1], 'selected image embeddings/order differ between seeds')
    require(metadata[0]['token_prefix'] == metadata[1]['token_prefix'], 'different seed token histories')
    return plan


def validate_result(result, plan):
    require(result['status'] == 'DIAGNOSTIC' and result['reference_tier_hits'] == 0
            and result['teacher_tokens'] == plan['teacher_tokens']
            and result['initial_context'] == plan['initial_context'], 'wrong native trajectory execution')
    require(result.get('image_decode', False) is bool(plan.get('image_decode', False)),
            'wrong multimodal decode route')
    require(len(result['steps']) == len(plan['teacher_tokens']), 'incomplete teacher window')
    for i, step in enumerate(result['steps']):
        require(step['step'] == i and step['teacher_token'] == plan['teacher_tokens'][i]
                and step['position'] == plan['initial_context'] + i, 'wrong forced token or position')
        if plan.get('image_decode'):
            require(type(result['mrope_delta']) is int and
                    step['mrope_position'] == step['position'] + result['mrope_delta'],
                    'wrong image completion axis coordinate')
        for key in ('identical_incoming_single_step', 'evolving_shared_seed', 'evolving_separate_serving_seeds'):
            row = step[key]
            require(row['states']['compared_states'] == 128 and row['logits']['elements'] == 248320
                    and all(math.isfinite(row['logits'][k]) for k in ('max_abs', 'rel_l2', 'reference_margin', 'candidate_margin')),
                    'incomplete/nonfinite full state/head metrics')
        for key in ('c1_state_sha256', 'c4_shared_state_sha256', 'c4_own_state_sha256'):
            require(len(step[key]) == 128 and all(isinstance(h, str) and len(h) == 64 for h in step[key].values()),
                    'missing complete state fingerprints')
    for key in ('same_geometry_repeat', 'same_geometry_moved_row_slot_page'):
        control = result[key]
        require(len(control['steps']) == len(plan['teacher_tokens']) and control['inactive_poisoned_bytes_checked'] > 0,
                'missing repeat/slot/poison control')
        for i, step in enumerate(control['steps']):
            require(step['step'] == i and step['states']['compared_states'] == 128
                    and step['states']['all_storage_exact'] is True and not step['states']['different_states']
                    and step['logits']['elements'] == 248320 and step['logits']['storage_exact'] is True
                    and step['logits']['different_elements'] == 0 and step['logits']['max_abs'] == 0,
                    'same-geometry repeat/slot control differs')
    require(set(result['inactive_poisoned_bytes_checked']) == {'c1', 'c4_shared', 'c4_own'}
            and all(n > 0 for n in result['inactive_poisoned_bytes_checked'].values()), 'missing inactive-owner checks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'capture', 'plan', 'binary', 'output', 'execution'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.execution.exists():
        parser.error('preserve previous trajectory evidence')
    report = {'status': 'FAIL', 'scope': __doc__.strip()}
    try:
        plan = validate_plan(args.capture, args.plan)
        require(digest(args.model / 'config.json') == CONFIG_SHA, 'pinned config differs')
        report['selected_model_sources'] = validate_model_sources(args.model, BLOCK_SOURCE)
        report.update(plan_sha256=digest(args.plan), binary_sha256=digest(args.binary),
                      input_hashes_verified_before_native=True)
        command = [str(args.binary.resolve()), str(args.model.resolve()), str(args.capture.resolve()),
                   str(args.output.resolve()), '--trajectory', str(args.plan.resolve())]
        report['command'] = command
        report['native_exit_code'] = subprocess.run(command).returncode
        require(report['native_exit_code'] == 0 and args.output.exists(), 'native trajectory failed')
        result = json.loads(args.output.read_text())
        require(result['plan_sha256'] == report['plan_sha256'], 'executed trajectory plan differs')
        if plan.get('image_decode'):
            expected = json.loads((args.capture / plan['c1_checkpoint']['file']).read_text())['mrope_delta']
            require(result['mrope_delta'] == expected, 'executed image position delta differs from admitted checkpoint')
        validate_result(result, plan)
        report['status'] = 'DIAGNOSTIC'
        code = 0
    except Exception as error:
        report['error'] = repr(error)
        code = 1
    args.execution.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'native_exit_code', 'error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
