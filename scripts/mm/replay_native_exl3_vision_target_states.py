#!/usr/bin/env python3
"""Source-pinned full native C1 target state replay, never a serving qualification."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess

from analyze_native_exl3_vision_prefix import digest, load as load_ordinary, metrics, require
from analyze_native_exl3_vision_mtp_prefix import load
from replay_native_exl3_vision_first_gdn import BLOCK_SOURCE, CONFIG_SHA, validate_model_sources
from native_vision_attribution_pins import load_attribution_pins


def shard_digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def analyze_layer3_substages(directory):
    """Compare logical row0 only; other rows intentionally have different tokens."""
    manifest = directory / 'manifest.tsv'
    require(0 < manifest.stat().st_size <= 16 * 1024, 'substage manifest bound')
    rows, bindings = {}, []
    for line in manifest.read_text().splitlines():
        step, layer, stage, dtype, count, columns, size, filename = line.split('\t')
        step, layer, count, columns, size = map(int, (step, layer, count, columns, size))
        require(step in (0, 1) and layer == 3 and count == 4 and dtype in ('f16', 'f32')
                and Path(filename).name == filename, 'invalid substage layout')
        require(0 < size <= 128 * 1024 and size == count * columns * {'f16': 2, 'f32': 4}[dtype],
                'invalid substage size')
        path = directory / filename
        require(path.stat().st_size == size and (step, stage) not in rows, 'missing/duplicate substage')
        raw = path.read_bytes()
        rows[step, stage] = dtype, columns, raw[:size // count]
        bindings.append({'file': filename, 'sha256': digest(path), 'bytes': size,
                         'step': step, 'layer': layer, 'dtype': dtype, 'rows': count, 'cols': columns})
    stages = {'post_input_norm', 'post_input_norm_DIRECT', 'post_input_norm_RECHECK',
              'attn_qgate', 'attn_key_raw', 'attn_value', 'attn_q_rope', 'attn_k_rope',
              'attn_gate', 'attn_core', 'attn_gated', 'block_out', 'post_attn_norm', 'mlp_out'}
    require(set(rows) == {(step, stage) for step in (0, 1) for stage in stages}, 'incomplete substages')
    result = {}
    for stage in sorted(stages):
        dtype, columns, a = rows[0, stage]
        other_dtype, other_columns, b = rows[1, stage]
        require((dtype, columns) == (other_dtype, other_columns), 'different substage shapes')
        fmt = '<e' if dtype == 'f16' else '<f'
        left = [x[0] for x in struct.iter_unpack(fmt, a)]
        right = [x[0] for x in struct.iter_unpack(fmt, b)]
        require(all(math.isfinite(x) for x in left + right), 'nonfinite substage')
        result[stage] = {**metrics(left, right), 'storage_exact': a == b}
    for step in (0, 1):
        require(rows[step, 'post_input_norm'][2] == rows[step, 'post_input_norm_DIRECT'][2]
                == rows[step, 'post_input_norm_RECHECK'][2], 'normalization buffer lifetime differs')
    return {'scope': 'layer3 logical row0 only; Q4 versus B4/Q1; no full-state qualification',
            'manifest_sha256': digest(manifest), 'bindings': bindings, 'stages': result,
            'norm_direct_recheck_exact': True}


def validate_result(result, native_exit, ordinary_recurrence, match_attention_reference=False):
    require(result['ordinary_recurrence_setting'] == ordinary_recurrence and result['reference_tier_hits'] == 0,
            'wrong recurrence setting or fallback')
    anchor = result['actual_serving_spec_anchor']
    require(anchor['compared_states'] == 128 and anchor['all_storage_exact'] is True
            and anchor['different_states'] == [] and anchor['all_four_logit_rows_storage_exact'] is True,
            'full actual serving anchor not reproduced')
    controls = [result['same_m_target_control']]
    if match_attention_reference:
        matched = result['same_m_matched_attention_control']
        require(matched['attention_mode'] == 'reference', 'wrong matched attention mode')
        controls.append(matched)
    for control in controls:
        logits = control['first_logit_metrics']
        require(control['compared_states'] == 128 and control['all_storage_exact'] is True
                and control['different_states'] == [] and logits['storage_exact'] is True
                and logits['elements'] == 248320 and logits['different_elements'] == 0
                and logits['max_abs'] == 0 and logits['rel_l2'] == 0,
                'full matched-M4 target state/logit control differs')
    require(native_exit == 0 and result['status'] == 'DIAGNOSTIC', 'full target native control failed')


def validate_acceptance_controls(result):
    cases = result['matched_acceptance_controls']
    require(len(cases) == 4, 'missing full-target acceptance length')
    for accepted, case in enumerate(cases):
        require(type(case['accepted_drafts']) is int and case['accepted_drafts'] == accepted
                and case['valid_context'] == 19 + accepted and case['correction_context'] == 20 + accepted
                and case['selected_snapshot_slot'] == (3, 1, 6, 4)[accepted]
                and type(case['teacher_forced_correction_token']) is int
                and 0 <= case['teacher_forced_correction_token'] < 248320,
                'wrong acceptance/correction selector or context')
        for name in ('prefix_states', 'correction_states'):
            state = case[name]
            require(state['compared_states'] == 128 and state['all_storage_exact'] is True
                    and state['different_states'] == [], 'full accepted/corrected target states differ')
        for name in ('prefix_logits', 'correction_logits'):
            logits = case[name]
            require(logits['elements'] == 248320 and logits['storage_exact'] is True
                    and logits['different_elements'] == 0 and logits['max_abs'] == 0 and logits['rel_l2'] == 0,
                    'full accepted/corrected logits differ')


def validate_ordinary_pair(result, native_exit):
    require(native_exit == 0 and result['status'] == 'DIAGNOSTIC' and result['reference_tier_hits'] == 0
            and result['ordinary_recurrence_setting'] == 'auto', 'ordinary replay failed or wrong arithmetic')
    require(set(result['actual_ordinary_anchors']) == {'c1', 'mixed_selected'}, 'missing actual ordinary anchor')
    for anchor in result['actual_ordinary_anchors'].values():
        logits = anchor['logits']
        require(anchor['compared_states'] == 128 and anchor['all_storage_exact'] is True
                and anchor['different_states'] == [] and logits['elements'] == 248320
                and logits['storage_exact'] is True and logits['different_elements'] == 0
                and logits['max_abs'] == 0 and logits['rel_l2'] == 0, 'actual ordinary state/head anchor differs')
    for name in ('shared_c1_incoming_new_m4_vs_m1', 'shared_c1_incoming_original_m4_vs_m1',
                 'shared_c1_incoming_new_vs_original_m4'):
        control = result[name]
        require(control['compared_states'] == 128 and control['logits']['elements'] == 248320,
                'incomplete shared-input shape diagnostic')
    layout = result['shared_c1_incoming_new_vs_original_m4']
    require(layout['all_storage_exact'] is True and layout['different_states'] == []
            and layout['logits']['storage_exact'] is True and layout['logits']['different_elements'] == 0
            and layout['logits']['max_abs'] == 0 and layout['logits']['rel_l2'] == 0,
            'matched-M4 selected-row/layout control differs')


def run(args, report):
    if os.environ.get('VT_DUMP_ACT_SUB'):
        directory = Path(os.environ['VT_DUMP_ACT_SUB'])
        require(directory.is_dir() and not any(directory.iterdir()), 'preserve existing substage evidence')
        report['optional_substage_capture'] = {'directory': str(directory), 'layer': 3}
    require(digest(args.model / 'config.json') == CONFIG_SHA, 'pinned target config changed')
    report['selected_model_sources'] = validate_model_sources(args.model, BLOCK_SOURCE)
    artifact_pins = load_attribution_pins()
    selected_pins = artifact_pins['adjacent_metadata' if args.ordinary_prefix_pair else 'first_mtp_metadata']
    pins = {b['file']: b['sha256'] for b in selected_pins}
    report['capture_metadata'] = []
    for index in range(4) if args.ordinary_prefix_pair else (2, 3):
        name = f'prefix-{index}.json'
        require(digest(args.capture / name) == pins[name], 'frozen actual MTP metadata changed')
        _, _, binding = load_ordinary(args.capture, index) if args.ordinary_prefix_pair else load(args.capture, name)
        report['capture_metadata'].append(binding)
    if args.ordinary_prefix_pair:
        require(args.ordinary_recurrence == 'auto' and not args.match_attention_reference
                and not args.check_acceptance_states and not report.get('optional_substage_capture')
                and not os.environ.get('VT_DUMP_ACT') and os.environ.get('VT_XPU_ATTENTION', 'auto') == 'auto',
                'ordinary pair requires original auto arithmetic without unrelated probes')
        original_pins = artifact_pins['original_last_prefix_metadata']
        report['original_last_prefix_metadata'] = []
        for index, pin in enumerate(original_pins):
            original = args.capture.parent / 'c2-alpha-prefix-v2'
            require(pin['file'] == f'prefix-{index}.json' and digest(original / pin['file']) == pin['sha256'],
                    'original last-prefix geometry changed')
            _, _, binding = load_ordinary(original, index)
            report['original_last_prefix_metadata'].append(binding)
    index = args.model / 'model.safetensors.index.json'
    mapping = json.loads(index.read_text())['weight_map']
    shards = sorted(set(mapping.values()))
    require(0 < len(shards) <= 128 and all(Path(s).name == s for s in shards), 'unsafe target shard set')
    report['whole_checkpoint_shards'] = []
    for shard in shards:
        path = args.model / shard
        require(0 < path.stat().st_size <= 16 * 1024**3, 'target shard bound exceeded')
        report['whole_checkpoint_shards'].append({'file': shard, 'bytes': path.stat().st_size, 'sha256': shard_digest(path)})
    report['index_sha256'] = digest(index)
    report['config_sha256'] = CONFIG_SHA
    report['binary_sha256'] = digest(args.binary)
    report['input_hashes_verified_before_native'] = True
    command = [str(args.binary.resolve()), str(args.model.resolve()), str(args.capture.resolve()), str(args.output.resolve())]
    if args.ordinary_prefix_pair:
        command.append('--ordinary-prefix-pair')
    report['ordinary_prefix_pair'] = args.ordinary_prefix_pair
    if args.match_attention_reference:
        require(args.ordinary_recurrence == 'reference' and os.environ.get('VT_XPU_ATTENTION', 'auto') == 'auto',
                'matched-attention diagnostic requires scalar recurrence and original auto attention')
        command.append('--match-attention-reference')
    if args.check_acceptance_states:
        require(args.match_attention_reference and not report.get('optional_substage_capture'),
                'acceptance control requires matched attention without two-step substage capture')
        command.append('--check-acceptance-states')
    report['check_acceptance_states'] = args.check_acceptance_states
    report['match_attention_reference'] = args.match_attention_reference
    report['command'] = command
    report['ordinary_recurrence_setting'] = args.ordinary_recurrence
    environment = dict(os.environ, VT_XPU_GDN_DECODE=args.ordinary_recurrence)
    report['native_exit_code'] = subprocess.run(command, env=environment, timeout=240).returncode
    require(args.output.exists(), 'full target native result missing')
    result = json.loads(args.output.read_text())
    report['native_result_sha256'] = digest(args.output)
    if args.ordinary_prefix_pair:
        validate_ordinary_pair(result, report['native_exit_code'])
        report['status'] = 'DIAGNOSTIC'
        return
    if report.get('optional_substage_capture'):
        report['optional_substage_analysis'] = analyze_layer3_substages(directory)
    if args.check_acceptance_states:
        validate_acceptance_controls(result)
        report['matched_acceptance_state_control'] = 'EXACT_DIAGNOSTIC_NOT_DEFAULT_QUALIFICATION'
    validate_result(result, report['native_exit_code'], args.ordinary_recurrence, args.match_attention_reference)
    report['status'] = 'DIAGNOSTIC'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'model', 'capture', 'output', 'execution'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--ordinary-recurrence', choices=('auto', 'reference'), default='auto')
    parser.add_argument('--match-attention-reference', action='store_true',
                        help='Additional matched XPU-attention attribution; retains original strict FAIL')
    parser.add_argument('--check-acceptance-states', action='store_true',
                        help='Teacher-forced complete target snapshots/corrections for0..3 accepted drafts')
    parser.add_argument('--ordinary-prefix-pair', action='store_true',
                        help='Replay adjacent actual prefix20/21 pairs and shared incoming-state M1/M4 shapes')
    args = parser.parse_args()
    if args.output.exists() or args.execution.exists():
        parser.error('preserve previous evidence')
    report = {'status': 'FAIL', 'scope': __doc__.strip()}
    try:
        run(args, report)
        code = 0
    except Exception as error:
        report['error'] = repr(error)
        code = 1
    args.execution.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'native_exit_code', 'error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
