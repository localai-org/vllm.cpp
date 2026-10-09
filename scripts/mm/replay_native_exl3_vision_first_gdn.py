#!/usr/bin/env python3
"""Validate frozen mixed-case inputs, then run the native first-layer control.

External qualification diagnostic; missing/changed inputs fail before native
starts, including under Python -O. No numerical or serving parity waiver.
Run without another GPU instance. Binary and runtime are supplied explicitly.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
import subprocess

from analyze_native_exl3_vision_prefix import analyze, analyze_paired

CONFIG_SHA = 'dab9abc478e1b1b928ef354f27a26fc90e8191a4a3ce5f3d19aa21558216b267'
PAIRED_SHA = [
    '3434c67d7a632ec4d461297fb1c2af532fa59a6373468fe3b511c8d9318e8328',
    '0e1d84a47a3773a4982236d743b0caf19c771c594faeca18a9add9168ef6c07c',
    'f81eb75e5bee92d3a182ee8a12273bfbcda213c73e97ccae6bfffbcf64ee427a',
    '9fc06f1d14307c75b87bb15a3a1a49e99fd1a657ca5a852cdedf3395b975392e']
M4_SHA = [
    '6d9090eb702f1c6b688822d67efecbcf19071bd40bdd1cf7685983bfba883629',
    '46a6c10c289e5b2e7cd7d2feeec866efa56f62098e80012ed488b5dc4e9ac162',
    '8c884834f3692650c8fa0acc35c0b4960d8814b71d5120f0d92a1447b2bb1d71']


SOURCE = {
    "model.language_model.layers.0.input_layernorm.weight": {
        "dtype": "BF16",
        "sha256": "853b5989bfb677b1761540ee97c09bbf8d90fd825c9ae6ca74ef5853cba3c075",
        "shape": [
            5120
        ]
    },
    "model.language_model.layers.0.linear_attn.in_proj_qkv.mul1": {
        "dtype": "I32",
        "sha256": "20feb8ff6dbb9b28746f70893914cd343f8a92629ad812923fb29772030dd082",
        "shape": []
    },
    "model.language_model.layers.0.linear_attn.in_proj_qkv.suh": {
        "dtype": "F16",
        "sha256": "f8ee54a06132c35683940897d8199aaa91f57222af148e747add732370b2ba88",
        "shape": [
            5120
        ]
    },
    "model.language_model.layers.0.linear_attn.in_proj_qkv.svh": {
        "dtype": "F16",
        "sha256": "a6c6d0227db9b312a930af93f34f9d98141a7efddf4f21d79b6dacdec5dadacf",
        "shape": [
            10240
        ]
    },
    "model.language_model.layers.0.linear_attn.in_proj_qkv.trellis": {
        "dtype": "I16",
        "sha256": "79dee98050e4ba1fb0c2df661dfb3680fd14786cbfd9382ebb231acea90a6f3d",
        "shape": [
            320,
            640,
            64
        ]
    },
    "model.language_model.layers.0.linear_attn.in_proj_z.mul1": {
        "dtype": "I32",
        "sha256": "20feb8ff6dbb9b28746f70893914cd343f8a92629ad812923fb29772030dd082",
        "shape": []
    },
    "model.language_model.layers.0.linear_attn.in_proj_z.suh": {
        "dtype": "F16",
        "sha256": "7ca8cdc1dc099510b80e544e7301892dc82a2c6baf699b15d3b11f2d19582cef",
        "shape": [
            5120
        ]
    },
    "model.language_model.layers.0.linear_attn.in_proj_z.svh": {
        "dtype": "F16",
        "sha256": "60fc14deecfc20d5e75e9b4c7888a9921ceda2077b0ca9f7c3ea0c29e5d78d29",
        "shape": [
            6144
        ]
    },
    "model.language_model.layers.0.linear_attn.in_proj_z.trellis": {
        "dtype": "I16",
        "sha256": "923a44c6a94ad397bb9675b3ff368c40ea0239a5db3e586eceb9880fc975aa26",
        "shape": [
            320,
            384,
            64
        ]
    }
}
EMBEDDING_ROW = {
    "f16_sha256": "7dda586a74b47ae3727edb5b197d05870313a62e4788a27d4d11253c5436cf42",
    "source_dtype": "BF16",
    "source_sha256": "34f3d1434ffe3e7068202f426460d22195efa19c9b157dd65b6177d2d8319fd2",
    "token_id": 271
}

BLOCK_SOURCE = {
    "model.language_model.layers.0.linear_attn.A_log": {
        "dtype": "BF16",
        "sha256": "429d53cd15ec0ea1a2d638100c3bbdf8d53505e69110fb17ee09d055c1fdbd34",
        "shape": [
            48
        ]
    },
    "model.language_model.layers.0.linear_attn.conv1d.weight": {
        "dtype": "BF16",
        "sha256": "803a2591c4f5e7b1c2a7bdbbad78bbd0b68acdd0deab9552933ee7c337489dad",
        "shape": [
            10240,
            1,
            4
        ]
    },
    "model.language_model.layers.0.linear_attn.dt_bias": {
        "dtype": "BF16",
        "sha256": "969ce6b2149d20f0c365373fd89613167e373905c15f0e1cf5af62adf52228de",
        "shape": [
            48
        ]
    },
    "model.language_model.layers.0.linear_attn.in_proj_a.weight": {
        "dtype": "F16",
        "sha256": "6f6c67353cc409c6d4a347f8b27ec42250181f3af3eed95f8ab44e0bc1a7042c",
        "shape": [
            48,
            5120
        ]
    },
    "model.language_model.layers.0.linear_attn.in_proj_b.weight": {
        "dtype": "F16",
        "sha256": "e073eeb79d3cea4e3df65d17933fb09441d6f1fd03ba8a23b137d9056731b767",
        "shape": [
            48,
            5120
        ]
    },
    "model.language_model.layers.0.linear_attn.in_proj_qkv.mul1": {
        "dtype": "I32",
        "sha256": "20feb8ff6dbb9b28746f70893914cd343f8a92629ad812923fb29772030dd082",
        "shape": []
    },
    "model.language_model.layers.0.linear_attn.in_proj_qkv.suh": {
        "dtype": "F16",
        "sha256": "f8ee54a06132c35683940897d8199aaa91f57222af148e747add732370b2ba88",
        "shape": [
            5120
        ]
    },
    "model.language_model.layers.0.linear_attn.in_proj_qkv.svh": {
        "dtype": "F16",
        "sha256": "a6c6d0227db9b312a930af93f34f9d98141a7efddf4f21d79b6dacdec5dadacf",
        "shape": [
            10240
        ]
    },
    "model.language_model.layers.0.linear_attn.in_proj_qkv.trellis": {
        "dtype": "I16",
        "sha256": "79dee98050e4ba1fb0c2df661dfb3680fd14786cbfd9382ebb231acea90a6f3d",
        "shape": [
            320,
            640,
            64
        ]
    },
    "model.language_model.layers.0.linear_attn.in_proj_z.mul1": {
        "dtype": "I32",
        "sha256": "20feb8ff6dbb9b28746f70893914cd343f8a92629ad812923fb29772030dd082",
        "shape": []
    },
    "model.language_model.layers.0.linear_attn.in_proj_z.suh": {
        "dtype": "F16",
        "sha256": "7ca8cdc1dc099510b80e544e7301892dc82a2c6baf699b15d3b11f2d19582cef",
        "shape": [
            5120
        ]
    },
    "model.language_model.layers.0.linear_attn.in_proj_z.svh": {
        "dtype": "F16",
        "sha256": "60fc14deecfc20d5e75e9b4c7888a9921ceda2077b0ca9f7c3ea0c29e5d78d29",
        "shape": [
            6144
        ]
    },
    "model.language_model.layers.0.linear_attn.in_proj_z.trellis": {
        "dtype": "I16",
        "sha256": "923a44c6a94ad397bb9675b3ff368c40ea0239a5db3e586eceb9880fc975aa26",
        "shape": [
            320,
            384,
            64
        ]
    },
    "model.language_model.layers.0.linear_attn.norm.weight": {
        "dtype": "BF16",
        "sha256": "70d11f294923a94aac97bedd27e596dbf496746c6b9e3b2b7c38798a0d2522cc",
        "shape": [
            128
        ]
    },
    "model.language_model.layers.0.linear_attn.out_proj.mul1": {
        "dtype": "I32",
        "sha256": "20feb8ff6dbb9b28746f70893914cd343f8a92629ad812923fb29772030dd082",
        "shape": []
    },
    "model.language_model.layers.0.linear_attn.out_proj.suh": {
        "dtype": "F16",
        "sha256": "9066aec9e0ce9533ac836aa0fd08955a61eba3e02a4eda3e51e4a00da3a75554",
        "shape": [
            6144
        ]
    },
    "model.language_model.layers.0.linear_attn.out_proj.svh": {
        "dtype": "F16",
        "sha256": "38c094c3c6ead19b6d913d4282fb07a1c25acd53584e5e5b49d3f13c512fca5b",
        "shape": [
            5120
        ]
    },
    "model.language_model.layers.0.linear_attn.out_proj.trellis": {
        "dtype": "I16",
        "sha256": "1715cd7f722cd9c128c19cac2a2a12a8f64feeeb8fb89a06241c2f2bcb68b68e",
        "shape": [
            384,
            320,
            64
        ]
    }
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_metadata(directory, hashes):
    for index, checksum in enumerate(hashes):
        require(digest(directory / f'prefix-{index}.json') == checksum,
                'frozen request metadata differs')


def validate_model_sources(model, extra_sources=None):
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    verified = {}
    sources = dict(SOURCE)
    sources.update(extra_sources or {})
    for name, expected in [*sources.items(), ('model.language_model.embed_tokens.weight',
            {'dtype': 'BF16', 'shape': [248320, 5120], 'sha256': EMBEDDING_ROW['source_sha256']})]:
        filename = index[name]
        require(Path(filename).name == filename, 'unsafe checkpoint shard')
        path = model / filename
        with path.open('rb') as stream:
            header_size = struct.unpack('<Q', stream.read(8))[0]
            require(0 < header_size <= 2 * 1024 * 1024, 'checkpoint header exceeds bound')
            entry = json.loads(stream.read(header_size))[name]
            require(entry['dtype'] == expected['dtype'] and entry['shape'] == expected['shape'],
                    'pinned tensor storage differs: ' + name)
            begin, end = entry['data_offsets']
            count = math.prod(entry['shape']) * (4 if entry['dtype'] in ('I32', 'F32') else 2)
            require(0 <= begin < end and end - begin == count
                    and header_size + 8 + end <= path.stat().st_size, 'invalid tensor span')
            if name.endswith('embed_tokens.weight'):
                begin += 271 * 5120 * 2
                count = 5120 * 2
            require(count <= 64 * 1024 * 1024, 'selected tensor exceeds diagnostic budget')
            stream.seek(header_size + 8 + begin)
            checksum = hashlib.sha256()
            left = count
            while left:
                chunk = stream.read(min(left, 1024 * 1024))
                require(chunk, 'incomplete selected tensor')
                checksum.update(chunk)
                left -= len(chunk)
            require(checksum.hexdigest() == expected['sha256'], 'pinned tensor checksum differs: ' + name)
            verified[name] = expected
    return verified


def run(args, report):
    require(not (args.block and args.spec_block), 'choose one block mode')
    block_mode = args.block or args.spec_block
    require(digest(args.model / 'config.json') == CONFIG_SHA, 'pinned model config differs')
    report['model_sources_verified_before_native'] = validate_model_sources(args.model, BLOCK_SOURCE if block_mode else None)
    validate_metadata(args.capture, PAIRED_SHA)
    paired = analyze_paired(args.capture)
    report['paired_metadata'] = paired['metadata']
    if block_mode:
        require(args.norm_row and not args.m4_capture, 'block mode requires norm row and no M4 capture')
        require(args.norm_row.stat().st_size == 5120 * 2 and digest(args.norm_row) ==
                '34eb05c8f923ba017ce333f0ce43d40bd623a2e6b0088631e7f20110da2a1b88',
                'actual normalized token271 row differs')
    if args.m4_capture:
        validate_metadata(args.m4_capture, M4_SHA)
        other = analyze(args.m4_capture)
        require(other['actual_batches'] == [2, 4, 1]
                and other['actual_token_rows'] == [246, 4, 1], 'wrong original first-decode shapes')
        report['m4_metadata'] = other['metadata']
    report['config_sha256'] = CONFIG_SHA
    report['binary_sha256'] = digest(args.binary)
    report['hashes_verified_before_native'] = True
    command = [str(args.binary.resolve()), str(args.model.resolve()),
               str(args.capture.resolve()), str(args.output.resolve())]
    if block_mode:
        command = [str(args.binary.resolve()), '--spec-block' if args.spec_block else '--block', str(args.model.resolve()),
                   str(args.capture.resolve()), str(args.norm_row.resolve()), str(args.output.resolve())]
    if args.m4_capture:
        command.append(str(args.m4_capture.resolve()))
    report['command'] = command
    report['native_exit_code'] = subprocess.run(command, timeout=180).returncode
    require(args.output.exists(), 'native control produced no diagnostic result')
    result = json.loads(args.output.read_text())
    require(report['native_exit_code'] == 0 and result['status'] == 'DIAGNOSTIC',
            'native first-layer control failed')
    require(result['reference_tier_hits'] == 0, 'reference fallback executed')
    require(result['weights'] == (BLOCK_SOURCE if block_mode else SOURCE),
            'native source tensor provenance differs')
    if args.spec_block:
        require(result['all_compared_values_exact'] is True and result['initial_output']['storage_exact'] is True,
                'initial typed speculative output differs')
        require([c['accepted_drafts'] for c in result['cases']] == [0, 1, 2, 3], 'missing accepted prefix')
        require([c['selected_ssm_slot'] for c in result['cases']] == [3, 1, 6, 4], 'wrong selected state slots')
        require(result['query_shapes'] == {'ordinary': [1, 1, 1, 1], 'speculative': [4]}, 'wrong logical query shapes')
        for case in result['cases']:
            require(case['physical_m'] == 4 and case['num_accepted_tokens'] == case['accepted_drafts'] + 1,
                    'wrong shape or accepted selector')
            for field in ('snapshot_conv', 'snapshot_ssm', 'correction_output', 'correction_conv', 'correction_ssm'):
                require(case[field]['storage_exact'] is True, 'typed speculative state/output differs')
        report['result_sha256'] = digest(args.output)
        report['status'] = 'DIAGNOSTIC'
        return
    if block_mode:
        require([c['m'] for c in result['cases']] == [1, 509], 'wrong actual block shapes')
        for case in result['cases']:
            require(case['conv_vs_actual_serving']['storage_exact'] is True
                    and case['ssm_vs_actual_serving']['storage_exact'] is True,
                    'actual first-block persistent states not reproduced')
            entry = case['output']
            require(entry['dtype'] == 'F16' and entry['shape'] == [5120]
                    and Path(entry['file']).name == entry['file'], 'invalid block output row')
            path = args.output.parent / entry['file']
            require(path.stat().st_size == 5120 * 2 and digest(path) == entry['sha256'],
                    'block output row differs')
        report['result_sha256'] = digest(args.output)
        report['status'] = 'DIAGNOSTIC'
        return
    require(result['embedding_row'] == EMBEDDING_ROW, 'native embedding provenance differs')
    require([case['m'] for case in result['cases']] == [1, 4, 18, 246, 509], 'wrong controlled shapes')
    for case in result['cases']:
        for stage, columns in (('norm', 5120), ('chain', 16384), ('fixed_norm_projection', 16384)):
            entry = case[stage]
            require(entry['dtype'] == 'F16' and entry['shape'] == [columns]
                    and Path(entry['file']).name == entry['file'], 'invalid controlled row layout/path')
            path = args.output.parent / entry['file']
            require(path.stat().st_size == columns * 2 and digest(path) == entry['sha256'],
                    'controlled row payload differs')
        if case['m'] in (1, 509):
            require(case['chain_vs_actual_new_conv']['storage_exact'] is True,
                    'actual serving boundary is not reproduced')
        if case['m'] == 4 and args.m4_capture:
            require(case['row1_chain_vs_actual_new_conv']['storage_exact'] is True,
                    'actual M4 row1 boundary is not reproduced')
        if case['m'] > 1:
            require(case['repeated_row1_vs_row0']['storage_exact'] is True
                    and case['repeated_last_row_vs_row0']['storage_exact'] is True,
                    'identical activation rows disagree')
    report['result_sha256'] = digest(args.output)
    report['status'] = 'DIAGNOSTIC'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'model', 'capture', 'output', 'execution'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--m4-capture', type=Path)
    parser.add_argument('--block', action='store_true', help='shared first GDN block and persistent states')
    parser.add_argument('--spec-block', action='store_true', help='matched-M4 typed first-block accept/reject state control')
    parser.add_argument('--norm-row', type=Path, help='exact native token271 normalized row for block mode')
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
    print(json.dumps({key: report.get(key) for key in ('status', 'native_exit_code', 'error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
