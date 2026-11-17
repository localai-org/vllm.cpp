#!/usr/bin/env python3
"""Verify the existing all-block capture, then replay first-image blocks or merger.

Descriptive diagnosis only: no new golden, numerical pass, or serving claim.
The native utility shares the production typed block and accepts frozen inputs
only in its diagnostic mode. Run without another GPU instance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

CAPTURE_SHA256 = '08e4e08cb266c4a311edab62252629523751bc33b9ae6064fb7ce513f52756a0'
CONFIG_SHA256 = 'dab9abc478e1b1b928ef354f27a26fc90e8191a4a3ce5f3d19aa21558216b267'
REFERENCE_IMAGE = 'sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, report):
    manifest_path = args.reference / 'vision-boundaries-capture.json'
    require(digest(manifest_path) == CAPTURE_SHA256, 'existing all-block capture changed')
    require(digest(args.model / 'config.json') == CONFIG_SHA256, 'checkpoint config differs')
    manifest = json.loads(manifest_path.read_text())
    captures = manifest['worker']['captures']
    require(manifest['image'] == REFERENCE_IMAGE and manifest['worker']['encoder_calls'] == 2
            and len(captures) == 382, 'incomplete/wrong actual worker capture')
    require(manifest['worker']['tower_config'] == {'depth': 27, 'norm_eps': [1e-6] * 27,
            'dtype': 'torch.float16'}, 'wrong captured tower configuration')
    if args.merger:
        require(not args.layers and not args.operators_reference and not args.stage_reference and not args.computed_only,
                'merger mode cannot select blocks/operator captures')
    else:
        require(args.layers and len(args.layers) <= 5 and len(set(args.layers)) == len(args.layers)
                and all(0 <= layer < 27 for layer in args.layers), 'select one to five distinct blocks')
    first = {}
    for entry in captures:
        name = entry['name']
        if name == 'tower-input-0' and name in first:
            break
        require(name not in first, 'ambiguous first-image boundary')
        filename = entry['file']
        require(Path(filename).name == filename, 'invalid captured boundary path')
        path = args.reference / filename
        require(digest(path) == entry['sha256'], 'captured boundary hash differs: ' + name)
        size = {'torch.float16': 2, 'torch.int64': 8, 'torch.int32': 4}[entry['dtype']]
        for dim in entry['shape']:
            require(isinstance(dim, int) and dim > 0, 'invalid captured shape')
            size *= dim
        require(path.stat().st_size == size, 'captured shape/byte mismatch')
        first[name] = entry
    require(first['tower-input-0']['shape'] == [768, 1536], 'wrong small input geometry')
    grid = first['tower-kwargs-grid_thw']
    import struct
    require(struct.unpack('<3q', (args.reference / grid['file']).read_bytes()) == (1, 24, 32),
            'wrong captured image grid')
    for layer in range(27):
        require(first[f'block{layer}-input-0']['shape'] == [768, 1, 1152]
                and first[f'block{layer}-output']['shape'] == [768, 1, 1152], 'wrong block geometry')
        if layer:
            require(first[f'block{layer}-input-0']['sha256'] == first[f'block{layer-1}-output']['sha256'],
                    'captured reference block chain differs')
    if args.stage_reference:
        require(args.layers in ([12], [26]) and not args.operators_reference,
                'captured operator mode requires one block12/26 only')
        path = args.stage_reference / 'capture.json'
        captured = json.loads(path.read_text())
        layer = args.layers[0]
        require(captured['status'] == 'CAPTURED' and captured['block'] == layer
                and captured['reference_image'] == REFERENCE_IMAGE
                and captured['capture_sha256'] == CAPTURE_SHA256
                and captured['config_sha256'] == CONFIG_SHA256
                and captured['reference_input_sha256'] == first[f'block{layer}-input-0']['sha256'],
                'wrong pinned block reference/input')
        from capture_native_exl3_vision_reference import EXPECTED
        require(captured['runtime'] == EXPECTED and captured['source_sha256']['block'] ==
                '22f02a1ee1faba276a93243d23ecada1094484e39e5bde79452d8a44f32238ba',
                'wrong executing reference runtime/source')
        columns = {'input': 1152, 'norm1': 1152, 'qkv': 3456, 'q': 1152, 'k': 1152, 'v': 1152,
                   'rotated_q': 1152, 'rotated_k': 1152, 'attention': 1152, 'projection': 1152,
                   'residual1': 1152, 'norm2': 1152, 'fc1': 4304, 'gelu': 4304, 'fc2': 1152, 'output': 1152}
        require(set(captured['stages']) == set(columns), 'missing/unknown captured operator')
        for name, width in columns.items():
            entry = captured['stages'][name]
            filename = entry['file']
            require(Path(filename).name == filename and entry['dtype'] == 'torch.float16',
                    'wrong stage path/dtype')
            size = 1
            for dim in entry['shape']:
                require(isinstance(dim, int) and not isinstance(dim, bool) and 0 < dim <= 4304,
                        'invalid stage shape')
                size *= dim
            require(size == 768 * width, 'wrong captured operator size')
            require(digest(args.stage_reference / filename) == entry['sha256']
                    and (args.stage_reference / filename).stat().st_size == size * 2,
                    'captured operator hash/bytes differ: ' + name)
        require(captured['stages']['input']['sha256'] == first[f'block{layer}-input-0']['sha256'],
                'actual stage input differs')
        require(captured['matches_original_block_output'] is True and
                captured['stages']['output']['sha256'] == first[f'block{layer}-output']['sha256'],
                'standalone block does not reproduce original output; diagnose reference before native replay')
        report['stage_reference_manifest_sha256'] = digest(path)
        report['stage_reference_input_chain_verified'] = True
        report['scope'] += ' All captured block stages isolated; no composed tower execution.'
    if args.merger:
        shapes = {'merger-input-0': [768, 1, 1152], 'merger-norm-input-0': [768, 1, 1152],
                  'merger-norm-output': [768, 1, 1152], 'merger-fc1-input-0': [192, 4608],
                  'merger-fc1-output-0': [192, 4608], 'merger-fc2-input-0': [192, 4608],
                  'merger-fc2-output-0': [192, 5120], 'merger-output': [192, 5120]}
        require(all(first[name]['shape'] == shape and first[name]['dtype'] == 'torch.float16'
                    for name, shape in shapes.items()), 'wrong merger boundary geometry/dtype')
        for a, b in [('block26-output', 'merger-input-0'), ('merger-input-0', 'merger-norm-input-0'),
                     ('merger-norm-output', 'merger-fc1-input-0'), ('merger-fc2-output-0', 'merger-output')]:
            require(first[a]['sha256'] == first[b]['sha256'], 'captured merger chain differs')
        report.update(capture_sha256=CAPTURE_SHA256, config_sha256=CONFIG_SHA256,
                      binary_sha256=digest(args.binary), verified_first_image_boundaries=len(first),
                      hashes_verified_before_native=True, merger_input_chain_verified=True)
        report['scope'] += ' Exact merger input and isolated suboperations only; no composed tower execution.'
        command = [str(args.binary.resolve()), '--merger', str(args.model.resolve()),
                   str(args.reference.resolve()), str(args.output.resolve())]
        report['command'] = command
        report['native_exit_code'] = subprocess.run(command).returncode
        require(args.output.exists(), 'native replay did not write diagnostic output')
        result = json.loads(args.output.read_text())
        require(report['native_exit_code'] == 0 and result['status'] == 'DIAGNOSTIC', 'native replay failed')
        require(result['composed_tower_executed'] is False and result['reference_tier_hits'] == 0,
                'unexpected full tower/fallback execution')
        require(set(result['exact_upstream_stage_replay']) == {'norm', 'fc1', 'gelu', 'output'},
                'missing isolated merger operators')
        require(result['same_input_repeat_storage_exact'] is True, 'same-input native merger repeat differs')
        report['status'] = 'DIAGNOSTIC'
        return
    if args.operators_reference:
        require(args.layers == [0], 'isolated operator mode requires only block0')
        expected_manifests = {
            'qkv-real-replays.json': '373c645b6d7591109eb620add321b5c11f920d87b2852f6d5cefdbf59ba7820e',
            'vision-rope-reference.json': '982f5f46a3a4ad1222aab64e6df9f18a2dea5f26dcdee12c81195cc2bacd7b7b',
            'attention-real-replays.json': '119e789a988b382b60d44aaeea9db90d8d13f6ebbfef1530ff90b021e301daa0'}
        manifests = {}
        for name, sha in expected_manifests.items():
            path = args.operators_reference / name
            require(digest(path) == sha, 'existing operator manifest changed: ' + name)
            manifests[name] = json.loads(path.read_text())
            require(manifests[name]['reference_image'] == REFERENCE_IMAGE, 'wrong operator oracle')
        qkv = manifests['qkv-real-replays.json']['cases'][0]
        rope = manifests['vision-rope-reference.json']
        attention = manifests['attention-real-replays.json']['cases'][0]
        require(qkv['input']['sha256'] == first['block0-norm1-output']['sha256'], 'QKV reference input differs')
        require(rope['combined_cos']['sha256'] == first['encoder-metadata-rotary_pos_emb_cos']['sha256']
                and rope['combined_sin']['sha256'] == first['encoder-metadata-rotary_pos_emb_sin']['sha256'],
                'operator rotary metadata differs')
        require(rope['cases'][0]['q_in']['sha256'] == qkv['files']['q']['sha256']
                and rope['cases'][0]['k_in']['sha256'] == qkv['files']['k']['sha256'], 'rotary input chain differs')
        require(attention['files']['q']['sha256'] == rope['cases'][0]['q_out']['sha256']
                and attention['files']['k']['sha256'] == rope['cases'][0]['k_out']['sha256']
                and attention['files']['v']['sha256'] == qkv['files']['v']['sha256'], 'attention input chain differs')
        require(struct.unpack('<I', struct.pack('<f', attention['scale']))[0] == 0x3df15bef
                and attention['causal'] is False, 'operator scale/mask differs')
        for entry in [*qkv['files'].values(), *attention['files'].values()]:
            require(Path(entry['file']).name == entry['file'], 'invalid operator filename')
            path = args.operators_reference / entry['file']
            require(digest(path) == entry['sha256'], 'operator boundary differs: ' + entry['file'])
        report['operator_manifest_sha256'] = expected_manifests
        report['operator_input_chain_verified'] = True
        report['scope'] += ' Isolated block0 stage replay only; no composed tower execution.'
    report.update(capture_sha256=CAPTURE_SHA256, config_sha256=CONFIG_SHA256,
                  binary_sha256=digest(args.binary), verified_first_image_boundaries=len(first),
                  selected_layers=args.layers, hashes_verified_before_native=True)
    command = [str(args.binary.resolve()), '--blocks', str(args.model.resolve()),
               str(args.reference.resolve()), str(args.output.resolve()), ','.join(map(str, args.layers))]
    if args.operators_reference:
        command.append(str(args.operators_reference.resolve()))
    if args.stage_reference:
        command.extend(['--stage-reference', str(args.stage_reference.resolve())])
    if args.computed_only:
        command.append('--computed-only')
    report['command'] = command
    report['native_exit_code'] = subprocess.run(command).returncode
    require(args.output.exists(), 'native replay did not write diagnostic output')
    result = json.loads(args.output.read_text())
    require(report['native_exit_code'] == 0 and result['status'] == 'DIAGNOSTIC', 'native replay failed')
    if args.computed_only:
        require(result['computed_only'] is True and result['reference_outputs_injected'] is False
                and result['composed_tower_executed'] is False, 'computed-only execution contract differs')
        for block in result['blocks']:
            require('composed_output' not in block and 'only_norm_references_output' not in block
                    and not block.get('exact_upstream_stage_replay')
                    and 'exact_upstream_final_add' not in block, 'unexpected injected replay')
        report['scope'] = 'Computed same-reference-input blocks only; reference outputs compared on host, never injected; no full tower.'
        report['reference_outputs_injected'] = False
    require([block['block'] for block in result['blocks']] == args.layers, 'wrong selected blocks executed')
    require(result['reference_tier_hits'] == 0, 'reference fallback executed')
    report['status'] = 'DIAGNOSTIC'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'model', 'reference', 'output', 'execution'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--layers', nargs='+', type=int)
    parser.add_argument('--computed-only', action='store_true',
                        help='compute selected blocks only; never inject intermediate references or run a full tower')
    parser.add_argument('--merger', action='store_true',
                        help='replay existing merger boundaries only; no tower or selected blocks')
    parser.add_argument('--stage-reference', type=Path,
                        help='bind separately captured actual pinned block12/26 stages; no tower execution')
    parser.add_argument('--operators-reference', type=Path,
                        help='bind existing QKV/rotary/attention references; isolated block0 only, no full tower')
    args = parser.parse_args()
    if args.output.exists() or args.execution.exists():
        parser.error('output/receipt exists; preserve previous evidence')
    report = {'status': 'FAIL', 'scope': __doc__.strip()}
    code = 1
    try:
        run(args, report)
        code = 0
    except Exception as error:
        report['error'] = repr(error)
    args.execution.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'native_exit_code', 'error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
