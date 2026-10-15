#!/usr/bin/env python3
"""Pinned Python GDN core self-control on the captured first-layer operands.

This calls the installed _xpu_C core, not a reconstructed Torch formula. Z is
zero because only the pre-gated core and persistent Conv/SSM are compared.
Sibling states start at zero, as in the native local replay. This is neither a
whole reference model trajectory nor a quality or serving qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path

CONFIG_SHA = 'dab9abc478e1b1b928ef354f27a26fc90e8191a4a3ce5f3d19aa21558216b267'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a, b):
    import numpy as np
    require(a.shape == b.shape, 'different logical shape')
    x, y = a.astype(np.float64), b.astype(np.float64)
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'nonfinite core/state')
    d = y - x
    return {'elements': d.size, 'storage_exact': a.dtype == b.dtype and a.tobytes() == b.tobytes(),
            'different_elements': int(np.count_nonzero(d)), 'max_abs': float(np.max(np.abs(d))),
            'rel_l2': float(np.linalg.norm(d) / max(np.linalg.norm(x), 1e-30))}


def run(args, report):
    import numpy as np
    import torch
    import vllm._xpu_ops  # Registers the actual installed custom operator.
    import vllm_xpu_kernels
    from safetensors import safe_open

    require(sha(args.model / 'config.json') == CONFIG_SHA, 'model config changed')
    require(torch.__version__ == '2.13.0+xpu', 'unpinned Torch')
    require(torch.xpu.device_count() == 1, 'exactly one reference XPU required')
    properties = torch.xpu.get_device_properties(0)
    report['reference_device'] = str(properties)
    require(properties.device_id == 0xE223 and
            properties.name == 'Intel(R) Arc(TM) Pro B70 Graphics', 'wrong reference device')
    for key, name in [('vllm_xpu_source', '_xpu_ops.py'), ('gdn_python_source', 'qwen_gdn_linear_attn.py')]:
        root = Path(vllm._xpu_ops.__file__).parent
        path = root / name if key == 'vllm_xpu_source' else root / 'model_executor/layers/mamba/gdn' / name
        report[key] = {'path': str(path), 'sha256': sha(path)}
    root = Path(vllm_xpu_kernels.__file__).parent
    report['installed_kernel_binaries'] = [{'file': str(p), 'sha256': sha(p)} for p in root.rglob('*.so')]
    require(report['installed_kernel_binaries'], 'missing installed kernel identity')
    report['torch_version'] = torch.__version__
    report['operator_schema'] = str(torch.ops._xpu_C.gdn_attention.default._schema)
    manifest = args.stages / 'manifest.tsv'
    require(sha(manifest) == args.manifest_sha256, 'native stage manifest changed')
    native = json.loads(args.native_result.read_text())
    require(sha(args.native_result) == args.native_sha256 and native['status'] == 'DIAGNOSTIC',
            'native local replay identity/status changed')
    stage_metrics = json.loads(args.stage_metrics.read_text())
    bindings = {x['file']: x for x in stage_metrics['bindings']}
    stages = {}
    for line in manifest.read_text().splitlines():
        step, layer, name, dtype, rows, cols, size, filename = line.split('\t')
        require(Path(filename).name == filename and layer == '0' and step in ('0', '1'), 'unsafe stage')
        path = args.stages / filename
        require(path.stat().st_size == int(size) and sha(path) == bindings[filename]['sha256'],
                'stage payload changed')
        stages[step, name] = np.fromfile(path, dtype='<f2' if dtype == 'f16' else '<f4').reshape(int(rows), int(cols))

    index = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']
    weights = {}
    prefix = 'model.language_model.layers.0.linear_attn.'
    for name, dtype in [('conv1d.weight', torch.float16), ('A_log', torch.float32), ('dt_bias', torch.float16)]:
        key = prefix + name
        shard = index[key]
        require(Path(shard).name == shard, 'unsafe model shard')
        with safe_open(args.model / shard, framework='pt', device='cpu') as f:
            value = f.get_tensor(key)
        raw = value.contiguous().view(torch.uint8).numpy().tobytes()
        require(hashlib.sha256(raw).hexdigest() == native['weights'][key]['sha256'], 'source weight changed')
        weights[name] = value.to(dtype=dtype, device='xpu').contiguous()
    weights['conv1d.weight'] = weights['conv1d.weight'].reshape(10240, 4)

    report['cases'] = []
    saved = []
    with torch.inference_mode():
        for arm in (0, 1):
            md = json.loads((args.capture / f'prefix-{arm}.json').read_text())
            rows = 224 if arm == 0 else 507
            begin = 0 if arm == 0 else 283
            slot = md['gdn_slot']
            require(md['actual_token_rows'] == rows and md['row'] == (0 if arm == 0 else 3)
                    and md['query_start_loc'] == ([0, 224] if arm == 0 else [0, 1, 142, 283, 507])
                    and md['seq_len'] == 224 and md['mrope_delta'] == -176, 'wrong frozen geometry')
            mixed = stages[str(arm), 'gdn_mixed']
            require(mixed.shape == (rows, 10240), 'wrong input projection')
            packed = np.concatenate([mixed, np.zeros((rows, 6144), dtype=np.float16)], axis=1)
            ba = np.concatenate([stages[str(arm), 'gdn_ba_b'], stages[str(arm), 'gdn_ba_a']], axis=1)
            qkvz = torch.from_numpy(packed).to('xpu')
            ba = torch.from_numpy(ba).to('xpu')
            qsl = torch.tensor(md['query_start_loc'], dtype=torch.int32, device='xpu')
            initial = torch.tensor([False] if arm == 0 else [True, False, False, False], device='xpu')
            ids = md['gdn_indices']
            require(len(ids) == (1 if arm == 0 else 4) and len(set(ids)) == len(ids), 'aliased owners')
            result = []
            for control in ('original', 'repeat', 'moved_slot') if arm else ('original',):
                mapped = ids[:]
                selected = slot
                if control == 'moved_slot':
                    require(slot == 3, 'wrong selected slot')
                    mapped = [2 if x == 3 else 3 if x == 2 else x for x in mapped]
                    selected = 2
                state_indices = torch.tensor(mapped, dtype=torch.int32, device='xpu')
                # Installed chunk Conv indexes [slot, history, channel]. Native
                # owns [slot, channel, history]; normalize the logical witness.
                conv = torch.zeros((4, 3, 10240), dtype=torch.float16, device='xpu')
                ssm = torch.zeros((4, 48, 128, 128), dtype=torch.float32, device='xpu')
                core = torch.empty((rows, 48, 128), dtype=torch.float16, device='xpu')
                z = torch.empty_like(core)
                torch.ops._xpu_C.gdn_attention(core, z, qkvz, ba, 16, 48, 128, 128,
                    conv, ssm, weights['conv1d.weight'], None, 'silu', weights['A_log'], weights['dt_bias'],
                    1 if arm == 0 else 3, 0 if arm == 0 else 1, 0, initial, qsl, None, state_indices,
                    None, None, None, None, rows, 1, True)
                outputs = {'conv': conv[selected].transpose(0, 1).contiguous().cpu().numpy(), 'ssm': ssm[selected].cpu().numpy(),
                           'core': core[begin:begin + 224].cpu().numpy()}
                torch.xpu.synchronize()  # Bounded diagnostic observation only.
                entry = {'arm': arm, 'control': control, 'selected_slot': selected, 'payloads': {}}
                for name, value in outputs.items():
                    filename = f'arm-{arm}-{control}-{name}.bin'
                    path = args.output / filename
                    require(not path.exists(), 'preserve reference payload')
                    path.write_bytes(value.tobytes())
                    entry['payloads'][name] = {'file': filename, 'shape': list(value.shape),
                                             'dtype': str(value.dtype), 'sha256': sha(path)}
                if control != 'original':
                    entry['same_geometry'] = {k: compare(result[0][k], v) for k, v in outputs.items()}
                    require(all(x['storage_exact'] for x in entry['same_geometry'].values()),
                            'reference repeat/slot control differs')
                else:
                    for name, dtype, shape in [('conv', '<f2', (10240, 3)), ('ssm', '<f4', (48, 128, 128))]:
                        p = native['cases'][arm][name]
                        path = args.native_result.parent / p['file']
                        require(sha(path) == p['sha256'], 'native state payload changed')
                        entry['native_' + name] = compare(outputs[name], np.fromfile(path, dtype=dtype).reshape(shape))
                    core_start = 0 if arm == 0 else 282  # Native prefill core excludes leading decode row.
                    entry['native_core'] = compare(outputs['core'], stages[str(arm), 'gdn_core'][core_start:core_start + 224].reshape(224, 48, 128))
                    if arm:
                        entry['reference_c1_vs_c4'] = {k: compare(saved[0][k], v) for k, v in outputs.items()}
                    saved.append(outputs)
                result.append(outputs)
                report['cases'].append(entry)
    report['status'] = 'DIAGNOSTIC'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('model', 'capture', 'stages', 'native-result', 'stage-metrics', 'output'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--native-sha256', required=True)
    args = p.parse_args()
    args.output.mkdir()  # Never overwrite an earlier experiment.
    report = {'status': 'FAIL', 'scope': __doc__, 'input_arguments': {k: str(v) for k, v in vars(args).items()}}
    try:
        run(args, report)
    except Exception as e:
        report['error'] = f'{type(e).__name__}: {e}'
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    require(report['status'] == 'DIAGNOSTIC', report.get('error', 'incomplete core replay'))


if __name__ == '__main__':
    main()
