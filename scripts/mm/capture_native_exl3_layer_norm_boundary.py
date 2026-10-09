#!/usr/bin/env python3
"""N1: installed LayerNorm R on frozen, full-shape block operands only.

No tower/LLM is constructed. Capture an original output and aten statistics,
check both against the prior worker output, and preserve all original files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

from capture_native_exl3_vision_reference import EXPECTED, REFERENCE_IMAGE
from replay_native_exl3_vision_blocks import CONFIG_SHA256, require


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_operands(root, expected_sha256, block):
    """Admission runs before importing Torch or creating a queue, including -O."""
    path = root / 'capture.json'
    require(path.stat().st_size <= 128 * 1024, 'capture manifest too large')
    require(digest(path) == expected_sha256, 'frozen capture identity differs')
    doc = json.loads(path.read_text())
    require(doc['status'] == 'CAPTURED' and doc['block'] == block and
            doc['matches_original_block_output'], 'original worker capture required')
    require(doc['runtime'] == EXPECTED and doc['config_sha256'] == CONFIG_SHA256,
            'capture runtime/config pin differs')
    result = []
    for norm, input_name in [('norm1', 'input'), ('norm2', 'residual1')]:
        entries = {}
        for role, stage in [('input', input_name), ('reference', norm)]:
            entry = doc['stages'][stage]
            name = entry['file']
            require(isinstance(name, str) and Path(name).name == name and
                    not (root / name).is_symlink(), 'invalid operand path')
            require(entry['dtype'] == 'torch.float16' and
                    entry['shape'] == [768, 1, 1152], 'original full shape/dtype required')
            operand = root / name
            require(operand.stat().st_size == 768 * 1152 * 2 and
                    digest(operand) == entry['sha256'], 'frozen operand bytes differ')
            entries[role] = entry | {'path': operand}
        result.append({'label': f'block{block}-{norm}', 'norm': norm, 'entries': entries,
                       'parameter_sha256': {role: doc['parameter_f16_sha256'][norm + '.' + role]
                                            for role in ('weight', 'bias')}})
    return result


def capture(args):
    cases = read_operands(args.reference, args.capture_sha256, args.block)
    require(args.runtime_image == REFERENCE_IMAGE, 'pinned reference image required')
    require(digest(args.model / 'config.json') == CONFIG_SHA256, 'model config differs')
    require(not args.output.exists(), 'new output directory required; never overwrite a receipt')
    import numpy as np
    import torch
    import importlib.metadata
    from safetensors import safe_open

    packages = {name: importlib.metadata.version(name)
                for name in ('torchvision', 'transformers', 'vllm')}
    runtime = {'torch': torch.__version__, 'torch_git': torch.version.git_version,
               'installed_packages': packages}
    require(torch.__version__ == EXPECTED['torch'] and
            torch.version.git_version == EXPECTED['torch_git'] and
            packages == {'torchvision': '0.28.0+xpu', 'transformers': '5.16.1', 'vllm': '0.30.0+xpu'} and
            torch.xpu.is_available(), 'pinned XPU runtime required')
    args.output.mkdir()
    index = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']
    eps = 1e-6
    report = {'status': 'CAPTURED', 'scope': 'N1 installed R, two frozen full-shape norms only',
              'runtime': runtime, 'runtime_image': args.runtime_image,
              'capture_sha256': args.capture_sha256, 'source_sha256': digest(Path(__file__)),
              'config_sha256': CONFIG_SHA256, 'device': torch.xpu.get_device_name(0),
              'epsilon_double': eps, 'epsilon_fp32_bits': struct.pack('<f', eps).hex(),
              'cases': []}
    torch_lib = Path(torch.__file__).parent / 'lib' / 'libtorch_xpu.so'
    report['installed_binary'] = {'name': torch_lib.name, 'sha256': digest(torch_lib),
                                  'bytes': torch_lib.stat().st_size}
    (args.output / 'torch-build-config.txt').write_text(torch.__config__.show())
    (args.output / 'dispatch.txt').write_text(torch._C._dispatch_dump_table('aten::native_layer_norm'))
    with torch.inference_mode():
        for case in cases:
            label, norm = case['label'], case['norm']
            entry = case['entries']['input']
            x = torch.from_numpy(np.frombuffer(entry['path'].read_bytes(), dtype=np.float16)
                                 .copy().reshape(entry['shape'])).to('xpu')
            parameters = {}
            for role in ('weight', 'bias'):
                key = f'model.visual.blocks.{args.block}.{norm}.{role}'
                name = index[key]
                require(Path(name).name == name, 'invalid model shard path')
                with safe_open(args.model / name, framework='pt', device='cpu') as shard:
                    value = shard.get_tensor(key)
                require(value.dtype == torch.bfloat16 and list(value.shape) == [1152],
                        'pinned affine shape/dtype differs')
                value = value.to(torch.float16)
                require(hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest() ==
                        case['parameter_sha256'][role], 'affine FP16 parameter identity differs')
                parameters[role] = value.to('xpu')
            weight, bias = parameters['weight'], parameters['bias']
            if os.environ.get('N1_TRACE_DIR'):
                os.environ['N1_TRACE_ACTIVE'] = '1'
            original = torch.nn.functional.layer_norm(x, [1152], weight, bias, eps)
            y, mean, rstd = torch.ops.aten.native_layer_norm(x, [1152], weight, bias, eps)
            repeat, repeat_mean, repeat_rstd = torch.ops.aten.native_layer_norm(x, [1152], weight, bias, eps)
            torch.xpu.synchronize()
            def raw(t):
                return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            frozen = case['entries']['reference']['path'].read_bytes()
            require(raw(original) == raw(y) == raw(repeat) == frozen,
                    'aten/original/repeat output does not reproduce captured worker norm')
            require(raw(mean) == raw(repeat_mean) and raw(rstd) == raw(repeat_rstd),
                    'R statistics repeat differs')
            files = {}
            for role, tensor in [('input', x), ('weight', weight), ('bias', bias),
                                 ('reference', y), ('mean', mean), ('rstd', rstd)]:
                data = raw(tensor if role not in ('mean', 'rstd') else tensor.float())
                filename = label + '-' + role + ('.float32' if role in ('mean', 'rstd') else '.float16')
                (args.output / filename).write_bytes(data)
                files[role] = {'file': filename, 'sha256': hashlib.sha256(data).hexdigest(),
                               'bytes': len(data), 'original_dtype': str(tensor.dtype),
                               'shape': list(tensor.shape), 'strides': list(tensor.stride()),
                               'pointer_alignment_mod64': tensor.data_ptr() % 64}
            # The profiler observes a third execution, checked against the same
            # original. Its kernel name/launch metadata are evidence, not a guess.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.XPU]) as profile:
                observed, _, _ = torch.ops.aten.native_layer_norm(x, [1152], weight, bias, eps)
                torch.xpu.synchronize()
            require(raw(observed) == frozen, 'R profiler changes the output being explained')
            profile.export_chrome_trace(str(args.output / (label + '-profile.json')))
            report['cases'].append({'label': label, 'shape': list(x.shape), 'strides': list(x.stride()),
                                    'N': 1152, 'M': 768, 'eps': eps,
                                    'epsilon_fp32_bits': struct.pack('<f', eps).hex(),
                                    'files': files, 'R_original_aten_repeat_exact': True,
                                    'R_profiler_output_preserved': True,
                                    'hidden_M2_variance': 'not observable in installed aten API'})
    (args.output / 'cases.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'cases': [c['label'] for c in report['cases']],
                      'original_aten_repeat_and_profiler_exact': True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--block', type=int, required=True)
    parser.add_argument('--runtime-image', required=True)
    capture(parser.parse_args())


if __name__ == '__main__':
    main()
