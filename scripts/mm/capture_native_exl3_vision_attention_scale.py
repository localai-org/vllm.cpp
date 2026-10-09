#!/usr/bin/env python3
"""Observe the pinned attention scalar on one existing real Q/K/V replay.

Reference-only, not a full worker/model capture. No weights are loaded and no
new golden is generated. Run separately from native GPU work.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import struct

from capture_native_exl3_vision_reference import EXPECTED, REFERENCE_IMAGE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(args, report):
    import numpy as np
    import torch
    import torchvision
    import transformers
    import vllm
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.attention import MMEncoderAttention
    import vllm.model_executor.models.qwen3_vl as model_source
    import vllm_xpu_kernels.flash_attn_interface as interface

    runtime = {'torch': torch.__version__, 'torch_git': torch.version.git_version,
               'torchvision': torchvision.__version__, 'transformers': transformers.__version__,
               'vllm': vllm.__version__}
    report['runtime'] = runtime
    if runtime != EXPECTED or args.runtime_image != REFERENCE_IMAGE or not torch.xpu.is_available():
        raise RuntimeError('pinned XPU reference required')
    report['model_source_sha256'] = digest(Path(inspect.getfile(model_source)))
    if report['model_source_sha256'] != '22f02a1ee1faba276a93243d23ecada1094484e39e5bde79452d8a44f32238ba':
        raise RuntimeError('installed model source differs')
    report['vision_attention_source_sha256'] = digest(Path(inspect.getfile(model_source.Qwen2_5_VisionAttention)))
    report['vision_constructor_source'] = inspect.getsource(model_source.Qwen2_5_VisionAttention.__init__)
    if 'scale=self.hidden_size_per_attention_head**-0.5' not in report['vision_constructor_source']:
        raise RuntimeError('actual vision scalar construction differs; inspect before proceeding')
    reference_path = args.reference / 'attention-real-replays.json'
    reference = json.loads(reference_path.read_text())
    if reference['reference_image'] != REFERENCE_IMAGE or len(reference['cases']) != 2:
        raise RuntimeError('existing two-image real-input capture required')
    case = reference['cases'][0]
    if case['shape'] != [768, 16, 72] or case['causal'] is not False:
        raise RuntimeError('existing small non-causal operand set required')
    report['reference_manifest_sha256'] = digest(reference_path)
    report['input_sha256'] = {}
    values = {}
    for name in ('q', 'k', 'v', 'expected'):
        entry = case['files'][name]
        if Path(entry['file']).name != entry['file']:
            raise RuntimeError('invalid captured path')
        path = args.reference / entry['file']
        if digest(path) != entry['sha256'] or path.stat().st_size != 768 * 16 * 72 * 2:
            raise RuntimeError('captured operand/hash differs: ' + name)
        report['input_sha256'][name] = entry['sha256']
        values[name] = np.frombuffer(path.read_bytes(), dtype=np.float16).copy().reshape(768, 16, 72)
    routes = []
    original = torch.ops._vllm_fa2_C.varlen_fwd
    if len(original.overloads()) != 1:
        raise RuntimeError('inspect ambiguous registered operator signature')
    schema = original.default._schema
    names = [argument.name for argument in schema.arguments]
    scalar_names = [name for name in names if name in ('scale', 'softmax_scale')]
    if len(scalar_names) != 1:
        raise RuntimeError('cannot bind actual scalar from registered schema')
    scalar_name = scalar_names[0]
    report['registered_operator_schema'] = str(schema)

    def observed(*positional, **keywords):
        bound = dict(zip(names, positional)) | keywords
        scalar = bound[scalar_name]
        bits = struct.unpack('<I', struct.pack('<f', scalar))[0]
        routes.append({'operator': '_vllm_fa2_C.varlen_fwd', 'scalar_argument': scalar_name,
                       'python_scalar': scalar, 'effective_f32_bits': hex(bits)})
        if bits != 0x3df15bef:
            raise RuntimeError('actual operator scale differs from pinned FP32 scalar')
        result = original(*positional, **keywords)
        routes[-1]['returned'] = True
        return result

    torch.ops._vllm_fa2_C.varlen_fwd = observed
    config = VllmConfig(compilation_config=CompilationConfig(mode=0, cudagraph_mode='NONE', custom_ops=['none']))
    try:
        with set_current_vllm_config(config):
            torch.set_default_dtype(torch.float16)
            with torch.device('xpu'):
                operation = MMEncoderAttention(num_heads=16, head_size=72, scale=72**-0.5)
            report['module_scale'] = operation.scale
            report['module_scale_f32_bits'] = hex(struct.unpack('<I', struct.pack('<f', operation.scale))[0])
            operands = [torch.from_numpy(values[name]).to('xpu') for name in ('q', 'k', 'v')]
            offsets = torch.tensor([0, 768], dtype=torch.int32, device='xpu')
            outputs = []
            for _ in range(2):
                result = operation(*(value.unsqueeze(0) for value in operands), cu_seqlens=offsets,
                                   max_seqlen=torch.tensor(768, device='cpu'))
                raw = result.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                outputs.append(hashlib.sha256(raw).hexdigest())
            if outputs != [case['files']['expected']['sha256']] * 2 or len(routes) != 2:
                raise RuntimeError('actual reference repeat/output differs; frozen capture unchanged')
            report.update(status='PASS', routes=routes, output_sha256=outputs,
                          selected_method=operation._forward_method.__qualname__)
    finally:
        torch.ops._vllm_fa2_C.varlen_fwd = original
        report['routes'] = routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--runtime-image', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve previous capture')
    report = {'status': 'FAIL', 'reference_image': args.runtime_image, 'scope': __doc__.strip()}
    code = 1
    try:
        capture(args, report)
        code = 0
    except Exception as error:
        report['error'] = repr(error)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'module_scale_f32_bits', 'routes', 'error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
