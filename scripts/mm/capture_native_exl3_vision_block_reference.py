#!/usr/bin/env python3
"""Observe one pinned vision block on its existing exact worker input.

Diagnostic only: invokes the installed Qwen3_VisionBlock, not a reconstructed
tower or language model. Writes separately named operands; preserves goldens.
Run the oracle and native GPU jobs sequentially.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

from capture_native_exl3_vision_reference import EXPECTED, REFERENCE_IMAGE
from replay_native_exl3_vision_blocks import CAPTURE_SHA256, CONFIG_SHA256, require


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(args, report):
    import numpy as np
    import torch
    import torchvision
    import transformers
    import vllm
    from safetensors import safe_open
    from vllm import envs
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel,
                                 init_distributed_environment, initialize_model_parallel)
    import vllm.model_executor.models.qwen3_vl as source

    runtime = {'torch': torch.__version__, 'torch_git': torch.version.git_version,
               'torchvision': torchvision.__version__, 'transformers': transformers.__version__,
               'vllm': vllm.__version__}
    require(runtime == EXPECTED and args.runtime_image == REFERENCE_IMAGE, 'pinned runtime required')
    require(torch.xpu.is_available(), 'XPU required; no CPU learned fallback')
    require(digest(Path(inspect.getfile(source))) ==
            '22f02a1ee1faba276a93243d23ecada1094484e39e5bde79452d8a44f32238ba', 'model source differs')
    require(digest(args.model / 'config.json') == CONFIG_SHA256, 'checkpoint config differs')
    manifest_path = args.reference / 'vision-boundaries-capture.json'
    require(digest(manifest_path) == CAPTURE_SHA256, 'existing capture changed')
    manifest = json.loads(manifest_path.read_text())
    require(manifest['image'] == REFERENCE_IMAGE and manifest['worker']['encoder_calls'] == 2,
            'wrong actual worker capture')
    first = {}
    for entry in manifest['worker']['captures']:
        if entry['name'] == 'tower-input-0' and first:
            break
        require(entry['name'] not in first, 'duplicate captured boundary')
        first[entry['name']] = entry
    prefix = f'block{args.block}'
    report.update(runtime=runtime, capture_sha256=CAPTURE_SHA256, config_sha256=CONFIG_SHA256,
                  block=args.block, weight_n_contiguous=envs.VLLM_XPU_FORCE_N_CONTIG_WEIGHT)
    report['source_sha256'] = {name: digest(Path(inspect.getfile(cls))) for name, cls in
                             [('block', source.Qwen3_VisionBlock), ('attention', source.Qwen2_5_VisionAttention)]}

    def load(name):
        entry = first[name]
        require(Path(entry['file']).name == entry['file'], 'invalid boundary path')
        path = args.reference / entry['file']
        require(digest(path) == entry['sha256'], 'captured operand hash differs: ' + name)
        dtype = {'torch.float16': np.float16, 'torch.int32': np.int32}[entry['dtype']]
        values = np.frombuffer(path.read_bytes(), dtype=dtype).copy().reshape(entry['shape'])
        return torch.from_numpy(values).to(entry['device'])

    require(first[prefix + '-input-0']['shape'] == [768, 1, 1152], 'small existing block required')
    require(first[prefix + '-input-0']['sha256'] == first[f'block{args.block-1}-output']['sha256'],
            'original block input chain differs')
    x = load(prefix + '-input-0')
    kwargs = {name: load(prefix + '-kwargs-' + name) for name in
              ('cu_seqlens', 'rotary_pos_emb_cos', 'rotary_pos_emb_sin', 'max_seqlen')}
    kwargs['sequence_lengths'] = None  # Actual FLASH_ATTN metadata, source.prepare_encoder_metadata.
    config = VllmConfig(compilation_config=CompilationConfig(mode=0, cudagraph_mode='NONE', custom_ops=['none']))
    parameters = {}
    with set_current_vllm_config(config):
        torch.set_default_dtype(torch.float16)
        init_distributed_environment(1, 0, 'file:///tmp/b70-vision-block-reference', 0, backend='gloo')
        initialize_model_parallel(1, 1, backend='gloo')
        saved = []
        try:
            with torch.device('xpu'):
                block = source.Qwen3_VisionBlock(1152, 16, 4304,
                    act_fn=source._ACTIVATION_REGISTRY['gelu_pytorch_tanh'], prefix=f'model.visual.blocks.{args.block}')
            index = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']
            with torch.no_grad():
                for name, parameter in block.named_parameters():
                    key = f'model.visual.blocks.{args.block}.{name}'
                    filename = index[key]
                    require(Path(filename).name == filename, 'invalid model shard path')
                    with safe_open(args.model / filename, framework='pt', device='cpu') as shard:
                        value = shard.get_tensor(key)
                    require(value.dtype == torch.bfloat16 and value.shape == parameter.shape, 'checkpoint parameter differs')
                    value = value.to(torch.float16)
                    parameters[name] = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
                    parameter.copy_(value)
                for module in block.modules():
                    if getattr(module, 'quant_method', None) is not None:
                        module.quant_method.process_weights_after_loading(module)
            report['parameter_f16_sha256'] = parameters
            report['linear_weight_strides'] = {name: list(module.weight.stride()) for name, module in
                [('qkv', block.attn.qkv), ('projection', block.attn.proj),
                 ('fc1', block.mlp.linear_fc1), ('fc2', block.mlp.linear_fc2)]}
            require(block.attn.attn._forward_method.__qualname__ == 'MMEncoderAttention.forward_xpu', 'wrong attention route')
            report['attention_scale'] = block.attn.attn.scale
            report['stages'] = {}
            used = 0

            def save(name, tensor):
                nonlocal used
                if isinstance(tensor, tuple):
                    tensor = tensor[0]
                require(tensor.dtype == torch.float16 and tensor.device.type == 'xpu', 'wrong observed stage dtype/device')
                raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                used += len(raw)
                require(used <= 64 * 1024 * 1024, 'capture byte budget exceeded: ' + name)
                require(name not in report['stages'], 'duplicate observed stage: ' + name)
                filename = name + '.float16'
                (args.output / filename).write_bytes(raw)
                report['stages'][name] = {'file': filename, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                                        'sha256': hashlib.sha256(raw).hexdigest()}

            modules = {id(module): name for name, module in [('norm1', block.norm1), ('norm2', block.norm2),
                ('qkv', block.attn.qkv), ('rotary', block.attn.apply_rotary_emb), ('attention', block.attn.attn),
                ('projection', block.attn.proj), ('fc1', block.mlp.linear_fc1),
                ('gelu', block.mlp.act_fn), ('fc2', block.mlp.linear_fc2)]}
            observed_modules = [m for m in block.modules() if id(m) in modules]
            require(len(observed_modules) == len(modules), 'inspect callable stage not registered as module')
            # Snapshot every inherited callable BEFORE installing any wrapper.
            # QKVParallelLinear inherits ColumnParallelLinear: looking it up
            # after wrapping the parent can wrap the observer itself twice.
            originals = [(cls, cls.__dict__.get('__call__'), cls.__call__)
                         for cls in {type(m) for m in observed_modules}]
            for cls, own, old in originals:

                def observed(module, *positional, _old=old, **keywords):
                    name = modules.get(id(module))
                    if name == 'norm2': save('residual1', positional[0])
                    if name == 'rotary':
                        save('q', positional[0][0]); save('k', positional[0][1])
                    if name == 'attention': save('v', keywords['value'][0])
                    result = _old(module, *positional, **keywords)
                    if name == 'rotary':
                        save('rotated_q', result[0]); save('rotated_k', result[1])
                    elif name is not None: save(name, result)
                    return result

                saved.append((cls, own)); cls.__call__ = observed
            with torch.inference_mode():
                save('input', x)
                output = block(x, **kwargs)
                save('output', output)
            report['reference_input_sha256'] = first[prefix + '-input-0']['sha256']
            report['original_output_sha256'] = first[prefix + '-output']['sha256']
            report['matches_original_block_output'] = report['stages']['output']['sha256'] == report['original_output_sha256']
            report['status'] = 'CAPTURED'
        finally:
            for cls, own in reversed(saved):
                if own is None: delattr(cls, '__call__')
                else: cls.__call__ = own
            destroy_model_parallel(); destroy_distributed_environment()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'reference', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--runtime-image', required=True)
    parser.add_argument('--block', required=True, type=int, choices=(12, 26))
    args = parser.parse_args()
    if args.output.exists(): parser.error('preserve previous evidence')
    args.output.mkdir()
    report = {'status': 'FAIL', 'scope': __doc__.strip(), 'reference_image': args.runtime_image}
    try:
        capture(args, report)
    except Exception as error:
        report['status'] = 'FAIL'; report['error'] = repr(error)
    (args.output / 'capture.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'block', 'matches_original_block_output', 'error')}))
    return 0 if report['status'] == 'CAPTURED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
