#!/usr/bin/env python3
"""Capture four bounded actual-worker tower geometries in the pinned oracle.

Reference-only diagnostic readbacks. No captured tensors enter native serving
and no capture timing is performance evidence. Existing checkpoints only.
"""
import argparse
import hashlib
import json
from pathlib import Path

from capture_native_exl3_vision_reference import EXPECTED, REFERENCE_IMAGE
from native_exl3_vision_facts import load_tasks


class TowerCapture:
    def install_tower_capture(self, directory):
        import inspect
        import torch
        visual = self.model_runner.model.visual
        self._tower_directory = Path(directory)
        self._tower_cases = []
        self._tower_handles = []
        self._tower_classes = []
        self._tower_bytes = 0
        parameters = list(visual.named_parameters())
        if len(parameters) != 333 or any(p.dtype != torch.float16 or p.device.type != 'xpu'
                                         for _, p in parameters):
            raise RuntimeError('actual 333-parameter FP16/XPU tower required')

        def save(name, value):
            if not isinstance(value, torch.Tensor):
                raise RuntimeError('missing tensor boundary: ' + name)
            if value.numel() > 16384 * 1536 or value.dtype not in (torch.float16, torch.int64):
                raise RuntimeError('unbounded/unqualified capture: ' + name)
            case = self._tower_cases[-1]
            if name in case:
                raise RuntimeError('duplicate boundary: ' + name)
            self._tower_bytes += value.numel() * value.element_size()
            if self._tower_bytes > 256 * 1024 * 1024:
                raise RuntimeError('per-image capture exceeds 256 MiB')
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            filename = 'image-' + str(len(self._tower_cases) - 1) + '-' + name + '.bin'
            with (self._tower_directory / filename).open('xb') as output:
                output.write(raw)
            case[name] = {'file': filename, 'sha256': hashlib.sha256(raw).hexdigest(),
                          'dtype': str(value.dtype), 'shape': list(value.shape), 'device': str(value.device)}

        def before(module, args, kwargs):
            self._tower_bytes = 0
            self._tower_cases.append({})
            save('pixels', args[0])
            grid = kwargs.get('grid_thw', args[1] if len(args) > 1 else None)
            if not isinstance(grid, torch.Tensor):
                grid = torch.tensor(grid, dtype=torch.int64)
            save('grid', grid)
            metadata = kwargs.get('encoder_metadata')
            if metadata:
                for name, key in [('position', 'pos_embeds'), ('cos', 'rotary_pos_emb_cos'),
                                  ('sin', 'rotary_pos_emb_sin')]:
                    save(name, metadata[key])

        self._tower_handles.append(visual.register_forward_pre_hook(before, with_kwargs=True))
        modules = {'patch': visual.patch_embed, 'block0': visual.blocks[0], 'merger': visual.merger}
        selected = {id(module): name for name, module in modules.items()}
        # The pinned modules can call forward through explicit __call__ methods;
        # use the same class-call observation seam as the earlier small capture.
        for cls in {type(module) for module in modules.values()}:
            own, original = cls.__dict__.get('__call__'), cls.__call__

            def call(module, *args, _original=original, **kwargs):
                result = _original(module, *args, **kwargs)
                name = selected.get(id(module))
                if name is not None:
                    save(name, result)
                return result

            self._tower_classes.append((cls, own))
            cls.__call__ = call
        self._tower_visual = visual
        self._tower_metadata = visual.prepare_encoder_metadata

        def metadata(*args, **kwargs):
            result = self._tower_metadata(*args, **kwargs)
            for name, key in [('position', 'pos_embeds'), ('cos', 'rotary_pos_emb_cos'),
                              ('sin', 'rotary_pos_emb_sin')]:
                save(name, result[key])
            return result

        visual.prepare_encoder_metadata = metadata
        return {'parameters': len(parameters), 'dtype': 'torch.float16', 'device': 'xpu',
                'source_sha256': hashlib.sha256(Path(inspect.getfile(type(visual))).read_bytes()).hexdigest()}

    def finish_tower_capture(self):
        self._tower_visual.prepare_encoder_metadata = self._tower_metadata
        for handle in self._tower_handles:
            handle.remove()
        for cls, own in self._tower_classes:
            if own is None:
                delattr(cls, '__call__')
            else:
                cls.__call__ = own
        expected = {'pixels', 'grid', 'patch', 'position', 'cos', 'sin', 'block0', 'merger'}
        if len(self._tower_cases) != 4 or any(set(case) != expected for case in self._tower_cases):
            raise RuntimeError('missing actual four-image tower boundary')
        return self._tower_cases


def capture(args, report):
    import torch
    import torchvision
    import transformers
    import vllm
    from PIL import Image
    from vllm import LLM, SamplingParams
    runtime = {'torch': torch.__version__, 'torch_git': torch.version.git_version,
               'torchvision': torchvision.__version__, 'transformers': transformers.__version__,
               'vllm': vllm.__version__}
    report['runtime'] = runtime
    if runtime != EXPECTED or args.runtime_image != REFERENCE_IMAGE or not torch.xpu.is_available():
        raise RuntimeError('pinned XPU oracle required')
    root = args.fixtures
    fixtures = {r['name']: r for r in json.loads((root / 'native_vision_http/fixtures.json').read_text())}
    portrait = next(r for r in load_tasks(root / 'native_vision_qualification') if r['name'] == 'field-day-portrait')
    maximum = json.loads((root / 'native_vision_http/max-eviction-mrope-reference.json').read_text())['cases'][0]
    cases = [dict(name=name, image='native_vision_http/' + fixtures[name]['file'],
                  image_sha256=fixtures[name]['sha256']) for name in ('orbit', 'comet-unaligned')]
    cases += [dict(name='portrait', image='native_vision_qualification/' + portrait['file'], image_sha256=portrait['sha256']),
              dict(name='maximum', image='native_vision_http/' + maximum['file'], image_sha256=maximum['image_sha256'])]
    for case in cases:
        if hashlib.sha256((root / case['image']).read_bytes()).hexdigest() != case['image_sha256']:
            raise RuntimeError('frozen fixture changed')
    report['config_sha256'] = hashlib.sha256((args.model / 'config.json').read_bytes()).hexdigest()
    options = {'model': str(args.model), 'dtype': 'float16', 'trust_remote_code': True,
        'max_model_len': 8192, 'max_num_seqs': 1, 'max_num_batched_tokens': 4096,
        'gpu_memory_utilization': .8, 'kv_cache_dtype': 'fp8', 'enforce_eager': True,
        'enable_prefix_caching': False, 'skip_mm_profiling': True,
        'limit_mm_per_prompt': {'image': 1, 'video': 0},
        'mm_processor_kwargs': {'size': {'longest_edge': 4194304, 'shortest_edge': 65536}},
        'compilation_config': {'mode': 0, 'cudagraph_mode': 'NONE', 'custom_ops': ['none']},
        'worker_extension_cls': 'capture_native_exl3_vision_tower_reference.TowerCapture'}
    model = None
    try:
        model = LLM(**options)
        report['tower'] = model.collective_rpc('install_tower_capture', args=(str(args.output),), timeout=60)[0]
        for case in cases:
            prompt = ('<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.'
                      '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
            with Image.open(root / case['image']) as image:
                result = model.generate({'prompt': prompt, 'multi_modal_data': {'image': image.copy()}},
                    SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
            case['output_ids'] = list(result[0].outputs[0].token_ids)
            report['cases'].append(case)
            print('TOWER_REFERENCE_IMAGE ' + json.dumps(case), flush=True)
        boundaries = model.collective_rpc('finish_tower_capture', timeout=60)[0]
        for case, boundary in zip(report['cases'], boundaries):
            case['boundaries'] = boundary
        report['status'] = 'CAPTURED'
    finally:
        if model is not None:
            model.llm_engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--runtime-image', required=True, help='caller-declared; independently pin container command')
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[2] / 'tests/fixtures')
    parser.add_argument('--output', type=Path, required=True, help='new capture directory; raw tensors stay local')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve captures')
    args.output.mkdir(parents=True)
    report = {'status': 'FAIL', 'cases': [], 'scope': 'four actual eager oracle tower shapes; capture is not native parity or performance',
              'reference_image': args.runtime_image}
    code = 1
    try:
        capture(args, report)
        code = 0
    except Exception as error:
        report['error'] = repr(error)
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'cases': len(report['cases']), 'error': report.get('error')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
