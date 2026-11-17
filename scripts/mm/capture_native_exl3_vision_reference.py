#!/usr/bin/env python3
"""Reproduce six frozen image answers in the pinned EXL3 Python environment.

Reference-only tooling: no imports from this script enter native inference.
Run the oracle and native GPU jobs sequentially. --check-environment validates
the runtime/fixture pins without constructing an LLM or exposing a GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path

from native_exl3_vision_facts import evaluate, qualification_inputs


REFERENCE_IMAGE = 'sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918'
EXPECTED = {'torch': '2.13.0+xpu', 'torch_git': 'cf30153c4c131c8164ee7798e5022d810682e2cb',
            'torchvision': '0.28.0+xpu', 'transformers': '5.16.1', 'vllm': '0.30.0'}


class ResizeCapture:
    """Observe the actual oracle tower input, only in the explicit resize case."""
    def install_resize_capture(self):
        import torch
        self._resize_inputs = []

        def before(module, args, kwargs):
            pixels = args[0]
            grid = kwargs.get('grid_thw', args[1] if len(args) > 1 else None)
            if isinstance(grid, torch.Tensor):
                grid = grid.detach().cpu().tolist()
            if (list(pixels.shape) != [2160, 1536] or pixels.dtype != torch.float16 or
                    pixels.device.type != 'xpu' or grid != [[1, 40, 54]]):
                raise RuntimeError('actual resized FP16/XPU tower input differs')
            self._resize_inputs.append({'grid_thw': grid[0], 'pixels_shape': list(pixels.shape),
                                        'pixels_dtype': str(pixels.dtype), 'device': str(pixels.device)})

        self._resize_hook = self.model_runner.model.visual.register_forward_pre_hook(before, with_kwargs=True)

    def finish_resize_capture(self):
        self._resize_hook.remove()
        if len(self._resize_inputs) != 1:
            raise RuntimeError('exactly one actual resize encoder submission required')
        return self._resize_inputs[0]


def capture(args, report):
    import torch
    import torchvision
    import transformers
    import vllm

    runtime = {'torch': torch.__version__, 'torch_git': torch.version.git_version,
               'torchvision': torchvision.__version__, 'transformers': transformers.__version__,
               'vllm': vllm.__version__}
    report['runtime'] = runtime
    if runtime != EXPECTED or args.runtime_image != REFERENCE_IMAGE:
        raise RuntimeError('runtime differs from the separately pinned custom EXL3 reference')
    tasks, manifest_name, _ = qualification_inputs(args.fixtures, args.resize_screenshot)
    report['manifest_sha256'] = hashlib.sha256((args.fixtures / manifest_name).read_bytes()).hexdigest()
    report['config_sha256'] = hashlib.sha256((args.model / 'config.json').read_bytes()).hexdigest()
    if args.check_environment:
        report.update(status='ENVIRONMENT_VERIFIED', task_count=len(tasks), learned_inference=False)
        return
    if not torch.xpu.is_available():
        raise RuntimeError('the pinned XPU reference requires a GPU; no CPU learned fallback')
    report['device'] = torch.xpu.get_device_name(0)
    from PIL import Image
    from vllm import LLM, SamplingParams

    options = {'model': str(args.model), 'dtype': 'float16', 'trust_remote_code': True,
        'max_model_len': 8192, 'max_num_seqs': 1, 'max_num_batched_tokens': 4096,
        'gpu_memory_utilization': .8, 'kv_cache_dtype': 'fp8', 'enforce_eager': True,
        'enable_prefix_caching': False, 'skip_mm_profiling': True,
        'limit_mm_per_prompt': {'image': 1, 'video': 0},
        'mm_processor_kwargs': {'size': {'longest_edge': 4194304, 'shortest_edge': 65536}},
        'compilation_config': {'mode': 0, 'cudagraph_mode': 'NONE', 'custom_ops': ['none']}}
    if args.resize_screenshot:
        options['worker_extension_cls'] = 'capture_native_exl3_vision_reference.ResizeCapture'
    # The model location is supplied by the caller, never a portable identity.
    report['options'] = options | {'model': '<caller-supplied existing checkpoint>'}
    report['template_variant'] = 'pinned_qwen_no_thinking'
    model = None
    try:
        model = LLM(**options)
        if args.resize_screenshot:
            model.collective_rpc('install_resize_capture', timeout=60)
        for task in tasks:
            prompt = ('<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>' + task['prompt'] +
                      '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
            with Image.open(args.fixtures / task['file']) as image:
                outputs = model.generate({'prompt': prompt, 'multi_modal_data': {'image': image.copy()}},
                    SamplingParams(temperature=0, max_tokens=128), use_tqdm=False)
            if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                raise RuntimeError('wrong reference output count')
            output, generated = outputs[0], outputs[0].outputs[0]
            score = evaluate(generated.text, task)
            case = {'name': task['name'], 'image_sha256': task['sha256'],
                'prompt_ids': list(output.prompt_token_ids), 'output_ids': list(generated.token_ids),
                'reference_text': generated.text, 'score': score, 'completion_tokens': len(generated.token_ids),
                'finish_reason': generated.finish_reason, 'stop_reason': generated.stop_reason}
            report['cases'].append(case)
            print('HELDOUT_REFERENCE_ANSWER ' + json.dumps({k: case[k] for k in
                  ('name', 'reference_text', 'score', 'finish_reason')}), flush=True)
        if args.resize_screenshot:
            report['actual_tower_input'] = model.collective_rpc('finish_resize_capture', timeout=60)[0]
            report['processed_size_wh'] = tasks[0]['processed_size_wh']
        if not all(case['score']['pass'] and case['finish_reason'] == 'stop' for case in report['cases']):
            raise RuntimeError('frozen named facts or natural EOS failed; do not retune the expectations')
        report['status'] = 'PASS'
    finally:
        if model is not None:
            model.llm_engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--runtime-image', required=True, help='Caller-declared immutable image identity; command must also record the actual image')
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[2] / 'tests/fixtures/native_vision_qualification')
    parser.add_argument('--check-environment', action='store_true')
    parser.add_argument('--resize-screenshot', action='store_true', help='one frozen derived resize screenshot; original six-task default unchanged')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve previous evidence')
    report = {'status': 'FAIL', 'reference_image': args.runtime_image,
              'scope': 'six bounded pinned Python target-only image answers; named facts, not tensor or speed parity',
              'cases': []}
    if args.resize_screenshot:
        report['scope'] = 'one derived resize-required screenshot; actual oracle tower geometry and named facts, not held-out or numerical parity'
    code = 1
    try:
        capture(args, report)
        code = 0
    except Exception as error:
        report.update(status='FAIL', error=repr(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'runtime', 'error')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
