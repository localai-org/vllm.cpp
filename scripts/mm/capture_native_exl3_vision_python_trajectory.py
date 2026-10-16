#!/usr/bin/env python3
"""Bounded within-Python C1/C4 teacher windows; not native/reference equality.

Keep actual reference sibling histories. Extra same-token forwards restore all
active owners before each control, then restore the actual post-step owners.
Save complete unmasked heads before imposing the predeclared teacher token.
The diagnostic sampling mask is never an arithmetic/greedy-parity repair.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

from capture_native_exl3_vision_reference import EXPECTED, REFERENCE_IMAGE
from capture_native_exl3_vision_python_alpha import recipe
from replay_native_exl3_vision_first_gdn import CONFIG_SHA


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(a, b):
    import numpy as np
    x, y = a.astype(np.float64), b.astype(np.float64)
    require(x.shape == y.shape == (248320,) and np.isfinite(x).all() and np.isfinite(y).all(),
            'complete finite unmasked heads required')
    top = [np.argsort(-z, kind='stable')[:2] for z in (x, y)]
    d = y - x
    return {'elements': len(x), 'different_elements': int(np.count_nonzero(d)),
            'max_abs': float(np.max(np.abs(d))), 'rel_l2': float(np.linalg.norm(d) / max(np.linalg.norm(x), 1e-30)),
            'reference_top2_ids': top[0].tolist(), 'candidate_top2_ids': top[1].tolist(),
            'reference_margin': float(x[top[0][0]] - x[top[0][1]]),
            'candidate_margin': float(y[top[1][0]] - y[top[1][1]])}


class TrajectorySelfControl:
    def install_trajectory_control(self, output, specification):
        from vllm.forward_context import get_forward_context
        import torch
        self._tr_root = Path(output)
        self._tr_spec = specification
        self._tr_phase = 'c1'
        self._tr_step = 0
        self._tr_prefill_row = None
        self._tr_pending = None
        self._tr_c1 = []
        self._tr_records = {'c1': [], 'c4': []}
        self._tr_shared = None
        model = self.model_runner.model
        self._tr_forward = model.forward
        self._tr_logits = model.compute_logits

        def forward(*args, **kwargs):
            ctx = get_forward_context()
            embeds = kwargs.get('inputs_embeds')
            ids = kwargs.get('input_ids')
            if ids is None and isinstance(embeds, torch.Tensor):
                require(type(self.model_runner).__qualname__ == 'XPUModelRunnerV2', 'unexamined runner')
                ids = self.model_runner.input_buffers.input_ids[:embeds.shape[0]]
            require(isinstance(ids, torch.Tensor) and ids.ndim == 1, 'actual token buffer required')
            full = next(layer.self_attn.attn for layer in model.language_model.model.layers if hasattr(layer, 'self_attn'))
            am = ctx.attn_metadata[full.layer_name]
            qsl = am.query_start_loc.detach().cpu().tolist()
            seq = am.seq_lens.detach().cpu().tolist()
            n = len(seq)
            lengths = [b - a for a, b in zip(qsl, qsl[1:])]
            candidates = [i for i in range(n) if lengths[i] == specification['context'] and seq[i] == specification['context']]
            if candidates:
                require(len(candidates) == 1 and n == (1 if self._tr_phase == 'c1' else 4)
                        and self._tr_step == 0, 'wrong selected actual prefill')
                row = candidates[0]
                actual = ids[qsl[row]:qsl[row + 1]].detach().cpu().tolist()
                require(actual == specification['expanded_prompt_ids'], 'reference prompt expansion differs')
                self._tr_prefill_row = row
                self._tr_pending = {'kind': 'prefill', 'row': row, 'rows': n, 'seq_lens': seq,
                                    'query_start_loc': qsl, 'input_token_ids': actual}
                return self._tr_forward(*args, **kwargs)
            if lengths != [1] * n or n not in (1, 4):
                return self._tr_forward(*args, **kwargs)
            step = self._tr_step
            if step >= len(specification['teacher_tokens']):
                return self._tr_forward(*args, **kwargs)
            context = specification['context'] + step
            positions = kwargs['positions'].detach().cpu().tolist()
            require(len(positions) == 3 and all(len(axis) == n for axis in positions), 'actual M-RoPE axes required')
            candidates = [i for i in range(n) if seq[i] == context + 1 and
                          [axis[i] for axis in positions] == [context + specification['mrope_delta']] * 3]
            if not candidates:
                return self._tr_forward(*args, **kwargs)
            require(len(candidates) == 1 and n == (1 if self._tr_phase == 'c1' else 4), 'selected ordinary geometry changed')
            row = candidates[0]
            token_ids = ids.detach().cpu().tolist()
            require(token_ids[row] == specification['teacher_tokens'][step], 'teacher input changed')
            require(isinstance(embeds, torch.Tensor) and list(embeds.shape) == [n, 5120]
                    and embeds.dtype == torch.float16, 'selected actual model embeddings required')
            info = {'kind': 'decode', 'step': step, 'row': row, 'rows': n, 'context': context,
                    'positions': positions, 'input_token_ids': token_ids, 'seq_lens': seq,
                    'query_start_loc': qsl, 'embedding_sha256': self._tr_hash(embeds[row])}
            before = [self._tr_views(ctx.attn_metadata, i, n, seq[i] - 1) for i in range(n)]
            own_pre = self._tr_fingerprints(before[row])
            if self._tr_phase == 'c1':
                self._tr_c1.append({'incoming': self._tr_clone(before[row]), 'info': dict(info)})
                result = self._tr_forward(*args, **kwargs)
                info.update(incoming=own_pre, post=self._tr_fingerprints(self._tr_views(ctx.attn_metadata, row, n, context + 1)))
            else:
                c1 = self._tr_c1[step]
                require(info['embedding_sha256'] == c1['info']['embedding_sha256'], 'same-token embedding differs')
                all_before = [self._tr_clone(v) for v in before]
                result = self._tr_forward(*args, **kwargs)
                after = [self._tr_views(ctx.attn_metadata, i, n, seq[i]) for i in range(n)]
                all_after = [self._tr_clone(v) for v in after]
                own_head = self._tr_logits(result)[row].detach().cpu().numpy().copy()
                info.update(incoming=own_pre, post=self._tr_fingerprints(after[row]))
                controls = {}
                for label, seed in [('identical_incoming_single_step', c1['incoming']),
                                    ('evolving_shared_seed', self._tr_shared or self._tr_c1[0]['incoming'])]:
                    # Restore every actual sibling before each extra forward.
                    for views, saved in zip(before, all_before):
                        self._tr_restore(views, saved)
                    self._tr_restore(before[row], seed)
                    restored = self._tr_fingerprints(self._tr_views(ctx.attn_metadata, row, n, context))
                    expected = self._tr_fingerprints({k: (v, None) for k, v in seed.items()})
                    require(restored == expected, 'selected incoming control not restored exactly')
                    extra = self._tr_forward(*args, **kwargs)
                    post = self._tr_views(ctx.attn_metadata, row, n, context + 1)
                    control_head = self._tr_logits(extra)[row].detach().cpu().numpy().copy()
                    controls[label] = {'incoming': restored, 'post': self._tr_fingerprints(post),
                        'head': self._tr_write_head(f'c4-{step}-{label}', control_head),
                        'logits': metrics(c1['head_array'], control_head)}
                    if label == 'evolving_shared_seed':
                        self._tr_shared = self._tr_clone(post)
                # Normal generation continues from exactly its own first pass.
                for i, saved in enumerate(all_after):
                    self._tr_restore(self._tr_views(ctx.attn_metadata, i, n, seq[i]), saved)
                require(self._tr_fingerprints(self._tr_views(ctx.attn_metadata, row, n, context + 1)) == info['post'],
                        'actual selected post-state changed by controls')
                info['controls'] = controls
                info['own_head_array'] = own_head
                info['sibling_count_restored_per_control'] = n
            self._tr_pending = info
            return result

        def logits(hidden):
            result = self._tr_logits(hidden)
            info = self._tr_pending
            if info is None:
                return result
            row, n = info['row'], info['rows']
            require(result.shape == (n, 248320) and result.dtype == torch.float16, 'full installed reference head required')
            head = result[row].detach().cpu().numpy().copy()
            if info['kind'] == 'prefill':
                self._tr_records[self._tr_phase + '_prefill'] = dict(info, head=self._tr_write_head(self._tr_phase + '-prefill', head))
                next_token = specification['teacher_tokens'][0]
            else:
                step = info['step']
                info['head'] = self._tr_write_head(f'{self._tr_phase}-{step}-own', head)
                if self._tr_phase == 'c1':
                    self._tr_c1[step]['head_array'] = head
                else:
                    require((head == info.pop('own_head_array')).all(), 'actual head changed by control forwards')
                    info['evolving_separate_serving_seeds'] = metrics(self._tr_c1[step]['head_array'], head)
                self._tr_records[self._tr_phase].append(info)
                self._tr_step += 1
                next_token = specification['teacher_tokens'][self._tr_step] if self._tr_step < len(specification['teacher_tokens']) else 13
            self._tr_pending = None
            # Preserve completed observations if a later guarded step fails.
            (self._tr_root / 'partial-frames.json').write_text(json.dumps(self._tr_records, indent=2) + '\n')
            # Diagnostic teacher forcing only AFTER recording the unmasked
            # original. This tensor does not enter any arithmetic comparison.
            sampled = result.clone()
            sampled[row].fill_(-float('inf'))
            sampled[row, next_token] = 0
            return sampled

        model.forward = forward
        model.compute_logits = logits
        return {'model_class': type(model).__qualname__, 'runner_class': type(self.model_runner).__qualname__}

    def _tr_views(self, metadata, row, n, context):
        import torch
        result = {}
        gdn = attn = 0
        for layer in self.model_runner.model.language_model.model.layers:
            if hasattr(layer, 'linear_attn'):
                module = layer.linear_attn
                md = metadata[module.prefix]
                require(md.num_prefills == 0 and md.num_decodes == n and md.num_actual_tokens == n
                        and md.spec_sequence_masks is None, 'ordinary GDN metadata required')
                slots = md.non_spec_state_indices_tensor.detach().cpu().tolist()[:n]
                require(len(slots) == n and len(set(slots)) == n, 'aliased GDN owners')
                for label, owner in zip(('conv', 'ssm'), module.kv_cache):
                    value = owner[slots[row]]
                    require(value.dtype == (torch.float16 if label == 'conv' else torch.float32)
                            and value.numel() == (30720 if label == 'conv' else 786432), 'GDN owner contract differs')
                    result[f'gdn{gdn}-{label}'] = (value, lambda x, target=value: target.copy_(x))
                gdn += 1
            else:
                module = layer.self_attn.attn
                md = metadata[module.layer_name]
                cache = module.kv_cache
                seq = md.seq_lens.detach().cpu().tolist()[:n]
                require(cache.ndim == 4 and cache.shape[1] == 4 and cache.shape[-1] == 512
                        and cache.element_size() == 1 and seq[row] in (context, context + 1), 'initialized FP8 KV contract differs')
                require(md.query_start_loc.detach().cpu().tolist() == list(range(n + 1)), 'ordinary attention rows required')
                require([float(getattr(module, '_' + k + '_scale').item()) for k in ('k', 'v')] == [1.0, 1.0], 'KV scales differ')
                pages = md.block_table[:, 0].detach().cpu().tolist()[:n]
                require(len(set(pages)) == n and cache.shape[2] >= max(seq), 'distinct single-page owners required')
                indices = torch.arange(context, device=cache.device)
                physical = md.block_table[row].index_select(0, indices // cache.shape[2]).long()
                offsets = indices % cache.shape[2]
                raw = cache.view(torch.uint8)
                for which, label in enumerate(('k', 'v')):
                    lo, hi = which * 256, (which + 1) * 256
                    def store(saved, target=raw, p=physical, o=offsets, left=lo, right=hi):
                        target[p, :, o, left:right] = saved
                    result[f'attn{attn}-{label}'] = (raw[physical, :, offsets, lo:hi], store)
                attn += 1
        require(gdn == 48 and attn == 16 and len(result) == 128, 'incomplete actual state')
        return result

    @staticmethod
    def _tr_hash(tensor):
        import torch
        return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    @staticmethod
    def _tr_clone(views):
        return {k: v.clone() for k, (v, _) in views.items()}

    @staticmethod
    def _tr_restore(views, saved):
        require(views.keys() == saved.keys(), 'state set differs')
        for k, (v, store) in views.items():
            require(v.shape == saved[k].shape and v.dtype == saved[k].dtype, 'state shape/dtype differs')
            store(saved[k])

    @staticmethod
    def _tr_fingerprints(views):
        import torch
        total = sum(v.numel() * v.element_size() for v, _ in views.values())
        require(0 < total <= 256 * 1024**2, 'selected observation bound exceeded')
        host = {}
        for k, (v, _) in views.items():
            cpu = torch.empty(v.shape, dtype=v.dtype, device='cpu', pin_memory=True)
            cpu.copy_(v, non_blocking=True)
            host[k] = cpu
        torch.xpu.synchronize()  # One bounded observation drain, not a serving path.
        return {k: hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k, v in host.items()}

    def _tr_write_head(self, name, array):
        import numpy as np
        require(array.shape == (248320,) and array.dtype == np.float16 and np.isfinite(array).all(), 'bad complete head')
        path = self._tr_root / (name + '.bin')
        with path.open('xb') as f:
            f.write(array.tobytes())
        return {'file': path.name, 'shape': [248320], 'dtype': 'float16', 'sha256': digest(path)}

    def trajectory_c4_phase(self):
        require(self._tr_phase == 'c1' and self._tr_step == 8 and len(self._tr_records['c1']) == 8, 'incomplete C1 window')
        self._tr_phase = 'c4'
        self._tr_step = 0
        self._tr_prefill_row = None

    def finish_trajectory_control(self):
        require(self._tr_step == 8 and len(self._tr_records['c4']) == 8, 'incomplete C4 window')
        model = self.model_runner.model
        model.forward, model.compute_logits = self._tr_forward, self._tr_logits
        self._tr_c1 = []
        self._tr_shared = None
        return self._tr_records


def run(args, report):
    import torch, torchvision, transformers, vllm
    from PIL import Image
    from vllm import LLM, SamplingParams
    runtime = {'torch': torch.__version__, 'torch_git': torch.version.git_version,
               'torchvision': torchvision.__version__, 'transformers': transformers.__version__, 'vllm': vllm.__version__}
    require(runtime == EXPECTED and args.runtime_image == REFERENCE_IMAGE and torch.xpu.is_available(), 'unpinned reference runtime')
    require(digest(args.model / 'config.json') == CONFIG_SHA and os.environ.get('EXL3_INT8_PREFILL') == '1', 'model/prefill policy differs')
    require(torch.xpu.device_count() == 1, 'exactly one reference XPU required')
    properties = torch.xpu.get_device_properties(0)
    require(properties.device_id == 0xE223 and properties.name == 'Intel(R) Arc(TM) Pro B70 Graphics', 'wrong reference device')
    import vllm._xpu_ops, vllm_xpu_kernels
    root = Path(vllm._xpu_ops.__file__).parent
    sources = [root / '_xpu_ops.py', root / 'model_executor/models/qwen3_5.py',
               root / 'v1/worker/gpu/model_runner.py']
    report['installed_sources'] = [{'file': str(path), 'sha256': digest(path)} for path in sources if path.is_file()]
    report['installed_kernel_binaries'] = [{'file': str(path), 'sha256': digest(path)}
        for path in Path(vllm_xpu_kernels.__file__).parent.rglob('*.so')]
    require(report['installed_kernel_binaries'], 'installed reference kernel identity missing')
    report['reference_device'] = str(properties)
    plan_path = args.receipts / ('n3-alpha-trajectory-v1-plan.json' if args.case == 'alpha' else 'n3-image-trajectory-v1-plan.json')
    plan = json.loads(plan_path.read_text())
    require(len(plan['teacher_tokens']) == 8, 'exact predeclared eight-token window required')
    if args.case == 'alpha':
        inputs = recipe(args.receipts, args.fixtures)
        prompt_ids = inputs['alpha']['ids']
        target = {'prompt_token_ids': prompt_ids}
        siblings = []
        for name in ('beta', 'comet', 'orbit'):
            item = inputs[name]
            p = {'prompt_token_ids': item['ids']}
            if 'image' in item:
                with Image.open(item['image']) as image:
                    p['multi_modal_data'] = {'image': image.copy()}
            siblings.append(p)
        delta = 0
    else:
        frozen = json.loads((args.receipts / 'n3-image-tokenize-v2-result.json').read_text())
        image_path = args.fixtures / 'native_vision_http/orbit.png'
        require(digest(image_path) == '9ed13de9eb9d4c68576714a041d5690ab696ba269597002b71b154c0956aebe5', 'image changed')
        with Image.open(image_path) as image:
            target = {'prompt_token_ids': frozen['pre_expansion_ids'], 'multi_modal_data': {'image': image.copy()}}
        prompt_ids = frozen['prompt_ids']
        # Same mixed image/text subject. The reference tokenizer encodes these
        # ordinary sibling texts; its histories are not native state tensors.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        siblings = [{'prompt_token_ids': tokenizer.encode('Request ' + name + '.' + ' x' * 128 +
                    '. Continue with a detailed explanation of elementary arithmetic.', add_special_tokens=False)}
                    for name in ('beta', 'gamma', 'delta')]
        delta = -176
    spec = {'teacher_tokens': plan['teacher_tokens'], 'context': plan['initial_context'],
            'expanded_prompt_ids': prompt_ids, 'mrope_delta': delta, 'original_plan_sha256': digest(plan_path)}
    require(len(prompt_ids) == spec['context'], 'prompt context differs')
    options = {'model': str(args.model), 'dtype': 'float16', 'trust_remote_code': True, 'max_model_len': 8192,
        'max_num_seqs': 4, 'max_num_batched_tokens': 4096, 'gpu_memory_utilization': .8, 'kv_cache_dtype': 'fp8',
        'block_size': 1600, 'enforce_eager': True, 'enable_prefix_caching': False, 'skip_mm_profiling': True,
        'limit_mm_per_prompt': {'image': 1, 'video': 0},
        'mm_processor_kwargs': {'size': {'longest_edge': 4194304, 'shortest_edge': 65536}},
        'compilation_config': {'mode': 0, 'cudagraph_mode': 'NONE', 'custom_ops': ['none']},
        'worker_extension_cls': 'capture_native_exl3_vision_python_trajectory.TrajectorySelfControl'}
    report.update(runtime=runtime, options=options, specification=spec, source_sha256=digest(Path(__file__)))
    model = None
    try:
        model = LLM(**options)
        report['worker'] = model.collective_rpc('install_trajectory_control', args=(str(args.output), spec), timeout=60)[0]
        sampling = SamplingParams(temperature=0, max_tokens=9, ignore_eos=True)
        first = model.generate(target, sampling, use_tqdm=False)
        require(list(first[0].outputs[0].token_ids) == plan['teacher_tokens'] + [13], 'C1 teacher sequence differs')
        model.collective_rpc('trajectory_c4_phase', timeout=60)
        model.sleep(level=0, mode='keep')
        try:
            model.enqueue([siblings[0], target, *siblings[1:]], sampling, use_tqdm=False)
        finally:
            model.wake_up(tags=['scheduling'])
        second = model.wait_for_completion(use_tqdm=False)
        require(len(second) == 4 and list(second[1].outputs[0].token_ids) == plan['teacher_tokens'] + [13], 'C4 teacher sequence differs')
        report['frames'] = model.collective_rpc('finish_trajectory_control', timeout=60)[0]
        report['actual_prompt_lengths'] = [len(x.prompt_token_ids) for x in second]
        report['status'] = 'DIAGNOSTIC'
    finally:
        if model is not None:
            model.llm_engine.engine_core.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('model', 'receipts', 'fixtures', 'output'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--runtime-image', required=True)
    p.add_argument('--case', choices=('alpha', 'image'), required=True)
    args = p.parse_args()
    args.output.mkdir()
    report = {'status': 'FAIL', 'scope': __doc__, 'case': args.case}
    try:
        run(args, report)
    except Exception as error:
        report['error'] = repr(error)
        traceback.print_exc()
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'case', 'error', 'actual_prompt_lengths')}), flush=True)
    return 0 if report['status'] == 'DIAGNOSTIC' else 1


if __name__ == '__main__':
    raise SystemExit(main())
