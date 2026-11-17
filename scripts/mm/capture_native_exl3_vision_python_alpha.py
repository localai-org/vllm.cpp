#!/usr/bin/env python3
"""Bounded pinned Python C1/mixed M4 self-control at actual alpha prefix21.

Teacher force only the known current token2702. Observe complete unmasked
logits before sampling, and restore the selected C1 incoming cache in M4.
This is reference-only diagnosis, never native qualification or performance.
"""
import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

from analyze_native_exl3_vision_adjacent_prefix import PREFIX
from analyze_native_exl3_vision_mtp_prefix import PROMPT_IDS
from analyze_native_exl3_vision_prefix import digest, require
from capture_native_exl3_vision_reference import EXPECTED, REFERENCE_IMAGE
from replay_native_exl3_vision_first_gdn import CONFIG_SHA, BLOCK_SOURCE, validate_model_sources
from replay_native_exl3_vision_target_states import shard_digest
from native_vision_attribution_pins import load_attribution_pins

FROZEN_IMAGES = {
    'orbit': '9ed13de9eb9d4c68576714a041d5690ab696ba269597002b71b154c0956aebe5',
    'comet': '0442d568bf10dffce6afdfd557fe471158a9d8ba87e95ba5029249f599418b02',
}


def recipe(receipts, fixtures):
    artifact_pins = load_attribution_pins()
    pins = {x['file']: x['sha256'] for x in artifact_pins['first_mtp_metadata']}
    path = receipts / 'c2-alpha-mtp-state-v1/prefix-1.json'
    require(digest(path) == pins['prefix-1.json'], 'frozen first-mixed prompt metadata changed')
    md = json.loads(path.read_text())
    historical = receipts / 'vision-mixed-mtp0-v3-result.json'
    pin = artifact_pins['original_mixed_result_sha256']
    require(digest(historical) == pin, 'original failed mixed result changed')
    wave = next(x for x in json.loads(historical.read_text())['waves'] if x['label'] == 'c4-first')
    outputs = {x['label']: x['ids'] for x in wave['cases']}
    require(outputs['alpha'][:21] == PREFIX and md['query_start_loc'] == [0, 4, 22, 267, 512], 'wrong frozen recipe')
    result = {'alpha': {'ids': PROMPT_IDS + PREFIX[:-1], 'query_token': PREFIX[-1]},
              'beta': {'ids': md['input_token_ids'][4:22] + outputs['beta'][:21],
                       'query_token': outputs['beta'][21]}}
    manifest = {x['name']: x for x in json.loads((fixtures / 'native_vision_http/fixtures.json').read_text())}
    # Multimodal identity hashes include processing identity, and are not PNG
    # file digests. Bind rows through the frozen serving response request IDs.
    labels = {x['response']['id']: x['label'] for x in wave['cases']}
    for row in (2, 3):
        image = next(x for x in md['image_features'] if x['request_id'] == md['request_ids'][row])
        label = labels[image['request_id']]
        require(label in ('orbit', 'comet'), 'wrong sibling image')
        entry = manifest[label]
        path = fixtures / 'native_vision_http' / entry['file']
        require(entry['sha256'] == FROZEN_IMAGES[label] and digest(path) == entry['sha256'],
                'frozen sibling image changed')
        ids = md['input_token_ids'][md['query_start_loc'][row]:md['query_start_loc'][row + 1]]
        begin = ids.index(248053) + 1
        end = ids.index(248054)
        require(ids[begin:end] == [248056] * 192 and len(ids) == 245, 'wrong native image expansion')
        # TokensPrompt gets one unexpanded placeholder; the real processor
        # expands it again, avoiding text decode/re-tokenization of the prefix.
        result[label] = {'ids': ids[:begin] + [248056] + ids[end:] + outputs[label][:20],
                         'query_token': outputs[label][20], 'image': str(path),
                         'image_sha256': entry['sha256'], 'native_mm_hash': image['hash']}
    require(set(result) == {'alpha', 'beta', 'orbit', 'comet'} and len(result['alpha']['ids']) == 38,
            'incomplete frozen prompt set')
    return result


class AlphaSelfControl:
    def install_alpha_self_control(self, output, alpha_ids):
        import torch
        from vllm.forward_context import get_forward_context
        self._alpha_root = Path(output)
        self._alpha_ids = alpha_ids
        self._alpha_phase = 'c1'
        self._alpha_frames = {}
        self._alpha_saved = None
        self._alpha_pending = None
        self._alpha_calls = []
        runner = self.model_runner
        model = runner.model
        self._alpha_original_forward = model.forward
        self._alpha_original_logits = model.compute_logits

        def forward(*args, **kwargs):
            # Observe the public model call used by both pinned runner versions;
            # V2 does not have the V1 _model_forward/input_batch interfaces.
            ids = kwargs.get('input_ids')
            embeds = kwargs.get('inputs_embeds')
            # The real multimodal V2 model call uses embeddings even for text
            # decode, setting input_ids=None. Its token buffer remains real.
            if ids is None and isinstance(embeds, torch.Tensor) and embeds.ndim == 2:
                require(type(runner).__qualname__ == 'XPUModelRunnerV2', 'unaudited embedding-only runner')
                ids = runner.input_buffers.input_ids[:embeds.shape[0]]
            if len(self._alpha_calls) < 8:
                self._alpha_calls.append({'phase': self._alpha_phase,
                    'ids_shape': list(ids.shape) if isinstance(ids, torch.Tensor) else None,
                    'positions_shape': list(kwargs['positions'].shape)})
            if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.numel() not in (1, 4):
                return self._alpha_original_forward(*args, **kwargs)
            n = ids.numel()
            positions = kwargs['positions'].detach().cpu()
            require(positions.shape == (3, n), 'actual M-RoPE axes required')
            token_ids = ids.detach().cpu().tolist()
            selected = [i for i in range(n) if token_ids[i] == 2702 and positions[:, i].tolist() == [38, 38, 38]]
            if not selected:
                return self._alpha_original_forward(*args, **kwargs)
            require(len(selected) == 1 and n == (1 if self._alpha_phase == 'c1' else 4), 'wrong selected decode batch')
            row = selected[0]
            ctx = get_forward_context()
            info = {'row': row, 'rows': n, 'positions': positions.tolist(),
                    'input_token_ids': token_ids, 'metadata': {}}
            require(isinstance(embeds, torch.Tensor) and embeds.shape == (n, 5120)
                    and embeds.dtype == torch.float16, 'actual selected embedding required')
            embedding_sha = hashlib.sha256(embeds[row].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            info['selected_embedding_sha256'] = embedding_sha
            if self._alpha_phase == 'c4':
                require(embedding_sha == self._alpha_frames['c1_pre']['info']['selected_embedding_sha256'],
                        'selected model embedding differs')
            pre = self._alpha_views(ctx.attn_metadata, row, n, 38, info)
            phase = self._alpha_phase
            require(phase + '_pre' not in self._alpha_frames, 'duplicate selected forward')
            self._alpha_write_frame(phase + '_pre', pre, info)
            if phase == 'c1':
                self._alpha_saved = {name: value.clone() for name, (value, _) in pre.items()}
            else:
                require(self._alpha_saved is not None and pre.keys() == self._alpha_saved.keys(), 'missing C1 initial state')
                for name, (value, store) in pre.items():
                    saved = self._alpha_saved[name]
                    require(value.shape == saved.shape and value.dtype == saved.dtype, 'different reference state contract')
                    store(saved)
                restored = self._alpha_views(ctx.attn_metadata, row, n, 38, info)
                self._alpha_write_frame(phase + '_restored', restored, info)
                require(all(self._alpha_frames['c1_pre']['blobs'][k]['sha256'] ==
                            self._alpha_frames['c4_restored']['blobs'][k]['sha256'] for k in pre),
                        'selected M4 state not restored exactly')
            result = self._alpha_original_forward(*args, **kwargs)
            post = self._alpha_views(ctx.attn_metadata, row, n, 39, info)
            self._alpha_write_frame(phase + '_post', post, info)
            self._alpha_pending = (phase, row, n)
            return result

        def logits(hidden):
            result = self._alpha_original_logits(hidden)
            if self._alpha_pending is not None:
                phase, row, n = self._alpha_pending
                require(isinstance(result, torch.Tensor) and result.shape == (n, 248320)
                        and result.dtype == torch.float16,
                        'complete unmasked FP16 reference head required: ' + str(getattr(result, 'shape', None))
                        + ' ' + str(getattr(result, 'dtype', None)))
                self._alpha_write_frame(phase + '_logits', {'logits': (result[row], None)}, {'row': row, 'rows': n})
                self._alpha_pending = None
            return result

        model.forward = forward
        model.compute_logits = logits
        return {'layers': len(model.language_model.model.layers), 'model_class': type(model).__qualname__,
                'runner_class': type(runner).__qualname__}

    def _alpha_views(self, metadata, row, n, context, info):
        import torch
        require(isinstance(metadata, dict), 'real forward metadata dictionary required')
        result = {}
        gdn = attn = 0
        for layer in self.model_runner.model.language_model.model.layers:
            if hasattr(layer, 'linear_attn'):
                module = layer.linear_attn
                md = metadata[module.prefix]
                require(md.num_prefills == 0 and md.num_decodes == n and md.num_actual_tokens == n
                        and md.spec_sequence_masks is None, 'ordinary GDN decode metadata required')
                slot = int(md.non_spec_state_indices_tensor[row].item())
                slots = md.non_spec_state_indices_tensor.detach().cpu().reshape(-1).tolist()[:n]
                require(len(slots) == n and len(set(slots)) == n and slots[row] == slot, 'aliased recurrent request owners')
                for label, owner in zip(('conv', 'ssm'), module.kv_cache):
                    value = owner[slot]
                    require(value.dtype == (torch.float16 if label == 'conv' else torch.float32), 'actual GDN dtype differs')
                    require(value.numel() == (30720 if label == 'conv' else 786432), 'actual GDN extent differs')
                    result[f'gdn{gdn}-{label}'] = (value, lambda saved, target=value: target.copy_(saved))
                info['metadata'][module.prefix] = {'kind': 'gdn', 'slot': slot, 'slots': slots,
                                                  'num_actual_tokens': md.num_actual_tokens}
                gdn += 1
            else:
                module = layer.self_attn.attn
                md = metadata[module.layer_name]
                cache = module.kv_cache
                require(cache.ndim == 4 and cache.shape[1] == 4 and cache.shape[-1] == 512
                        and cache.element_size() == 1, 'actual packed FP8 KV layout differs: ' + str(cache.shape))
                require(int(md.seq_lens[row].item()) == 39, 'wrong selected KV length')
                seq_lens = md.seq_lens.detach().cpu().tolist()
                require(sorted(seq_lens) == ([39] if n == 1 else [39, 40, 266, 266]), 'sibling geometry differs')
                require(md.query_start_loc.detach().cpu().tolist() == list(range(n + 1)), 'ordinary attention rows required')
                scales = [float(getattr(module, '_'+kind+'_scale').item()) for kind in ('k', 'v')]
                require(scales == [1.0, 1.0], 'fixed FP8 KV scale contract differs')
                block = cache.shape[2]
                physical_pages = md.block_table[:, 0].detach().cpu().tolist()[:n]
                require(block >= max(seq_lens) and len(set(physical_pages)) == n, 'single-page distinct request owners required')
                indices = torch.arange(context, device=cache.device)
                pages = md.block_table[row].index_select(0, indices // block).long()
                offsets = indices % block
                raw = cache.view(torch.uint8)
                for which, label in enumerate(('k', 'v')):
                    lo, hi = which * 256, (which + 1) * 256
                    value = raw[pages, :, offsets, lo:hi]
                    def store(saved, target=raw, p=pages, o=offsets, left=lo, right=hi):
                        target[p, :, o, left:right] = saved
                    result[f'attn{attn}-{label}'] = (value, store)
                info['metadata'][module.layer_name] = {'kind': 'attention', 'block_size': block,
                    'seq_lens': seq_lens, 'cache_shape': list(cache.shape), 'scales': scales,
                    'physical_pages': physical_pages,
                    'cache_dtype': str(cache.dtype), 'implementation': type(module.impl).__qualname__}
                attn += 1
        require(gdn == 48 and attn == 16 and len(result) == 128, 'incomplete actual model caches')
        return result

    def _alpha_write_frame(self, name, values, info):
        import torch
        require(name not in self._alpha_frames, 'preserve reference frame')
        total = sum(t.numel() * t.element_size() for t, _ in values.values())
        require(0 < total <= 256 * 1024 * 1024, 'bounded reference frame exceeded')
        host = {}
        for key, (value, _) in values.items():
            target = torch.empty(value.shape, dtype=value.dtype, device='cpu', pin_memory=True)
            target.copy_(value, non_blocking=True)
            host[key] = target
        torch.xpu.synchronize()  # Diagnostic batch drain; no polling/hot-path change.
        frame = {'info': info, 'blobs': {}}
        for key, tensor in host.items():
            raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            filename = name + '-' + key + '.bin'
            with (self._alpha_root / filename).open('xb') as output:
                output.write(raw)
            frame['blobs'][key] = {'file': filename, 'sha256': hashlib.sha256(raw).hexdigest(),
                                  'bytes': len(raw), 'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}
        self._alpha_frames[name] = frame

    def set_alpha_phase(self):
        require('c1_logits' in self._alpha_frames and self._alpha_phase == 'c1',
                'C1 reference not captured: ' + json.dumps(self._alpha_calls))
        self._alpha_phase = 'c4'

    def finish_alpha_self_control(self):
        self.model_runner.model.forward = self._alpha_original_forward
        self.model_runner.model.compute_logits = self._alpha_original_logits
        self._alpha_saved = None
        return {'frames': self._alpha_frames, 'calls': self._alpha_calls}


def run(args, report):
    inputs = recipe(args.receipts, args.fixtures)
    require(digest(args.model / 'config.json') == CONFIG_SHA and args.runtime_image == REFERENCE_IMAGE, 'wrong model/runtime pin')
    report['model_sources'] = validate_model_sources(args.model, BLOCK_SOURCE)
    pins = load_attribution_pins()['checkpoint']
    require(digest(args.model / 'model.safetensors.index.json') == pins['index_sha256'], 'target index changed')
    report['whole_checkpoint_shards'] = []
    for pin in pins['whole_checkpoint_shards']:
        path = args.model / pin['file']
        require(Path(pin['file']).name == pin['file'] and path.stat().st_size == pin['bytes'], 'target shard changed')
        actual = shard_digest(path)
        require(actual == pin['sha256'], 'target shard checksum differs')
        report['whole_checkpoint_shards'].append(dict(pin))
    import torch, torchvision, transformers, vllm
    from PIL import Image
    from vllm import LLM, SamplingParams
    runtime = {'torch': torch.__version__, 'torch_git': torch.version.git_version, 'torchvision': torchvision.__version__,
               'transformers': transformers.__version__, 'vllm': vllm.__version__}
    require(runtime == EXPECTED and torch.xpu.is_available(), 'pinned XPU runtime required')
    report['runtime'] = runtime
    report['environment'] = {k: v for k, v in os.environ.items() if k.startswith(('EXL3_', 'VLLM_'))}
    require(os.environ.get('EXL3_INT8_PREFILL') == '1', 'explicit pinned W8A8 prefill required')
    report['recipe'] = inputs
    options = {'model': str(args.model), 'dtype': 'float16', 'trust_remote_code': True, 'max_model_len': 8192,
        'max_num_seqs': 4, 'max_num_batched_tokens': 4096, 'gpu_memory_utilization': .8, 'kv_cache_dtype': 'fp8',
        'block_size': 1600, 'enforce_eager': True, 'enable_prefix_caching': False, 'skip_mm_profiling': True,
        'limit_mm_per_prompt': {'image': 1, 'video': 0},
        'mm_processor_kwargs': {'size': {'longest_edge': 4194304, 'shortest_edge': 65536}},
        'compilation_config': {'mode': 0, 'cudagraph_mode': 'NONE', 'custom_ops': ['none']},
        'worker_extension_cls': 'capture_native_exl3_vision_python_alpha.AlphaSelfControl'}
    report['options'] = options
    model = None
    try:
        model = LLM(**options)
        report['worker'] = model.collective_rpc('install_alpha_self_control', args=(str(args.output), inputs['alpha']['ids']), timeout=60)[0]
        def sampling(token):
            return SamplingParams(temperature=0, max_tokens=2, ignore_eos=True, allowed_token_ids=[token])
        selected_sampling = sampling(2702)
        first = model.generate({'prompt_token_ids': inputs['alpha']['ids']}, selected_sampling, use_tqdm=False)
        require(first[0].prompt_token_ids == inputs['alpha']['ids'], 'C1 token prefix changed')
        require(list(first[0].outputs[0].token_ids) == [2702, 2702], 'C1 query forcing failed')
        model.collective_rpc('set_alpha_phase', timeout=60)
        prompts = []
        for name in ('beta', 'alpha', 'comet', 'orbit'):
            item = inputs[name]
            prompt = {'prompt_token_ids': item['ids']}
            if 'image' in item:
                with Image.open(item['image']) as image:
                    prompt['multi_modal_data'] = {'image': image.copy()}
            prompts.append(prompt)
        names = ('beta', 'alpha', 'comet', 'orbit')
        # Input processing of PIL images takes long enough for a live engine
        # to finish the earlier text requests before all four are admitted.
        # The documented level-0 pause queues all requests without offloading
        # weights or caches; resume once all CPU input preparation is complete.
        model.sleep(level=0, mode='keep')
        try:
            report['queued_request_ids'] = model.enqueue(prompts,
                [sampling(inputs[name]['query_token']) for name in names], use_tqdm=False)
        finally:
            model.wake_up(tags=['scheduling'])
        second = model.wait_for_completion(use_tqdm=False)
        require(len(second) == 4 and all(list(x.outputs[0].token_ids) == [inputs[name]['query_token']] * 2
                                       for name, x in zip(names, second)), 'mixed query forcing failed')
        capture = model.collective_rpc('finish_alpha_self_control', timeout=60)[0]
        report.update(capture)
        require(set(report['frames']) == {'c1_pre', 'c1_post', 'c1_logits', 'c4_pre', 'c4_restored', 'c4_post', 'c4_logits'},
                'missing actual reference phase: ' + json.dumps(report['calls']))
        report['prompt_lengths'] = [len(x.prompt_token_ids) for x in second]
        require(report['prompt_lengths'] == [39, 38, 265, 265], 'actual prompt expansion differs')
        report['status'] = 'CAPTURED'
    finally:
        if model is not None:
            model.llm_engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'receipts', 'fixtures', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--runtime-image', required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve prior reference capture')
    args.output.mkdir()
    report = {'status': 'FAIL', 'scope': __doc__.strip()}
    try:
        run(args, report)
    except Exception as error:
        report['error'] = repr(error)
        traceback.print_exc()
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'error', 'prompt_lengths')}), flush=True)
    return 0 if report['status'] == 'CAPTURED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
