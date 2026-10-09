#!/usr/bin/env python3
"""Observe one real pinned MTP1 prefill and its selected-row compact head.

All forwards delegate to the original implementation. No captured hidden/state
is substituted into the live worker. The frozen call is bounded R06 evidence,
not autonomous native MTP or full-target numerical qualification.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path

from capture_projection import IMAGE
from capture_runtime_layout import PROFILE, active_metadata, verify_inputs
from extract_projection import digest, headers, write_safetensors
from runtime_layout import tensor_layout


class MtpDraftCapture:
    def install_mtp_capture(self, output):
        import torch
        runner = self.model_runner
        speculator = runner.speculator
        draft = speculator.model
        self._mtp_output = Path(output)
        self._mtp_tensors, self._mtp_records = {}, {}
        self._mtp_seen = False

        def save(name, tensor):
            host = tensor.detach().cpu().contiguous()
            headers.require(host.numel() <= 2_000_000, 'unbounded MTP observation: ' + name)
            dtype = {torch.float16: 'F16', torch.float32: 'F32',
                     torch.int32: 'I32', torch.int64: 'I64'}[host.dtype]
            headers.require(not host.is_floating_point() or torch.isfinite(host).all().item(),
                            'nonfinite observed MTP boundary: ' + name)
            self._mtp_tensors[name] = (dtype, list(host.shape), host.numpy().tobytes())

        self._mtp_original_run = speculator._run_model

        def run(num_tokens, attn_metadata, slot_mappings, *args, **kwargs):
            observe = not self._mtp_seen
            headers.require(speculator.pcp_manager is None, 'PCP draft input remapping unsupported')
            ids = speculator.input_buffers.input_ids[:num_tokens]
            positions = speculator.input_buffers.positions[:num_tokens]
            hidden = speculator.hidden_states[:num_tokens]
            if observe:
                before(ids, positions, hidden, attn_metadata)
            result = self._mtp_original_run(num_tokens, attn_metadata, slot_mappings, *args, **kwargs)
            if observe:
                headers.require(result[0] is result[1], 'unexpected distinct feedback/logits hidden')
                save('draft_hidden', result[0])
            return result

        def before(ids, positions, hidden, metadata):
            headers.require(ids.shape == (128,) and hidden.shape == (128, 5120),
                            'requires real P128 draft prefill')
            save('input_ids', ids)
            save('positions', positions)
            save('target_hidden', hidden)
            save('selected_rows', speculator.last_token_indices[:1])
            layer = draft.model.layers[0].self_attn.attn
            self._mtp_records['attention_metadata'] = active_metadata(metadata[layer.layer_name])
            self._mtp_records['draft_cache_layout'] = tensor_layout(layer.kv_cache)

        speculator._run_model = run
        self._mtp_original_logits = draft.compute_logits

        def logits(hidden, *args, **kwargs):
            result = self._mtp_original_logits(hidden, *args, **kwargs)
            if not self._mtp_seen:
                headers.require(hidden.shape == (1, 5120), 'draft head must gather one row first')
                compact = draft.lm_head.exl3_draft
                idx = compact['idx']
                headers.require(idx.shape == (65536,), 'incorrect resolved draft subset')
                save('head_hidden', hidden)
                save('token_map', idx)
                save('compact_logits', result.index_select(1, idx))
                save('global_argmax', result.argmax(dim=-1))
                headers.require(result.shape == (1, 248320),
                                'unexpected widened sampler vocabulary')
                self._mtp_records['resolved_sampling'] = {
                    'method': runner.speculative_config.method,
                    'tokens': runner.speculative_config.num_speculative_tokens,
                    'draft_sample_method': runner.speculative_config.draft_sample_method,
                    'rejection_sample_method': runner.speculative_config.rejection_sample_method}
                self._mtp_seen = True
            return result

        draft.compute_logits = logits
        return {'draft_class': type(draft).__module__ + '.' + type(draft).__qualname__}

    def finish_mtp_capture(self):
        self.model_runner.speculator.model.compute_logits = self._mtp_original_logits
        self.model_runner.speculator._run_model = self._mtp_original_run
        headers.require(self._mtp_seen and len(self._mtp_tensors) == 9,
                        'missing real draft-forward/head observations: ' + str(list(self._mtp_tensors)))
        write_safetensors(self._mtp_output, self._mtp_tensors, {'image': IMAGE, 'scope': __doc__})
        return self._mtp_records | {
            'capture_sha256': digest(self._mtp_output.read_bytes()),
            'tensors': {name: {'dtype': tensor[0], 'shape': tensor[1], 'sha256': digest(tensor[2])}
                        for name, tensor in self._mtp_tensors.items()}}


def capture(args):
    report = args.output.with_suffix('.json')
    headers.require(not args.output.exists() and not report.exists(), 'refusing to overwrite MTP capture')
    reference = verify_inputs(args.reference_manifest, args.model, args.image_identity)
    import yaml
    profile = yaml.safe_load(PROFILE.read_text())
    for key, value in profile['env'].items():
        if value is not None:
            os.environ[key] = str(value).replace('{model_dir}', str(PROFILE.parent))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['EXL3_LOADER_REPORT_DIR'] = str(args.output.parent / 'loader')
    from vllm import LLM, SamplingParams
    from vllm.config import ReasoningConfig
    from vllm.engine.arg_utils import EngineArgs
    allowed = {f.name for f in dataclasses.fields(EngineArgs)}
    kwargs = {k: v for k, v in profile['vllm'].items() if k in allowed}
    if isinstance(kwargs.get('reasoning_config'), dict):
        kwargs['reasoning_config'] = ReasoningConfig(**kwargs['reasoning_config'])
    overrides = dict(enforce_eager=True, enable_prefix_caching=False, max_model_len=4352,
        max_num_seqs=1, max_num_batched_tokens=4096,
        speculative_config={'method': 'mtp', 'num_speculative_tokens': 1},
        compilation_config={'mode': 0, 'cudagraph_mode': 'NONE', 'custom_ops': ['none']})
    kwargs.update(overrides, model=str(args.model), worker_extension_cls='capture_mtp_draft.MtpDraftCapture')
    tokens = [1000 + (i * 37) % 4096 for i in range(128)]
    params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
    llm = None
    try:
        llm = LLM(**kwargs)
        ordinary = llm.generate({'prompt_token_ids': tokens}, params, use_tqdm=False)
        installed = llm.collective_rpc('install_mtp_capture', timeout=60, args=(str(args.output),))
        observed = llm.generate({'prompt_token_ids': tokens}, params, use_tqdm=False)
        ordinary_ids = [list(o.outputs[0].token_ids) for o in ordinary]
        observed_ids = [list(o.outputs[0].token_ids) for o in observed]
        headers.require(ordinary_ids == observed_ids, 'observation changed emitted greedy IDs')
        records = llm.collective_rpc('finish_mtp_capture', timeout=60)
        headers.require(len(records) == 1, 'requires one pinned worker')
        result = records[0] | dict(schema=1, image=IMAGE, checkpoint=reference['checkpoint']['identity'],
            profile_sha256=digest(PROFILE.read_bytes()), overrides=overrides, observer=installed,
            ordinary_output_ids=ordinary_ids, observed_output_ids=observed_ids,
            capture_tool_sha256=digest(Path(__file__).read_bytes()), scope=__doc__)
        with report.open('x') as stream:
            json.dump(result, stream, indent=2); stream.write('\n')
        print('MTP_CAPTURE_DONE', result['capture_sha256'], flush=True)
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference-manifest', type=Path, required=True)
    parser.add_argument('--image-identity', required=True)
    parser.add_argument('--output', type=Path, required=True)
    capture(parser.parse_args())
