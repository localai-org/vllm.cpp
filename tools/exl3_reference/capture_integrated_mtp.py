#!/usr/bin/env python3
"""Observe P128 followed by two real C1/Q4 target forwards with original MTP.

Read-only eager target hooks; draft generation and rejection sampling remain
original. Full consumed seeds, per-token FP32 SSM snapshots, valid Conv history
and initialized FP8 KV are copied. This is an original trace for native replay,
not native parity, ordinary-reference admission or a performance measurement.
"""
import argparse
import dataclasses
import functools
import json
import os
from pathlib import Path

from capture_projection import IMAGE
from capture_runtime_layout import PROFILE, verify_inputs
from capture_target import TargetCapture, load_prompt, selected_block_layers
from extract_projection import digest, headers, write_safetensors
from mtp_state_scope import initialized_kv_addresses, speculative_state_reads
from runtime_layout import tensor_layout


def observe_prepared_batch(original, save):
    """Observe the real batch object; preserve call arguments and return identity."""
    @functools.wraps(original)
    def prepared(*args, **kwargs):
        result = original(*args, **kwargs)
        save(result)
        return result
    return prepared


class IntegratedMtpCapture(TargetCapture):
    def install_integrated_mtp_capture(self, output, layer0_only=False):
        import torch
        from vllm.forward_context import get_forward_context
        runner = self.model_runner
        headers.require(type(layer0_only) is bool, "invalid attribution scope")
        self._imt_layer0_only = layer0_only
        headers.require(runner.speculative_config.num_speculative_tokens == 3,
                        "requires actual MTP depth3")
        self._imt_output = Path(output)
        headers.require(not self._imt_output.exists(), "refusing existing capture directory")
        self._imt_output.mkdir(parents=True)
        self._imt_records, self._imt_hooks = [], []
        self._imt_current = None
        self._imt_batch = None
        self._imt_embedding_tokens = None
        self._imt_conv_widths, self._imt_ssm_slots = {}, {}
        self._imt_prepare_original = runner.prepare_inputs
        runner.prepare_inputs = observe_prepared_batch(
            runner.prepare_inputs, lambda batch: setattr(self, "_imt_batch", batch))
        modules = dict(runner.model.named_modules())
        first = next(n for n in modules if n.endswith("language_model.model.layers.0"))
        layers = selected_block_layers(modules, first)
        headers.require(all(b.layer_type == ("full_attention" if i % 4 == 3 else "linear_attention")
                            for i, b in layers), "requires actual48GDN/16attention layers")
        embedding = modules[first.rsplit("layers.", 1)[0] + "embed_tokens"]

        def embedding_witness(module, args, kwargs):
            if len(self._imt_records) >= 3:
                return
            value = (dict(zip(("input_ids",), args)) | kwargs)["input_ids"]
            headers.require(value.dtype in (torch.int32, torch.int64) and value.numel() <= 128,
                            "unbounded actual embedding input")
            self._imt_embedding_tokens = value.detach().cpu().tolist()

        self._imt_hooks.append(embedding.register_forward_pre_hook(embedding_witness, with_kwargs=True))

        def save(key, value):
            tensors = self._imt_tensors
            headers.require(key not in tensors and 0 < value.numel() <= 4_000_000,
                            "duplicate/unbounded integrated tensor: " + key)
            dtype = {torch.float16: "F16", torch.float32: "F32", torch.uint8: "U8"}.get(value.dtype)
            headers.require(dtype is not None, "unexpected integrated tensor dtype")
            host = value.detach().cpu().contiguous()
            headers.require(not host.is_floating_point() or torch.isfinite(host).all().item(),
                            "nonfinite integrated tensor: " + key)
            tensors[key] = (dtype, list(host.shape), host.numpy().tobytes())

        def begin_target(module, args, kwargs):
            if len(self._imt_records) >= 3:
                return
            context = get_forward_context().attn_metadata
            if not context or not getattr(runner.req_states, "req_id_to_index", {}):
                return
            batch = self._imt_batch
            headers.require(batch is not None and batch.num_reqs == 1 and
                            len(batch.req_ids) == 1 and batch.num_tokens_after_padding == batch.num_tokens,
                            "capture requires real unpadded C1 batch")
            values = dict(zip(("input_ids", "positions"), args)) | kwargs
            # VLM runners may embed the text before the root forward and pass
            # input_ids=None. Cross-check these batch IDs against the actual
            # embedding-module call before accepting the completed capture.
            token_input = values["input_ids"]
            tokens = (batch.input_ids if token_input is None else token_input).detach().cpu().tolist()
            positions = values["positions"].detach().cpu().tolist()
            step = len(self._imt_records)
            rows = 128 if step == 0 else 4
            qsl = batch.query_start_loc.detach().cpu().tolist()
            seq = batch.seq_lens.detach().cpu().tolist()
            headers.require(len(tokens) == rows and all(type(t) is int and 0 <= t < 248320 for t in tokens)
                            and qsl == [0, rows] and len(seq) == 1 and 128 <= seq[0] <= 160 and
                            positions == [list(range(seq[0] - rows, seq[0]))] * 3,
                            "unexpected integrated target input/position scope")
            self._imt_current = {"step": step, "request_ids": list(batch.req_ids),
                                 "token_ids": tokens, "positions": positions,
                                 "root_input_ids_present": token_input is not None,
                                 "query_start_loc": qsl, "seq_lens": seq,
                                 "gdn": {}, "attention": {}}
            self._imt_tensors = {}

        def end_target(module, args, kwargs, result):
            if self._imt_current is None:
                return
            record = self._imt_current
            headers.require(len(record["gdn"]) == (1 if layer0_only else 48) and
                            len(record["attention"]) == (0 if layer0_only else 16),
                            "missing integrated layer boundaries")
            headers.require(self._imt_embedding_tokens == record["token_ids"],
                            "actual consumed embedding tokens differ from target witness")
            record["embedding_token_ids_exact"] = True
            save("target_hidden", result)
            path = self._imt_output / ("step-" + str(record["step"]) + ".safetensors")
            write_safetensors(path, self._imt_tensors, {"image": IMAGE, "scope": __doc__})
            record["file"] = path.name
            record["sha256"] = digest(path.read_bytes())
            record["tensors"] = {k: {"dtype": v[0], "shape": v[1], "sha256": digest(v[2])}
                                 for k, v in self._imt_tensors.items()}
            self._imt_records.append(record)
            with path.with_suffix(".json").open("x") as stream:
                json.dump(record, stream, indent=2, allow_nan=False); stream.write("\n")
            self._imt_current, self._imt_tensors = None, {}
            print("INTEGRATED_MTP_STEP", record["step"], len(record["token_ids"]), flush=True)

        self._imt_hooks.append(runner.model.register_forward_pre_hook(begin_target, with_kwargs=True))
        self._imt_hooks.append(runner.model.register_forward_hook(end_target, with_kwargs=True))

        def gdn_boundary(index, mixer, stage):
            record = self._imt_current
            if record is None:
                return
            meta = get_forward_context().attn_metadata[mixer.prefix]
            conv, ssm = mixer.kv_cache
            headers.require(conv.dtype == torch.float16 and conv.ndim == 3 and
                            conv.shape[1] == 6 and conv.shape[2] == 10240 and
                            ssm.dtype == torch.float32 and tuple(ssm.shape[1:]) == (48, 128, 128),
                            "unexpected integrated GDN cache layout")
            key = "l" + str(index)
            if stage == "before":
                if record["step"] == 0:
                    headers.require(meta.num_prefills == 1 and meta.num_spec_decodes == 0 and
                                    not meta.prefill_has_initial_state.detach().cpu().item(),
                                    "requires cold initialized-by-prefill seed")
                    base = int(meta.prefill_state_indices.detach().cpu().item())
                    headers.require(0 <= base < min(conv.shape[0], ssm.shape[0]), "invalid prefill slot")
                    plan = {"conv_slot": base, "conv_after": [0, 3], "ssm_after": [base]}
                    entry = {"cold_unconsumed_seed_excluded": True, "plan": plan}
                else:
                    headers.require(meta.num_prefills == meta.num_decodes == 0 and meta.num_spec_decodes == 1,
                                    "requires pure integrated C1 spec step")
                    qsl = meta.spec_query_start_loc.detach().cpu().tolist()
                    indices = meta.spec_state_indices_tensor.detach().cpu().tolist()
                    accepted = meta.num_accepted_tokens.detach().cpu().tolist()
                    plan = speculative_state_reads(qsl, indices, accepted,
                                                  min(conv.shape[0], ssm.shape[0]),
                                                  self._imt_conv_widths[index], self._imt_ssm_slots[index])[0]
                    entry = {"cold_unconsumed_seed_excluded": False, "plan": plan,
                             "spec_query_start_loc": qsl, "spec_state_indices": indices,
                             "previous_accepted_tokens": accepted}
                    begin, end = plan["conv_before"]
                    save(key + "_conv_before", conv[plan["conv_slot"], begin:end])
                    save(key + "_ssm_before", ssm[plan["ssm_before"]])
                entry["conv_cache_layout"] = tensor_layout(conv)
                entry["ssm_cache_layout"] = tensor_layout(ssm)
                record["gdn"][key] = entry
            else:
                plan = record["gdn"][key]["plan"]
                begin, end = plan["conv_after"]
                save(key + "_conv_after", conv[plan["conv_slot"], begin:end])
                for token, slot in enumerate(plan["ssm_after"]):
                    save(key + "_ssm_after_t" + str(token), ssm[slot])
                # Replace initialization provenance: old rejected/spare snapshots
                # remain allocated but are not eligible consumed seeds next time.
                self._imt_conv_widths[index] = {plan["conv_slot"]: end}
                self._imt_ssm_slots[index] = set(plan["ssm_after"])

        def kv_boundary(index, attn, stage):
            record = self._imt_current
            if record is None:
                return
            meta = get_forward_context().attn_metadata[attn.layer_name]
            kv = attn.kv_cache
            headers.require(kv.dtype == torch.uint8 and kv.ndim == 4 and
                            tuple(kv.shape[1:]) == (4, 1600, 512), "unexpected integrated FP8 KV layout")
            seq, positions = record["seq_lens"][0], record["positions"][0]
            headers.require(meta.num_actual_tokens == len(record["token_ids"]) and
                            meta.query_start_loc.detach().cpu().tolist() == record["query_start_loc"] and
                            meta.seq_lens.detach().cpu().tolist() == record["seq_lens"],
                            "attention metadata differs from actual target batch")
            blocks = meta.block_table[0, :(seq + 1599) // 1600].detach().cpu().tolist()
            slots = meta.slot_mapping[:len(positions)].detach().cpu().tolist()
            addresses = initialized_kv_addresses(1600, int(kv.shape[0]), seq, positions, blocks, slots, stage)
            key = "l" + str(index)
            entry = {"addresses": addresses, "written_logical_length": len(addresses),
                     "cache_layout": tensor_layout(kv)}
            for name in ("_k_scale", "_v_scale"):
                scale = float(getattr(attn, name).detach().cpu().item())
                headers.require(scale == 1.0, "unexpected original FP8 scale")
                entry[name] = scale
            record["attention"].setdefault(key, {})[stage] = entry
            if addresses:
                bi = torch.tensor([a[0] for a in addresses], dtype=torch.int64, device=kv.device)
                pi = torch.tensor([a[1] for a in addresses], dtype=torch.int64, device=kv.device)
                values = kv.transpose(1, 2)[bi, pi]
                save(key + "_key_" + stage, values[..., :256])
                save(key + "_value_" + stage, values[..., 256:])

        for index, block in layers:
            if layer0_only and index != 0:
                continue
            if index % 4 != 3:
                observer = lambda stage, i=index, m=block.linear_attn: gdn_boundary(i, m, stage)
            else:
                observer = lambda stage, i=index, a=block.self_attn.attn: kv_boundary(i, a, stage)
            self._imt_hooks.append(block.register_forward_pre_hook(
                lambda module, args, kwargs, observe=observer: observe("before"), with_kwargs=True))
            self._imt_hooks.append(block.register_forward_hook(
                lambda module, args, kwargs, result, observe=observer: observe("after"), with_kwargs=True))
        if layer0_only:
            block = layers[0][1]
            for label, module in (("input_norm", block.input_layernorm),
                                  ("qkvz", block.linear_attn.in_proj_qkvz),
                                  ("ba", block.linear_attn.in_proj_ba)):
                def projection(module, args, kwargs, result, label=label):
                    if self._imt_current is not None:
                        save("l0_" + label, result[0] if isinstance(result, tuple) else result)
                self._imt_hooks.append(module.register_forward_hook(projection, with_kwargs=True))

            def core_boundary(module, args, kwargs):
                if self._imt_current is not None:
                    values = dict(zip(("x", "z"), args)) | kwargs
                    save("l0_core", values["x"].reshape(-1, 48, 128))
                    save("l0_z", values["z"].reshape(-1, 48, 128))
            self._imt_hooks.append(block.linear_attn.norm.register_forward_pre_hook(core_boundary, with_kwargs=True))
        return {"layers": 1 if layer0_only else 64, "target_forwards": 3,
                "changes_arithmetic": False, "layer0_attribution_only": layer0_only}

    def finish_integrated_mtp_capture(self):
        for hook in self._imt_hooks:
            hook.remove()
        self.model_runner.prepare_inputs = self._imt_prepare_original
        headers.require(len(self._imt_records) == 3 and self._imt_current is None,
                        "missing complete P128/twoQ4 integrated sequence")
        return {"steps": self._imt_records}


def capture(args):
    headers.require(not args.output_dir.exists(), "refusing to overwrite integrated capture")
    tokens = load_prompt(args.prompt_ids)
    reference = verify_inputs(args.reference_manifest, args.model_dir, args.image_identity)
    import yaml
    profile = yaml.safe_load(PROFILE.read_text())
    for key, value in profile["env"].items():
        if value is not None:
            os.environ[key] = str(value).replace("{model_dir}", str(PROFILE.parent))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["EXL3_LOADER_REPORT_DIR"] = str(args.output_dir.parent / (args.output_dir.name + "-loader"))
    from vllm import LLM, SamplingParams
    from vllm.config import ReasoningConfig
    from vllm.engine.arg_utils import EngineArgs
    import torch
    headers.require(torch.__version__ == "2.13.0+xpu" and "B70" in torch.xpu.get_device_name(0),
                    "requires pinned Torch XPU/B70")
    allowed = {f.name for f in dataclasses.fields(EngineArgs)}
    kwargs = {k: v for k, v in profile["vllm"].items() if k in allowed}
    if isinstance(kwargs.get("reasoning_config"), dict):
        kwargs["reasoning_config"] = ReasoningConfig(**kwargs["reasoning_config"])
    # Keep the profile's actual MTP/draft-head/rejection settings. Only the
    # execution mode is eager so Python observers see each target boundary.
    overrides = {"enforce_eager": True,
                 "compilation_config": {"mode": 0, "cudagraph_mode": "NONE", "custom_ops": ["none"]}}
    kwargs.update(overrides, model=str(args.model_dir),
                  worker_extension_cls="capture_integrated_mtp.IntegratedMtpCapture")
    llm, ba_policy = None, None
    try:
        llm = LLM(**kwargs)
        if args.deterministic_ba:
            ba_policy = llm.collective_rpc("install_deterministic_ba", timeout=60)
        params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
        ordinary = llm.generate({"prompt_token_ids": tokens}, params, use_tqdm=False)
        headers.require(llm.reset_prefix_cache(), "prefix reset failed")
        installed = llm.collective_rpc("install_integrated_mtp_capture", timeout=60,
                                      args=(str(args.output_dir), args.layer0_only))
        observed = llm.generate({"prompt_token_ids": tokens}, params, use_tqdm=False)
        result = llm.collective_rpc("finish_integrated_mtp_capture", timeout=60)
        headers.require(len(result) == 1, "requires one original worker")
        ids = lambda outputs: [list(o.outputs[0].token_ids) for o in outputs]
        ordinary_exact = ids(ordinary) == ids(observed)
        schema = "b70-integrated-mtp-layer0-attribution-v1" if args.layer0_only else "b70-integrated-mtp-C1-Q4-v1"
        result = result[0] | {"schema": schema, "image": IMAGE,
                  "checkpoint": reference["checkpoint"]["identity"], "observer": installed,
                  "prompt_token_ids": tokens, "output_ids": ids(observed),
                  "ordinary_output_ids": ids(ordinary), "ordinary_ids_exact": ordinary_exact,
                  "reference_label": "controlled_deterministic_BA" if args.deterministic_ba else "default",
                  "ba_policy": ba_policy, "overrides": overrides,
                  "capture_tool_sha256": digest(Path(__file__).read_bytes()),
                  "scope_helper_sha256": digest(Path(__file__).with_name("mtp_state_scope.py").read_bytes()),
                  "profile_sha256": digest(PROFILE.read_bytes()),
                  "reference_manifest_sha256": digest(args.reference_manifest.read_bytes()),
                  "scope": __doc__ if not args.layer0_only else
                      "Layer0 attribution only: actual target norm/projections/core and full active GDN snapshots; no attention KV or whole-model state parity"}
        with (args.output_dir / "capture.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
        # Preserve the observed trace and both trajectories even on a failed
        # repeatability gate. Do not silently turn default nondeterminism into
        # an accepted controlled reference or report a successful worker exit.
        headers.require(ordinary_exact, "observation/repeat changed emitted greedy IDs")
        print("INTEGRATED_MTP_CAPTURE_DONE", args.output_dir, flush=True)
    finally:
        if llm is not None:
            try:
                if ba_policy is not None:
                    headers.require(llm.collective_rpc("restore_deterministic_ba", timeout=60) ==
                                    [{"restored_modules": 48}], "BA restoration failed")
            finally:
                llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--prompt-ids", type=Path)
    parser.add_argument("--deterministic-ba", action="store_true")
    parser.add_argument("--layer0-only", action="store_true",
                        help="small separate attribution capture; preserves full-state captures and their gates")
    capture(parser.parse_args())
