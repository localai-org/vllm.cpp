#!/usr/bin/env python3
"""Full-array digest observation of actual original C4 -> C2 -> C1 target MTP.

Cold four-request P128, then three real target forwards with explicit request
cancellations between steps. Original MTP3/drafter/rejection remain unchanged.
Every consumed/produced GDN array and initialized FP8 KV array is copied in full
and hashed; only digests/metadata persist. Not serving timing or native parity.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path

from capture_integrated_mtp import observe_prepared_batch
from capture_projection import IMAGE
from capture_runtime_layout import PROFILE, verify_inputs
from capture_target import TargetCapture, load_prompt, selected_block_layers
from extract_projection import digest, headers
from mtp_state_scope import integrated_batch_reads, initialized_kv_addresses, speculative_state_reads
from runtime_layout import tensor_layout


class MtpTransitionCapture(TargetCapture):
    def install_mtp_transition_capture(self, output):
        import torch
        from vllm.forward_context import get_forward_context
        runner = self.model_runner
        headers.require(runner.speculative_config.num_speculative_tokens == 3, "requires actual MTP3")
        self._tr_output = Path(output)
        headers.require(not self._tr_output.exists(), "refusing existing transition capture")
        self._tr_output.mkdir(parents=True)
        self._tr_steps, self._tr_hooks, self._tr_current = [], [], None
        self._tr_widths, self._tr_slots, self._tr_previous = {}, {}, {}
        self._tr_batch, self._tr_embedding = None, None
        self._tr_original = runner.prepare_inputs
        runner.prepare_inputs = observe_prepared_batch(self._tr_original,
                                                       lambda batch: setattr(self, "_tr_batch", batch))
        modules = dict(runner.model.named_modules())
        first = next(n for n in modules if n.endswith("language_model.model.layers.0"))
        layers = selected_block_layers(modules, first)
        headers.require(len(layers) == 64 and all(
            block.layer_type == ("full_attention" if i % 4 == 3 else "linear_attention")
            for i, block in layers), "requires actual48GDN/16attention layers")

        def embedding(module, args, kwargs):
            value = (dict(zip(("input_ids",), args)) | kwargs)["input_ids"]
            headers.require(value.numel() <= 512, "unbounded target embedding witness")
            self._tr_embedding = value.detach().cpu().tolist()

        embed = modules[first.rsplit("layers.", 1)[0] + "embed_tokens"]
        self._tr_hooks.append(embed.register_forward_pre_hook(embedding, with_kwargs=True))

        def save(key, value):
            headers.require(key not in self._tr_current["tensors"] and
                            0 < value.numel() <= 4_000_000, "duplicate/unbounded array: " + key)
            value = value.detach().cpu().contiguous()
            dtype = {torch.float16: "F16", torch.float32: "F32", torch.uint8: "U8"}.get(value.dtype)
            headers.require(dtype is not None and
                            (not value.is_floating_point() or torch.isfinite(value).all().item()),
                            "unexpected/nonfinite array: " + key)
            raw = value.numpy().tobytes()
            self._tr_current["tensors"][key] = {"dtype": dtype, "shape": list(value.shape),
                                                  "bytes": len(raw), "sha256": digest(raw), "finite": True}
            return raw

        def begin(module, args, kwargs):
            headers.require(len(self._tr_steps) < 4, "unexpected extra target forward")
            batch = self._tr_batch
            headers.require(batch is not None and batch.num_tokens_after_padding == batch.num_tokens and
                            batch.num_reqs_after_padding == batch.num_reqs, "requires actual unpadded batch")
            values = dict(zip(("input_ids", "positions"), args)) | kwargs
            token_input = values["input_ids"]
            tokens = (batch.input_ids if token_input is None else token_input).detach().cpu().tolist()
            positions = values["positions"].detach().cpu().tolist()
            qsl, seq = batch.query_start_loc.detach().cpu().tolist(), batch.seq_lens.detach().cpu().tolist()
            rows = integrated_batch_reads(list(batch.req_ids), tokens, positions, qsl, seq)
            step = len(self._tr_steps)
            headers.require(len(rows) == (4, 4, 2, 1)[step] and
                            all(r["cold_prefill"] == (step == 0) for r in rows),
                            "actual target did not execute bounded C4->C2->C1 transition")
            self._tr_current = {"step": step, "request_ids": list(batch.req_ids), "token_ids": tokens,
                                "positions": positions, "query_start_loc": qsl, "seq_lens": seq,
                                "rows": rows, "root_input_ids_present": token_input is not None,
                                "tensors": {}, "gdn": {}, "attention": {}, "continuity_checks": 0}

        def end(module, args, kwargs, result):
            record = self._tr_current
            headers.require(record is not None and len(record["gdn"]) == 48 and
                            len(record["attention"]) == 16 and self._tr_embedding == record["token_ids"],
                            "incomplete layer or actual embedding witness")
            record["embedding_token_ids_exact"] = True
            save("target_hidden", result)
            path = self._tr_output / ("step-" + str(record["step"]) + ".json")
            with path.open("x") as stream:
                json.dump(record, stream, indent=2, allow_nan=False); stream.write("\n")
            self._tr_steps.append(record)
            self._tr_current = None
            print("MTP_TRANSITION_STEP", record["step"], len(record["rows"]), len(record["tensors"]), flush=True)

        self._tr_hooks.append(runner.model.register_forward_pre_hook(begin, with_kwargs=True))
        self._tr_hooks.append(runner.model.register_forward_hook(end, with_kwargs=True))

        def gdn(index, mixer, stage):
            record = self._tr_current
            if record is None:
                return
            meta = get_forward_context().attn_metadata[mixer.prefix]
            conv, ssm = mixer.kv_cache
            headers.require(conv.dtype == torch.float16 and tuple(conv.shape[1:]) == (6, 10240) and
                            ssm.dtype == torch.float32 and tuple(ssm.shape[1:]) == (48, 128, 128),
                            "unexpected original cache geometry")
            key, cold, count = "l" + str(index), record["step"] == 0, len(record["rows"])
            previous = self._tr_previous.setdefault(key, {})
            if stage == "before":
                if cold:
                    headers.require(meta.num_prefills == count and meta.num_spec_decodes == meta.num_decodes == 0 and
                                    meta.prefill_query_start_loc.detach().cpu().tolist() == record["query_start_loc"] and
                                    meta.prefill_has_initial_state.detach().cpu().tolist() == [False] * count,
                                    "requires actual cold C4 prefill")
                    indices = meta.prefill_state_indices.detach().cpu().tolist()
                    headers.require(len(indices) == count and len(set(indices)) == count and
                                    all(0 <= s < min(conv.shape[0], ssm.shape[0]) for s in indices),
                                    "invalid actual prefill state ownership")
                    plans = [{"row": r, "conv_slot": s, "conv_after": [0, 3], "ssm_after": [s]}
                             for r, s in enumerate(indices)]
                    entry = {"cold_unconsumed_seed_excluded": True, "plans": plans}
                else:
                    headers.require(meta.num_spec_decodes == count and meta.num_prefills == meta.num_decodes == 0,
                                    "requires pure real speculative transition")
                    qsl = meta.spec_query_start_loc.detach().cpu().tolist()
                    indices = meta.spec_state_indices_tensor.detach().cpu().tolist()
                    accepted = meta.num_accepted_tokens.detach().cpu().tolist()
                    token_indices = meta.spec_token_indx.detach().cpu().tolist()
                    headers.require(qsl == record["query_start_loc"] and
                                    token_indices == list(range(len(record["token_ids"]))),
                                    "speculative compact mapping differs from actual rows")
                    plans = speculative_state_reads(qsl, indices, accepted, min(conv.shape[0], ssm.shape[0]),
                                                     self._tr_widths[index], self._tr_slots[index])
                    entry = {"cold_unconsumed_seed_excluded": False, "plans": plans,
                             "spec_query_start_loc": qsl, "spec_state_indices": indices,
                             "spec_token_indx": token_indices, "previous_accepted_tokens": accepted}
                    for plan in plans:
                        r = plan["row"]; req = record["request_ids"][r]; prefix = "r" + str(r) + "_" + key
                        begin, finish = plan["conv_before"]
                        cv = save(prefix + "_conv_before", conv[plan["conv_slot"], begin:finish])
                        sv = save(prefix + "_ssm_before", ssm[plan["ssm_before"]])
                        prev = previous[req]
                        headers.require(prev["conv_slot"] == plan["conv_slot"] and
                                        cv == prev["conv"][begin * 20480:finish * 20480] and
                                        digest(sv) == prev["ssm"][plan["ssm_before"]],
                                        "consumed original Conv/SSM differs from completed producer")
                        record["continuity_checks"] += 2
                entry["conv_cache_layout"], entry["ssm_cache_layout"] = tensor_layout(conv), tensor_layout(ssm)
                record["gdn"][key] = entry
            else:
                widths, slots, next_previous = {}, set(), {}
                for plan in record["gdn"][key]["plans"]:
                    r = plan["row"]; req = record["request_ids"][r]; prefix = "r" + str(r) + "_" + key
                    begin, finish = plan["conv_after"]
                    cv = save(prefix + "_conv_after", conv[plan["conv_slot"], begin:finish])
                    hashes = {}
                    for token, slot in enumerate(plan["ssm_after"]):
                        hashes[slot] = digest(save(prefix + "_ssm_after_t" + str(token), ssm[slot]))
                    widths[plan["conv_slot"]] = finish; slots.update(plan["ssm_after"])
                    next_previous[req] = {"conv_slot": plan["conv_slot"], "conv": cv, "ssm": hashes}
                self._tr_widths[index], self._tr_slots[index] = widths, slots
                self._tr_previous[key] = next_previous

        def attention(index, attn, stage):
            record = self._tr_current
            if record is None:
                return
            meta = get_forward_context().attn_metadata[attn.layer_name]
            kv = attn.kv_cache
            headers.require(kv.dtype == torch.uint8 and tuple(kv.shape[1:]) == (4, 1600, 512) and
                            meta.num_actual_tokens == len(record["token_ids"]) and
                            meta.query_start_loc.detach().cpu().tolist() == record["query_start_loc"] and
                            meta.seq_lens.detach().cpu().tolist() == record["seq_lens"],
                            "attention layout/actual active metadata mismatch")
            headers.require(all(float(getattr(attn, name).detach().cpu().item()) == 1.0
                                for name in ("_k_scale", "_v_scale")), "unexpected FP8 scales")
            key = "l" + str(index)
            previous = self._tr_previous.setdefault(key, {})
            entries, next_previous = [], {}
            for row in record["rows"]:
                r, begin, finish, req = row["row"], row["begin"], row["end"], row["request_id"]
                seq = record["seq_lens"][r]; positions = record["positions"][0][begin:finish]
                blocks = meta.block_table[r, :(seq + 1599) // 1600].detach().cpu().tolist()
                slots = meta.slot_mapping[begin:finish].detach().cpu().tolist()
                addresses = initialized_kv_addresses(1600, int(kv.shape[0]), seq, positions, blocks, slots, stage)
                entries.append({"row": r, "addresses": addresses, "written_logical_length": len(addresses)})
                if not addresses:
                    continue
                bi = torch.tensor([a[0] for a in addresses], dtype=torch.int64, device=kv.device)
                pi = torch.tensor([a[1] for a in addresses], dtype=torch.int64, device=kv.device)
                values = kv[bi, :, pi, :]
                prefix = "r" + str(r) + "_" + key
                raw = {name: save(prefix + "_" + name + "_" + stage, value)
                       for name, value in (("key", values[..., :256]), ("value", values[..., 256:]))}
                if stage == "before":
                    headers.require(all(data == previous[req][name][:len(data)] for name, data in raw.items()),
                                    "initialized original KV prefix differs from completed producer")
                    record["continuity_checks"] += 2
                else:
                    next_previous[req] = raw
            record["attention"].setdefault(key, {"cache_layout": tensor_layout(kv), "_k_scale": 1.0,
                                                "_v_scale": 1.0})[stage] = entries
            if stage == "after":
                self._tr_previous[key] = next_previous

        for index, block in layers:
            observer = (lambda stage, i=index, m=block.linear_attn: gdn(i, m, stage)) if index % 4 != 3 else (
                lambda stage, i=index, m=block.self_attn.attn: attention(i, m, stage))
            self._tr_hooks.append(block.register_forward_pre_hook(
                lambda module, args, kwargs, fn=observer: fn("before"), with_kwargs=True))
            self._tr_hooks.append(block.register_forward_hook(
                lambda module, args, kwargs, result, fn=observer: fn("after"), with_kwargs=True))
        return {"layers": 64, "expected_target_counts": [4, 4, 2, 1], "full_array_digest_only": True}

    def finish_mtp_transition_capture(self):
        for hook in self._tr_hooks:
            hook.remove()
        self.model_runner.prepare_inputs = self._tr_original
        return {"steps": self._tr_steps, "actual_target_counts": [len(s["rows"]) for s in self._tr_steps]}


def drive(engine, prompts, sampling):
    """Synchronous engine steps with explicit cancellation, no target forcing."""
    ids, outputs = {}, []
    for i, prompt in enumerate(prompts):
        external = "transition-" + str(i)
        ids[external] = engine.add_request(external, {"prompt_token_ids": prompt}, sampling)
    try:
        for step in range(4):
            headers.require(engine.has_unfinished_requests(), "transition ended before target step")
            values = engine.step()
            outputs.append({o.request_id: list(o.outputs[0].token_ids) for o in values})
            if step == 1:
                engine.abort_request(["transition-0", "transition-1"])
            elif step == 2:
                engine.abort_request(["transition-2"])
    finally:
        engine.abort_request(list(ids))
    headers.require(not engine.has_unfinished_requests(), "cancelled requests remain live")
    return {"internal_request_ids": ids, "step_outputs": outputs,
            "cancellations_after_target_steps": {"1": ["transition-0", "transition-1"], "2": ["transition-2"],
                                                  "3": ["transition-3"]}}


def capture(args):
    headers.require(not args.output_dir.exists(), "refusing existing transition output")
    reference = verify_inputs(args.reference_manifest, args.model_dir, args.image_identity)
    import yaml
    profile = yaml.safe_load(PROFILE.read_text())
    for key, value in profile["env"].items():
        if value is not None:
            os.environ[key] = str(value).replace("{model_dir}", str(PROFILE.parent))
    # In-process synchronous EngineCore lets cancellation precede the next
    # scheduled forward. This diagnostic override is not a serving score.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
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
    overrides = {"max_num_seqs": 4, "max_num_batched_tokens": 512, "async_scheduling": False,
                 "enforce_eager": True, "compilation_config": {"mode": 0, "cudagraph_mode": "NONE", "custom_ops": ["none"]}}
    kwargs.update(overrides, model=str(args.model_dir), worker_extension_cls="capture_mtp_transition.MtpTransitionCapture")
    base = load_prompt(None)
    prompts = [[t + i * 19 for t in base] for i in range(4)]
    llm, ba_policy = None, None
    try:
        llm = LLM(**kwargs)
        ba_policy = llm.collective_rpc("install_deterministic_ba", timeout=60)
        params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
        ordinary = drive(llm.llm_engine, prompts, params)
        headers.require(llm.reset_prefix_cache(), "cold prefix reset failed")
        installed = llm.collective_rpc("install_mtp_transition_capture", timeout=60, args=(str(args.output_dir),))
        observed = drive(llm.llm_engine, prompts, params)
        records = llm.collective_rpc("finish_mtp_transition_capture", timeout=60)
        headers.require(len(records) == 1, "requires one original worker")
        exact = ordinary["step_outputs"] == observed["step_outputs"]
        result = records[0] | {"schema": "b70-integrated-mtp-C4-C2-C1-digests-v1", "image": IMAGE,
                  "checkpoint": reference["checkpoint"]["identity"], "reference_label": "controlled_deterministic_BA",
                  "ordinary_observed_outputs_exact": exact, "ordinary": ordinary, "observed": observed,
                  "prompt_token_ids": prompts, "observer": installed, "ba_policy": ba_policy, "overrides": overrides,
                  "environment_overrides": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
                  "capture_tool_sha256": digest(Path(__file__).read_bytes()),
                  "scope_helper_sha256": digest(Path(__file__).with_name("mtp_state_scope.py").read_bytes()),
                  "profile_sha256": digest(PROFILE.read_bytes()),
                  "reference_manifest_sha256": digest(args.reference_manifest.read_bytes()), "scope": __doc__}
        with (args.output_dir / "capture.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
        headers.require(result["actual_target_counts"] == [4, 4, 2, 1] and exact,
                        "actual transition or ordinary/observed output gate fails; evidence preserved")
        print("MTP_TRANSITION_CAPTURE_PASS", flush=True)
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
    for name in ("output-dir", "reference-manifest", "model-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    capture(parser.parse_args())
