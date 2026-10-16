#!/usr/bin/env python3
"""Read-only original layer3 P128/D1 attention and active FP8 byte capture."""
import argparse
import dataclasses
import functools
import json
import os
from pathlib import Path

from capture_block import BlockCapture
from capture_projection import IMAGE
from capture_runtime_layout import PROFILE, active_metadata, verify_inputs
from extract_projection import digest, headers, write_safetensors
from runtime_layout import describe, initialized_values, tensor_layout


def attention_capture_step(selected_step, captured_steps):
    headers.require(type(selected_step) is int and selected_step in (-1, 29),
                    "attention observation supports P128/D1 or selected D29 only")
    headers.require(type(captured_steps) is int and captured_steps >= 0,
                    "invalid captured attention step count")
    if selected_step == 29:
        headers.require(captured_steps == 0, "repeated selected D29 attention")
        return "d29", 1, 157, 156
    headers.require(captured_steps < 2, "extra P128/D1 attention step")
    return ("p128", 128, 128, 0) if captured_steps == 0 else ("d1", 1, 129, 128)


def active_cache_addresses(page_size, capacity, seq_len, blocks, slots, positions, *, max_seq_len=129):
    """Validate written rows against actual metadata before reading cache bytes."""
    headers.require(type(page_size) is int and page_size > 0 and
                    type(capacity) is int and capacity > 0 and
                    type(max_seq_len) is int and max_seq_len in (129, 157) and
                    type(seq_len) is int and 0 < seq_len <= max_seq_len, "bad bounded cache geometry")
    headers.require(len(blocks) == (seq_len + page_size - 1) // page_size and
                    all(type(b) is int and 0 <= b < capacity for b in blocks),
                    "invalid active block table")
    addresses = [(blocks[p // page_size], p % page_size) for p in range(seq_len)]
    headers.require(bool(positions) and positions == list(range(seq_len - len(positions), seq_len)) and
                    len(slots) == len(positions) and
                    all(type(s) is int for s in slots) and
                    slots == [addresses[p][0] * page_size + addresses[p][1] for p in positions],
                    "slot mapping differs from written logical positions")
    return addresses


def observe_projection(original, save):
    @functools.wraps(original)
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        save(result)
        return result
    return observed


class AttentionCapture(BlockCapture):
    def install_block_capture(self, output, selected_step=-1, step_counter=None):
        attention_capture_step(selected_step, 0)
        headers.require(selected_step == -1 or callable(step_counter), "selected attention needs actual step counter")
        import torch
        from vllm.forward_context import get_forward_context
        runner = self.model_runner
        layers = [(n, m) for n, m in runner.model.named_modules()
                  if n.endswith("language_model.model.layers.3")]
        headers.require(len(layers) == 1 and layers[0][1].layer_type == "full_attention",
                        "requires unique real full-attention layer3")
        name, layer = layers[0]
        mixer, attn = layer.self_attn, layer.self_attn.attn
        self._block_output = Path(output)
        self._block_tensors, self._block_records, self._block_hooks = {}, [], []
        self._block_current = None
        self._attention_selected_step = selected_step

        def save(stage, value):
            headers.require(value is not None and self._block_current is not None,
                            "missing attention boundary")
            key = self._block_current + "_" + stage
            headers.require(key not in self._block_tensors and value.numel() <= 5_000_000,
                            "duplicate/unbounded attention stage: " + key)
            host = value.detach().cpu().contiguous()
            dtype = {torch.float16: "F16", torch.float32: "F32", torch.int32: "I32",
                     torch.int64: "I64", torch.uint8: "U8"}.get(host.dtype)
            headers.require(dtype is not None and
                            (not host.is_floating_point() or torch.isfinite(host).all().item()),
                            "invalid attention stage: " + key)
            self._block_tensors[key] = (dtype, list(host.shape), host.numpy().tobytes())

        def cache(stage, length, addresses):
            # Actual producer layout is [block,head,page,2*D], with K/V
            # interleaved within each head. Gather only initialized logical rows.
            kv = attn.kv_cache
            headers.require(kv.dtype == torch.uint8 and kv.ndim == 4 and
                            tuple(kv.shape[1:]) == (4, 1600, 512), "unexpected actual FP8 cache")
            selected = addresses[:length]
            bi = torch.tensor([a[0] for a in selected], device=kv.device, dtype=torch.int64)
            pi = torch.tensor([a[1] for a in selected], device=kv.device, dtype=torch.int64)
            rows = kv.transpose(1, 2)[bi, pi]
            save("key_bytes_" + stage, rows[..., :256])
            save("value_bytes_" + stage, rows[..., 256:])

        def begin(module, args, kwargs):
            if selected_step >= 0 and step_counter() != selected_step:
                return
            metadata = get_forward_context().attn_metadata
            if not metadata or not getattr(runner.req_states, "req_id_to_index", {}):
                return
            phase, expected_rows, expected_seq, first_position = attention_capture_step(
                selected_step, len(self._block_records))
            meta = metadata[attn.layer_name]
            values = dict(zip(("positions", "hidden_states", "residual"), args)) | kwargs
            positions = values["positions"]
            host_positions = positions.detach().cpu().tolist()
            rows = int(meta.num_actual_tokens)
            headers.require(rows == expected_rows and
                            host_positions == [list(range(first_position, first_position + rows))] * 3,
                            "unexpected attention positions/step shape")
            seq = int(meta.seq_lens[:1].detach().cpu().item())
            headers.require(seq == expected_seq and
                            meta.seq_lens.numel() == 1 and tuple(meta.query_start_loc.shape) == (2,) and
                            meta.query_start_loc.detach().cpu().tolist() == [0, rows],
                            "unexpected attention request metadata")
            kv = attn.kv_cache
            blocks = meta.block_table[0, :(seq + 1599) // 1600].detach().cpu().tolist()
            slots = meta.slot_mapping[:rows].detach().cpu().tolist()
            addresses = active_cache_addresses(1600, int(kv.shape[0]), seq, blocks, slots, host_positions[0],
                                               max_seq_len=157 if selected_step == 29 else 129)
            self._block_current = phase
            self._attention_positions = positions[0]
            self._attention_addresses = addresses
            self._block_records.append({"phase": phase, "metadata": active_metadata(meta),
                                        "cache_layout": tensor_layout(kv), "logical_addresses": addresses,
                                        "scales": {a: initialized_values(getattr(attn, a), 1)
                                                   for a in ("_k_scale", "_v_scale")},
                                        "initial_cache_consumed": phase != "p128"})
            save("positions", positions)
            save("slot_mapping", meta.slot_mapping[:rows])
            save("block_table", meta.block_table[:1, :len(blocks)])
            save("seq_lens", meta.seq_lens[:1])
            save("query_start_loc", meta.query_start_loc)
            save("hidden_in", values["hidden_states"])
            if values.get("residual") is not None:
                save("residual_in", values["residual"])
            if phase != "p128":
                cache("before", seq - rows, addresses)

        def end(module, args, kwargs, result):
            if self._block_current is not None:
                save("hidden_out", result[0]); save("residual_out", result[1])
                print("ATTENTION_CAPTURE_STEP", self._block_current, flush=True)
                self._block_current = None

        self._block_hooks.append(layer.register_forward_pre_hook(begin, with_kwargs=True))
        self._block_hooks.append(layer.register_forward_hook(end, with_kwargs=True))
        for label, module in (("input_norm", layer.input_layernorm), ("qkv", mixer.qkv_proj),
                              ("q_norm", mixer.q_norm), ("k_norm", mixer.k_norm),
                              ("o_proj", mixer.o_proj), ("mixer", mixer)):
            def after(module, args, kwargs, result, label=label):
                if self._block_current is not None:
                    save(label + "_output", result[0] if isinstance(result, tuple) else result)
            self._block_hooks.append(module.register_forward_hook(after, with_kwargs=True))

        def projected(result):
            if self._block_current is not None:
                for label, value in zip(("q_rope", "k_rope", "value", "gate"), result):
                    save(label, value)
        self._attention_mixer = mixer
        self._attention_original_projection = mixer._project_qkv_gate
        mixer._project_qkv_gate = observe_projection(self._attention_original_projection, projected)

        def rope_inputs(module, args, kwargs):
            if self._block_current is not None:
                values = dict(zip(("positions", "query", "key", "offsets"), args)) | kwargs
                save("rope_q_input", values["query"])
                save("rope_k_input", values["key"])
        self._block_hooks.append(mixer.rotary_emb.register_forward_pre_hook(rope_inputs, with_kwargs=True))
        self._attention_original_cos_cache = mixer.rotary_emb._match_cos_sin_cache_dtype
        def cos_cache(result):
            if self._block_current is not None:
                # Observe the cache actually selected by original forward_native,
                # including its dtype, rather than reconstructing coefficients.
                save("rope_cos_sin", result[self._attention_positions])
        mixer.rotary_emb._match_cos_sin_cache_dtype = observe_projection(
            self._attention_original_cos_cache, cos_cache)

        def attention_done(module, args, kwargs, result):
            if self._block_current is not None:
                save("attention_output", result)
                cache("after", len(self._attention_addresses), self._attention_addresses)
        self._block_hooks.append(attn.register_forward_hook(attention_done, with_kwargs=True))
        def gated(module, args):
            if self._block_current is not None:
                save("gated_attention", args[0])
        self._block_hooks.append(mixer.o_proj.register_forward_pre_hook(gated))
        return {"layer": name, "observer": "original module hooks and single-delegation projection observer",
                "use_fused_qk_norm_rope_gate": mixer.use_fused_qk_norm_rope_gate}

    def finish_block_capture(self):
        self._attention_mixer._project_qkv_gate = self._attention_original_projection
        self._attention_mixer.rotary_emb._match_cos_sin_cache_dtype = self._attention_original_cos_cache
        if self._attention_selected_step == 29:
            import torch
            torch.xpu.synchronize()
            for hook in self._block_hooks:
                hook.remove()
            headers.require([r["phase"] for r in self._block_records] == ["d29"], "missing selected attention")
            for label in ("key", "value"):
                before = self._block_tensors[f"d29_{label}_bytes_before"]
                after = self._block_tensors[f"d29_{label}_bytes_after"]
                headers.require(before[1] == [156, 4, 256] and after[1] == [157, 4, 256] and
                                after[2][:len(before[2])] == before[2], "D29 overwrote earlier initialized KV")
            write_safetensors(self._block_output, self._block_tensors, {"image": IMAGE, "selected_step": "29"})
            return {"path": str(self._block_output), "steps": self._block_records,
                    "capture_sha256": digest(self._block_output.read_bytes()),
                    "tensor_hashes": {n: {"dtype": t[0], "shape": t[1], "sha256": digest(t[2])}
                                      for n, t in sorted(self._block_tensors.items())},
                    "compilation_config": describe(self.model_runner.compilation_config),
                    "previous_initialized_cache_unchanged": True}
        result = super().finish_block_capture()
        for label in ("key", "value"):
            p = self._block_tensors["p128_" + label + "_bytes_after"]
            d = self._block_tensors["d1_" + label + "_bytes_before"]
            headers.require(p == d, "P128/D1 cache continuity differs: " + label)
            after = self._block_tensors["d1_" + label + "_bytes_after"]
            headers.require(after[2][:len(p[2])] == p[2], "D1 overwrote existing cache rows")
        return result | {"active_cache_continuity_exact": True}


def capture(args):
    headers.require(not args.output.exists() and not args.output.with_suffix(".json").exists(),
                    "refusing to overwrite attention capture")
    reference = verify_inputs(args.reference_manifest, args.model_dir, args.image_identity)
    import yaml
    profile = yaml.safe_load(PROFILE.read_text())
    for key, value in profile["env"].items():
        if value is not None:
            os.environ[key] = str(value).replace("{model_dir}", str(PROFILE.parent))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["EXL3_LOADER_REPORT_DIR"] = str(args.output.parent / "attention-loader")
    from vllm import LLM, SamplingParams
    from vllm.config import ReasoningConfig
    from vllm.engine.arg_utils import EngineArgs
    import torch
    headers.require(torch.__version__ == "2.13.0+xpu" and "B70" in torch.xpu.get_device_name(0),
                    "requires pinned Torch XPU and B70")
    allowed = {f.name for f in dataclasses.fields(EngineArgs)}
    kwargs = {k: v for k, v in profile["vllm"].items() if k in allowed}
    if isinstance(kwargs.get("reasoning_config"), dict):
        kwargs["reasoning_config"] = ReasoningConfig(**kwargs["reasoning_config"])
    overrides = {"enforce_eager": True, "speculative_config": None,
                 "compilation_config": {"mode": 0, "cudagraph_mode": "NONE", "custom_ops": ["none"]}}
    kwargs.update(overrides, model=str(args.model_dir), worker_extension_cls="capture_attention.AttentionCapture")
    tokens = [1000 + (i * 37) % 4096 for i in range(128)]
    params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
    llm = None
    try:
        llm = LLM(**kwargs)
        ordinary = llm.generate({"prompt_token_ids": tokens}, params, use_tqdm=False)
        headers.require(llm.reset_prefix_cache(), "prefix reset failed")
        installed = llm.collective_rpc("install_block_capture", timeout=60, args=(str(args.output),))
        observed = llm.generate({"prompt_token_ids": tokens}, params, use_tqdm=False)
        ordinary_ids = [list(o.outputs[0].token_ids) for o in ordinary]
        observed_ids = [list(o.outputs[0].token_ids) for o in observed]
        headers.require(ordinary_ids == observed_ids, "observation changed greedy output IDs")
        records = llm.collective_rpc("finish_block_capture", timeout=60)
        headers.require(len(records) == 1, "requires one worker")
        result = records[0] | {"schema": 1, "kind": "pinned_layer3_eager_P128_D1_attention",
                              "image": IMAGE, "checkpoint": reference["checkpoint"]["identity"],
                              "profile_sha256": digest(PROFILE.read_bytes()), "s1_overrides": overrides,
                              "capture_tool_sha256": digest(Path(__file__).read_bytes()), "observer": installed,
                              "prompt_token_ids": tokens, "ordinary_output_ids": ordinary_ids,
                              "observed_output_ids": observed_ids,
                              "scope": "Original layer3 attention stages and written FP8 bytes only; no native parity or performance claim."}
        with args.output.with_suffix(".json").open("x") as stream:
            json.dump(result, stream, indent=2); stream.write("\n")
        print("ATTENTION_CAPTURE_DONE", args.output, result["capture_sha256"], flush=True)
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    capture(parser.parse_args())
