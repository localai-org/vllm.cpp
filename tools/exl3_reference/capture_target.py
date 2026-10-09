#!/usr/bin/env python3
"""Bounded cold eager P128/D1 or P128/D64 original-target captures.

Reference reproducibility evidence only. No changed arithmetic or native pass.
"""
import argparse
import dataclasses
import functools
import heapq
import json
import math
from pathlib import Path
import os
import struct

from capture_block import BlockCapture
from capture_projection import IMAGE
from capture_runtime_layout import PROFILE, verify_inputs, active_metadata
from compare_projection import blob, metrics
from extract_projection import digest, headers
from runtime_layout import tensor_layout


def observe_logits(original, save):
    """Delegate exactly once, preserving arguments and the result object."""
    @functools.wraps(original)
    def observed(hidden_states, *args, **kwargs):
        result = original(hidden_states, *args, **kwargs)
        save(hidden_states, result)
        return result
    return observed


def deterministic_ba_forward(original, read_setting, write_setting):
    """Scope the original dense matmul policy; delegate once and restore on failure."""
    @functools.wraps(original)
    def forward(*args, **kwargs):
        previous = read_setting()
        write_setting((True, False))
        try:
            return original(*args, **kwargs)
        finally:
            write_setting(previous)
    return forward


def target_phases(decode_steps):
    headers.require(decode_steps in (1, 29, 64), "only focused D1/D29/D64 captures are supported")
    return ["p128"] + [f"d{i}" for i in range(1, decode_steps + 1)]


def selected_detail_kind(block_step, detail_layer):
    headers.require(type(detail_layer) is int and detail_layer in (-1, 1, 3, 21) and
                    (detail_layer == -1 or block_step == 29),
                    "detailed observation supports D29 GDN1/GDN21 or attention3 only")
    return {-1: "none", 1: "gdn", 3: "attention", 21: "gdn"}[detail_layer]


def validate_gdn_history(block_step, detail_layer, enabled):
    headers.require(type(enabled) is bool and
                    (not enabled or (block_step == 29 and detail_layer == 21)),
                    "early GDN history is bounded to the selected D29 GDN21 capture")


def selected_boundary_phase(step, index, block_step, gdn_history):
    if step == block_step:
        return "d29"
    if gdn_history and step == 0 and 0 <= index <= 21:
        return "p128"
    return None


def selected_block_layers(modules, first_layer):
    prefix = first_layer.rsplit(".", 1)[0] + "."
    layers = [(int(name[len(prefix):]), module) for name, module in modules.items()
              if name.startswith(prefix) and name[len(prefix):].isdigit()]
    headers.require(sorted(index for index, _ in layers) == list(range(64)),
                    "selected-step observation requires all 64 target blocks")
    return sorted(layers)


def load_trace(path, decode_steps):
    record = json.loads(path.read_text())
    ids = record["output_ids"]
    headers.require(len(ids) == 1 and len(ids[0]) == decode_steps + 1 and
                    all(type(t) is int and 0 <= t < 248320 for t in ids[0]),
                    "trace must contain one complete bounded target token sequence")
    return ids[0]


def load_prompt(path):
    """Keep the legacy input; optional held-out witnesses are exactly P128."""
    if path is None:
        return [1000 + (i * 37) % 4096 for i in range(128)]
    ids = headers.read_json(path)["prompt_token_ids"]
    headers.require(isinstance(ids, list) and len(ids) == 128 and
                    all(type(t) is int and 0 <= t < 248320 for t in ids),
                    "held-out capture requires exactly128 valid target token IDs")
    return ids


def validate_all_gdn_states(decode_steps, block_step, detail_layer, gdn_history, enabled):
    headers.require(type(enabled) is bool and
                    (not enabled or (decode_steps == 1 and block_step == -1 and
                                     detail_layer == -1 and not gdn_history)),
                    "all-GDN-state observation supports standalone P128/D1 only")


def validate_all_attention_kv(all_gdn_states, enabled):
    headers.require(type(enabled) is bool and (not enabled or all_gdn_states is True),
                    "all-attention KV observation requires standalone all-GDN-state capture")


def all_target_gdn_layers(modules, first_layer):
    layers = selected_block_layers(modules, first_layer)
    expected = [i for i in range(64) if i % 4 != 3]
    selected = [(i, block) for i, block in layers if block.layer_type == "linear_attention"]
    headers.require([i for i, _ in selected] == expected and
                    all(block.layer_type == "full_attention" for i, block in layers if i % 4 == 3),
                    "all-state capture requires actual48GDN/16attention target blocks")
    return selected


def validate_prefix_witnesses(witnesses, prompt, outputs):
    phases = ["p128"] + [f"d{i}" for i in range(1, len(outputs))]
    headers.require(list(witnesses) == phases, "missing or reordered actual input witnesses")
    for step, phase in enumerate(phases):
        ids = prompt if step == 0 else outputs[step - 1:step]
        positions = list(range(len(prompt))) if step == 0 else [len(prompt) + step - 1]
        headers.require(witnesses[phase]["token_ids"] == ids and
                        witnesses[phase]["positions"] == [positions] * 3,
                        "actual input tokens/positions differ from recorded prefix: " + phase)


def probability_metrics(actual, expected):
    headers.require(len(actual) == len(expected) and len(actual) > 0,
                    "logit lengths must match and be nonempty")
    headers.require(all(math.isfinite(x) for row in (actual, expected) for x in row),
                    "nonfinite logits")
    def log_probs(row):
        maximum = max(row)
        normalizer = math.log(math.fsum(math.exp(x - maximum) for x in row))
        return [x - maximum - normalizer for x in row]
    a, e = log_probs(actual), log_probs(expected)
    pa, pe = [math.exp(x) for x in a], [math.exp(x) for x in e]
    top = min(10, len(actual))
    ai = set(heapq.nlargest(top, range(len(actual)), key=lambda i: actual[i]))
    ei = set(heapq.nlargest(top, range(len(expected)), key=lambda i: expected[i]))
    tv = 0.5 * math.fsum(abs(x - y) for x, y in zip(pa, pe))
    kl = math.fsum(p * (x - y) for p, x, y in zip(pe, e, a))
    overlap = len(ai & ei)
    return {"TV": tv, "KL_reference_to_actual": kl, "top10_overlap": overlap,
            "investigation_trigger": tv > 0.02 or kl > 0.002 or overlap < min(9, top)}


class TargetCapture(BlockCapture):
    def install_deterministic_ba(self):
        import torch
        headers.require(not hasattr(self, "_deterministic_ba_originals"), "BA policy already installed")
        modules = [(name, module) for name, module in self.model_runner.model.named_modules()
                   if name.endswith("linear_attn.in_proj_ba")]
        headers.require(len(modules) == 48 and all(module.weight.dtype == torch.float16 and
                        module.weight.shape == (96, 5120) and
                        type(module.quant_method).__name__ == "UnquantizedLinearMethod"
                        for _, module in modules), "requires all48 actual unquantized target BA modules")
        def read():
            return (torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled())
        def write(setting):
            torch.use_deterministic_algorithms(setting[0], warn_only=setting[1])
        self._deterministic_ba_originals = [(module, module.forward) for _, module in modules]
        record = []
        for name, module in modules:
            record.append({"name": name, "weight_layout": tensor_layout(module.weight),
                           "quant_method": type(module.quant_method).__name__})
            module.forward = deterministic_ba_forward(module.forward, read, write)
        return {"kind": "diagnostic_original_deterministic_BA_only", "modules": record,
                "scope": "Original forwards/operands unchanged; deterministic policy only inside each BA forward, restored before return. Not a replacement of frozen qualification."}

    def restore_deterministic_ba(self):
        for module, original in self._deterministic_ba_originals:
            module.forward = original
        del self._deterministic_ba_originals
        return {"restored_modules": 48}

    def install_target_capture(self, output, decode_steps=1, block_step=-1, detail_layer=-1,
                               gdn_history=False, all_gdn_states=False, all_attention_kv=False):
        import torch
        headers.require(block_step == -1 or (decode_steps == 29 and block_step == 29),
                        "selected block observation is bounded to D29")
        detail_kind = selected_detail_kind(block_step, detail_layer)
        validate_gdn_history(block_step, detail_layer, gdn_history)
        validate_all_gdn_states(decode_steps, block_step, detail_layer, gdn_history,
                                all_gdn_states)
        validate_all_attention_kv(all_gdn_states, all_attention_kv)
        installed = self.install_block_capture(output)
        self._gdn_history_capture = None
        if gdn_history:
            # Reuse the bounded P128/D1 observer. It skips uninitialized,
            # unconsumed cold-prefill state and delegates every original op.
            observer = BlockCapture()
            observer.model_runner = self.model_runner
            path = Path(output)
            observer.install_block_capture(
                path.with_name(path.stem + "-gdn21-history.safetensors"), 21)
            self._gdn_history_capture = observer
        self._target_phases = []
        self._target_expected_phases = target_phases(decode_steps)
        self._target_layouts = []
        self._target_witnesses, self._target_hooks = {}, []
        model = self.model_runner.model
        modules = dict(model.named_modules())
        layer = modules[installed["layer"]]
        self._all_gdn_states = all_gdn_states
        self._all_gdn_state_records = {}
        if all_gdn_states:
            from vllm.forward_context import get_forward_context

            def save_state(index, mixer, stage):
                context = get_forward_context().attn_metadata
                if not context or not getattr(self.model_runner.req_states, "req_id_to_index", {}):
                    return
                step = len(self._target_phases)
                headers.require(step in (0, 1), "extra all-state target step")
                meta = context[mixer.prefix]
                phase, rows = ("p128", 128) if step == 0 else ("d1", 1)
                headers.require(meta.num_actual_tokens == rows and bool(meta.num_prefills) == (step == 0),
                                "unexpected all-state phase or active rows")
                key = f"{phase}_l{index}_{stage}"
                headers.require(key not in self._all_gdn_state_records, "duplicate all-state observation")
                indices = meta.prefill_state_indices if step == 0 else meta.non_spec_state_indices_tensor
                headers.require(indices.numel() == 1, "all-state requires one active slot")
                slot = int(indices.detach().cpu().item())
                cache = mixer.kv_cache
                headers.require(len(cache) == 2 and 0 <= slot < cache[0].shape[0] and
                                slot < cache[1].shape[0], "invalid all-state cache slot")
                initial = bool(meta.prefill_has_initial_state.detach().cpu().item()) if step == 0 else True
                record = {"phase": phase, "layer": index, "stage": stage,
                          "active_slot": slot, "initial_state_consumed": initial}
                # Never read the unspecified seed of a cold prefill.
                if stage == "before" and not initial:
                    record["copied"] = False
                else:
                    for label, value, dtype, shape in (
                        ("conv", cache[0][slot], torch.float16, (3, 10240)),
                        ("ssm", cache[1][slot], torch.float32, (48, 128, 128))):
                        headers.require(value.dtype == dtype and tuple(value.shape) == shape,
                                        "unexpected active GDN state layout")
                        host = value.detach().cpu().contiguous()
                        headers.require(torch.isfinite(host).all().item(), "nonfinite active GDN state")
                        tensor_key = key + "_" + label
                        headers.require(tensor_key not in self._block_tensors, "duplicate active state tensor")
                        self._block_tensors[tensor_key] = (
                            "F16" if dtype == torch.float16 else "F32", list(shape), host.numpy().tobytes())
                        record[label + "_layout"] = tensor_layout(value)
                    record["copied"] = True
                self._all_gdn_state_records[key] = record

            for index, block in all_target_gdn_layers(modules, installed["layer"]):
                mixer = block.linear_attn
                self._target_hooks.append(block.register_forward_pre_hook(
                    lambda module, args, kwargs, i=index, m=mixer: save_state(i, m, "before"),
                    with_kwargs=True))
                self._target_hooks.append(block.register_forward_hook(
                    lambda module, args, kwargs, result, i=index, m=mixer: save_state(i, m, "after"),
                    with_kwargs=True))
        self._all_attention_kv = all_attention_kv
        self._all_attention_kv_records = {}
        self._kv_preimage_originals = []
        if all_attention_kv:
            from capture_attention import active_cache_addresses, observe_projection
            from runtime_layout import initialized_values

            # Causal probe for the observed layer43 signed-zero key mismatch.
            # Observe actual operands/results; do not recompute or replace them.
            probe_block = modules[installed["layer"].rsplit(".", 1)[0] + ".43"]
            headers.require(probe_block.layer_type == "full_attention", "missing KV preimage layer43")
            probe_mixer = probe_block.self_attn
            self._kv_preimage_active = False

            def begin_preimage(module, args, kwargs):
                self._kv_preimage_active = True

            def end_preimage(module, args, kwargs, result):
                self._kv_preimage_active = False

            self._target_hooks.append(probe_block.register_forward_pre_hook(begin_preimage, with_kwargs=True))
            self._target_hooks.append(probe_block.register_forward_hook(end_preimage, with_kwargs=True))

            def save_preimage(label, value):
                context = get_forward_context().attn_metadata
                if not self._kv_preimage_active or not context or not getattr(
                        self.model_runner.req_states, "req_id_to_index", {}):
                    return
                step = len(self._target_phases)
                headers.require(step in (0, 1), "extra KV preimage step")
                phase = "p128" if step == 0 else "d1"
                key = f"{phase}_l43_preimage_{label}"
                headers.require(key not in self._block_tensors and value.numel() <= 5_000_000 and
                                value.dtype == torch.float16, "duplicate/unbounded KV preimage: " + key)
                host = value.detach().cpu().contiguous()
                headers.require(torch.isfinite(host).all().item(), "nonfinite KV preimage")
                self._block_tensors[key] = ("F16", list(host.shape), host.numpy().tobytes())

            for label, module in (("input_norm_output", probe_block.input_layernorm),
                                  ("qkv_output", probe_mixer.qkv_proj),
                                  ("k_norm_output", probe_mixer.k_norm)):
                self._target_hooks.append(module.register_forward_hook(
                    lambda module, args, kwargs, result, label=label: save_preimage(
                        label, result[0] if isinstance(result, tuple) else result), with_kwargs=True))
            self._target_hooks.append(probe_mixer.k_norm.register_forward_pre_hook(
                lambda module, args, kwargs: save_preimage("k_norm_input", args[0] if args else kwargs["x"]),
                with_kwargs=True))

            def rope_preimage(module, args, kwargs):
                context = get_forward_context().attn_metadata
                if not self._kv_preimage_active or not context or not getattr(
                        self.model_runner.req_states, "req_id_to_index", {}):
                    return
                values = dict(zip(("positions", "query", "key", "offsets"), args)) | kwargs
                self._kv_preimage_positions = values["positions"][0]
                save_preimage("rope_k_input", values["key"])
            self._target_hooks.append(probe_mixer.rotary_emb.register_forward_pre_hook(
                rope_preimage, with_kwargs=True))
            for module, name, save_result in (
                    (probe_mixer, "_project_qkv_gate", lambda result: save_preimage("k_rope", result[1])),
                    (probe_mixer.rotary_emb, "_match_cos_sin_cache_dtype",
                     lambda result: save_preimage("rope_cos_sin", result[self._kv_preimage_positions])
                     if self._kv_preimage_active else None)):
                original = getattr(module, name)
                self._kv_preimage_originals.append((module, name, original))
                setattr(module, name, observe_projection(original, save_result))

            def save_kv(index, attn, args, kwargs, stage):
                context = get_forward_context().attn_metadata
                if not context or not getattr(self.model_runner.req_states, "req_id_to_index", {}):
                    return
                step = len(self._target_phases)
                headers.require(step in (0, 1), "extra all-KV target step")
                phase, rows, seq = ("p128", 128, 128) if step == 0 else ("d1", 1, 129)
                meta = context[attn.layer_name]
                values = dict(zip(("positions", "hidden_states", "residual"), args)) | kwargs
                positions = values["positions"].detach().cpu().tolist()
                headers.require(meta.num_actual_tokens == rows and
                                positions == [list(range(seq - rows, seq))] * 3 and
                                meta.seq_lens.numel() == 1 and int(meta.seq_lens[:1].cpu().item()) == seq and
                                tuple(meta.query_start_loc.shape) == (2,) and
                                meta.query_start_loc.cpu().tolist() == [0, rows],
                                "unexpected active all-KV positions/metadata")
                kv = attn.kv_cache
                headers.require(kv.dtype == torch.uint8 and kv.ndim == 4 and
                                kv.shape[1] == 4 and kv.shape[3] == 512 and kv.shape[2] > 0,
                                "unexpected original FP8 KV layout")
                page = int(kv.shape[2])
                blocks = meta.block_table[0, :(seq + page - 1) // page].cpu().tolist()
                slots = meta.slot_mapping[:rows].cpu().tolist()
                addresses = active_cache_addresses(page, int(kv.shape[0]), seq, blocks, slots, positions[0])
                length = seq if stage == "after" else seq - rows
                key = f"{phase}_l{index}_{stage}"
                headers.require(key not in self._all_attention_kv_records, "duplicate all-KV boundary")
                record = {"phase": phase, "layer": index, "stage": stage,
                          "written_logical_length": length, "copied": length > 0,
                          "page_size": page, "logical_addresses": addresses[:length],
                          "cache_layout": tensor_layout(kv),
                          "scales": {name: initialized_values(getattr(attn, name), 1)
                                     for name in ("_k_scale", "_v_scale")}}
                if length:
                    selected = addresses[:length]
                    bi = torch.tensor([a[0] for a in selected], dtype=torch.int64, device=kv.device)
                    pi = torch.tensor([a[1] for a in selected], dtype=torch.int64, device=kv.device)
                    active = kv.transpose(1, 2)[bi, pi]
                    for label, value in (("key", active[..., :256]), ("value", active[..., 256:])):
                        host = value.detach().cpu().contiguous()
                        headers.require(tuple(host.shape) == (length, 4, 256), "unbounded active KV copy")
                        tensor_key = key + "_" + label
                        headers.require(tensor_key not in self._block_tensors, "duplicate active KV tensor")
                        self._block_tensors[tensor_key] = ("U8", list(host.shape), host.numpy().tobytes())
                self._all_attention_kv_records[key] = record

            for index, block in selected_block_layers(modules, installed["layer"]):
                if index % 4 != 3:
                    continue
                attn = block.self_attn.attn
                self._target_hooks.append(block.register_forward_pre_hook(
                    lambda module, args, kwargs, i=index, a=attn: save_kv(i, a, args, kwargs, "before"),
                    with_kwargs=True))
                self._target_hooks.append(block.register_forward_hook(
                    lambda module, args, kwargs, result, i=index, a=attn: save_kv(i, a, args, kwargs, "after"),
                    with_kwargs=True))
        self._selected_block_hooks = []
        self._selected_block_counts = {}
        if block_step >= 0:
            def save_boundary(key, value, rows):
                headers.require(key not in self._block_tensors and
                                value.dtype == torch.float16 and value.shape == (rows, 5120),
                                "duplicate or unbounded selected block boundary: " + key)
                host = value.detach().cpu().contiguous()
                headers.require(torch.isfinite(host).all().item(), "nonfinite selected block boundary")
                self._block_tensors[key] = ("F16", [rows, 5120], host.numpy().tobytes())
            for index, block in selected_block_layers(modules, installed["layer"]):
                def before(module, args, kwargs, index=index):
                    phase = selected_boundary_phase(len(self._target_phases), index, block_step, gdn_history)
                    if phase is None:
                        return
                    rows = 128 if phase == "p128" else 1
                    values = dict(zip(("positions", "hidden_states", "residual"), args)) | kwargs
                    save_boundary(f"{phase}_l{index}_hidden_in", values["hidden_states"], rows)
                    residual = values.get("residual")
                    if residual is not None:
                        save_boundary(f"{phase}_l{index}_residual_in", residual, rows)
                def after(module, args, kwargs, output, index=index):
                    phase = selected_boundary_phase(len(self._target_phases), index, block_step, gdn_history)
                    if phase is None:
                        return
                    rows = 128 if phase == "p128" else 1
                    headers.require(isinstance(output, tuple) and len(output) == 2,
                                    "unexpected target block result")
                    save_boundary(f"{phase}_l{index}_hidden_out", output[0], rows)
                    save_boundary(f"{phase}_l{index}_residual_out", output[1], rows)
                    if phase == "d29":
                        self._selected_block_counts[index] = self._selected_block_counts.get(index, 0) + 1
                self._selected_block_hooks.append(block.register_forward_pre_hook(before, with_kwargs=True))
                self._selected_block_hooks.append(block.register_forward_hook(after, with_kwargs=True))
                for label, norm in (("post_input_norm", block.input_layernorm),
                                    ("post_attn_norm", block.post_attention_layernorm)):
                    def norm_after(module, args, kwargs, output, index=index, label=label):
                        phase = selected_boundary_phase(len(self._target_phases), index, block_step, gdn_history)
                        if phase is not None:
                            value = output[0] if isinstance(output, tuple) else output
                            save_boundary(f"{phase}_l{index}_{label}", value, 128 if phase == "p128" else 1)
                    self._selected_block_hooks.append(norm.register_forward_hook(norm_after, with_kwargs=True))
        if gdn_history:
            # The first preceding P128 divergence is block20's MLP. Observe
            # its three real module outputs on that forward only.
            mlp = modules[installed["layer"].rsplit(".", 1)[0] + ".20"].mlp
            for label, module, width in (("gate_up", mlp.gate_up_proj, 34816),
                                         ("swiglu", mlp.act_fn, 17408),
                                         ("down", mlp.down_proj, 5120)):
                def mlp_after(module, args, kwargs, output, label=label, width=width):
                    if self._target_phases:
                        return
                    value = output[0] if isinstance(output, tuple) else output
                    key = "p128_l20_detail_" + label
                    headers.require(key not in self._block_tensors and
                                    value.dtype == torch.float16 and value.shape == (128, width),
                                    "duplicate/unbounded P128 MLP20 boundary")
                    host = value.detach().cpu().contiguous()
                    headers.require(torch.isfinite(host).all().item(), "nonfinite P128 MLP20 boundary")
                    self._block_tensors[key] = ("F16", [128, width], host.numpy().tobytes())
                self._selected_block_hooks.append(module.register_forward_hook(mlp_after, with_kwargs=True))
        self._selected_block_step = block_step
        self._selected_detail_record = None
        self._selected_attention_capture = None
        if detail_kind == "attention":
            from capture_attention import AttentionCapture
            observer = AttentionCapture()
            observer.model_runner = self.model_runner
            path = Path(output)
            observer.install_block_capture(path.with_name(path.stem + "-attention3.safetensors"),
                                           selected_step=29, step_counter=lambda: len(self._target_phases))
            self._selected_attention_capture = observer
        if detail_kind == "gdn":
            from vllm.forward_context import get_forward_context
            block = modules[installed["layer"].rsplit(".", 1)[0] + f".{detail_layer}"]
            headers.require(block.layer_type == "linear_attention", "selected layer must be GDN")
            mixer = block.linear_attn
            def save_detail(label, value):
                key = f"d29_l{detail_layer}_detail_" + label
                headers.require(key not in self._block_tensors and value.numel() <= 1_000_000,
                                "duplicate or unbounded D29 selected GDN detail")
                host = value.detach().cpu().contiguous()
                dtype = {torch.float16: "F16", torch.float32: "F32"}.get(host.dtype)
                headers.require(dtype is not None and torch.isfinite(host).all().item(),
                                "invalid D29 selected GDN detail")
                self._block_tensors[key] = (dtype, list(host.shape), host.numpy().tobytes())
            def mixer_before(module, args, kwargs):
                if len(self._target_phases) != block_step:
                    return
                meta = get_forward_context().attn_metadata[mixer.prefix]
                indices = meta.non_spec_state_indices_tensor
                headers.require(meta.num_actual_tokens == 1 and not meta.num_prefills and
                                indices.numel() == 1, "requires one actual D29 decode state")
                slot = int(indices.detach().cpu().item())
                headers.require(0 <= slot < mixer.kv_cache[0].shape[0] and
                                self._selected_detail_record is None, "invalid or repeated active D29 slot")
                self._selected_detail_record = {"layer": detail_layer, "step": 29, "slot": slot,
                    "metadata": active_metadata(meta),
                    "conv_layout": tensor_layout(mixer.kv_cache[0]),
                    "ssm_layout": tensor_layout(mixer.kv_cache[1])}
                save_detail("conv_before", mixer.kv_cache[0][slot])
                save_detail("ssm_before", mixer.kv_cache[1][slot])
            def mixer_after(module, args, kwargs, output):
                if len(self._target_phases) != block_step:
                    return
                slot = self._selected_detail_record["slot"]
                save_detail("mixer_output", output[0] if isinstance(output, tuple) else output)
                save_detail("conv_after", mixer.kv_cache[0][slot])
                save_detail("ssm_after", mixer.kv_cache[1][slot])
            self._selected_block_hooks.append(mixer.register_forward_pre_hook(mixer_before, with_kwargs=True))
            self._selected_block_hooks.append(mixer.register_forward_hook(mixer_after, with_kwargs=True))
            for label, module in (("qkvz", mixer.in_proj_qkvz), ("ba", mixer.in_proj_ba),
                                  ("gated_norm", mixer.norm), ("out_proj", mixer.out_proj)):
                def detail_after(module, args, kwargs, output, label=label):
                    if len(self._target_phases) == block_step:
                        save_detail(label + "_output", output[0] if isinstance(output, tuple) else output)
                self._selected_block_hooks.append(module.register_forward_hook(detail_after, with_kwargs=True))
        embedding = modules[installed["layer"].rsplit("layers.", 1)[0] + "embed_tokens"]
        def witness(module, args, kwargs, tokens=False):
            step = len(self._target_phases)
            headers.require(step < len(self._target_expected_phases), "extra model input witness")
            phase = self._target_expected_phases[step]
            values = dict(zip(("input_ids",) if tokens else ("positions",), args)) | kwargs
            value = values["input_ids" if tokens else "positions"]
            headers.require(value.dtype in (torch.int32, torch.int64) and
                            value.numel() <= (128 if tokens else 384), "unbounded actual input witness")
            row = self._target_witnesses.setdefault(phase, {})
            key = "token_ids" if tokens else "positions"
            headers.require(key not in row, "duplicate actual input witness")
            row[key] = value.detach().cpu().tolist()
        self._target_hooks.append(embedding.register_forward_pre_hook(
            lambda module, args, kwargs: witness(module, args, kwargs, True), with_kwargs=True))
        self._target_hooks.append(layer.register_forward_pre_hook(witness, with_kwargs=True))
        self._target_original_logits = model.compute_logits
        def save(hidden, logits):
            step = len(self._target_phases)
            headers.require(step < len(self._target_expected_phases), "unexpected extra target-head step")
            phase = self._target_expected_phases[step]
            if step < 2:
                headers.require(self._block_records[-1]["phase"] == phase, "unexpected block/head order")
            headers.require(hidden.ndim == logits.ndim == 2 and
                            hidden.shape == (1, 5120) and logits.shape == (1, 248320),
                            "requires gathered final prompt row and full target vocabulary")
            self._target_layouts.append({"phase": phase, "hidden": tensor_layout(hidden),
                                         "logits": tensor_layout(logits)})
            for label, value in (("logits_hidden", hidden), ("logits", logits)):
                host = value.detach().cpu().contiguous()
                dtype = {torch.float16: "F16", torch.float32: "F32"}.get(host.dtype)
                headers.require(dtype is not None and torch.isfinite(host).all().item(),
                                "invalid target-head boundary")
                self._block_tensors[phase + "_" + label] = (
                    dtype, list(host.shape), host.numpy().tobytes())
            self._target_phases.append(phase)
            if len(self._target_phases) == 2:
                # The existing bounded block observer covers only P128/D1.
                # Later decode steps retain head observations, not block states.
                for hook in self._block_hooks:
                    hook.remove()
                self._block_hooks.clear()
                if self._gdn_history_capture is not None:
                    for hook in self._gdn_history_capture._block_hooks:
                        hook.remove()
                    self._gdn_history_capture._block_hooks.clear()
        model.compute_logits = observe_logits(self._target_original_logits, save)
        return installed | {"target_observer": "compute_logits delegates original once; bounded active copies"}

    def finish_target_capture(self):
        history_record = (self._gdn_history_capture.finish_block_capture()
                          if self._gdn_history_capture is not None else None)
        attention = self._selected_attention_capture
        attention_record = attention.finish_block_capture() if attention is not None else None
        self.model_runner.model.compute_logits = self._target_original_logits
        for hook in self._target_hooks:
            hook.remove()
        for hook in self._selected_block_hooks:
            hook.remove()
        for module, name, original in self._kv_preimage_originals:
            setattr(module, name, original)
        if self._selected_block_step >= 0:
            headers.require(self._selected_block_counts == dict.fromkeys(range(64), 1),
                            "missing, repeated or reordered D29 block observations")
        headers.require(self._target_phases == self._target_expected_phases, "missing full-target head calls")
        if self._all_gdn_states:
            expected = {f"{phase}_l{i}_{stage}" for phase in ("p128", "d1")
                        for i in range(64) if i % 4 != 3 for stage in ("before", "after")}
            headers.require(set(self._all_gdn_state_records) == expected,
                            "missing all-GDN-state boundaries")
        if self._all_attention_kv:
            for phase in ("p128", "d1"):
                for stage in ("input_norm_output", "qkv_output", "k_norm_input", "k_norm_output",
                              "rope_k_input", "rope_cos_sin", "k_rope"):
                    headers.require(f"{phase}_l43_preimage_{stage}" in self._block_tensors,
                                    "missing actual KV preimage boundary")
            expected = {f"{phase}_l{i}_{stage}" for phase in ("p128", "d1")
                        for i in range(3, 64, 4) for stage in ("before", "after")}
            headers.require(set(self._all_attention_kv_records) == expected,
                            "missing all-attention KV boundaries")
            for i in range(3, 64, 4):
                for label in ("key", "value"):
                    prefill = self._block_tensors[f"p128_l{i}_after_{label}"][2]
                    before = self._block_tensors[f"d1_l{i}_before_{label}"][2]
                    after = self._block_tensors[f"d1_l{i}_after_{label}"][2]
                    headers.require(prefill == before and after[:len(before)] == before,
                                    "original KV continuation overwrote initialized rows")
        return self.finish_block_capture() | {"selected_block_step": self._selected_block_step,
                                              "selected_attention_record": attention_record,
                                              "selected_detail_record": self._selected_detail_record,
                                              "gdn_history_record": history_record,
                                              "all_gdn_state_records": self._all_gdn_state_records,
                                              "all_attention_kv_records": self._all_attention_kv_records,
                                              "selected_block_counts": self._selected_block_counts,
                                              "target_layouts": self._target_layouts,
                                              "actual_input_witnesses": self._target_witnesses}


def compare_repeats(paths, decode_steps=1):
    reports = [headers.read_shard_header(p) for p in paths]
    comparisons = []
    for j in range(1, len(paths)):
        for i in range(j):
            stages = {}
            for phase in target_phases(decode_steps):
                labels = ("ba_output", "ssm_state_after", "hidden_out") if phase in ("p128", "d1") else ()
                for label in labels + ("logits_hidden", "logits"):
                    key = phase + "_" + label
                    a, raw_a = blob(paths[j], reports[j], key)
                    e, raw_e = blob(paths[i], reports[i], key)
                    headers.require(a["shape"] == e["shape"] and a["dtype"] == e["dtype"],
                                    "repeat stage contract differs")
                    result = metrics(raw_a, raw_e, a["dtype"])
                    headers.require(result["finite"], "nonfinite repeat stage")
                    if label in ("logits", "ssm_state_after"):
                        fmt = "<f" if a["dtype"] == "F32" else "<e"
                        av = [v for (v,) in struct.iter_unpack(fmt, raw_a)]
                        ev = [v for (v,) in struct.iter_unpack(fmt, raw_e)]
                        if label == "logits":
                            result["distribution"] = probability_metrics(av, ev)
                        else:
                            result["fixed_pointwise_failures"] = sum(
                                abs(x-y) > 1e-5 + 1e-4 * abs(y) for x,y in zip(av, ev))
                    stages[key] = result
            comparisons.append({"reference_repeat": i, "actual_repeat": j, "stages": stages})
    return comparisons


def capture(args):
    target_phases(args.decode_steps)
    validate_all_gdn_states(args.decode_steps, args.block_step, args.detail_layer,
                            args.gdn_history, args.all_gdn_states)
    validate_all_attention_kv(args.all_gdn_states, args.all_attention_kv)
    tokens = load_prompt(args.prompt_ids)
    headers.require(1 <= args.repeats <= 3, "repeat count must be bounded by three")
    headers.require(not args.output_dir.exists(), "refusing to overwrite target repeats")
    reference = verify_inputs(args.reference_manifest, args.model_dir, args.image_identity)
    import yaml
    profile = yaml.safe_load(PROFILE.read_text())
    for key, value in profile["env"].items():
        if value is not None:
            os.environ[key] = str(value).replace("{model_dir}", str(PROFILE.parent))
    args.output_dir.mkdir(parents=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["EXL3_LOADER_REPORT_DIR"] = str(args.output_dir / "loader")
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
    trace = load_trace(args.trace_capture_json, args.decode_steps) if args.trace_capture_json else None
    if trace is not None:
        overrides["enable_trace_replay"] = True
    kwargs.update(overrides, model=str(args.model_dir), worker_extension_cls="capture_target.TargetCapture")
    params = SamplingParams(temperature=0, max_tokens=args.decode_steps + 1, ignore_eos=True,
                            trace_decode_token_ids=trace)
    llm = None
    ba_policy = None
    paths, records = [], []
    try:
        llm = LLM(**kwargs)
        if args.deterministic_ba:
            ba_policy = llm.collective_rpc("install_deterministic_ba", timeout=60)
        ordinary = llm.generate({"prompt_token_ids": tokens}, params, use_tqdm=False)
        ordinary_ids = [list(o.outputs[0].token_ids) for o in ordinary]
        for repeat in range(args.repeats):
            headers.require(llm.reset_prefix_cache(), "prefix reset failed")
            path = args.output_dir / f"repeat-{repeat}.safetensors"
            llm.collective_rpc("install_target_capture", timeout=60,
                               args=(str(path), args.decode_steps, args.block_step,
                                     args.detail_layer, args.gdn_history, args.all_gdn_states,
                                     args.all_attention_kv))
            observed = llm.generate({"prompt_token_ids": tokens}, params, use_tqdm=False)
            ids = [list(o.outputs[0].token_ids) for o in observed]
            result = llm.collective_rpc("finish_target_capture", timeout=60)
            headers.require(len(result) == 1, "requires one worker")
            result = result[0] | {"output_ids": ids, "ordinary_ids_exact": ids == ordinary_ids}
            with path.with_suffix(".json").open("x") as stream:
                json.dump(result, stream, indent=2); stream.write("\n")
            validate_prefix_witnesses(result["actual_input_witnesses"], tokens, ids[0])
            if args.decode_steps == 1:
                headers.require(ids == ordinary_ids, "different continuation prevents matched-prefix D1 comparison")
            if trace is not None:
                headers.require(ids == [trace], "original trace replay did not follow the requested prefix")
            if records:
                headers.require(ids == records[0]["output_ids"],
                                "different repeat continuations prevent matched-prefix comparisons")
            paths.append(path); records.append(result)
            print("TARGET_REPEAT_DONE", repeat, ids, flush=True)
    finally:
        if llm is not None:
            try:
                if ba_policy is not None:
                    restored = llm.collective_rpc("restore_deterministic_ba", timeout=60)
                    headers.require(restored == [{"restored_modules": 48}], "BA policy restoration failed")
            finally:
                llm.llm_engine.engine_core.shutdown()
    kind_prefix = "diagnostic_deterministic_BA_" if args.deterministic_ba else ""
    report = {"schema": 1, "kind": f"{kind_prefix}original_eager_target_P128_D{args.decode_steps}_reproducibility",
              "decode_steps": args.decode_steps,
              "sampling_mode": "original_trace_token_replay" if trace is not None else "original_greedy",
              "trace_source_sha256": digest(args.trace_capture_json.read_bytes()) if trace is not None else None,
              "trace_token_ids": trace,
              "image": IMAGE, "checkpoint": reference["checkpoint"]["identity"],
              "profile_sha256": digest(PROFILE.read_bytes()), "s1_overrides": overrides,
              "capture_tool_sha256": digest(Path(__file__).read_bytes()),
              "prompt_token_ids": tokens, "ordinary_output_ids": ordinary_ids,
              "prompt_source_sha256": digest(args.prompt_ids.read_bytes()) if args.prompt_ids else None,
              "observed_output_ids": [r["output_ids"] for r in records],
              "repeats": [{"path": str(p), "sha256": digest(p.read_bytes())} for p in paths],
              "comparisons": compare_repeats(paths, args.decode_steps),
              "selected_block_step": args.block_step,
              "selected_detail_layer": args.detail_layer,
              "selected_gdn_history": args.gdn_history,
              "all_gdn_states": args.all_gdn_states,
              "all_attention_kv": args.all_attention_kv,
              "diagnostic_deterministic_ba": bool(args.deterministic_ba), "ba_policy": ba_policy,
              "scope": "Bounded original full-target capture; layer0 states P128/D1, optional read-only D29 block/norm boundaries or standalone all48GDN active P128/D1 states and all16initialized attention KV. Cold unconsumed seeds and unwritten KV capacity excluded. No native pass or new numerical envelope."}
    with (args.output_dir / "comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print("TARGET_REPEAT_CAPTURE_DONE", args.output_dir, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--decode-steps", type=int, choices=(1, 29, 64), default=1)
    parser.add_argument("--block-step", type=int, choices=(-1, 29), default=-1)
    parser.add_argument("--detail-layer", type=int, choices=(-1, 1, 3, 21), default=-1)
    parser.add_argument("--gdn-history", action="store_true",
                        help="also observe selected GDN21 P128/D1 states and operands")
    parser.add_argument("--deterministic-ba", action="store_true",
                        help="separate diagnostic policy: original BA matmul deterministic; frozen gates unchanged")
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--trace-capture-json", type=Path)
    parser.add_argument("--prompt-ids", type=Path,
                        help="frozen held-out JSON prompt_token_ids, exactly128 valid IDs")
    parser.add_argument("--all-gdn-states", action="store_true",
                        help="read-only all48GDN active Conv/SSM boundaries for standalone P128/D1")
    parser.add_argument("--all-attention-kv", action="store_true",
                        help="also copy only initialized active FP8 KV of all16attention layers")
    capture(parser.parse_args())
