#!/usr/bin/env python3
"""Observe one real GDN layer (0 or 1), eager P128/D1 and active state.

Pinned independent worker; hooks copy bounded active tensors without replacing
arithmetic. No MTP/graphs. Never read an uninitialized cold state or capacity.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

from capture_projection import IMAGE
from capture_runtime_layout import PROFILE, VLLM_ROOT, active_metadata, verify_inputs
from extract_projection import digest, headers, write_safetensors
from runtime_layout import describe, tensor_layout


def select_gdn_layer(modules, layer_index):
    headers.require(type(layer_index) is int and layer_index in (0, 1, 21),
                    "bounded block capture supports only target GDN layers0/1/21")
    suffix = f"language_model.model.layers.{layer_index}"
    layers = [(name, module) for name, module in modules if name.endswith(suffix)]
    headers.require(len(layers) == 1, "requires one unique real target GDN layer")
    headers.require(layers[0][1].layer_type == "linear_attention", "selected layer must be GDN")
    return layers[0]


class BlockCapture:
    def install_block_capture(self, output, layer_index=0):
        import torch
        from vllm.forward_context import get_forward_context
        runner = self.model_runner
        name, layer = select_gdn_layer(runner.model.named_modules(), layer_index)
        mixer = layer.linear_attn
        self._block_output = Path(output)
        self._block_tensors, self._block_records, self._block_hooks = {}, [], []
        self._block_current = None

        def save(stage, value):
            if value is None:
                return
            prefix = self._block_current
            headers.require(prefix is not None, "stage outside captured block")
            key = prefix + "_" + stage
            headers.require(key not in self._block_tensors and value.numel() <= 5_000_000,
                            "duplicate/unbounded block stage: " + key)
            host = value.detach().cpu().contiguous()
            dtype = {torch.float16: "F16", torch.float32: "F32", torch.int32: "I32",
                     torch.int64: "I64"}.get(host.dtype)
            headers.require(dtype is not None and
                            (not host.is_floating_point() or torch.isfinite(host).all().item()),
                            "invalid/nonfinite block stage: " + key)
            self._block_tensors[key] = (dtype, list(host.shape), host.numpy().tobytes())

        def state(stage, meta):
            indices = meta.prefill_state_indices if meta.num_prefills else meta.non_spec_state_indices_tensor
            headers.require(indices.numel() == 1, "capture requires one active state slot")
            slot = int(indices.detach().cpu().item())
            cache = mixer.kv_cache
            headers.require(len(cache) == 2 and 0 <= slot < cache[0].shape[0], "bad active cache")
            # Cold-prefill initial contents are not initialized/consumed. Record
            # that contract rather than copying unknown bytes as a zero fixture.
            initial = bool(meta.prefill_has_initial_state.detach().cpu().item()) if meta.num_prefills else True
            if stage == "before" and not initial:
                self._block_records[-1]["initial_state_consumed"] = False
                return
            save("conv_state_" + stage, cache[0][slot])
            save("ssm_state_" + stage, cache[1][slot])

        def begin(module, args, kwargs):
            metadata = get_forward_context().attn_metadata
            if not metadata or not getattr(runner.req_states, "req_id_to_index", {}):
                return
            headers.require(len(self._block_records) < 2, "unexpected extra target step")
            meta = metadata[mixer.prefix]
            rows = meta.num_actual_tokens
            phase = "p128" if meta.num_prefills else "d1"
            headers.require((phase, rows) == ("p128", 128) if not self._block_records
                            else (phase, rows) == ("d1", 1), "unexpected step shape/order")
            self._block_current = phase
            self._block_records.append({"phase": phase, "metadata": active_metadata(meta),
                                        "conv_cache_layout": tensor_layout(mixer.kv_cache[0]),
                                        "ssm_cache_layout": tensor_layout(mixer.kv_cache[1])})
            values = dict(zip(("positions", "hidden_states", "residual"), args)) | kwargs
            save("positions", values["positions"])
            save("hidden_in", values["hidden_states"])
            save("residual_in", values.get("residual"))
            state("before", meta)

        def end(module, args, kwargs, output):
            if self._block_current is None:
                return
            save("hidden_out", output[0]); save("residual_out", output[1])
            meta = get_forward_context().attn_metadata[mixer.prefix]
            state("after", meta)
            print("BLOCK_CAPTURE_STEP", self._block_current, flush=True)
            self._block_current = None

        self._block_hooks.append(layer.register_forward_pre_hook(begin, with_kwargs=True))
        self._block_hooks.append(layer.register_forward_hook(end, with_kwargs=True))
        # Every captured value belongs to an original module boundary. Mixed
        # projection pairs return (output,bias); norm may return (hidden,res).
        for label, module in (("input_norm", layer.input_layernorm),
                              ("post_norm", layer.post_attention_layernorm),
                              ("qkvz", mixer.in_proj_qkvz), ("ba", mixer.in_proj_ba),
                              ("gated_norm", mixer.norm), ("out_proj", mixer.out_proj),
                              ("mixer", mixer), ("gate_up", layer.mlp.gate_up_proj),
                              ("swiglu", layer.mlp.act_fn), ("down", layer.mlp.down_proj)):
            def before(module, args, kwargs, label=label):
                if self._block_current is None:
                    return
                if label == "gated_norm":
                    save("core_out", args[0]); save("z", args[1])
                elif label == "post_norm":
                    save("post_norm_input", args[0]); save("post_norm_residual_input", args[1])
            def after(module, args, kwargs, output, label=label):
                if self._block_current is None:
                    return
                if isinstance(output, tuple):
                    save(label + "_output", output[0])
                    if label == "post_norm":
                        save("post_norm_residual_output", output[1])
                else:
                    save(label + "_output", output)
            self._block_hooks.append(module.register_forward_pre_hook(before, with_kwargs=True))
            self._block_hooks.append(module.register_forward_hook(after, with_kwargs=True))
        return {"layer": name, "observer": "bounded read-only eager module hooks"}

    def finish_block_capture(self):
        import torch
        torch.xpu.synchronize()
        for hook in self._block_hooks:
            hook.remove()
        headers.require([r["phase"] for r in self._block_records] == ["p128", "d1"], "missing steps")
        write_safetensors(self._block_output, self._block_tensors, {"image": IMAGE})
        return {"steps": self._block_records, "capture_sha256": digest(self._block_output.read_bytes()),
                "tensor_hashes": {n: {"dtype": t[0], "shape": t[1], "sha256": digest(t[2])}
                                  for n, t in sorted(self._block_tensors.items())},
                "compilation_config": describe(self.model_runner.compilation_config)}


def capture(args):
    headers.require(not args.output.exists() and not args.output.with_suffix(".json").exists(),
                    "refusing to overwrite block capture")
    reference = verify_inputs(args.reference_manifest, args.model_dir, args.image_identity)
    import yaml
    profile = yaml.safe_load(PROFILE.read_text())
    for key, value in profile["env"].items():
        if value is not None:
            os.environ[key] = str(value).replace("{model_dir}", str(PROFILE.parent))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["EXL3_LOADER_REPORT_DIR"] = str(args.output.parent / "block-loader")
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
    # S1 eager target: preserve the resolved profile's 'none' CustomOp base,
    # explicitly disable compilation/graphs and speculation. Prefix reset below
    # ensures a cold input, without changing the original cache configuration.
    overrides = {"enforce_eager": True, "speculative_config": None,
                 "compilation_config": {"mode": 0, "cudagraph_mode": "NONE", "custom_ops": ["none"]}}
    kwargs.update(overrides, model=str(args.model_dir), worker_extension_cls="capture_block.BlockCapture")
    tokens = [1000 + (i * 37) % 4096 for i in range(128)]
    params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
    llm = None
    try:
        print("BLOCK_CAPTURE_START", flush=True)
        llm = LLM(**kwargs)
        ordinary = llm.generate({"prompt_token_ids": tokens}, params, use_tqdm=False)
        headers.require(llm.reset_prefix_cache(), "prefix reset failed")
        installed = llm.collective_rpc("install_block_capture", timeout=60,
                                       args=(str(args.output), args.layer_index))
        observed = llm.generate({"prompt_token_ids": tokens}, params, use_tqdm=False)
        ordinary_ids = [list(o.outputs[0].token_ids) for o in ordinary]
        observed_ids = [list(o.outputs[0].token_ids) for o in observed]
        headers.require(ordinary_ids == observed_ids, "observation changed greedy output IDs")
        records = llm.collective_rpc("finish_block_capture", timeout=60)
        headers.require(len(records) == 1, "requires one worker")
        result = records[0] | {"schema": 1, "kind": f"pinned_layer{args.layer_index}_eager_P128_D1",
                              "layer_index": args.layer_index,
                              "image": IMAGE, "checkpoint": reference["checkpoint"]["identity"],
                              "profile_sha256": digest(PROFILE.read_bytes()), "s1_overrides": overrides,
                              "capture_tool_sha256": digest(Path(__file__).read_bytes()),
                              "observer": installed, "prompt_token_ids": tokens,
                              "ordinary_output_ids": ordinary_ids, "observed_output_ids": observed_ids,
                              "ordinary_vs_observed_greedy_ids_exact": True,
                              "scope": "Actual selected GDN layer eager P128 then D1 stages and active states only. Greedy ID observer check is not bitwise whole-model parity or performance."}
        with args.output.with_suffix(".json").open("x") as stream:
            json.dump(result, stream, indent=2); stream.write("\n")
        print("BLOCK_CAPTURE_DONE", args.output, result["capture_sha256"], flush=True)
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--layer-index", type=int, choices=(0, 1, 21), default=0)
    capture(parser.parse_args())
