#!/usr/bin/env python3
"""Replay original fused/split XPU GDN on immutable real layer0/1/21 operands.

The split operators are from the same pinned binary, not reimplementations.
Require exact fused/split/full-worker endpoints before accepting intermediates.
"""
import argparse
import json
from pathlib import Path

from capture_projection import IMAGE
from capture_runtime_layout import verify_inputs
from extract_projection import digest, headers, write_safetensors


def block_layer_index(receipt):
    index = receipt.get("layer_index", 0)
    headers.require(type(index) is int and index in (0, 1, 21), "unsupported captured GDN layer")
    return index


def capture(args):
    headers.require(not args.output.exists() and not args.output.with_suffix(".json").exists(),
                    "refusing to overwrite GDN capture")
    verify_inputs(args.reference_manifest, args.model_dir, args.image_identity)
    receipt = json.loads(args.block.with_suffix(".json").read_text())
    layer_index = block_layer_index(receipt)
    headers.require(receipt["image"] == IMAGE and
                    digest(args.block.read_bytes()) == receipt["capture_sha256"], "block identity mismatch")
    import torch
    import vllm._xpu_ops  # registers the original pinned operators
    from safetensors import safe_open
    from safetensors.torch import load_file
    headers.require(torch.__version__ == "2.13.0+xpu" and "B70" in torch.xpu.get_device_name(0),
                    "requires pinned Torch XPU and B70")
    host = load_file(str(args.block))
    for name, record in receipt["tensor_hashes"].items():
        headers.require(list(host[name].shape) == record["shape"] and
                        digest(host[name].numpy().tobytes()) == record["sha256"], "bad block tensor: " + name)
    weights = {}
    source_weights = {}
    with safe_open(str(args.model_dir / "model-00001-of-00002.safetensors"), framework="pt") as shard:
        for suffix, dtype in (("conv1d.weight", torch.float16), ("A_log", torch.float32),
                              ("dt_bias", torch.float16)):
            name = f"model.language_model.layers.{layer_index}.linear_attn." + suffix
            value = shard.get_tensor(name)
            source_weights[name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                                    "sha256": digest(value.view(torch.uint8).numpy().tobytes())}
            weights[suffix] = value.to(dtype=dtype, device="xpu").contiguous()
    weights["conv1d.weight"] = weights["conv1d.weight"].view(10240, 4)
    tensors, endpoints = {}, []

    def save(name, value):
        value = value.detach().cpu().contiguous()
        headers.require(value.numel() <= 5_000_000 and torch.isfinite(value).all().item(),
                        "unbounded/nonfinite GDN stage")
        tensors[name] = ({torch.float16: "F16", torch.float32: "F32"}[value.dtype],
                         list(value.shape), value.numpy().tobytes())

    # Preserve active slot4 and both original first-dimension strides. Limit
    # inactive capacity to five slots; kernels consume only the active slot.
    def caches():
        storage = torch.empty(5 * 3276800, dtype=torch.uint8, device="xpu")
        conv = storage.view(torch.float16).as_strided((5, 3, 10240), (1638400, 10240, 1))
        ssm = storage.view(torch.float32).as_strided((5, 48, 128, 128), (819200, 16384, 128, 1), 15360)
        conv.fill_(-2); ssm.fill_(7)
        return conv, ssm

    fused_conv, fused_ssm = caches()
    split_conv, split_ssm = caches()
    for phase, rows in (("p128", 128), ("d1", 1)):
        prefill = phase == "p128"
        qkvz = host[phase + "_qkvz_output"].to("xpu")
        ba = host[phase + "_ba_output"].to("xpu")
        qsl = torch.tensor([0, rows], dtype=torch.int32, device="xpu")
        idx = torch.tensor([4], dtype=torch.int32, device="xpu")
        initial = torch.tensor([False], dtype=torch.bool, device="xpu") if prefill else None
        common = dict(num_prefills=int(prefill), num_decodes=int(not prefill), num_spec_decodes=0,
                      has_initial_state=initial, non_spec_query_start_loc=qsl,
                      non_spec_token_indx=None, non_spec_state_indices_tensor=idx,
                      num_actual_tokens=rows, tp_size=1)
        conv_args = dict(projected_states_qkvz=qkvz, projected_states_ba=ba,
                         num_k_heads=16, num_v_heads=48, head_k_dim=128, head_v_dim=128,
                         conv_weights=weights["conv1d.weight"], conv_bias=None,
                         activation="silu", reorder_input=True)
        delta_args = dict(A_log=weights["A_log"], dt_bias=weights["dt_bias"])
        core = torch.empty((rows, 48, 128), dtype=torch.float16, device="xpu")
        z = torch.empty_like(core)
        torch.ops._xpu_C.gdn_attention(core, z, conv_state=fused_conv, ssm_state=fused_ssm,
                                      **conv_args, **delta_args, **common,
                                      spec_query_start_loc=None, spec_token_indx=None,
                                      spec_state_indices_tensor=None, num_accepted_tokens=None)
        split_core = torch.empty_like(core)
        split_z = torch.empty_like(z)
        parts = torch.ops._xpu_C.causal_conv1d_non_spec(split_z, conv_state=split_conv,
                                                      **conv_args, **common)
        headers.require(len(parts) == 5, "original split Conv must return Q/K/V/B/A")
        for label, value in zip(("q", "k", "v", "b", "a"), parts):
            save(phase + "_conv_" + label, value)
        torch.ops._xpu_C.gated_delta_rule_non_spec(split_core, *parts, num_v_heads=48,
                                                  head_v_dim=128, ssm_state=split_ssm,
                                                  **delta_args, **common)
        torch.xpu.synchronize()
        for label, actual, split, expected in (
                ("core", core, split_core, host[phase + "_core_out"].reshape(rows, 48, 128)),
                ("z", z, split_z, host[phase + "_z"].reshape(rows, 48, 128)),
                ("conv_state", fused_conv[4], split_conv[4], host[phase + "_conv_state_after"]),
                ("ssm_state", fused_ssm[4], split_ssm[4], host[phase + "_ssm_state_after"])):
            headers.require(torch.equal(actual.cpu(), expected) and torch.equal(split.cpu(), expected),
                            "original endpoint mismatch: " + phase + "." + label)
            endpoints.append(phase + "." + label)
            save(phase + "_" + label, actual)
        for label, value in zip(("q", "k", "v", "b", "a"), parts):
            save(phase + "_delta_" + label, value)
        print("GDN_CAPTURE_EXACT", phase, flush=True)
    write_safetensors(args.output, tensors, {"image": IMAGE})
    package = Path("/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels")
    result = {"image": IMAGE, "kind": "original_fused_split_GDN_real_P128_D1",
              "layer_index": layer_index,
              "block_sha256": receipt["capture_sha256"], "source_weights": source_weights,
              "capture_sha256": digest(args.output.read_bytes()), "exact_endpoints": endpoints,
              "tool_sha256": digest(Path(__file__).read_bytes()),
              "binary_hashes": {name: digest((package / name).read_bytes())
                                for name in ("_xpu_C.abi3.so", "libgdn_attn_kernels_xe_2.so")},
              "tensor_hashes": {name: {"dtype": t[0], "shape": t[1], "sha256": digest(t[2])}
                                for name, t in tensors.items()},
              "scope": "Same original fused and split operators, real captured operands; active slot4/strides preserved, inactive capacity bounded to5. No whole-model or performance claim."}
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print("GDN_CAPTURE_DONE", result["capture_sha256"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    capture(parser.parse_args())
