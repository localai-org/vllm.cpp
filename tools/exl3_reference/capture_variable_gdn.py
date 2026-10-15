#!/usr/bin/env python3
"""Original fused/split variable C1 GDN on repeated real layer0 projections.

Each P127/128/129/256/4096 is one original prefill call, followed by an actual
three-token prefill continuation and D1 decode. No rechunked CPU reference.
Cold/continued Conv and FP32 states are owned by the original operators.
"""
import argparse
import json
from pathlib import Path

from capture_projection import IMAGE
from capture_runtime_layout import verify_inputs
from extract_projection import digest, headers, write_safetensors


def capture(args):
    headers.require(not args.output.exists() and not args.output.with_suffix(".json").exists(),
                    "refusing to overwrite variable GDN capture")
    verify_inputs(args.reference_manifest, args.model, args.image_identity)
    receipt = json.loads(args.block.with_suffix(".json").read_text())
    headers.require(receipt["image"] == IMAGE and digest(args.block.read_bytes()) == receipt["capture_sha256"],
                    "real block identity mismatch")
    import torch
    import vllm._xpu_ops
    from safetensors import safe_open
    from safetensors.torch import load_file
    headers.require(torch.__version__ == "2.13.0+xpu" and "B70" in torch.xpu.get_device_name(0),
                    "pinned Torch/B70 required")
    host = load_file(str(args.block))
    weights = {}
    tensors, cases = {}, []

    def save(name, value):
        t = value.detach().cpu().contiguous()
        headers.require(torch.isfinite(t).all().item(), f"nonfinite stage {name}")
        tensors[name] = {torch.float16: "F16", torch.float32: "F32"}[t.dtype], list(t.shape), t.numpy().tobytes()

    with safe_open(str(args.model / "model-00001-of-00002.safetensors"), framework="pt") as shard:
        for suffix, dtype in (("conv1d.weight", torch.float16), ("A_log", torch.float32),
                              ("dt_bias", torch.float16)):
            name = "model.language_model.layers.0.linear_attn." + suffix
            value = shard.get_tensor(name).to(dtype=dtype, device="xpu").contiguous()
            if suffix == "conv1d.weight": value = value.view(10240, 4)
            weights[suffix] = value
            save("weight_" + suffix.replace(".", "_"), value)

    def caches():
        # Original physical strides; only logical slots/states are compared.
        storage = torch.empty(5 * 3276800, dtype=torch.uint8, device="xpu")
        conv = storage.view(torch.float16).as_strided((5, 3, 10240), (1638400, 10240, 1))
        state = storage.view(torch.float32).as_strided((5, 48, 128, 128), (819200, 16384, 128, 1), 15360)
        conv.fill_(-2); state.fill_(7)
        conv[4].zero_(); state[4].zero_()
        return conv, state

    def inputs(rows, decode=False):
        prefix = "d1" if decode else "p128"
        qkv = host[prefix + "_qkvz_output"]
        ba = host[prefix + "_ba_output"]
        return (qkv.repeat((rows + qkv.shape[0] - 1) // qkv.shape[0], 1)[:rows].contiguous().to("xpu"),
                ba.repeat((rows + ba.shape[0] - 1) // ba.shape[0], 1)[:rows].contiguous().to("xpu"))

    for length in args.rows:
        headers.require(length in (127, 128, 129, 256, 4096), "unsupported reference prefill length")
        fused_conv, fused_state = caches()
        split_conv, split_state = caches()
        for suffix, rows, prefill, initial in (("", length, True, False),
                                               ("_append3", 3, True, True),
                                               ("_d1", 1, False, True)):
            label = f"p{length}" + suffix
            qkv, ba = inputs(rows, not prefill)
            save(label + "_qkvz", qkv); save(label + "_ba", ba)
            qsl = torch.tensor([0, rows], dtype=torch.int32, device="xpu")
            index = torch.tensor([4], dtype=torch.int32, device="xpu")
            his = torch.tensor([initial], dtype=torch.bool, device="xpu") if prefill else None
            common = dict(num_prefills=int(prefill), num_decodes=int(not prefill), num_spec_decodes=0,
                          has_initial_state=his, non_spec_query_start_loc=qsl, non_spec_token_indx=None,
                          non_spec_state_indices_tensor=index, num_actual_tokens=rows, tp_size=1)
            conv_args = dict(projected_states_qkvz=qkv, projected_states_ba=ba,
                             num_k_heads=16, num_v_heads=48, head_k_dim=128, head_v_dim=128,
                             conv_weights=weights["conv1d.weight"], conv_bias=None,
                             activation="silu", reorder_input=True)
            delta_args = dict(A_log=weights["A_log"], dt_bias=weights["dt_bias"])
            core = torch.empty((rows, 48, 128), dtype=torch.float16, device="xpu")
            z = torch.empty_like(core)
            torch.ops._xpu_C.gdn_attention(core, z, conv_state=fused_conv, ssm_state=fused_state,
                                          **conv_args, **delta_args, **common,
                                          spec_query_start_loc=None, spec_token_indx=None,
                                          spec_state_indices_tensor=None, num_accepted_tokens=None)
            split_core, split_z = torch.empty_like(core), torch.empty_like(z)
            parts = torch.ops._xpu_C.causal_conv1d_non_spec(split_z, conv_state=split_conv,
                                                          **conv_args, **common)
            headers.require(len(parts) == 5, "missing original Conv stages")
            if prefill:
                for name, tensor in zip(("q", "k", "v"), parts[:3]): save(label + "_" + name, tensor[:rows])
                save(label + "_beta", parts[3][:, :rows].transpose(0, 1))
            else:
                # Decode normalization belongs to the original recurrence;
                # these three Conv outputs remain the packed unnormalized row.
                save(label + "_decode_mixed", torch.cat([t.reshape(1, -1) for t in parts[:3]], dim=-1))
            torch.ops._xpu_C.gated_delta_rule_non_spec(split_core, *parts, num_v_heads=48, head_v_dim=128,
                                                      ssm_state=split_state, **delta_args, **common)
            torch.xpu.synchronize()
            for name, fused, split in (("core", core, split_core), ("z", z, split_z),
                                       ("conv_state", fused_conv[4], split_conv[4]),
                                       ("state", fused_state[4], split_state[4])):
                headers.require(torch.equal(fused.view(torch.uint8), split.view(torch.uint8)),
                                f"original fused/split mismatch {label} {name}")
                save(label + "_" + name, fused)
            for conv, state in ((fused_conv, fused_state), (split_conv, split_state)):
                headers.require((conv[:4] == -2).all().item() and (state[:4] == 7).all().item(),
                                f"original inactive slot changed {label}")
            cases.append({"label": label, "rows": rows, "prefill": prefill, "has_initial_state": initial,
                          "cold_length": length, "capacity": rows + 63 if prefill else rows,
                          "inactive_slots_unchanged": True, "fused_split_bit_exact": True})
            print("VARIABLE_GDN_CAPTURE_EXACT", label, flush=True)
    write_safetensors(args.output, tensors, {"image": IMAGE, "scope": __doc__})
    args.output.chmod(0o644)
    package = Path("/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels")
    result = {"schema": 1, "image": IMAGE, "scope": __doc__, "cases": cases,
              "block_sha256": receipt["capture_sha256"], "tool_sha256": digest(Path(__file__).read_bytes()),
              "capture_sha256": digest(args.output.read_bytes()),
              "binary_hashes": {name: digest((package / name).read_bytes())
                                for name in ("_xpu_C.abi3.so", "libgdn_attn_kernels_xe_2.so")},
              "tensor_hashes": {name: {"dtype": t[0], "shape": t[1], "sha256": digest(t[2])}
                                for name, t in sorted(tensors.items())}}
    with args.output.with_suffix(".json").open("x") as f: json.dump(result, f, indent=2); f.write("\n")
    print("VARIABLE_GDN_CAPTURE_SHA256", result["capture_sha256"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[127, 128, 129, 256, 4096])
    capture(parser.parse_args())
