#!/usr/bin/env python3
"""Attribute the first real C1 Q4 GDN step with pinned original split operators.

Captured seeds enter isolated original operator copies only. Require full-worker
core/Conv/all four FP32 snapshots to match before trusting split intermediates.
This is neither native state injection nor whole-model/default qualification.
"""
import argparse
import json
from pathlib import Path

from capture_projection import IMAGE
from capture_runtime_layout import verify_inputs
from extract_projection import digest, headers, write_safetensors


def capture(args):
    headers.require(not args.output.exists() and not args.output.with_suffix(".json").exists(),
                    "refusing to overwrite speculative GDN attribution")
    verify_inputs(args.reference_manifest, args.model_dir, args.image_identity)
    report = headers.read_json(args.capture_dir / "capture.json")
    headers.require(report["schema"] == "b70-integrated-mtp-layer0-attribution-v1" and
                    report["image"] == IMAGE and report["ordinary_ids_exact"] is True and
                    report["reference_label"] == "controlled_deterministic_BA" and
                    report["reference_manifest_sha256"] == digest(args.reference_manifest.read_bytes()),
                    "requires identified controlled layer0 capture")
    step = report["steps"][1]
    meta = step["gdn"]["l0"]
    plan = meta["plan"]
    headers.require(step["step"] == 1 and step["query_start_loc"] == [0, 4] and
                    meta["spec_query_start_loc"] == [0, 4] and
                    meta["previous_accepted_tokens"] == [1] and
                    meta["spec_state_indices"] == [[13, 12, 11, 10]] and
                    plan["conv_before"] == [0, 3] and plan["conv_after"] == [0, 6] and
                    plan["conv_slot"] == plan["ssm_before"] == 13 and
                    plan["ssm_after"] == [13, 12, 11, 10] and
                    step["file"] == "step-1.safetensors",
                    "requires first captured C1 Q4 step and its actual active slots")
    source = args.capture_dir / step["file"]
    headers.require(digest(source.read_bytes()) == step["sha256"], "capture payload mismatch")
    import torch
    import vllm._xpu_ops  # registers the pinned original operators
    from safetensors import safe_open
    from safetensors.torch import load_file
    headers.require(torch.__version__ == "2.13.0+xpu" and "B70" in torch.xpu.get_device_name(0),
                    "requires pinned Torch XPU and B70")
    host = load_file(str(source))
    headers.require(set(host) == set(step["tensors"]), "capture keys mismatch")
    for name, value in host.items():
        record = step["tensors"][name]
        headers.require(list(value.shape) == record["shape"] and
                        {torch.float16: "F16", torch.float32: "F32"}[value.dtype] == record["dtype"] and
                        digest(value.numpy().tobytes()) == record["sha256"], "bad capture tensor: " + name)
    weights, source_weights = {}, {}
    with safe_open(str(args.model_dir / "model-00001-of-00002.safetensors"), framework="pt") as shard:
        for suffix, dtype in (("conv1d.weight", torch.float16), ("A_log", torch.float32),
                              ("dt_bias", torch.float16)):
            name = "model.language_model.layers.0.linear_attn." + suffix
            value = shard.get_tensor(name)
            source_weights[name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                                    "sha256": digest(value.view(torch.uint8).numpy().tobytes())}
            weights[suffix] = value.to(dtype=dtype, device="xpu").contiguous()
    weights["conv1d.weight"] = weights["conv1d.weight"].view(10240, 4)
    conv_layout, ssm_layout = meta["conv_cache_layout"], meta["ssm_cache_layout"]
    headers.require(conv_layout["strides_elements"] == [1638400, 10240, 1] and
                    conv_layout["storage_offset_elements"] == 0 and
                    ssm_layout["strides_elements"] == [819200, 16384, 128, 1] and
                    ssm_layout["storage_offset_elements"] == 30720,
                    "requires actual captured shared cache strides/offsets")

    def caches():
        # Keep original physical slots/strides/Conv-SSM storage relation; bound
        # capacity to14 and poison spare/unwritten rows. Only captured seeds read.
        storage = torch.empty(14 * 3276800, dtype=torch.uint8, device="xpu")
        conv = storage.view(torch.float16).as_strided((14, 6, 10240), (1638400, 10240, 1))
        ssm = storage.view(torch.float32).as_strided((14, 48, 128, 128),
                                                   (819200, 16384, 128, 1), 30720)
        conv.fill_(-2); ssm.fill_(7)
        conv[13, :3].copy_(host["l0_conv_before"].to("xpu"))
        ssm[13].copy_(host["l0_ssm_before"].to("xpu"))
        return conv, ssm

    tensors, endpoints = {}, []

    def save(name, value):
        value = value.detach().cpu().contiguous()
        headers.require(value.numel() <= 5_000_000 and torch.isfinite(value).all().item(),
                        "unbounded/nonfinite speculative GDN stage")
        tensors[name] = ({torch.float16: "F16", torch.float32: "F32"}[value.dtype],
                         list(value.shape), value.numpy().tobytes())

    common = dict(num_prefills=0, num_decodes=0, num_spec_decodes=1, num_actual_tokens=4, tp_size=1,
                  spec_query_start_loc=torch.tensor([0, 4], dtype=torch.int32, device="xpu"),
                  # Pure active C1 four-token identity mapping; full-worker
                  # endpoint equality below is required, not inferred from it.
                  spec_token_indx=torch.arange(4, dtype=torch.int32, device="xpu"),
                  spec_state_indices_tensor=torch.tensor([[13, 12, 11, 10]], dtype=torch.int32, device="xpu"),
                  num_accepted_tokens=torch.tensor([1], dtype=torch.int32, device="xpu"))
    conv_args = dict(projected_states_qkvz=host["l0_qkvz"].to("xpu"),
                     projected_states_ba=host["l0_ba"].to("xpu"),
                     num_k_heads=16, num_v_heads=48, head_k_dim=128, head_v_dim=128,
                     conv_weights=weights["conv1d.weight"], conv_bias=None,
                     activation="silu", reorder_input=True)
    delta_args = dict(num_v_heads=48, head_v_dim=128, A_log=weights["A_log"], dt_bias=weights["dt_bias"])
    fused_conv, fused_ssm = caches()
    split_conv, split_ssm = caches()
    core = torch.empty((4, 48, 128), dtype=torch.float16, device="xpu")
    z = torch.empty_like(core)
    torch.ops._xpu_C.gdn_attention(core, z, conv_state=fused_conv, ssm_state=fused_ssm,
                                  A_log=weights["A_log"], dt_bias=weights["dt_bias"],
                                  has_initial_state=None, non_spec_query_start_loc=None,
                                  non_spec_token_indx=None, non_spec_state_indices_tensor=None,
                                  **conv_args, **common)
    split_core, split_z = torch.empty_like(core), torch.empty_like(z)
    parts = torch.ops._xpu_C.causal_conv1d_spec(split_z, conv_state=split_conv, **conv_args, **common)
    headers.require(len(parts) == 5, "original split Conv must return Q/K/V/B/A")
    for label, value in zip(("q", "k", "v", "b", "a"), parts):
        save("conv_" + label, value)
    torch.ops._xpu_C.gated_delta_rule_spec(split_core, *parts, ssm_state=split_ssm, **delta_args, **common)
    torch.xpu.synchronize()
    boundaries = [("core", core, split_core, host["l0_core"]),
                  ("z", z, split_z, host["l0_z"]),
                  ("conv_after", fused_conv[13], split_conv[13], host["l0_conv_after"])]
    boundaries += [("ssm_after_t" + str(t), fused_ssm[slot], split_ssm[slot],
                    host["l0_ssm_after_t" + str(t)]) for t, slot in enumerate((13, 12, 11, 10))]
    for label, actual, split, expected in boundaries:
        save(label, actual); save("split_" + label, split)
        expected_bytes = expected.contiguous().numpy().tobytes()
        endpoints.append({"boundary": label, "fused_exact": tensors[label][2] == expected_bytes,
                          "split_exact": tensors["split_" + label][2] == expected_bytes})
    for label, value in zip(("q", "k", "v", "b", "a"), parts):
        save("delta_" + label, value)
    write_safetensors(args.output, tensors, {"image": IMAGE})
    package = Path("/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels")
    result = {"schema": "b70-original-spec-GDN-layer0-attribution-v1", "image": IMAGE,
              "reference_label": report["reference_label"], "source_payload_sha256": step["sha256"],
              "source_capture_sha256": digest((args.capture_dir / "capture.json").read_bytes()),
              "reference_manifest_sha256": digest(args.reference_manifest.read_bytes()),
              "source_weights": source_weights, "exact_endpoints": endpoints,
              "all_endpoints_exact": all(e["fused_exact"] and e["split_exact"] for e in endpoints),
              "capture_sha256": digest(args.output.read_bytes()), "tool_sha256": digest(Path(__file__).read_bytes()),
              "operator_schemas": {name: str(getattr(torch.ops._xpu_C, name).default._schema)
                                   for name in ("gdn_attention", "causal_conv1d_spec", "gated_delta_rule_spec")},
              "binary_hashes": {name: digest((package / name).read_bytes())
                                for name in ("_xpu_C.abi3.so", "libgdn_attn_kernels_xe_2.so")},
              "tensor_hashes": {name: {"dtype": t[0], "shape": t[1], "sha256": digest(t[2])}
                                for name, t in tensors.items()},
              "scope": __doc__}
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    headers.require(result["all_endpoints_exact"], "original fused/split/full-worker endpoint mismatch; evidence preserved")
    print("SPEC_GDN_ATTRIBUTION_EXACT", len(endpoints), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    capture(parser.parse_args())
