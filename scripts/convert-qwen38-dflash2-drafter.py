#!/usr/bin/env python3
"""Validate and pin the Qwen3.8 DFlash2 drafter checkpoint for the C++ loader.

W1 of `QWEN38-DFLASH2-STAGE-C`
(issue `.agents/issues/QWEN38-DFLASH2-STAGE-C/ISSUE-LOCAL-01M73J3QJ5CEBP2YS3ESK2ZMSE.md`,
spec `.agents/specs/qwen38-dflash2-stage-c.md`).

**Where the bytes come from.** `incoai/Qwen3.8-27B-DFlash2` @
`dedf8df68adfb1afeaf7b7480c0a0243108177b4` — the drafter-weights pin the
dflash2-vision bundle's PROVENANCE.json names (`weights.drafter`); the bundle's
own drafter code (`extra_code/qwen38/dflash2.py:39`, `DEFAULT_WEIGHTS`) resolves
the same repository. One flat `model.safetensors` plus `config.json`.

**Why this script does NOT re-encode anything (the recorded format decision).**
The loader the W2/W3 forward consumes already exists: the SPEC-DFLASH draft
path (`LoadDflashDraft`, `src/vllm/entrypoints/model_loader.cpp:512+` and
`src/vllm/model_executor/models/qwen3_dflash_weights.cpp`) reads a DFlash
draft checkpoint as a plain HF safetensors directory — exactly the layout this
repository publishes — and `include/vllm/model_executor/models/qwen3_dflash.h:14-111`
documents this very geometry (hidden 5120, 32q/8kv heads, head_dim 128, vocab
248320, `fc` [H, H*5], `kernel_projection` (1280, 5120), selector
`hidden_projection` (256, 5120)). Any re-encode would add a second format with
no consumer. The conversion is therefore the IDENTITY: the converter's job is
to (a) validate every tensor the loader reads — key set, dtype, shape — against
that contract and refuse BY NAME what is missing or mis-shaped, and (b) write
`meta.json` beside the checkpoint with the full enumeration and the source pin.

**Refuse-by-name.** Both halves of the W1 deliverable live here and in the
loader. This script refuses a checkpoint whose key set or shapes fail the
loader contract, naming each bad tensor. At RUNTIME the loader already refuses
by name (`qwen3_dflash_weights.cpp` `refuse_unowned` / "tensor not found: <name>"),
so no loader-side change is owed for W1; W2 wires the drafter forward through
the existing `LoadDflashDraft` entry point and inherits that refusal.

Usage:
    python3 convert-qwen38-dflash2-drafter.py <checkpoint_dir> [--write-meta]

`--write-meta` writes `<checkpoint_dir>/meta.json` (enumeration + source pin).
Without it the script validates and prints only.
"""
import hashlib
import json
import os
import struct
import sys

SOURCE = {
    "repository": "incoai/Qwen3.8-27B-DFlash2",
    "revision": "dedf8df68adfb1afeaf7b7480c0a0243108177b4",
    "file": "model.safetensors",
    "bundle_provenance": (
        "mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv @ "
        "28add9a1fbd0f259cba61f955023a65a10748064 (PROVENANCE.json weights.drafter)"
    ),
}

# The loader contract, specialized to this arm (qwen3_dflash.h:14-111 +
# qwen3_dflash_weights.cpp:862-866, 1044-1091). Shapes follow the checkpoint's
# torch row-major convention: [out, in] for projections.
H, L, TAPS, INTER, VOCAB = 5120, 5, 5, 17408, 248320
HDIM, SEL_RANK, CONV_KP_ROWS = 128, 256, 1280
QROWS, KVROWS = 32 * HDIM, 8 * HDIM

FIXED = {
    "fc.weight": ([H, H * TAPS], "BF16"),
    "hidden_norm.weight": ([H], "BF16"),
    "norm.weight": ([H], "BF16"),
}
PER_LAYER = {
    "input_layernorm.weight": ([H], "BF16"),
    "post_attention_layernorm.weight": ([H], "BF16"),
    "self_attn.q_proj.weight": ([QROWS, H], "BF16"),
    "self_attn.k_proj.weight": ([KVROWS, H], "BF16"),
    "self_attn.v_proj.weight": ([KVROWS, H], "BF16"),
    "self_attn.o_proj.weight": ([H, QROWS], "BF16"),
    "self_attn.q_norm.weight": ([HDIM], "BF16"),  # per-head RMSNorm (qwen3_dflash.h:158)
    "self_attn.k_norm.weight": ([HDIM], "BF16"),
    "mlp.gate_proj.weight": ([INTER, H], "BF16"),
    "mlp.up_proj.weight": ([INTER, H], "BF16"),
    "mlp.down_proj.weight": ([H, INTER], "BF16"),
    "attention_conv.base_kernel": ([2, 2, H], "BF16"),
    "attention_conv.kernel_projection.weight": ([CONV_KP_ROWS, H], "BF16"),
    "mlp_conv.base_kernel": ([2, 2, H], "BF16"),
    "mlp_conv.kernel_projection.weight": ([CONV_KP_ROWS, H], "BF16"),
}
SELECTOR = {
    "candidate_selector.predecessor_codebook": ([VOCAB, SEL_RANK], "BF16"),
    "candidate_selector.successor_codebook": ([VOCAB, SEL_RANK], "BF16"),
    "candidate_selector.hidden_projection.weight": ([SEL_RANK, H], "BF16"),
}

EXPECTED = dict(FIXED)
for i in range(L):
    for suffix, (shape, dt) in PER_LAYER.items():
        EXPECTED[f"layers.{i}.{suffix}"] = (shape, dt)
EXPECTED.update(SELECTOR)


def read_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    return {k: v for k, v in header.items() if k != "__metadata__"}, 8 + hlen


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write_meta = "--write-meta" in sys.argv
    if len(args) != 1:
        print(__doc__)
        return 2
    d = args[0]
    src = os.path.join(d, "model.safetensors")
    cfg_path = os.path.join(d, "config.json")

    with open(cfg_path) as f:
        cfg = json.load(f)
    dc = cfg.get("dflash_config") or {}
    problems = []
    for field, want in (
        ("hidden_size", H), ("num_hidden_layers", L),
        ("intermediate_size", INTER), ("vocab_size", VOCAB),
        ("head_dim", HDIM),
    ):
        if cfg.get(field) != want:
            problems.append(f"config.{field} = {cfg.get(field)}, loader contract wants {want}")
    if dc.get("target_layer_ids") != [5, 19, 33, 47, 61]:
        problems.append(f"dflash_config.target_layer_ids = {dc.get('target_layer_ids')}, wants [5, 19, 33, 47, 61]")
    if dc.get("block_size") != 8:
        problems.append(f"dflash_config.block_size = {dc.get('block_size')}, wants 8")

    header, data_start = read_header(src)
    size = os.path.getsize(src)

    missing = sorted(set(EXPECTED) - set(header))
    if missing:
        print("REFUSED: the drafter checkpoint is missing %d tensor(s) the "
              "loader reads: %s" % (len(missing), ", ".join(missing)))
        return 1
    for key, (want_shape, want_dt) in sorted(EXPECTED.items()):
        e = header[key]
        shape = [int(x) for x in e["shape"]]
        if shape != want_shape or e["dtype"] != want_dt:
            problems.append(f"{key}: {e['dtype']}{shape}, contract wants {want_dt}{want_shape}")

    extra = sorted(set(header) - set(EXPECTED))
    if problems:
        print("REFUSED: %d contract violation(s):" % len(problems))
        for p in problems:
            print("  " + p)
        return 1

    enumeration = []
    for key in sorted(header):
        e = header[key]
        enumeration.append({
            "name": key,
            "shape": [int(x) for x in e["shape"]],
            "dtype": e["dtype"],
            "bytes": e["data_offsets"][1] - e["data_offsets"][0],
        })

    print("OK: %d tensors, %d B, all match the loader contract" % (len(enumeration), size))
    if extra:
        print("note: %d tensor(s) outside the enumerated contract, copied through: %s"
              % (len(extra), ", ".join(extra)))
    meta = {
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "config": {k: cfg.get(k) for k in (
            "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "intermediate_size",
            "sliding_window", "rope_theta", "vocab_size", "layer_types",
            "dflash_config")},
        "consumed_by": "LoadDflashDraft (src/vllm/entrypoints/model_loader.cpp) — "
                       "the checkpoint IS the loader format; no re-encode",
        "source": dict(SOURCE, size=size, sha256=sha256(src),
                       config_sha256=sha256(cfg_path)),
        "tensors": enumeration,
    }
    if write_meta:
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print("wrote %s/meta.json" % d)
    else:
        print(json.dumps(meta["source"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
