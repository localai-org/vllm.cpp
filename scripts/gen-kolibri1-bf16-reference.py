#!/usr/bin/env python3
"""Dequantize the Kolibri-1 FP8 block checkpoint to a bf16 reference.

Streams the 32-shard Aleph-Alpha/Kolibri-1 fp8 safetensors checkpoint through
the W1 loader math — every F8_E4M3 tensor [N, K] with its F32 weight_scale_inv
sibling [N/128, K/128] decodes to f32 as `fp8_e4m3(byte) * scale_inv[block]`
per 128x128 block (the DeepSeek-style weight_block_size [128, 128] grid,
quantization_config in the checkpoint's config.json) — and writes bf16
safetensors shards with the SAME shard layout and a new index. Non-converted
tensors (the bf16 router gates, norms, embedding; the f32 scales) are copied
byte-identically. The result is a plain bf16 model with NO
quantization_config: the dequantized weights ARE the reference, so
transformers-class runtimes load it directly (MODEL-TEXT-kolibri-1 W3,
.agents/specs/kolibri-1-cpu.md risk R4).

Usage:
  python3 scripts/gen-kolibri1-bf16-reference.py \
      /mnt/models/Aleph-Alpha/Kolibri-1 /mnt/models/Aleph-Alpha/Kolibri-1-bf16
"""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file

BLOCK = 128


def dequantize(fp8: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """[N, K] F8_E4M3 x [N/128, K/128] F32 grid -> bf16, f32 arithmetic."""
    rows, cols = fp8.shape
    br, bc = scale_inv.shape
    if rows != br * BLOCK or cols != bc * BLOCK:
        raise ValueError(
            f"block grid mismatch: weight {(rows, cols)} vs scale {tuple(scale_inv.shape)}"
        )
    wide = fp8.to(torch.float32).view(br, BLOCK, bc, BLOCK)
    wide = wide * scale_inv.view(br, 1, bc, 1)
    return wide.view(rows, cols).to(torch.bfloat16)


def main() -> int:
    src_dir, dst_dir = Path(sys.argv[1]), Path(sys.argv[2])
    index = json.loads((src_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Strip the quantization_config: the output IS the dequantized model.
    config = json.loads((src_dir / "config.json").read_text())
    removed = config.pop("quantization_config", None)
    if removed is None:
        raise SystemExit("source config has no quantization_config — nothing to dequantize")
    (dst_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for sidecar in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        if (src_dir / sidecar).exists():
            shutil.copy(src_dir / sidecar, dst_dir / sidecar)

    # Group tensor names by source shard, preserving each shard's partition.
    shard_names: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shard_names.setdefault(shard, []).append(name)

    # Resumable: a shard file already present was written whole by save_file.
    todo = [s for s in sorted(shard_names) if not (dst_dir / s).exists()]
    print(f"{len(shard_names) - len(todo)} shards already written, "
          f"{len(todo)} to go", flush=True)

    census: Counter[str] = Counter()
    for shard in todo:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(src_dir / shard, framework="pt") as f:
            for name in shard_names[shard]:
                tensor = f.get_tensor(name)
                if tensor.dtype == torch.float8_e4m3fn:
                    scale_name = name + "_scale_inv"
                    scale = f.get_tensor(scale_name)
                    tensor = dequantize(tensor, scale)
                    census["fp8_dequantized"] += 1
                elif tensor.dtype == torch.bfloat16:
                    census["bf16_copied"] += 1
                else:
                    census[str(tensor.dtype)] += 1
                tensors[name] = tensor
        # A bf16 shard is ~2x the fp8 shard; write once, keep the name.
        save_file(tensors, dst_dir / shard, metadata={"format": "pt"})
        print(f"{shard}: {len(tensors)} tensors written", flush=True)
        del tensors

    (dst_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    print(f"census: {dict(census)}")
    print(f"wrote {dst_dir} ({len(shard_names)} shards, quantization_config stripped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
