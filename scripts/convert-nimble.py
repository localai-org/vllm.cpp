#!/usr/bin/env python3
"""Convert a Bespoke-Nimble LoRA adapter into a self-contained NimbleModel dir.

MODEL-NIMBLE (.agents/specs/nimble.md). Nimble ships an UNMERGED PEFT adapter
(bespokelabs/Bespoke-Nimble-9B, r=16, alpha=32) for Qwen/Qwen3.5-9B. This
engine has no LoRA on the server path, so the adapter is merged here, once,
exactly as PEFT's LoraLayer.merge does it on a BF16 base with F32 LoRA
weights:

    W' = bf16( float32(W) + (B @ A) * (lora_alpha / r) )

This is the kev precedent (scripts/convert-kev.py). The output directory holds
the base's shard layout with merged tensors, a config.json whose architecture
is NimbleModel, and the adapter's tokenizer and chat template.

Usage:
    python3 scripts/convert-nimble.py <adapter_dir> \
        --base-model-dir <Qwen/Qwen3.5-9B @ c202236235762e1c871ad0ccb60c8ee5ba337b9a> \
        --output-dir <out>

The adapter dir is a download of bespokelabs/Bespoke-Nimble-9B (T=1.0) or
bespokelabs/Bespoke-Nimble-9B-v2 (T=2.179078721266035). Both use the same
prompt code; the temperature comes from the adapter's temperature_config.json.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

# parallel_schema.py, the prompt contract NimbleDecide implements. Both
# published revisions pin this hash as schema_config.json prompt_code_sha256.
PROMPT_CODE_SHA256 = "a0a0f94d0f65e972bc20d088678ad3d595ff1c42b9c0d78f96034526303f63fc"
SUPPORTED_TASKS = ("schema_candidate_classification_v1",
                   "schema_candidate_classification_v2")
ADAPTER_PREFIX = "base_model.model.model.language_model.layers."
BASE_PREFIX = "model.language_model.layers."
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
PROVENANCE_FILES = ("schema_config.json", "temperature_config.json", "adapter_config.json")


class ConvertError(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_sha256sums(adapter_dir):
    """Check every file SHA256SUMS lists. A missing listed file is an error."""
    sums = Path(adapter_dir) / "SHA256SUMS"
    if not sums.is_file():
        raise ConvertError(f"{sums} is missing; download the complete adapter repo")
    checked = []
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = Path(adapter_dir) / name
        if not path.is_file():
            raise ConvertError(f"SHA256SUMS lists {name}, which is missing")
        if sha256_file(path) != digest:
            raise ConvertError(f"sha256 mismatch for {name}")
        checked.append(name)
    return checked


def map_adapter_key(name):
    """Adapter tensor name -> (base weight name, "A" | "B"), or None."""
    if not name.startswith(ADAPTER_PREFIX) or not name.endswith(".weight"):
        return None
    stem = name[len(ADAPTER_PREFIX):-len(".weight")]
    module, sep, which = stem.rpartition(".lora_")
    if not sep or which not in ("A", "B"):
        return None
    return BASE_PREFIX + module + ".weight", which


def plan_merge(adapter_shapes, base_shapes, rank):
    """Pair every adapter module with its base weight and check the shapes.

    adapter_shapes: {adapter tensor name: shape}; base_shapes: {name: shape}.
    Returns {base name: (lora_A name, lora_B name)}. Raises ConvertError on an
    unpaired, unmatched or mis-shaped module, because a silently skipped
    module is a model that loads and answers wrong.
    """
    pairs = {}
    for name in adapter_shapes:
        mapped = map_adapter_key(name)
        if mapped is None:
            raise ConvertError(f"unrecognized adapter tensor {name}")
        base, which = mapped
        pairs.setdefault(base, {})[which] = name
    plan = {}
    for base, ab in sorted(pairs.items()):
        if set(ab) != {"A", "B"}:
            raise ConvertError(f"{base}: adapter carries only lora_{''.join(ab)}")
        if base not in base_shapes:
            raise ConvertError(f"{base}: no such tensor in the base checkpoint")
        a_shape, b_shape = tuple(adapter_shapes[ab["A"]]), tuple(adapter_shapes[ab["B"]])
        out_f, in_f = tuple(base_shapes[base])
        if a_shape != (rank, in_f) or b_shape != (out_f, rank):
            raise ConvertError(f"{base}: lora_A {a_shape} / lora_B {b_shape} do not "
                               f"fit weight {(out_f, in_f)} at rank {rank}")
        plan[base] = (ab["A"], ab["B"])
    return plan


def build_config(base_config, temperature, max_length, provenance):
    cfg = dict(base_config)
    cfg["architectures"] = ["NimbleModel"]
    cfg["nimble_temperature"] = float(temperature)
    cfg["nimble_max_length"] = int(max_length)
    cfg["nimble_provenance"] = provenance
    return cfg


def read_contract(adapter_dir):
    adapter_dir = Path(adapter_dir)
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    if adapter_config.get("peft_type") != "LORA" or adapter_config.get("use_dora") \
            or adapter_config.get("use_rslora") or adapter_config.get("bias", "none") != "none":
        raise ConvertError("only a plain LoRA (no DoRA, no rsLoRA, no bias) is supported")
    schema = json.loads((adapter_dir / "schema_config.json").read_text())
    if schema.get("task") not in SUPPORTED_TASKS:
        raise ConvertError(f"unsupported training contract {schema.get('task')!r}")
    if schema.get("prompt_code_sha256") != PROMPT_CODE_SHA256:
        raise ConvertError("the adapter's prompt contract is not the parallel_schema.py "
                           "this engine implements")
    temperature = json.loads((adapter_dir / "temperature_config.json").read_text())["temperature"]
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) \
            or not temperature > 0:
        raise ConvertError("temperature_config.json temperature must be positive")
    return adapter_config, schema, float(temperature)


def merge_tensor(weight, lora_a, lora_b, scaling):
    """PEFT LoraLayer.merge: base += (B @ A) * scaling, in F32, stored BF16."""
    delta = (lora_b.float() @ lora_a.float()) * scaling
    return (weight.float() + delta).to(weight.dtype)


def convert(adapter_dir, base_dir, output_dir):
    import torch  # noqa: F401  (merge_tensor's tensors)
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    adapter_dir, base_dir, output_dir = Path(adapter_dir), Path(base_dir), Path(output_dir)
    checked = verify_sha256sums(adapter_dir)
    adapter_config, schema, temperature = read_contract(adapter_dir)
    rank = int(adapter_config["r"])
    scaling = float(adapter_config["lora_alpha"]) / rank

    base_config = json.loads((base_dir / "config.json").read_text())
    if base_config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ConvertError("the base must be a Qwen3_5ForConditionalGeneration checkpoint")
    snapshot = base_dir.resolve().name
    if len(snapshot) == 40 and snapshot != schema["revision"]:
        raise ConvertError(f"base snapshot {snapshot} is not the pinned {schema['revision']}")

    index_path = base_dir / "model.safetensors.index.json"
    if index_path.is_file():
        shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    base_shapes = {}
    for shard in shards:
        with safe_open(str(base_dir / shard), framework="pt") as f:
            for name in f.keys():
                base_shapes[name] = f.get_slice(name).get_shape()

    adapter = load_file(str(adapter_dir / "adapter_model.safetensors"))
    plan = plan_merge({k: list(v.shape) for k, v in adapter.items()}, base_shapes, rank)
    print(f"LoRA merge: r={rank} alpha={adapter_config['lora_alpha']} "
          f"scaling={scaling} modules={len(plan)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged = 0
    for shard in shards:
        tensors = load_file(str(base_dir / shard))
        for name in list(tensors):
            if name in plan:
                a_name, b_name = plan[name]
                tensors[name] = merge_tensor(tensors[name], adapter[a_name],
                                             adapter[b_name], scaling).contiguous()
                merged += 1
        save_file(tensors, str(output_dir / shard), metadata={"format": "pt"})
        print(f"  wrote {shard}")
        del tensors
    if merged != len(plan):
        raise ConvertError(f"merged {merged} of {len(plan)} planned modules")
    if index_path.is_file():
        shutil.copy2(index_path, output_dir / index_path.name)

    provenance = {
        "adapter_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
        "base_model": schema["model"],
        "base_revision": schema["revision"],
        "task": schema["task"],
        "sha256sums_checked": checked,
    }
    config = build_config(base_config, temperature, schema.get("max_length", 8192), provenance)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in TOKENIZER_FILES + PROVENANCE_FILES:
        shutil.copy2(adapter_dir / name, output_dir / name)
    print(f"Wrote {output_dir} (NimbleModel, T={temperature}, {merged} merged modules)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("adapter_dir")
    ap.add_argument("--base-model-dir", required=True)
    ap.add_argument("--output-dir", "-o", required=True)
    args = ap.parse_args()
    try:
        convert(args.adapter_dir, args.base_model_dir, args.output_dir)
    except ConvertError as e:
        print(f"convert-nimble: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
