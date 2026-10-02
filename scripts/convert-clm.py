#!/usr/bin/env python3
"""Convert a CLM projection-head checkpoint into a self-contained ClmModel dir.

MODEL-CLM (.agents/specs/clm.md). Contrastive-LM/CLM-v0.1-8B ships its two
projection heads as a torch pickle, CLM_v0.1-8B.pt: a dict with state_head
and action_head state dicts, the raw logit_scale (a log), and the head cfg.
The encoder is the unmodified base, Qwen/Qwen3-8B. This engine reads
safetensors and one config.json, so the conversion writes:

  <out>/config.json          the base config, architectures ["ClmModel"], and
                             the head cfg and raw logit_scale as clm_* keys
                             (the engine applies exp(.).clamp(max=100) itself)
  <out>/head.safetensors     state_head.<name> / action_head.<name>, F32, with
                             the checkpoint's own names (inp, hidden.<i>,
                             norms.<i>, out), as heads.py make_head defines them
  <out>/model-*.safetensors  the base shards and index, copied unchanged
  <out>/tokenizer*.json ...  the base tokenizer (the CLM repo carries none)

Usage:
    python3 scripts/convert-clm.py <CLM_v0.1-8B.pt or its repo dir> \\
        --base-model-dir <Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218> \\
        --output-dir <out>

Needs torch and safetensors.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

# The checkpoint and base this converter was verified against (docs/models/clm.md).
PUBLISHED_SHA256 = "b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5"
BASE_MODEL = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
CHECKPOINT_NAME = "CLM_v0.1-8B.pt"
ACTIVATIONS = ("gelu", "relu", "silu")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "generation_config.json")
HEADS = ("state_head", "action_head")


class ConvertError(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def head_config(ck):
    """The make_head arguments, read the way heads.py HeadPair._load reads them."""
    if not isinstance(ck, dict) or not all(k in ck for k in (*HEADS, "logit_scale", "cfg")):
        raise ConvertError("not a CLM checkpoint: expected state_head, action_head, "
                           "logit_scale and cfg")
    cfg = dict(ck["cfg"])
    for key in ("width", "depth"):
        if key not in cfg:
            raise ConvertError(f"cfg has no {key!r}")
    out = {"width": int(cfg["width"]), "depth": int(cfg["depth"]),
           "projection_dim": int(ck.get("projection_dim", cfg.get("projection_dim", 512))),
           "activation": cfg.get("activation", "gelu"),
           "layernorm": bool(cfg.get("layernorm", False)),
           "residual": bool(cfg.get("residual", False)),
           "hidden_size": int(cfg.get("hidden_size", 4096))}
    if out["activation"] not in ACTIVATIONS:
        raise ConvertError(f"unsupported activation {out['activation']!r}")
    if out["depth"] < 2:
        raise ConvertError("cfg depth must be >= 2")
    return out


def expected_shapes(cfg):
    """{state-dict name: shape} for one make_head(**cfg)."""
    w, h, p = cfg["width"], cfg["hidden_size"], cfg["projection_dim"]
    shapes = {"inp.weight": (w, h), "inp.bias": (w,), "out.weight": (p, w), "out.bias": (p,)}
    for i in range(cfg["depth"] - 2):
        shapes[f"hidden.{i}.weight"] = (w, w)
        shapes[f"hidden.{i}.bias"] = (w,)
        if cfg["layernorm"]:
            shapes[f"norms.{i}.weight"] = (w,)
            shapes[f"norms.{i}.bias"] = (w,)
    return shapes


def head_tensors(ck, cfg):
    """{safetensors name: F32 tensor} for both heads, every name and shape checked.

    A tensor that is missing, extra or mis-shaped is refused: the engine would
    otherwise load a head that answers wrong.
    """
    want = expected_shapes(cfg)
    for head in HEADS:
        sd = ck[head]
        missing = sorted(set(want) - set(sd))
        extra = sorted(set(sd) - set(want))
        if missing:
            raise ConvertError(f"{head}: missing {missing[0]}")
        if extra:
            raise ConvertError(f"{head}: unexpected tensor {extra[0]}")
        for name, shape in want.items():
            if tuple(sd[name].shape) != shape:
                raise ConvertError(f"{head}.{name}: shape {tuple(sd[name].shape)}, "
                                   f"expected {shape}")
    # Checked in full first, so a bad action head never leaves a half result.
    return {f"{head}.{name}": ck[head][name].detach().float().contiguous()
            for head in HEADS for name in want}


def build_config(base_config, cfg, logit_scale, provenance):
    if base_config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise ConvertError("the base must be a Qwen3ForCausalLM checkpoint")
    if base_config.get("hidden_size") != cfg["hidden_size"]:
        raise ConvertError(f"the heads take {cfg['hidden_size']}-d embeddings, the base "
                           f"hidden_size is {base_config.get('hidden_size')}")
    out = dict(base_config)
    out["architectures"] = ["ClmModel"]
    out["clm_hidden_size"] = cfg["hidden_size"]
    out["clm_width"] = cfg["width"]
    out["clm_depth"] = cfg["depth"]
    out["clm_projection_dim"] = cfg["projection_dim"]
    out["clm_activation"] = cfg["activation"]
    out["clm_layernorm"] = cfg["layernorm"]
    out["clm_residual"] = cfg["residual"]
    # The raw log, as stored; the engine computes exp(.).clamp(max=100.0).
    out["clm_logit_scale"] = float(logit_scale)
    out["clm_provenance"] = provenance
    return out


def base_shards(base_dir):
    index = base_dir / "model.safetensors.index.json"
    if index.is_file():
        return sorted(set(json.loads(index.read_text())["weight_map"].values())), index
    shards = sorted(p.name for p in base_dir.glob("*.safetensors"))
    if not shards:
        raise ConvertError(f"{base_dir} holds no safetensors shards")
    return shards, None


def find_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        path = path / CHECKPOINT_NAME
    if not path.is_file():
        raise ConvertError(f"{path} does not exist")
    return path


def convert(checkpoint, base_dir, output_dir, allow_other_checkpoint=False):
    import torch
    from safetensors.torch import save_file

    checkpoint, base_dir, output_dir = find_checkpoint(checkpoint), Path(base_dir), Path(output_dir)
    digest = sha256_file(checkpoint)
    if digest != PUBLISHED_SHA256 and not allow_other_checkpoint:
        raise ConvertError(f"{checkpoint.name} sha256 {digest} is not the verified "
                           f"CLM-v0.1-8B checkpoint; pass --allow-other-checkpoint for "
                           f"a head you trained")
    base_config = json.loads((base_dir / "config.json").read_text())
    ck = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    cfg = head_config(ck)
    tensors = head_tensors(ck, cfg)
    snapshot = base_dir.resolve().name
    provenance = {"checkpoint": checkpoint.name, "checkpoint_sha256": digest,
                  "base_model": ck["cfg"].get("model", BASE_MODEL),
                  "base_revision": snapshot if re.fullmatch(r"[0-9a-f]{40}", snapshot) else None,
                  "reference": "https://github.com/Contrastive-LM/CLM"}
    config = build_config(base_config, cfg, float(torch.as_tensor(ck["logit_scale"]).float()),
                          provenance)

    output_dir.mkdir(parents=True, exist_ok=True)
    shards, index = base_shards(base_dir)
    for shard in shards:
        shutil.copyfile(base_dir / shard, output_dir / shard)
        print(f"  copied {shard}")
    if index is not None:
        shutil.copyfile(index, output_dir / index.name)
    for name in TOKENIZER_FILES:
        if (base_dir / name).is_file():
            shutil.copyfile(base_dir / name, output_dir / name)
    if not (output_dir / "tokenizer.json").is_file():
        raise ConvertError(f"{base_dir} has no tokenizer.json")
    save_file(tensors, str(output_dir / "head.safetensors"), metadata={"format": "pt"})
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {output_dir} (ClmModel, {len(tensors)} head tensors, "
          f"logit_scale {config['clm_logit_scale']:.6f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint", help=f"{CHECKPOINT_NAME} or the directory holding it")
    ap.add_argument("--base-model-dir", required=True)
    ap.add_argument("--output-dir", "-o", required=True)
    ap.add_argument("--allow-other-checkpoint", action="store_true")
    args = ap.parse_args()
    try:
        convert(args.checkpoint, args.base_model_dir, args.output_dir,
                args.allow_other_checkpoint)
    except ConvertError as e:
        print(f"convert-clm: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
