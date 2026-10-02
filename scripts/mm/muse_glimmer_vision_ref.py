#!/usr/bin/env python3
"""Muse Glimmer perception-encoder REFERENCE dump (W3 gate fixture).

Emits `tests/vllm/models/muse_glimmer_vision_goldens.inc` — the per-stage
reference tensors the C++ vision-tower gate
(`tests/vllm/models/test_muse_glimmer_vision.cpp`) compares against.

WHERE THE REFERENCE COMES FROM
------------------------------
Every function below is a VERBATIM transcription of vllm PR #51655 head
`075d645af`, `vllm/model_executor/models/muse_glimmer.py`, with vLLM's
parallel-linear / MMEncoderAttention wrappers replaced by their plain-torch
equivalents (`nn.Linear`, a block-diagonal varlen SDPA). The transcribed spans:

  _make_2d_rope                muse_glimmer.py:741-759
  _get_pos_emb                 muse_glimmer.py:761-820
  _pixel_shuffle_downsample    muse_glimmer.py:822-842
  _get_sparse_permutation      muse_glimmer.py:844-867
  _patchify                    muse_glimmer.py:902-935
  MuseGlimmerVisionEncoder.forward   muse_glimmer.py:937-1034
  MuseGlimmerVisionMLP.forward       muse_glimmer.py:641-648
  MuseGlimmerVisionBlock.forward     muse_glimmer.py:651-689
  MuseGlimmerVisionAttention.forward muse_glimmer.py:555-638
  MuseGlimmerVisionAdapter.forward   muse_glimmer.py:1036-1044
  ApplyRotaryEmb.forward_static      rotary_embedding/common.py:143-184

NOT the pinned oracle. The vLLM parity pin `a7c23ac96d` registers Muse Glimmer
(it did not exist at the older pin `555967922`), but this script is a torch
transcription, not the vLLM runtime, so NO speed claim is derivable from this
fixture. See
`.agents/specs/muse-glimmer.md` §0.

The synthetic weights are produced by an explicit LCG that the C++ test
reproduces BIT-EXACTLY (same integer recurrence, same double-precision
mapping), so no weight blobs are committed — only the reference OUTPUTS.

Usage:  python3 scripts/mm/muse_glimmer_vision_ref.py [--out PATH]
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn.functional as F

torch.use_deterministic_algorithms(True)


# --- the deterministic synthetic-weight LCG (mirrored in the C++ test) --------
def lcg(seed: int, n: int, scale: float) -> torch.Tensor:
    """values[i] = (((s>>8) / 2^24) * 2 - 1) * scale, s advancing by a Numerical-
    Recipes LCG. Computed in double, rounded ONCE to f32 — exactly what the C++
    test does, so both sides hold identical bits."""
    s = seed & 0xFFFFFFFF
    out = []
    for _ in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out.append(((((s >> 8) / 16777216.0) * 2.0) - 1.0) * scale)
    return torch.tensor(out, dtype=torch.float32)


# --- the fixture config (small, but every code path is live) ------------------
class Cfg:
    hidden_size = 32
    num_attention_heads = 4          # head_dim 8, spatial_dim 4, 2 freqs/axis
    num_hidden_layers = 3
    intermediate_size = 48
    patch_size = 2
    patch_temporal = 2
    merge_kernel_size = 2
    pos_emb_height = 4               # window block = 4x4 = 16 tokens
    pos_emb_width = 4
    adapter_dim = 16
    layer_norm_eps = 1e-5
    layer_types = ["window_attention", "full_attention", "window_attention"]

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def patch_dim(self):
        return self.patch_temporal * 3 * self.patch_size**2

    @property
    def output_dim(self):
        return self.hidden_size * self.merge_kernel_size**2


CFG = Cfg()

# Two images: 12x12 (grid 6x6 -> padded 8x8: blocks of 16/8/8/4 valid tokens,
# and pos-emb interpolation off BOTH ends of the 4x4 learned grid) and 8x8 with
# 6 channels (grid 4x4 -> exactly one block; the patch_temporal*3 branch of
# _patchify).
IMAGES = [(3, 12, 12), (CFG.patch_temporal * 3, 8, 8)]


# --- muse_glimmer.py:741-759 -------------------------------------------------
def make_2d_rope(cfg, grid_height, grid_width):
    spatial_dim = cfg.head_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim)
    )
    height = torch.arange(1, grid_height + 1, dtype=torch.float32)
    width = torch.arange(1, grid_width + 1, dtype=torch.float32)
    height = height.unsqueeze(1).expand(-1, grid_width).reshape(-1)
    width = width.unsqueeze(0).expand(grid_height, -1).reshape(-1)
    freq_w = torch.outer(width, inv_freq)
    freq_h = torch.outer(height, inv_freq)
    freqs = torch.cat([freq_w, freq_h], dim=-1)
    return torch.cos(freqs), torch.sin(freqs)


# --- muse_glimmer.py:761-820 -------------------------------------------------
def get_pos_emb(cfg, pos_emb, grid_height, grid_width):
    h_grid = (torch.arange(grid_height, dtype=torch.float32) + 0.5) * (
        cfg.pos_emb_height / grid_height
    ) - 0.5
    w_grid = (torch.arange(grid_width, dtype=torch.float32) + 0.5) * (
        cfg.pos_emb_width / grid_width
    ) - 0.5
    h_floor = torch.floor(h_grid).long()
    w_floor = torch.floor(w_grid).long()
    h_ceil = h_floor + 1
    w_ceil = w_floor + 1
    h_frac = h_grid - h_floor.float()
    w_frac = w_grid - w_floor.float()

    h_floor_valid = (h_floor >= 0) & (h_floor < cfg.pos_emb_height)
    h_ceil_valid = (h_ceil >= 0) & (h_ceil < cfg.pos_emb_height)
    w_floor_valid = (w_floor >= 0) & (w_floor < cfg.pos_emb_width)
    w_ceil_valid = (w_ceil >= 0) & (w_ceil < cfg.pos_emb_width)
    h_floor = h_floor.clamp(0, cfg.pos_emb_height - 1)
    h_ceil = h_ceil.clamp(0, cfg.pos_emb_height - 1)
    w_floor = w_floor.clamp(0, cfg.pos_emb_width - 1)
    w_ceil = w_ceil.clamp(0, cfg.pos_emb_width - 1)

    h_floor_offset = h_floor * cfg.pos_emb_width
    h_ceil_offset = h_ceil * cfg.pos_emb_width
    indices = torch.stack(
        [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
    )
    weights = torch.stack(
        [
            (
                (1 - h_frac)[:, None]
                * (1 - w_frac)[None, :]
                * (h_floor_valid[:, None] & w_floor_valid[None, :])
            ).flatten(),
            (
                (1 - h_frac)[:, None]
                * w_frac[None, :]
                * (h_floor_valid[:, None] & w_ceil_valid[None, :])
            ).flatten(),
            (
                h_frac[:, None]
                * (1 - w_frac)[None, :]
                * (h_ceil_valid[:, None] & w_floor_valid[None, :])
            ).flatten(),
            (
                h_frac[:, None]
                * w_frac[None, :]
                * (h_ceil_valid[:, None] & w_ceil_valid[None, :])
            ).flatten(),
        ]
    )
    return (pos_emb[indices] * weights[..., None]).sum(0)


# --- muse_glimmer.py:822-842 -------------------------------------------------
def pixel_shuffle_downsample(cfg, hidden_states, grid_height, grid_width):
    factor = cfg.merge_kernel_size
    output_tokens = (grid_height // factor) * (grid_width // factor)
    permutation = torch.arange(grid_height * grid_width)
    permutation = permutation.view(
        grid_height // factor, factor, grid_width // factor, factor
    )
    permutation = permutation.permute(0, 2, 1, 3).reshape(-1)
    hidden_states = hidden_states.squeeze(0)[permutation]
    hidden_size = hidden_states.shape[-1]
    hidden_states = (
        hidden_states.view(output_tokens, factor * factor, hidden_size)
        .permute(0, 2, 1)
        .contiguous()
        .view(output_tokens, hidden_size * factor * factor)
    )
    return hidden_states.unsqueeze(0)


# --- muse_glimmer.py:844-867 -------------------------------------------------
def get_sparse_permutation(cfg, grid_height, grid_width):
    block_height = cfg.pos_emb_height
    block_width = cfg.pos_emb_width
    padded_height = math.ceil(grid_height / block_height) * block_height
    padded_width = math.ceil(grid_width / block_width) * block_width
    indices = torch.arange(grid_height * grid_width).view(grid_height, grid_width)
    indices = F.pad(
        indices,
        (0, padded_width - grid_width, 0, padded_height - grid_height),
        value=-1,
    ).flatten()
    indices = indices.view(
        padded_height // block_height,
        block_height,
        padded_width // block_width,
        block_width,
    )
    indices = indices.permute(0, 2, 1, 3).reshape(-1)
    valid = (indices != -1).view(-1, block_height * block_width)
    return indices[indices != -1], valid.sum(dim=1).tolist()


# --- muse_glimmer.py:902-935 -------------------------------------------------
def patchify(cfg, pixels):
    patch_size = cfg.patch_size
    _, channels, height, width = pixels.shape
    grid_height = height // patch_size
    grid_width = width // patch_size
    if channels == 3:
        patches = pixels.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size
        )
        patches = patches.contiguous().view(
            1, channels, grid_height, grid_width, patch_size, patch_size
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.unsqueeze(3).expand(-1, -1, -1, cfg.patch_temporal, -1, -1, -1)
    elif channels == cfg.patch_temporal * 3:
        frame_patches = []
        for frame_idx in range(cfg.patch_temporal):
            frame = pixels[:, frame_idx * 3 : (frame_idx + 1) * 3]
            frame = frame.unfold(2, patch_size, patch_size).unfold(
                3, patch_size, patch_size
            )
            frame = frame.contiguous().view(
                1, 3, grid_height, grid_width, patch_size, patch_size
            )
            frame_patches.append(frame.permute(0, 2, 3, 1, 4, 5).contiguous())
        patches = torch.stack(frame_patches, dim=3)
    else:
        raise ValueError("bad channel count")
    return patches.reshape(1, grid_height * grid_width, -1)


# --- rotary_embedding/common.py:143-184 (is_neox_style=True, fp32) -----------
def apply_rotary_emb(x, cos, sin):
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


# --- MMEncoderAttention, as a block-diagonal varlen SDPA ---------------------
def varlen_attention(query, key, value, seq_lens, scale):
    """query/key/value [L, heads, head_dim]; attends only within each segment."""
    out = torch.empty_like(query)
    off = 0
    for n in seq_lens:
        q = query[off : off + n].transpose(0, 1).float()  # [h, n, d]
        k = key[off : off + n].transpose(0, 1).float()
        v = value[off : off + n].transpose(0, 1).float()
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        out[off : off + n] = torch.matmul(probs, v).transpose(0, 1).to(out.dtype)
        off += n
    return out


# --- the weights ------------------------------------------------------------
class Weights:
    def __init__(self, cfg):
        h, i = cfg.hidden_size, cfg.intermediate_size
        self.conv1 = lcg(1001, h * cfg.patch_dim, 0.1).view(h, cfg.patch_dim)
        self.pos_emb = lcg(1002, cfg.pos_emb_height * cfg.pos_emb_width * h, 0.5).view(
            cfg.pos_emb_height * cfg.pos_emb_width, h
        )
        self.ln_pre_w = lcg(1003, h, 0.3) + 1.0
        self.ln_pre_b = lcg(1004, h, 0.1)
        self.ln_post_w = lcg(1005, h, 0.3) + 1.0
        self.ln_post_b = lcg(1006, h, 0.1)
        self.blocks = []
        for l in range(cfg.num_hidden_layers):
            s = 2000 + 100 * l
            self.blocks.append(
                dict(
                    ln1_w=lcg(s + 1, h, 0.3) + 1.0,
                    ln1_b=lcg(s + 2, h, 0.1),
                    ln2_w=lcg(s + 3, h, 0.3) + 1.0,
                    ln2_b=lcg(s + 4, h, 0.1),
                    qkv_w=lcg(s + 5, 3 * h * h, 0.1).view(3 * h, h),
                    qkv_b=lcg(s + 6, 3 * h, 0.1),
                    o_w=lcg(s + 7, h * h, 0.1).view(h, h),
                    o_b=lcg(s + 8, h, 0.1),
                    fc_w=lcg(s + 9, i * h, 0.1).view(i, h),
                    fc_b=lcg(s + 10, i, 0.1),
                    proj_w=lcg(s + 11, h * i, 0.1).view(h, i),
                    proj_b=lcg(s + 12, h, 0.1),
                )
            )
        self.ad_fc = lcg(9001, cfg.adapter_dim * cfg.output_dim, 0.1).view(
            cfg.adapter_dim, cfg.output_dim
        )
        self.ad_proj = lcg(9002, cfg.adapter_dim * cfg.adapter_dim, 0.1).view(
            cfg.adapter_dim, cfg.adapter_dim
        )


# --- muse_glimmer.py:937-1034 ------------------------------------------------
def encoder_forward(cfg, w, pixel_values, capture):
    has_sparse = any(t != "full_attention" for t in cfg.layer_types)
    all_hidden, all_cos, all_sin = [], [], []
    sparse_seq_lens, global_seq_lens, metadata = [], [], []

    for pixels in pixel_values:
        grid_height = pixels.shape[-2] // cfg.patch_size
        grid_width = pixels.shape[-1] // cfg.patch_size
        num_tokens = grid_height * grid_width
        patched = patchify(cfg, pixels)
        capture.setdefault("patchify", []).append(patched.reshape(-1))
        hidden_states = F.linear(patched, w.conv1)          # bias=False (:710)
        pos = get_pos_emb(cfg, w.pos_emb, grid_height, grid_width)
        capture.setdefault("pos_emb", []).append(pos.reshape(-1))
        # `.to(dtype)` (_get_pos_emb is f32; the sum is in the tower dtype).
        hidden_states = hidden_states + pos.unsqueeze(0).to(hidden_states.dtype)
        hidden_states = F.layer_norm(
            hidden_states.view(-1, cfg.hidden_size),
            (cfg.hidden_size,),
            w.ln_pre_w,
            w.ln_pre_b,
            cfg.layer_norm_eps,
        ).view(1, -1, cfg.hidden_size)
        capture.setdefault("ln_pre", []).append(hidden_states.reshape(-1))
        cos, sin = make_2d_rope(cfg, grid_height, grid_width)
        capture.setdefault("rope_cos", []).append(cos.reshape(-1))
        capture.setdefault("rope_sin", []).append(sin.reshape(-1))

        permutation = None
        if has_sparse:
            permutation, seq_lens = get_sparse_permutation(cfg, grid_height, grid_width)
            capture.setdefault("perm", []).append(permutation.to(torch.int32))
            capture.setdefault("seq_lens", []).extend(seq_lens)
            hidden_states = hidden_states[:, permutation]
            cos = cos[permutation]
            sin = sin[permutation]
            sparse_seq_lens.extend(seq_lens)

        all_hidden.append(hidden_states.squeeze(0))
        all_cos.append(cos)
        all_sin.append(sin)
        global_seq_lens.append(num_tokens)
        metadata.append((grid_height, grid_width, num_tokens, permutation))

    hidden_states = torch.cat(all_hidden).unsqueeze(0)
    cos = torch.cat(all_cos)
    sin = torch.cat(all_sin)
    scale = cfg.head_dim**-0.5
    nh = cfg.num_attention_heads
    hd = cfg.head_dim

    for layer_idx, (layer_type, blk) in enumerate(zip(cfg.layer_types, w.blocks)):
        seq_lens = global_seq_lens if layer_type == "full_attention" else sparse_seq_lens
        flattened = hidden_states.view(-1, cfg.hidden_size)
        normed = F.layer_norm(
            flattened, (cfg.hidden_size,), blk["ln1_w"], blk["ln1_b"], cfg.layer_norm_eps
        )
        qkv = F.linear(normed, blk["qkv_w"], blk["qkv_b"])
        qkv = qkv.view(flattened.shape[0], 3, nh, hd)
        query, key, value = qkv.unbind(1)
        stacked = apply_rotary_emb(torch.stack([query, key]).float(), cos, sin)
        query, key = stacked.to(qkv.dtype).unbind(0)
        attn = varlen_attention(query, key, value, seq_lens, scale)
        attn = attn.reshape(flattened.shape[0], -1)
        attn = F.linear(attn, blk["o_w"], blk["o_b"])
        flattened = flattened + attn
        mlp_in = F.layer_norm(
            flattened, (cfg.hidden_size,), blk["ln2_w"], blk["ln2_b"], cfg.layer_norm_eps
        )
        mlp = F.linear(F.gelu(F.linear(mlp_in, blk["fc_w"], blk["fc_b"])), blk["proj_w"], blk["proj_b"])
        flattened = flattened + mlp
        hidden_states = flattened.view(1, -1, cfg.hidden_size)
        if layer_idx == 0:
            capture["block0"] = hidden_states.reshape(-1)

    features, offset = [], 0
    for grid_height, grid_width, num_tokens, permutation in metadata:
        item = hidden_states[:, offset : offset + num_tokens]
        offset += num_tokens
        if permutation is not None:
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(len(permutation))
            item = item[:, inverse]
        item = F.layer_norm(
            item.view(-1, cfg.hidden_size),
            (cfg.hidden_size,),
            w.ln_post_w,
            w.ln_post_b,
            cfg.layer_norm_eps,
        ).view(1, -1, cfg.hidden_size)
        features.append(
            pixel_shuffle_downsample(cfg, item, grid_height, grid_width).squeeze(0)
        )
    return torch.cat(features)


# --- muse_glimmer.py:1036-1044 ----------------------------------------------
def adapter_forward(w, hidden_states):
    return F.gelu(F.linear(F.gelu(F.linear(hidden_states, w.ad_fc)), w.ad_proj))


def emit(fh, name, values):
    fh.write(f"inline constexpr float {name}[] = {{\n")
    for i in range(0, len(values), 6):
        # `.9e` (never `.9g`): a `g` format renders 0.0 as `0`, and `0f` is an
        # invalid C++ literal suffix on an integer.
        row = ", ".join(f"{float(v):.9e}f" for v in values[i : i + 6])
        fh.write(f"    {row},\n")
    fh.write("};\n\n")


def emit_int(fh, name, values):
    fh.write(f"inline constexpr int {name}[] = {{\n")
    for i in range(0, len(values), 16):
        row = ", ".join(str(int(v)) for v in values[i : i + 16])
        fh.write(f"    {row},\n")
    fh.write("};\n\n")


# --- REAL mode: the released tensors (a first reference run) ---------------
# `--real-dir DIR` reads `config.json` and every `*.safetensors` in DIR whose
# names start `model.vision_` (the tower from shard 1, the adapter and the
# projection from shard 2 of meta-models/Muse-Glimmer-30B @ a4e59da5), runs the
# SAME transcription on a deterministic pixel tensor in f32 and in the
# production bf16, and writes raw little-endian f32 binaries to `--out-dir`
# for the env-gated tests/vllm/models/test_muse_glimmer_vision_real.cpp.
# The reference is the pinned vLLM `a7c23ac96d` formulas; their vision
# semantics are unchanged from 075d645af (see .agents/specs/muse-glimmer-parity.md).
def load_st_dir(path):
    import glob
    import json as _json
    import struct as _struct

    out = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = _struct.unpack("<Q", fh.read(8))[0]
            hdr = _json.loads(fh.read(n))
            base = 8 + n
            for k, v in hdr.items():
                if k == "__metadata__" or not k.startswith("model.vision_"):
                    continue
                a, b = v["data_offsets"]
                fh.seek(base + a)
                raw = bytearray(fh.read(b - a))
                dt = {"BF16": torch.bfloat16, "F32": torch.float32}[v["dtype"]]
                out[k] = torch.frombuffer(raw, dtype=dt).reshape(v["shape"]).clone()
    return out


class RealCfg(Cfg):
    pass


def real_cfg(config_path):
    import json as _json

    c = _json.load(open(config_path))
    v = c["vision_config"]
    cfg = RealCfg()
    cfg.hidden_size = v["hidden_size"]
    cfg.num_attention_heads = v["num_attention_heads"]
    cfg.num_hidden_layers = v["num_hidden_layers"]
    cfg.intermediate_size = v["intermediate_size"]
    cfg.patch_size = v["patch_size"]
    cfg.patch_temporal = v["patch_temporal"]
    cfg.merge_kernel_size = v["merge_size"]
    cfg.pos_emb_height = v["pos_emb_height"]
    cfg.pos_emb_width = v["pos_emb_width"]
    cfg.adapter_dim = c["projector_hidden_size"]
    cfg.layer_norm_eps = v["layer_norm_eps"]
    cfg.layer_types = list(v["layer_types"])
    text = c["text_config"]
    cfg.text_hidden = text["hidden_size"]
    cfg.rms_norm_eps = text["rms_norm_eps"]
    # `normalize_tok_embeddings` defaults True when absent (configs/muse_glimmer.py:67)
    cfg.normalize_tok_embeddings = text.get("normalize_tok_embeddings", True)
    if cfg.normalize_tok_embeddings is None:
        cfg.normalize_tok_embeddings = True
    return cfg


def real_weights(st, cfg, dtype):
    w = Weights.__new__(Weights)
    t = "model.vision_tower."
    g = lambda k: st[k].to(torch.float32).to(dtype)  # noqa: E731
    w.conv1 = g(t + "patch_embedder.patch_embedding.weight")
    # The positional table stays f32-valued: upstream gathers it with f32
    # fractions and casts the SUM to the activation dtype (_get_pos_emb + `.to(dtype)`).
    w.pos_emb = st[t + "patch_embedder.position_embedding_table.weight"].to(torch.float32)
    w.ln_pre_w, w.ln_pre_b = g(t + "ln_pre.weight"), g(t + "ln_pre.bias")
    w.ln_post_w, w.ln_post_b = g(t + "ln_post.weight"), g(t + "ln_post.bias")
    w.blocks = []
    for l in range(cfg.num_hidden_layers):
        p = f"{t}layers.{l}."
        w.blocks.append(dict(
            ln1_w=g(p + "norm1.weight"), ln1_b=g(p + "norm1.bias"),
            ln2_w=g(p + "norm2.weight"), ln2_b=g(p + "norm2.bias"),
            qkv_w=torch.cat([g(p + "attn.q_proj.weight"), g(p + "attn.k_proj.weight"),
                             g(p + "attn.v_proj.weight")]),
            qkv_b=torch.cat([g(p + "attn.q_proj.bias"), g(p + "attn.k_proj.bias"),
                             g(p + "attn.v_proj.bias")]),
            o_w=g(p + "attn.proj.weight"), o_b=g(p + "attn.proj.bias"),
            fc_w=g(p + "mlp.fc1.weight"), fc_b=g(p + "mlp.fc1.bias"),
            proj_w=g(p + "mlp.fc2.weight"), proj_b=g(p + "mlp.fc2.bias")))
    w.ad_fc = g("model.vision_adapter.fc1.weight")
    w.ad_proj = g("model.vision_adapter.fc2.weight")
    w.projection = g("model.vision_projection.weight")
    return w


def real_pixels(h, wd):
    """A deterministic 3-channel pixel tensor in the normalized range: an LCG
    image in [0, 255] passed through the CLIP-style normalization the released
    processor_config declares. The tower gate does not need the processor."""
    raw = lcg(7777, 3 * h * wd, 1.0).view(1, 3, h, wd)
    img = (raw + 1.0) * 127.5
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return (img / 255.0 - mean) / std


def real_chain(cfg, w, pixels, dtype):
    capture = {}
    feats = encoder_forward(cfg, w, [pixels.to(dtype)], capture)
    adapted = adapter_forward(w, feats)
    soft = F.linear(adapted, w.projection)
    if cfg.normalize_tok_embeddings:
        x = soft.float()
        soft = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)).to(soft.dtype)
    return capture, feats, adapted, soft


def real_main(real_dir, out_dir, h, wd):
    cfg = real_cfg(os.path.join(real_dir, "config.json"))
    st = load_st_dir(real_dir)
    pixels = real_pixels(h, wd)
    os.makedirs(out_dir, exist_ok=True)
    pixels.reshape(-1).numpy().tofile(os.path.join(out_dir, "pixels_f32.bin"))
    for name, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        w = real_weights(st, cfg, dtype)
        cap, feats, adapted, soft = real_chain(cfg, w, pixels, dtype)
        cap["ln_pre"][0].float().numpy().tofile(os.path.join(out_dir, f"ln_pre_{name}.bin"))
        cap["block0"].float().numpy().tofile(os.path.join(out_dir, f"block0_{name}.bin"))
        feats.float().reshape(-1).numpy().tofile(os.path.join(out_dir, f"tower_{name}.bin"))
        adapted.float().reshape(-1).numpy().tofile(os.path.join(out_dir, f"adapter_{name}.bin"))
        soft.float().reshape(-1).numpy().tofile(os.path.join(out_dir, f"soft_{name}.bin"))
        print(name, "tower", tuple(feats.shape), "soft", tuple(soft.shape), flush=True)
    import json as _json
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        _json.dump({"height": h, "width": wd, "torch": torch.__version__,
                    "normalize_tok_embeddings": bool(cfg.normalize_tok_embeddings)}, fh)


def main():
    ap = argparse.ArgumentParser()
    default = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tests/vllm/models/muse_glimmer_vision_goldens.inc",
    )
    ap.add_argument("--out", default=default)
    ap.add_argument("--real-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--image-hw", nargs=2, type=int, default=[336, 448])
    args = ap.parse_args()
    if args.real_dir:
        real_main(args.real_dir, args.out_dir, args.image_hw[0], args.image_hw[1])
        return

    cfg, w = CFG, Weights(CFG)
    pixel_values = []
    for idx, (c, h, wd) in enumerate(IMAGES):
        pixel_values.append(lcg(5001 + idx, c * h * wd, 1.0).view(1, c, h, wd))

    capture: dict = {}
    tower = encoder_forward(cfg, w, pixel_values, capture)
    adapted = adapter_forward(w, tower)

    # A standalone pixel-shuffle probe on a known ramp: hidden[i, d] = i*10 + d.
    ramp = torch.arange(36 * 4, dtype=torch.float32).view(1, 36, 4)
    ramp_out = pixel_shuffle_downsample(cfg, ramp, 6, 6).squeeze(0).reshape(-1)

    with open(args.out, "w") as fh:
        fh.write(
            "// GENERATED by scripts/mm/muse_glimmer_vision_ref.py — DO NOT EDIT.\n"
            "// Muse Glimmer perception-encoder per-stage reference, transcribed\n"
            "// from vllm PR #51655 head 075d645af, muse_glimmer.py:555-1044.\n"
            "// NOT an oracle run: a torch transcription, not the vLLM runtime, so\n"
            "// this fixture establishes NUMERICS ONLY and licenses NO speed claim.\n"
            "#pragma once\n\n"
            "namespace muse_glimmer_vision_ref {\n\n"
        )
        for i in range(len(IMAGES)):
            emit(fh, f"kPatchify{i}", capture["patchify"][i].tolist())
            emit(fh, f"kPosEmb{i}", capture["pos_emb"][i].tolist())
            emit(fh, f"kRopeCos{i}", capture["rope_cos"][i].tolist())
            emit(fh, f"kRopeSin{i}", capture["rope_sin"][i].tolist())
            emit(fh, f"kLnPre{i}", capture["ln_pre"][i].tolist())
            emit_int(fh, f"kSparsePerm{i}", capture["perm"][i].tolist())
        emit_int(fh, "kSparseSeqLens", capture["seq_lens"])
        emit(fh, "kPixelShuffleRamp", ramp_out.tolist())
        emit(fh, "kBlock0", capture["block0"].tolist())
        emit(fh, "kTowerOut", tower.reshape(-1).tolist())
        emit(fh, "kAdapterOut", adapted.reshape(-1).tolist())
        fh.write("}  // namespace muse_glimmer_vision_ref\n")
    print(f"wrote {args.out}")
    print(f"tower {tuple(tower.shape)} adapter {tuple(adapted.shape)}")


if __name__ == "__main__":
    main()
