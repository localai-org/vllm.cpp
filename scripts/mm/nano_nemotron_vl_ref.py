#!/usr/bin/env python3
"""Nemotron Nano VL / Omni image-path REFERENCE (RADIO tower + mlp1 + processor).

Two modes.

1. Synthetic (default). Emits `tests/vllm/models/nano_nemotron_vl_goldens.inc`,
   the per-stage reference tensors the C++ gate
   (`tests/vllm/models/test_nano_nemotron_vl_vision.cpp`) compares against. The
   weights are an explicit LCG, rounded to bf16, that the C++ test reproduces
   bit-exactly, so no weight blob is committed, only reference OUTPUTS.

2. Real (`--real-weights PATH --out-dir DIR`). Loads the released
   `vision_model.*` and `mlp1.*` tensors (a subset file cut from
   `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` shard 1 at revision
   `e5e9932441de940c9a62185c870ea5bcd4cd24e2`), runs the same transcription in
   the production bf16 dtype on a deterministic image, and writes raw little-
   endian binaries the env-gated real arm of the C++ test reads.

WHERE THE REFERENCE COMES FROM
------------------------------
Every function below is a transcription of vLLM at the parity pin `e126687a9a`,
with vLLM's parallel-linear and MMEncoderAttention wrappers replaced by
`nn.functional.linear` and a per-segment SDPA (the block-diagonal mask
`cu_seqlens` builds):

  ViTPatchGenerator.forward / apply_pos_enc_dynamic /
    cls_token_dynamic / _get_pos_embeddings     radio.py:198-421
  RadioVisionEncoderLayer.forward              radio.py:493-505
  InternParallelAttention / InternMLP          intern_vit.py:145-290
  RadioInternVisionModel.forward (mask meta)   radio.py:579-644
  RadioModel._extract_final                    radio.py:745-774
  NemotronH_Nano_VL_V2.pixel_shuffle (v2)      nano_nemotron_vl.py:1012-1029
  pixel_shuffle_dynamic_res                    nano_nemotron_vl.py:1031-1048
  extract_feature_dynamic                      nano_nemotron_vl.py:1050-1058
  mlp1                                         nano_nemotron_vl.py:955-976
  DynamicResolutionImageTiler                  processors/nano_nemotron_vl.py:252-570
  _bicubic_resize_and_normalize                processors/nano_nemotron_vl.py:61-83

The resize and the positional interpolation call torch's own
`F.interpolate`, which IS the upstream code path; only the surrounding module
plumbing is transcribed.

Usage:
  python3 scripts/mm/nano_nemotron_vl_ref.py [--out PATH]
  python3 scripts/mm/nano_nemotron_vl_ref.py --real-weights W.safetensors \
      --out-dir DIR [--image-hw 333 517]
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import json

import torch
import torch.nn.functional as F

torch.use_deterministic_algorithms(True)
torch.set_num_threads(max(1, torch.get_num_threads()))


# ─── the LCG the C++ test reproduces bit-exactly ────────────────────────────
def lcg(seed: int, n: int, scale: float) -> torch.Tensor:
    out = []
    s = seed & 0xFFFFFFFF
    for _ in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        u = (s >> 8) / 16777216.0
        out.append((u * 2.0 - 1.0) * scale)
    # float64 -> float32 -> bf16 (round to nearest even), then back to f32 so
    # both arms see the SAME bf16-exact weights.
    return torch.tensor(out, dtype=torch.float64).to(torch.float32).to(torch.bfloat16).to(torch.float32)


def lcg_image(seed: int, h: int, w: int) -> torch.Tensor:
    s = seed & 0xFFFFFFFF
    vals = []
    for _ in range(h * w * 3):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        vals.append((s >> 24) & 0xFF)
    return torch.tensor(vals, dtype=torch.uint8).reshape(h, w, 3)


# ─── processor: DynamicResolutionImageTiler (processors/...:252-570) ─────────
class Tiler:
    PIXEL_SHUFFLE = True
    CONV_MERGING = False

    def __init__(self, *, max_model_len, patch_size, min_num_patches, max_num_patches,
                 downsample_ratio, norm_mean, norm_std, factor_max=1.0):
        self._patch_size = patch_size
        self._max_model_len = max_model_len
        self._min_num_patches = min_num_patches
        self._max_num_patches = max_num_patches if max_num_patches > 0 else float("inf")
        self._factor_max = factor_max
        self.norm_mean = torch.tensor(norm_mean).reshape(3, 1, 1)
        self.norm_std = torch.tensor(norm_std).reshape(3, 1, 1)
        self._downsample_ratio = int(1 / downsample_ratio) ** (self.PIXEL_SHUFFLE + self.CONV_MERGING)

    def _get_num_embeddings(self, width, height):
        num_patches = (width // self._patch_size) * (height // self._patch_size)
        return num_patches // (self._downsample_ratio ** 2)

    def max_num_tokens_available(self, text_prompt_length):
        return self._max_model_len - text_prompt_length - 4

    def process_media(self, orig_width, orig_height, num_tokens_available):
        current = num_tokens_available
        closest_patch_height = round(orig_height / self._patch_size + 0.5)
        closest_patch_width = round(orig_width / self._patch_size + 0.5)
        patches = closest_patch_height * closest_patch_width
        factor = min(math.sqrt(current / patches), self._factor_max)
        tph = math.floor(factor * closest_patch_height)
        tpw = math.floor(factor * closest_patch_width)
        if current > self._min_num_patches and tph * tpw < self._min_num_patches:
            up = math.sqrt(self._min_num_patches / (tph * tpw))
            tph = math.ceil(up * tph)
            tpw = math.ceil(up * tpw)
        d = 2
        rem_h = tph % d
        if rem_h != 0:
            inc = d - rem_h
            if (tph + inc) * tpw <= current:
                tph += inc
            else:
                tph = max(d, tph - rem_h)
        rem_w = tpw % d
        if rem_w != 0:
            inc = d - rem_w
            if tph * (tpw + inc) <= current:
                tpw += inc
            else:
                tpw = max(d, tpw - rem_w)
        num_embeddings = self._get_num_embeddings(tpw * self._patch_size, tph * self._patch_size)
        return (tpw, tph, num_embeddings), tpw * tph

    def compute_params(self, sizes_wh, num_tokens_available):
        num_tokens_available = num_tokens_available * 4
        num_tokens_available = max(num_tokens_available, self._min_num_patches * len(sizes_wh))
        per = [int(max(min(num_tokens_available, self._max_num_patches), self._min_num_patches))
               for _ in sizes_wh]
        for _ in range(10):
            params, counts = [], []
            for (w, h), t in zip(sizes_wh, per):
                p, c = self.process_media(w, h, t)
                params.append(p)
                counts.append(c)
            total = sum(counts)
            if total <= num_tokens_available:
                return params
            sf = num_tokens_available / total
            scaled = [max(self._min_num_patches, int(c * sf)) for c in counts]
            if not any(scaled[i] < per[i] for i in range(len(per))):
                per = [self._min_num_patches] * len(sizes_wh)
            else:
                per = scaled
        raise ValueError("unreachable")

    def apply(self, image_hwc_u8, tpw, tph, dtype):
        # _pil_to_nhwc_tensor + _bicubic_resize_and_normalize (:61-92)
        t = image_hwc_u8.unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
        t = F.interpolate(t, size=(tph * self._patch_size, tpw * self._patch_size),
                          mode="bicubic", align_corners=False, antialias=True)
        return ((t / 255.0 - self.norm_mean) / self.norm_std).to(dtype).contiguous()[0]


def patchify(img_chw: torch.Tensor, p: int) -> torch.Tensor:
    # DynamicResolutionImageTiler.stack.rearrange_img (:555-566):
    # "c (py yy) (px xx) -> (py px) (c yy xx)"
    c, h, w = img_chw.shape
    py, px = h // p, w // p
    return img_chw.reshape(c, py, p, px, p).permute(1, 3, 0, 2, 4).reshape(py * px, c * p * p)


# ─── RADIO tower (radio.py) ─────────────────────────────────────────────────
def get_pos_embeddings(pos_embed, num_rows, num_cols, input_dims, cpe_mode):
    # _get_pos_embeddings (:386-421)
    if (num_rows, num_cols) == input_dims:
        return pos_embed
    pe = pos_embed.reshape(1, num_rows, num_cols, -1).permute(0, 3, 1, 2)

    def window_select(pe):
        if input_dims[0] < pe.shape[-2]:
            pe = pe[..., : input_dims[0], :]
        if input_dims[1] < pe.shape[-1]:
            pe = pe[..., :, : input_dims[1]]
        return pe

    if cpe_mode:
        max_dim = max(input_dims)
        pe = F.interpolate(pe.float(), size=(max_dim, max_dim), align_corners=False,
                           mode="bilinear").to(pe.dtype)
        pe = window_select(pe)
    else:
        pe = window_select(pe)
    if pe.shape[-2:] != input_dims:
        pe = F.interpolate(pe.float(), size=input_dims, align_corners=False,
                           mode="bilinear").to(pe.dtype)
    return pe.flatten(2).permute(0, 2, 1)


def radio_forward(w, cfg, patches_flat, imgs_sizes, dtype):
    """RadioModel.forward on the dynamic path; returns the per-image feature
    rows with the CLS/register tokens stripped, concatenated [sum, hidden]."""
    P = cfg["patch_size"]
    H = cfg["hidden"]
    nh = cfg["heads"]
    hd = H // nh
    num_skip = cfg["num_cls"] + cfg["num_registers"]
    x = patches_flat.to(dtype).unsqueeze(0)
    # ViTPatchGenerator.forward with imgs_sizes (:201-206)
    patches = F.linear(x, w["embedder"].to(dtype))
    cur = 0
    out = []
    for (h, wd) in imgs_sizes:
        n = (h // P) * (wd // P)
        pe = get_pos_embeddings(w["pos_embed"].to(dtype), cfg["pos_rows"], cfg["pos_cols"],
                                (h // P, wd // P), cfg["cpe_mode"])
        seg = patches[:, cur:cur + n, :] + pe
        out.append(w["cls_token"].to(dtype).unsqueeze(0))
        out.append(seg)
        cur += n
    hs = torch.cat(out, dim=1)[0]
    seq_lens = [(h // P) * (wd // P) + num_skip for (h, wd) in imgs_sizes]
    eps = cfg["ln_eps"]
    for blk in w["blocks"]:
        # RadioVisionEncoderLayer.forward (:493-505); ls1 = ls2 = 1.0
        # (initializer_factor, intern_vit.py:323-324; the checkpoint ships none).
        n1 = F.layer_norm(hs, (H,), blk["norm1_w"].to(dtype), blk["norm1_b"].to(dtype), eps)
        qkv = F.linear(n1, blk["qkv_w"].to(dtype), blk["qkv_b"].to(dtype))
        q, k, v = qkv.chunk(3, dim=-1)
        att = torch.empty_like(q)
        off = 0
        for L in seq_lens:
            qs = q[off:off + L].reshape(L, nh, hd).transpose(0, 1)
            ks = k[off:off + L].reshape(L, nh, hd).transpose(0, 1)
            vs = v[off:off + L].reshape(L, nh, hd).transpose(0, 1)
            o = F.scaled_dot_product_attention(qs, ks, vs, scale=hd ** -0.5)
            att[off:off + L] = o.transpose(0, 1).reshape(L, H)
            off += L
        hs = hs + F.linear(att, blk["proj_w"].to(dtype), blk["proj_b"].to(dtype))
        n2 = F.layer_norm(hs, (H,), blk["norm2_w"].to(dtype), blk["norm2_b"].to(dtype), eps)
        f1 = F.gelu(F.linear(n2, blk["fc1_w"].to(dtype), blk["fc1_b"].to(dtype)))
        hs = hs + F.linear(f1, blk["fc2_w"].to(dtype), blk["fc2_b"].to(dtype))
    # _extract_final (:745-774): strip num_skip per image
    feats = []
    cur = 0
    for (h, wd) in imgs_sizes:
        n = (h // P) * (wd // P)
        feats.append(hs[cur + num_skip: cur + num_skip + n])
        cur += num_skip + n
    return torch.cat(feats, dim=0)


def pixel_shuffle_v2(x, scale_factor=0.5):
    # nano_nemotron_vl.py:1012-1029, ps_version "v2"
    n, h, w, c = x.size()
    r = int(1 / scale_factor)
    x = x.view(n, h // r, r, w // r, r, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(n, h // r, w // r, c * r * r)


def shuffle_dynamic(feats, imgs_sizes, P):
    # pixel_shuffle_dynamic_res (:1031-1048)
    out = []
    cur = 0
    for (h, wd) in imgs_sizes:
        gh, gw = h // P, wd // P
        sv = feats[cur:cur + gh * gw].reshape(1, gh, gw, -1)
        sv = pixel_shuffle_v2(sv).flatten(1, 2)[0]
        out.append(sv)
        cur += gh * gw
    return torch.cat(out, dim=0)


def mlp1(w, x, dtype):
    # nano_nemotron_vl.py:962-976: RMSNorm(eps 1e-5) -> Linear -> relu^2 -> Linear.
    # vLLM's RMSNorm.forward_native (layernorm.py) computes in f32 and casts back.
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xn = (xf * torch.rsqrt(var + 1e-5)).to(dtype) * w["norm_w"].to(dtype)
    h = F.linear(xn, w["fc1_w"].to(dtype))
    h = torch.square(F.relu(h))
    return F.linear(h, w["fc2_w"].to(dtype))


# ─── synthetic mode ─────────────────────────────────────────────────────────
SYN = dict(hidden=64, heads=4, layers=2, inter=128, patch_size=4, num_cls=4,
           num_registers=6, pos_rows=8, pos_cols=8, cpe_mode=True, ln_eps=1e-6,
           proj_hidden=96, llm_hidden=48)


def syn_weights():
    c = SYN
    H, I, P = c["hidden"], c["inter"], c["patch_size"]
    seed = 1000
    w = {}

    def nxt(n, scale):
        nonlocal seed
        seed += 1
        return lcg(seed, n, scale)

    w["embedder"] = nxt(H * 3 * P * P, 0.2).reshape(H, 3 * P * P)
    w["pos_embed"] = nxt(c["pos_rows"] * c["pos_cols"] * H, 0.5).reshape(1, c["pos_rows"] * c["pos_cols"], H)
    w["cls_token"] = nxt((c["num_cls"] + c["num_registers"]) * H, 0.5).reshape(-1, H)
    w["blocks"] = []
    for _ in range(c["layers"]):
        b = {}
        b["norm1_w"] = 1.0 + nxt(H, 0.1)
        b["norm1_b"] = nxt(H, 0.1)
        b["qkv_w"] = nxt(3 * H * H, 0.15).reshape(3 * H, H)
        b["qkv_b"] = nxt(3 * H, 0.1)
        b["proj_w"] = nxt(H * H, 0.15).reshape(H, H)
        b["proj_b"] = nxt(H, 0.1)
        b["norm2_w"] = 1.0 + nxt(H, 0.1)
        b["norm2_b"] = nxt(H, 0.1)
        b["fc1_w"] = nxt(I * H, 0.15).reshape(I, H)
        b["fc1_b"] = nxt(I, 0.1)
        b["fc2_w"] = nxt(H * I, 0.1).reshape(H, I)
        b["fc2_b"] = nxt(H, 0.1)
        # LayerNorm weights are 1 + small: round the SUM to bf16 so both sides
        # hold the same bf16-exact value.
        for k in ("norm1_w", "norm2_w"):
            b[k] = b[k].to(torch.bfloat16).to(torch.float32)
        w["blocks"].append(b)
    m = {}
    m["norm_w"] = (1.0 + nxt(4 * H, 0.1)).to(torch.bfloat16).to(torch.float32)
    m["fc1_w"] = nxt(c["proj_hidden"] * 4 * H, 0.1).reshape(c["proj_hidden"], 4 * H)
    m["fc2_w"] = nxt(c["llm_hidden"] * c["proj_hidden"], 0.1).reshape(c["llm_hidden"], c["proj_hidden"])
    return w, m


def emit(f, name, t):
    t = t.detach().to(torch.float32).reshape(-1).tolist()
    f.write(f"static const float {name}[{len(t)}] = {{\n")
    for i in range(0, len(t), 8):
        f.write("  " + ", ".join(f"{v:.9e}f" for v in t[i:i + 8]) + ",\n")
    f.write("};\n")


def emit_int(f, name, vals):
    f.write(f"static const int64_t {name}[{len(vals)}] = {{" + ", ".join(str(int(v)) for v in vals) + "};\n")


def synthetic(out_path):
    c = SYN
    P = c["patch_size"]
    w, m = syn_weights()
    tiler = Tiler(max_model_len=4096, patch_size=P, min_num_patches=16, max_num_patches=64,
                  downsample_ratio=0.5, norm_mean=[0.48145466, 0.4578275, 0.40821073],
                  norm_std=[0.26862954, 0.26130258, 0.27577711])
    # Two images: one downscaled (budget-bound, 50x70 -> 24x40), one upscaled
    # (min-patch bound, 9x6 -> 24x16).
    imgs = [lcg_image(7, 50, 70), lcg_image(9, 9, 6)]
    sizes_wh = [(im.shape[1], im.shape[0]) for im in imgs]
    text_len = 11
    avail = tiler.max_num_tokens_available(text_len)
    params = tiler.compute_params(sizes_wh, avail)
    pix_bf16, pix_f32, imgs_sizes = [], [], []
    for im, (tpw, tph, _) in zip(imgs, params):
        pix_f32.append(tiler.apply(im, tpw, tph, torch.float32))
        pix_bf16.append(tiler.apply(im, tpw, tph, torch.bfloat16))
        imgs_sizes.append((tph * P, tpw * P))
    flat = torch.cat([patchify(p.float(), P) for p in pix_bf16], dim=0)

    feats32 = radio_forward(w, c, flat, imgs_sizes, torch.float32)
    feats16 = radio_forward(w, c, flat, imgs_sizes, torch.bfloat16)
    sh32 = shuffle_dynamic(feats32, imgs_sizes, P)
    sh16 = shuffle_dynamic(feats16.to(torch.bfloat16), imgs_sizes, P)
    proj32 = mlp1(m, sh32, torch.float32)
    proj16 = mlp1(m, sh16, torch.bfloat16)
    # A second, non-square CPE case for the positional table alone.
    pe_ns = get_pos_embeddings(w["pos_embed"], c["pos_rows"], c["pos_cols"], (5, 11), True)

    with open(out_path, "w") as f:
        f.write("// GENERATED by scripts/mm/nano_nemotron_vl_ref.py -- do not edit.\n")
        f.write("// Reference: vLLM e126687a9a radio.py / nano_nemotron_vl.py /\n")
        f.write("// processors/nano_nemotron_vl.py, transcribed in torch "
                f"{torch.__version__}.\n#pragma once\n#include <cstdint>\n")
        f.write(f"static const int64_t kNnvlTextLen = {text_len};\n")
        emit_int(f, "kNnvlImgHW", [v for im in imgs for v in (im.shape[0], im.shape[1])])
        emit_int(f, "kNnvlParams", [v for p in params for v in p])
        for i, (a, b) in enumerate(zip(pix_f32, pix_bf16)):
            emit(f, f"kNnvlPixF32_{i}", a)
            emit(f, f"kNnvlPixBf16_{i}", b.float())
        emit(f, "kNnvlPosNonSquare", pe_ns)
        emit(f, "kNnvlFeatsF32", feats32)
        emit(f, "kNnvlFeatsBf16", feats16.float())
        emit(f, "kNnvlProjF32", proj32)
        emit(f, "kNnvlProjBf16", proj16.float())
    print("wrote", out_path, "params", params)


# ─── real mode ──────────────────────────────────────────────────────────────
def load_st(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            fh.seek(base + a)
            raw = bytearray(fh.read(b - a))
            dt = {"BF16": torch.bfloat16, "F32": torch.float32}[v["dtype"]]
            out[k] = torch.frombuffer(raw, dtype=dt).reshape(v["shape"]).clone()
    return out


def real(weights_path, out_dir, img_h, img_w, text_len, max_model_len):
    st = load_st(weights_path)
    pre = "vision_model.radio_model.model."
    w = {"embedder": st[pre + "patch_generator.embedder.weight"],
         "pos_embed": st[pre + "patch_generator.pos_embed"],
         "cls_token": st[pre + "patch_generator.cls_token.token"], "blocks": []}
    for i in range(32):
        b = pre + f"blocks.{i}."
        w["blocks"].append({
            "norm1_w": st[b + "norm1.weight"], "norm1_b": st[b + "norm1.bias"],
            "qkv_w": st[b + "attn.qkv.weight"], "qkv_b": st[b + "attn.qkv.bias"],
            "proj_w": st[b + "attn.proj.weight"], "proj_b": st[b + "attn.proj.bias"],
            "norm2_w": st[b + "norm2.weight"], "norm2_b": st[b + "norm2.bias"],
            "fc1_w": st[b + "mlp.fc1.weight"], "fc1_b": st[b + "mlp.fc1.bias"],
            "fc2_w": st[b + "mlp.fc2.weight"], "fc2_b": st[b + "mlp.fc2.bias"]})
    m = {"norm_w": st["mlp1.0.weight"], "fc1_w": st["mlp1.1.weight"], "fc2_w": st["mlp1.3.weight"]}
    cfg = dict(hidden=1280, heads=16, patch_size=16, num_cls=4, num_registers=6,
               pos_rows=128, pos_cols=128, cpe_mode=True, ln_eps=1e-6)
    tiler = Tiler(max_model_len=max_model_len, patch_size=16, min_num_patches=1024,
                  max_num_patches=13312, downsample_ratio=0.5,
                  norm_mean=[0.48145466, 0.4578275, 0.40821073],
                  norm_std=[0.26862954, 0.26130258, 0.27577711])
    im = lcg_image(11, img_h, img_w)
    avail = tiler.max_num_tokens_available(text_len)
    (tpw, tph, n_emb), = tiler.compute_params([(img_w, img_h)], avail)
    pix = tiler.apply(im, tpw, tph, torch.bfloat16)
    sizes = [(tph * 16, tpw * 16)]
    flat = patchify(pix.float(), 16)
    feats = radio_forward(w, cfg, flat, sizes, torch.bfloat16).to(torch.bfloat16)
    sh = shuffle_dynamic(feats, sizes, 16)
    proj = mlp1(m, sh, torch.bfloat16)
    # The f32 arm: the same bf16 weights and bf16 pixels, f32 arithmetic. It
    # separates an arithmetic defect from the bf16 envelope, which the
    # production arm alone cannot do on a 32-block tower.
    w32 = {k: (v.float() if torch.is_tensor(v) else [{kk: vv.float() for kk, vv in b.items()} for b in v])
           for k, v in w.items()}
    m32 = {k: v.float() for k, v in m.items()}
    feats32 = radio_forward(w32, cfg, flat, sizes, torch.float32)
    proj32 = mlp1(m32, shuffle_dynamic(feats32, sizes, 16), torch.float32)
    os.makedirs(out_dir, exist_ok=True)
    im.numpy().tofile(os.path.join(out_dir, "image_u8.bin"))
    pix.float().numpy().tofile(os.path.join(out_dir, "pixels_f32.bin"))
    feats.float().numpy().tofile(os.path.join(out_dir, "feats_f32.bin"))
    proj.float().numpy().tofile(os.path.join(out_dir, "proj_f32.bin"))
    feats32.numpy().tofile(os.path.join(out_dir, "feats_f32arm.bin"))
    proj32.numpy().tofile(os.path.join(out_dir, "proj_f32arm.bin"))
    meta = dict(img_h=img_h, img_w=img_w, text_len=text_len, max_model_len=max_model_len,
                grid_w=tpw, grid_h=tph, num_embeddings=n_emb, torch=torch.__version__)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    print(json.dumps(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "tests",
                                                  "vllm", "models", "nano_nemotron_vl_goldens.inc"))
    ap.add_argument("--real-weights")
    ap.add_argument("--out-dir")
    ap.add_argument("--image-hw", nargs=2, type=int, default=[333, 517])
    ap.add_argument("--text-len", type=int, default=9)
    ap.add_argument("--max-model-len", type=int, default=16384)
    a = ap.parse_args()
    if a.real_weights:
        real(a.real_weights, a.out_dir, a.image_hw[0], a.image_hw[1], a.text_len, a.max_model_len)
    else:
        synthetic(os.path.normpath(a.out))


if __name__ == "__main__":
    main()
