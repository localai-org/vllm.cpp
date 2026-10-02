#!/usr/bin/env python3
"""Cohere2MoeForCausalLM (North) REFERENCE transcription.

Two modes.

1. Synthetic (default). Emits `tests/vllm/models/cohere2_moe_goldens.inc`, the
   logits the C++ gate (`tests/vllm/models/test_cohere2_moe.cpp`) compares
   against, for four tiny configs that between them switch every mechanism
   of the spec on and off. The weights are an explicit LCG rounded to bf16 that
   the C++ test reproduces bit-exactly, so no weight blob is committed. Each
   config has an f32 arm (bf16-valued weights, f32 arithmetic: the tight gate)
   and a bf16 arm (the model-dtype stores of the model file: the production
   envelope). Config A also carries a greedy continuation for the runner gate.

2. Real (`--real DIR`). Reads the North-Mini-Code-1.0 tensor subset fetched by
   HTTP range into DIR (layers 0, 1, 4 and `model.norm`, from
   `CohereLabs/North-Mini-Code-1.0` @ d11e61a842617a22dc328552fa5bb86231ee4f37)
   plus DIR/config.json, runs each of those decoder layers and the final norm on
   one deterministic hidden-state input, and writes raw little-endian binaries
   the env-gated real arm of the C++ test reads (`VLLM_COHERE2_MOE_REAL_DIR`).

WHERE THE REFERENCE COMES FROM
------------------------------
A transcription of vLLM at the parity pin `a7c23ac96d`,
`vllm/model_executor/models/cohere2_moe.py`, with vLLM's parallel linears
replaced by plain matmuls and `Attention` by an explicit masked softmax:

  token_choice_with_bias              cohere2_moe.py:56-72
  rms_norm_func                       cohere2_moe.py:75-82
  layer_norm_func                     commandr.py:68-75 (imported at :37)
  select_norm_impl                    cohere2_moe.py:97-103
  Cohere2MoeMLP.forward               cohere2_moe.py:142-146
  Cohere2MoeAttention (window, RoPE)  cohere2_moe.py:205-246
  Cohere2Moe.forward                  cohere2_moe.py:319-328
  Cohere2MoeDecoderLayer.forward      cohere2_moe.py:369-384
  mlp_layer_types normalization       cohere2_moe.py:410-418
  compute_logits (tied, logit_scale)  cohere2_moe.py:510-513, :537-541
  RotaryEmbedding cache / GPT-J apply rotary_embedding/base.py:94-103,
                                      rotary_embedding/common.py:134-170
  per-layer window W -> FA (W-1, 0)   v1/attention/backends/flash_attn.py:1076-1080
  softmax top-k (expert_selection_fn != "sigmoid"): FusedMoE's default
                                      fused_topk (softmax, top-k, renormalize)

Config defaults for absent keys are those of transformers' `Cohere2MoeConfig`
(the class vLLM instantiates), including its layer_types derivation.

The bf16 arm rounds at the model file's stores. Kernel-internal accumulation
(the fused-MoE top-k sum, the attention softmax) is carried in f32, which is
what the C++ path does too; it is an envelope, not a mechanism. The logits are
f32 on both sides (the C++ carrier is f32).

Usage:
  ~/venvs/music3-oracle/bin/python scripts/cohere2-moe-ref.py [--out PATH]
  ~/venvs/music3-oracle/bin/python scripts/cohere2-moe-ref.py --real DIR --out-dir DIR2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct

import torch
import torch.nn.functional as F

torch.use_deterministic_algorithms(True)


# --- the deterministic synthetic-weight LCG (mirrored in the C++ test) --------
def lcg(seed: int, n: int, scale: float) -> torch.Tensor:
    s = seed & 0xFFFFFFFF
    out = []
    for _ in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out.append(((((s >> 8) / 16777216.0) * 2.0) - 1.0) * scale)
    return torch.tensor(out, dtype=torch.float32)


# --- config resolution (transformers Cohere2MoeConfig defaults) ---------------
class Params:
    def __init__(self, raw: dict):
        g = raw.get
        self.H = raw["hidden_size"]
        self.L = raw["num_hidden_layers"]
        self.V = raw["vocab_size"]
        self.nh = raw["num_attention_heads"]
        self.nkv = g("num_key_value_heads") or self.nh
        self.hd = g("head_dim", 128)
        self.I = raw["intermediate_size"]
        self.Ip = g("prefix_dense_intermediate_size") or self.I
        self.E = g("num_experts", 8)
        self.K = g("num_experts_per_tok", 2)
        self.ns = g("num_shared_experts", 0)
        self.avg = self.ns > 0 and g("shared_expert_combination_strategy", "average") == "average"
        self.sigmoid = g("expert_selection_fn", "softmax") == "sigmoid"
        self.renorm = g("norm_topk_prob", True)
        self.rms = g("rms_norm_eps") is not None
        self.eps = g("rms_norm_eps") if self.rms else g("layer_norm_eps", 1e-5)
        self.logit_scale = g("logit_scale", 0.0625)
        self.theta = g("rope_theta", 10000.0)
        self.maxpos = g("model_max_length") or g("max_position_embeddings", 8192)
        fk = g("first_k_dense_replace", 0) or 0
        pp = g("prefix_dense_sliding_window_pattern", 1)
        lt = g("layer_types")
        if lt is None:  # Cohere2MoeConfig.__post_init__
            swp = g("sliding_window_pattern", 4)
            lt = ["sliding_attention" if ((i + 1) % pp) != 0 else "full_attention" for i in range(fk)]
            lt += ["sliding_attention" if ((i + 1) % swp) != 0 else "full_attention"
                   for i in range(self.L - fk)]
        mt = g("mlp_layer_types")
        if mt is None:  # cohere2_moe.py:410-418
            mt = ["dense"] * fk + ["sparse"] * (self.L - fk)
        sw = raw["sliding_window"] if "sliding_window" in raw else 4096
        sw = int(sw) if sw is not None else None
        self.window, self.rope, self.dense = [], [], []
        for l in range(self.L):
            w = sw + 1 if lt[l] == "sliding_attention" else None       # :205-211
            prefix = all(t == "dense" for t in mt[: l + 1])            # :49-53
            force = prefix and pp == 1                                 # :216-222
            self.window.append(w)
            self.rope.append(bool(w) or force)                         # :242
            self.dense.append(mt[l] == "dense")


def tensor_specs(p: Params):
    """(name, shape, kind) in the C++ EnumerateCohere2MoeTensors order."""
    q, kv = p.nh * p.hd, p.nkv * p.hd
    out = [("model.embed_tokens.weight", [p.V, p.H], "embed")]

    def mlp(prefix, inter):
        out.append((prefix + "gate_proj.weight", [inter, p.H], "linear"))
        out.append((prefix + "up_proj.weight", [inter, p.H], "linear"))
        out.append((prefix + "down_proj.weight", [p.H, inter], "linear"))

    for l in range(p.L):
        b = f"model.layers.{l}."
        out.append((b + "input_layernorm.weight", [p.H], "norm"))
        out.append((b + "self_attn.q_proj.weight", [q, p.H], "linear"))
        out.append((b + "self_attn.k_proj.weight", [kv, p.H], "linear"))
        out.append((b + "self_attn.v_proj.weight", [kv, p.H], "linear"))
        out.append((b + "self_attn.o_proj.weight", [p.H, q], "linear"))
        if p.dense[l]:
            mlp(b + "mlp.", p.Ip)
            continue
        out.append((b + "mlp.gate.weight", [p.E, p.H], "router"))
        for e in range(p.E):
            mlp(b + f"mlp.experts.{e}.", p.I)
        if p.ns > 0:
            mlp(b + "mlp.shared_experts.", p.I * p.ns)
    out.append(("model.norm.weight", [p.H], "norm"))
    return out


def synth_weights(p: Params):
    ws = {}
    for i, (name, shape, kind) in enumerate(tensor_specs(p)):
        n = math.prod(shape)
        seed = 1000 + 7919 * i
        if kind == "norm":
            v = 1.0 + lcg(seed, n, 0.25)
        elif kind == "embed":
            v = lcg(seed, n, 1.0)
        elif kind == "router":
            v = lcg(seed, n, 3.0 / math.sqrt(shape[1]))
        else:
            v = lcg(seed, n, 1.5 / math.sqrt(shape[1]))
        ws[name] = v.reshape(shape).to(torch.bfloat16).to(torch.float32)
    return ws


# --- the transcription ---------------------------------------------------------
def rnd(x, dt):
    return x.to(dt)


def linear(x, w, dt):
    return rnd(x.float() @ w.float().t(), dt)


def norm(x, w, p, dt):
    h = x.float()
    if p.rms:  # rms_norm_func
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + p.eps)
    else:  # layer_norm_func
        mean = h.mean(-1, keepdim=True)
        var = (h - mean).pow(2).mean(-1, keepdim=True)
        h = (h - mean) * torch.rsqrt(var + p.eps)
    return rnd(w.float() * h, dt)


def rope_cache(p: Params, rows: int, dt):
    inv_freq = 1.0 / (p.theta ** (torch.arange(0, p.hd, 2, dtype=torch.float) / p.hd))
    t = torch.arange(rows, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dt)


def rope_gptj(x, positions, cache):
    cs = cache.index_select(0, positions)
    cos, sin = cs.chunk(2, dim=-1)
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def attention(q, k, v, window, scale):
    T, nh, hd = q.shape
    nkv = k.shape[1]
    g = nh // nkv
    kk = k.float().repeat_interleave(g, dim=1)
    vv = v.float().repeat_interleave(g, dim=1)
    s = torch.einsum("ihd,jhd->hij", q.float(), kk) * scale
    i = torch.arange(T).unsqueeze(1)
    j = torch.arange(T).unsqueeze(0)
    mask = j > i
    if window is not None:  # FA window (W-1, 0): i - j <= W - 1
        mask = mask | ((i - j) > (window - 1))
    s = s.masked_fill(mask.unsqueeze(0), float("-inf"))
    pr = torch.softmax(s, dim=-1)
    return torch.einsum("hij,jhd->ihd", pr, vv)


def attn_block(x, ws, b, l, p, positions, cache, dt):
    q = linear(x, ws[b + "self_attn.q_proj.weight"], dt).view(-1, p.nh, p.hd)
    k = linear(x, ws[b + "self_attn.k_proj.weight"], dt).view(-1, p.nkv, p.hd)
    v = linear(x, ws[b + "self_attn.v_proj.weight"], dt).view(-1, p.nkv, p.hd)
    if p.rope[l]:
        q = rope_gptj(q, positions, cache)
        k = rope_gptj(k, positions, cache)
    o = rnd(attention(q, k, v, p.window[l], p.hd ** -0.5), dt)
    return linear(o.reshape(-1, p.nh * p.hd), ws[b + "self_attn.o_proj.weight"], dt)


def mlp_block(x, ws, prefix, dt):
    g = linear(x, ws[prefix + "gate_proj.weight"], dt)
    u = linear(x, ws[prefix + "up_proj.weight"], dt)
    a = rnd(F.silu(g.float()) * u.float(), dt)  # SiluAndMul
    return linear(a, ws[prefix + "down_proj.weight"], dt)


def moe_block(x, ws, b, p, dt):
    logits = linear(x, ws[b + "mlp.gate.weight"], dt)
    if p.sigmoid:  # token_choice_with_bias
        scores = logits.float().sigmoid()
    else:  # FusedMoE default fused_topk
        scores = torch.softmax(logits.float(), dim=-1)
    # torch.topk leaves the order of EXACT ties unspecified; bf16 router logits
    # produce them at the k-th boundary on these configs. The C++ router keeps
    # the house rule (lowest expert index wins a tie, ops.h MoeRouterTopK
    # "DETERMINISM DEVIATION"), so the reference selects the same way.
    tw, ti = torch.sort(scores, dim=-1, descending=True, stable=True)
    tw, ti = tw[:, : p.K], ti[:, : p.K]
    if p.renorm:
        tw = tw / tw.sum(dim=-1, keepdim=True)
    routed = torch.zeros(x.shape[0], p.H, dtype=torch.float32)
    for t in range(x.shape[0]):
        for j in range(p.K):
            e = int(ti[t, j])
            y = mlp_block(x[t:t + 1], ws, b + f"mlp.experts.{e}.", dt)
            routed[t] += tw[t, j] * y[0].float()
    out = rnd(routed, dt)
    if p.ns > 0:
        out = rnd(out + mlp_block(x, ws, b + "mlp.shared_experts.", dt), dt)
        if p.avg:
            out = out / 2
    return out


def decoder_layer(h, ws, l, p, positions, cache, dt):
    b = f"model.layers.{l}."
    n = norm(h, ws[b + "input_layernorm.weight"], p, dt)
    a = attn_block(n, ws, b, l, p, positions, cache, dt)
    m = mlp_block(n, ws, b + "mlp.", dt) if p.dense[l] else moe_block(n, ws, b, p, dt)
    return rnd(rnd(h + a, dt) + m, dt)


def forward(tokens, ws, p, dt):
    positions = torch.arange(len(tokens))
    cache = rope_cache(p, len(tokens), dt)
    h = rnd(ws["model.embed_tokens.weight"][torch.tensor(tokens)], dt)
    for l in range(p.L):
        h = decoder_layer(h, ws, l, p, positions, cache, dt)
    h = norm(h, ws["model.norm.weight"], p, dt)
    return (h.float() @ ws["model.embed_tokens.weight"].float().t()) * p.logit_scale


# --- the synthetic configs ---------------------------------------------------
BASE = {
    "architectures": ["Cohere2MoeForCausalLM"],
    "model_type": "cohere2_moe",
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 16,
    "num_experts": 8,
    "num_experts_per_tok": 3,
    "vocab_size": 96,
    "max_position_embeddings": 64,
    "rope_theta": 100.0,
    "sliding_window": 3,
    "dtype": "bfloat16",
}
CONFIGS = {
    # North's own switches: RMSNorm (both eps keys present), sigmoid without
    # renormalization, no shared expert, one forced-RoPE dense prefix layer,
    # interleaved window, NoPE full MoE layers, logit_scale != 1.
    "a": dict(BASE, num_hidden_layers=5, first_k_dense_replace=1,
              prefix_dense_intermediate_size=48, prefix_dense_sliding_window_pattern=1,
              layer_types=["full_attention", "sliding_attention", "sliding_attention",
                           "full_attention", "sliding_attention"],
              rms_norm_eps=1e-6, layer_norm_eps=1e-5, expert_selection_fn="sigmoid",
              norm_topk_prob=False, num_shared_experts=0,
              shared_expert_combination_strategy="average", logit_scale=0.5,
              use_qk_norm=False),
    # LayerNorm (no rms_norm_eps), renormalized sigmoid, two shared experts
    # averaged, derived layer_types with a NoPE full prefix-dense layer
    # (pattern 2), explicit mlp_layer_types.
    "b": dict(BASE, num_hidden_layers=4, first_k_dense_replace=2,
              prefix_dense_intermediate_size=40, prefix_dense_sliding_window_pattern=2,
              mlp_layer_types=["dense", "dense", "sparse", "sparse"],
              layer_norm_eps=1e-5, expert_selection_fn="sigmoid", norm_topk_prob=True,
              num_shared_experts=2, shared_expert_combination_strategy="average",
              logit_scale=0.75),
    # Softmax router (expert_selection_fn absent), norm_topk_prob and
    # logit_scale absent (class defaults), one shared expert summed, no dense
    # prefix.
    "c": dict(BASE, num_hidden_layers=3,
              layer_types=["sliding_attention", "full_attention", "sliding_attention"],
              rms_norm_eps=1e-6, num_shared_experts=1,
              shared_expert_combination_strategy="sum"),
    # A NON-contiguous dense layer: mlp_layer_types dense, sparse, dense with
    # pattern 1. Layer 0 is the dense prefix (forced RoPE); layer 2 is dense but
    # not in the contiguous prefix (is_prefix_dense_layer, cohere2_moe.py:49-53),
    # so as a full-attention layer it is NoPE. sliding_window given as 3.0.
    "d": dict(BASE, num_hidden_layers=4, sliding_window=3.0,
              prefix_dense_intermediate_size=24, prefix_dense_sliding_window_pattern=1,
              mlp_layer_types=["dense", "sparse", "dense", "sparse"],
              layer_types=["full_attention", "sliding_attention", "full_attention",
                           "sliding_attention"],
              rms_norm_eps=1e-6, expert_selection_fn="sigmoid", norm_topk_prob=False,
              logit_scale=0.5),
}
TOKENS = [5, 17, 42, 8, 61, 3, 88, 29, 71, 14, 50]
GREEDY_PROMPT = [70, 29, 44, 29, 86, 28]
GREEDY_STEPS = 8


def c_floats(name, t):
    vals = ", ".join(f"{float(x):.9e}f" for x in t.flatten().tolist())
    return f"inline constexpr float {name}[] = {{{vals}}};\n"


def c_ints(name, xs):
    return f"inline constexpr int32_t {name}[] = {{{', '.join(str(int(x)) for x in xs)}}};\n"


def synthetic(out_path):
    lines = ["// GENERATED by scripts/cohere2-moe-ref.py; do not edit.\n",
             "// Reference logits of the pinned vLLM cohere2_moe.py (a7c23ac96d)\n",
             "// transcription on LCG weights the C++ test regenerates bit-exactly.\n",
             "#pragma once\n#include <cstdint>\n\nnamespace c2m_golden {\n\n",
             c_ints("kTokens", TOKENS)]
    for key, raw in CONFIGS.items():
        p = Params(raw)
        ws = synth_weights(p)
        f32 = forward(TOKENS, ws, p, torch.float32)
        b16 = forward(TOKENS, ws, p, torch.bfloat16)
        rel = (b16 - f32).abs().max().item() / f32.abs().max().item()
        print(f"config {key}: max|logit| {f32.abs().max().item():.4f}, bf16 vs f32 rel {rel:.3e}")
        lines.append(f'inline constexpr char kConfig_{key}[] = R"JSON({json.dumps(raw)})JSON";\n')
        lines.append(c_floats(f"kLogitsF32_{key}", f32))
        lines.append(c_floats(f"kLogitsBf16_{key}", b16))
    # Greedy continuation on config a, bf16 arm (the runner's production dtype).
    p = Params(CONFIGS["a"])
    ws = synth_weights(p)
    seq = list(GREEDY_PROMPT)
    gaps = []
    for _ in range(GREEDY_STEPS):
        lg = forward(seq, ws, p, torch.bfloat16)[-1]
        top = torch.topk(lg, 2)
        gaps.append(float(top.values[0] - top.values[1]))
        seq.append(int(top.indices[0]))
    print("greedy", seq[len(GREEDY_PROMPT):], "min top1-top2 gap", min(gaps))
    lines.append(c_ints("kGreedyPrompt", GREEDY_PROMPT))
    lines.append(c_ints("kGreedyOut", seq[len(GREEDY_PROMPT):]))
    lines.append("\n}  // namespace c2m_golden\n")
    with open(out_path, "w") as f:
        f.writelines(lines)
    print("wrote", out_path)


# --- the real-tensor mode --------------------------------------------------------
REAL_LAYERS = [0, 1, 4]
REAL_T = 24
REAL_POS0 = 1000


def real(src_dir, out_dir):
    from safetensors import safe_open

    raw = json.load(open(os.path.join(src_dir, "config.json")))
    p = Params(raw)
    ws = {}
    for fn in sorted(os.listdir(src_dir)):
        if fn.startswith("sub_") and fn.endswith(".safetensors"):
            with safe_open(os.path.join(src_dir, fn), "pt") as f:
                for k in f.keys():
                    ws[k] = f.get_tensor(k).float()
    os.makedirs(out_dir, exist_ok=True)
    hidden = lcg(20260930, REAL_T * p.H, 0.5).reshape(REAL_T, p.H).to(torch.bfloat16).float()
    positions = torch.arange(REAL_POS0, REAL_POS0 + REAL_T)

    def dump(name, t, fmt="f"):
        with open(os.path.join(out_dir, name), "wb") as f:
            flat = t.flatten().tolist()
            f.write(struct.pack(f"<{len(flat)}{fmt}", *flat))

    dump("hidden_in.f32", hidden)
    dump("positions.i32", positions, "i")
    for dt, tag in ((torch.float32, "f32"), (torch.bfloat16, "bf16")):
        cache = rope_cache(p, REAL_POS0 + REAL_T, dt)
        for l in REAL_LAYERS:
            y = decoder_layer(hidden.to(dt), ws, l, p, positions, cache, dt).float()
            dump(f"layer{l}_{tag}.f32", y)
            print(f"layer {l} ({tag}): max|out| {y.abs().max().item():.4f}")
        dump(f"norm_{tag}.f32", norm(hidden.to(dt), ws["model.norm.weight"], p, dt).float())
    print("wrote", out_dir)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "..", "tests", "vllm", "models",
                                                  "cohere2_moe_goldens.inc"))
    ap.add_argument("--real", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if a.real:
        real(a.real, a.out_dir or a.real)
    else:
        synthetic(os.path.normpath(a.out))


if __name__ == "__main__":
    main()
