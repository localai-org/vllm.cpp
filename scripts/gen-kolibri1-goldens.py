#!/usr/bin/env python3
"""Generate the Kolibri-1 golden run (MODEL-TEXT-kolibri-1 W3).

Runs greedy decode (temperature 0, 32 tokens, seed 0) over a fixed 8-prompt
set against the DEQUANTIZED bf16 reference checkpoint
(/mnt/models/Aleph-Alpha/Kolibri-1-bf16) and writes compact per-prompt
fingerprints (input ids, generated ids, final-step top-k logits and their
sums) to tests/vllm/models/kolibri1_goldens.json.

PROVENANCE AND ADAPTATION (spec risk R4): `transformers` has no native
Kolibri1ForCausalLM, and the pinned plugin's modeling file
(aleph_alpha_inference/kolibri1.py, pin 049a6a7bd240, Apache-2.0,
SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH) imports vLLM
throughout, so it cannot run under a plain torch/transformers stack. This
script therefore transcribes the plugin's math — the same transcription the
W2 scalar reference (tests/vllm/models/test_kolibri1_w2.cpp) verified line by
line — into pure torch: GQA with per-head qk-norm, RoPE on sliding layers
only (RNoPE on full layers), sandwich norms, sigmoid-logit-add routing
(top-k on fp32 logits + e_score_correction_bias, weights = ungated sigmoid,
norm_topk_prob false), and an ungated shared expert added on every layer.
No KV cache: each decode step recomputes the full sequence, which is
numerically identical to incremental decoding for this fixed math.

Usage:
  python3 scripts/gen-kolibri1-goldens.py /mnt/models/Aleph-Alpha/Kolibri-1-bf16 \
      tests/vllm/models/kolibri1_goldens.json
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH (transcribed math)

import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import safe_open
from tokenizers import Tokenizer

MAX_NEW_TOKENS = 32
TOP_K = 8
SEED = 0

# 4 synthetic word-pattern prompts (the bench generator's style) + 4 natural
# prompts, one German (the model is English-German).
PROMPTS = [
    "alpha beta gamma delta epsilon zeta",
    "der Mond scheint hell über den Bergen",
    "The capital of Australia is",
    "Wissen ist Macht, aber Zeit ist Geld",
    "x1 = 3, x2 = 7, x3 = 11, x4 =",
    "Translation to German: the weather is beautiful today",
    "Ein gutes Buch liest man am besten",
    "counting: one two three four five six seven",
]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """vLLM RMSNorm: fp32 variance, fp32 multiply, cast back."""
    var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    xn = x.to(torch.float32) * torch.rsqrt(var + eps)
    return (xn * weight.to(torch.float32)).to(x.dtype)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float):
    inv = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    angles = positions.to(torch.float32)[:, None] * inv[None, :]
    return torch.cos(angles), torch.sin(angles)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class Kolibri1Ref(torch.nn.Module):
    """Pure-torch transcription of the pinned plugin's kolibri1.py math."""

    def __init__(self, model_dir: Path):
        super().__init__()
        cfg = json.loads((model_dir / "config.json").read_text())
        self.cfg = cfg
        self.eps = cfg["rms_norm_eps"]
        self.head_dim = cfg["head_dim"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.window = cfg["sliding_window"]
        self.theta = cfg["rope_theta"]
        self.n_experts = cfg["num_experts"]
        self.topk = cfg["num_experts_per_tok"]
        self.renormalize = cfg["norm_topk_prob"]
        self.n_layers = cfg["num_hidden_layers"]
        self.layer_types = cfg["layer_types"]
        self.files = {}
        self.index = json.loads(
            (model_dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self._cache: dict[str, torch.Tensor] = {}
        self.dir = model_dir

    def t(self, name: str) -> torch.Tensor:
        if name not in self._cache:
            path = self.dir / self.index[name]
            if path not in self.files:
                self.files[path] = safe_open(
                    path if path.is_absolute() else path, framework="pt"
                )
            self._cache[name] = self.files[path].get_tensor(name)
        return self._cache[name]

    def linear(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        w = self.t(prefix + ".weight")  # bf16 [out, in]
        return torch.nn.functional.linear(x, w)

    def attention(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """h: [T, hidden]. Returns [T, hidden]."""
        p = f"model.layers.{layer}.self_attn"
        t_len = h.shape[0]
        qkv = torch.cat(
            [
                self.linear(h, p + ".q_proj"),
                self.linear(h, p + ".k_proj"),
                self.linear(h, p + ".v_proj"),
            ],
            dim=-1,
        )
        q = qkv[:, : self.n_heads * self.head_dim]
        k = qkv[
            :,
            self.n_heads * self.head_dim : (self.n_heads + self.n_kv) * self.head_dim,
        ]
        v = qkv[:, (self.n_heads + self.n_kv) * self.head_dim :]
        q = q.view(t_len, self.n_heads, self.head_dim)
        k = k.view(t_len, self.n_kv, self.head_dim)
        v = v.view(t_len, self.n_kv, self.head_dim)
        q = rms_norm(q, self.t(p + ".q_norm.weight"), self.eps)
        k = rms_norm(k, self.t(p + ".k_norm.weight"), self.eps)

        positions = torch.arange(t_len)
        if self.layer_types[layer] == "sliding_attention":
            cos, sin = rope_cos_sin(positions, self.head_dim, self.theta)
            # NeoX half-rotation: each half shares the same angle vector.
            cos = torch.cat([cos, cos], dim=-1)[:, None, :]
            sin = torch.cat([sin, sin], dim=-1)[:, None, :]
            q = (q.to(torch.float32) * cos + rotate_half(q.to(torch.float32)) * sin).to(
                q.dtype
            )
            k = (k.to(torch.float32) * cos + rotate_half(k.to(torch.float32)) * sin).to(
                k.dtype
            )

        # GQA repeat -> [T, n_heads, head_dim], fp32 attention math.
        k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        qf, kf, vf = (x.transpose(0, 1).to(torch.float32) for x in (q, k, v))
        scores = torch.bmm(qf * self.head_dim**-0.5, kf.transpose(1, 2))
        # Causal mask + the 513 sliding window on sliding layers.
        pos = torch.arange(t_len)
        causal = pos[:, None] >= pos[None, :]
        if self.layer_types[layer] == "sliding_attention":
            causal &= (pos[:, None] - pos[None, :]) < self.window
        scores = scores.masked_fill(~causal[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.bmm(probs, vf)
        out = out.transpose(0, 1).reshape(t_len, -1).to(torch.bfloat16)
        return self.linear(out, p + ".o_proj")

    def mlp(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """sigmoid-logit-add routing (kolibri1.py:128-142) + shared expert."""
        p = f"model.layers.{layer}"
        logits = self.linear(h, p + ".mlp.gate").to(torch.float32)
        bias = self.t(p + ".moe.router.expert_bias").to(torch.float32)
        topk_ids = torch.topk(logits + bias, k=self.topk, dim=-1)[1]
        topk_weights = torch.sigmoid(logits.gather(1, topk_ids))
        if self.renormalize:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )

        t_len = h.shape[0]
        out = torch.zeros_like(h)
        flat_ids = topk_ids.reshape(-1)
        flat_w = topk_weights.reshape(-1)
        for e in torch.unique(flat_ids).tolist():
            mask = flat_ids == e
            rows = torch.nonzero(mask, as_tuple=True)[0]
            token = rows // self.topk
            ep = f"{p}.mlp.experts.{e}"
            gate = self.linear(h[token], ep + ".gate_proj")
            up = self.linear(h[token], ep + ".up_proj")
            ye = self.linear(
                torch.nn.functional.silu(gate) * up, ep + ".down_proj"
            ).to(torch.bfloat16)
            out.index_add_(0, token, ye * flat_w[mask][:, None].to(torch.bfloat16))

        sp = f"{p}.mlp.shared_experts"
        gate = self.linear(h, sp + ".gate_proj")
        up = self.linear(h, sp + ".up_proj")
        out = out + self.linear(
            torch.nn.functional.silu(gate) * up, sp + ".down_proj"
        ).to(torch.bfloat16)
        return out

    def forward_last_logits(self, ids: list[int]) -> torch.Tensor:
        """Full-sequence forward; returns fp32 last-position logits [vocab]."""
        h = self.t("model.embed_tokens.weight")[torch.tensor(ids)].to(torch.bfloat16)
        residual = None
        for layer in range(self.n_layers):
            p = f"model.layers.{layer}"
            # Fused add-norm polarity (kolibri1.py:210-253): the residual
            # accumulates the PRE-norm sum / the POST-normed branch, and the
            # norm normalizes that sum.
            if residual is None:
                residual = h
                x = rms_norm(h, self.t(p + ".input_layernorm.weight"), self.eps)
            else:
                residual = residual + h
                x = rms_norm(residual, self.t(p + ".input_layernorm.weight"), self.eps)
            x = self.attention(x, layer)
            x = rms_norm(x, self.t(p + ".post_attn_norm.weight"), self.eps)
            residual = residual + x
            x = rms_norm(
                residual, self.t(p + ".post_attention_layernorm.weight"), self.eps
            )
            x = self.mlp(x, layer)
            h = rms_norm(x, self.t(p + ".post_ffn_norm.weight"), self.eps)
        # The final norm is fused with the residual too (the W2 scalar
        # reference's "final norm carries the residual").
        h = rms_norm(
            h + residual, self.t("model.norm.weight"), self.eps
        )
        logits = torch.nn.functional.linear(
            h[-1:], self.t("lm_head.weight")
        ).to(torch.float32)
        return logits[0]

    def generate(self, prompt_ids: list[int]) -> tuple[list[int], dict]:
        ids = list(prompt_ids)
        last = None
        for step in range(MAX_NEW_TOKENS):
            logits = self.forward_last_logits(ids)
            last = logits
            nxt = int(torch.argmax(logits))
            ids.append(nxt)
        return ids[len(prompt_ids):], last


def fingerprint(logits: torch.Tensor) -> dict:
    top = torch.topk(logits, k=TOP_K)
    return {
        "topk_ids": top.indices.tolist(),
        "topk_logits": [round(v, 4) for v in top.values.tolist()],
        "logits_sum": round(float(logits.sum()), 4),
        "abs_sum": round(float(logits.abs().sum()), 4),
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    model_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    torch.manual_seed(SEED)
    torch.set_num_threads(128)

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    model = Kolibri1Ref(model_dir)

    entries = []
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        gen, last = model.generate(ids)
        entry = {
            "prompt": prompt,
            "input_ids": ids,
            "generated_ids": gen,
            "final_logits": fingerprint(last),
        }
        entries.append(entry)
        print(f"[{len(entries)}/8] {prompt!r}: {len(gen)} tokens, "
              f"argmax {gen[-1]}", flush=True)

    out = {
        "model": str(model_dir),
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "top_k": TOP_K,
        "decode": "greedy, temperature 0, full recompute per step",
        "provenance": "pure-torch transcription of aleph-alpha-inference "
                      "kolibri1.py @ 049a6a7bd240 on the dequantized bf16 "
                      "reference (scripts/gen-kolibri1-bf16-reference.py)",
        "prompts": entries,
    }
    out_path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
