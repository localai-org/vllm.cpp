# Oracle: `vllm-tt-plugin-qwen38` — the dflash2-vision serving bundle for qwen3.8-27B

A scoped, second `vllm-tt-plugin` record, admitted for one narrow reason, in
the same shape as [`llama-cpp-flint`](llama-cpp-flint.md). It is a concrete,
self-contained instance of the [`vllm-tt-plugin`](vllm-tt-plugin.md) oracle
serving qwen3.8-27B: the model the QWEN38-P150-REBENCH row measured at 0.03
tok/s output on our single-P150 path, with no correctness reference except
itself. The bundle's greedy output is token-for-token exact to the target
model, which makes it the correctness reference that path has never had. It
does not replace the [`vllm-tt-plugin`](vllm-tt-plugin.md) record, it does not
outrank vLLM, and it is never a mirror source.

## Why the `vllm-tt-plugin` file cannot carry this pin

`scripts/check-oracle-pins.py` admits exactly one ` ```oracle-pin ` block per
file, and the `vllm-tt-plugin` pin is deliberately the plugin's main HEAD as
of 2026-09-21 (#3261). Folding a third-party serving bundle into that record
would change what that pin means for every measurement already taken against
it.

## What it is, and what it is not

`mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4` is a tt-model
serving BUNDLE (code, not weights): vLLM + the tt-model plugin + custom
kernels, DFlash2 speculative decoding (draft + verify on-card, the host
touches each step once), TWO p300c boards = 4 Blackhole chips (1x4 mesh,
FABRIC_1D). Weights are `Qwen/Qwen3.8-27B` bf16 safetensors, downloaded on
first use. OpenAI-compatible endpoints; 8 slots; 262,144-token context.

- **A correctness reference, not a performance one.** Greedy output is exact
  to the target model; sampled output follows the target distribution exactly
  (rejection sampling). That property is what our qwen3.8-27B TT path has
  never had on any Tenstorrent box: a denominator that runs the same
  architecture from the same weights.
- **NOT a throughput denominator on different hardware.** The bundle runs 4
  Blackhole chips + DFlash2 speculative decoding + upstream's kernel set; our
  path is one P150 without speculative decoding. The gap to our 0.03 tok/s
  decomposes into hardware (4x), speculative decoding (multi-token per step),
  kernel maturity, and stack. Their numbers are recorded as *their* numbers;
  ours are recorded against our own path's profile.
- **NOT token-comparable across architectures.** A 4-chip mesh's reduction
  order differs from a single P150's, so greedy outputs may diverge on
  near-tie positions for reasons that are the ARCHITECTURE's, not either
  implementation's. Any token comparison is architecture-dependent and must
  be labeled as such (the kolibri1 band-gate principle: same-weights
  comparisons only where the reduction order matches).

## Provenance

Repo `mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4`, revision
`7855cf0d2982064350991bdb8e1ec1db82f914b3`, verified against its HF metadata
and README on 2026-11-03. The bundle is self-contained: it pulls vLLM, the
tt-model plugin, and its custom kernels, then downloads `Qwen/Qwen3.8-27B`
bf16 on first use. Serving numbers from its README (one user): 78–110 tok/s
DFlash, 43 tok/s plain, ~93 tok/s @4 users, ~140 tok/s @8 users; TTFT cached
0.4–0.6s. Those are the bundle's own measurements on its own 4-chip hardware
— recorded here as provenance, never as this repository's denominator.

## Gateability

The bundle CANNOT run on this host: it requires 4 Blackhole chips in a 1x4
mesh (TWO p300c boards), and this host has one P150. The honest state is
`gateable = no` with the hardware mismatch named and the external resource
that would make it gateable — a 4x-Blackhole box — recorded below. The
qwen3.8-27B correctness reference this row wants is therefore DEFERRED until
a gateable instance exists: the row delivers the registration and the honest
gateability record, not a comparison.

```oracle-pin
id = vllm-tt-plugin-qwen38
role = secondary
upstream = https://huggingface.co/mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4
scope = the qwen3.8-27B correctness reference on Tenstorrent Blackhole (4-chip mesh): the dflash2-vision serving bundle's exact greedy output — a reference our single-P150 path has never had; never a throughput denominator on different hardware
pin = 7855cf0d2982064350991bdb8e1ec1db82f914b3
pin_label = the dflash2-vision v6-thin bundle, cp8.1
pinned_on = 2026-11-03
gateable = no
evidence = #3547
```
