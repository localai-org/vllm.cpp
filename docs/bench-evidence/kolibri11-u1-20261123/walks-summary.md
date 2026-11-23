# Kolibri-1.1 U1 — four-arm verification and header/tensor walks

Row: `MODEL-TEXT-kolibri-1-1-kolibri1-1-for-causal-lm`. Issue:
ISSUE-LOCAL-01M84HPYRT3BNNF61BKGNRQ740. Date: 2026-11-23.
Status: **walk evidence only**. No token parity is claimed; there is no
gateable CPU oracle for `kolibri1_1` today.

## Artifact pin and verification

HF repo `Aleph-Alpha/Kolibri-1.1-GGUF`, revision
`90116812a1669494a894b729e677ed75d6068320`. Local directory
`/mnt/models/Aleph-Alpha-Kolibri-1.1-GGUF`.

`sha256sum -c` against the four HF LFS oids (log: `verify-all-four.log`):

| Arm | Bytes | GiB | LFS sha256 check |
|---|---:|---:|---|
| IQ1_M-GSQ-RCO-1.75bpw | 16,576,614,848 | 15.44 | OK |
| IQ2_XXS-GSQ-RCO-2.00bpw | 18,941,665,728 | 17.65 | OK |
| IQ2_XS-GSQ-RCO-2.25bpw | 21,336,904,128 | 19.87 | OK |
| Q4_K_M | 45,508,762,048 | 42.38 | OK |

The Q4_K_M file was copied from an independently verified peer copy of the
same revision and re-hashed locally; no bytes were assumed from the source.

## Walks

One file per arm: `walk-iq1_m.txt`, `walk-iq2_xxs.txt`, `walk-iq2_xs.txt`,
`walk-q4_k_m.txt`. Each was produced by the pinned llama.cpp `gguf-py` reader
and contains the full metadata KV list, type-id names, per-tensor name/shape/
type/offset census and a JSON summary. Tensor payloads were never read.

| Arm | Tensors | Type census (id: count) |
|---|---:|---|
| IQ1_M | 875 | 0:391 (F32), 13:306 (Q5_K), 14:34 (Q6_K), 16:12 (IQ2_XXS), 19:127 (IQ1_S), 29:5 (IQ1_M) |
| IQ2_XXS | 875 | 0:391, 8:56 (Q8_0), 13:209, 14:75, 16:45, 17:12 (IQ2_XS), 19:61, 22:9 (IQ2_S), 29:17 |
| IQ2_XS | 875 | 0:391, 8:114, 13:154, 14:72, 16:27, 17:33, 18:7 (IQ3_XXS), 19:37, 21:5 (IQ3_S), 22:20, 29:15 |
| Q4_K_M | 875 | 0:391, 12:413 (Q4_K), 14:71 |

**Every ggml type id in all four arms is already tabulated in our
`gguf_reader.cpp` and already has a dequant arm in `gguf_dequant.cpp`.** No new
quant encoding is needed to read any selected arm. The first implementation
barrier is the architecture, not the quantization.

All four files carry the 1.1 marker KVs:
`kolibri1.attention.indexer.top_k`, `.types`, `.kv_source_map` and
`kolibri1.leading_dense_block_count`. All four omit `attn_k.weight`,
`attn_v.weight` and `attn_k_norm.weight` on exactly layers 9, 14, 34, 39, 44
and 49. The official `general.architecture` is `kolibri1` on every arm — the
arch-collision trap is confirmed on the real artifacts, not only on the
assessment's Q4_K_M walk.

## Fit finding (accepted as U1 evidence by oc-41)

Byte census of the verified Q4_K_M: routed experts 40.75 GiB (96% of the
file), attention 0.89 GiB, embed/head/norm 0.42 GiB, shared experts 0.11 GiB,
other 0.21 GiB. Resident non-expert state is only ~1.63 GiB, so a 32 GB P150
is blocked almost entirely by the expert tower.

Experts are packed three tensors per MoE layer with the expert dimension last,
e.g. `[2560,512,384]`; that is ~2.26 MiB compressed per expert over 48 MoE
layers. Top-6 routing across 48 layers is ~652 MiB of compressed expert bytes
per token, before any dequantization. Expanded to bf16 a full expert tower is
~138 GiB, so a hot set is mandatory. Host bandwidth at 25-60 GB/s puts a cold
stream at ~11-26 ms/token. **Fit and speed are separate questions:** our
general expert-streaming stack exists, but Kolibri-1 currently has only a
device-free policy planner (`kolibri1_tt.cpp:241`), not a streaming lane.

The three GSQ-RCO arms (15.44-19.87 GiB) are the only ones within reach of a
32 GB P150 budget; KV-cache and activation headroom are still uncalculated for
1.1's indexed attention and NVFP4 KV cache.

## What this evidence supports

1. The U1 guard is required on real artifacts: all four files would be claimed
   as Kolibri-1 by an arch-string-only reader.
2. Later waves should port the indexed-attention / dense-prefix architecture
   first; no new quant decoder is needed for these four arms.
3. No token gate or throughput number is established by this evidence. A
   gateable oracle is still owed (llama.cpp PR #32891 unmerged; the
   aleph-alpha-inference v2 plugin is CUDA-only).
