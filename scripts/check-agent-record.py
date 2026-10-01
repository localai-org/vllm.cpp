#!/usr/bin/env python3
"""Validate the canonical .agents tables and lifecycle contracts."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / ".agents"

MATRICES = {
    # 358 since 2026-08-05: +31 architectures vLLM's registry defines that we had
    # NEVER inventoried (found by scripts/upstream-inventory.py). All INVENTORIED;
    # inventorying is not committing.
    # 360 since 2026-08-07: +`MODEL-AUDIO-PARAKEET-ENCODER`, the first row of the
    # new `MODEL-AUDIO` section (Parakeet / FastConformer encoder + CTC head,
    # spike parakeet-conformer-encoder.md work item P4). It is NOT one of the 328
    # registry architectures: vLLM ships it as the audio COMPONENT of
    # `nano_nemotron_vl.py` and delegates the encoder itself to transformers, so
    # there is no `registry.py` entry to inventory. A genuinely new row, never a
    # count relaxed to make a transition pass.
    # 361 since 2026-08-07: +`MODEL-AUDIO-PARAKEET-TRANSDUCER` (the Parakeet RNN-T
    # and TDT heads over that same encoder: `ParakeetForRNNT` / `ParakeetForTDT`,
    # spike work item P6). A SEPARATE row rather than an advance of the encoder
    # row, because it is a different upstream model class with its own state dict,
    # its own decode and its own checkpoints. Like the encoder row it is not one
    # of the 328 registry architectures: vLLM has no transducer call site at all
    #: so there is nothing in `registry.py` to inventory. Bumped because a new
    # row EXISTS, never to make a transition pass.
    # 362 since 2026-08-10: +`MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation`
    # (Meta's Muse Glimmer 30B, released 2026-08-08). A THIRD beyond-pin row: it is
    # not one of the 326 registry architectures at `555967922`, and unlike the two
    # Parakeet rows it is absent because it did not exist yet, not because vLLM
    # delegates it. Its only upstream implementation is the still-OPEN
    # vllm#51655; see porting-inventory.md §9 deviation 16. Bumped because a new
    # row EXISTS, never to make a transition pass.
    # 382 since 2026-09-25: +20 rows landed since the muse-glimmer bump — the
    # and the `MODEL-DSV4-EXL3` checkpoint-campaign and `MODEL-GLINER25`
    # decision rows restored/recorded in the same record-rot repair (382).
    # DeepSeek-V4/V4.1, Qwen3.5/3.5-MoE, Qwen4-Exp, Bailing/Bailing-MM, dots.3-note
    # (+ its MTP), GLM5-Next, MiniMax-Music3, LTX-2.5 video transformer, Higgs-Audio-V3,
    # IndexTTS2 (s2-mel + talker), Moss-TTS (delay + realtime), Qwen3-TTS,
    # Voxtral-Realtime and dSpark-V4.1 rows. Bumped because a new row EXISTS,
    # never to make a transition pass.
    "MODEL": (AGENTS / "model-matrix.md", 384),
    # 82 since 2026-07-21: +`QUANT-NVFP4-CT-W4A16` (compressed-tensors NVFP4A16 /
    # W4A16 — NVFP4 weights with BF16 activations, distinct from the existing
    # `QUANT-NVFP4-CT-W4A4` and `QUANT-NVFP4-MO-W4A16` rows in both scheme
    # discovery and kernel selection). This count is the inventory size, so it is
    # bumped when a genuinely new scheme is inventoried — never to make a failing
    # state transition pass.
    # 2026-07-29: the pre-existing INVENTORIED `QUANT-GGUF-IQ3_XXS` (id 18) row was
    # ADVANCED to `ACTIVE` (keep-quant compute landed) by `CLAIM-DEEPSEEK-V4-W8` —
    # the `UD-IQ2_XXS` down-projection routed experts (`ffn_down_exps`) are IQ3_XXS;
    # no row count change (an in-place advance, not a new row).
    # 87 since 2026-09-25: +`QUANT-EXL3`, `QUANT-EXL3-MUL1`, `QUANT-EXL3-PERF`
    # (the EXL3 trellis family) and the `QUANT-QWEN38-27B-GGUF-ARM` /
    # `QUANT-QWEN38-27B-NVFP4-ARM` model-quant arm rows. Bumped because a new
    # row EXISTS, never to make a transition pass.
    "QUANT": (AGENTS / "quantization-matrix.md", 87),
    # 34 since 2026-07-22: +`KERNEL-GEMM-CPU-ELEM` (the elementwise f32/f16/bf16 CPU
    # GEMM — a genuinely separate family from `QUANT-GGUF-CIQ-GEMM`'s block-quantized
    # `kMatmulBTQuant`: it serves every safetensors CPU path and every non-block
    # tensor of a mixed GGUF). 33 since 2026-07-22: +`KERNEL-ACCEL-PROVIDER-SELECT`
    # (which IMPLEMENTATION of an op runs when a device has more than one — the
    # selection layer above every kernel family here, distinct from
    # `KERNEL-CUDA-DISPATCH-AOT`, which selects an ARCH for one implementation).
    # 35 since 2026-07-26: +`KERNEL-ATTN-DFLASH-BLOCK` (the DFlash draft's in-block
    # attention — the project's FIRST non-causal / bidirectional attention primitive,
    # a genuinely separate op from the causal `kAttention`/`kPagedAttention`; SPEC-DFLASH
    # D2, `CLAIM-DFLASH-D2`).
    # 36 since 2026-07-27: +`KERNEL-ATTN-DFLASH-PAGED-BLOCK` (the CAPTURE-SAFE paged
    # variant of the DFlash in-block attention — a genuinely separate op
    # `kDFlashPagedBlockAttention` reading the growing context from a paged K/V cache +
    # persistent device block_table instead of a materialized combined buffer, the
    # CUDA-graph draft-attention primitive; SPEC-DFLASH D12 Part B, `CLAIM-DFLASH-D12`).
    # 37 since 2026-07-28: +`KERNEL-ATTN-DENSE-FLASH` (the SHARED-MEMORY-TILED flash
    # variant of `AttentionDenseFast` for long non-causal contexts — a genuinely
    # separate op `kAttentionDenseFlash` that tiles K/V across a block of query-warps
    # in shared memory, for the Whisper AUDIO encoder; multimodal-speed §14,
    # `CLAIM-MM-SPEED-AUDIO-ENC-KERNEL`).
    # 38 since 2026-07-28: +`KERNEL-ATTN-DSA-SPARSE-INDEX` (the DeepSeek-V4 DSA
    # "Lightning Indexer" sparse-attention SELECTION op — weighted MQA logits with a
    # per-head ReLU + causal top-k token selection, the project's FIRST sparse-
    # attention candidate-selection primitive, distinct from every dense/paged/MLA
    # attention family above which score ALL keys. W3 landed a portable host
    # reference + unit gate; the device kernel is a W7 residual. `SPIKE`,
    # `CLAIM-DEEPSEEK-V4-W3`, spec specs/deepseek-v4-flash.md).
    # 39 since 2026-07-28: +`KERNEL-KDA-DELTA` (the Kimi Delta Attention gated-
    # linear-attention delta vs plain GDN — the per-channel [H,D] low-rank decay
    # (f_a/f_b bottleneck), the sigmoid-gated output norm (FusedRMSNormGated), the
    # 3 q/k/v short convs + q/k L2-norm. A genuinely new gated-linear-attention
    # family distinct from GDN, which has only a per-head scalar decay and no gated
    # output norm; SUBCLASSES GDN so its recurrence is REUSED, not re-ported. W1
    # landed a portable host reference + unit gate; the device kernel + the
    # Kimi-Linear-48B proxy e2e gate are named residuals. Shared unblocker for
    # Kimi-Linear-48B and Kimi-K3 (W4). `SPIKE`, `CLAIM-KDA-KERNEL`, spec
    # specs/kda-kernel-delta.md).
    # 40 since 2026-07-29: +`KERNEL-ATTN-DSA-COMPRESSOR` (the DeepSeek-V4 DSA
    # COMPRESSOR — the softmax-weighted window POOL that compresses
    # `(1+overlap)*compress_ratio` KV-state rows into one compressed latent (per
    # head-dim-column softmax, then RMSNorm) + the fused save-time APE add + the
    # fp8_ds_mla KV-cache STATE layout (448 fp8 UE8M0 per-64 block scales + 64
    # bf16 rope, 576B token stride, 7+1 scale bytes). A genuinely separate op from
    # `KERNEL-ATTN-DSA-SPARSE-INDEX` (which SELECTS keys): this one POOLS +
    # QUANTIZES the selected/windowed KV into the latent the MLA reads and how it
    # is cached across steps. W4 landed a portable host reference + unit gate; the
    # device kernel (the fused `_fused_kv_compress_norm_rope_insert_sparse_attn`)
    # is a W7 residual. `SPIKE`, `CLAIM-DEEPSEEK-V4-W4`, spec
    # specs/deepseek-v4-flash.md).
    # 41 since 2026-07-29: +`KERNEL-MHC-SINKHORN` (the DeepSeek-V4 Manifold/Markov
    # Hyper-Connections topology — the `[tokens, hc_mult, hidden]` residual
    # manifold mixed by a 20-iteration Sinkhorn-normalized doubly-stochastic
    # matrix, the mHC pre/post mixes with the FOLDED attn/ffn RMSNorms, and the
    # hc_head collapse). A genuinely new residual-stream topology distinct from the
    # plain residual+RMSNorm every other family uses. W5 landed a portable host
    # reference + unit gate (ported from the vLLM eager reference mhc/torch.py —
    # correcting the W0 "no eager reference" premise — and gated against a from-
    # first-principles double-precision Sinkhorn derivation); the device kernel +
    # DeepseekV4Model::Forward assembly are W7 residuals. `SPIKE`,
    # `CLAIM-DEEPSEEK-V4-W5`, spec specs/deepseek-v4-flash.md).
    # 42 since 2026-07-29: +`KERNEL-MOE-SQRTSOFTPLUS-HASH` (the DeepSeek-V4 MoE
    # router + clamped-SwiGLU deltas — the `sqrt(softplus(·))` score function, the
    # noaux_tc bias-for-selection top-k with weights gathered from the UNBIASED
    # scores, the hash `tid2eid` token-id→expert route that BYPASSES top-k, and the
    # asymmetric clamped SwiGLU expert activation `SiluAndMulWithClamp`). The three
    # genuinely-new-vs-V2/V3 MoE pieces; the shared grouped-GEMM / expert / shared-
    # expert machinery is REUSED, not re-ported. W6 landed a portable host reference
    # + unit gate; the device kernels + DeepseekV4Model::Forward assembly are W7
    # residuals. `SPIKE`, `CLAIM-DEEPSEEK-V4-W6`, spec specs/deepseek-v4-flash.md).
    # 43 since 2026-07-29: +`KERNEL-DSV4-W7-DEVICE` (the DeepSeek-V4-Flash W7-DEVICE
    # CUDA kernels — the four NEW V4 op families' device kernels: MHC Sinkhorn+pre/
    # post+head, DSA indexer weight-fold/MQA-logits/causal-topk + sink softmax +
    # grouped output-LoRA, compressor pool+norm + fp8_ds_mla KV encode/decode, and
    # the sqrtsoftplus/hash router + clamped SwiGLU. Each a 1:1 device port of the
    # landed host reference (KERNEL-{MHC-SINKHORN,ATTN-DSA-SPARSE-INDEX,ATTN-DSA-
    # COMPRESSOR,MOE-SQRTSOFTPLUS-HASH}) registered through the OpProvider seam and
    # RUNTIME-VERIFIED on the DGX GB10 at small shape vs its host-ref oracle (11/11
    # cases · 153 assertions, compute-sanitizer memcheck 0 errors, RED-first proven).
    # The 512-wide MLA attn + expert grouped-GEMM REUSE the existing NVFP4/FP8
    # kernels — NOT re-ported. `IMPL`, `CLAIM-DEEPSEEK-V4-W7-DEVICE`, spec
    # specs/deepseek-v4-flash.md; real-checkpoint e2e stays W8, multi-Spark).
    # 44 since 2026-07-29: +`KERNEL-QUANT-CIQ-IQUANT` (the DeepSeek-V4 W8 keep-quant
    # `vec_dot` for the ~2-3-bit codebook encodings IQ2_XXS/IQ3_XXS/Q2_K — the
    # single-Spark GGUF memory enabler; extends `QUANT-GGUF-COMPUTE`'s six-type
    # `kMatmulBTQuant` so the 158 B routed experts stay COMPRESSED instead of
    # OOM-expanding to bf16. CPU tier only, like the six existing k-quants. 1:1
    # ports of ggml `vec_dot_q2_K/iq2_xxs/iq3_xxs_q8_K_generic`; gated 19 cases /
    # 130444 assertions, RED-first proven. `SPIKE`, `CLAIM-DEEPSEEK-V4-W8`, spec
    # specs/deepseek-v4-flash.md §W8).
    # 45 since 2026-07-29: +`KERNEL-QUANT-CIQ-GEMM-CUDA` (the FIRST CUDA keep-quant
    # GGUF k-quant GEMM — the kCUDA provider for `kMatmulBTQuant`. MMVQ-style
    # dequant-in-kernel dot: quantize the activation to Q8_K on-GPU, integer-dot
    # against the compressed Q8_K-family weight blocks (IQ2_XXS/IQ3_XXS/Q2_K +
    # Q3_K/Q4_K/Q5_K/Q6_K) kept COMPRESSED in the unified pool. Registering it flips
    # GgufQuantComputeAvailable TRUE on kCUDA so DeepSeek-V4's routed experts run on
    # the GPU instead of the 20 ARM cores. A NEW kernel family — the CPU
    # KERNEL-QUANT-CIQ-IQUANT vec_dot is a separate row/impl. RUNTIME-VERIFIED on
    # the DGX GB10: 2/2 cases · 92401 assertions vs the CPU oracle + f64 dequant,
    # memcheck 0, RED-first proven. `ACTIVE`, `CLAIM-CUDA-KEEPQUANT-GEMM`, spec
    # specs/deepseek-v4-flash.md §W8.)
    # 46 since 2026-08-06: +`KERNEL-CPU-A76-Q8-DOT`, a separately gateable
    # Cortex-A76 Q8_0 x Q8_0 DotProd/assembly family. The physical-Pi trace
    # proves the portable dot is reached at 20.10% of Qwen3.5-2B user cycles;
    # the row owns exact-order C++ SDOT vs scheduled AAPCS64, independent of
    # the broad CPU-backend row.
    # Inventory size, bumped for a genuinely new family — never to make a failing
    # state transition pass.
    # 47 since 2026-08-06: +`KERNEL-GEMM-CPU-ELEM-X86WIDE` (the AVX2/AVX-512
    # elementwise tiers; our x86 tier is SSE2 while the box has avx512f, a
    # measured 3.5x) and +`KERNEL-GEMM-CPU-TILED` (the tinyBLAS-style tiled
    # sgemm; controls proved our NEON kernel is at ggml-stock parity and the
    # whole 16-bit deficit is llamafile, ~1.9x Arm / ~2.4x x86).
    # 50 since 2026-08-06: +`KERNEL-CPU-CONV2D-SUBSAMPLE`, +`KERNEL-DEPTHWISE-CONV1D`
    # and +`KERNEL-ATTN-RELPOS` — the three conformer/FastConformer audio-encoder
    # primitives the tree had no device op for at all (Conv2d existed only as a host
    # std::vector loop; the only depthwise conv1d was the CAUSAL Mamba/GDN one; every
    # attention path was RoPE + paged/flash KV). Spike specs/parakeet-conformer-encoder.md.
    # 51 since 2026-08-06 (PR #79): +`KERNEL-CPU-A76-Q8-DOT`, a separately gateable
    # Cortex-A76 Q8_0 x Q8_0 DotProd/assembly family. The physical-Pi trace
    # proves the portable dot is reached at 20.10% of Qwen3.5-2B user cycles;
    # the row owns exact-order C++ SDOT vs scheduled AAPCS64, independent of
    # the broad CPU-backend row.
    # 59 since 2026-09-25: +the eight rows the active claims and committed specs named and the
    # specs promised but the matrix never carried (record rot): the four
    # megakernel SPIKE row the #3084 spec and #3085 record-gate repair already
    # declared, plus the four
    # DFlash2 kernel rows (`KERNEL-DFLASH2-GROUPED-CONV`, `-SELECTOR-EDGES`,
    # `-TOPK-PAIRS`, `-PATH-WALK` — each marked LANDED in the DFlash2 spec's
    # port map), the LTX-2.5 `KERNEL-CONV3D` and `KERNEL-LTX2-VAE` device arms,
    # and `KERNEL-QUANT-CIQ-GEMM-ROCM` (four merged PRs; the handover spec
    # already called the row ACTIVE).
    # 52 since 2026-08-10 (issue #284): +`KERNEL-GEMM-CPU-ELEM-A76`, the
    # Raspberry Pi 5 Cortex-A76 BF16 specialization of the elementwise GEMM.
    # The C++/NEON row is separately gateable from the broad portable family;
    # assembly is explicitly blocked until its C++ path beats llama.cpp E2E.
    "KERNEL": (AGENTS / "kernel-matrix.md", 60),
    # 56 since 2026-07-22: +`BACKEND-ACCEL-PROVIDER` (the acceleration-provider seam
    # itself, which is a cross-backend platform concern rather than a platform).
    # 57 since 2026-07-22: +`BACKEND-SEAM-AUDIT` (the accelerator-seam AUDIT — does
    # our MLX/Vulkan/Metal architecture actually port vLLM's CUDA-path strategy, and
    # what is the shared layer's device-leakage budget. A cross-backend structural
    # concern that owns no kernel and no platform, hence its own row).
    # 60 since 2026-07-27: +`BACKEND-CUDA-SM060` + `BACKEND-CUDA-SM061` (the Pascal
    # targets for the beyond-vLLM fp16/non-tensor-core breadth lane) + the
    # `BACKEND-GATE-CUDA-LLAMACPP-LEGACY` competitor floor (llama.cpp on the same
    # old card, since vLLM has no entry on Pascal/Volta/Turing). See
    # specs/cuda-arch-breadth-fp16.md.
    # 65 since 2026-07-28: +the distributed / scale-out family
    # `BACKEND-DISTRIBUTED-COMM`/`-TP`/`-PP`/`-MULTINODE-SPARK`/`-MLX-RING` (the NEW
    # scale-out capability dimension — one `vt::` collective abstraction with
    # nccl/RDMA/MLX-ring transports; single-GPU today). All `SPIKE`,
    # `CLAIM-SCALE-OUT-SPIKE`, spec specs/scale-out-distributed.md.
    # 68 since 2026-07-28: +the parallelism-MODE rows `BACKEND-DISTRIBUTED-DP`/
    # `-EP`/`-SP` (the enumeration of vLLM's parallelism modes onto our transport
    # abstraction — DP engine-replica scale-out, EP whole-expert all-to-all, SP the
    # TP-mode reduce-scatter compilation pass; TP/PP already existed, context/CP
    # rides the same abstraction and takes no row). All `SPIKE`,
    # `CLAIM-PARALLELISM-MODES-SPIKE`, spec specs/parallelism-modes.md.
    # 79 since 2026-08-05: +11 BACKEND-GGML-* rows, the llama.cpp ggml
    # backends folded into scope (user-directed). All INVENTORIED and
    # spike-gated; inventorying is not committing.
    # 80 since 2026-08-09: +`BACKEND-TENSTORRENT`, an extension platform
    # proposal (Tenstorrent Blackhole, ttnn C++ adapter) in the same class as
    # Metal/Vulkan. INVENTORIED; spec-only, not yet reviewed or accepted.
    # 90 since 2026-09-25: +the llama.cpp competitor floors
    # `BACKEND-GATE-CUDA-LLAMACPP` / `BACKEND-GATE-ROCM-LLAMACPP` and the eight
    # Tenstorrent rows (`BACKEND-TENSTORRENT-GDN`, `-GDN-DEVICE-PURE`,
    # `-HOST-FREE-FORWARD`, `-KEEPQUANT`, `-MISTRAL`, `-QWEN35`,
    # `-RESIDUAL-GOLDEN`, `-TRACE-RUNNER`). Bumped because a new row EXISTS,
    # never to make a transition pass.
    "BACKEND": (AGENTS / "backend-matrix.md", 90),
}

ENGINE_MATRIX = AGENTS / "engine-matrix.md"
ENGINE_PREFIXES = (
    "ENG",
    "KV",
    "PAR",
    "SAMPLE",
    "TOOLS",
    "SPEC",
    "SERVE",
    "LORA",
    "ATTN",
    "LOAD",
)
# 117 since 2026-07-25: +`ENG-MM-AUDIO-ENCODER` (the Whisper-class AUDIO encoder
# tower, audio-track A2 — the encoder half of audio understanding, proven faithful
# in isolation; a genuinely-new engine capability, distinct from the A0/A1
# `ENG-MM-AUDIO-PIPELINE` INPUT row). Bumped for a real new row, never to make a
# failing state transition pass.
# 120 since 2026-07-27: +`KV-SGLANG-RADIX-CACHE` (SGLang RadixAttention
# behavior-parity scope — verdict: already fused into our block-hash APC, the
# `--enable-radix-attention` flag is an alias) and +`ENG-SGLANG-BEHAVIOR-FLAG`
# (the SGLang-alike runtime survey + the enable/disable control: cache-aware LPM
# scheduling is the one genuinely-distinct flag-worthy behavior; overlap ==
# `ENG-ASYNC-SCHED`, jump-forward deferred). Both `SPIKE`,
# `CLAIM-SGLANG-RADIX-SCOPE`, spec `specs/sglang-radixattention.md`.
# 121 since 2026-07-28: +`SAMPLE-N` (parallel sampling — a request with n>1 fans
# out into n prompt-sharing child sequences aggregated into one RequestOutput /
# n OpenAI choices; the n==1 default path stays byte-identical). A real new engine
# capability, `ACTIVE`, `CLAIM-C7-N-SAMPLING`.
# 122 since 2026-07-28: +`SAMPLE-BEST-OF` (the OpenAI `best_of` endpoint control —
# generate best_of children via the SAMPLE-N fan-out, rank by cumulative logprob,
# return the top-n; a distinct request-surface capability from raw n-sampling). The
# best_of==n default path is byte-identical. `ACTIVE`, `CLAIM-C7-BESTOF-BEAM-API`.
# 123 since 2026-07-28: +`SPEC-MTP-GGUF` (MTP speculative decoding from a GGUF
# TARGET). Distinct from `SPEC-MTP` (`DONE`, safetensors): the engine currently
# REFUSES mtp+GGUF on the assumption that GGUF exports carry no head, but
# llama.cpp's Qwen3.5 converter does emit it and `HfConfigFromGguf` already reads
# the announcing metadata key. `READY`, spiked.
# 124 since 2026-07-28: +`SPEC-DFLASH-GGUF` (DFlash draft, then target, from GGUF).
# Distinct again: a separate draft checkpoint with its own llama.cpp `dflash` arch
# contract, plus a shared-head coupling to the target that `SPEC-MTP-GGUF` does not
# have. `READY`, spiked.
# 125 since 2026-07-28: +`ENG-POOLER-SEQ` (the non-generative POOLER OP — the
# sequence pooling methods CLS/LAST/MEAN + the normalize/classify activation heads
# that turn hidden states into a pooled embedding/logit row instead of a sampled
# token). HIGH-priority feature-gap #2 (pooling task class); a genuinely-new
# engine capability, W1 CPU brick landed + unit-gated. `ACTIVE`, `CLAIM-POOLING`,
# spec `specs/pooling-task-class.md`. Bumped for a real new row, never to make a
# failing state transition pass. (`SERVE-POOLING-ENDPOINTS` INVENTORIED→SPIKE in
# the same change is a state move on the existing row, not a new row.)
# 126 since 2026-07-28: +`KV-PREFIX-MATCH-UNIT` (`--prefix-match-unit` / config
# `prefix_match_unit`, NEW in vLLM 0.26). A distinct capability: the fine-grained
# prefix-cache matching unit (`resolve_kv_cache_block_sizes` -> `hash_block_size`),
# not covered by the `KV-PREFIX-CACHE` block-hash row (which hashes at
# `block_size`). W0 spike + W1 resolver landed; `PARTIAL`.
# 127 since 2026-07-29: +`ENG-PLUGIN-SYSTEM` (the out-of-core PLUGIN system —
# `LoadGeneralPlugins()` + the general-plugin registration seam that lets an
# external TU / shared object register a model factory / platform / quant method
# through the existing `REGISTER_VLLM_MODEL`-style seams WITHOUT editing engine
# core; the RECORDS-GAP the feature-gap analysis named for row creation). A real
# new engine capability, W0 spike + W1 CPU brick landed + unit-gated RED-first;
# `ACTIVE`, `CLAIM-PLUGIN-SYSTEM`, spec `specs/plugin-system.md`. Bumped for a
# real new row, never to make a failing state transition pass.
# 128 since 2026-07-29: +`SERVE-BATCH-API` (the offline OpenAI Batch API runner —
# read a JSONL of BatchRequestInput, dispatch each line to the matching serving
# handler, write a BatchRequestOutput JSONL; the RECORDS-GAP the feature-gap
# analysis named for row creation, recommending SERVE-BATCH-API). A real new
# serving capability, W0 spike + W1 CPU brick (chat dispatch orchestrator over
# the existing OpenAIServingChat, NO reimplemented generation) landed +
# unit-gated RED-first; `ACTIVE`, `CLAIM-BATCH-API`, spec `specs/batch-api.md`.
# Bumped for a real new row, never to make a failing state transition pass.
# 130 since 2026-07-29: +`SPEC-DRAFT-MODEL` (the classic model-agnostic SEPARATE
# draft-model speculator — a full smaller standalone LM runs K autoregressive
# greedy steps to propose K drafts, target verifies in one forward, longest
# accepted prefix emitted; distinct from MTP/EAGLE/DFlash: no target-hidden tap,
# no shared embed/lm_head) and +`SPEC-MEDUSA` (Medusa N-head single-pass
# speculator). The two RECORDS-GAP items the feature-gap analysis named for row
# creation (lines 82-83). `SPEC-DRAFT-MODEL` ACTIVE (W0 spike + W1 CPU greedy
# propose brick landed + unit-gated RED-first, reusing the landed SPEC-REJECTION
# verify); `SPEC-MEDUSA` SPIKE (W0 spike only, proposer deferred to W2).
# `CLAIM-SPEC-DRAFT-MEDUSA`, spec `specs/draft-model-medusa-spec.md`. Bumped for
# two real new rows, never to make a failing state transition pass.
# 131 since 2026-07-29: +`ENG-POOLING-RUNNER` (the pooling RUNNER path — where the
# generation runner samples a token, `PoolingRunner` applies the model's `Pooler`
# to the last hidden state and returns the POOLED DATA / embedding vector; the W3
# brick of the pooling task class). A real new engine capability, CPU brick landed
# + structurally cosine-gated RED-first; `ACTIVE`, `CLAIM-POOLING`, spec
# `specs/pooling-task-class.md`. (Its sibling `ENG-POOLER-SEQ` advanced W1→W2 in
# the same change — the heads/`SequencePooler`/`DispatchPooler` composite — not a
# new row.) Bumped for a real new row, never to make a failing state transition
# pass.
# 141 since 2026-08-06: +`SERVE-VIDEOS-OAI` (the `/v1/videos` request surface in
# OpenAI's Sora shape plus `GET /v1/videos/{id}/content`) — a real new serving
# capability, not a restatement of the MiniMax-H3 model row: an OpenAI video
# client works unmodified, the MP4 is fetchable over HTTP at all, and the
# fl2va/ref2va exclusivity is enforced at the request boundary. CPU-landed +
# gated, `PARTIAL`, `CLAIM-SERVE-VIDEOS-OAI`, spec `specs/minimax-h3.md` §9.
# Bumped for a real new row, never to make a failing state transition pass.
# 142 since 2026-08-06: +`SERVE-VIDEOS-REFS` (reference CONDITIONING over
# `/v1/videos`: `input_reference` -> fl2va, plus the two `metadata` ref2va
# modalities) — a real new capability stacked on `SERVE-VIDEOS-OAI`, not a
# restatement of it: that row made an OpenAI body PARSE, this one makes its
# references reach the pipeline. Before it no reference modality was reachable
# over HTTP at all. CPU-landed + gated, `CLAIM-SERVE-VIDEOS-REFS`, spec
# `specs/minimax-h3.md` §10.
# Bumped for a real new row, never to make a failing state transition pass.
# 143 since 2026-08-07: +`ENG-RELEASE-BINARIES` (downloadable, backend-specific
# server bundles and their static/runtime dependency contract) — a real
# distribution capability requested in issue #117, not a restatement of the
# server implementation. Inventoried while its release-matrix spike is written;
# no packaging support is claimed by the count bump.
# Bumped for a real new row, never to make a failing state transition pass.
# 144 since 2026-08-08: +`ENG-RELEASE-CONTAINERS` (published GHCR container
# images built by GitHub Actions — a distribution channel distinct from the
# downloadable archives in `ENG-RELEASE-BINARIES`: different artifact format,
# registry, tag contract, multi-arch manifest and publish flow, sharing only the
# staged bundle. User-directed, issue #170; inventoried while its spike is
# written, and no image, workflow or registry package is claimed by the bump.
# Bumped for a real new row, never to make a failing state transition pass.
# 145 since 2026-08-09: +`ENG-DOCS-SITE` (publish `docs/` as a GitHub Pages site
# that mounts the existing markdown read-only rather than copying it — a real
# distribution surface for the documentation, distinct from the binary and
# container channels above and from the docs themselves, which it does not
# modify). User-directed, issue #224; `READY` on its committed spec, and no
# site, workflow or published page is claimed by the bump.
# 147 since 2026-08-10: +`KV-MOONCAKE-STORE` (`MooncakeStoreConnector`, Mooncake's
# distributed KV object store as an external cache pool). Split out of
# `KV-CONNECTORS`, whose blanket "Mooncake NOT SCHEDULED" verdict conflated the
# P2P `MooncakeConnector` with the store connector; the P2P half keeps that
# verdict. User-directed, issue #287; `SPIKE` on its committed spec. No client,
# no connector, no build flag and no gate result is claimed by the bump.
# 148 since 2026-08-11: +`ENG-RECORD-CONFLICT-SURFACES` (retire the shared record
# surfaces that make concurrent PRs conflict by construction — the `STATUS_RATCHET`
# global, the `NOW.md` byte budget, and the insert-at-one-anchor claims table).
# MEASURED at `origin/main` `d928e2c3`: 16 of 29 open PRs conflict and 13 of those
# 16 conflict in bookkeeping only. User-directed, issue #364; `READY` on its
# committed spec. No checker semantic, no doc content and no gate result is
# changed by the bump — this row is the record of the work, not the work.
# 149 since 2026-08-11: +`ENG-NOW-DERIVED` (the live position is DERIVED and the
# freshness obligation moves to the row's own spec, so `.agents/NOW.md` stops
# being a surface every row-advancing PR must write). Follow-up to #364, which
# removed the file's byte budget but not the doc-checkpoint requirement that
# marched every PR into it. User-directed, issue #374; `ACTIVE` on its committed
# spec. No checker semantic beyond the row's own scope and no product source is
# changed by the bump.
# Bumped for a real new row, never to make a failing state transition pass.
# 179 since 2026-09-25: +30 rows landed since the dbd0d51cd bump — the record
# gates (ENG-RECORD-*), release and CI rows, MM forward/tower rows, expert-stream,
# CUDA-graph break/dedup/diffusion, KV-DSV4-MULTICACHE, the dSpark spec-decode
# family, SERVE-RECIPE-ARGS / -REQUEST-LENGTH-GUARD, LOAD-GGUF-MMPROJ and the
# attention-window row. Bumped because a new row EXISTS, never to make a
# transition pass.
# 180 since 2026-09-29: +1 row (ENG-HOST-EMBEDDING, the host-resident token
# table). Bumped because a new row EXISTS, never to make a transition pass.
ENGINE_ROWS = 180

ENGINE_SUMMARY_SECTIONS = (
    ("Engine and scheduling", "Engine core and scheduling"),
    ("KV cache and memory", "KV cache and memory"),
    ("Parallelism", "Parallelism and scale-out"),
    ("Sampling and generation", "Sampling and generation controls"),
    ("Structured output and tools", "Structured outputs and tool calling"),
    ("Speculative decoding", "Speculative decoding"),
    ("Serving, API, CLI, library", "Serving surface, CLI, and library"),
    ("LoRA and adapters", "LoRA and adapters"),
    ("Long context and attention", "Long context and attention breadth"),
    ("Loading, tokenizer, config", "Loading, tokenizer, and config"),
)

MATRIX_PATHS = [ENGINE_MATRIX, *(path for path, _ in MATRICES.values())]
REQUIRED = [
    ROOT / "AGENTS.md",
    ROOT / "README.md",
    ROOT / "docs/BENCHMARKS.md",
    AGENTS / "roadmap_v1.md",
    AGENTS / "coordination.md",
    AGENTS / "feature-matrix.md",
    AGENTS / "specs/model-family-inventory.md",
    AGENTS / "specs/feature-anchor-backfill.md",
    *MATRIX_PATHS,
]

STATES = {
    "INVENTORIED",
    "SPIKE",
    "READY",
    "ACTIVE",
    "GATING",
    "PARTIAL",
    "DONE",
    "BLOCKED",
    "ANCHOR-BACKFILL",
    "BUILD-ONLY",
    "UNTRACED",
}
READY_STATES = {"READY", "ACTIVE", "GATING", "DONE", "BLOCKED"}
EVIDENCED_STATES = {
    "PARTIAL",
    "ANCHOR-BACKFILL",
    "GATING",
    "DONE",
    "BUILD-ONLY",
    "UNTRACED",
}

ALL_PREFIXES = (*ENGINE_PREFIXES, *MATRICES)
ID_RE = re.compile(
    rf"(?:{'|'.join(re.escape(prefix) for prefix in ALL_PREFIXES)})-"
    r"[A-Za-z0-9_.-]+"
)
STATE_RE = re.compile(r"`(" + "|".join(re.escape(state) for state in STATES) + r")`")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
CLAIM_RE = re.compile(r"CLAIM-[A-Za-z0-9_.-]+")
LINE_FRAGMENT_RE = re.compile(r"L(\d+)(?:-L?(\d+))?")
COMMIT_RE = re.compile(r"[0-9a-f]{7,40}")
RAW_LOCAL_ANCHOR_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])"
    r"((?:src|include|tests|examples|cmake|scripts|tools|\.github/workflows)/"
    r"[A-Za-z0-9_./-]+|CMakeLists\.txt):(\d+)(?:-(\d+))?"
)

CODE_ANCHOR_PREFIXES = (
    "src/",
    "include/",
    "examples/",
    "cmake/",
    "scripts/",
    "tools/",
    ".github/workflows/",
)
TEST_ANCHOR_PREFIXES = (
    "tests/",
    "scripts/",
    ".github/workflows/",
)
CODE_ANCHOR_FILES = {"CMakeLists.txt"}
EVIDENCE_ANCHOR_FILES = {
    ".agents/parity-ledger.md",
}

SPEC_REQUIREMENTS = {
    "Scope": ("scope",),
    "Upstream chain": ("upstream chain",),
    "Our baseline": ("our baseline",),
    "Port map": ("port map", "port and harness map"),
    "Tests to port": ("tests to port",),
    "Gates": ("gates",),
    "Dependencies": ("dependencies",),
    "Work breakdown": (
        "work breakdown",
        "non overlapping work breakdown",
        "work decomposition",
    ),
    "Risks/decisions": ("risks", "risk /", "risk and", "risks/decisions"),
}


@dataclass(frozen=True)
class ClaimRow:
    path: Path
    line_no: int
    item_id: str
    state: str
    header: tuple[str, ...]
    cells: tuple[str, ...]
    raw: str

    def field(self, name: str) -> str:
        index = field_index(self.header, name)
        return self.cells[index] if index is not None and index < len(self.cells) else ""


def markdown_files() -> list[Path]:
    return [
        ROOT / "AGENTS.md",
        ROOT / "README.md",
        ROOT / "docs/BENCHMARKS.md",
        *sorted(AGENTS.rglob("*.md")),
    ]


def split_cells(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", body)]


def normalize_header(value: str) -> str:
    value = value.replace("`", "").replace("*", "").lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def field_index(header: tuple[str, ...], field: str) -> int | None:
    for index, value in enumerate(header):
        if field == "id" and value == "id":
            return index
        if field == "item" and (
            value.startswith("item")
            or value == "encoding"
            or "method scheme" in value
            or "cache dtype mode" in value
            or value == "scope"
            or value == "surface"
        ):
            return index
        if field == "upstream" and "upstream" in value:
            return index
        if field == "code" and ("our code" in value or "local evidence" in value):
            return index
        if field == "tests" and (
            "tests evidence" in value
            or "test evidence" in value
            or "local evidence" in value
            or value == "evidence"
        ):
            return index
        if field == "spec" and "spike" in value:
            return index
        if field == "state" and (value == "state" or value.endswith(" state")):
            return index
        if field == "owner" and "owner" in value:
            return index
    return None


def is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def parse_claim_rows(path: Path, errors: list[str]) -> list[ClaimRow]:
    rows: list[ClaimRow] = []
    header: tuple[str, ...] | None = None
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|"):
            header = None
            continue
        cells = split_cells(line)
        normalized = tuple(normalize_header(cell) for cell in cells)
        if "id" in normalized and any(value == "state" or value.endswith(" state") for value in normalized):
            header = normalized
            continue
        if is_separator(cells):
            continue
        if header is None or not cells:
            continue

        item_id = cells[0].strip().strip("`")
        if not ID_RE.fullmatch(item_id):
            continue
        # A row that fails its shape check is REPORTED and still parsed, never
        # dropped: dropping it hid every downstream contract defect behind the
        # shape error and corrupted the matrix ratchet counts, so each fix
        # exposed the next defect only after another run (three defects masked
        # this way in the ORPHAN-MODEL-ROWS repair;
        # ISSUE-LOCAL-01M3NC14GE995V9E6F7GTYSQJ3). The one error list reports
        # the shape and the contracts in the same run.
        malformed = False
        if len(cells) != len(header):
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: {item_id} has {len(cells)} cells; "
                f"header has {len(header)}"
            )
            malformed = True
        state_index = field_index(header, "state")
        state_cell = cells[state_index] if state_index is not None else ""
        state_matches = STATE_RE.findall(state_cell)
        if len(state_matches) != 1:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: {item_id} must have exactly one canonical state"
            )
            malformed = True
        if malformed:
            # Parsed with an empty state so the ratchet, duplicate detection
            # and the summary rollups still see the row; no state-conditional
            # contract can fire on the empty state, so the reported defects
            # stay the shape ones.
            rows.append(
                ClaimRow(path, line_no, item_id, state_matches[0] if state_matches else "", header, tuple(cells), line)
            )
            continue
        rows.append(
            ClaimRow(path, line_no, item_id, state_matches[0], header, tuple(cells), line)
        )
    return rows


def link_base(source: Path, text: str) -> Path:
    """Resolve migrated legacy links from their original .agents/ location."""
    if (
        source.is_relative_to(AGENTS / "completed/state-events")
        and "<!-- legacy-payload:begin -->" in text
    ):
        return AGENTS
    return source.parent


def check_links(errors: list[str]) -> None:
    for source in markdown_files():
        text = source.read_text(encoding="utf-8")
        base = link_base(source, text)
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            target_path, _, fragment = target.partition("#")
            resolved = (base / target_path).resolve()
            if not resolved.exists():
                errors.append(f"{source.relative_to(ROOT)}: dangling link {raw_target}")
                continue
            line_match = LINE_FRAGMENT_RE.fullmatch(fragment)
            if line_match is None or not resolved.is_file():
                continue
            line_count = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
            start = int(line_match.group(1))
            end = int(line_match.group(2) or start)
            if start < 1 or end < start or end > line_count:
                errors.append(
                    f"{source.relative_to(ROOT)}: out-of-range line anchor {raw_target} "
                    f"(file has {line_count} lines)"
                )


def table_ids(prefix: str, path: Path, errors: list[str]) -> list[str]:
    return [
        row.item_id
        for row in parse_claim_rows(path, errors)
        if row.item_id.startswith(prefix + "-")
    ]


def check_matrices(errors: list[str]) -> tuple[list[ClaimRow], dict[str, ClaimRow]]:
    rows: list[ClaimRow] = []
    for path in MATRIX_PATHS:
        rows.extend(parse_claim_rows(path, errors))

    by_id: dict[str, ClaimRow] = {}
    for row in rows:
        prior = by_id.get(row.item_id)
        if prior is not None:
            errors.append(
                f"{row.path.relative_to(ROOT)}:{row.line_no}: duplicate ID {row.item_id}; "
                f"first at {prior.path.relative_to(ROOT)}:{prior.line_no}"
            )
        else:
            by_id[row.item_id] = row

    for prefix, (path, expected) in MATRICES.items():
        count = sum(row.item_id.startswith(prefix + "-") for row in rows if row.path == path)
        if count != expected:
            errors.append(f"{path.relative_to(ROOT)}: {count} {prefix} rows; expected {expected}")

    engine_count = sum(
        any(row.item_id.startswith(prefix + "-") for prefix in ENGINE_PREFIXES)
        for row in rows
        if row.path == ENGINE_MATRIX
    )
    if engine_count != ENGINE_ROWS:
        errors.append(
            f"{ENGINE_MATRIX.relative_to(ROOT)}: {engine_count} engine rows; expected {ENGINE_ROWS}"
        )
    return rows, by_id


def check_engine_summary(rows: list[ClaimRow], errors: list[str]) -> None:
    lines = ENGINE_MATRIX.read_text(encoding="utf-8").splitlines()
    header: list[str] | None = None
    total: list[str] | None = None
    summaries: dict[str, list[str]] = {}
    section_lines: dict[str, int] = {}
    for line_no, line in enumerate(lines, 1):
        if line.startswith("## "):
            section_lines[line.removeprefix("## ").strip()] = line_no
        if line.startswith("| Area | Rows |"):
            header = [normalize_header(cell) for cell in split_cells(line)]
        elif header is not None and total is None and line.startswith("|"):
            cells = [cell.replace("*", "").strip() for cell in split_cells(line)]
            if is_separator(cells):
                continue
            if cells[0] == "Total":
                total = cells
            else:
                summaries[cells[0]] = cells
    if header is None or total is None or len(header) != len(total):
        errors.append(f"{ENGINE_MATRIX.relative_to(ROOT)}: missing or malformed lifecycle summary")
        return

    actual_rows = [row for row in rows if row.path == ENGINE_MATRIX]

    def check_counts(label: str, recorded_cells: list[str], scoped_rows: list[ClaimRow]) -> None:
        if len(recorded_cells) != len(header):
            errors.append(f"{ENGINE_MATRIX.relative_to(ROOT)}: malformed {label} lifecycle summary")
            return
        expected = {"rows": len(scoped_rows)}
        expected.update(
            {normalize_header(state): sum(row.state == state for row in scoped_rows) for state in STATES}
        )
        for index, name in enumerate(header[1:], 1):
            if name not in expected:
                continue
            try:
                recorded = int(recorded_cells[index])
            except ValueError:
                errors.append(
                    f"{ENGINE_MATRIX.relative_to(ROOT)}: non-numeric {label} summary for {name}"
                )
                continue
            if recorded != expected[name]:
                errors.append(
                    f"{ENGINE_MATRIX.relative_to(ROOT)}: {label} summary {name}={recorded}; "
                    f"actual {expected[name]}"
                )

    check_counts("total", total, actual_rows)
    for area, section in ENGINE_SUMMARY_SECTIONS:
        recorded_cells = summaries.get(area)
        section_line = section_lines.get(section)
        if recorded_cells is None or section_line is None:
            errors.append(f"{ENGINE_MATRIX.relative_to(ROOT)}: missing {area} lifecycle summary")
            continue
        next_section_line = min(
            (line_no for line_no in section_lines.values() if line_no > section_line),
            default=len(lines) + 1,
        )
        scoped_rows = [
            row for row in actual_rows if section_line < row.line_no < next_section_line
        ]
        check_counts(area, recorded_cells, scoped_rows)


def is_placeholder(value: str) -> bool:
    normalized = value.strip().strip("`").lower()
    return not normalized or normalized in {"-", "none", "unassigned", "open", "leaf open"}


def local_line_anchors(value: str, source: Path) -> list[str]:
    anchors: list[str] = []
    for raw_target in LINK_RE.findall(value):
        target = raw_target.strip().strip("<>")
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        target_path, _, fragment = target.partition("#")
        line_match = LINE_FRAGMENT_RE.fullmatch(fragment)
        if line_match is None:
            continue
        resolved = (source.parent / target_path).resolve()
        if not resolved.is_file():
            continue
        line_count = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
        start = int(line_match.group(1))
        end = int(line_match.group(2) or start)
        if start < 1 or end < start or end > line_count:
            continue
        try:
            anchors.append(resolved.relative_to(ROOT).as_posix())
        except ValueError:
            continue
    for match in RAW_LOCAL_ANCHOR_RE.finditer(value):
        resolved = (ROOT / match.group(1)).resolve()
        if not resolved.is_file():
            continue
        line_count = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
        start = int(match.group(2))
        end = int(match.group(3) or start)
        if start < 1 or end < start or end > line_count:
            continue
        try:
            anchors.append(resolved.relative_to(ROOT).as_posix())
        except ValueError:
            continue
    return anchors


def is_code_anchor(value: str, source: Path) -> bool:
    if is_placeholder(value):
        return False
    return any(
        path in CODE_ANCHOR_FILES or path.startswith(CODE_ANCHOR_PREFIXES)
        for path in local_line_anchors(value, source)
    )


def is_test_anchor(value: str, source: Path) -> bool:
    if is_placeholder(value):
        return False
    return any(
        path in EVIDENCE_ANCHOR_FILES or path.startswith(TEST_ANCHOR_PREFIXES)
        for path in local_line_anchors(value, source)
    )


def ledger_line_anchors(value: str, source: Path) -> list[str]:
    return [
        path
        for path in local_line_anchors(value, source)
        if path == ".agents/parity-ledger.md"
    ]


def commit_exists(commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def local_spec_paths(row: ClaimRow) -> list[Path]:
    paths: list[Path] = []
    spec_root = (AGENTS / "specs").resolve()
    for raw_target in LINK_RE.findall(row.field("spec")):
        target = raw_target.strip().strip("<>").split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (row.path.parent / target).resolve()
        try:
            resolved.relative_to(spec_root)
        except ValueError:
            continue
        if resolved.suffix == ".md" and resolved.is_file():
            paths.append(resolved)
    return paths


def has_substantive_spec_content(lines: list[str]) -> bool:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or re.match(r"^#{1,6}\s+", stripped) or stripped.startswith("```"):
            continue
        if stripped.startswith("|"):
            cells = split_cells(stripped)
            if is_separator(cells):
                continue
            next_nonempty = next(
                (candidate.strip() for candidate in lines[index + 1 :] if candidate.strip()),
                "",
            )
            if next_nonempty.startswith("|") and is_separator(split_cells(next_nonempty)):
                continue
            if len(cells) >= 2 and any(not is_placeholder(cell) for cell in cells[1:]):
                return True
            continue
        if not is_placeholder(stripped):
            return True
    return False


def structured_spec_fields(text: str) -> set[str]:
    lines = text.splitlines()
    fields: set[str] = set()
    for index, line in enumerate(lines):
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading is not None:
            level = len(line) - len(line.lstrip("#"))
            end = len(lines)
            for candidate in range(index + 1, len(lines)):
                next_heading = re.match(r"^(#{1,6})\s+", lines[candidate])
                if next_heading is not None and len(next_heading.group(1)) <= level:
                    end = candidate
                    break
            if has_substantive_spec_content(lines[index + 1 : end]):
                fields.add(normalize_header(heading.group(1)))
            continue
        if line.startswith("|"):
            cells = split_cells(line)
            next_nonempty = next(
                (
                    candidate.strip()
                    for candidate in lines[index + 1 :]
                    if candidate.strip()
                ),
                "",
            )
            is_header = next_nonempty.startswith("|") and is_separator(
                split_cells(next_nonempty)
            )
            if (
                len(cells) >= 2
                and not is_header
                and not is_separator(cells)
                and any(not is_placeholder(cell) for cell in cells[1:])
            ):
                fields.add(normalize_header(cells[0]))
    return fields


def missing_spec_requirements(text: str) -> list[str]:
    fields = structured_spec_fields(text)
    missing: list[str] = []
    for label, alternatives in SPEC_REQUIREMENTS.items():
        normalized = [normalize_header(alternative) for alternative in alternatives]
        if not any(
            field == alternative or field.startswith(alternative + " ")
            for field in fields
            for alternative in normalized
        ):
            missing.append(label)
    return missing


def check_spec(row: ClaimRow, errors: list[str]) -> None:
    specs = local_spec_paths(row)
    location = f"{row.path.relative_to(ROOT)}:{row.line_no}"
    if not specs:
        errors.append(f"{location}: {row.item_id} {row.state} has no real .agents/specs link")
        return
    token = f"`{row.item_id}`"
    matching = [path for path in specs if token in path.read_text(encoding="utf-8")]
    if not matching:
        errors.append(f"{location}: no linked spec names exact stable token `{row.item_id}`")
        return
    text = matching[0].read_text(encoding="utf-8")
    for label in missing_spec_requirements(text):
        errors.append(
            f"{matching[0].relative_to(ROOT)}: spec for {row.item_id} lacks structured {label}"
        )


def claim_sources() -> list[Path]:
    """Every file that may carry an active claim row.

    One file per claim in .agents/claims/ (ENG-RECORD-CONFLICT-SURFACES, #364),
    plus the legacy table in coordination.md. Both are read, so a claim is
    equally valid in either and no existing row had to be migrated -- the table
    empties as its claims close.

    The per-claim file exists because the table is insert-at-one-anchor: every
    concurrent claim appended at the same line, which made coordination.md the
    largest single conflict source in the repository (8 of the 16 conflicting
    open PRs at origin/main d928e2c3, six of them one author's sequential ROCm
    stack whose only conflict was this). A file with one writer cannot collide.
    """
    sources = [AGENTS / "coordination.md"]
    claims_dir = AGENTS / "claims"
    if claims_dir.is_dir():
        sources.extend(sorted(claims_dir.glob("CLAIM-*.md")))
    return sources


def parse_active_claims(errors: list[str]) -> dict[str, set[str]]:
    claims: dict[str, set[str]] = {}
    origin: dict[str, str] = {}
    for path in claim_sources():
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.startswith("| `CLAIM-"):
                continue
            cells = split_cells(line)
            claim_match = CLAIM_RE.search(cells[0])
            if claim_match is None:
                continue
            claim = claim_match.group(0)
            if claim in claims:
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_no}: duplicate active claim "
                    f"{claim} (already declared in {origin[claim]})"
                )
                continue
            origin[claim] = str(path.relative_to(ROOT))
            claims[claim] = set(ID_RE.findall(cells[1])) if len(cells) > 1 else set()
    return claims


def check_row_contracts(
    rows: list[ClaimRow], by_id: dict[str, ClaimRow], errors: list[str]
) -> None:
    active_claims = parse_active_claims(errors)
    required_fields = ("id", "item", "upstream", "code", "tests", "spec", "state", "owner")

    for row in rows:
        location = f"{row.path.relative_to(ROOT)}:{row.line_no}"
        for field in required_fields:
            if field_index(row.header, field) is None:
                errors.append(f"{location}: {row.item_id} table lacks semantic {field} column")

        if is_placeholder(row.field("item")):
            errors.append(f"{location}: {row.item_id} has no item description")
        if is_placeholder(row.field("upstream")):
            errors.append(f"{location}: {row.item_id} has no upstream anchor/target")

        if row.state in EVIDENCED_STATES:
            if not is_code_anchor(row.field("code"), row.path):
                errors.append(f"{location}: {row.item_id} {row.state} lacks exact local code anchor")
            if not is_test_anchor(row.field("tests"), row.path):
                errors.append(f"{location}: {row.item_id} {row.state} lacks exact test/evidence anchor")

        if row.state in READY_STATES:
            check_spec(row, errors)

        if row.state in {"SPIKE", "ACTIVE"}:
            claim_match = CLAIM_RE.search(row.field("owner"))
            if claim_match is None:
                errors.append(f"{location}: {row.state} row {row.item_id} has no CLAIM-* owner")
            else:
                claim = claim_match.group(0)
                if row.item_id not in active_claims.get(claim, set()):
                    errors.append(
                        f"{location}: owner {claim} does not claim active row {row.item_id} in any claim source"
                    )

        if row.state == "DONE":
            if not ledger_line_anchors(row.field("tests"), row.path):
                errors.append(f"{location}: DONE row {row.item_id} lacks exact parity-ledger link")
            owner = row.field("owner").strip().strip("`")
            if COMMIT_RE.fullmatch(owner) is None:
                errors.append(
                    f"{location}: DONE row {row.item_id} owner is not the hexadecimal closing commit"
                )
            elif not commit_exists(owner):
                errors.append(
                    f"{location}: DONE row {row.item_id} closing commit {owner} does not exist"
                )

    for claim, item_ids in active_claims.items():
        if not item_ids:
            errors.append(f"active claim {claim} has no stable row IDs")
        for item_id in item_ids:
            row = by_id.get(item_id)
            if row is None:
                errors.append(f"active claim {claim} references unknown row {item_id}")
            elif row.state not in {"SPIKE", "ACTIVE"}:
                errors.append(
                    f"active claim {claim} references {item_id} in state {row.state}, not SPIKE/ACTIVE"
                )


def check_model_invariants(errors: list[str]) -> None:
    path = AGENTS / "model-matrix.md"
    rows: list[tuple[list[str], str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `MODEL-"):
            continue
        cells = split_cells(line)
        if len(cells) < 3:
            continue
        aliases = re.findall(r"`([^`]+)`", cells[1])
        targets = [value for value in re.findall(r"`([^`]+)`", cells[2]) if "::" in value]
        if aliases and targets:
            rows.append((aliases, targets[-1]))

    actual = {
        "rows": len(rows),
        "memberships": sum(len(aliases) for aliases, _ in rows),
        "architectures": len({alias for aliases, _ in rows for alias in aliases}),
        "targets": len({target for _, target in rows}),
        "modules": len({target.split("::", 1)[0] for _, target in rows}),
    }
    # Regenerated from the matrix rows: the 4b0939230 record correction
    # deduplicated the target and module citations (a relocation and a
    # Transformers-backend migration collapsed five stale upstream anchors), and
    # the MODEL-LAYA row that briefly raised these two numbers again was closed
    # by bc5dbc367. The record below is what the current inventory rows sum to.
    expected = {
        "rows": 324,
        "memberships": 373,
        "architectures": 356,
        "targets": 309,
        "modules": 245,
    }
    if actual != expected:
        errors.append(f"{path.relative_to(ROOT)}: model inventory {actual}, expected {expected}")


def check_table_shapes(paths: list[Path], errors: list[str]) -> None:
    for path in sorted(set(paths)):
        expected_pipes: int | None = None
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.startswith("|"):
                expected_pipes = None
                continue
            pipes = len(re.findall(r"(?<!\\)\|", line))
            if expected_pipes is None:
                expected_pipes = pipes
            elif pipes != expected_pipes:
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_no}: table has {pipes} pipes; expected {expected_pipes}"
                )


def check_spec_location(errors: list[str]) -> None:
    misplaced = re.compile(r"(?:spec|scoping|semantics|feasibility|notes)", re.I)
    allowed = {"benchmark-protocol.md"}
    for path in AGENTS.glob("*.md"):
        if path.name not in allowed and misplaced.search(path.stem):
            errors.append(f"{path.relative_to(ROOT)}: feature spec/scoping file belongs in .agents/specs/")


ISSUE_ROW = re.compile(
    r"^\|\s*\[#(\d+)\]\((https://github\.com/[^)]+/issues/(\d+))\)\s*\|"
    r"\s*(?:`([A-Z0-9][A-Za-z0-9_.-]*)`|—)\s*\|"
)


def check_issue_table(errors: list[str]) -> None:
    """Every tracked issue is well-formed and its row link is consistent.

    Deliberately NETWORK-FREE. Querying GitHub would make this gate fail on
    connectivity, which is exactly the class of flake this protocol exists to
    remove. It checks the FORM and the internal consistency; whether the issue
    is still open is the agent's job at intake, not a CI blocker.
    """

    path = AGENTS / "roadmap_v1.md"
    text = path.read_text(encoding="utf-8")
    if "## Open issues" not in text:
        errors.append(f"{path.relative_to(ROOT)}: missing the '## Open issues' intake table")
        return

    section = text.split("## Open issues", 1)[1].split("\n## ", 1)[0]
    seen: set[str] = set()
    rows = 0
    for line in section.splitlines():
        if not line.startswith("|") or line.startswith("| Issue") or set(line) <= set("|-: "):
            continue
        match = ISSUE_ROW.match(line)
        if not match:
            errors.append(
                f"{path.relative_to(ROOT)}: malformed issue row {line[:60]!r}; "
                "expected | [#N](https://github.com/.../issues/N) | `ROW-ID` or — | title | kind |"
            )
            continue
        rows += 1
        number, url, url_number, row_id = match.group(1), match.group(2), match.group(3), match.group(4)
        if number != url_number:
            errors.append(
                f"{path.relative_to(ROOT)}: issue #{number} links to {url}, a different issue"
            )
        if number in seen:
            errors.append(f"{path.relative_to(ROOT)}: issue #{number} listed twice")
        seen.add(number)
    if rows == 0:
        errors.append(f"{path.relative_to(ROOT)}: the open-issue table has no rows")


def check_roadmap(by_id: dict[str, ClaimRow], errors: list[str]) -> None:
    path = AGENTS / "roadmap_v1.md"
    expected_blocks = [
        "ROAD-V1-A",
        "ROAD-V1-C1",
        "ROAD-V1-C2",
        "ROAD-V1-C3",
        "ROAD-V1-C4",
        "ROAD-V1-C5",
        "ROAD-V1-C6",
        "ROAD-V1-C7",
        "ROAD-V1-C8",
        "ROAD-V1-C9",
        "ROAD-V1-D1",
        "ROAD-V1-D2",
        "ROAD-V1-D3",
        "ROAD-V1-D4",
        "ROAD-V1-D5",
        # +D6 2026-08-05: llama.cpp device breadth folded into scope (user-directed).
        "ROAD-V1-D6",
    ]
    seen: list[tuple[int, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|"):
            continue
        cells = split_cells(line)
        if len(cells) < 6 or not cells[0].isdigit():
            continue
        block_id = cells[1].strip().strip("`")
        if not re.fullmatch(r"ROAD-V1-[A-Z0-9]+", block_id):
            continue
        seen.append((int(cells[0]), block_id))
        if STATE_RE.fullmatch(cells[5].strip()) is None:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: portfolio State cell needs exactly one state"
            )
    if seen != list(enumerate(expected_blocks)):
        errors.append(f"{path.relative_to(ROOT)}: portfolio order {seen}; expected {list(enumerate(expected_blocks))}")

    for source in (path, AGENTS / "coordination.md"):
        text = source.read_text(encoding="utf-8")
        for item_id in set(re.findall(r"`(" + ID_RE.pattern + r")`", text)):
            if item_id not in by_id:
                errors.append(f"{source.relative_to(ROOT)}: references unknown stable row {item_id}")


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED:
        if not path.is_file():
            errors.append(f"missing canonical record: {path.relative_to(ROOT)}")

    rows: list[ClaimRow] = []
    by_id: dict[str, ClaimRow] = {}
    if not errors:
        check_links(errors)
        check_issue_table(errors)
        rows, by_id = check_matrices(errors)
        check_engine_summary(rows, errors)
        check_row_contracts(rows, by_id, errors)
        check_model_invariants(errors)
        spec_paths = [path for row in rows if row.state in READY_STATES for path in local_spec_paths(row)]
        check_table_shapes(
            [AGENTS / "roadmap_v1.md", AGENTS / "coordination.md", *MATRIX_PATHS, *spec_paths],
            errors,
        )
        check_spec_location(errors)
        check_roadmap(by_id, errors)

    if errors:
        for error in dict.fromkeys(errors):
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    counts = [
        "ENGINE="
        + str(
            sum(
                any(row.item_id.startswith(prefix + "-") for prefix in ENGINE_PREFIXES)
                for row in rows
                if row.path == ENGINE_MATRIX
            )
        )
    ]
    for prefix, (path, _) in MATRICES.items():
        counts.append(
            f"{prefix}="
            + str(sum(row.item_id.startswith(prefix + "-") for row in rows if row.path == path))
        )
    print("agent record OK: " + " ".join(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
