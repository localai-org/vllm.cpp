# FP8 merged QKV projection (35B c1/c2 launch-reduction lever)

Row: `KERNEL-GEMM-FP8` (merged-QKV FP8 sub-lever). Claim: `CLAIM-FP8-MERGED-QKV-1`.

## Scope

In: extend the fp4-only merged-QKV projection fusion to the 35B **FP8 (W8A8)**
full-attention path. The three logical Q/K/V shards (`ProjectFullAttnQkv`,
`qwen3_5.cpp`) run as THREE separate `MatmulFp8CutlassD`/`PreQuantD` GEMMs on
35B; collapse them into ONE fp8 GEMM over the N-concatenated operand + split.
Target the M=1 decode launch/heuristic overhead that dominates the last 35B
c1/c2 residual (~2-4%, low-batch).

Out (this row): GDN `in_proj_qkvz`/`z` merge and GDN `BA` merge (30 layers) —
follow-up sub-levers; the same primitive (`MulColVecF32`) + resident-concat
pattern applies, but GDN qkv/z carry DIFFERENT output dtypes (qkv F32, z
`outdt`) so the split needs an extra cast branch. MoE `gate_up` fp8 (already
per-shard-equal in the checkpoint) is a separate row.

## Upstream chain (mirror)

- vLLM merges Q/K/V into ONE `QKVParallelLinear` parameter and runs ONE GEMM
  before `torch.split`: `vllm/model_executor/models/qwen3_5.py:279-288`,
  `vllm/model_executor/layers/linear.py:942-1050`.
- Per-tensor FP8 merged handling: `requantize_with_max_scale`
  (`vllm/model_executor/layers/quantization/utils/w8a8_utils.py:76-107`) —
  dequant each shard, requant all to the MAX weight_scale, ONE per-tensor scale.
- Runtime: `torch._scaled_mm` / cuBLASLt fp8 (`nvjet_sm121_qqtst_*`), one GEMM.

## Our baseline

- fp4 merge (the infra we extend): `ResidentNvfp4Qkv` (`qwen3_5.cpp:991`) byte-
  concatenates the packed rows + per-block scales along N (swizzle-once) and
  `MergedQkvCutlassD` (`qwen3_5.cpp:1668`) runs ONE `MatmulNvfp4CutlassModel`;
  `MergedQkvEligible` (`qwen3_5.cpp:1607`) gates it FP4-only.
- fp8 split path: `ProjectFullAttnQkv` (`qwen3_5.cpp:~1808`) `project` lambda ->
  `MatmulFp8CutlassPreQuantD`/`MatmulFp8CutlassD` (`qwen3_5.cpp:1199,1221`),
  three separate GEMMs, F32 output.

## KEY FINDING — fp8 is PER-TENSOR scaled (the brief's byte-exact premise refined)

Unlike fp4 (per-BLOCK scales that concatenate losslessly under ONE global
alpha), the 35B fp8 weights are PER-TENSOR: each `Fp8Weight` carries a single
`weight_scale` and a folded scalar `alpha = input_scale * weight_scale`
(`qwen3_5_weights.h:135-147`). A single-scalar-alpha GEMM over the concatenated
Q/K/V weight is therefore NOT correct — each shard needs its own alpha. Two
faithful realizations:

- **(A) per-column alpha (CHOSEN):** concat the RAW fp8 bytes (unchanged) into
  `[Nq+Nk+Nv,K]`, run ONE fp8 GEMM with `alpha=1` (raw f32 accumulation), then
  apply each output column's shard scalar via a resident per-column vector
  (`vt::MulColVecF32`, new op). Because Q/K/V share ONE `input_scale`, the
  activation quant is identical across them, so this is **byte-identical to the
  three separate GEMMs** iff cuBLASLt picks the same per-column accumulation for
  the larger-N GEMM. This mirrors the fp4 merge's per-column-scale philosophy
  and needs NO weight requant / fp8 host encoder.
- (B) requantize_with_max_scale (vLLM's exact per-tensor mirror): dequant+requant
  each shard to the max scale, one scalar alpha, reuse the GEMM directly. Needs
  an fp8 requant pass and is NOT byte-exact vs our separate path (requant re-
  rounds). Rejected as the first cut: more numerics risk, more surface.

## Port map

| Upstream / pattern | Local |
|---|---|
| new per-column dequant primitive | `vt::MulColVecF32` — `ops.h` (enum `kMulColVecF32`, decl), `ops.cpp` (validate+dispatch), `cpu_ops.cpp` (`MulColVecF32Kernel`), `cuda_glue.cu` (`MulColVecF32KernelCuda` + register) |
| `ResidentNvfp4Qkv` byte-concat | `ResidentFp8Qkv` (`qwen3_5.cpp`) — concat raw fp8 bytes + build per-column alpha vector, resident-once |
| `MergedQkvEligible` (fp4) | `MergedFp8QkvEnabled`/`MergedFp8QkvEligible` (`VT_FP8_MERGED_QKV`, opt-in) |
| `MergedQkvCutlassD` (fp4) | `MergedFp8QkvD` — ONE `MatmulFp8CublasLt`(alpha=1) + `MulColVecF32` |
| resident fields | `FullAttnLayerWeights::d_qkv_fp8_packed`/`d_qkv_fp8_alpha` (`qwen3_5_weights.h`) |

## Tests to port / add

- `vt::MulColVecF32` byte-exact CPU refs (contiguous + padded-row/strided packed
  view): `tests/vt/test_ops_glue.cpp` (DONE, exact `==`).
- CUDA (DGX): byte-exact `merged == separate` on the 35B q/k/v shapes (verify
  cuBLASLt tiling matches at M=1/decode); if NOT byte-exact, the token gate is
  the bar.
- Model gate: 35B 315/315 (default OFF unaffected + `VT_FP8_MERGED_QKV=1` arm
  token-exact) + 27B 235/235 (fp8-gated, 27B inert).

## Gates

1. CPU: clean -Werror build, full ctest, tools 164, checkers. (DONE local:
   glue 10/10 incl. 2 new, fp8_cutlass 6/6, matmul 7/7, clean build.)
2. Correctness (DGX, one flock): merged==separate byte-exact OR 315/315 token
   gate; 27B 235/235; memcheck.
3. Perf (DGX): in-situ 35B c1/c2/c4 A/B (`VT_FP8_MERGED_QKV` on/off, same
   binary), c8+ neutral/win; launches/step before/after (30->10 attn QKV GEMMs).
   Build flags: `-DVLLM_CPP_CUTLASS_DIR=$HOME/venvs/vllm-oracle/.../flashinfer/
   data/cutlass -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
   -DVLLM_CPP_TRITON=ON`.

## Land

Flip `VT_FP8_MERGED_QKV` default ON for 35B fp8 iff byte/token-exact + faster;
else land opt-in honest. Records: this spec, `KERNEL-GEMM-FP8`, ledger, state,
README/BENCHMARKS, coordination — same commit.

## Dependencies / risks

- Depends on the shared-`input_scale` invariant (guarded; the real 35B q/k/v
  share it). GGUF/synthetic/27B leave fp8 fields empty -> inert.
- Risk: cuBLASLt may tile the larger-N GEMM differently -> not byte-exact ->
  token-gate fallback (a near-tie greedy flip is the only blocker).
- Product calls: none (vLLM-defined behavior).
## Owed

- **`ISSUE-GH-469` — `test_ops_glue.cpp:190`'s `CHECK_THROWS` is satisfied by
  the CPU kernel's second guard, not the dispatch guard it names**, so mutation
  M8 survives. This spec introduced `vt::MulColVecF32` together with its
  validate+dispatch arm, which is the guard the case intends to pin. Behavior is
  correct; test strength only.
