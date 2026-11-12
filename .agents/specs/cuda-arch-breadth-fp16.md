# CUDA architecture breadth — the fp16 / non-tensor-core lane (Pascal / Volta / Turing)

Status: **SPIKE (scoping) — DERIVE-AND-SHIP framed. W1 (Turing `sm_75` bf16-WMMA
guard) has LANDED and is BUILD-VERIFIED + SASS on nvcc 13.0 (base `034be66e`,
`CLAIM-CUDA-TURING-SM75`); W2+ (the fp16 fast body) and all runtime gates remain
open. V0 (the full-library compile audit W1 never ran) is MEASURED as of
2026-08-06 — see §V0: 18 of 20 unconditional CUDA TUs already compile clean at
`sm_75`, and the residual is exactly TWO files.** Read-only on production code
except the W1 guard; this document is the
committed scope for a "support MORE than vLLM" lane
that vLLM **drops** but llama.cpp **still supports**: the pre-Ampere NVIDIA
arches Pascal (`sm_60`/`sm_61`), Volta (`sm_70`) and Turing (`sm_75`). The port
source is **llama.cpp's `ggml-cuda`** (vLLM has NO kernel for any of these); the
competitor floor is **llama.cpp on the same old card**. Nothing here claims
runtime support; the deliverable moves the Volta/Turing rows to `SPIKE` and
proposes new Pascal rows, all honestly scoped.

Owner: `CLAIM-CUDA-BREADTH-SCOPE`.
Rows: `BACKEND-CUDA-SM075` (Turing), `BACKEND-CUDA-SM070` (Volta), NEW
`BACKEND-CUDA-SM060`/`BACKEND-CUDA-SM061` (Pascal); competitor floor NEW
`BACKEND-GATE-CUDA-LLAMACPP-LEGACY`.
Roadmap wiring: `ROAD-V1-D1` (NVIDIA target fan-out), the "beyond-vLLM breadth"
leaf.

Pins: llama.cpp `/home/mudler/_git/llama.cpp` @ **`237ad9b96`** (`ggml-cuda`;
the same llama.cpp pin the Vulkan row already cites); ggml `c044a8ee`. vLLM
`555967922` (0.26.0.dev0) — for the record only, it has no kernel here. vllm.cpp
base `11e0ba36`.

---

## The core constraint (measured, not assumed)

Our portable paged-attention TU instantiates **bf16 WMMA** tensor-core fragments
at file scope:

```
// src/vt/cuda/cuda_paged_attn.cu:706-717
namespace attn_wmma = nvcuda::wmma;
using AccFrag = attn_wmma::fragment<attn_wmma::accumulator, 16,16,16, float>;
attn_wmma::fragment<attn_wmma::matrix_a, 16,16,16, __nv_bfloat16, row_major>;   // :713
attn_wmma::fragment<attn_wmma::matrix_b, 16,16,16, __nv_bfloat16, col_major>;   // :715
```

`nvcuda::wmma::fragment<…,__nv_bfloat16,…>` is only a **complete type on
`sm_80`+** (bf16 tensor cores are Ampere and later). On Turing/Volta/Pascal the
type is incomplete and the whole TU fails to compile — the exact failure W10
measured on dgx: `incomplete type "…fragment<matrix_a,16,16,16,__nv_bfloat16,…>"`
at `cuda_paged_attn.cu:1797`. Independently, **nvcc 13.0 rejects `sm_70`
outright** (`nvcc fatal: Unsupported gpu architecture 'sm_70'`), and by the same
CUDA-13 arch drop, `sm_60`/`sm_61` as well; only `sm_75` is accepted by nvcc
13.0 (it compiles a trivial kernel, but not our attn TU). So this lane is a
**new fp16 / non-tensor-core kernel body**, exactly what these rows have said
they need — and llama.cpp is where that body already exists.

**One de-risking baseline already in the tree.** We are NOT starting from zero:
`cuda_paged_attn.cu` already carries a **scalar CUDA-core register-tiled flash
path** reachable at runtime via `VT_ATTN_WMMA=0`
(`cuda_paged_attn.cu:2349-2352`, "falls back to the CUDA-core register-tiled
flash path"). It runs no tensor cores. What blocks the old arches is not the
*absence* of a non-tensor-core path but that the bf16-WMMA fragment types are
**compiled unconditionally** even when that path is selected. The cheapest
correct first step is therefore a compile-time guard, not a new kernel (see
§Work breakdown W1).

## Toolkit split — precise, because it decides what ships NOW

| Arch | CC | nvcc 13.0 | Buildable today? | Lane |
|---|---|---|---|---|
| Turing | `sm_75` | **accepts** (compiles trivial + fp16 WMMA) | **YES**, once the bf16-WMMA TU is guarded and an fp16 body lands | **DERIVE-AND-SHIP candidate NOW** |
| Volta | `sm_70` | **rejects** (`Unsupported gpu architecture 'sm_70'`) | NO — needs a CUDA **< 13** toolkit | NOT-YET-BUILDABLE (blocked on `<13` toolkit) |
| Pascal | `sm_60`/`sm_61` | **rejects** (CUDA 13 dropped Pascal offline codegen) | NO — needs a CUDA **< 13** toolkit | NOT-YET-BUILDABLE (blocked on `<13` toolkit) |

A CUDA `< 13` toolkit is **not provisioned anywhere in this environment** (`ls
/usr/local/cuda*` = none on the dev box; dgx builds with `cuda-13.0`). So Volta
and Pascal cannot even be BUILD-verified here until a 12.x toolkit is wired;
they are honestly `NOT-YET-BUILDABLE`, not "scoped forever". Turing is the only
one that can be derived, compiled and SASS-verified on the toolkit we already
run.

## The per-arch SIGNAL matrix (help-wanted: hardware testing)

The whole point of this lane is beyond-vLLM breadth, shipped the way llama.cpp
itself ships old-arch support: **derived, build-verified, honestly labeled
untested, community testing welcome.** Each row therefore carries an explicit
SIGNAL:

| SIGNAL | Meaning |
|---|---|
| `RUNTIME-VERIFIED` | A gate model ran token-checked on the real card here (none of these are, and no such card exists here) |
| `DERIVED+BUILD-VERIFIED (testing-welcome)` | fp16 body ported 1:1 from llama.cpp (cited file:line), compiles `-Werror` clean, real per-arch SASS in `cuobjdump`; NOT hardware-tested here — help wanted |
| `NOT-YET-BUILDABLE` | Needs a toolkit (`<13`) or seam not yet present; nothing compiles yet |

| Arch | Card examples | Reachability | Target SIGNAL after this lane's W-work | Why not higher |
|---|---|---|---|---|
| Turing `sm_75` | T4 (cloud-ubiquitous), RTX 20-series, GTX 16-series, Quadro RTX | **common + cloud-reachable** (T4 on every major cloud) | `DERIVED+BUILD-VERIFIED (testing-welcome)` | no Turing board here; W-work stops at build+SASS |
| Volta `sm_70` | V100 (cloud), Titan V, Quadro GV100 | cloud-reachable (V100) | `NOT-YET-BUILDABLE` → then `DERIVED+BUILD-VERIFIED` | needs `<13` toolkit first |

**Prior art on Volta, and independent confirmation of the toolkit pin
(2026-08-06, user-supplied).** [`1CatAI/1Cat-vLLM`](https://github.com/1CatAI/1Cat-vLLM)
is a V100 / `sm_70`-focused **fork of vLLM** serving Qwen-class AWQ (and
experimental fp8) models on Volta. It **pins CUDA 12.8**, which is an
independent confirmation of the `<13` toolkit constraint W5 records rather than
an assumption of ours. Its `sm_70` kernels derive from **TurboMind** plus a
custom Volta FlashAttention, so it is a SECOND reference body for this lane
next to llama.cpp's `ggml-cuda` (`fattn-tile`/`fattn-vec`) — useful when W5/W6
pick a source, and evidence that the arch is worth the work to somebody. Its
own stated limits are worth carrying into any claim we make: slow first-request
warmup on V100, MTP degrading long-context throughput there, fp8 experimental.
Not a vLLM upstream capability: upstream vLLM still drops Volta, so this changes
nothing about the "no vLLM oracle on these cards" rule in Risks §2.
| Pascal `sm_60` | P100 (datacenter, fast fp16) | old, some cloud | `NOT-YET-BUILDABLE` → then `DERIVED+BUILD-VERIFIED` | needs `<13` toolkit first |
| Pascal `sm_61` | P40, GTX 10-series, Titan X/Xp | old, plentiful used | `NOT-YET-BUILDABLE` → then `DERIVED+BUILD-VERIFIED` | needs `<13` toolkit + fp32-accum (slow fp16, see below) |

**Value ranking (honest):** Turing `sm_75` is by far the highest-value target —
T4 is the single most cloud-available "old" NVIDIA GPU and it is buildable on
our current nvcc 13 — so it leads the queue. Volta `sm_70` (V100) is second,
gated only on a toolkit. Pascal is real but oldest; `sm_61` additionally needs
an fp32-accumulate path because consumer Pascal has quarter-rate fp16.

---

## Scope

**In.**
- Rows `BACKEND-CUDA-SM075`, `BACKEND-CUDA-SM070` moved `INVENTORIED` → `SPIKE`
  referencing this spec; NEW `BACKEND-CUDA-SM060`, `BACKEND-CUDA-SM061` created
  at `SPIKE`; NEW competitor row `BACKEND-GATE-CUDA-LLAMACPP-LEGACY`.
- The fp16 / non-tensor-core kernel INVENTORY from llama.cpp (attention, the
  core float GEMM/mat-vec, dequant, and the compute-capability gating), each
  with a `file:line` and the algorithm/port strategy.
- The derive-and-ship plan into the `vt::` op layer behind a FEATURE-TABLE cell
  + tactic registration (mirroring the `BACKEND-CUDA-ARCH-ADDITIVITY` seam), so
  a fp16 attention body is a registered tactic keyed on `major∈{6,7}` /
  `sm_75`, not a global `#if`.
- The correctness-gate strategy (with the honest vLLM-oracle gap) and the
  performance-gate floor (llama.cpp on the same old card).

**Out.**
- Any actual kernel port, build, or SASS run in THIS change (read-only spike; no
  legacy card and no `<13` toolkit here). The first port is a claimable W-row.
- vLLM's fast paths (fp4/CUTLASS/Marlin/FA) for these arches — vLLM has none;
  never applicable.
- GGUF-quantized weight kernels (mmvq/mmq int-dot paths). Our gate models are
  bf16/fp16; the quantized mat-vec kernels are inventoried for completeness but
  are a separate follow-on, NOT on the bring-up critical path.
- Ampere/Hopper/datacenter-Blackwell rows (sibling agents own them). Untouched.

**Dispatch behavior.** A fp16 attention tactic registers for the old-arch
capability and the runtime selector (`SelectArchTactic`, keyed on cached
`DeviceCaps`) picks it when `major∈{6,7}` or `sm_75`; on `sm_80`+ the existing
bf16-WMMA path is unchanged and wins. Behavior-preserving on GB10 by
construction: no tactic supports `major==12`, so the default path is untouched.

## Upstream chain — the llama.cpp fp16 / non-tensor-core inventory (file:line)

All at llama.cpp `237ad9b96`, `ggml/src/ggml-cuda/`.

### Compute-capability gating (the arch map to mirror)

| Concern | llama.cpp file:line | What it says |
|---|---|---|
| CC constants | `common.cuh:50-55` | `PASCAL 600`, `DP4A 610`, `VOLTA 700`, `TURING 750`, `AMPERE 800` |
| fp16 arithmetic exists | `common.cuh:257-259` | `FP16_AVAILABLE` ⇔ `__CUDA_ARCH__ >= PASCAL` — half math from Pascal up |
| **fast** fp16 (the P40 trap) | `common.cuh:261-263` (`FAST_FP16_AVAILABLE`), `303-307` (`fast_fp16_available`) | fast fp16 is `FP16_AVAILABLE && arch != 610`: **consumer Pascal `sm_61` (P40/10-series) is quarter-rate fp16** → must accumulate in fp32; P100 `sm_60` and Volta+ have real fast fp16 |
| fp16 tensor cores | `common.cuh:316-320` (`fp16_mma_hardware_available`) | `cc >= VOLTA` — Volta/Turing DO have **fp16** WMMA (just not bf16) |
| bf16 tensor cores | `common.cuh:322-326` (`bf16_mma_hardware_available`) | `cc >= AMPERE` — the exact line that excludes Pascal/Volta/Turing, i.e. our blocker |
| Volta vs Turing MMA | `common.cuh:344-350` (`volta_mma_available`, `turing_mma_available`) | Volta uses the legacy `wmma` PTX; Turing+ uses the newer `mma` PTX |
| dp4a int-dot (quant) | `common.cuh:723-729` (`ggml_cuda_dp4a`) | `__dp4a` from `sm_61` (`DP4A 610`); scalar fallback below — only relevant to the deferred quantized lane |

### Attention — the two non-tensor-core kernels that run on ANY of these arches

| Kernel | llama.cpp file:line | Algorithm | Arch role |
|---|---|---|---|
| **`fattn-tile`** (generic flash tile) | `fattn-tile.cu:5` (`ggml_cuda_flash_attn_ext_tile`), body `fattn-tile.cuh` | Flash-attention with a shared-memory KV tile and **fp16/fp32 register accumulation, NO tensor cores**; online softmax | The Pascal fallback and the small-matrix Volta path |
| **`fattn-vec`** (vector decode) | `fattn-vec.cuh:21` (`flash_attn_ext_vec`), fp16 accum `:134`, fp32 accum `:137`, V-dequant `:100-102` | Per-thread dot-product flash decode; `half2` accumulation when fast-fp16, `float2` otherwise (the P40/`sm_61` branch) — batch-size-1 decode | Decode on all old arches |
| `fattn-wmma-f16` (fp16 tensor cores) | `fattn-wmma-f16.cu`, gate `fattn-wmma-f16.cuh:26` (`ggml_cuda_should_use_wmma_fattn`, true for Volta) | fp16 WMMA (`nvcuda::wmma` with `half`, complete on `sm_70`+) | Volta's large-matrix path; optional speedup, not required for correctness |

**The dispatch that proves these run on the old arches** —
`ggml_cuda_get_best_fattn_kernel`, `fattn.cu:340-537`:
- Turing (`turing_mma_available`, `fattn.cu:457-478`): newer-`mma` fp16 tensor
  cores for large batches, `BEST_FATTN_KERNEL_VEC` for decode.
- Volta (`volta_mma_available`, `fattn.cu:487-495`): `TILE` for `Q->ne[1]*gqa
  <= 16` ("On Volta tensor cores are only faster for sufficiently large
  matrices"), `VEC` for tiny, fp16-WMMA otherwise.
- **Pascal / no-tensor-core (`fattn.cu:523-537`): "If there are no tensor cores
  available, use the generic tile kernel" — `VEC` for decode, else `TILE`.**
  This is the exact beyond-vLLM path.

The enum `best_fattn_kernel` (`fattn.cu:332-338`): `TILE=200`, `VEC=100`,
`WMMA_F16=300`, `MMA_F16=400`. **Our port needs only `TILE` + `VEC`** for
correctness on every old arch; `WMMA_F16` (Volta/Turing fp16 tensor cores) is a
later speed tactic.

### Core GEMM / mat-vec — the float (non-quant) paths our bf16 models use

The high-level router is `ggml_cuda_mul_mat` (`ggml-cuda.cu:2725`), selecting
among five paths by five booleans (`ggml-cuda.cu:2751-2786`), dispatched at
`:2820-2837`:

| Path | llama.cpp file | Algorithm | Arch role |
|---|---|---|---|
| **`mul_mat_vec_f`** (mmvf) | `mmvf.cu` / `mmvf.cuh`, gate `ggml_cuda_should_use_mmvf` | Float (fp16/bf16/f32) **mat-vec, no tensor cores**; the decode GEMV | Decode weight GEMMs on every old arch |
| `mul_mat_f` (mmf) | `mmf.cu` | Float tile GEMM, uses tensor cores where present | Prefill; on old arch falls to cuBLAS |
| `mul_mat_vec_q` (mmvq) | `mmvq.cu` | Quantized mat-vec via `dp4a` (`sm_61`+) | **Deferred** quant lane |
| `mul_mat_q` (mmq) | `mmq.cu` / `mmq.cuh` | Quantized tile GEMM, int tensor cores (Turing+) / dp4a | **Deferred** quant lane |
| **`mul_mat_cublas`** (dequant→cuBLAS) | `ggml-cuda.cu:1668-1890`; `fast_fp16 = fast_fp16_hardware_available(cc)` at `:1705` | Dequant to fp16/f32 then `cublasGemmEx`/`cublasSgemm`; fp16 compute when fast-fp16, else fp32 | Prefill GEMM fallback on old arch |

For OUR bf16 gate path the relevant bodies are **mmvf** (decode GEMV) and the
**cuBLAS fp16/fp32 prefill GEMM** — and our own `cuda_matmul.cu` portable
tiled GEMM (fp32-accumulate) already compiles on `sm_75` (only the attn TU
fails, W10), so the GEMM side is a much smaller lift than the attention side.

### Runtime trace plan

Dispatch here is a runtime capability probe, so a passing test is not proof the
old-arch tactic ran. Reuse the `BACKEND-CUDA-ARCH-ADDITIVITY` mechanism: the
tactic registry's `ArchTacticStats` counters + `VT_ARCH_TACTIC_STATS=1`; the
gate asserts `selections>0 ∧ last_selected=="fattn-fp16-tile/legacy"` for a
synthetic `DeviceCaps{sm_major=7}` (host-side, no GPU), exactly as W9 asserts
the sm12x tactic declines a synthetic Hopper cap.

## Our baseline

| Anchor | State |
|---|---|
| `src/vt/cuda/cuda_paged_attn.cu:706-717` | bf16-WMMA fragment types instantiated unconditionally — the compile blocker on `<sm_80` |
| `cuda_paged_attn.cu:1797` | the exact incomplete-type failure site W10 reported on `sm_75` |
| `cuda_paged_attn.cu:2349-2352` | **`VT_ATTN_WMMA=0` scalar CUDA-core flash fallback ALREADY EXISTS** — a non-tensor-core path, just not compile-isolated |
| `cuda_paged_attn.cu:2562-2570` | `wmma = is_prefill && d==256 && …` runtime gate — already a predicate, needs an arch term |
| `src/vt/cuda/cuda_matmul.cu` | portable tiled GEMM, fp32-accumulate, **compiles on `sm_75`** (not a blocker) |
| `cmake/CudaArchFeatures.cmake` `VT_CUDA_FEATURE_TABLE` | the DATA seam; a new `fattn-fp16` feature row + `sm_75`/`70`/`61`/`60` cells is where support is declared |
| `src/vt/cuda/cuda_arch_tactics.{h,cu}` | the runtime tactic registry; a fp16 attention tactic registers here keyed on old-arch caps |
| `src/vt/cuda/cuda_device_caps.h` | cached `(major,minor)` probe — already the single source of truth |

**Honest gaps:** (1) ~~the bf16-WMMA TU is not compile-guarded~~ — W1 guarded
`cuda_paged_attn.cu`, and §V0 now MEASURES the rest: two more TUs remain
(`cuda_gdn.cu`, `cuda_matmul_nvfp4.cu`); (2) no fp16 attention body ported;
(3) FEATURE-TABLE has no `fattn-fp16` row; (4) no `sm_60`/`sm_61` rows exist;
(5) no `<13` toolkit wired for Volta/Pascal; (6) no legacy card anywhere here
for a runtime gate.

## §V0 — the full-library compile audit (MEASURED 2026-08-06)

W1 build-verified exactly ONE translation unit (`cuda_paged_attn.cu`) and the
B1 row honestly recorded "full-lib link … await W2". That left the size of the
remaining `<sm_80` compile surface UNKNOWN, and an unknown-size pile of `#if`s
is not a costable plan. V0 closes that: **compile every unconditionally-built
CUDA TU at `sm_75` and count what breaks.**

**Method.** `git archive` of `249697b7` onto dgx (nvcc **13.0.88**, cutlass
4.5.0), configured `-DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=ON
-DVLLM_CPP_CUDA_ARCHITECTURES=75 -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_METAL=OFF
-DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` (configure EXIT=0), then each `.cu` entry in
`compile_commands.json` re-run VERBATIM with only `-o` redirected to a throwaway
(no objects retained — the box was at 20 GB free). Compile-only: **no GPU lock
taken, no GPU touched.**

**The arch-gating seam holds — measured, not assumed.** The configure reports
**all eight** fast-path FEATURE-TABLE cells `DISABLED (no requested arch in [75]
provides it)`: `fp4-mma`, `cutlass-nvfp4`, `cutlass-nvfp4-sm100`, `cutlass-fp8`,
`scaledmm-c3x-sm90`, `scaledmm-c3x-sm100`, `marlin-nvfp4`, `fa2`. So the audit
surface is bounded at the **20 TUs** in the unconditional `target_sources` list
(`CMakeLists.txt:896-916`), not the 37 `.cu` files in the tree.

**Result: 20 TUs · 18 PASS (0 errors, 0 warnings) · 2 FAIL.**

| TU | Result | Root cause |
|---|---|---|
| 18 of 20 (`cuda_arch_tactics`, `cuda_backend`, `cuda_cache`, `cuda_combine_tokens`, `cuda_deepseek_v4`, `cuda_dropin`, `cuda_glue`, `cuda_laguna`, `cuda_layernorm`, `cuda_matmul`, `cuda_mla_attn`, `cuda_mla_prefill`, `cuda_moe`, `cuda_ops`, `cuda_paged_attn`, `cuda_quant_dot`, `cuda_sample`, `nccl_communicator`) | **PASS** | — (`cuda_paged_attn` passes *because of* W1) |
| `cuda_gdn.cu` | **FAIL**, 110 error lines | two classes, below |
| `cuda_matmul_nvfp4.cu` | **FAIL**, 10 errors | bf16 WMMA fragments, W1's exact class |

### V0-a — `cuda_gdn.cu` is a second W1, plus a class W1 never faced

The file carries **zero `__CUDA_ARCH__` guards**. Two distinct root causes:

| Class | Count | Sites |
|---|---|---|
| `wmma::precision::tf32` — *"name followed by `::` must be a class or namespace name"* | 4 | `cuda_gdn.cu:2896-2899` |
| tf32 accumulator `fragment<accumulator,16,16,8,float>` incomplete | 16 | span `:3032-4113` |
| bf16 `fragment<…,__nv_bfloat16,…>` incomplete | 37 | span `:3035-4117` |
| cascades from the above (`<error-type>`) + follow-on | 53 | span `:3033-4117` |

**TF32 is the new class, and it is strictly harder than W1's.** `wmma::precision::tf32`
is Ampere-and-later, so on `sm_75`/`sm_70` the failure is a **namespace lookup
failure at the type-alias definition** (`WmmaCfg<float>`, `:2896-2899`), not an
incomplete type at a use site. W1's pattern — guard the kernel *body*, `#else
__trap()` — is therefore NOT sufficient here: the alias block itself must be
conditionally defined, or the TU fails before any kernel body is reached. The
bf16 half (37) is W1's familiar class and takes W1's familiar fix.

### V0-b — `cuda_matmul_nvfp4.cu` compiles unconditionally with its feature OFF

Ten errors, one class: bf16 WMMA fragments at five paired sites
(`:427-428`, `:737-738`, `:1087-1088`, `:1269-1270`, `:2721-2722`).

The finding is the *build wiring*, not the kernel: this TU sits in the
**unconditional** source list (`CMakeLists.txt:903`) even though its own
`fp4-mma` FEATURE-TABLE cell resolves DISABLED at `sm_75`. Every other fp4/CUTLASS
TU is correctly gated behind its cell (`CMakeLists.txt:953`, `:1063`, `:1368`).
So the fix looks like a choice — guard the bodies W1-style, or move the TU behind
its already-existing feature cell.

**CORRECTION (W1b, 2026-08-06): "move the TU behind its cell" is WRONG and was
retracted before implementation.** The name is misleading: this 3,032-line TU is
not an fp4 TU. It also carries the **generic bf16 MoE grouped GEMMs**
(`MoeGroupedGemmBf16Naive`, `…SplitK`, `…Wmma`, `…WmmaPipe`), which every arch
needs and which have nothing to do with `fp4-mma`. Gating the TU on that cell
would strip them from `sm_80/90a/100a/110` and break builds that pass today.
The correct fix is the conservative one: guard the five WMMA kernel bodies
W1-style and leave the TU compiled for every arch. Recorded because the wrong
call here would have been silent — the `fp4` in the filename argues for it, and
only reading the TU's contents refutes it.

### What V0 changes about the plan

1. **The surface is small and finite.** 90% of the unconditional CUDA tree
   already compiles at `sm_75`. The pre-Ampere blocker is two files, not a
   diffuse sweep — this lane is far cheaper than "no old arch compiles" implied.
2. **bf16-as-a-dtype is NOT a blocker, and no fp16 model path is needed.** There
   are **zero** bf16 *arithmetic* intrinsics anywhere in the CUDA TUs; the tree's
   pattern is convert-on-load then compute in fp32 (`cuda_layernorm.cu:56`,
   `__bfloat162float`), and those conversions are available pre-Ampere. The three
   TUs touching `__nv_bfloat162` use it for vectorized *loads*
   (`cuda_laguna.cu:1140`, `cuda_ops.cu:140`), not packed math. `cuda_quant_dot.cu:327`
   uses `__dp4a`, which needs `sm_61` — V100 is `sm_70`, so it is satisfied.
   **Models stay bf16 on Volta**; only WMMA fragment *instantiation* is Ampere-gated.
3. **W1 is not one row, it is three.** `cuda_paged_attn.cu` (DONE),
   `cuda_gdn.cu` (bf16 + tf32), `cuda_matmul_nvfp4.cu` (bf16 or re-gate).
4. **`sm_70` is still not compilable here** — nvcc 13.0 rejects it outright, so
   V0 could only be run at `sm_75`. Both failing TUs fail for capability reasons
   (`sm_70` has neither bf16 nor tf32 tensor cores) that hold on Volta by
   construction, so the fix list transfers; the SASS proof does not, and still
   waits on a CUDA 12.x toolkit (W5).

### Honest scope of a shipped V100 row

What would actually RUN on a V100 is **dense bf16 models** (Llama / Qwen dense)
on the portable fp32-accumulate GEMM plus the ported fp16 tile/vec attention.
GDN (Qwen3-Next), NVFP4, the MLA fast paths and fp8 are Ampere+ by construction;
their TUs compile to `__trap()` stubs and those models are simply not supported
there. The claim to publish is **"V100 runs dense bf16 models"**, never "V100
runs vllm.cpp".

## Port map

| llama.cpp origin | Local destination | Notes |
|---|---|---|
| `fattn-tile.cuh` + `fattn-tile.cu:5` | `src/vt/cuda/cuda_paged_attn_fp16.cu` (new TU) | 1:1 port of the generic non-tensor-core flash tile; fp16 accum for `sm_60/70/75`, fp32 accum for `sm_61` |
| `fattn-vec.cuh:21` (`half2`/`float2` split `:134/:137`) | same new TU | the decode GEMV flash path; the `arch != 610` fp16/fp32 accum split ported verbatim |
| `common.cuh:257-263,303-326` gates | `cmake/CudaArchFeatures.cmake` (`fattn-fp16` feature row) + `cuda_device_caps.h` (accum-dtype predicate) | mirror `FP16_AVAILABLE` / `FAST_FP16_AVAILABLE`; the `!=610` rule becomes a cap predicate |
| `fattn.cu:340-537` dispatch | `src/vt/cuda/cuda_arch_tactics.*` tactic registration + `cuda_paged_attn.cu` selector term | register `fattn-fp16-tile`/`fattn-fp16-vec` for old-arch caps; add the arch term to the `:2562` gate |
| `cuda_paged_attn.cu:706-717` (ours) | `#if __CUDA_ARCH__ >= 800 …` guard | isolate the bf16-WMMA instantiation so the TU compiles on `<sm_80` (W1, mechanical) |
| `mmvf.cu` (optional, speed) | `cuda_matmul.cu` fp16 decode-GEMV tactic | only if the portable GEMV is measured slow vs llama.cpp on the card |

**Deviations, recorded up front.** (1) We port only the `TILE`+`VEC`
correctness bodies first; `WMMA_F16` (Volta/Turing fp16 tensor cores) and the
Turing newer-`mma` path are follow-on speed tactics. (2) The FEATURE-TABLE cell
lists only arches with a built+validated body (deviation #2 of the additivity
spec), so a `sm_75` cell appears only after W2 lands and SASS is verified. (3)
Consumer-Pascal `sm_61` gets the fp32-accumulate variant per llama.cpp's `!=610`
rule; P100 `sm_60` and Volta/Turing get fp16 accum.

## Tests to port

llama.cpp validates fattn by `tests/test-backend-ops.cu` (`FLASH_ATTN_EXT`
cases across head dims, KV lens, GQA ratios, mask/ALiBi) — the executable spec.
Re-expressed locally:

| Case | Local tier | Anchor |
|---|---|---|
| fp16 tile/vec attention numerics vs the CPU oracle across head dims/KV lens/GQA | doctest, NMSE ≤ 5e-4 (reducing op) | `tests/vt/test_ops_paged_attn.cpp` new arch-tactic cases |
| FEATURE-TABLE resolves `fattn-fp16` for `75`/`70`/`61`/`60`, EMPTY on `80`/`121a` | `cmake -P`, no GPU, CI-gated | `cmake/CudaArchFeaturesTest.cmake` new expectations |
| registry SELECTS the fp16 tactic for a synthetic old-arch `DeviceCaps`, declines on `sm_80`+ | doctest, no GPU | `tests/vt/test_ops_paged_attn.cpp` / `test_cuda_backend.cpp` |
| a gate model end to end on a real old card | parity, **checked in SKIPPED (no hardware here)** with a tracked "help-wanted: Turing/Volta/Pascal testing" reason | `tests/parity/` |

## Gates

Because no legacy card and no `<13` toolkit exist here, gates are staged and the
top stage is honestly hardware-blocked.

| Gate | Requirement | Reachable here? |
|---|---|---|
| **B0 guard** ✅ PASSED | bf16-WMMA TU guarded; `sm_121a` GB10 build byte-identical (default path untouched) | **PASSED** (base `034be66e`): guard landed; sm_121a same TU `-Werror` 0-warn + 0 SASS instruction diffs vs unguarded (byte-identical) |
| **B1 build (Turing)** ✅ PASSED (TU) | clean `-DVLLM_CPP_CUDA_ARCHITECTURES=75 -DVLLM_CPP_TRITON=OFF` build, `-Werror` 0-warn; `cuobjdump -lelf` shows real `sm_75` cubins | **PASSED for the `cuda_paged_attn` TU** (dgx nvcc 13.0.88): single-arch `75` compile `-Werror=all-warnings` 0-warn EXIT=0, `cuobjdump -lelf` → real `cuda_paged_attn.cu.1.sm_75.cubin`. **W1a/§V0 2026-08-06 widened this from one TU to all 20: 18 PASS, 2 FAIL (`cuda_gdn.cu`, `cuda_matmul_nvfp4.cu`).** Full-lib LINK still awaits W1b+W2 — a per-TU compile pass is not a link |
| **B1 build (Volta/Pascal)** | same for `70`/`61`/`60` | **NO** until a `<13` toolkit is provisioned (`NOT-YET-BUILDABLE`) |
| **C1 kernel correctness** | ported tile/vec numerics vs CPU oracle (NMSE ≤ 5e-4); FEATURE-TABLE + registry-selection asserted host-side | YES (no GPU needed for the host asserts; numerics run on any CUDA dev) |
| **C3 e2e / P performance** | a gate model token-checked AND ≥ llama.cpp on the same old card | **NO — hardware-blocked**; this is the `testing-welcome` frontier |

**Ship criterion (derive-and-ship).** A row moves to
`DERIVED+BUILD-VERIFIED (testing-welcome)` when B0+B1(its arch)+C1 pass and the
port is a faithful 1:1 of the cited llama.cpp `file:line`, LABELED "derived from
llama.cpp, not hardware-tested here, community testing welcome". It moves to
`RUNTIME-VERIFIED` only when someone runs stages C3/P on the real card.

### Correctness gate — the honest oracle problem

Our standing correctness bar is token-exactness vs the pinned **vLLM oracle**.
**vLLM cannot be the oracle here:** vLLM dropped these arches, so the oracle
binary will not even run on a Turing/Volta/Pascal card (and it is an `sm_121a`
build regardless). So the correctness reference for a REAL on-card test is:

1. **llama.cpp itself on the same old card** — it is the port source AND the
   thing users run there; matching its logits/argmax on the identical GGUF/bf16
   weights is the primary on-card correctness signal (this is literally how
   llama.cpp validates its own old-arch kernels via `test-backend-ops`).
2. **A newer-card or CPU cross-check** — the SAME `vt::` fp16 tile/vec tactic,
   compiled for `sm_80`+ or run on the CPU reference, produces the reference
   output; the old-arch run must match it within the near-tie band. This proves
   the KERNEL is correct independent of the card, which is what build-verify +
   a portable-tier cross-check can establish with NO legacy hardware.

Be explicit in every row: **there is no vLLM oracle on these cards**; the gate
is llama.cpp-on-card + a portable cross-check, and until a card is available the
claim is build-verified, not runtime-verified.

### Performance gate — the real competitor floor

vs **llama.cpp on the same old card** (T4/V100/P40…), pinned commit per run,
same GGUF/weights, same prompt, under the GPU lock, ≥2 reps: ours ≥ llama.cpp on
total + output throughput and ≤ on TTFT/TPOT/peak memory. This is the honest
"beyond-vLLM" scoreboard — vLLM has no entry to compare against, so llama.cpp is
the only floor. Recorded as `BACKEND-GATE-CUDA-LLAMACPP-LEGACY`, `INVENTORIED`
until a card exists.

## Dependencies

| Kind | Item |
|---|---|
| Rows | `BACKEND-CUDA-ARCH-ADDITIVITY` seam (DONE) — the FEATURE-TABLE + tactic registry this lane plugs into |
| Toolchain | nvcc 13.0 for Turing; a CUDA **12.x** toolkit for Volta/Pascal (NOT present here — the gating dependency) |
| Hardware | a Turing (T4 cloud), Volta (V100), or Pascal (P40/P100) card for any runtime/perf gate; none here |
| Source | llama.cpp `237ad9b96` `ggml-cuda` (MIT) — the port source; cite file:line per body |
| Models | a small bf16 or GGUF model that fits these cards' memory + its golden; llama.cpp as the on-card reference |

## Work breakdown

| # | Row | Deliverable | Blocked on |
|---|---|---|---|
| W1 ✅ **DONE** | guard the bf16-WMMA TU | `#if __CUDA_ARCH__ >= 800` wraps the bodies of all 5 bf16-WMMA prefill kernels (`cuda_paged_attn.cu:732,958,1197,1472,1716`; `#else __trap()`), so `__CUDA_ARCH__ < 800` compiles the TU selecting the existing scalar path. **Build-verified (dgx nvcc 13.0.88 + cutlass 4.5.0, base `034be66e`):** single-arch `75` `-Werror=all-warnings` 0-warn EXIT=0, `cuobjdump -lelf` → real `cuda_paged_attn.cu.1.sm_75.cubin`; the `:1797 __nv_bfloat16 fragment` error GONE (RED: unguarded HEAD FAILS 21 errors). GB10 sm_121a byte-identical — same TU `-Werror` 0-warn AND 0 SASS instruction diffs vs unguarded. NO Turing board ran it | DONE — (mechanical, nvcc 13) |
| W1a ✅ **DONE** | **V0 full-library compile audit** — every unconditionally-built CUDA TU compiled at `sm_75`, failures enumerated and classified. **MEASURED 2026-08-06 (base `249697b7`, dgx nvcc 13.0.88): 20 TUs, 18 PASS (0 err / 0 warn), 2 FAIL** (`cuda_gdn.cu` 110 errors, `cuda_matmul_nvfp4.cu` 10). All 8 fast-path FEATURE-TABLE cells confirmed DISABLED at `75`, bounding the surface at the `CMakeLists.txt:896-916` list. Compile-only, no GPU. Full detail + classification in §V0 | DONE — nvcc 13, no card |
| W1b ✅ **DONE** | **finish the guard set.** `cuda_gdn.cu`: both `WmmaCfg` specializations' members guarded (bf16 fragments AND the tf32 alias block — a lookup failure at the alias, so a body-only guard does NOT compile), 8 device bodies guarded `#if __CUDA_ARCH__ >= 800` / `#else __trap()`, plus the `V128<T>` staging helpers and `WyMerge`. `cuda_matmul_nvfp4.cu`: 5 WMMA bodies guarded, TU left compiled for every arch (see the §V0-b correction — gating it on `fp4-mma` would have stripped the generic bf16 MoE GEMMs from `sm_80/90a/100a/110`). **VERIFIED (dgx nvcc 13.0.88): `sm_75` 20/20 TUs PASS, 0 errors 0 warnings** (was 18/20). **`sm_121a` byte-identity HELD:** both TUs `-Werror=all-warnings` 0-warn, `cuda_gdn.cu` SASS bit-identical across 824,704 lines, `cuda_matmul_nvfp4.cu` **zero instruction-level diffs** (all 148 differing lines are `Function :` headers carrying the anon-namespace hash, which shifts on any edit — same artifact W1 recorded). NO board ran any of it | DONE |
| W1c ✅ **DONE** | **arch-gate the SELECTORS — the guards alone were a live trap.** W1/W1b made the WMMA bodies compile on `<sm_80` behind `__trap()`, but every predicate that SELECTS them was host-side only (shape/dtype/env), so a pre-Ampere board would still pick a trapping kernel. Three chokepoints, each now requiring `DeviceCaps::sm_major >= 8` and each falling through to an EXISTING portable path: (a) `cuda_paged_attn.cu:2611` → `LaunchPrefillFlash` (CUDA-core register-tiled flash); (b) `cuda_gdn.cu:5359` `GdnPrefillKernelCuda` → `GdnScanCuda` (sequential scan) — the single chokepoint, since all 7 guarded GDN launches live in `LaunchChunkedPrefill`, itself reached only from there, and the `TSc=float` instantiations use the TF32 `WmmaCfg` so f32 is not a way around it; (c) `cuda_matmul_nvfp4.cu` `WmmaEnabled():77` → naive/tiled/split-K. (c) is folded into the predicate itself rather than its six call sites because all six mean the same thing and each already has a CUDA-core fallthrough; it is queried LIVE, not latched in the static env cache, since the device context need not exist at static-init time. All three fail safe (caps invalid → portable). **VERIFIED (dgx nvcc 13.0.88): all 3 TUs `-Werror=all-warnings` rc=0 0-warn at BOTH `sm_75` and `sm_121a`; `sm_121a` SASS IDENTICAL for all three** (933,178 + 825,294 + 137,542 lines, anon-namespace hash normalized) — host-side change, device code untouched. NO board ran it | W1b |
| W2 | port `fattn-tile`+`fattn-vec` fp16 body — **RESCOPED to a SPEED brick by W1c.** The bf16-WMMA path is only `is_prefill && d == 256 && bf16 q+KV` (`cuda_paged_attn.cu:2611`); all decode and every other prefill shape already run portable kernels, and W1c routes the d=256 case to `LaunchPrefillFlash` on `<sm_80`. So pre-Ampere CORRECTNESS is covered by paths that exist today, and this port buys prefill throughput only. ~2,000 lines adapted from llama.cpp's contiguous KV to our paged block-table layout, against a floor (llama.cpp on-card) that needs hardware we do not have. **Do this only for a genuine speed claim on T4/V100, and only once a card is reachable** | W1b; hardware for any payoff |
| W3 | FEATURE-TABLE + tactic registration | `fattn-fp16` feature row + `sm_75/70/61/60` cells; register `fattn-fp16-tile/vec` tactics; selector arch term at `:2562`; `CudaArchFeaturesTest.cmake` + registry-selection tests | W2 |
| W4 | **Turing derive-and-ship** | `sm_75` `-Werror` build + `cuobjdump` SASS proof; row → `DERIVED+BUILD-VERIFIED (testing-welcome)`; labeled untested | W3, nvcc 13 (**doable now**) |
| W5 | wire a `<13` toolkit | provision CUDA 12.x; Volta/Pascal build-verify (`70`/`61`/`60` SASS); those rows → `DERIVED+BUILD-VERIFIED` | a 12.x toolkit |
| W6 | fp16-WMMA speed tactic (opt) | port `fattn-wmma-f16` for Volta/Turing fp16 tensor cores + Turing newer-`mma`; a speed tactic, correctness already covered | W4 |
| W7 | on-card runtime + perf gate | any legacy card: llama.cpp-on-card + portable cross-check correctness, then perf vs llama.cpp; row → `RUNTIME-VERIFIED` | hardware |
| W8 | quantized lane (opt, later) | port mmvq/mmq (`dp4a`, `sm_61`+) for GGUF-quant on old cards | separate |

## Risks / decisions

1. **P40/consumer-Pascal quarter-rate fp16 is a real correctness-of-speed trap.**
   `sm_61` must accumulate in fp32 (llama.cpp's `arch != 610` rule,
   `common.cuh:261-263`); using fp16 accum there is both slow and less accurate.
   Encoded as a cap predicate, not a global.
2. **No vLLM oracle on these cards — stated in every row.** The correctness
   reference is llama.cpp-on-card + a portable/newer-card cross-check; do not
   imply a vLLM comparison that cannot run.
3. **Build-verify is not run-verify.** A green `sm_75` link + SASS is
   `DERIVED+BUILD-VERIFIED`, never `RUNTIME-VERIFIED`; every shipped row carries
   the "not hardware-tested here, community testing welcome" label — llama.cpp's
   own posture for old arches.
4. **Volta/Pascal are toolkit-blocked, not scoped-out.** They become buildable
   the moment a 12.x toolkit is wired (W5); the honest state is
   `NOT-YET-BUILDABLE`, distinct from Turing's `DERIVE-AND-SHIP-NOW`.
5. **The GB10 default path must stay byte-identical.** W1's guard and the new
   tactic support only `major∈{6,7}`/`sm_75`; no tactic supports `major==12`, so
   the gate models' path is untouched (mirrors the additivity behavior-preserving
   contract).
6. **Turing HAS fp16 tensor cores** (`fp16_mma_hardware_available`, `cc>=VOLTA`)
   — the bf16 blocker is specifically bf16, so a fp16-WMMA speed tactic (W6) is
   available later; correctness never needs it (tile/vec suffice).
7. **TF32 is a SECOND Ampere-gated tensor-core type, and this spec missed it
   until W1a measured it.** `wmma::precision::tf32` (`cuda_gdn.cu:2896-2899`)
   fails as a *namespace lookup at the alias definition*, so it breaks the TU
   earlier and more thoroughly than an incomplete bf16 fragment does. Any future
   "is this arch additive?" audit must scan for `precision::tf32` alongside
   `__nv_bfloat16`, and must not assume W1's body-guard pattern generalizes.
8. **An unconditional TU can outlive its own feature gate.**
   `cuda_matmul_nvfp4.cu` is compiled for every arch while its `fp4-mma` cell is
   DISABLED — the FEATURE-TABLE seam is only as good as the `target_sources`
   list that honours it. Worth a standing check: any TU whose kernels are
   arch-specific should sit behind its cell, and `CMakeLists.txt:903` currently
   does not.
9. **A COMPILE guard without a SELECTOR guard is a trap, not a fix.** This is the
   single most important lesson of W1a-W1c. `#if __CUDA_ARCH__ >= 800` /
   `#else __trap()` makes a TU *compile* on an old arch; it does nothing about the
   host code that decides to launch it. Every predicate selecting a guarded kernel
   here was pure host-side shape/dtype/env, so the "fix" would have turned a
   compile error into a runtime crash on exactly the boards it was meant to
   enable. Any future arch-guarding must pair each `#if` with an arch term on its
   selector and name the portable path it falls through to — and if no portable
   path exists, that is a design problem, not a mechanical edit.
10. **Guarding a body orphans its helpers, and `-Werror=all-warnings` turns that
   into a build failure.** W1b needed three iterations for exactly this: after the
   8 GDN bodies were guarded, `nvcc` reported `#177-D "declared but never
   referenced"` for `WmmaCfg::WK`, every `V128<T>` member, and finally `WyMerge`
   — each promoted to an error. Two lessons for the remaining arch work: guard
   the helper structs on the SAME condition as their only consumers, and for a
   `__device__` function whose callers are all guarded, wrap the WHOLE function
   (a body-only guard leaves an emitted-but-uncalled definition that still trips
   `#177-D`). Expect this cascade on any future `<sm_80` guarding.
11. **A per-TU compile sweep is NOT a link.** W1a compiled 20 TUs to throwaway
   objects; it proves no TU has an `<sm_80` *compile* blocker beyond the two
   named, and nothing about undefined symbols, `__trap()` stubs reachable at
   link, or fatbin assembly. Do not quote 18/20 as "the library builds".
## Owed

- **`ISSUE-GH-148` — Volta (`sm_70`) and Turing (`sm_75`) support.** This lane
  owns the fp16 / non-tensor-core breadth arc, and W1 shipped only the Turing
  `sm_75` bf16-WMMA guard; the Volta and Turing runtime arms stay open. The
  record predates the cutover and carries `Row: -`; no row, date, state or
  resolution is invented for it. The owner is this spec because the architecture
  scope is the spec's own.
