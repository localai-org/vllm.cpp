# Leaf spec: GGUF compute-in-quant GEMM — ggml tensor-traits port into vt::cpu

**Row:** `QUANT-GGUF-CIQ-GEMM` (quantization-matrix.md, leaf of the
`QUANT-GGUF-COMPUTE` block) · **status:** `ACTIVE` — **G1-G4 landed** (2026-07-22): the portable tier-0 quant GEMM is complete, gated at the OP level, and **now routed end to end** — `vt::MatmulBT` dispatches a block-dtype weight to `kMatmulBTQuant`, keep-quant is the production DEFAULT wherever that op is registered, and the six routed encodings compute in quant. Measured on the binding host: decode **3.45×**, prefill **4.16×**, peak RSS **1.16×** less, output tokens **byte-identical**. G5 landed 2026-10-31; G8 open (G6 and G7 landed 2026-07-23), and the G4 measurement RE-RANKED them (see the G4 result below: the remaining CPU gap is the ELEMENTWISE bf16/f16 GEMM, not the quant one) ·
**upstream pin:** llama.cpp local fork `237ad9b96` (b9892) ·
**parent evidence:** B4 decision row
([parity-ledger.md L290](../parity-ledger.md#L290)) — the "vendor IF proven
faster" criterion is MET; this leaf is the structural fix: keep GGUF blocks
resident and quantize the ACTIVATION to int8 to meet them (llama.cpp's
mul_mat), instead of expanding weights to bf16 at load. Research estimate on
top of the threadpool: ~3.3× decode, ~10× prefill, ~3.3× weight RAM
([expansion map wave 3](expansion-map-2026-07-10.md)).

## Scope

- **In scope:** quantized-block weight storage in `vt::` (new block dtypes +
  traits); activation quantization `f32→Q8_0/Q8_K` (`from_float`); the
  per-type `vec_dot` GEMM core dispatched exactly like ggml's
  `type_traits_cpu` table; portable generic C++ kernels FIRST for the types
  our GGUF families need — **Q8_0, Q4_K, Q5_K, Q6_K, Q3_K** (APEX 35B
  Compact/Balanced + the Q8_K_XL bench file) plus already-materialized
  **Q4_0**; then per-arch SIMD tiers (x86 AVX2, Arm NEON/dotprod/i8mm for
  GB10) and the repack/interleave tier (`q4_Kx8`-style) as separate work
  rows. CPU backend only.
- **Out of scope:** loader/persistence of the quant weights (leaf
  `QUANT-GGUF-KEEPQ-LOADER`); the threadpool itself (leaf
  `QUANT-GGUF-CPU-THREADPOOL`); i-quants/TQ/MXFP4/NVFP4 GGUF types (their
  own encoding rows: `QUANT-GGUF-IQ2_S`, `QUANT-GGUF-IQ4_XS`, …); CUDA GGUF
  compute (future backend row); llamafile/KleidiAI/AMX accelerators (see
  Risks/decisions).
- **Dispatch behavior (mirrors ggml exactly):** for weight type T,
  `vec_dot_type(T)` selects the activation quant (Q8_0 for Q4_0/Q8_0; Q8_K
  for Q3_K/Q4_K/Q5_K/Q6_K per `type_traits_cpu`); src1 (activations) is
  quantized once into scratch, then each output element is one
  `vec_dot(K, w_row, act_row)`. Repack tier, when present for
  (type, arch, N%8), replaces this with interleaved `gemv/gemm` — same
  selection precedence as `ggml_repack_get_optimal_repack_type`.

## Upstream chain

Pinned local fork `/home/mudler/_git/llama.cpp` @ `237ad9b96`.

| Component | Upstream anchor | Content |
|---|---|---|
| Trait table | `ggml/src/ggml-cpu/ggml-cpu.c:211-406` | `type_traits_cpu[]`: `from_float`, `vec_dot`, `vec_dot_type`, `nrows` per type; Q8_0→(q8_0,q8_0,nrows 2 on i8mm) `:262-271`, Q3_K `:295-300`, Q4_K `:301-310`, Q5_K `:311-316`, Q6_K `:317-326`, Q4_0 `:230-239` |
| GEMM driver | `ggml-cpu.c:1245-1443` | src1→`vec_dot_type` quant into `wdata` (`:1313-1349`, block-split across threads), chunk grid + atomic stealing (threadpool leaf), `num_rows_per_vec_dot` 2-row mmla guard `:1426-1433` |
| Chunk worker | `ggml-cpu.c:1155-1243` | 16×16 tile, `vec_dot` per output, dst column memcpy |
| Block layouts | `ggml/src/ggml-common.h:242-245` (q8_0), `:305-310` (q3_K), `:317-327` (q4_K), `:334-345` (q5_K), `:352-357` (q6_K), `:361-365` (q8_K: f32 d, i8 qs[256], i16 bsums[16]) | byte-exact structs; our gguf_dequant.cpp already mirrors the weight-side layouts |
| Activation quant reference | `ggml/src/ggml-quants.c:238` (`quantize_row_q8_0_ref`), `:2696` (`quantize_row_q8_K_ref` incl. bsums) | portable reference; CPU wrappers `ggml-cpu/quants.c:45,117` |
| Generic vec_dot kernels (portable C++ tier) | `ggml-cpu/quants.c:174` (q4_0·q8_0), `:400` (q8_0·q8_0), `:566` (q3_K·q8_K), `:645` (q4_K·q8_K), `:720` (q5_K·q8_K), `:800` (q6_K·q8_K) | integer block dots + bsums tricks; this is the byte-for-byte port source for tier 0 |
| x86 SIMD tier | `ggml-cpu/arch/x86/quants.c:1170` (q8_0·q8_0), `:1900` (q4_K·q8_K), `:2288` (q6_K·q8_K), … | AVX2/AVX512 variants, generic fallback tail |
| Arm SIMD tier | `ggml-cpu/arch/arm/quants.c:1076` (q8_0·q8_0), `:2260` (q4_K·q8_K), `:2890` (q6_K·q8_K), … | NEON/dotprod/SVE/i8mm (`nrows==2` mmla paths) — GB10's tier |
| SIMD abstraction | `ggml-cpu/simd-mappings.h`, `arch-fallback.h` | per-ISA macro mapping; our tiers keep the same generic-fallback naming discipline |
| Tensor-traits hook | `ggml-cpu/traits.h:20-36`, `traits.cpp:9-38` | `tensor_traits::{work_size,compute_forward}` + `extra_buffer_type` probe — the seam repack plugs into |
| Repack tier | `ggml-cpu/repack.cpp:4153-4250` (traits class), `:4253` (`forward_mul_mat`), `:4528-4726` (`ggml_repack_get_optimal_repack_type`: q4_K_8x8 on AVX2/i8mm, q6_K_8x8/8x4, q8_0_4x4/4x8, q5_K, q2_K …), `:4727` (extra hook), `:4745-4830` (buffer type); activation `ggml_quantize_mat_t<4/8, Q8_0>` `:318-357`; generic gemv/gemm: q4_K `:887,:1822,:1905`, q5_K impl `:551,:647`, q6_K impl `:357,:447`, q8_0 `:1274-1368`; arch SIMD: `arch/x86/repack.cpp`, `arch/arm/repack.cpp` | prefill engine: 4/8 interleaved weight rows × 4/8 activation rows per kernel call |

Runtime-trace: CPU dispatch is static given (type, arch, N alignment); the B4
oracle run (20-thread llama.cpp, repack active on AVX2) is the traced
denominator. When gating on GB10 Arm, confirm which tier llama.cpp selects
there (i8mm repack expected) before comparing.

## Our baseline

- **No quantized compute exists.** All executable GGUF weights expand to bf16
  at load: [qwen3_5_gguf_weights.cpp:105-155](../../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp#L105);
  GEMMs then run the naive bf16 loops
  [cpu_ops.cpp:30-57](../../src/vt/cpu/cpu_ops.cpp#L30).
- Dequant kernels (weight-side block decode, byte-for-byte ports already
  reviewed against the same pin):
  [gguf_dequant.cpp:43-219](../../src/vt/cpu/cpu_quant_dequant.cpp#L48),
  dispatch `:246-279` — reuse their layout knowledge; they REMAIN as the
  `VT_CPU_REF` oracle path.
- `vt::` typing: [include/vt/dtype.h:21](../../include/vt/dtype.h#L21) has no
  block dtypes (`kF32,kF16,kBF16,kI8,kI32,kI64`); `vt::Tensor` assumes
  elementwise strides. [include/vt/ops.h:62](../../include/vt/ops.h#L62)
  `kMatmulBT` is the Linear-orientation GEMM
  ([N,K] row-major weight) — the exact orientation ggml's src0 has.
- Precedent for quant-weight structs + orientation flag:
  [qwen3_5_weights.h:37-80](../../include/vllm/model_executor/models/qwen3_5_weights.h#L37)
  (`OwnedTensor.nk`, `Nvfp4Weight` packed+scales kept raw in [N,K]).
- Tests today: [tests/vllm/test_gguf_dequant.cpp:25-223](../../tests/vllm/test_gguf_dequant.cpp#L25)
  (dequant units), [tests/parity/test_qwen36_gguf_engine.cpp:143](../../tests/parity/test_qwen36_gguf_engine.cpp#L143)
  (same-file greedy vs llama.cpp oracle goldens) — nothing exercises a quant
  GEMM. Honest gap: the whole `C` column of the GGUF encoding table.

## Port map

| Upstream | Local | Notes / deviations |
|---|---|---|
| `ggml.h` type ids + `ggml-common.h` block structs | extend `vt::DType` with `kQ4_0,kQ8_0,kQ3_K,kQ4_K,kQ5_K,kQ6_K,kQ8_K` + block traits (block_elems, block_bytes) in `src/vt/dtype.cpp`; cross-checked against `GgmlTraits` ([gguf_reader.cpp:191](../../src/vllm/model_executor/model_loader/gguf_reader.cpp#L191)) | design decision recorded below; `kQ8_K` is activation-only (never a weight/storage dtype in files) |
| `type_traits_cpu[]` (`ggml-cpu.c:213-471`) | new `src/vt/cpu/cpu_quant_traits.{h,cpp}`: `{from_float, vec_dot, vec_dot_type, nrows}` per block dtype | same table shape so upstream diffs port mechanically |
| `quantize_row_q8_0_ref` / `quantize_row_q8_K_ref` (`ggml-quants.c:238,2696`) | `src/vt/cpu/cpu_quant_act.cpp` | byte-for-byte incl. Q8_K bsums; f32 input (our activations are bf16 → widen first, matching ggml's f32 src1 contract) |
| generic `ggml_vec_dot_*` (`ggml-cpu/quants.c:174,400,566,645,720,800`) | `src/vt/cpu/cpu_quant_dot.cpp` (tier 0, pure portable C++) | keep `*_generic` names in comments + upstream line cites per kernel |
| `ggml_compute_forward_mul_mat` src1-quant + chunk loop (`ggml-cpu.c:1245-1443,1155-1243`) | new op `OpId::kMatmulBTQuant` in `include/vt/ops.h` + CPU kernel in `cpu_ops.cpp`; `vt::MatmulBT` call sites route on `b.dtype` being a block dtype (no signature change) | scratch for quantized activations: per-call 64B-aligned buffer sized `rows × row_size(vec_dot_type, K)` mirroring `ggml_graph_plan` wdata (`ggml-cpu.c:2752-2980`); reuses `ParallelFor` from the threadpool leaf |
| `arch/x86/quants.c`, `arch/arm/quants.c` | `src/vt/cpu/arch/{x86,arm}/cpu_quant_dot_{avx2,neon}.cpp` (tier 1) | compile-time ISA gates + runtime CPU-feature probe mirroring `ggml_cpu_has_*`; generic fallback always built (CI runs it) |
| `repack.cpp` traits + gemv/gemm + `quantize_mat_t` | `src/vt/cpu/cpu_quant_repack.cpp` (tier 2) | weight repack at LOAD (from the keep-quant loader's resident blocks) selected per (dtype, arch, N%8) exactly as `:4528-4726`; brings the 4×8/8×8 GEMM used for prefill |
| `traits.h` `tensor_traits`/`extra_buffer_type` classes | NOT ported as a polymorphic hook | our dispatch is the static op registry; ggml needs the hook for pluggable backends we don't have — recorded deviation, revisit if a second accelerator (AMX/KleidiAI-class) is ported |

## Tests to port

| Upstream test | Local tier / file | Notes |
|---|---|---|
| llama.cpp `tests/test-quantize-fns.cpp` (thresholds `:17-28`: total quant error ≤0.002 4-bit+, ≤0.0040 3-bit, reference error ≤0.0001, dot-product error ≤0.02, ≤0.04 low-bit) | T-unit `tests/vt/test_ops_quant_dot.cpp` (new) | per-type: quantize→dequant RMSE bounds AND `vec_dot` vs f64 reference dot on synthetic rows; same thresholds, cite each |
| llama.cpp `tests/test-backend-ops.cpp` MUL_MAT cases (`:4260-4450`, NMSE ≤ `max_nmse_err()` `:1133-1151` ≈5e-4) | T-unit, same file | `kMatmulBTQuant` vs the dequant-to-f32 composite (`DequantGgufRowToF32` + `MatmulBTKernel`) at model shapes (K=hidden, N=proj dims, M∈{1,4,32,512}) per type |
| llama.cpp `tests/test-quantize-perf.cpp` | tooling only (bench harness), not a ctest gate | microbench per vec_dot tier; evidence for the ledger, never a pass/fail gate |
| our existing goldens | T-parity [tests/parity/test_op_parity.cpp:34](../../tests/parity/test_op_parity.cpp#L34) | add `matmul_bt_quant` runner + goldens; **goldens+runner land in the SAME commit or the op goes in `PendingRunnerOps()`** (workflow.md anti-stale-golden gate) |
| same-file engine gates | T-e2e [tests/parity/test_qwen36_gguf_engine.cpp:143](../../tests/parity/test_qwen36_gguf_engine.cpp#L143) + the 2B Q8_K_XL file | greedy 16/16 vs same-file llama.cpp oracle with compute-in-quant ACTIVE; new goldens (see Risks — tokens may legitimately differ from the bf16-expansion path; the oracle is llama.cpp, not our old path) |

## Gates

1. **Op correctness:** ported test-quantize-fns bounds green for all six
   types (generic tier on CI; SIMD tiers on their boxes); MUL_MAT NMSE
   ≤ 5e-4 vs the dequant-f32 composite at all listed shapes.
2. **Oracle discipline:** `VT_CPU_REF=1` env forces the dequant-to-bf16 +
   bf16-GEMM path (loader leaf provides it): existing goldens stay
   byte-identical there — the parity oracle is never lost. Compute-in-quant
   is NOT bit-identical to it by design; its correctness bar is the
   llama.cpp same-file oracle.
3. **E2E token gate:** greedy 16/16 token-exact vs same-file llama.cpp CPU
   (`237ad9b96` build) on (a) Qwen3.5-2B-UD-Q8_K_XL, (b) APEX 35B Compact,
   (c) APEX 35B Balanced — the existing gate harness recipe.
4. **Perf (B4-anchored, same box/file/recipe as
   [parity-ledger.md L290](../parity-ledger.md#L290), threadpool active, 20
   threads, ≥2 reproducing runs):**
   - tier 0 (generic vec_dot) milestone: decode ≥ 0.25× llama.cpp tg32
     (≥3.2 t/s), prefill ≥ 0.1× pp128;
   - tier 1 (arch SIMD) milestone: decode ≥ 0.7× tg32;
   - tier 2 (repack) closes the row: **match or beat** llama.cpp on decode
     t/s, prefill t/s AND peak RSS on the same file/box (mirror-as-floor;
     `BACKEND-GATE-CPU-LLAMACPP` consumes this evidence; GB10 Arm rerun
     recorded there too).
5. **Memory:** peak RSS with compute-in-quant ≤ 1.15× llama.cpp same-run
   (2.80 GiB on the 2B) — the bf16 expansion (7.43 GiB) must be gone
   (loader leaf) and no hidden f32 staging may reintroduce it.

## Dependencies

- `QUANT-GGUF-CPU-THREADPOOL` (`ParallelFor` + chunk loop; perf gates
  meaningless without it). Op-level tier-0 kernels + unit tests may land
  before it (single-thread), but the row cannot pass gate 4 without it.
- `QUANT-GGUF-KEEPQ-LOADER` for gates 3-5 (resident quant weights + the
  `VT_CPU_REF` switch). Unit gates 1-2 need only synthetic blocks.
- Hardware: 20-core x86 box (AVX2 tier + gates); GB10 (Arm NEON/i8mm tier);
  CI runs generic tier. No GPU lock; CPU runs need an otherwise-idle box per
  benchmark-protocol.
- Oracle: llama.cpp fork build at `237ad9b96` (same as B4; keep the local
  NVFP4/Q1_0 fork patch noted — irrelevant to these six types).
- Models: 2B Q8_K_XL + APEX 35B Compact/Balanced GGUFs (already on the boxes;
  checkpoint-gated tests skip when absent, as today).

## Work breakdown

| W | Row (claim-sized) | Content | Depends | Status |
|---|---|---|---|---|
| G1 | block dtypes + traits | `vt::DType` block entries, traits table, `kMatmulBTQuant` op skeleton + generic-composite fallback, trait cross-check vs `GgmlTraits` | - | **DONE** 2026-07-22 |
| G2 | activation quant | `quantize_row_q8_0/q8_K` ports + scratch sizing; unit tests | G1 | **DONE** 2026-07-22 |
| G3 | tier-0 vec_dot kernels | the six generic `vec_dot` ports + GEMM wiring + test-quantize-fns/backend-ops ports + parity goldens | G1, G2 | **DONE** 2026-07-22 |
| G4 | e2e enablement | route `MatmulBT` call sites on block dtype; engine gates (3) + `VT_CPU_REF` A/B; first B4-recipe measurement (tier-0 milestone) | G3, loader L2-L3, threadpool W2 | **DONE** 2026-07-22 |
| G5 | x86 AVX2 tier | `arch/x86` vec_dot ports + feature probe; tier-1 milestone measurement | G3 | **DONE** 2026-10-31 — AVX2 `vec_dot` for q4_K/q6_K/IQ3_S behind `VT_CPU_QUANT_AVX2`, portable fallback, 2.65× same-binary decode on Flare GSQ (see G5 result below) |
| G6 | Arm NEON/i8mm tier | `arch/arm` ports incl. `nrows==2` mmla; GB10 measurement | G3 | **DONE** 2026-07-23 — i8mm mmla `nrc==2` tier for q8_0/q4_0/q4_K/q6_K, 2x2-tiled into `kMatmulBTQuant` (see G6 result below) |
| G7 | repack tier | repack-at-load + gemv/gemm + `quantize_mat_t`; selection parity vs `:4528`; closes gates 4-5 | G4 + loader L2 | **DONE** 2026-07-23 — q8_0 `q8_0_4x8` repack-at-load (block_q8_0x4 interleave), bit-identical i8mm gemm+gemv, byte-permute weight + non-fused MAC (see G7 result below) |
| G8 | ledger + matrix closure | full A/B series, ledger row, flip `C`/`E`/`P` cells per encoding row (Q8_0/Q4_K/Q5_K/Q6_K/Q3_K/Q4_0), roadmap update | G4-G7 | open |

G5/G6 are parallel; G7 may start once G4's layout is fixed.

## Risks/decisions

- **`vt::DType` extension vs side-struct** (internal design, decided here):
  block dtypes IN the enum mirror ggml's "type carries layout" model, keep
  `Tensor` self-describing, and let dispatch key on `b.dtype` — the
  alternative (parallel `QuantTensor`) would fork every op signature. Blocked
  dtypes are storage-only: any elementwise op on them is a `VT_CHECK` failure.
- **Goldens shift:** enabling compute-in-quant changes CPU e2e numerics (not
  bit-equal to bf16 expansion). Decision: goldens for quant-GGUF engine gates
  are regenerated against the llama.cpp oracle in the SAME change that flips
  the default, with the `VT_CPU_REF` path keeping the old goldens alive. No
  silent tolerance widening.
- **Repack memory transient:** repack-at-load temporarily holds source+dest
  blocks per tensor; do it tensor-by-tensor (llama.cpp repacks in-buffer per
  tensor) to keep gate 5.
- **Accelerator hooks not ported** (llamafile sgemm `ggml-cpu.c:1286-1310`,
  AMX, KleidiAI): each is an extra tier behind the same dispatch seam;
  llama.cpp's B4 arm did not need them to hit the floor numbers (stock CPU
  build). Revisit only if gate 4 tier-2 stalls below 1.0× — record then.
- **`nrows==2` boundary guards** (`ggml-cpu.c:1426-1433`) must be ported with
  the mmla kernels or i8mm results corrupt on odd shapes — covered by the
  non-divisor shape cases in the ported MUL_MAT tests.
- Upstream behavior is not reopened: quant math, thresholds, and tier
  selection all mirror llama.cpp; the only product-ish calls are the two
  recorded above (enum design, golden regeneration protocol).

## G1 result (2026-07-22)

**Landed.** `vt::DType` gains the seven block entries (`kQ4_0 kQ8_0 kQ3_K kQ4_K
kQ5_K kQ6_K kQ8_K`) with block geometry in
[dtype.cpp](../../src/vt/dtype.cpp#L32) and the storage-only rule enforced —
`SizeOf` on a block dtype THROWS, so any elementwise kernel that reaches one
fails loudly. `RowSizeBytes` mirrors `ggml_row_size` (rows are whole blocks).
The `type_traits_cpu` mirror is [cpu_quant_traits.cpp](../../src/vt/cpu/cpu_quant_traits.cpp#L1)
behind the new public [vt/quant.h](../../include/vt/quant.h#L1); `vec_dot_type`
and `nrows` are populated, `vec_dot`/`from_float` stay `nullptr` until G3/G2.
`OpId::kMatmulBTQuant` + [the CPU kernel](../../src/vt/cpu/cpu_quant_gemm.cpp#L1)
run the generic dequant-composite fallback for every type.

**Deviation from the port map (recorded).** The six `dequantize_row_*` decoders
were MOVED, not duplicated: they now live at
[cpu_quant_dequant.cpp](../../src/vt/cpu/cpu_quant_dequant.cpp#L1) as the traits
table's `to_float` column (upstream keeps `to_float` in `ggml.c`'s
device-neutral `type_traits`; vt has no such second table, so it rides the CPU
one), and the GGUF loader
[delegates](../../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L75).
One implementation serves both the loader oracle and the GEMM fallback. The
code is byte-identical to what it replaced, so numerics are unchanged, and
`test_gguf_dequant` plus a new byte-for-byte equivalence case gate that.

**Trait cross-check — NO mismatch.** [test_ops_quant_traits.cpp](../../tests/vt/test_ops_quant_traits.cpp#L1)
compares THREE independent statements of the same llama.cpp facts: vt's block
geometry table, the GGUF reader's `GgmlTraits`, and the block-struct arithmetic
written out from `ggml-common.h`. All seven types agree on block_elems,
block_bytes and type id; the id map round-trips; `Q8_K` (id 15) is correctly
absent from the reader's table because it is activation-only. 8 cases /
5,694 assertions green.

## G2 + G3 result (2026-07-22)

**Landed together.** The portable tier-0 compute-in-quant GEMM is complete.

**G2 — activation quant.** [cpu_quant_act.cpp](../../src/vt/cpu/cpu_quant_act.cpp#L1)
ports `quantize_row_q8_0_ref` (`ggml-quants.c:238`) and `quantize_row_q8_K_ref`
(`ggml-quants.c:2696`, bsums included) plus `nearest_int` (`:563`); the CPU-tier
wrappers those sit behind are `ggml-cpu/quants.c:45,117`, which on the generic
tier are bare calls to the `_ref` forms. Scratch sizing (`QuantActRowBytes` /
`QuantActScratchBytes`) mirrors `ggml_row_size` + the mul_mat `wdata`
computation in `ggml_graph_plan` (`ggml-cpu.c:1313-1349, 2752-2980`). Only the
two ACTIVATION encodings get a `from_float`: nothing in this project quantizes
an activation into a k-quant, so upstream's k-quant encoders stay unported.

**G3 — the six generic `vec_dot`.** [cpu_quant_dot.cpp](../../src/vt/cpu/cpu_quant_dot.cpp#L1)
ports `ggml-cpu/quants.c` `:174` (q4_0·q8_0), `:400` (q8_0·q8_0), `:566`
(q3_K·q8_K), `:645` (q4_K·q8_K), `:720` (q5_K·q8_K), `:800` (q6_K·q8_K), with
upstream's auto-vectorization-shaped structure (`aux8` decode split, 8-wide
`aux16`/`aux32` staging, deferred `sums[8]`) preserved verbatim — that structure
also FIXES the reduction order, which is what makes the GEMM bit-reproducible.
Each kernel asserts `nrc == 1`; `nrows == 2` stays unreachable until G6 brings
the mmla kernels AND the `ggml-cpu.c:1426-1433` boundary guards.
[cpu_quant_blocks.h](../../src/vt/cpu/cpu_quant_blocks.h#L1) is a `static_assert`ed
mirror of the `ggml-common.h` block structs.
[cpu_quant_gemm.cpp](../../src/vt/cpu/cpu_quant_gemm.cpp#L1) now takes the
quantized path (quantize src1 ONCE into scratch, then one integer `vec_dot` per
output) for the six types; the generic dequant-composite remains for any block
dtype without a `vec_dot` (today only Q8_K, activation-only).

**Correctness — gated against an INDEPENDENT reference, not a second copy.**
[test_ops_quant_dot.cpp](../../tests/vt/test_ops_quant_dot.cpp#L1), 16 cases /
78,052 assertions green. The primary gate dequantizes BOTH operands through
`BlockToFloat` (the loader-side `dequantize_row_*` decoders — a separate port
that walks the layout differently from each `vec_dot`'s inline decode) and dots
them in **f64**, with the tolerance set relative to the dot's L1 magnitude so
sign cancellation cannot hide an error (`|got - ref| <= 1e-5 * L1`, actual
agreement ~1e-6). Coverage: all six types × nblocks ∈ {1, 2, 3, 5, 7, 16} —
single-block rows, ODD multiples that catch an unroll-by-2 assumption, and the
Q6_K/Q3_K super-block structure. Ragged K (not a whole number of blocks) is
gated to THROW at every layer (`from_float`, `vec_dot`, `QuantActRowBytes`,
`kMatmulBTQuant`) rather than round down. The G2→G3 handoff is covered by
feeding real `from_float` output into every `vec_dot` case. Upstream's own
thresholds are ported unwidened: `test-quantize-fns.cpp:17-28`
(total 0.002 / reference 0.0001 / dot-product 0.02) and
`test-backend-ops.cpp:4277` MUL_MAT NMSE ≤ 5e-4 at M ∈ {1,4,32,512} ×
N ∈ {1,7,16}. Determinism: `vec_dot` and the full GEMM are asserted
**bit-identical** run-to-run and across thread counts 1/2/4 (byte `memcmp`, not
`Approx`), and the GEMM is asserted bit-equal to a hand-driven per-element
`vec_dot` so the wiring provably adds no reordering or stride slip.

**Mutation-tested (14 mutants, 13 caught).** To prove the gate has teeth rather
than merely passing: q6_K −32 bias→−31, q4_K `mins[j/2]`→`[j/4]`, q3_K hmask
polarity flip, q5_K high-bit 16→8, q4_0 nibble bias −8→−7, q8_0 dropped
activation delta, q4_K `kmask1` corruption, q3_K scale bias −32→−31, q8_K bsums
short/doubled, q8_K `iscale` −127→−128, q8_0 `amax/127`→`/128`, q8_0 stored-delta
fed back into the quants — **all caught**. One mutant (`roundf`→truncation in
Q8_0) initially PASSED every statistical bound: upstream's 8-bit RMSE/NMSE
thresholds genuinely cannot distinguish rounding modes. That gap is closed by an
added **byte-exact encoder gate** comparing `from_float` output against a
reference written from upstream's prose (round-half-away-from-zero for Q8_0,
round-half-to-even for `nearest_int`) rather than by calling the same library
function; it also catches the `nearest_int`→truncation and stored-delta mutants.
The single uncaught mutant is Q8_K's `MIN(127, v)`, which is **provably
unreachable** (|iscale| = 127/amax and |x| ≤ amax ⇒ |iscale·x| ≤ 127); it is kept
for upstream fidelity and the test records why no case can distinguish it.

**DGX CONFIRMATION RUN (`dgx.casa`, `~/work/vllm.cpp-ciq-g3`, transferred by `git archive` — never rsync — production flags `-DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DVLLM_CPP_TRITON=ON -DCMAKE_CUDA_ARCHITECTURES=121a`, GPU verified idle by `nvidia-smi` with no compute apps, every stage under `flock $HOME/gpu.lock`):** clean CUDA `-Werror` build to 100%, **0 warnings**. Regression set ALL UNCHANGED, each gate run STANDALONE — 27B **235/235**, 35B **315/315**, Qwen3-Coder **6/6** (138 assertions; strict 5/6, max gap 0 nats), Qwen3-dense **16/16** on BOTH 0.6B and 4B (664 assertions; strict 10/16 and 11/16, max gap 0.25 nats), OPT **6/6** (36 assertions), DeepSeek-V2-Lite **8/8** (223 assertions; strict 5/8, 92/128 tokens strictly exact, 0 forward-divergent). `test_qwen36_gguf_engine` run STANDALONE on the real APEX files: **28/28 assertions, 2/2 cases, 16/16 tokens each on Compact and Balanced**. The three new/updated CPU units also pass on dgx's **aarch64** with identical counts (`test_ops_quant_dot` 16 cases / 78,052 assertions, `test_ops_quant_traits` 8 / 5,615, `test_gguf_dequant` 11 / 202) — the portable tier is architecture-independent, as intended. Golden corpus md5 (475 files under `tests/parity/goldens`) **identical before and after the series**: `2965ef5772b556d3f3f86fedf4221b2f`. **One transient to record honestly:** a first pass that ran four engine gates back-to-back inside a single shell aborted DeepSeek-V2 partway (95 of 223 assertions, 1 failure); re-run STANDALONE it passes 223/223 / 8/8, which is the known co-scheduled-memory effect on this box, not a regression.

**Nothing existing moved.** No model call site routes to `kMatmulBTQuant`
(`grep` over `src/vllm/` finds no reference — G4 owns that), every GGUF weight
still expands to bf16 at load, and the per-encoding `C` cells therefore stay `-`.
`test_gguf_dequant` 11 cases / 202 assertions and the `BlockToFloat` ==
loader-dequant byte-for-byte case remain green, so the decode numerics are
untouched. `test_ops_quant_traits` is 8 cases / **5,615** assertions (was 5,694):
its composite-fallback case now covers Q8_K alone, because the six weight types
legitimately no longer take the composite path — the drop is that retarget, not
lost coverage, and the six types' GEMM behaviour moved to the new file's
independent-reference and NMSE cases.

## G4 grounding — what the measurement says it is worth (2026-07-22)

`CLAIM-BENCH-CPU-LLAMA-REMEASURE-1` re-measured the floor and profiled the
engine. Full write-up:
[CPU floor re-measurement](cpu-llamacpp-floor-remeasure-2026-07-22.md). Three
results change how this row should be planned and judged.

**1. G4 is not an increment — it is most of the gap.** An op-dispatch profile
that accounted for 100.0 % of wall time found **`kMatmul` at 95.37 %** (748
calls, 10.915 ms/call); attention, the GDN recurrence, norms, conv, sampling and
all host glue are under 5 % combined. The CPU plan is a GEMM plan.

**2. The tier-0 kernels this row already landed are 14–44× faster than the GEMM
production runs.** Op-level harness at this model's shapes (aarch64, 20 threads,
best of 3):

| shape (M×N×K) | production `kMatmul` `[K,N]` bf16 | `kMatmulBT` `[N,K]` bf16 | tier-0 `kMatmulBTQuant` Q8_0 | ratio |
|---|---|---|---|---|
| decode qkv 1×3072×2048 | 11.79 GFLOP/s | 17.91 | **287.54** | 24.4× |
| decode down 1×2048×6144 | 6.11 | 18.48 | **266.50** | 43.6× |
| decode lm_head 1×248320×2048 | 14.66 | 24.68 | **211.43** | 14.4× |
| prefill qkv 128×3072×2048 | 18.42 | 24.62 | **416.79** | 22.6× |
| prefill gate_up 128×12288×2048 | 14.86 | 24.52 | **393.48** | 26.5× |

x86 (indicative, contended box): production 6.1–12.6, Q8_0 tier-0 130–200, so
13–20×.

**3. Gate 4's tier-0 milestone is set far too low.** Projecting the table above
onto this model (arithmetic on measured op throughput, NOT an end-to-end run):
per-token decode GEMM 24 × (0.044+0.032+0.226+0.094) + 4.811 = **14.3 ms ⇒
~70 t/s** cache-warm, bounded in practice by DRAM at llama.cpp's own **25.80
t/s**; 128-token prefill GEMM 24 × (3.864+2.710+16.373+8.295) = **750 ms ⇒ ~171
t/s** against llama.cpp pp128 **174.63**; peak RSS ≈ file + activations ≈
**2.9–3.0 GiB** against llama.cpp's **2.798**. The stated tier-0 milestone
(decode ≥ 0.25× tg32, prefill ≥ 0.1× pp128) should be cleared by a wide margin.
**Measure G4 against PARITY, not against the milestone** — and re-scope **G7**
afterwards, because tier-0 at M=128 is already within ~2 % of pp128 on GEMM
alone, which is far less headroom than the repack tier was budgeted for.

**Orientation is a G4 PREREQUISITE, newly identified.** The GGUF loader today
dequantises to bf16 **and transposes** into `[K,N]`
([qwen3_5_gguf_weights.cpp:192](../../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp#L192)
— "dequant to bf16 and transpose to Matmul-B [K, N]"), so the hot path runs
`kMatmul`, whose inner loop strides by N down K. Measured, that orientation
alone costs **1.3–3.0×**. GGUF's native layout is already `[N,K]` — exactly what
`MatmulBTQuant` requires — so G4's first step is to stop doing extra load-time
work to reach the slower kernel.

**Inertness re-confirmed at `1cb5f64`** by three independent checks: no
`MatmulBTQuant` reference under `src/vllm/`; `vt::MatmulBT`
([ops.cpp:163](../../src/vt/ops.cpp#L163)) hard-requires `IsFloat(b.dtype)` with
no block branch; and `VT_GGUF_KEEP_QUANT=1` **fails at load on both boxes** with
`vt: matmul_bt: float inputs and f32/bf16 output required`.

## G4 result (2026-07-22) — routed, default-ON, measured

**Landed.** The machinery G1-G3 and L1-L3 built is now on the executed path.

**What routes.** ONE dispatch point does the whole job:
[`vt::MatmulBT`](../../src/vt/ops.cpp#L158) sends a block-dtype `b` to
`MatmulBTQuant` and falls through to its unchanged validation + `kMatmulBT` for
everything else. That is sufficient because every model matmul helper already
routes an `nk == true` weight to `MatmulBT`
([qwen3_5.cpp:1067,1078](../../src/vllm/model_executor/models/qwen3_5.cpp#L1067)
device-in/out, [:743,760](../../src/vllm/model_executor/models/qwen3_5.cpp#L743)
host), and the keep-quant loader already produces exactly that: a block-typed
`OwnedTensor` in the file's `[N,K]` order. No forward, no signature and no call
site changed. Safetensors paths are unreachable from the new branch by
construction — no safetensors loader produces a block dtype.

**The default flip.** [`GgufLoadPolicy::FromEnv`](../../src/vllm/model_executor/model_loader/gguf_keep_quant.cpp#L95)
now defaults `keep_quant` to `GgufQuantComputeAvailable()` — `vt::OpRegistered(kMatmulBTQuant, CurrentPlatform().device_type())`
— rather than to `false`. The condition is the honest one ("does a block weight
have a consumer on the device this process will run on"), so a CUDA build keeps
expanding to bf16 (CUDA GGUF compute-in-quant remains a future backend row) and
the day that kernel is registered elsewhere the default follows with no edit.
`VT_GGUF_KEEP_QUANT=0/off/false` is the opt-out; `VT_CPU_REF=1` still wins over
everything.

**The transpose is gone — for BOTH residencies.** Kept blocks never had one
(L2). The measurement's lever 2 was folded in as
`GgufLoadPolicy::expand_nk`: a matmul weight that must EXPAND (an f16/f32 file
tensor, or an unported encoding) is now materialised in the file's own `[N,K]`
order with `nk = true` instead of being transposed into Matmul-B `[K,N]`. It
rides the same availability condition and is forced off by `VT_CPU_REF`. This
is numerically free on the CPU tier by construction —
[`MatmulOneChunk<kBT>`](../../src/vt/cpu/cpu_ops.cpp#L70) differs from `<false>`
ONLY in the weight offset, keeping the same sequential f32 K reduction — and a
unit asserts the two loads are element-for-element transposes of each other. It
is deliberately NOT applied on a device whose GEMM picks its algorithm from the
operand layout (cuBLASLt), where orientation is not numerically free.

**This mattered more than expected.** The bench file is MIXED, which is the
normal case: `Qwen3.5-2B-UD-Q8_K_XL.gguf` is 103 `q8_0` + **56 `f16`** + 176
`f32` tensors, and the f16 side is not small tensors — it is 1.615 GiB
including the 970 MiB tied `token_embd`/`lm_head` and whole layers of ffn.
Without `expand_nk` the largest GEMM in the model would still have been paying
the slow orientation.

### Correctness — NO token movement anywhere

- **E2E gate, quant path DEFAULT-ON, run STANDALONE on a CPU-only dgx build**
  (which is where keep-quant is live): `test_qwen36_gguf_engine` **PASSES** on
  the real APEX files — 2/2 cases, 16/16 greedy tokens each on Compact
  (`{F32,Q3_K,Q4_K,Q6_K}`) and Balanced (`{F32,Q8_0,Q5_K,Q6_K}`) against the
  same-file llama.cpp oracle. Five of the six routed encodings are exercised
  end to end by that pair. Peak RSS during it was 18.7 GiB on the 17.3 GiB
  Compact file — visibly the file, not a bf16 expansion of it.
- **Oracle A/B on the bench model:** the output token streams of the BEFORE
  arm (`VT_GGUF_KEEP_QUANT=0`), the AFTER arm (production default) and the
  ORACLE arm (`VT_CPU_REF=1`) are **byte-identical** — one md5,
  `d235db12f2cd304007530286a1755c95`, across all three. So `VT_CPU_REF`
  reproduces the historical result exactly, and compute-in-quant did not move a
  single token on this file.
- **A CPU-only build fails four SACRED gates — CONTROLLED as pre-existing, not
  a G4 regression.** Running the full dgx ctest against a CPU-only build (the
  configuration that exercises the quant path) fails `test_qwen36_weights`,
  `test_qwen3_paged_engine`, `test_qwen27_paged_engine` and
  `test_deepseek_v2_paged_engine`, because those gates' goldens were captured on
  CUDA and a CPU forward is not bit-identical to it. The BASE tree (`c603ebc`)
  was built CPU-only on the same box and run against the same four: **identical
  failures, identical assertion counts** (51/2, 245/2, 11/2, 95/1). None of the
  four loads a GGUF file, and all four pass on the CUDA build (235/235, 315/315,
  664/664, 223/223). The regression bar for this repo is the CUDA build; a
  CPU-only run of a CUDA-goldened gate is not one.
- The spec anticipated that quantized compute "is NOT bit-identical to the
  dequant path by design". On the measured workloads it did not need the
  latitude: **zero divergence, so nothing had to be diagnosed and no golden was
  regenerated.** Q8_0 · bf16 is the favourable case (the weight values are
  identical either way; only the accumulation differs, and the integer path is
  the more accurate one), so this is a result about these files, not a proof
  that no k-quant file can ever differ.

### Benchmark — binding, dgx.casa (GB10, aarch64, 20 cores), idle

Whole series under one `flock $HOME/gpu.lock`; load average 0.11 at series
start, `nvidia-smi` reporting no compute apps; transferred by `git archive`;
goldens md5-verified 475 files / `2965ef5772b556d3f3f86fedf4221b2f` before and
after. Model `Qwen3.5-2B-UD-Q8_K_XL.gguf`, llama.cpp `237ad9b96` (b9892).

**llama.cpp** (`llama-bench -t 20 -ngl 0 -r 5`): pp512 212.49±0.74, **pp128
175.71±2.54**, tg128 25.90±0.69, **tg32 25.87±1.36**; isolated tg32
25.26±1.34 with peak RSS 2,934,124 KB = **2.798 GiB**. (Yesterday's numbers on
this box reproduce to within noise, so the denominator is fresh, not carried.)

**Ours — SAME BINARY, three arms**, `VLLM_CPP_CPU_THREADS=20`,
`--input-len 128 --output-len 32 --concurrency 1 --seed 0 --temperature 0`,
3 reps each, medians:

| arm | TTFT (ms) | TPOT (ms) | peak RSS (KB) |
|---|---|---|---|
| BEFORE — `VT_GGUF_KEEP_QUANT=0` (the historical path) | 24,859.25 | 451.27 | 7,788,196 |
| **AFTER — production default** | **5,970.21** | **130.72** | **6,712,368** |
| ORACLE — `VT_CPU_REF=1` | 24,822.20 | 451.89 | 7,787,940 |

Rep spread on the AFTER arm: TTFT 1.41 %, TPOT 2.03 %. The BEFORE arm
reproduces the 2026-07-22 binding baseline (24,594 ms / 450.7 ms / 7,788,220 KB)
to 1.1 % / 0.1 % / 24 KB, which is what makes this a valid same-binary A/B.

| axis | BEFORE | AFTER | ours: A/B | vs llama.cpp BEFORE | **vs llama.cpp AFTER** |
|---|---|---|---|---|---|
| Prefill (128 / TTFT) | 5.149 t/s | **21.44 t/s** | **4.16×** | 34.1× behind | **8.20× behind** |
| Decode (1000 / TPOT) | 2.216 t/s | **7.650 t/s** | **3.45×** | 11.7× behind | **3.38× behind** (3.30× vs isolated tg32) |
| Peak RSS | 7.428 GiB | **6.401 GiB** | **1.16× less** | 2.66× worse | **2.29× worse** |

### Did the 9-17× projection hold? NO — and the reason is measurable

**It did not.** The projection was 9-17× end to end with RSS falling to
~2.9 GiB; the measurement is **3.45× decode / 4.16× prefill and RSS 6.40 GiB**.
The projection was arithmetic over the quant GEMM's op-level throughput applied
to the WHOLE model, and that step is where it broke: it assumed every GEMM takes
the quant path. On this file most of the weight mass cannot.

| | bytes in file | params | routed to `kMatmulBTQuant`? |
|---|---|---|---|
| `q8_0` (103 tensors) | 1.062 GiB | 1.073 B | **yes** |
| `f16` (56 tensors, incl. the 970 MiB tied `token_embd`/`lm_head`) | 1.615 GiB | 0.867 B | **no** — f16 is not a block encoding |
| `f32` (176 tensors) | 0.006 GiB | — | no (norms/vectors, never eligible) |

So **60 % of the weight bytes and 45 % of the parameters — including the single
biggest GEMM in the model — kept running the elementwise bf16 kernel at
17-25 GFLOP/s** while the routed 40 % moved to 211-417 GFLOP/s. The measured
3.4-4.2× is what that mixture predicts; the projection's 9-17× is what a
fully-q8_0 file would have given. Nothing about the quant kernels
under-delivered.

The same mixture explains the RSS floor: 2.68 GiB of touched file pages (blocks
are COPIED, not mmapped in place — the recorded L2 follow-up) + 1.615 GiB of
f16→bf16 expansion + a SECOND 970 MiB copy of the tied head + 1.06 GiB of kept
blocks ≈ 6.35 GiB, against 6.401 GiB measured.

### Consequences for the plan (supersedes the ranking in the floor re-measurement)

0. **RESOLVED (2026-07-22, later the same day).** Item 1 below was acted on
   immediately and is now the landed row
   [`KERNEL-GEMM-CPU-ELEM`](cpu-elementwise-gemm.md) **E1-E4**: the elementwise
   GEMM went 18-24 -> 69-351 GFLOP/s **bit-exactly** (memcmp gate, same token
   md5), taking the CPU position from 3.38x / 8.20x / 2.29x to **decode 1.03x
   (parity within 3.1 %) / prefill 2.34x / RSS 2.29x**. Its own measured
   negative re-ranks items 2-3 once more: M-blocking that kernel bought 1.63x at
   the op level and **0.0 % end to end**, so prefill is no longer GEMM-bound of
   EITHER kind, and the 95.37 % `kMatmul` attribution below is **STALE**. **G5,
   G6 and G7 must not be started against it** — a fresh op-dispatch profile of
   the current binary is owed first.
1. **The elementwise bf16/f16 GEMM is now the #1 CPU lever, ahead of G5-G7.**
   It owns 60 % of this file's weight mass, EVERY safetensors CPU path, and
   every mixed GGUF (which is what published quants actually are). Its two known
   defects are stated in
   [the floor re-measurement lever 3](cpu-llamacpp-floor-remeasure-2026-07-22.md):
   no SIMD `vec_dot_bf16/f16`, and `MatmulOneChunk`'s per-element `LoadF32`
   dtype switch INSIDE the K loop. Closing it is what would let this file
   approach llama.cpp, which computes those same f16 rows with
   `ggml_vec_dot_f16`.
2. **G5/G6 (SIMD quant tiers) now improve only the 40 % that is already fast**,
   so their end-to-end share on a mixed file is bounded by Amdahl at well under
   what the spec assumed. **G7 (repack) should stay parked**: prefill is 8.2×
   behind and the quant GEMM is not what is holding it.
3. **RSS needs two loader changes, not a kernel:** mmap-in-place residency
   instead of copying (already recorded as an L2 follow-up, llama.cpp supports
   both), and sharing the tied `lm_head` with the embedding table instead of
   materialising the vocab matrix twice. Together they are ~1.9 GiB of the
   3.6 GiB that separates us from llama.cpp; the rest is the f16 expansion,
   which the lever-1 work removes as a side effect if f16 weights stay f16.
4. **Gate 4's tier-0 milestone is MET.** Its bar is decode ≥ 0.25× tg32 and
   prefill ≥ 0.1× pp128: measured **0.296×** (7.650 / 25.87) and **0.122×**
   (21.44 / 175.71), against 0.086× and 0.029× before G4. **Gate 5 (peak RSS
   ≤ 1.15× llama.cpp) is NOT met** — 2.29× — and stays open with L4 plus the two
   loader changes in item 3.

### Reproduction

```sh
# dgx.casa, aarch64, idle, ONE flock for the whole series
R=$HOME/work/bench-cpu-llama
M=$R/models/Qwen3.5-2B-UD-Q8_K_XL.gguf
LB=$R/llamacpp/build/bin
B=$HOME/work/vllm.cpp-ciq-g4/build-cpu/examples/vllm-bench

cmake -S vllmcpp -B build-cpu -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build-cpu -j 20 --target vllm-bench

flock $HOME/gpu.lock sh -c '
  uptime; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  LD_LIBRARY_PATH='"$LB"' '"$LB"'/llama-bench -m '"$M"' -p 512,128 -n 128,32 -t 20 -r 5 -ngl 0
  LD_LIBRARY_PATH='"$LB"' /usr/bin/time -v '"$LB"'/llama-bench -m '"$M"' -p 0 -n 32 -t 20 -r 3 -ngl 0
  for arm in "VT_GGUF_KEEP_QUANT=0" "" "VT_CPU_REF=1"; do
    for rep in 1 2 3; do
      env $arm VLLM_CPP_CPU_THREADS=20 /usr/bin/time -v '"$B"' --model '"$M"' \
        --num-prompts 1 --input-len 128 --output-len 32 --concurrency 1 \
        --seed 0 --temperature 0 --output-token-ids /tmp/ids.$rep.txt
    done
  done
'
# The three arms MUST agree on /tmp/ids.*.txt (md5 d235db12f2cd304007530286a1755c95).
```

**Gates at G1:** clean CPU `-Werror` build (0 warnings); full CPU ctest
**151/151**; gguf ctest 4/4. **DGX CONFIRMATION RUN (`dgx.casa`, `~/work/vllm.cpp-ciq-g1`, production flags `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DVLLM_CPP_CUTLASS_DIR=/home/mudler/cutlass-4.5.0 -DVLLM_CPP_TRITON=ON`, one `flock $HOME/gpu.lock` for the whole series):** clean CUDA `-Werror` build to 100%, **0 warnings**. Regression set ALL UNCHANGED — 27B **235/235**, 35B **315/315**, Qwen3-Coder **6/6** (strict 5/6, max gap 0 nats), Qwen3-dense **16/16** (strict 11/16, max gap 0.25 nats), OPT **6/6** (36 assertions), DeepSeek-V2-Lite **8/8** (223 assertions). `test_qwen36_gguf_engine` run STANDALONE (it OOMs co-scheduled) on the real APEX files: **28/28 assertions, 2/2 cases, 16/16 tokens each on Compact and Balanced** — the k-quant dequant path is byte-stable across the decoder move. Gates 1-5 of this spec are
untouched: they need G2/G3 (op correctness) and the loader leaf (e2e/perf/RSS).

## G6 result (2026-07-23) — Arm i8mm (mmla) tier, landed + measured

**What landed.** The `nrc==2` `vmmlaq_s32` block dot products for the FOUR
encodings llama.cpp gives an mmla path — q8_0, q4_0, q4_K, q6_K — ported
byte-for-byte from `arch/arm/quants.c` @ `237ad9b96` (q4_0 `:241`, q8_0 `:1094`,
q4_K `:2495` `#elif MATMUL_INT8`, q6_K `:3101` `#elif MATMUL_INT8`) into
[cpu_quant_dot_arm.cpp](../../src/vt/cpu/cpu_quant_dot_arm.cpp#L1). q3_K and q5_K
have NO upstream mmla path (SVE + NEON only), so they legitimately stay on the
portable nrc==1 tier — "port the mmla path where upstream has it", nothing
invented. [`kMatmulBTQuant`](../../src/vt/cpu/cpu_quant_gemm.cpp#L85) grew a 2x2
register-tiled path (`QuantChunkMmla`) that mirrors
`ggml_compute_forward_mul_mat_one_chunk`'s `num_rows_per_vec_dot=2` tile
(`ggml-cpu.c:1233-1239`): each `vec_dot(nrc=2)` computes two weight rows × two
activation rows.

**Feature probe (illegal-instruction-safe).** The TU is compiled with
`-march=armv8.2-a+i8mm+dotprod` **for this one file only** (CMakeLists
`set_source_files_properties`, surgical exactly like the x86 tier's per-function
`target("f16c")`) so `<arm_neon.h>` exposes `vmmlaq_s32`; the whole body is
`#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)`, else empty
stubs so every other build links and the portable tier serves. At RUNTIME the
mmla kernels are handed out only when `getauxval(AT_HWCAP2) & HWCAP2_I8MM`
([QuantMmlaActive](../../src/vt/cpu/cpu_quant_dot_arm.cpp#L1)); a wrong-tier
selection on a non-i8mm CPU can't happen. `VT_CPU_QUANT_MMLA=0|off|false` forces
the tier off for a same-binary A/B (portable nrc==1 then serves every shape),
and `VT_CPU_REF=1` still reproduces the historical dequant path.

**Determinism by construction (unpin is thread-safe).** mmla engages only at
`M%2==0 && N%2==0`; decode (M=1) and any odd dim keep the portable nrc==1 path —
the exact "num_rows_per_vec_dot falls to 1" guard ggml uses (`ggml-cpu.c:1432`).
The GEMM parallelizes over WEIGHT-ROW PAIRS, so the (weight-row, activation-row)
pairing is GLOBAL and independent of how pairs chunk across threads → the result
is bit-identical run-to-run AND across thread counts. Verified 1/2/4/20.

**Bit-exact vs NMSE — determined, not assumed.** The int8 products accumulate
into i32 EXACTLY, and q8_0/q4_0's only float step is a block-by-block
`vmlaq_f32` MAC in the scalar kernel's order, non-fused under `-ffp-contract=off`
— so **q8_0 and q4_0 mmla are BIT-IDENTICAL to the portable/scalar tier**
(`CHECK(exact == total)` passes for all shapes). q4_K/q6_K add a `vpaddq`/`vmull`
bias reduction that reassociates and are gated at the ratified **NMSE ≤ 5e-4**
vs the portable per-element dot (measured well under). Both are asserted in
[test_ops_quant_dot.cpp](../../tests/vt/test_ops_quant_dot.cpp#L1)'s new G6
tier-cross-check (cases run only where the tier is live; skip coherently
elsewhere). On dgx the suite is 19 cases / **78,162 assertions** green; on x86
(portable) 19 / 78,058 with the mmla numeric cases skipping.

**Correctness — NO token movement.** `test_qwen36_gguf_engine` STANDALONE with
the mmla tier live: **2/2 cases, 16/16 greedy tokens** on APEX-Compact
(`{F32,Q3_K,Q4_K,Q6_K}`) and APEX-Balanced (`{F32,Q8_0,Q5_K,Q6_K}`) vs the
same-file llama.cpp oracle — q8_0/q4_K/q6_K all exercised at prefill through
mmla, no token moved. On the bench file the output-token md5 is
`d235db12f2cd304007530286a1755c95` — byte-identical across the mmla-OFF, mmla-ON
and `VT_CPU_REF=1` arms.

### Benchmark — binding, dgx.casa (GB10, aarch64, 20 cores), one flock

Loadavg-gated inside the lock (start 1.30). `git archive` transfer.

**Op-level GFLOP/s (kMatmulBTQuant, best-of-N, portable vs i8mm):**

| shape (M×N×K) | q8_0 portable→i8mm | q4_K portable→i8mm | q6_K portable→i8mm |
|---|---|---|---|
| prefill qkv 128×3072×2048 | 450–492→**518** ~1.1–1.2× | 148→**1109** **7.5×** | 240→**930** **3.9×** |
| prefill gate_up 128×12288×2048 | 492→**580** 1.18× | 152→**1270** **8.4×** | 248→**1070** **4.3×** |
| prefill down 128×2048×6144 | 454→**514** 1.13× | 148→**1058** **7.1×** | 238→**905** **3.8×** |
| decode qkv 1×3072×2048 | 412→412 (mmla off at M=1) | 139→143 | 227→227 |

The k-quant win is large (mmla does the block unpack in-register vs the portable
scalar `aux8` decode); q8_0's portable kernel was already near-optimal so its
mmla uplift is modest.

**E2e prefill A/B — same binary, `Qwen3.5-2B-UD-Q8_K_XL`, 3 reps, medians:**

| arm | TTFT (ms) | prefill t/s |
|---|---|---|
| mmla OFF (portable) | 1135.68 | 112.7 |
| **mmla ON (default)** | **1047.70** | **122.2** |

Same-binary **1.084× prefill**; vs llama.cpp pp128 (refreshed same session,
175.41±1.62): **1.56× → 1.44× behind**. Decode TPOT ~44 ms unchanged (mmla off
at M=1). This file is the PESSIMAL case for G6 — it is q8_0 (whose portable was
already fast) + 60 % `f16` (elementwise GEMM, not G6's domain) — so the e2e
share is Amdahl-bounded; the op-level table shows the real kernel lever, and the
k-quant-heavy APEX 35B production files are where the 3.8–8.4× lands end to end.

**Fresh bottleneck.** The prior profile had kMatmulBTQuant 50 % + kMatmul 16 % +
kMatmulBT 14 % = 80 % of prefill. mmla makes the q8_0 kMatmulBTQuant ~1.2×
faster on this file (the 8.4 % e2e is exactly `0.50·(1−1/1.2)`), so its share
drops to ~44 % and **the elementwise f16/f32 GEMM (kMatmul + kMatmulBT, ~30 %,
UNCHANGED — the 60 % f16 mass incl. the tied lm_head/embed) is now co-dominant
and the next CPU prefill lever on this file**; on a k-quant file kMatmulBTQuant
collapses (7–8× on q4_K) and the elementwise GEMM + attention dominate outright.
G5 (x86 AVX2/AVX512 quant tier) and the NEON-dotprod nrc==1 tier for i8mm-absent
aarch64 remain open follow-ups; G7 repack stays parked.

**Gates at G6:** clean CPU `-Werror` build 0 warnings; the new arm TU compiles
clean on dgx aarch64; CPU units green (test_ops_quant_dot 19/78,162,
test_ops_quant_traits 8/5,615, test_gguf_dequant, test_gguf_keep_quant,
test_cpu_threadpool, test_ops_matmul_elem); e2e engine gate 2/2 · 16/16 vs
llama.cpp. CUDA regression set UNCHANGED (see the parity ledger G6 row).

## G7 result (2026-07-23) — q8_0 repack-at-load, landed + measured; CPU prefill crosses llama.cpp parity

**What landed.** llama.cpp's repack-at-load for q8_0, the `q8_0_4x8_q8_0` tier
`ggml_repack_get_optimal_repack_type` selects on NEON + i8mm
(`repack.cpp:4683` @ `237ad9b96`). The keep-quant loader repacks each eligible
q8_0 weight ONCE at load into the `block_q8_0x4` interleave (`nrows_interleaved=4`,
`interleave_block=8`), and `kMatmulBTQuant` dispatches the pre-shuffled i8mm
gemm/gemv, which read contiguous aligned blocks with NO in-register row shuffles
(the mmla tier's per-block `vzip1q_s64` are gone).

**Ported (file:line both sides).**
- weight interleave `make_block_q8_0x4` (`repack.cpp:2725`) +
  `repack_q8_0_to_q8_0_4_bl` (`repack.cpp:3480`) → byte-permute in
  [cpu_quant_repack.cpp](../../src/vt/cpu/cpu_quant_repack.cpp#L1)
  (`RepackQ8_0Rows4`/`QuantRepackWeight`), arch-independent.
- prefill gemm `ggml_gemm_q8_0_4x8_q8_0` (NEON+i8mm, `arch/arm/repack.cpp:5091`)
  and gemv `ggml_gemv_q8_0_4x8_q8_0` (NEON+dotprod, `arch/arm/repack.cpp:1779`) →
  [cpu_quant_repack_arm.cpp](../../src/vt/cpu/cpu_quant_repack_arm.cpp#L1)
  (`GemmTileQ8_0`/`GemvRowQ8_0`, parallelized over weight COL-groups).
- activation side: rather than port `ggml_quantize_mat_q8_0_4x8`'s round-to-even,
  the repack path quantizes each activation row with the SAME `from_float`
  (`quantize_row_q8_0`, round-half-away-from-zero) as the tier-0/mmla path, then
  interleaves the bytes (`InterleaveQ8_0Rows4`) — bit-preserving.
- feature probe: `QuantRepackActive()` = compile-time aarch64 i8mm AND runtime
  `HWCAP2_I8MM` AND not `VT_CPU_QUANT_REPACK=0`, mirroring the mmla tier's probe;
  the loader repacks (`GgufLoadPolicy::quant_repack`) only under it, so the layout
  and the kernel that consumes it can never disagree.

**BIT-IDENTICAL (target met).** The transform is a byte permutation and the
gemm/gemv fold the per-block scale in the SAME block order with a NON-FUSED
`vmlaq_f32` (upstream uses fused `vfmaq_f32`; the swap, plus the global
`-ffp-contract=off`, is the one recorded deviation) — so the integer `vmmlaq`/
`vdotq` sums are exact and the float accumulation matches the tier-0/mmla path
bit-for-bit. Proven by [test_ops_quant_repack](../../tests/vt/test_ops_quant_repack.cpp#L1)
(dgx aarch64, 305 assertions): the repacked gemm/gemv is `memcmp`-identical to the
non-repacked `kMatmulBTQuant` across decode (M=1) / leftover (M%4) / prefill
(M%4==0), N∈{4..3072}, K up to 6144, f32 AND bf16 output, contiguous AND
row-strided activations, and thread counts 1/2/4/20; the interleave also matches
`make_block_q8_0x4` byte-for-byte and round-trips. On CI (x86, no i8mm) the
transform round-trips and the gating is asserted total; the numeric cases skip
coherently.

**A wiring bug the E2E gate caught (recorded).** The op path was bit-identical
from day one, but the first E2E run produced ALL-ZERO tokens: `ResidentWeight`
(qwen3_5.cpp) builds the CPU forward's weight `vt::Tensor` via `MakeTensor`, which
dropped the new `OwnedTensor.repacked` marker, so the kernel read repacked bytes
as a plain q8_0 weight. Fixed by propagating `repacked` at that single
host→kernel chokepoint (covers attn/ffn/GDN/router/experts — every weight tensor
routes through it); `Tensor::Slice` now also throws on a repacked weight
(defence — a repacked weight has no coherently sliceable axis; the merged
GDN/conv row-slices are safetensors-only bf16, never repacked). After the fix the
output-token md5 is `d235db12f2cd304007530286a1755c95` — byte-identical across
repack-ON (default), repack-OFF (`VT_CPU_QUANT_REPACK=0`) and `VT_CPU_REF=1`.

### Benchmark — binding, dgx.casa (GB10, aarch64, 20 cores), idle, one flock

`git archive` transfer, `Qwen3.5-2B-UD-Q8_K_XL.gguf`, same-binary A/B, loadavg
gated. llama.cpp `237ad9b96` refreshed same session (`-t20 -r5 -ngl0`): **pp128
177.32±2.50**, tg32 25.40±0.80.

**Op-level q8_0 GFLOP/s (cache-warm best-of-N, mmla tier → repacked):**

| shape (M×N×K) | mmla | **repacked** | ratio |
|---|---|---|---|
| prefill qkv 128×3072×2048 | 518.75 | **2400.92** | **4.63×** |
| prefill gate_up 128×12288×2048 | 583.37 | **3455.86** | **5.92×** |
| prefill down 128×2048×6144 | 514.13 | **1901.66** | **3.70×** |
| decode qkv 1×3072×2048 | 408.75 | 418.54 | 1.02× |

The pre-interleaved contiguous loads + the 4×4 output tile (16 outputs / mmla vs
the mmla tier's 2×2) remove the shuffle overhead and quadruple arithmetic
intensity; decode (M=1, gemv) is memory-bound and neutral.

**E2E prefill A/B** (`--input-len 128 --output-len 32`, `VLLM_CPP_CPU_THREADS=20`,
6 interleaved reps to cancel drift, cold rep discarded, median 2–6):

| arm | TTFT median (ms) | prefill t/s | TPOT (ms) |
|---|---|---|---|
| BEFORE `VT_CPU_QUANT_REPACK=0` (mmla) | 1096.38 | 116.7 | ~40.5 |
| **AFTER — repack default** | **571.88** | **223.8** | ~40.5 |

**Same-binary prefill 1.92×** — the binding, box-state-robust metric. Prefill t/s
223.8 vs llama.cpp pp128 177.32 = **1.26× — AT/BEYOND llama.cpp parity**, up from
0.66× (1.52× behind) on the repack-OFF arm this session (and from the recorded
1.32×-behind idle baseline; the same-binary 1.92× on that baseline lands ~255 t/s,
likewise beyond llama). Decode TPOT ~40.5 ms unchanged (24.7 t/s ≈ llama tg32
25.4, at parity — repack is neutral at M=1). Peak RSS 3.884 GiB unchanged (the
1.39× RSS gap is loader-bound f16 expansion, not G7's target).

### Fresh post-change profile + what remains

Prefill dropped 1096→572 ms; the q8_0 quant GEMM (`kMatmulBTQuant`, 55 % of
prefill at ~510 GFLOP/s) went ~4.6× faster, so its share of the OLD time (603 ms)
collapses to ~131 ms and its share of the NEW prefill is ~21 %. The now-dominant
prefill cost is the **elementwise f16/bf16 GEMM** (`kMatmulBT`, the tied
`lm_head`/`token_embd` + f16 ffn mass, ~48 % of the new prefill, UNCHANGED — not
a block encoding, so repack does not touch it), then the q8_0 GEMM, then GDN +
attention. **But the CPU prefill gap vs llama.cpp is CLOSED on this file** —
prefill now matches/beats llama.cpp — so this closes the CPU prefill-lever search:
what remains between us and llama.cpp is **peak RSS (1.39×, loader/f16-expansion
bound, not compute)**, not prefill speed. A k-quant-heavy file would see the
`kMatmulBTQuant` share (already mmla-fast on q4_K/q6_K) matter less again; adding
the k-quant repack layouts is a clean follow-up but not required to reach parity.

**Gates at G7:** clean CPU `-Werror` 0 warnings + clean CUDA `-Werror` 0 warnings;
[test_ops_quant_repack](../../tests/vt/test_ops_quant_repack.cpp#L1) 5 cases /
305 assertions on dgx aarch64 (110 on x86, numeric cases skip); CPU quant/gguf
units green (test_ops_quant_dot, test_ops_quant_traits, test_gguf_keep_quant,
test_gguf_dequant); `test_qwen36_gguf_engine` STANDALONE 2/2 · 16/16 on APEX
Compact+Balanced with repack live; CUDA regression set UNCHANGED (see the ledger
G7 row), goldens content-hash identical before/after.

## G5 plan (2026-10-31) — x86 AVX2 tier, unblocked by a fresh same-file profile

**Why G5 can start now.** The roadmap marked G5 blocked on two conditions: a
fresh op-dispatch profile and a non-`VOID` x86 host. Both are satisfied by a
direct same-file A/B on the Flare GSQ artifact, not by a stale percentage:

| Arm (same GGUF, greedy, 32 tokens, warm) | Request 1 | Request 2 | Startup |
|---|---|---|---|
| LumenEngine `evidence/gsq-lumen/candidate-server` (`-t 18`, keep-quant) | 14.80 s | 13.95 s | 25.8 s |
| llama.cpp b10451 `llama-server -t 18 -tb 18 -fa on` | 3.91 s | 4.24 s | 3.0 s |

Both emitted ` 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, `. The model is 3.8 GB with 234
IQ3_S, 104 Q6_K and 42 Q4_K tensors, so decode is dominated by exactly the
three `vec_dot`s that `BlockVecDot` still routes to the portable tier
([cpu_quant_dot.cpp:22](../../src/vt/cpu/cpu_quant_dot.cpp#L22),
`BlockVecDot` cases at `:1245-1249`). This is a measured 3.5× end-to-end gap on
the user's model, not the q8_0/f16 mix that made the 2026-07-22 profile rank
G5 behind the elementwise GEMM.

**Scope.** Add `src/vt/cpu/cpu_quant_dot_avx2.cpp`, one translation unit built
with `-mavx2 -mf16c` (the existing per-file pattern at
[CMakeLists.txt:1545](../../CMakeLists.txt#L1545)), containing the three
upstream AVX2 kernels for the encodings this artifact uses. Keep the portable
tier as the universal fallback and select at runtime from
`DetectX86IsaCaps().avx2`, with `VT_CPU_QUANT_AVX2=auto|portable` as the test
and benchmark seam (the same shape as `VT_CPU_Q8_DOT`).

**Upstream anchors** (registered llama.cpp b10451 source,
`ggml/src/ggml-cpu/arch/x86/quants.c`):

| Component | Anchor | Ported as |
|---|---|---|
| `ggml_vec_dot_q4_K_q8_K` AVX2 | `:2018-2100` | `VecDotQ4_KQ8_K_AVX2` |
| `ggml_vec_dot_q6_K_q8_K` AVX2 | `:2406-2478` | `VecDotQ6_KQ8_K_AVX2` |
| `ggml_vec_dot_iq3_s_q8_K` AVX2 | `:3364-3456` | `VecDotIQ3_SQ8_K_AVX2` |
| `hsum_float_8`, `get_scale_shuffle_k4`, `get_scale_shuffle` | `:42-50`, `:495-530` | local helpers, same values |
| block layouts | `ggml-common.h:317-327, 352-357, 386-392` | already mirrored in `cpu_quant_blocks.h` |

The upstream FMA intrinsics become `_mm256_add_ps(_mm256_mul_ps(...))` because
this tree builds with global `-ffp-contract=off`; the reduction order is
otherwise upstream's. That keeps the AVX2 TU valid on AVX2 CPUs without FMA and
preserves the repository's non-fusion discipline.

**Tests.** Extend `tests/vt/test_ops_quant_dot.cpp`: the existing G3
`vec_dot == independent f64 dequantize-then-dot` case runs unchanged and now
exercises the AVX2 arm on this host; add a selection test that
`VT_CPU_QUANT_AVX2=portable` restores the portable kernel and that the two arms
agree within the G3 tolerance. The same-file runtime gate is token equality
against the recorded baseline plus `verify.py`, not a new golden.

**Gates.** (1) `test_ops_quant_dot` green with AVX2 auto and with
`VT_CPU_QUANT_AVX2=portable`; (2) the Flare server built from the same source
prints the identical 32-token completion as the recorded baseline; (3) warm
request latency improves on the same host and same binary recipe, reported with
the contention state; (4) `verify.py` passes. vLLM has no CPU GGUF decode path,
so this leaf makes no vLLM parity claim; llama.cpp remains the secondary oracle
used only for the denominator.

**Stop conditions.** Any token divergence, non-finite value, build warning, or
G3 failure under either arm stops the change. A measured regression means the
tier stays opt-in (`VT_CPU_QUANT_AVX2=portable` is the default) and the result
is recorded, not widened into the portable arm.

**Evidence.** `.agents/evidence/quant-g5-avx2/` holds the baseline log, the
post-change log, the exact commands, and the host contention state.

## G5 result (2026-10-31) — x86 AVX2 tier, landed + installed on the Flare host

**What landed.** `src/vt/cpu/cpu_quant_dot_avx2.cpp`, one translation unit built
with `-mavx2 -mf16c`, carrying the AVX2 `vec_dot` arms of llama.cpp b10451
`arch/x86/quants.c:2018-2100` (q4_K), `:2406-2478` (q6_K) and `:3364-3456`
(IQ3_S), selected at runtime by `DetectX86IsaCaps().avx2` through
`SelectQuantKVecDot`, with `VT_CPU_QUANT_AVX2=auto|portable|avx2`. The portable
tier stays the universal fallback and is also the defeat seam that makes a
same-binary A/B possible. `BlockVecDot` gained the three dispatch arms, so the
tier is reachable from production decode, not only from the selector.

**Same-binary A/B** (one candidate binary, Flare GSQ
`Nemotron-3.5-Bolt-E4B-GSQ-RCO-IQ3_M.gguf`, greedy 32 tokens, one warmup,
interleaved arms, Xeon E5-2696 v3 18c/36t AVX2/no-AVX-512, shared host at
load average 16-19):

| Arm | Request 1 (s) | Request 2 (s) | Mean (s) |
|---|---|---|---|
| `VT_CPU_QUANT_AVX2=portable` | 16.874 / 16.536 | 16.630 / 16.689 | 16.68 |
| `VT_CPU_QUANT_AVX2=avx2` | 6.158 / 6.403 | 6.188 / 6.459 | 6.30 |

**2.65×** decode from the three kernels, and all four runs emitted the same
32-token text byte for byte. Across the whole served artifact the ported
encodings are 234 IQ3_S + 104 Q6_K + 42 Q4_K tensors, so this is the
decode-dominant path, not a microbenchmark.

**Against the installed baseline and the secondary oracle.** The pre-G5
installed binary measured 14.39/14.76 s at 25.8 s startup; llama.cpp b10451
`llama-server -t 18 -tb 18 -fa on` measured 3.91/4.24 s at 3.0 s startup on the
same file. G5 therefore closes the decode gap from ~3.5× to ~1.5× and leaves it
open: the three `vec_dot`s now execute llama.cpp's own algorithms, so the
residual is elsewhere and is **not** declared a ceiling.

**Live service after install** (load average 17.22): 6.23 s and 6.21 s for the
same 32 tokens with identical text; `verify.py --expect-cache on` PASS
(prefix-cache queries 115, hits 96 at the re-run, completion ` a mammal.`,
`finish_reason=stop`). The default `gsq-rco-cpu` profile serves the AVX2 tier;
`VT_CPU_QUANT_AVX2=portable` and the pre-G5 binary are both recorded rollbacks.

**Defaults, and why each has its value.**

- `VT_CPU_QUANT_AVX2=auto` is the default because both gate 1 (auto and
  portable green) and gate 3 (latency improves) passed; the stop condition that
  would have kept the tier opt-in did not trigger.
- `-mavx2 -mf16c` is per-file, matching the existing per-TU ISA pattern, so the
  rest of the binary keeps its baseline ISA and an AVX2-less x86 CPU falls back
  at runtime rather than faulting.
- FMA intrinsics are written as `_mm256_add_ps(_mm256_mul_ps(...))` because the
  tree builds with `-ffp-contract=off`; the reduction order is otherwise
  upstream's, which is what keeps the token gate meaningful.

**Rejected alternatives.**

- *AVX-512 tier.* The gating host has no AVX-512, so it cannot be exercised
  here. Recorded as owed, not shipped blind.
- *Adding IQ3_S to `kWeightCases`.* The shared MatmulBTQuant NMSE case runs at
  m=1,n=1, where an IQ3_S dot cancels near zero and the unmutated NMSE is
  already ~1.10. A G5-local case list covers IQ3_S without weakening that
  ceiling.
- *One kernel with a runtime branch inside the block loop.* It reintroduces a
  per-block branch on the decode hot path; llama.cpp keeps separate TUs and so
  does this port.
- *Widening the portable kernels instead of adding a tier.* The portable tier
  is the fallback and the A/B baseline; changing it would remove both.

**Correctness.** `test_ops_quant_dot` is 35 cases / 285,691 assertions green
under `auto`, `portable` and `avx2`. The G5 case proves all three things a
selector-only test cannot: `auto` chooses the AVX2 arm, `portable` really
restores the portable kernel, and both stay within `1e-5 * L1` of an independent
f64 dequantize-then-dot reference for q4_K, q6_K and IQ3_S. The mutation battery
reddens the numeric checks for an IQ3_S sign change and for the `2*ls+1`
weight, and deleting the production `BlockVecDot` arm reddens the routing
checks, so the case measures a reached capability.

**Owed.** A merged-head x86 profile to attribute the remaining ~1.5× (the
2026-08-06 aarch64 profile put 47 % of decode in threadpool synchronisation at
M=1; that hypothesis is untested on this host). Startup is 25.8 s against
llama.cpp's 3.0 s. AVX-512, and the AVX2 arms for q3_K/q5_K/q8_0/q4_0, are not
ported because this artifact does not use them.
