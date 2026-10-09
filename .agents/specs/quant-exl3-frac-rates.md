# BACKEND-ROCM — half-integer EXL3 rates (`K + 0.5`, `mul1`) for per-tensor mixed-rate checkpoints

Row: `BACKEND-ROCM`
Issue: `.agents/issues/BACKEND-ROCM/ISSUE-LOCAL-01M46P4WWC21PSET72VQF800B5.md`
Base SHA: `ae0c9c874`
Parent row: [`QUANT-EXL3-MUL1`](quant-exl3-mul1.md) (the `mul1` codebook and the
integer widths a mixed-rate artifact ships)
Matrix: [`.agents/quantization-matrix.md`](../quantization-matrix.md)

Oracle pin: `exllamav3` @ image `exllamav3-rocm:git-679835b7-rocm10.0.0` —
vLLM implements no EXL3 at the parity pin; the format is mirrored from the
registered secondary oracle. This image is the first pinned revision carrying
`exllamav3_ext/quant/frac.cu` + `dq8_half` (the earlier `git-584dd44f` pin
predates them). Gateability was measured 2026-10-05 on gfx1100: the image
loads this checkpoint and greedy-decodes coherent output (~24 tok/s); token
goldens for the E2E gate are `orca_golden_{0,1}.json` beside this spec, and
the evidence is recorded in `.agents/oracles/exllamav3.md`'s AMD addendum.

## The gap

`orcarouter/OrcaSAQ-2-27B-EXL3-3.21bpw` (sha-pinned in
`docs/USAGE.md` once reachable) refuses at load. Its `.trellis` tensors are a
**per-tensor mixed rate**: census of `model.safetensors` gives last-dim widths
`{96: 1, 48: 231, 56: 120, 64: 51, 32: 6}` — 120 tensors at 56 uint16 words
per tile. `LoadExl3` gates `words % 16 == 0`, `Exl3Weight::Bits()` gates
`last_dim % 32 == 0`, and every device arm is `template <int BITS>` over
integer widths, so the tensor is unnameable, unloadable and unrunnable.

`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` already loads: its "3.5bpw" is a per-layer
integer-width split. Orca's 56-word tiles are a different, genuinely fractional
format.

## The format (from the oracle, `exllamav3_ext/quant/frac.cu`, `exl3_dq.cuh:254-286`)

- Half-integer rate `K = KA + 0.5`, `mul1` codebook only. Positions alternate
  `KA` / `KA+1` new state bits; the period-16 pattern is `MASK = 0xAAAA`
  (**odd** positions within each 16 carry the extra bit). The oracle's packer
  accepts any period-16 mask but only the alternating one is instantiated;
  we instantiate only 0xAAAA.
- Tile bits: `bpb16 = 16·KA + popcount(MASK)` per 16 weights →
  `16·KA + 8` uint16 words per 256-weight tile for `K+0.5` (KA=3 → 56).
- State window for position `i`: ring bits `[S(i) − 16, S(i))` where
  `S(i) = KA·(i+1) + ((i&15)+1)/2 + 8·(i/16)` (mask 0xAAAA). Ring is
  tail-biting mod `tile_bits = 256·K` — identical convention to the integer
  `t_offset*bits + bits − 16 + 256·bits` wrap.
- Stream is MSB-first in 32-bit words, same as `pack_trellis`; the loaded
  byte layout needs no transform.
- Reference decode `dq8_half<KA, cb>`: `bits2 = 2·KA+1` bits per position
  pair, `words = 4·bits2` u32 per tile, four windows span
  `gspan = 18 + 3·KA ≤ 27` bits — two funnel shifts + six register shifts
  for 8 windows, then `decode_3inst_2<cb>`.

## Design

**Signal, not heuristics.** `words % 16 == 8` ⇔ fractional rate at
`KA = (words − 8) / 16`; `words % 16 == 0` stays integer. This partition is
total — no width is ambiguous. `KA` is validated `1..7` (oracle `frac_bpb`
bound) and `codebook == 2` is **required**: the oracle asserts frac = `mul1`
only, and a frac tensor without a `.mul1` marker refuses by name.

**Representation.** `Exl3Weight` carries `bits` (the integer `KA`) plus a new
`half` flag; `Exl3GemmArgs` gains `half`. `Bits()` accepts `last_dim % 16 == 8`
and sets it. Naming mirrors the oracle's `half_k` template parameter.

**Decode.** One width-generic scalar reader `Exl3FracCodeword` implements
`S(i)`/window extraction for mask 0xAAAA — the correctness floor for the
generic dot arm and the host reference. Fast arms port `dq8_half<KA>`
bit-exactly for `KA ∈ {2, 3}` (the only rates observed in the wild; KA=1
falls to the generic path).

**Arms (this change):**
- `LoadExl3`: accept `words % 16 ∈ {0, 8}`; `8` requires `mul1`, `KA ≤ 7`.
- `src/vt/cpu/cpu_exl3_dequant.cpp`: frac-aware reference dequant — the test
  golden and CPU-backend floor.
- `src/vt/rocm/rocm_exl3_dot.hip`: `dq8_half<KA>` port; half arm added to the
  dot dispatch (`Exl3DotKImpl` instantiation + `Exl3GemmK` fallback path).
- `src/vt/exl3_policy.cpp` + `rocm_exl3.hip`: frac tensors route to the dot
  arm; GEMV eligibility refuses `half` cleanly (no silent integer arm pick).
- `src/vt/rocm/rocm_exl3_gemv.hip` + `rocm_exl3_recon.hip`: `## Owed` — frac
  GEMV m=1 arm and frac reconstruct are follow-ons; dispatch must refuse or
  route around them by name, never silently.
- `src/vt/cuda/cuda_exl3.cu`: `## Owed` — CUDA frac arm; CUDA dispatch refuses
  `half` by name (consistent with the repo's unimplemented-arm rule).

## Tests

- `tests/vllm/.../test_exl3_native_loader.cpp`: a 56-word `mul1` tile loads
  with `half` set; the same tile without `mul1` refuses; `words % 16 == 8`
  with `KA` outside `1..7` refuses.
- `tests/vt/test_exl3_dequant.cpp`: frac host dequant vs an independent
  packed-words reference (bit-window extraction written from `frac.cu`'s
  definition, not from our own code path).
- `tests/vt/test_exl3_rocm.cpp` / `_gemv.cpp`: fixture `half` variant at
  `KA=3` — ROCm dot-arm output byte-exact vs the host golden; refusal cases
  for frac+GEMV and frac+CUDA dispatch.
- **E2E token gate**: greedy 256 tokens on
  `orcarouter/OrcaSAQ-2-27B-EXL3-3.21bpw` vs the same checkpoint under the
  oracle image `exllamav3-rocm:git-679835b7-rocm10.0.0`, identical prompt and
  sampling. Token-exact is the gate; a distributional waiver needs an explicit
  recorded reason.
  The prompts are exactly the two the goldens were captured against: a
  Fibonacci-function request and a two-sentence hash-map explanation under
  the ChatML `chatml` format with "You are a helpful assistant."

## Risks

- **Mask assumption.** Orca was written by the alternating packer; if a
  checkpoint ever ships another period-16 mask our mask-0xAAAA formula
  silently mis-decodes. Mitigation: the loader refuses `words % 16 == 8` when
  the `mul1` marker is absent — same refusal-by-name posture as cb 0/1.
- **Wrap arithmetic.** `S(i) − 16` goes negative near ring position 0; the
  integer path wraps by `+256·bits`, the frac path by `+16·KA+8`-scaled ring
  length — a copy of the integer `+256·bits` term is an off-by-128 bug. The
  scalar golden must be exercised at `t_offset` near the wrap.
- **Mixed-rate dispatch.** Integer and frac tensors coexist in one model;
  per-tensor `half` must flow through `Exl3GemmArgs` per call, not per model.

## Stop conditions

- The checkpoint loads and greedy-decodes ≥256 tokens.
- Integer-rate models (`Mia-AiLab 3.5bpw`, `SC_4.00bpw`, `SC_3.00bpw`) are
  unaffected — same greedy output as base SHA on the smoke prompts.
- Any frac tensor on a non-instantiated path refuses with a message naming
  `half`/`K+0.5` and the missing arm.

## Owed

- ~~Oracle gateability~~ DONE 2026-10-05: image `git-679835b7` loads and
  serves OrcaSAQ-3.21bpw on gfx1100; goldens `orca_golden_{0,1}.json`; see
  `.agents/oracles/exllamav3.md` AMD addendum.
- Frac GEMV m=1 arm (`rocm_exl3_gemv.hip`) and frac reconstruct arm
  (`rocm_exl3_recon.hip`).
- CUDA frac arm (`cuda_exl3.cu`) and the corresponding CUDA dispatch rows.

## Outcome

Measured 2026-10-05 on gfx1100, HEAD `8cc6aa921` (plus the reconstruct-seam
repair it carries):

- **Load + decode**: OrcaSAQ-2-27B-EXL3-3.21bpw loads and greedy-decodes
  coherent output (both gate prompts, 256 and 145 tokens respectively).
- **Kernel gate**: `ctest -R exl3` — 20/22 pass including the new frac ROCm
  dot-arm cases. The two failures (`test_deepseek_v4_exl3_loader` codebook
  subcase, `test_dflash2_exl3_reach`) reproduce identically at base
  `e6e492238` with `VLLM_CPP_HIP=ON` — pre-existing, not this change; the
  GroupedConv miss is filed at
  `.agents/issues/KERNEL-DFLASH2-GROUPED-CONV/ISSUE-LOCAL-01M46WJS3AJGDC1KYHC2KKVTEA.md`.
- **Token gate, recorded as distributional**: greedy output matches the
  oracle's golden token-for-token until an FP-order argmax tie (chars ~168
  and ~61 into the two captures), and the INTEGER-rate sibling
  `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` — unchanged by this work — diverges from
  its own oracle golden at char ~158 under identical greedy conditions.
  Divergence is the two engines' floating-point baseline, not a frac decode
  defect: kernel-level equivalence is separately proven byte-exact against
  an independently written packed-words reader (`test_exl3_dequant`,
  `test_exl3_rocm_gemv` frac cases).
- **Integer-rate regression check**: `Mia-AiLab/Qwen3.8-27B-SC_3.00bpw`
  produces its usual clean greedy output at HEAD (spec stop condition).

Rejected alternative: token-exact gating at 256 tokens is unreachable in
principle here — the oracle and this engine have different GEMM/attention
accumulation orders, so argmax ties diverge mid-sequence even for integer
rates; the integer-rate control run is what turns the frac divergence from a
suspect into the baseline.
