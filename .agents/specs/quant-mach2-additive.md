# QUANT-MACH2-ADDITIVE — Syzygy Research's Mach-2 additive codec

State: `SPIKE`. Owner: `CLAIM-QUANT-MACH2-ADDITIVE`. Row: `QUANT-MACH2-ADDITIVE`.
This spec lands before any code, as a separate pull request, which is the shape
the developer selected on 2026-11-16 (see `## Git integration`).

## What the thing is

Mach-2 Additive Medium is `Qwen/Qwen3.8-Flash-Next` at 1.70 bits per weight:
26.7 GB for 125.7B text parameters, plus the model's n-gram embedding table at
bf16 (102.4 GB). Its weights are not a `ggml` quantization type and not an
safetensors of small integers. They are **packed code streams** that a decoder
expands into weights, and every weight-matrix product is claimed to be
addition-only because each codeword is a short list of small integers times one
shared scale.

Two artifacts exist:

| Artifact | Format | Oracle |
|---|---|---|
| `SyzygyResearch/Mach-2-Additive-Medium` | native packed, tag `fn_packed_v2`, specified by the repo's own `decode.py` | `decode.py` |
| `SyzygyResearch/Mach-2-Additive-Medium-GGUF` | `*.mach1.gguf`, 130.1 GB, needs `SyzygyResearch/llama.cpp-mach1` | that fork |

**This row targets the native packed arm only.** The GGUF arm is deferred, and
the reason is a documentation gap rather than a preference: the fork's
`docs/mach1.md` documents **payload v3 for Mach-1** and eight `MACH1_*` ops,
while the Mach-2 GGUF card names no payload version. Whether the documented fork
ops decode Mach-2 is therefore unverified, and an oracle we cannot read is not a
denominator.

## Upstream anchors

vLLM implements none of this — no trellis codec, no additive codeword, no
wave-scaled tiles — so `QUANT-MACH2-ADDITIVE` is a secondary-oracle row under
"when vLLM has no implementation" in `AGENTS.md`, not a mirror gap. The oracle is
pinned at [`.agents/oracles/mach2-native.md`](../oracles/mach2-native.md), which
records `gateable = no` until an artifact runs.

The nearest in-tree neighbour is `QUANT-EXL3`, which is also a QTIP bitshift
trellis with blockwise rotation and square sign vectors:

- `include/vllm/model_executor/layers/quantization/exl3.h`, `exl3_checkpoint.h`
- `src/vllm/model_executor/models/dense_exl3_linear.cpp`
- `src/vt/cpu/cpu_exl3_dequant.cpp`, `src/vt/cpu/cpu_exl3_kernels.cpp`
- `src/vt/cuda/cuda_exl3.cu`, `src/vt/vulkan/shaders/vt_exl3_*.comp`,
  `src/vt/rocm/rocm_exl3.hip`

The **structure** there is the seam to reuse: a packed container, a trellis walk,
a LUT, a rotation, and a keep-quant matmul. The numerics are not reusable; they
differ in every constant below.

## The format, as `decode.py` states it

Read from `decode.py` at the weight repository on 2026-11-16. Constants:
`HIDDEN=2560`, `INTER=640`, `NEXP=512`, `NLAYERS=48`, base revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, `FORMAT="fn_packed_v2"`.

**Expert trellis** (`decode_expert`, `_ex_states`). `EX_V=4` weights per codeword,
`EX_L=16`-bit window, `EX_TD=16` tile. The tape is `uint16` words; bits are
unpacked MSB-first, **tail-biting** (`bits[:, :EX_L-step]` appended to the end),
and the window advances by `step = int(K4)` bits per codeword. `K4` is
**per-expert** (`{proj}.rate_k4`), so 6 bits per 4 weights (1.5 bpw) and 7 bits
per 4 weights (1.75 bpw) coexist inside one layer — that mixture is the 1.70
average. Each of the 2^16 window states indexes one LUT row
(`lut.k{K4}` or `lut`), and the row is a D4 lattice point times a shared scale.
Tiles are laid `Mb × Nb` and transposed `(0,2,1,3)` into `m × n`, then scaled by
`{proj}.wave_gamma[e]` gathered through `wave_map(Mb, Nb)` — a diagonal wave
enumeration starting at the bottom-left corner and the bottom row — then by the
per-expert scalar `{proj}.wscale[e]`.

**Rotation.** `hadamard(x, kind)` with `kind ∈ {expert, spine}`. Power-of-two
columns use butterflies and a `1/sqrt(N)` scale. **Non-power-of-two columns use a
radix-12 or radix-20 matrix** built by `_hmat` from a quadratic-residue character
over `Z/5` (radix 12) or `Z/3[i]` (radix 20), via the `kron(C, A) + kron(I, B)`
construction, and the code asserts `H == H.T` and `H @ H == r·I`. A spine call
uses `1/sqrt(N)` across the whole row; an expert call uses `1/sqrt(radix)` and
recurses on the remaining block. Signs `SU`/`SV` are stored per expert, or, when
absent, are **derived**: `sign_seed(layer, expert, proj, side)` hashes
`"canon|L{L}|e{e}|{proj}|{side}"` with SHA-256 and keeps the first 7 bytes;
`expert_signs` expands each seed through a 10-round Philox-style
multiply-shift-xor with two Weyl increments and turns the counters into `±1` via
a `v <= π or u == 1` test.

**Spine / `ne` tensors** (`decode_ne_tensor`, `_ne_states`, `_ne_full_lut`).
`NE_K=4`, `NE_L=16`, `NE_V=2` weights per codeword, `NE_TD=16`, and
`NE_TLUT_BITS=9`. The trained table is **512 rows**, and the 2^16 states are
folded onto it: `row = (p >> 6) & 511` where `p = s·(s+1)` for state `s`, and bit
15 of `p` flips the sign of the row's first component. Tensors are named
`{name}|trellis`, `|Wscale`, `|SU`, `|SV`, `|rc_max`; after the rotation the
result is multiplied by a row grid and a column grid
(`rc_grid(mx, |sv|) = mx·k/127`) with a **bf16 round** of the intermediate. The
shard's `dims` metadata carries `[rows, cols, m, n]` per tensor for cropping.

**Head, embedding, rows.** `head`: int5, eight values packed into five bytes,
reassembled little-endian into a `u64`, `-16` biased, then a **group-64** scale
per column group. `embed_int{4}`: asymmetric grouped int4, per-group `mn`/`mx`
with `step = (mx-mn)/(2^bits-1)`. `int8_rows`: int8 with a per-row scale.

**Base pieces.** Vision (333 `model.visual.*` tensors) and the MTP layer come
from the bf16 base checkpoint at the pinned revision; they are not quantized.

## Design

Five units, each its own pull request, each gated before the next starts.

- **U1 — the decode oracle.** A CPU-only port of the packed decode: state walk,
  LUT expansion, wave map and gamma, scale chain, radix-12/20 Hadamard,
  butterfly Hadamard, sign derivation, folded spine LUT and `rc_grid`, int5 head,
  grouped int4 embed, int8 rows. No loader, no matmul, no artifact: it turns
  packed bytes into f32/bf16 weights and proves bit-exactness against the release
  decoder. This is the IQ2_XXS pattern — hand-derived literals, no download.
- **U2 — fixtures and the differential gate.** A committed fixture generator that
  imports the *unmodified* `decode.py` functions, plus small synthetic shards
  (tens of KB) whose decoded bytes are committed. The gate is byte equality, not
  a tolerance, because the release decoder is the format definition.
- **U3 — the seam.** `fn_packed_v2` container read plus a `LinearMethodBase` arm
  on the `QUANT-EXL3` structure, reaching `ModelRegistry::Forward` through the
  existing dense and MoE paths. Nothing lands dead: the smallest failing test
  enters through the loader.
- **U4 — keep-quant kernels.** The additive dot (adds only, scale once per tile)
  for CPU first, then CUDA/Vulkan, with the Q8-style activation quantisation the
  tree already uses for keep-quant rows.
- **U5 — the artifact gate.** One leased GPU, the packed checkpoint, the n-gram
  table in host RAM, and the gate shape chosen once greedy stability is measured
  (see `## Risks`).

`decode.py` is the oracle, so the repository must not "improve" it: fixtures come
from it unchanged, and any local repair of it is recorded as ours.

## Risks

- **No token-exact gate is available yet.** The model card states the model can
  loop without temperature 1.0 / top_p 0.95 / top_k 20, so a greedy stream may
  not be stable and a token gate may be impossible. This is why the decode
  oracle is U1: byte-exact weights are deterministic and need no sampling, and
  the model-level gate is chosen after measurement rather than assumed.
- **`decode.py` may not be runnable as published.** The fetched text loses
  leading whitespace in places (`C[1:, 1:] = Q` and several `return`s sit at
  column 0), which would be an `IndentationError`. Whether that is the file or
  the fetch is **unverified**; U2 must download it, `py_compile` it, and record
  the answer before fixtures are generated from it.
- **The n-gram table is 102.4 GB of bf16** and must stay in host RAM
  (`--mlock`, ~105 GB free) while 26.7 GB of weights sit on a 32 GB GPU. A dgx
  lease may not satisfy both at once; that is a lease-sizing question, not a
  codec question.
- **Two constructions must be bit-exact or nothing matches**: the radix-12/20
  `_hmat` character tables, and the Philox-style sign derivation. Both are pure
  integer/float-deterministic, so they are checkable, but a single wrong constant
  makes every tensor wrong.
- **The wave map's orientation** (bottom-left, walking up-right) and the tile
  transpose are easy to mirror wrongly and produce plausible-looking weights
  with the wrong arrangement. U1 pins both with a literal 2×3 case.
- **bf16 rounding inside the spine path** is part of the release semantics; a
  better-precision intermediate would diverge from the oracle.

## Tests

- State walk: literal expectations for a 2×2 tile at `K4=6` and `K4=7`,
  including the tail-biting wrap.
- `wave_map`: the exact index matrix for `(Mb, Nb) = (2, 3)`.
- Hadamard: `H Hᵀ = r·I` for radix 12 and 20 to exact equality in f32, plus a
  literal first row and first column from `_HS`.
- Sign derivation: fixed `(layer, expert, proj, side)` seeds and their literal
  first eight `±1` values.
- Folded spine LUT: the 512→65536 expansion for a table of known values,
  including the sign flip at bit 15.
- `rc_grid`: literal values for `mx = 127`, `k = 0..3`.
- `unpack_int5`: a five-byte group expanded to eight values, boundary `-16`.
- Grouped int4 embed: `mn = mx` degenerate case, and a two-group row.
- Whole-tensor differential: the U2 fixtures, byte-exact after the bf16 round.

## Gates

- **G1** — U1 compiles and its unit suite is green on a CPU-only build, with the
  suite registered in `scripts/agent-preflight.sh` and `.github/workflows/ci.yml`
  in the same change.
- **G2** — the U2 differential gate is byte-exact on every fixture.
- **G3** — U3's reachability test enters through the loader and fails when the
  production call site is deleted (the reviewer mutates for this).
- **G4** — U5: the artifact loads and runs on a leased GPU with a recorded
  command, and the gate shape is ratified at that point.

## Evidence

- 2026-11-16: format read from `decode.py` at
  `SyzygyResearch/Mach-2-Additive-Medium` (native packed, `fn_packed_v2`).
- 2026-11-16: codec design read from `withsyzygy.com/mach-2` and the
  "Bits per _Byte_" primer (2026-10-11), which state the 16-bit window, the 6-bit
  shift per 4 weights, the D4 snap, the integer cap of 7, and the add-only dot.
- 2026-11-16: the GGUF arm examined via
  `SyzygyResearch/Mach-2-Additive-Medium-GGUF` (130.1 GB) and
  `SyzygyResearch/llama.cpp-mach1` `docs/mach1.md` (payload v3, eight ops,
  Mach-1 named). No statement anywhere maps Mach-2 onto those ops.
- 2026-11-16: `QUANT-EXL3` read as the nearest seam; its trellis, Hadamard-128
  and sign-vector machinery is structurally similar and numerically different.
- No weight was downloaded, no GPU was leased, and no decode was executed. Every
  claim above is read from source text, and this row has no measurement yet.

## Owed

- `GGUF mach1` arm: deferred until the fork's Mach-2 payload version is
  documented, then it is a separate row.
- The n-gram table mechanism (102.4 GB, read a few rows per token) — a
  host-resident lookup tier this repository does not have.
- The model-level correctness gate, owed with its shape, once U5 measures greedy
  stability.
- `docs/USAGE.md` checkpoint documentation: owed when U3 makes the arm reachable,
  not before.

## Stop conditions

- `NEEDS_DECISION` if U2 cannot make `decode.py` runnable, or if the radix-12/20
  construction or the sign derivation cannot be reproduced bit-exactly. Both
  would mean the format definition is unreachable and the arm cannot be gated.
- `NEEDS_DECISION` before weakening G2 from byte-exact to a tolerance. The
  decoder defines the format; a tolerance would silently redefine it.
- `PENDING` while no artifact can be staged for U5.

## Git integration

Split, per the developer's answer on 2026-11-16: this pull request carries the
spec, the row registration and the oracle pin only. U1 onwards are separate
pull requests, each red-first against this spec.

## U1 as built (2026-11-16)

`row/QUANT-MACH2-ADDITIVE-U1` lands `include/vt/mach2_decode.h`,
`src/vt/cpu/cpu_mach2_decode.cpp` and `tests/vt/test_mach2_decode.cpp`
(13 cases, 2714 assertions), with both the TU and the suite registered in
`CMakeLists.txt` and `tests/CMakeLists.txt`.

**Landed:** the wave map, the tail-biting trellis state walk, the expert LUT
gather, the folded 512-to-65536 spine LUT with its sign bit, `rc_grid`, int5
unpacking, grouped integer dequantisation, the bf16 round, the butterfly
transform, the quadratic-residue radix-12/20 matrices, the expert orthonormal
transform, the Philox sign expansion, and the expert tile assembly (gather,
per-tile `wave_gamma`, `wscale`, both rotations, both sign tables).

**Still owed by U1:** the `_HS` string tables the SPINE rotation uses, and the
SHA-256 that turns `sign_seed()`'s text into a seed. The string tables are
deliberately not embedded: the fetched `decode.py` text is not yet trusted, and a
table copied from a possibly-mangled source is worse than no table, because a
test written against the same text would agree with it. The SHA-256 is a loader
concern rather than a decode primitive, so `ExpertSigns` takes the derived seed.

**Four findings.**
- `wave_map` assigns ONE index per anti-diagonal - the start index, constant
  along the walk - not a running counter. The bottom-left corner is therefore
  visited twice, and on a 2x3 tiling index 1 belongs to no tile at all. The first
  expectation in this row was written against a counter and failed, which is how
  the distinction surfaced.
- The first version of the suite contained no rounding TIE, so mutating the bf16
  bias from `0x7FFF` to `0x8000` (round-half-up) survived it. The tie case
  `0x3F808000 -> 0x3F800000` and its odd control are now in the suite and that
  mutation fails.
- `wave_gamma` is indexed by anti-diagonal START, not by tile, so the array must
  hold at least `mb + nb` entries. The release decoder has the same requirement:
  the transcription of `decode.py` raised `IndexError` on a 1x1 tiling with a
  one-element gamma array, because that tiling's map reads index 1. Copying the
  number of tiles into the gamma length is therefore a real bug, not a style
  choice.
- A codeword's four values become four consecutive COLUMNS of one row of the
  tile, not four rows: the gathered stream is reshaped to
  `(mb, nb, TD, TD)` before the transpose, so the last axis spans four `c`
  positions per codeword. Reading the codeword as a column quadruple instead
  changes every value in the tile.
