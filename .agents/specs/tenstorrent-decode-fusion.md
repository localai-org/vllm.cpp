# Spec: fused single-program decode for the Tenstorrent keep-quant word chain — one program at the 1,024 B floor replaces ~85

Row: `TT-DECODE-FUSION`. State: DRAFT (2026-09-29).
Issue: `ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ` — this row RIDES the
existing issue; see `## Issue routing` below.
Builds on: the 2026-09-29 attribution leg (the trace-config-page bisect,
`docs/bench-evidence/tt-trace-config-page-repro-20260929.md`) that proved
the keepquant region's 2,965,504 B record is pure program COUNT — 8 chunks
× ~85 eltwise decode programs + ~8 matmul-chain programs, each paying the
4-17 KB per-program record floor that the packed relay cannot collapse —
and the RTA-fix commit (`a1661114b` + successor) that already put the
keepquant program on SetCommonRuntimeArgs-only launch with in-kernel row
derivation.
Precedent: `kKeepQuantInt8DotKernelSrc`
(`src/vt/tenstorrent/tenstorrent_keepquant.cpp:1558`) is the existence
proof at the floor — ONE full-grid custom kernel, 1,024 B per captured
launch, 4 uniform CBs, one common-args vector, per-core row0/rowc derived
in-kernel from the core coordinate. This row gives the DECODE arm the same
shape.
Git integration: ONE pull request (spec + implementation + tests +
evidence), continuing the developer's recorded 2026-09-28 choice under
`## Git integration` in `.agents/developer-preferences.md`.

## Scope

The W4a grouped decode chain launches, per ceil(N/8)=8-row chunk,
~85 small eltwise device programs from
`DecodeKeepQuantWordsF32` (`src/vt/tenstorrent/tenstorrent_keepquant.cpp:629-719`,
Q6_K arm) to dequantize packed quant words into f32 activations. At ~4-17
KB of trace record per program that is ~2.9 MB per captured chunk-chain
launch class — the measured 2,965,504 B region record — and the whole
27B decode cannot fit DRAM staging. This row collapses the ~85-program
eltwise chain to ONE custom kernel per chunk-chain launch (full grid,
one program), turning the red gate at
`tests/vt/test_tenstorrent_backend.cpp:11527` (keepquant region ≤ 64 KiB;
measured 2,965,504 B) green.

OUT of scope: the matmul chain (~8 programs per chunk) — it stays as
today unless the fused decode kernel's plumbing makes folding it trivial
(a recorded decision either way, not silent scope creep). OUT of scope:
other encodings' decode arms beyond what the per-enc-select bit-exactness
tests demand — Q6_K first, the other enc-select arms through the same
kernel dispatch as the tests require, each bit-exact before it serves.

The current eltwise chain STAYS in the tree as the named-decline arm
(env-gated, default flipping to fused only after the bit-exactness gate
holds): a shape or encoding the fused kernel does not yet serve declines
by name and runs the proven chain, never silently.

## Upstream anchors

- vLLM dequant-in-kernel polarity: vLLM's quantized GEMM paths
  (`vllm/model_executor/layers/quantized_linear`, the marlin/machete
  kernel family) unpack quant weights INSIDE the kernel and never
  materialize a dequantized f32 weight tensor on device — the eltwise
  dequant chain this row replaces is a tt-metal-shape artifact of the
  ttnn op-by-op port, not vLLM behavior. vLLM defines the CONTRACT
  (bit-exact W4a grouped dequant feeding the dot), the int8dot kernel
  defines the HOUSE STYLE it is delivered in.
- Our own precedent: `kKeepQuantInt8DotKernelSrc` +
  `keepquant_kernel_code.h` (the shared host/device soft-float header,
  little-endian explicit loads, `-ffast-math` provenance-hiding via
  `volatile` where a division defines a value, accumulation order ported
  verbatim). The fused decode kernel reuses the same header include path
  (`KeepQuantKernelIncludeDir()`) and the same doctrine.
- The ae27ec78a doctrine: full-grid uniform slices with in-kernel row
  derivation from the core coordinate; ONE common runtime-args vector;
  self-cycled CBs (reserve -> use -> push -> pop, no consumer core).

## Design

### Stage inventory: what the ~85 programs actually do

Read from `tenstorrent_keepquant.cpp:629-757` (Q6_K), grouped. Counts
are per chunk of B=8 rows; every ttnn op below is one or more small
full-grid programs at the packed record floor.

1. **Word-range extraction and widening (~12 programs).** `ttnn::slice`
   of word 52 (the f16 `d` half), `bitwise_and 0xFFFF`,
   `f16_bits_to_f32`, reshape; three `KeepQuantByteRange` calls
   (ql bytes 0..128, qh 128..192, sc 192..208 — each an extract + layout
   round-trip); `signed_byte_f32(sc_bytes)`; the sc sign-bit chain
   (`bitwise_right_shift 7`, `and 1`, `typecast f32`).
2. **Per-(h,r) nibble unpack loop (8 runs × ~12 ≈ 96 program slots —
   the bulk).** For each half h in {0,1} and run r in {0..3}: two
   `slice`s (ql at qoff = 64h + 32(r%2), qh at 32h..32h+32), the 4-bit
   low/high nibble op (`and 0xF` or `shift 4`), the 2-bit high op
   (`shift 2r`, `and 3`), the OR into bit 4, `typecast f32` + `subtract
   32.0` (the q6 value in [-32,31]), the scale slice s2, the two-step
   multiply `(d*sc)*q` (host left-to-right association, kept), and the
   integer compare `lt(nib6, 32)` + typecast for the sign mask.
3. **Concat/reshape plumbing (~8 programs).** Two concats per half,
   the two half concats, the `{B,16,16}` reshapes of prod and qsign.
4. **Zero-product sign algebra (~10 programs).** The three-sign XOR
   predicate (d's f16 bit 15, sc's bit 7, sign(q)) written as the exact
   f32 polynomial `a+b+c-2(ab+ac+bc)+4abc`, and the `repair` +
   `zero_mask_f32` + `neg0_scalar` chain that re-imposes -0 where the
   XOR lands on a zero product.

This sign-algebra group exists ONLY because ttnn's broadcast multiply
cannot carry the IEEE sign of a zero product through the op chain. A
scalar RISC soft-float multiply produces the IEEE sign bit naturally —
group 4 (and with it group 3's qsign plumbing) DELETES rather than ports.

### The fused kernel: per-core contract

ONE `CreateKernelFromString` full-grid data-movement kernel
(RISCV_0, DM_DEDICATED_NOC, O2), compiled against
`keepquant_kernel_code.h`, dispatched on `enc_sel` exactly as the
int8dot kernel dispatches (the CARG_ENC select). Per core c = y*grid_x + x:

- Derive its uniform row slice in-kernel: `row0 = c * rows_per_core`,
  `rowc = clamp(rows - row0, 0, rows_per_core)` — zero extra runtime
  args, the a1661114b doctrine. Idle tails (rowc == 0) exit without
  touching DRAM; partial tails read/write only rowc rows.
- Read its rows' packed Q6_K word pages from the words shadow through an
  interleaved `TensorAccessor` (page = the 16B-aligned per-row word
  bytes; Q6_K = nb × 210 B, `d` at word byte 208, `ql/qh/sc` at
  0..128/128..192/192..208 exactly as the C++ comment at
  `tenstorrent_keepquant.cpp:630-637` documents).
- Per 256-column block, unpack in registers: the 16 sub-block scales as
  signed i8, per (h, r) the low nibble (ql byte, low or high half by
  r<2), the 2 high bits `(qh[32h+l] >> 2r) & 3` OR'd into bit 4,
  subtract 32, multiply `d * sc` FIRST then `* q` — the host's
  left-to-right association, written in that order under
  `-ffp-contract=off` semantics the shared header already pins.
  No divisions anywhere on this path, so no `volatile` provenance
  hiding is needed (record: revisit only on evidence).
- Write the f32 activation page (`nb*256` floats per row) through the
  output accessor.

Everything outside the unpack — the matmul chain, the activation
quantize, the dot — stays exactly where it is.

### CB / RTA plan (packed-floor discipline)

- CBs: 4 uniform CBs, self-cycled, mirroring the int8dot geometry —
  c_0 one weight row's packed words (aligned row bytes), c_1 the widened
  f32 out page (nb*256*4 B), c_2 spare staging (only if the row does not
  fit L1 in one page; prefer direct DRAM->regs->DRAM without a staging
  CB when the row page fits), c_3 reserved. All CBs identical across
  cores: the bisect proved identical pages collapse into the packed
  relay, so the record is 1,024 B/launch.
- ONE common-args vector (the CARG_* table pattern): the words bank
  base, the out bank base, rows, nb, enc_sel, grid_x, rows_per_core.
  NO per-core SetRuntimeArgs anywhere — that stream is the already-spent
  228.9 MB lever and it must not come back.
- Warm-first: the workload caches by the same key shape as int8dot
  (enc × M/K/N × grid); an uncached shape refuses under capture by name
  (`tt_capture_active()` check, the mesh_workload.cpp:153 contract).

### Capture-safety by construction

The kernel is capture-safe because nothing about it is capture-conditional:
common args only, uniform CBs, no capture-scope uploads, binaries warmed
eagerly on first call, the same chain in both passes. The trace record is
the proven 1,024 B floor × 1 program per chunk-chain launch instead of
×~93 — the region record falls from 2,965,504 B to ~8-16 KB (8 chunks +
the ~8 matmul programs at their own floor), under the 64 KiB gate.

### Bit-exactness strategy

The fused kernel must produce bit-identical f32 to the current chain.
The chain's own construction pins what "identical" means:

- The unpack is exact integer work — identical bits by construction,
  checked per enc-select against golden dumps of the current chain's
  output (the existing capture-x2 byte-identity harness is the
  vehicle).
- The arithmetic is two IEEE f32 multiplies in a fixed order:
  `(d*sc)*q`. The soft-float RISC build and the ttnn chain both execute
  exactly that order (`-ffp-contract=off`, no reassociation in a scalar
  sequence). The zero-sign repair the chain performs by algebra, the
  kernel gets from IEEE multiply semantics for free — both must produce
  -0 on the same elements, which the goldens check.
- Partial and idle tails: the last core's rowc < rows_per_core rows and
  a fully-idle core both get dedicated golden cases (a rows % grid
  misalignment on purpose).
- Per enc-select: Q6_K first; any other encoding the fused dispatch
  claims runs the same golden-vs-chain comparison before its default
  flips. An encoding not yet ported keeps the chain via the named
  decline.

## Risks

- **SFPU/compute budget per core.** The int8dot kernel already does
  per-block soft-float work at full grid on these shapes, so the unpack
  (strictly less arithmetic per row than the dot) is bounded by that
  precedent. Risk shape: a per-row cost blowup that makes the fused arm
  slower EAGER than the chain, trading a trace win for a TPOT loss.
  Mitigation: the money legs measure TPOT both arms before the default
  flips; the env gate keeps the chain one flag away.
- **Register pressure in the unpack loop.** 16 scales + 4 nibble
  streams held per block is modest, but the RISC build's soft-float
  calls can spill. The int8dot kernel's register discipline (block-at-
  -a-time, no cross-block state) is the pattern; a spill shows up as an
  eager-time regression the money legs catch.
- **Capture-replay interaction.** The known fatal classes
  (mid-capture writes/reads, per-core RTA pages, capture-scope uploads,
  first-time binaries) are each structurally absent (above), and the
  red-first tests include a capture-x2 byte-identity case on the fused
  arm so a regression cannot land silently.
- **A parallel path beside the chain.** The fused kernel routes through
  the same `DecodeKeepQuantWordsF32` entry and the same workload cache
  shape; the chain stays as the DECLINE arm, not as a twin serving path.
  The env gate selects between two arms of one seam, documented in both
  directions.

## Tests

1. **The red gate goes green.**
   `tests/vt/test_tenstorrent_backend.cpp:11527` (keepquant region ≤
   64 KiB, measured 2,965,504 B on HEAD, red by design since `7a0f1ca3f`)
   passes with the fused arm default-on. Red-before is already
   recorded; this is the row's arbiter.
2. **Per-enc-select bit-exact doctests vs the chain.** For each served
   encoding (Q6_K first): fused output bits == current-chain output
   bits on the golden shapes, including a partial-last-core case and an
   idle-core case. Run eager AND capture-x2 (capture, replay, byte
   identity) per the existing harness.
3. **The named decline.** A shape or encoding the fused arm does not
   serve declines with a message naming the missing part and the chain
   serves it — the refused-arm rule, red-first (fused-forced +
   unsupported enc must not silently pass).
4. **Capture-safety regression guard.** The fused arm under capture
   issues no per-core RTA writes and no capture-scope uploads (the
   VT_TT_TRACE_DEBUG census and [TT-UP] routes are the instruments);
   the region census for a fused chunk-chain is printed and bounded.
5. **The money legs.** 27B c1 and c2, `BENCH_EXIT`, TPOT table, and the
   whole-graph trace demand re-measured (the 2,925,109,248 B bound must
   fall; the fit wall is this row's downstream judge). One fresh
   process per leg, the recorded anchor recipe, both
   VT_TT_KEEPQUANT_INT8DOT settings.

## Gates

1. Every test above red-before (or, for the goldens, mutation-proven:
   flip the multiply association or the -32 bias in a scratch copy and
   the golden must fail), green after.
2. The device suite at the standing bar with only the pre-recorded
   2261 RAC flake failing.
3. The red gate at 11527 green with the fused default on.
4. 27B c1/c2 legs: the trace-fit boundary moves; TPOT recorded on both
   arms; served == eager token-exact.
5. Standard gates: `agent-preflight.sh --staged`, commit-style,
   commit-trailers, `check-agent-record` on the changed files.

## Evidence plan

Dated `docs/bench-evidence/tt-decode-fusion-<date>.md`: the per-launch
record census before/after (the 2,965,504 B -> floor number), the golden
bit-identity table per enc-select and tail class, the eager-time A/B
(fused vs chain on the grouped shapes), and the 27B leg table (trace
demand, BENCH_EXIT, TPOT). The bench-evidence file is the row's single
measurement record; no number lives in two files.

## Stop conditions

- The fused kernel cannot be made bit-exact against the chain on Q6_K
  after the accumulation-order and sign-semantics avenues are exhausted
  (each tried with a recorded golden diff) -> stop, the chain stays the
  serving arm, the fit wall keeps its bound, the row records the
  arithmetic divergence with the diff bits.
- The fused arm is bit-exact but eager-time slower than the chain by
  more than the trace win justifies (the TPOT table shows a net loss on
  the anchor arm) -> stop, record the numbers; the chain serves, the
  fit wall stays open, and the escalation beside tt-metal#57970 carries
  the measured trade.
- The record does not fall under the 64 KiB gate with ONE program per
  chunk-chain launch -> the packed-floor model is wrong somewhere; stop
  and re-attribute before any further lever (the bisect harness from
  `tt-trace-config-page-repro-20260929.md` is the instrument).

## Issue routing

This row RIDES `ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ`; no new local
issue. Reasoning: the issue's 2026-09-29 entries already name this exact
lever as the fit site's "NEXT LEVER (one step)" and the 64 KiB gate as
its arbiter — the fusion IS the next increment of the same unit of work
(the 27B decode-trace DRAM fit), not a separately-owned behavior. The
row ID `TT-DECODE-FUSION` identifies the spec and branch; the issue's
Row: field stays `BACKEND-TENSTORRENT-QWEN35` because the fit site it
owns is what closes. The resolution entry for the fused kernel lands in
that same issue file when the row lands, per the issue's dated-entry
pattern.

## Owed

- The matmul chain (~8 programs per chunk, the other ~8 × floor term
  per chunk): owed to this row's own follow-up unless the fuse is
  trivial; the fit wall's remaining demand after the decode fuse is
  re-measured first so the follow-up is sized on numbers.
- Other encodings' decode arms (Q4_K/Q5_K/IQ arms) through the fused
  dispatch: each owed its golden before its default flips; the named
  decline covers them until then.
- The 27B whole-graph fit after the decode fuse: the demand will still
  include the matmul chain and the rest of the graph; c1/c2 may still
  not serve — the leg numbers decide whether the follow-up (matmul
  fuse) or the escalation is next.
- The pre-existing owed items (the 2261 RAC flake bisect, the INT8DOT
  re-measure, the sampler-bracket re-baseline) carry forward unchanged.

## Git integration

One pull request: spec, fused kernel, tests, evidence, and records land
together on `row/tt-decode-fusion`, continuing the developer's recorded
2026-09-28 `## Git integration` choice (one PR per row; not re-asked per
policy).

## Now

2026-09-29 (DRAFT, spec committed on `row/tt-decode-fusion-spec`, stacked
on the region-capture branch HEAD `f92eae33f`): design only — the stage
inventory of the ~85, the single-program kernel contract, the CB/RTA
plan, the bit-exactness strategy, and the tests above. Nothing
implemented. NEXT: the fresh implementer ports the Q6_K fused kernel
smallest-test-first against the already-red gate at
`tests/vt/test_tenstorrent_backend.cpp:11527`. The row's arbiter, the
red gate, is planted; the money legs stay blocked behind the fusion per
the fit site's record in ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ.
