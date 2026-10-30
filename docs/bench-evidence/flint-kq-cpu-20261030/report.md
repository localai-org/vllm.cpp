# FLINT-KQ-CPU-DISPATCH — the CPU paged arm serves the keep-quant tree

Row: `FLINT-KQ-CPU-DISPATCH`
Spec: `.agents/specs/flint-kq-cpu-dispatch.md`
Issue: `ISSUE-LOCAL-01M64QZJ4WDDX439EJDVY3M4E4`
Date: 2026-10-30 (benches 2026-10-29/30 night). Host: the shared aarch64
128-core dev box, bench zone `taskset -c 32-47` (16t legs) / `-c 32-63` (32t
legs), governor **schedutil** on all 128 cores — the SAME governor state the
floor was measured under (flint-cpu-parity-20261029), so the comparison is
governor-clean. Idle gate (1-min load < 16, no other process above 50% CPU)
re-checked before every rep by the leg driver; box claimed in
`~/.config/box-claims.txt`.

Artifact: `/mnt/models/flint-GGUF/flint-Q4_K_M.gguf`, sha256
`20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e` — the same
byte-identical file the floor and the goldens were built against.

## RED (re-verified at HEAD before the change)

`VT_FLINT_KEEP_QUANT=1` through the paged serving path
(`vllm-bench`, full engine stack) fires the named refusal at
`flint.cpp:868-881`: "flint CPU paged arm: the weight tree carries keep-quant
block tensors ... this arm reads bf16 elementwise weights only". Re-firing
captured during this row from the pre-change binary (same message text as the
committed `../flint-cpu-parity-20261029/keepquant-refusal.log`).

## GREEN — the serving gate

`VT_FLINT_KEEP_QUANT=1` loads `LoadFlintKeepQuantFromGguf`; the paged arm's
linear layers and `lm_head` route through the per-dtype dispatch seam
(`FlintLinearMatmul` / the lm_head call site): a block weight
(`vt::IsBlockQuant`) goes to `vt::MatmulBT`, whose hand-off reaches the landed
CPU `kMatmulBTQuant` CIQ GEMM (`src/vt/cpu/cpu_quant_gemm.cpp`, the
llama.cpp-mirrored mul_mat port); a float weight keeps the unchanged
`vt::Matmul` path byte-for-byte. The bf16 arm (env unset) is bit-identical by
construction.

Gate: `test_flint_gguf`, keep-quant paged token gate (new TEST_CASE, production
entry point `ModelRegistry::Load/Forward` with hand-built per-layer paged KV
caches, the same llama.cpp oracle goldens, the kolibri1 contract):

**53/60 greedy positions token-exact, 7 flips, 7 near-tie, 0 hard flips,
worst gap 2.486 nats (band 2.5)** — llama.cpp uses integer block dots, so
token-exactness is not expected; the band governs. PASS.

Existing suites, all green: `test_flint_gguf` bf16 token gate unchanged
(46/53, 0 hard), `test_flint_paged` 208/208 assertions,
`test_flint_paged_engine` 11/11 (three continuous-arrival requests through
the full LLMEngine stack).

## Throughput vs the floor

`vllm-bench` through the full engine stack, single stream, greedy,
ignore-eos, 8-token prompt / 200-token window (the floor's tg200 shape),
3-rep medians, `VT_FLINT_KEEP_QUANT=1`:

| threads | keep-quant paged arm (tok/s, median of 3) | floor: llama.cpp tg200 (tok/s) |
|---:|---:|---:|
| 16 | **1.20** (1.10 / 1.32 / 1.20) | **26.29** |
| 32 | **2.46** (2.45 / 2.50 / 2.46) | **26.38** |

Raw logs: `kq-t16-t200-rep{1,2,3}.log`, `kq-t32-t200-rep{1,2,3}.log`.

## The compute-class decision (measured, not preferred)

The spec's open question was which compute class each dtype gets: the
simplest correct class through the shared seam is the existing CPU CIQ
integer-dot GEMM (dequant of the ACTIVATION once per call + integer block
dots, the llama.cpp mirror) — it landed here first and is what the TT row's
dispatch already reaches. Measured above: 1.20 / 2.46 tok/s.

Context number from the parity row, same harness/zone/shape: the
DEQUANT-AT-LOAD bf16 arm measured 3.25 tok/s @16t
(`../flint-cpu-parity-20261029/report.md`). MEASUREMENT SAYS the CIQ
per-expert path is ~2.7x SLOWER than dequant-at-load bf16 in this harness at
batch 1: the MoE expert loop runs one tiny `kMatmulBTQuant` call per expert
projection (activation re-quantized per call, no grouped launch), and that
per-call overhead dominates decode.

Disposition: the arm serves and is in-band; the throughput gap stays OPEN
(12-22x vs the floor, no ceiling declared). Next traceable levers, in order:
(1) the grouped expert GEMM the kernel file already carries
(`kMatmulBTQuantGrouped` / `MoeGateUpSwiGLUGrouped`, unused by this arm) to
amortize the per-call activation quantization across the top-k experts;
(2) llama.cpp-style Q8_K block-dot gathers (the mistral.rs 63→23 ms lever
recorded in the spec); (3) the MoE gather/scatter fusion the paged report
already names. This row does not do that kernel work.

## Raw logs in this directory

`kq-t16-t200-rep{1,2,3}.log`, `kq-t32-t200-rep{1,2,3}.log`,
`host-state.txt` (governor + date).
