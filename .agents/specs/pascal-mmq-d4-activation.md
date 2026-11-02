# Pascal MMQ D4 activation contract

Issue: `ISSUE-LOCAL-01M6405SHVT2X5PWAXM1FJHXXG`.
Row: `BACKEND-CUDA-SM061`. Base: `561ee7b38`.

## Scope and source

The pinned CUDA oracle uses F32 D4 activation scales for standard Q5_0/Q8_0
prefill. On identical actual SmolLM2 input and packed weights, D4 matches Q8 V
with RMSE4.46e-9 and Q5 up/gate with4.13e-8/5.31e-8. F16-scale formulas instead
produce8.4e-6 and8.29e-5/1.04e-4. One actual activation integer changes when the
reciprocal arithmetic changes. Diagnosis: `/tmp/lumen-smol-diagnosis.md`,
section `Exact remaining prefill quantization mismatch and correction boundary`.

Pinned llama.cpp `62318cd57dc63900ecb153272c0b10f78550855d`:
`ggml/src/ggml-cuda/mmq.cuh:60-76` selects D4 for Q5_0/Q8_0;
`:564-581` reconstructs signed Q5 for the Q8-style integer dot.
`quantize.cu:457-550,574-602` computes per32 maximum, `inv=127/amax`,
`q=roundf(x*inv)`, then `d=1/inv`, storing F32 d without original sums.
`ggml-cuda.cu:1955-2010` checks normal MUL_MAT and F32 carriers, then MMVQ
before MMQ. `mmvq.cuh:3`, `mmvq.cu:318-380` retain the Pascal batch8 cutoff.
`mmq.cu:355-445` checks supported format,48KiB shared memory and compiled
DP4A support. Pascal has no FP16 MMA hardware. M>8 alone is insufficient.

## Design

Add a separate Pascal hardware header for D4 scratch, quantization and signed
integer contraction. Enter only from the existing registered MatmulBTQuant
standard dense path. Retain original packed weights on the GPU. No dequantized
weight materialization, host drain, grouped path or parallel model forward.

Eligibility requires Pascal sm61/sm62, native DP4A, at least48KiB shared memory,
M>8, K divisible by32, contiguous activation and canonical NK packed weights.
Reject this arm for CPU-repacked, Q8-aligned or elementwise-transposed weights.
Only Q5_0/Q8_0 qualify. The API's F32/BF16 carriers adapt the upstream F32
contract by widening input values before quantization and rounding output only
at the final store. No model dtype policy changes. M1/M8, sm60, Volta, modern
CUDA, grouped/MoE, other formats and arbitrary repacks keep existing paths.
The published measured shapes are M16/M128,K576,N192 or1536.

## Tests and evidence

First write and run failing registered MatmulBTQuant tests. Independently
construct D4 quantization and decode unchanged packed weights. Cover Q5/Q8,
F32/BF16 input/output, M9/M16/M128,K32/K576 and ragged N. Include zero,tiny and
signed values, F16-scale-sensitive and reciprocal-boundary vectors. M1/M8
controls retain each existing activation contract. Load exact actual oracle
input files and packed checkpoint tensors for the measured projections;
compare against independent formulas and actual captured oracle outputs.
External artifacts remain pinned; fixtures need not copy model weights into Git.

Mutation narrows D4 scale to F16, then separately reuses the old reciprocal
route. Both focused controls must fail. Delete production dispatch and require
registered tests to fail. Restore sources and repeat focused green. Compile
sm60/sm61/sm70 with the existing Release recipe and run the bounded full quant
suite. All GPU runs hold `/home/user/gpu.lock` on the approved nonfleet P620;
use at most four build jobs and avoid additional full archive copies.

The operator reruns actual Smol public completion with the correct Tiny head
fix linked. This correction may not establish whole-model token parity;
remaining differences remain open, with no correctness waiver.

## Risks and stop conditions

Round-half-away-from-zero and reciprocal order are observable contracts.
Keep amax-zero behavior finite. A wrong eligibility predicate can alter decode
or other hardware behavior. Stop with NEEDS_CONTEXT for missing source or
actual fixture contracts; do not infer a larger oracle arm. Stop for a new
quant-residency loss, unreviewed policy change or an unfixed focused failure.
Independent immutable review and the operator's row gates remain required.

## Now

ACTIVE. Canonical issue and spec precede tests and implementation. One PR is
recorded repository integration policy. No main merge or remote write.
