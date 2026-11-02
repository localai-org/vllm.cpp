# Pascal register decode attention

Issue: `ISSUE-LOCAL-01M63YBNVGEYMHETJ67HE5XS2K`.
Row: `BACKEND-CUDA-SM061`. Base: `962e005db`.

## Scope and design

The real TinyLlama P620 trace records 2772 portable decode launches taking
2647 ms. At head dimension 64, the existing 256-thread kernel reduces each
key through shared memory and eight block barriers, with additional state
barriers. Reuse the shared register/warp-strided online-softmax kernel for
Pascal CUDA head dimensions 64 and 128. Add the missing two-element-per-lane
loads in a separate legacy hardware header. Keep Volta on its existing policy until delivered hardware is validated (TODO).
Preserve modern CUDA dispatch,
existing environment overrides, paged strides, GQA mapping, causal and local
masking, softcap, and BF16/F32 input/output handling. FP8 remains on its existing
scaled block path. No ABI, configuration, model or quantization changes.

## Upstream anchors

Pinned llama.cpp `62318cd57dc63900ecb153272c0b10f78550855d`,
`ggml/src/ggml-cuda/fattn-vec.cuh:4-10,86-118,152-170,274,439-496` uses
warp reductions, resident Q, register online state, and a final cross-warp
combination. The local existing decode-opt implementation supplies these same
seams. The primary vLLM pin cannot execute Pascal; the registered pinned
llama.cpp remains the executable secondary oracle. No tensor cores assumed.

## Tests and evidence

First capture a public registered PagedAttention latency regression against
its rollback kernel on actual small contexts (16, 128, 256) and dimension64.
Require a substantial same-process warmed speedup; the existing barrier kernel
must fail. Compare deterministic BF16/F32 outputs against the CPU reference for
head dimensions64/128, GQA, paged noncontiguous blocks, partial blocks, causal
and local masking. Existing synthetic public model completion verifies loader,
registry, dense forward and sampling. Mutate the production legacy dispatch and
require the speedup regression to fail. Compile the changed production TU using
the existing Release recipe; build at most four jobs. GPU runs hold
`/home/user/gpu.lock` on the explicitly approved local nonfleet P620.

The operator reruns all identical TinyLlama checkpoint token gates before any
performance acceptance, then records same-binary idle warmed A/B throughput.
SmolLM cross-runtime numerical divergence is separately owned and must not
worsen. P100 and Volta remain hardware-unverified; Volta optimization is TODO.

## Risks and stop conditions

Warp-reordered accumulation can change near-tied logits. Correctness precedes
performance. Stop on a new token mismatch or masked/output dtype regression.
An empty attention interval retains the prior behavior. Unavailable hardware
or oracle execution remains explicitly pending, never green.

## Now

ACTIVE. Root cause identified; spec precedes implementation. Independent review
and operator full-model gates remain required. No main merge or remote write.

## Explicit d128 override repair

Issue: ISSUE-LOCAL-01M63ZPB4JB2634AYK6XN4S6EH.
Independent review found that the Pascal ternary bypasses DecodeD128Enabled.
An explicit VT_ATTN_DECODE_D128=0 therefore retains the optimized kernel.
Keep Pascal's unset default enabled. Let the existing helper resolve explicit values before hardware alignment decides eligibility.
Keep other architectures' unset default disabled. The master VT_ATTN_DECODE_OPT=0 still disables all optimized decode.

First add a production-path override regression using the registered attention benchmark.
Require unset and explicit 1 to select the fast path, and explicit 0 and master 0 to select the portable path.
The same executable checks numerical correctness in each mode.
A dispatch deletion and an override bypass must fail the regression.
Compile the repair for sm60, sm61, and sm70, and rerun the existing 197 numerical cases.
Performance acceptance remains pending fresh review and the operator gate.

### Override repair evidence

On 29 October 2026, the regression failed against the reviewed head with a disabled-to-enabled ratio of 1.000984.
The repair passes all 197 numerical cases in each of four isolated environment configurations.
Unset and explicit 1 select the optimized d128 path. Explicit 0 and master opt 0 select the portable path.
All 16 dispatch comparisons exceed 4x. The smallest observed ratio is 12.519x.
The changed production translation unit compiles for sm60, sm61, and sm70.
An override-bypass mutation fails the regression at 0.996622x.
A production d128 dispatch deletion fails at 0.920076x.
The restored unchanged production object passes the four-mode regression again.
Scratch mutations never change the production worktree.
The offline canonical record check and whitespace check pass.
Evidence: `/tmp/attention-override-evidence`, with driver `/tmp/attention-override-build.py`.
Independent review and the operator gate remain required before performance acceptance.

## Prepared real-model A/B evidence

Issue: `ISSUE-LOCAL-01M63YBNVGEYMHETJ67HE5XS2K`.
[Measured TinyLlama A/B](../evidence/pascal-volta-enable/attention/model-ab/README.md)
records the executed pre-quant-correction binary at attention source `22d433baf`.
The same binary passes all32 cold/warm completions,1280 exact generated IDs,
with warm throughput ratios1.0276/1.0834/1.1590/1.2834 across the four axes.
Peak memory is equal within each pair. The whole idle-start campaign holds
one GPU file mutex. Later d128 repair `c93e92b94` leaves d64 unchanged.
These prior results do not validate later Q5/Q6/D4 quant candidates. Current
Smol parity and the later Q6-prefill Tiny regression remain FAIL. Lifecycle
and final operator gates remain unchanged. This evidence rides the capability
integration; it is not a benchmark publication or a new standalone feature.
