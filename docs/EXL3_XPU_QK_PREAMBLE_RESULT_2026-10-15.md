# B70 Wave 4: Q/K preamble, 15 October 2026

## Decision and scope

Keep native FP32 multiply/add nodes in the existing fused subgroup Q/K norm,
RoPE and gate producer. The old explicit RN device-library helper path remains
available with `VT_XPU_ATTN_PREAMBLE_NATIVE_FP32=0`; the qualified default is 1.
The FP32 reduction tree, inverse, epsilon, Gemma `1 + weight`, FP16 narrowing,
RoPE operations and output layout are unchanged. This is separate from the
explicitly approved Wave 2 GDN reassociation; no additional tolerance was relaxed.

The source base is `bef524f0cb9df20089da2ed9cebce9773deecba4` plus the
uncommitted campaign changes. The qualified binary SHA256 is
`7897028668df1c5e3341812b579183dea0fce09f9aca4eb770fede57aba20313`.
The external `wave4/qk-preamble/qualified-identity.json` records per-file hashes
and the adjacent source snapshot. Qualification ran before committing; the
checkpoint commit preserves the tested implementation bytes.

Touched product surfaces for this wave are `src/vt/xpu/xpu_attention.cpp`,
`include/vt/xpu_attention_preamble_policy.h`, and the existing dense model graph
policy in `qwen3_5.cpp`/`qwen3_5_internal.h`. Dispatch and graph identity use a
shared default. The graph key includes the arithmetic control so captured target
and verifier graphs retire when it changes. The paged MTP draft forward currently
runs its layer directly, without an additional captured graph owner in that path.
The focused Q/K and graph-policy tests changed with those surfaces.

## Selection from the new profile

Separate post-Wave-3 PTI captures included replayed graph kernels: 384,576 GPU
kernel records for MTP0 and 138,827 for MTP3, with zero collector errors. Client
phase boundaries remain approximate; instrumented wall times are not scores.
The MTP0 decode window attributes about 6,565 ms to GEMV, 623 ms to attention,
513 ms to normalization and 435 ms to Hadamard. MTP3 attributes about 2,397 ms
to GEMM, 252 ms to normalization and 248 ms to attention. These are diagnostic
kernel-duration sums, not additive estimates of uninstrumented latency.

Q/K preamble alone accounted for 4,080 calls / 248 ms in the target-only decode
window and 1,539 calls / 98 ms with MTP3. Its norm still used explicit RN
multiply/add library helpers, although an earlier Gemma optimization had already
removed equivalent calls from a different producer. The pinned Python trace
showed a short fused Q/K producer, but different fusion boundaries prevent
claiming direct whole-region equivalence from those kernel names alone.

A first, distinct Gemma experiment folded the first reduction-tree step into
registers while preserving its arithmetic order. All four actual row-1/row-4,
plain/residual replays were byte-exact, but complete-operator time increased
26–66%. That candidate was removed before any serving score. The original norm
source and tests were restored byte-for-byte. Its patch and failed performance
result remain in `wave4/rejected-fold2`; it is not a shipped alternative profile.

## Exact operands and complete-operator comparison

A separate eager capture retained actual P4096/MTP3 row-1 and row-4 Q/K/gate
operands, their physical strides, weights, coefficients and old outputs. Replays
restore output storage each trial. Candidate and control Q, K and gate outputs
match every byte, as do immutable inputs. Warmups and balanced complete calls
include the fused producer; capture/download/restoration are outside timing.

| Rows | Control complete ms | Candidate complete ms | Control / candidate kernel us |
|---|---:|---:|---:|
| 1 | 0.137589 | 0.028090 | 128.281 / 18.907 |
| 4 | 0.136795 | 0.030075 | 127.552 / 20.417 |

These microbenchmarks do not predict proportional whole-model gains. Level Zero
properties report 1,024 private bytes and zero spill bytes for both instantiations;
GRF occupancy was not exposed. No register-pressure improvement is claimed.
The new path has no new tensor, workspace, model-weight cache or host fence.
A fresh full-context/driver-memory capacity qualification was not performed.

## Focused qualification

The final default build and graph-policy checks pass 3 cases / 109 assertions.
The default GPU run passes 4 cases / 385 assertions: signed zeros and exact
control outputs at rows 1/2/3/4/8/128, Gemma/plain weights, strided merged input,
unchanged inputs, and graph replay with changed activations and coefficients.
The actual layer-43 P128/D1 KV preimage has zero differing FP16 elements for
standalone norm, standalone RoPE and the fused path. Its separate bounded
complete-preamble test passes 1 case / 817 assertions.

Historical failed setup attempts remain recorded: the initial new test needed a
graph include; two fixture mounts missed required tensors/relative symlinks;
the first final timing invocation omitted an output directory, and combining
both real tests then collided with intentionally non-overwriting dump paths.
Separate processes and fresh output directories resolved the final setup errors.
No numerical check was weakened to make these runs pass.

Three counterbalanced fresh-process pairs per mode retain the original
C1/P4096/O256 greedy workload and all previous selected waves. Profiler/trace
controls are off. All 48 native replies match the frozen native token
representations, text and usage; direct pairs also match all speculative draft
and accepted-token counters, including primer and final values.

| Mode | Control prefill s | Candidate prefill s | Control / candidate decode tok/s | Control / candidate total s |
|---|---:|---:|---:|---:|
| MTP0 | 1.835481 | 1.779241 | 27.2625 / 27.9083 | 11.189641 / 10.917154 |
| MTP3 | 1.890490 | 1.830180 | 63.2466 / 63.8773 | 5.922883 / 5.819951 |

Median paired prefill changes are -2.98%/-2.91% for MTP0/MTP3. Decode throughput
changes are +2.36% (range +1.14 to +2.41%) and +0.89% (range +0.72 to +1.00%).
Total duration falls 2.44%/1.50%. Differences are small but have the same sign
in all three independent blocks. The predeclared 3% unaffected-duration limit
is met; it is not a definition of parity. Raw block dispersion is retained. Cgroup receipts show zero CPU throttling in
all 12 paired workers. Recorded requested GPU frequency has median 2800 MHz;
actual-frequency samples include idle periods and are not occupancy measures.
The inherited fixed hwmon path did not yield temperature/power samples in this
run; no retrospective thermal normalization or power-equivalence claim is made.

## Remaining work and limits

## Final standard-build confirmation

One fresh worker per engine/mode repeated the four-request protocol, using the
median of the last three requests. This confirms the default build; the paired
three-block experiment above establishes the incremental candidate effect.

| Mode | Native / Python prefill s | Extra native prefill time | Native / Python decode tok/s | Lower native throughput | Native / Python total s |
|---|---:|---:|---:|---:|---:|
| MTP0 | 1.752 / 1.475 | +18.81% | 28.14 / 29.62 | -4.99% | 10.821 / 10.089 |
| MTP3 | 1.823 / 1.479 | +23.24% | 64.26 / 68.14 | -5.69% | 5.794 / 5.226 |

All four decoded texts per mode, prompt token IDs and integer usage counts agree.
Python additionally emits nullable usage-detail fields. The initial campaign
comparator incorrectly required identical usage dictionaries; its correction
compares all three integer counts and verifies that additional reference fields
are null. Same-engine exact token/text/usage checks remain unchanged.
Cross-engine raw output-ID, full-logit and accepted-state parity is not established.
Native/Python proposed draft counts remain 972/984 for the four MTP3 requests;
this comparison includes that actual acceptance behavior rather than fixing it.

The bounded default-build vision regression passes seven existing HTTP requests:
PNG/JPEG fixtures, SSE, and ordered two-image ordinary/SSE requests. The server
reports 300 actual proposed drafts, valid integer usage and drained gauges.
No broad vision matrix or vision-performance parity campaign was repeated.

The final remaining gaps are measured, not strict or near parity. Relative to
the earlier Wave 3 snapshot, changes in both native and reference measurements
also contribute to the observed gap; only paired controls attribute the Q/K gain.
This closes the selected Q/K wave. A subsequently authorized section9.3 panel
comparison is recorded in `EXL3_XPU_W8A8_PANEL_RESULT_2026-10-15.md`.

This wave selected the measured Q/K producer; the alternative Gemma mapping was
rejected. Large-M panel policy was inspected, not changed: the native bounded
1024-column panels and the reference full-source-group oneDNN calls differ.
A future bounded 2048-column comparison could test that prefill region, with
whole-operation timing and fresh memory accounting. It is not an established
win or a requirement to exhaust the plan's conditional backlog now.

Wave 1's preparation-cap qualification, Wave 2's user-approved rounding change
and Wave 3's bounded attention gates remain distinct. The deferred long natural
generation-quality test for subgroup GDN is still open. This campaign does not
qualify all prompts, concurrency, maximum context, sampling RNG or cross-engine
raw output IDs/logits/state equivalence. Production remains stopped.

Commands, failed attempts, real operands, block samples and separate traces are
under `pro-optimization-2026-10-15/wave4` in the local recovery artifact root.
