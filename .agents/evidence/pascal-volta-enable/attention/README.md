# Pascal register decode attention

Issue: `ISSUE-LOCAL-01M63YBNVGEYMHETJ67HE5XS2K`.
Spec-first commits: `06e5ddce0`, `19b17a3ce`. Date: 29 October 2026.

The P620 TinyLlama trace supplied by the operator records 2772 portable
attention calls taking 2647 ms. The d64 path falls through to a 256-thread
block reduction with eight reduction barriers plus state barriers per key.
The existing warp-register kernel keeps Q and online softmax in registers,
strides keys over eight warps and merges once after the scan. Add its d64
vector loads and enable d64/d128 by default on Pascal only. Modern CUDA keeps
its previous policy. Volta keeps its existing implementation and tuning is
TODO until the GPU arrives. No Tensor Core capability is assumed.

Pinned llama.cpp `62318cd57dc63900ecb153272c0b10f78550855d`
`ggml/src/ggml-cuda/fattn-vec.cuh:86-118,152-170,274,439-496` supplies the
resident-Q, warp reduction and final combination reference. The local kernel
adapts this decomposition to paged NHD caches through the existing registered
`vt::PagedAttention` seam. The primary pinned vLLM cannot run Pascal; no
primary-oracle throughput claim is made.

## Focused evidence

`check_cuda_legacy_attention_perf.py` runs the same binary twice under one
file mutex, with `VT_ATTN_DECODE_OPT=0` and `=1`. Both arms first verify the
same independently composed, two-pass host attention reference. The initial
unmodified implementation fails the required fourfold improvement with
1.064x (`perf-red.log`). This is the intended red performance regression.
The final implementation passes all 16 d64 full-attention shapes at context
128/256 across F32/BF16 inputs, F32/BF16 outputs and GQA ratios 1/4:
13.496x to 15.042x faster (`perf-final.log`). These are warmed CUDA-event
operator measurements, not end-to-end model throughput.

The final numerical suite passes 197 cases (`final-run.log`): dimensions
64/128; contexts 1,7,16,17,128,256; F32/BF16 input/output cross product;
GQA 1/4; reversed physical block order; padded block stride; non-unit
block-table column stride; causal/full and local-window attention; softcap;
and four valid odd-padded cache-row cases. The operator's existing synthetic
public CPU/CUDA completion regression also passes all nine dimension/prompt
combinations, with identical greedy IDs (`public-final-run.log`).

Vector eligibility checks the base pointers and each cache stride. The
four odd-stride regressions fail with `misaligned address` before this guard
(`alignment-red-run.log`), then pass using the existing portable fallback.
No valid tensor contract is narrowed. FP8 keeps its existing scaled path.

Deleting the new production d64 dispatch in an independent scratch source
compiles and links successfully. The same A/B performance gate then fails
(`perf-mutation.log`), proving that registered production dispatch is reached.
The production tree is unchanged during mutation. No test calls an internal
kernel or constructs an implementation backend by hand.

## Build and limits

`/tmp/pascal-attention-driver.py` extracts the existing Release TU commands
from `/tmp/lumen-build-legacy/compile_commands.json`, substitutes only the
source/include root and scratch output, and uses the existing NVCC 12.4
sm60/sm61/sm70 recipe. Compile logs report success for the final kernel and
mutation. No full build cache is changed. A 4.7 MB thin archive references the
existing Tiny repair archive and removes its old attention object. Explicit
new attention and fixed untied-head objects supply the final model binary;
this avoids duplicate attention registrars and an additional full archive.
All GPU processes use `flock /home/user/gpu.lock` on the approved nonfleet P620.

The final object is `/tmp/pascal-attention/attention-final.o`; the public model
binary is `/tmp/pascal-attention/actual-frozen`. The scratch directory retains
commands, raw logs, resource usage and SASS. Initial d64 specialization uses
37-39 registers with no stack/local memory. A remaining 1360-instruction SASS
variant includes runtime 64-bit page arithmetic; matching the oracle's tile
and address decomposition is the next traceable hypothesis, not a ceiling.
No optional page-address change was made in this repair.

The startup preflight was bounded at 45 seconds and exited 124; it is not a
passing full gate. Offline record validation passes with ANCHOR-ROT=0 and
`git diff --check` passes. Independent review, the operator's full model token
and throughput axes, P100 hardware and Volta hardware remain separate gates.
Issues remain open until landing; no merge or remote write occurred.
