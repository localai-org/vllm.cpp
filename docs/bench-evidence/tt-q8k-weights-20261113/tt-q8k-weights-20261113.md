# TT-KEEPQUANT-Q8K-WEIGHTS — the Q8_K weights arm on the P150 (2026-11-13)

Row: `TT-KEEPQUANT-Q8K-WEIGHTS` · Issue: `ISSUE-LOCAL-01M7ADZAKNZTC45QPEE1WGW2DV`
Spec: `.agents/specs/tt-keepquant-q8k-weights.md` (amended layout section is
authoritative: `block_q8_K` = 292 B / 256 elems → 73 words → **80 padded
words** per the standing 64-B pad rule). Worktree `/tmp/tt-q8k`, branch
`row/TT-KEEPQUANT-Q8K-WEIGHTS`, build `/tmp/build-q8k` (recipe identical to
`/tmp/build-q6k-config.log`: durable-pin `CMAKE_PREFIX_PATH`, topk isystem
flag, Release; build dir OUTSIDE the repo tree).

## Artifact (W3 pin)

`/mnt/models/unsloth-qwen3.8-27B-gguf-v2/Qwen3.8-27B-UD-Q8_K_XL.gguf`
unsloth/Qwen3.8-27B-GGUF @ `ce331efc48d5722b4b8041f72793e530eae18d9b`,
31,457,991,680 B. Measured sha256 (`artifact-sha256.txt`): `af36ecb6...990377`
— matches the spec pin.

## Red-before (the absent arm)

- `red-route-pin.log` — `RouteGgufTensor(kQ8_K, kTENSTORRENT)` returned
  `kExpandBf16`: 1 failed / 17 passed.
- `red-device-tests.log` — staging test THREW at
  `tenstorrent_keepquant.cpp:112` ("EnsureKeepQuantWords: unsupported
  encoding"); composite test THREW at `:907` ("kQ8_K has no keep-quant decode
  on TENSTORRENT; the registered set is ..."): 2 failed.

## Green-after

- `green-route-pin.log` — 1/1 passed (Q8_K → `kKeepQuant`).
- `green-device-tests.log` — 2/2 cases, 337/337 assertions: staging
  byte-exact at 80 words/block incl. the f32 `d` word, zero pad words 73..79,
  and the composite matmul vs the CPU dequant oracle (f64-accumulated
  `BlockToFloat` reference).
- `green-full-loader2.log` — the full loader suite 63/63 cases, 12179/12179
  assertions (includes the re-grounded Q8_K pins: the gather discriminator
  case, the executable-encodings pin, the routing TOTAL table with the CPU
  GEMM term 36→38 and CUDA left at 36, and the Q8_K embedding/matrix case).

## Existing-arms byte-identity

`green-existing-arms2.log` — the keepquant/fused/residency subset: 23/24
cases pass, 3051/3111 assertions. The single failing case is
`kTENSTORRENT kGdnPostConv matches the CPU f32 oracle` (GDN conv, not a
keepquant arm). It fails IDENTICALLY on the untouched `/tmp/build-q6k` tree
(verified 2026-11-13, same envelope failures at
`test_tenstorrent_backend.cpp:5247-5249`) — a pre-existing host/device gap,
not a regression of this change. The Q4_K/Q6_K byte-exact envelopes
("E=1 grouped keep-quant chunked slice-decode is bit-exact vs BlockToFloat",
"dense keep-quant matmul stages PACKED words and builds NO twin (all four
encodings)", the fused Q4_K/Q6_K bit-exact cases) all pass unchanged — the
dtype-selection change does not perturb them.

## W3 device gate (standing recipe)

batch 1 (`--num-prompts 1`) / input 32 / output 200 / seed 0,
`VT_TT_KEEPQUANT_INT8DOT=1`, `--max-num-batched-tokens 64`, `taskset -c
32-63`, per leg: `luwen reset` + 15 s + `/dev/shm` cleanup, idle gate
load<16, whole run inside `flock -x` on `${GPU_LOCK:-$HOME/gpu.lock}`.

KV pool: **NONE FITS — unservable-on-P150.** `--num-blocks 8` was tried
first (the declared floor: 4 blocks = 128 tokens cannot serve the recipe's
input 32 + output 200 = 232 tokens at block-size 32). All three legs died
with the IDENTICAL deterministic OOM: `TT_FATAL: Out of Memory: Not enough
space to allocate 178257920 B DRAM buffer across 8 banks ... bank size is
4272341376 B (allocated: 4264775936 B, free: 7565440 B, largest free block:
2097152 B)` — a ~171 MB shortfall beside the 31.5 GB of weights. Backtrace:
`vt::tenstorrent::MatmulBTQuantInt8DotKernel` inside
`Qwen3_5DenseModel::ForwardDevice` — the working set of the NON-Q8_K layers
(the checkpoint mixes Q8_K with other keep-quant encodings) is what
overflows, not the Q8_K composite transient planes. Per the spec's stop
condition the row records **unservable-on-P150** and stops; no determinism
reps were servable (each leg died before the first token). Evidence:
`w3-gate.log`, `w3-rep{1,2,3}.log`, `w3-rep{1,2,3}.stderr.log`. Note: the
bench process exits 0 despite the TT_FATAL (the device fatal is swallowed
upstream of the harness exit code) — gate on the `TT_FATAL` line, not the
exit code. The W1/W2 capability (loader + staging + composite) is green and
landed; W3's e2e gate is a named external blocker: the artifact needs a
bigger board or a smaller mixed-quant variant.

## Arm decision (recorded)

Q8_K weights stage at 80 padded words and serve through the DEQUANT-COMPOSITE
arm — the tree-standard for Q8_K weights (the vec_dot selector at
`cpu_quant_dot.cpp` returns nullptr by design; the composite is both the CPU
oracle and the device path). NO new INT8-dot kernel in this row: the fused
Q8_K×Q8_K dot is a separate perf row, gated on the composite's measured cost.
`bsums` are stage-carried and never decoded (recorded: they cache the
per-16-group sums the Q4_K/Q5_K vec_dot arms consume; the composite's
to_float twin ignores them too — `tenstorrent_keepquant.cpp` Q8_K decode
case). The device dispatch bypasses `VT_TT_KEEPQUANT_INT8DOT` for Q8_K
unconditionally — the lever cannot route a Q8_K weight into a kernel that
cannot exist.
