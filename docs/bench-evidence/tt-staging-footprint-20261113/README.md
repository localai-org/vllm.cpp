# TT-STAGING-FOOTPRINT — the row-page repack serves UD-Q6_K_XL on one P150

Evidence for row `TT-STAGING-FOOTPRINT` (issue
`ISSUE-LOCAL-01M7B4HC9XPACJSFMRP4EVYZYM`). Implementation head
`e86625369`; spec arithmetic `722fa1d6d`; byte-identity evidence
`cd3e3309b`. Build: `/tmp/build-ttfp` (config `/tmp/build-ttfp-config.log`;
recipe = `cmake -B /tmp/build-ttfp -G Ninja -DCMAKE_BUILD_TYPE=Release
-DVLLM_CPP_TENSTORRENT=ON -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_BUILD_TESTS=ON
-DVLLM_CPP_BUILD_EXAMPLES=ON -DCMAKE_PREFIX_PATH=<tt-metal-pin-durable
cmake dirs>`, plus `-DCMAKE_CXX_FLAGS=-isystem <pin>/libexec/tt-metalium/
ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device` — a known
host gap: the durable pin's install include tree lacks
`kernels/topk_large_indices_compute_body_mode.hpp`; every previously green
compile rode ccache hits).

## The geometry (Design B, spec-amended)

kQ8_0/kQ6_K stage ONE WEIGHT ROW per accessor page: the row's `nb` GGML
blocks copied tight at their exact byte strides, zero-padded to
`row_bytes = ceil64(nb x block_bytes)` — a 64-B multiple, so the W1 leg33
page-stride contract holds by construction, with <=63 B of pad per ROW
(the legacy grid padded per BLOCK: Q8_0 x1.88, Q6_K x1.22). Staged
footprint for UD-Q6_K_XL drops ~36.9 GiB -> ~23.6 GiB (the spec carries
the full table; a fixed multi-block tile was rejected there — a
zero-waste Q6_K tile needs `32 | bpt` and the artifact's nb values are
20/24/40/68). Q4_K/Q5_K keep the untouched legacy grid.

## Gates

1. **Focused red/green.** `mutation-stride-red.log`: with the kernel's
   `xstride` corrupted to `word_bytes` (the row-packed per-block stride
   ignored), the multi-block focused test fails 18/43 assertions — the
   leg33 signature (rows spanning 4 blocks put blocks 1..3 at interior
   offsets the legacy geometry never produced). `focused-green.log`:
   restored, 43/43. `residency-green.log`: the four-encoding residency
   test 81/81, now probing kQ8_0/kQ6_K ROW-PACKED (and legacy-absent)
   and kQ4_K/kQ5_K legacy (and row-packed-absent).
2. **Existing-arm byte-identity.** `q4km-1-token-ids.json` is
   byte-identical (sha256 `d49c53d2...`) to the recorded pre-change
   stream in `docs/bench-evidence/qwen38-q6k-vs-q4km-20261113/`
   (q4km-1/2/3, recorded at `de3493a3742`, a code-identical keepquant
   base), at the recorded recipe: `--num-blocks 64`, batch 1, input 32,
   output 200, seed 0. TPOT 6624 ms vs recorded 6607. First attempt at
   `--num-blocks 8` produced a 64-token stream — the engine's KV
   auto-fit capped max_model_len at 96; the recorded stream was taken at
   64 blocks, so the gate reruns at the recorded recipe.
3. **UD-Q6_K_XL serves on the P150.** Model sha256
   `701d8fa9ed214ab21bfc130cd2a7df19ca89bbef7713e2dfb19f3c63696aa917`.
   - KV pool 16: OOM (`oom16-q6kl-1.stderr.log`) — the board is fully
     allocated (4269/4272 MB per bank) at the decode GDN scatter
     transient; weights+KV16+GDN state do not fit.
   - KV pool 8, CAPTURED arm: OOM at `end_trace_capture`
     (`oom8-q6kl-1.stderr.log`) — the 80,134,144 B trace region asks
     10,018,816 B per bank against 13.6 MB/bank free with a 4.8 MB
     largest free block: a ~80 MB shortfall, fragmentation-bound.
   - KV pool 8, EAGER decode (`VT_TT_DECODE_CAPTURE=0`, the engine's
     documented opt-out): **serves, 3/3 reps, byte-identical token IDs**
     (`eager8-{1,2,3}-token-ids.json`, sha256 `b36bc61a...`), Mean TPOT
     5340.63 / 5342.32 / 5339.67 ms, 64 generated tokens (the 8-block
     pool caps max_model_len at 96: 32 in + 64 out; output-len 200 is
     not runnable at this pool — recorded, not silently truncated).
   - Recipe per leg: mutex `${GPU_LOCK}`, luwen reset + 15 s + shm
     cleanup, idle gate load<16, `taskset -c 32-63`, governor/env/ps-top
     logged (`*-env.txt`, `ps-top-*.txt`); builds `taskset -c 64-127`.

## Known host gaps

- `check-windows-portability.py` fails on the missing `dotnet`
  (`preflight-staged.log`); `test_check_test_registration` passes. Full
  staged preflight otherwise green.
- The tt-metal include-tree gap above (build-only, workaround recorded).

## Streams are short on purpose

The degenerate [220,16,23066,...] repetition appears in BOTH arms'
streams (Q4_K_M at 64 blocks and Q6_K_XL at 8) — the synthetic prompt's
greedy attractor, not a copy error: the Q4_K_M gate compares against the
independently recorded 200-token stream byte-for-byte.
