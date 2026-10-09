ID: ISSUE-LOCAL-01M4GPEZ987KVAXCYBFB2EVNT7
Title: Port the TT backend to the updated tt-metal stack (tt_umd 0.9.12/KMD 2.11.1/FW 19.15.0)
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: -

## Problem

Host Tenstorrent stack updated (tt-smi 6.7.0, tt-umd 0.9.12, KMD 2.11.1-pre, board firmware 19.7.1 -> 19.15.0). The frozen tt-metal pin (~/Sources/tt/tt-metal-pin, tt_umd v0.9.11-427) can no longer run on the P150: its tt_umd reports num_hw_cqs=0 against the new KMD/firmware, so every vllm TT binary fails at first op with TT_FATAL mesh_device.cpp:874 cq_id 0 out of range. A/B-proven: the same board through tt_umd 0.9.12 (probe /tmp/umd-probe, trial build /tmp/tt-metal-umdtrial + /tmp/umdtrial-install) reports num_hw_cqs=1 and CQ0 creates fine. Root cause and trial recipe: memory note p150-ttumd-rootcause-confirmed.md (2026-10-09). Developer decision: option 3, full pin forward. Work owed: port src/vt/tenstorrent/ + include/vllm/ to the new tt-metal headers/APIs (first break: tt-metalium/tensor/spec/memory_config/memory_config.hpp moved to include/tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp), link+load against /tmp/umdtrial-install, then rerun device gates: test_tenstorrent_backend embedding exactness, test_kolibri1 27/234, test_kolibri1_tt_b2i staging, decode bench anchor 109726, and the B2b-ii full-model token gate (141/145 argmax positions, 4 near-tie flips inside 2.5-nat band, 0 hard flips). Stop conditions: if the new tt_umd rejects FW 19.15.0, or token-gate numerics move beyond the band, STOP and report.

## Resolution

- 2026-10-09 (helper, row/ci-tt-pin-forward `a55e50f44`): the compile+link
  port landed — tensor spec headers shimmed to the new
  `tt-metalium/experimental/tensor/spec/...` paths
  (`src/vt/tenstorrent/tt_tensor_spec_compat.hpp`), the 18th
  `chunk_gated_delta_rule` parameter (`use_mcast`) removed upstream, and one
  gcc-15 `-Wstringop-overflow` false positive avoided in
  `tests/vllm/models/test_dots3_note_vision.cpp`. The whole tree builds and
  links with `-DVLLM_CPP_TENSTORRENT=ON` against the updated tt-metal
  install (`CMAKE_PREFIX_PATH=/tmp/umdtrial-install/lib64/cmake;/tmp/umdtrial-install/share/cmake`).
- 2026-10-09 (device gates on the P150, new tt_umd 0.9.12 — no
  firmware-version assertion fires against FW 19.15.0):
  `tests/vt/test_tenstorrent_backend` full suite 101/103 cases passed
  (778547/778549 assertions); the embedding row-gather exactness case 1/1;
  `tests/test_kolibri1` 27/27 cases, 234/234 assertions;
  `test_kolibri1_tt_b2i` device bring-up 1/1 (3.084 GiB resident, 753
  tensors). Two op-level numeric failures remain open (both reproduce with
  a clean kernel cache): `kGdnDecode` wide-range state (state max_abs
  0.0051074 vs tol 0.002, near-zero elements) and
  `kMatmulBTQuantGrouped` decode P=1 Q4_K (154/1024 outputs past the
  elementwise envelope, worst_rel 0.130697). An exit-time segfault after
  the doctest summary (GraphTracker teardown ordering) is recorded but
  result-neutral.
- 2026-10-09: the B2b-ii device token gate CANNOT reach numerics on the new
  stack for a reason independent of this port: commit `d81f7d2b7` (landed
  2026-10-09 00:18, while the card was down) routes the TT streaming MoE
  through `vt::MoeCombine` on the tenstorrent queue
  (`kolibri1_tt_forward.cpp:513`), and no TT kernel exists for
  `kMoeCombine` — the portable CPU reference tier is correctly refused
  because the discrete device's memory is not host-addressable
  (`src/vt/op_provider.cpp:637`). All 9 other `test_kolibri1_tt_b2ii`
  cases pass. The native `kMoeCombine` kernel (a weighted expert combine)
  is owed before the 141/145 band can be re-measured. The decode bench
  anchor (109726 with the 101807/109726 chain) is likewise still owed.
- 2026-10-09 (B2b-ii device half, branch `row/ci-tt-pin-forward`):
  the native `kMoeCombine` TT kernel landed (host-staged f32 scalar loop,
  bit-exact to the CPU oracle `cpu_ops.cpp` `MoeCombineKernel`; red-first
  doctest ran RED at `REQUIRE(OpRegistered(kMoeCombine, kTENSTORRENT))`
  and GREEN 961/961 after). The token gate then ran for the first time and
  exposed TWO further defects, both fixed or characterized this session:
  (1) the gate itself decoded step 0 from a ZEROED KV cache while claiming
  `ctx_before = len-1` computed positions (0/8 meaningless hard flips) —
  fixed to the W3 one-token-per-step walk (a whole-prompt prefill step
  also exceeds the B1 per-step stream bound: 1242865920 B charged against
  1179936000 B at layer 15; the chunked walk is the same-math
  alternative); (2) the pinned tt-metal's device bf16 `rms_norm` is
  broken — with the previous arm the model's attention output was EXACTLY
  ZERO at every layer — fixed by the f32-shadow norm arm in
  `tenstorrent_ops.cpp` `RmsNormKernel` (regression test: `kRmsNorm bf16
  [48,128] per-head arm`). Gate state after both fixes, WITH the host-free
  decode paths DISABLED (`VT_TT_HOST_FREE_DECODE=0`): 26/33 compared
  positions match, all 7 flips are genuine near-ties whose nat gaps sit
  just outside the 2.5 band (2.69-5.60) — still a FAILING gate (141/145,
  0 hard flips required); with host-free ON the device handoff corrupts
  activations (0/8, nonsense tokens). OWED: root-cause the host-free
  device-shadow handoff at this pin (device rope / device
  ReshapeAndCache / device residual norm are the paths host
  materialization bypasses); the `kGdnDecode` wide-range state drift
  (re-confirmed: dot error 0.044 vs a double oracle with an EXACT f32
  multiply — a reduction-path property; a pairwise reshape-sum experiment
  made it worse, 0.0195, and was reverted) and the `kMatmulBTQuantGrouped`
  decode P=1 Q4_K envelope miss (unchanged, 154/1024, worst_rel
  0.130697; not exercised by the kolibri1 path). Also observed once: the
  `batched decode RAC is capture-safe` shadow check failed mid-suite
  (V 109/128) and passed in isolation and on a full-suite re-run — a new
  intermittent, unowned. The decode bench anchor 109726 remains owed
  behind the token gate.
