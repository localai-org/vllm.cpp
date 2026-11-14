# SPEC — `TT-EMBED-DUAL-RESIDENCY` — the tied embed table lives twice on device; eliminate the duplicate and unlock the captured arm

- Owning row: `TT-EMBED-DUAL-RESIDENCY` (record ID `BACKEND-TENSTORRENT-EMBED-DUAL-RESIDENCY`)
- Trigger: TT-STAGING-FOOTPRINT's W3 recorded the captured-arm serving gap (~80 MB trace-region short at KV 8, fragmentation-bound) and named the lever: the tied embed table's dual residency — **9.95 GiB keep-quant row words + 2.44 GiB bf16 twin**. Removing the twin frees ~2.4 GB: the captured-arm gap disappears with margin, KV headroom grows, and UD-Q8_K_XL viability (Q8_0 83.6% + BF16 16.4% — its BF16 slice benefits identically) gets re-checked on a board that is no longer full.

## The anchor (from the staging-footprint G6 investigation)

The dual copy is created by the tied keep-quant **non-share fall-through** in
`LoadEmbedAndHead`: `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:883`
(share gate) → `:897` (`*head = OwnMatmulWeight`; deliberate non-share,
:869-880). Device side: head row words `tenstorrent_keepquant.cpp:107`
(pre-pass `tenstorrent_capture.cpp:473`) sit beside the gather bf16 twin
`tenstorrent_residency.cpp:30` (call site `tenstorrent_ops.cpp:335`).

## Waves

- **W1 — consumer census**: determine exactly which device consumers read the bf16 twin vs the row words, at the call sites above. Deliverable: a written census (consumer -> representation -> file:line) and the elimination design: either (a) the gather reads the dequantized-on-demand view of the row words, (b) the lm_head matmul reads the bf16 twin, or (c) the twin is proven dead and simply dropped. The census names the smallest correct change.
- **W2 — elimination**: implement the census's design. Gate: the embedding/lm_head focused tests byte-identical for the existing arms (Q4_K_M/Q6_K_XL), and DRAM measured before/after (the ~2.4 GiB must actually disappear from the allocator's ledger).
- **W3 — the unlock gate**: with the twin gone, re-run the TT-STAGING-FOOTPRINT W3 ladder: (a) UD-Q6_K_XL on the CAPTURED arm at KV 8 (the ~80 MB gap must clear; deterministic x3 token IDs matching the recorded eager-arm stream `b36bc61a…` — captured vs eager MUST be token-identical), (b) TPOT comparison captured-vs-eager on the same artifact (the capture-default recipe), (c) KV pool headroom re-measured (largest fitting pool now), (d) UD-Q8_K_XL viability re-check at its largest fitting pool.

## W1 census (recorded 2026-11-14, worktree bd78cb172 + this commit)

**Why the twin exists.** The tied keep-quant share gate (`qwen3_5_gguf_weights.cpp:883`,
reason comment `:871-878`, introduced in e228d6893) refuses to share because the head
goes through `OwnMatmulWeight`, "which may REPACK the slice ... and a table sharing a
repacked buffer would gather permuted rows". That reason is **host-CPU-only**:
`QuantRepackForDevice` (`gguf_keep_quant.cpp:96-100`) requires `dev == vt::DeviceType::kCPU`,
so on kTENSTORRENT the head's words are never repacked — the row-page staging copies
each weight row's blocks tightly at their exact GGML block bytes
(`tenstorrent_keepquant.cpp:87-105`) and the legacy grid pads blocks without permuting them.
The second, decisive reason the pair still splits on TT is the gather arm:
`DeviceQuantGatherSupported` (`gguf_keep_quant.cpp:327-329`) answers
`OpRegistered(OpId::kEmbeddingQuant, dev)`, and `kEmbeddingQuant` is registered for
kCPU (`cpu_ops.cpp:4662`) and CUDA (`cuda_embedding_quant.h:47`) but **not** kTENSTORRENT.
So on TT the tied table routes `kExpandBf16` (host bf16 twin, `qwen3_5_gguf_weights.cpp:853`)
while the head routes `kKeepQuant` — the dual residency.

**Consumers of the bf16 twin** (device BFLOAT16 `[vocab, H]`, uploaded once by
`EnsureEmbedTableDevice` `tenstorrent_residency.cpp:29-54`, cached in `EmbedTableShadows()`
keyed on the host twin pointer; `ToHostF32` host dequant feeds the upload):

1. Eager gather (prefill, and the pre-capture warm step): `EmbeddingKernel`
   (`tenstorrent_ops.cpp:297`, table fetch `:335`) → `ttnn::embedding`.
2. Captured decode replay: `EmbedDeviceIdsInto` (`tenstorrent_capture.cpp:322-370`) →
   `ttnn::embedding` on the cached shadow with warmed ids (`WarmDecodeIds`,
   `tenstorrent_capture.cpp:287`); driver call sites `qwen3_5.cpp:13107` and `:13172`.

**Consumers of the row words** (device i32 word shadow staged by `StageKeepQuantWordsFor`
`tenstorrent_capture.cpp:464-487` at load/capture pre-pass, keyed on the packed host
pointer; row-packed form for Q8_0/Q6_K via `EnsureKeepQuantRowWords`
`tenstorrent_keepquant.cpp:105`, legacy grid otherwise via `EnsureKeepQuantWords:224`):

1. The lm_head keep-quant matmul: `MatmulBTQuantKernel` (`tenstorrent_keepquant.cpp:1055`),
   dense int8-dot and grouped expert arms, each call slice-decoding from the resident
   words. No other consumer.

**Both copies are live and hot** — the twin serves every gather, the words serve every
lm_head step — but they serve DIFFERENT tensors and neither is irreplaceable: the twin's
rows are exactly the row words' rows decoded. The CPU tree already proves the shape:
`vt::Embedding` decodes one row per gathered id (`ops.cpp:1825` dispatch,
W6a e228d6893), which is why a keep-quant gather table is legal residency at all
(`qwen3_5_gguf_weights.cpp:825-831`).

**Design chosen: (a) the gather dequantizes from the row words — via device work, per step,
on the gathered rows only.**

1. Register `kEmbeddingQuant` on kTENSTORRENT (`EmbeddingQuantKernel`): stage the packed
   table's word shadow (the SAME `EnsureKeepQuantRowWords`/`EnsureKeepQuantWords` cache the
   head uses), gather the `t` selected word rows on device (t ≤ batch — per-step cost is
   O(t) rows, not the 2.44 GiB table), run the existing decode chain
   (`DecodeKeepQuantWordsF32` is generic over `{rows, nb}`) → f32 `[t, H]` → bf16 out.
2. The loader picks the residency up with NO gate edit: `OpRegistered` answers true and
   `embed_r` becomes `kKeepQuant` on TT (`gguf_keep_quant.cpp:380-392`).
3. Extend the tied share gate with a keep-quant arm, allowed exactly when the device does
   not repack (`!QuantRepackForDevice(...)` — i.e. any non-CPU device): the embed borrows
   the head's packed bytes (`nk=false`), so ONE host packed copy and ONE device word shadow
   serve both the gather and the lm_head matmul. The CPU gate stays closed (repack live).
   Sharing is REQUIRED, not optional: an unshared keep-quant table would stage a SECOND
   ~9.95 GiB word shadow — worse than the twin it replaces.

Net device ledger: 9.95 GiB words + 2.44 GiB bf16 twin → 9.95 GiB words. The ~2.44 GiB twin
(disappears from host AND device, since the ExpandBf16 host expansion is never built either)
is the freed margin.

**Numerics risk, named.** The gathered rows switch representation: host-decoded bf16 twin →
device decode chain. That chain is the one the lm_head words already serve, and its Q4_K/
Q5_K/Q6_K/Q8_0 chains claim bit-exactness against the host dequant (decode chain comment,
`tenstorrent_ops.cpp:367-407`). The W2 gate (focused embed/lm_head tests byte-identical) and
W3(a) (captured stream vs `b36bc61a…`) test exactly this claim end to end. Divergence = STOP.

**Capture risk, named.** `EmbedDeviceIdsInto` must run the dequant-gather chain INSIDE the
captured region: warmed ids + warmed word shadow are both capture-time-resident, and the
gather/decode/copy ops are the same capture-safe op classes the chunked matmul arm already
bakes (`tenstorrent_keepquant.cpp` capture-demand comments). No from_vector write may appear
in the per-step path (the stage-B lesson, spec risk 1).

## Gates

1. W1 census recorded in the spec/issue with file:line citations.
2. Focused embed/lm_head tests red-before/green-after; existing arms byte-identical.
3. W3(a) is THE gate: captured-arm greedy stream byte-identical to the recorded eager stream (`b36bc61a…`) — divergence = wrong implementation, STOP.
4. DRAM ledger: the twin's bytes measurably freed (allocator numbers before/after in the evidence).
5. Preflight (known host gaps excepted); Q4_K_M baseline arm byte-identical throughout.

## Risks/decisions

- The bf16 twin may be load-bearing for a trace-capture consumer (capture bakes reads at fixed addresses; swapping representations mid-capture is the stage-B lesson — any representation change must land OUTSIDE captured regions or be captured consistently).
- Dequant-on-demand gather may cost decode latency (the embed gather runs per step on the hot path); if W1's census shows the gather is hot, measure the trade before choosing (a) — the twin might be cheaper than dequant-per-step, in which case the row records it and the lever is only partial (e.g. drop the ROW WORDS and serve lm_head from the twin instead — either direction frees ~2.4 GB).
- The share gate (:883-:880) deliberately non-shares for a reason — find that reason in the git history (`git log -L` on the gate) before overturning it.

## Stop conditions

- W1 census shows both representations are live-hot and neither can be dropped without a new kernel: record the finding, the row closes with the census (the lever may still be partial via whichever copy is cold).
- Captured-vs-eager divergence at W3: STOP.

## Git integration

One pull request; worktree `/tmp/tt-embed`; push to `mine`. W1 may land records-first.
