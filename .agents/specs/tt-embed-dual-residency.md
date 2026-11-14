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
