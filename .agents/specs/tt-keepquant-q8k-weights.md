# SPEC — `TT-KEEPQUANT-Q8K-WEIGHTS` — the Q8_K weights arm: serve UD-Q8_K_XL-class GGUF checkpoints on the TT keepquant path

- Owning row: `TT-KEEPQUANT-Q8K-WEIGHTS` (record ID `BACKEND-TENSTORRENT-Q8K-WEIGHTS`)
- Trigger: the developer downloaded `unsloth/Qwen3.8-27B-GGUF` @ `ce331efc48d5722b4b8041f72793e530eae18d9b` (UD-Q8_K_XL, 31,457,991,680 B, LFS sha256 `af36ecb6b5db1407953345b746c14ac93f0657dda413910b4348683a2d990377`) and the arm must serve it; today the loader refuses (`kQ8_K` is activation-only in include/vt/dtype.h:41).

## The block layout (CORRECTED 2026-11-13 — the original spec conflated block_q8_0 with block_q8_K)

`block_q8_K` (src/vt/cpu/cpu_quant_blocks.h:129-134): 1 f32 `d` + 256 int8
`qs` + 16 int16 `bsums` = **292 B per 256 elements** (static_assert 292,
kQK_K = 256). Staging geometry: 292 B -> 73 words -> padded to the standing
64-B-multiple grid = **80 words (320 B)** per super-block (the pad rule at
tenstorrent_keepquant.cpp:62-68 exists to remove page-size disagreements;
an unpadded 73-word row would reintroduce the leg33 disagreement). The
Q8_K-weight vec_dot does NOT exist even on the CPU oracle
(cpu_quant_dot.cpp:1268-1273 returns nullptr; the tree-standard reference
for Q8_K weights is the **dequant-composite** fallback,
cpu_quant_traits.cpp:85-95 carries the row for to_float only). Activation
side: `kQ8_K` is already the K-quant `vec_dot_type`. Implementation shape
therefore: W2 stages the 80-word rows and serves through the
**dequant-composite arm** (tree-standard oracle AND tree-standard device
path) in v1; a fused Q8_KxQ8_K INT8-dot kernel is a PERF row of its own
(M5-rule adjudication), gated on the composite's measured cost.

## Waves

- **W1 — loader**: the GGUF weights loader accepts `Q8_K` weight tensors for the qwen35 family and stages them through the keepquant path. Update the activation-only comment in include/vt/dtype.h:41 to name the weights arm (the comment is the record; do not delete the constraint silently). Refuse-by-name stays for anything outside {Q4_K, Q5_K, Q6_K, Q8_0, Q8_K, F32/BF16}.
- **W2 — keepquant staging**: word-count + zero-pad treatment for Q8_K (9 words, no pad), and the decode kernel arm: Q8_K-weight × Q8_K-activation INT8 dot (scale `d` f32 per block) with the same accuracy contract as the Q4_K/Q6_K arms (CPU f32 reference oracle, numeric envelope per the standing kernel-matrix discipline).
- **W3 — device gate**: e2e serving of the pinned UD-Q8_K_XL artifact on the P150 — greedy token run at the standing recipe (batch 1 / input 32 / output 200 / seed 0), KV pool sized to fit (the Q6 legs proved 64 blocks OOM at 25.3 GB; 31.5 GB needs `--num-blocks 16` or smaller — measure and record the largest pool that fits). Token gate: byte-identical across 3 reps of the SAME arm (determinism), plus the CPU-reference token comparison if the reference forward covers Q8_K (if not, the determinism gate + focused-kernel envelope gate stand in, with the gap named).

## Scope

IN: Q8_K weight-side loader/staging/kernel, the dtype-comment record, tests. OUT: any other quant type, CPU-side changes, the Q6 benchmark verdict (separate evidence).

## Weights

Pinned: `/mnt/models/unsloth-qwen3.8-27B-gguf-v2/Qwen3.8-27B-UD-Q8_K_XL.gguf`, unsloth/Qwen3.8-27B-GGUF @ `ce331efc48d5722b4b8041f72793e530eae18d9b`, sha256 `af36ecb6b5db1407953345b746c14ac93f0657dda413910b4348683a2d990377`. docs/USAGE.md gains the arm entry in the same change that makes it servable.

## Risks/decisions

- INT8DOT machinery: if the fused single-program arm cannot take Q8_K weights without a new kernel compile unit, prefer the composition that exists (staging + existing INT8 dot) over new kernels; a genuinely new kernel is its own adjudication per the M5-style rule.
- DRAM: 31.5 GB weights leave ~2.5 GB for KV+working set on the 34 GB board — the W3 leg must record the KV pool that fits and NOT silently shrink below what the recipe needs; if even 8 blocks don't fit, STOP and report (the artifact is unservable on one P150 and the row records it).
- The activation vec_dot_type for a Q8_K-weights layer: if upstream (llama.cpp) uses Q8_K activations for ALL K-quant weights, our per-layer activation dtype selection must not change existing Q4_K/Q6_K arms' behavior — byte-identity of the existing arms is a gate.

## Tests

1. Focused keepquant staging test for Q8_K (word count, no-pad, byte-exact staging) — red-before on the absent arm, green after.
2. Kernel envelope test vs the CPU f32 oracle (the standing pattern in test_tenstorrent_backend.cpp).
3. e2e W3 device gate with determinism x3.
4. Existing Q4_K_M/Q6_K arms byte-identical (the dtype-selection change must not perturb them).

## Git integration

One pull request, `row/TT-KEEPQUANT-Q8K-WEIGHTS`, worktree `/tmp/tt-q8k`, push to `mine`.

## Stop conditions

- 8 KV blocks don't fit beside 31.5 GB weights: STOP, record unservable-on-P150.
- Byte-identity of existing arms breaks: STOP (the change is wrong, not the arms).
