# SPEC — `TT-STAGING-FOOTPRINT` — the keepquant staging grid inflates high-bit weights past one P150's DRAM; make the new unsloth artifacts fit

- Owning row: `TT-STAGING-FOOTPRINT` (record ID `BACKEND-TENSTORRENT-STAGING-FOOTPRINT`)
- Trigger: UD-Q6_K_XL (23.56 GiB) and UD-Q8_K_XL (29.30 GiB) — both downloaded, byte-verified, metadata-servable — OOM on the P150 at EVERY KV pool (64/16/8 blocks; evidence docs/bench-evidence/qwen38-q6k-vs-q4km-20261113/{oom-64blocks,oom-16blocks}/ and docs/bench-evidence/tt-q8k-weights-20261113/ — the tt-q8k-weights directory itself is referenced-only in this tree; see W1 below). W1-corrected arithmetic (2026-11-13): the grid inflates Q6_K ×1.219 (210→256 B, 64 words — NOT the ×1.52/320 B this spec first recorded) and Q8_0 ×1.88 (34→64 B, 16 words), so the inflation is Q8_0-dominated: staged UD-Q6_K_XL ≈ 36.9 GiB (Q8_0 12.60×1.88 + Q6_K 9.93×1.22 + Q5_K 1.00×1.09) against a 34 GB board; staged UD-Q8_K_XL ≈ 50.9 GiB (Q8_0 24.49×1.88 + BF16 4.79). Dtype mixes measured off the artifacts with a raw GGUF metadata read (offsets sum to the file sizes exactly): UD-Q6_K_XL = Q8_0 53.5% / Q6_K 42.2% / Q5_K 4.2%; UD-Q8_K_XL = Q8_0 83.6% / BF16 16.4% and carries NO Q8_K (ggml type 9) tensors at all — the "Q8_K" in the name is UD-matrix naming, so this artifact never exercises the 032203666 Q8_K arm.
- The plugin contrast (developer request 2026-11-11): the tt-model bundle serves `Qwen/Qwen3.8-27B` bf16 via `ttnn.from_torch` tilized on FOUR Blackhole chips — no host-block staging at all. We have one board, so the fix is on our staging format.

## W1 finding — the leg33 constraint, derived from the evidence (2026-11-13)

**The leg33 record.** No standalone bench-evidence file records leg33; the only record is the code comment, introduced with the padding itself in d9298c85c (2026-09-08, `git log -S'leg33'`): src/vt/tenstorrent/tenstorrent_keepquant.cpp:15-22 — "a 144-B logical page inside 192-B DRAM slots read back as stream bytes 16 + 152*slot — the tensor write and the accessor disagreed on the layout". The pre-leg33 grid was TIGHT (e456c390f: Q4_K 36 words = 144 B, Q5_K 44 = 176 B, Q6_K 53 = 210 B, Q8_0 9 = 32 B — exact GGML block bytes, zero-filled 2-byte tails only); the 64-B zero pad is leg33's fix, not an original property. (leg41, keepquant.cpp:73-79, is a different bug — a freed-address aliasing hit in the shadow cache, fixed by `DropKeepQuantWordShadow`; it is not the page-size evidence.) The spec's earlier "Q5_K decode sweep" framing is wrong twice over: the mispadded page in the record is 144 B — a Q4_K block — and the record is a decode-sweep readback garbage signature, not a served-wrong-weight proof.

**The constraint.** tt-metal sizes an interleaved DRAM buffer's page stride at 64-B granularity: the only alignment hypothesis consistent with the recorded 144→192 B slot rounding (ceil16(144)=144 and ceil32(144)=160 both fail; ceil64(144)=192 matches), and the Q8_K arm commit 032203666 states the same reading ("the pad rule ... an unpadded row would reintroduce it"). Therefore:

> **Staged keepquant rows must be 64-B multiples, because the device consumer strides the staged words tensor by `acc_w.get_noc_addr(page)` — which advances by tt-metal's aligned DRAM page stride — while reading `word_bytes = wpb*4` bytes per page (src/vt/tenstorrent/tenstorrent_keepquant.cpp:1821-1822, stride at 1829+ via `kq_vec_dot_*`'s `xblock += block_word_bytes`, kernels/keepquant_kernel_code.h:267-273). When `wpb*4` is not the aligned stride, page r's bytes are read from the wrong slot offsets — exactly the leg33 signature.** The consumer is LIVE: this is the production dense arm (`MatmulBTQuantInt8DotKernel`, keepquant.cpp:2333, `wpb` at :2533, `weight_row_bytes` at :2504), not a removed path.

**Design A is KILLED, two ways.** (1) By the constraint: 224 B (7 words = 56×4) is not a 64-B multiple, so the 32-B grid reintroduces the leg33 disagreement. (2) By arithmetic: every registered encoding already sits at its ceil-64 minimum (Q4_K 192 = ceil64(144), Q5_K 192 = ceil64(176), Q6_K 256 = ceil64(210), Q8_0 64, Q3_K 128 = ceil64(110), IQ4_XS 192 = ceil64(136), IQ2_XS 128 = ceil64(74), IQ1_S/M 64, IQ3_XXS 128, IQ3_S 128, IQ2_XXS 128, Q2_K 128, IQ1_M 64) — there is no tighter legal grid, and even if there were, the Q8_0 ×1.88 inflation (53.5% of UD-Q6_K_XL, 83.6% of UD-Q8_K_XL) is untouched by any grid change that keeps one GGUF block per row. Design A cannot serve either target artifact.

**Design B survives and is selected.** A load-time repack into device-native tiles is trivially compatible with the constraint: the accessor page IS the tile, and we size tiles as 64-B multiples, amortizing the alignment overhead across many blocks per tile row (a 3360-B strip of 16 Q6_K blocks pays 64-B alignment once per 16 blocks instead of once per block). It requires a kernel-side read-geometry change (per-tile addressing, decode offsets inside the tile) — M5-adjudicated per the spec's existing rule — and the greedy-exactness gate against the Q4_K_M arm, whose 192-B grid is untouched only if the repack starts from the same bytes. Both designs were evaluated; there is no measurement that rescues A (the constraint is structural), so no separating measurement is owed.

**Corrections to this spec's first arithmetic.** Q6_K inflation is ×1.22 (256 B staged), not ×1.52 (320 B) — the first table read the pad as 80 words; `KeepQuantWordsPerBlock(kQ6_K)` returns 64 (keepquant.cpp:31). The staged-UB-Q6_K_XL total is ~36.9 GiB, still over the board, and Q8_0-dominated — so the OOM record and the design outcome stand, but the "staging to ~35 GB" line attributed the wrong multiplier to Q6_K. Side note recorded for the implementing wave: the inline word-table comment at keepquant.cpp:11-12 still says "Q8_0 32 -> 9" (the e456c390f tight grid); the code has returned 16 words since d9298c85c. Fixing that comment is product-file scope and is left to the Design B change.

## The two designs (W1 decides which; both are greedy-exact gated)

- **Design A — tighter grid** (**KILLED by W1, 2026-11-13**): the 32-B grid (Q6_K 210→224 B) violates the 64-B accessor-stride constraint (leg33), and every encoding already sits at its ceil-64 minimum, so no legal tightening exists — see the W1 finding above.
- **Design B — plugin-shaped load-time repack (SELECTED by W1, 2026-11-13)**: convert GGUF blocks at load time into tight device-native quant tiles (the staged copy IS the tile layout; no host-row inheritance), eliminating the grid's overhead for every dtype at once — Q8_0 included, which is the decisive case given the measured mixes. Tile rows are accessor pages sized as 64-B multiples, so the leg33 constraint is satisfied by construction; the kernel-side read-geometry change is M5-adjudicated.

## Design B — the row-page repack (W2 geometry, committed BEFORE implementation)

**The geometry.** For `kQ8_0` and `kQ6_K` — the two encodings whose combined payload is 95.7% of UD-Q6_K_XL — the load-time staging (`EnsureKeepQuantRowWords`, a new shadow map beside `EnsureKeepQuantWords`) packs ONE WEIGHT ROW per accessor page: the page is the row's `nb` GGML blocks copied tightly (`nb × block_bytes`, 34 B / 210 B, no per-block pad), zero-padded to the 64-B page multiple:

> `row_bytes = ceil64(nb × block_bytes)`, staged tensor `{rows, row_words}` with `row_words = row_bytes / 4`.

This amortizes the leg33 64-B alignment over the whole row (pad ≤ 63 B per ROW, not per block), so the W1 constraint holds by construction: the accessor's aligned page stride equals `row_bytes` exactly. The Q4_K and Q5_K grids stay on `EnsureKeepQuantWords` untouched — that is the byte-identity gate 3/4, and their residual inflation (Q5_K ×1.09 ≈ 1.09 GiB staged) is immaterial. Block-internal strides inside the page are the exact GGML block bytes, so the int8-dot kernel's decode offsets are unchanged; only the page walk changes (`grow` = one page per weight row; the dot's `block_word_bytes` stride becomes the natural `block_bytes`). Consumers that need per-block word rows — the decode chains (`DecodeKeepQuantWordsF32`), the grouped/E=1 arm, the `=0` opt-out — keep calling `EnsureKeepQuantWords` and stage their own legacy words on demand; the dense default path never reaches them, and the double-staging cost lands only in the explicit rollback scenario. `StageKeepQuantWordsFor` (the W4d W6 load-time pre-pass) routes `kQ8_0`/`kQ6_K` to the row-packed staging so the words exist once, at load, in the tight form.

**Why not fixed multi-block tiles:** a uniform tile of `bpt` blocks needs `bpt × block_bytes ≡ 0 (mod 64)` — for Q6_K that forces `32 | bpt` (210 = 2·3·5·7) — AND `bpt | nb`. Measured `nb = K / 256` values in UD-Q6_K_XL are 20 (K=5120), 24, 40 (K=10240), 68 (K=17408), … — never divisible by 32, so a zero-waste fixed tile does not exist for this artifact's dims, and a padded uniform tile wastes up to a full tile per row (52 KB × 17408 rows ≈ 0.9 GiB on ffn_up alone). The whole-row page has no `nb` divisibility requirement at all.

**Footprint arithmetic for UD-Q6_K_XL (measured off the artifact header, dims verified against the file size):**

| dtype | payload | staged (row-page) | staged (legacy grid) |
|---|---|---|---|
| Q8_0 | 12.596 GiB | 12.596 GiB + ~0.2 MiB row pad (K=48 GDN rows pad 32 B × 5120 rows) | ×1.88 ≈ 23.7 GiB |
| Q6_K | 9.931 GiB | 9.931 GiB + ~6 MiB row pad (only K=5120 rows: 4200 → 4224 B, 24 B × ~250k rows) | ×1.22 ≈ 12.1 GiB |
| Q5_K | 0.997 GiB | (legacy ×1.09) 1.09 GiB | 1.09 GiB |
| f32 (norms) | 0.010 GiB | 0.010 GiB | 0.010 GiB |

**Total staged ≈ 23.6 GiB against the 34 GB board** — under the ~32 GB bar with ~10 GB of headroom for the bf16 embed twin (~2.4 GiB), activations, and a 16-block KV pool. The legacy grid staged ≈ 36.9 GiB (W1's corrected arithmetic) and OOM'd at every pool. Largest staged row page: `210 × (17408/256) = 14,280 B` (ffn rows), well inside the CB page and L1 budgets (the CB_W page SHRINKS vs legacy: 14,280 B vs 17,408 B for the same row). All served `nb × block_bytes` values in the artifact are 64-B-exact except the two 48-column GDN rows, so the inflation is ×1.001 overall — the row's footprint lever is exhausted at ×1.0; no tighter legal geometry exists under the W1 constraint.

**The kernel change (M5), localized.** One new common arg (`CARG_BLK_BYTES`, 14; `kNumCommonArgs` 14 → 15). 0 = legacy per-block-page geometry (Q4_K/Q5_K and every GSQ/IQ encoding — byte-identical path); nonzero = row-packed, and the kernel reads ONE page per weight row (`acc_w.get_noc_addr(grow)`) and dots with `block_word_bytes = blk_bytes`. The 14-arm dot chain is untouched (it already takes the stride as a parameter); for legacy arms the stride argument evaluates to the same `word_bytes` as before. `wpb`/`weight_row_bytes`/the CB_W page derive from the row geometry on the host side per encoding.

## Gates

1. W1's written constraint + which design it selects (records).
2. DRAM-fit: the target artifact loads and serves on one P150 (deterministic x3 token IDs at the standing recipe; KV pool recorded).
3. Greedy-exactness vs the pre-change arm on the SAME artifact is impossible where the artifact previously OOM'd — for those, the gate is: existing Q4_K_M arm byte-identical (the staging change must not perturb it) + the focused staging/envelope tests.
4. Existing arms byte-identity + full preflight (known host gaps excepted).

## Risks/decisions

- W1 ANSWERED (2026-11-13): the constraint binds a LIVE consumer (see the W1 finding). Design A killed; Design B selected — its tile layout satisfies the constraint by construction, and its kernel read change is the M5 question. Q8_0's inflation is decisive: 53.5% of UD-Q6_K_XL and 83.6% of UD-Q8_K_XL is Q8_0, which no per-block grid change touches.
- Multi-device (the plugin's actual answer: more chips) is out of scope; one P150 is the fleet we have.

## Stop conditions

- W1 concludes the grid is required by a live consumer AND Design B's tile layout cannot satisfy it: record the finding; the artifacts stay unservable-on-one-P150 and the row closes with the arithmetic.
- Greedy divergence or existing-arm byte-identity breakage: STOP.

## Git integration

One pull request; W1 may land records-first if it selects a design worth staging separately. Push to `mine`.
