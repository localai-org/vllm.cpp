# SPEC — `TT-STAGING-FOOTPRINT` — the keepquant staging grid inflates high-bit weights past one P150's DRAM; make the new unsloth artifacts fit

- Owning row: `TT-STAGING-FOOTPRINT` (record ID `BACKEND-TENSTORRENT-STAGING-FOOTPRINT`)
- Trigger: UD-Q6_K_XL (25.3 GB) and UD-Q8_K_XL (31.5 GB) — both downloaded, byte-verified, metadata-servable — OOM on the P150 at EVERY KV pool (64/16/8 blocks; evidence docs/bench-evidence/qwen38-q6k-vs-q4km-20261113/{oom-64blocks,oom-16blocks}/ and docs/bench-evidence/tt-q8k-weights-20261113/), while the raw weights would fit: the keepquant 64-B staging grid inflates Q6_K ×1.52 and Q8_0 ×1.88, staging UD-Q6_K_XL to ~35 GB against a 34 GB board.
- The plugin contrast (developer request 2026-11-11): the tt-model bundle serves `Qwen/Qwen3.8-27B` bf16 via `ttnn.from_torch` tilized on FOUR Blackhole chips — no host-block staging at all. We have one board, so the fix is on our staging format.

## W1 — the leg33 investigation (gates everything)

The 64-B grid exists for a reason: the leg33 page-size disagreement (the Q5_K decode sweep leg where unpadded rows re-served another weight's words; see tenstorrent_keepquant.cpp:62-78 and its recorded leg). Before ANY design change, re-derive from that evidence what page property the padding actually protects (which consumer reads staged rows at page granularity, and what alignment that consumer requires). Deliverable: a written constraint ("staged rows must be N-byte aligned because X reads Y") with the leg33 citations — or the finding that the constraint binds a consumer that no longer exists. This wave is records-only and its output selects or kills the designs below.

## The two designs (W1 decides which; both are greedy-exact gated)

- **Design A — tighter grid**: admit 32-B-multiple rows for Q6_K (210 B -> 224 B = 7 words, ×1.07 instead of ×1.52) and Q8_0 (34 B -> 64 B unchanged at 32-B grid — no win there) — a small change IF W1's constraint allows; measured DRAM-fit test: UD-Q6_K_XL must load and serve at a workable KV pool.
- **Design B — plugin-shaped load-time repack**: convert GGUF blocks at load time into tight device-native quant tiles (the staged copy IS the tile layout; no host-row inheritance), eliminating the grid's overhead for every dtype at once. Bigger: new repack pass, new tile geometry, and possibly a kernel-side read change — M5-adjudicated if it touches kernels.

## Gates

1. W1's written constraint + which design it selects (records).
2. DRAM-fit: the target artifact loads and serves on one P150 (deterministic x3 token IDs at the standing recipe; KV pool recorded).
3. Greedy-exactness vs the pre-change arm on the SAME artifact is impossible where the artifact previously OOM'd — for those, the gate is: existing Q4_K_M arm byte-identical (the staging change must not perturb it) + the focused staging/envelope tests.
4. Existing arms byte-identity + full preflight (known host gaps excepted).

## Risks/decisions

- If W1 finds the 64-B grid is genuinely load-bearing for a LIVE consumer, Design A dies and Design B inherits the same question (tilized tiles have their own alignment guarantees — likely satisfiable).
- Q8_0's 2.0× inflation is NOT fixed by Design A; if UD-Q8_K_XL's mix is Q8_0-heavy, only Design B serves it — check the artifact's dtype mix before choosing.
- Multi-device (the plugin's actual answer: more chips) is out of scope; one P150 is the fleet we have.

## Stop conditions

- W1 concludes the grid is required by a live consumer AND Design B's tile layout cannot satisfy it: record the finding; the artifacts stay unservable-on-one-P150 and the row closes with the arithmetic.
- Greedy divergence or existing-arm byte-identity breakage: STOP.

## Git integration

One pull request; W1 may land records-first if it selects a design worth staging separately. Push to `mine`.
