ID: ISSUE-LOCAL-01M7B4HC9XPACJSFMRP4EVYZYM
Title: TT staging footprint: the 64-B grid inflates Q6_K x1.52 / Q8_0 x1.88 — the new unsloth high-bit artifacts cannot fit one P150
Row: TT-STAGING-FOOTPRINT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-13
Updated: 2026-11-13
Closed: -

## Problem

UD-Q6_K_XL (25.3 GB) and UD-Q8_K_XL (31.5 GB), both byte-verified and metadata-servable, OOM on the P150 at every KV pool (64/16/8 blocks; docs/bench-evidence/qwen38-q6k-vs-q4km-20261113/{oom-64blocks,oom-16blocks}/ + docs/bench-evidence/tt-q8k-weights-20261113/). Root arithmetic: the keepquant 64-B staging grid pads Q6_K rows 210->320 B (x1.52) and Q8_0 34->64 B (x1.88), staging UD-Q6_K_XL to ~35 GB against a 34 GB board. The tt-model bundle contrast (developer request): it serves bf16 tilized via ttnn.from_torch on four Blackhole chips — no host-block staging; we have one board, so our staging format is the lever. Owed: W1 the leg33 investigation (what page property the 64-B pad actually protects, written as a constraint with citations — it selects or kills the designs), then Design A (32-B grid for Q6_K rows; Q8_0 has no win there) or Design B (plugin-shaped load-time repack into tight device-native quant tiles), gated greedy-exact where comparable and on existing-arm byte-identity (Q4_K_M must stay byte-identical). Q8_0's inflation is only fixable by Design B — check the artifact dtype mix before choosing. Spec: .agents/specs/tt-staging-footprint.md

## Resolution

-
